# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from utils import BatchLayout, InternalError, get_name_or_self, repeat_interleave


class Wav2Vec2Masker(nn.Module, ABC):
    """Masks extracted wav2vec 2.0 features."""

    @abstractmethod
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.

        :returns:
            - The input sequences with mask applied. *Shape:* Same as ``seqs``.
            - The temporal mask that has been applied to ``seqs``. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math`S` is
              the sequence length.
        """

    if TYPE_CHECKING:
        __call__ = forward

    
    @staticmethod
    def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor) -> Tensor:
        """
        Extracts masked elements from ``seqs``.

        :param seqs: The sequences. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param temporal_mask: The temporal mask. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math`S` is the sequence length.
        """
        batch_size = seqs.size(0)

        # (N, S, M) -> (N x T, M)
        seqs = seqs[temporal_mask]

        # (N x T, M) -> (N, T, M)
        return seqs.unflatten(0, (batch_size, -1))  # type: ignore[no-any-return]


@final
class StandardWav2Vec2Masker(Wav2Vec2Masker):
    """Masks extracted wav2vec 2.0 features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    def __init__(
        self,
        model_dim: int,
        temporal_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        min_num_temporal_mask_spans: int = 2,
        spatial_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        min_num_spatial_mask_spans: int = 2,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param temporal_span_len:
            The length of each temporal mask span that is applied over time
            steps.
        :param max_temporal_mask_prob:
            The maximum probability of masking a time step. Note that, due to
            mask span overlap, the effective probability will be lower.
        :param spatial_span_len:
            The length of each spatial mask span that is applied over features.
        :param max_spatial_mask_prob:
            The maximum probability of masking a feature. Note that, due to mask
            span overlap, the effective probability will be lower.
        :param mask_factory:
            The row mask factory. If ``None``, :func:`compute_row_mask` will be
            used.
        """
        super().__init__()

        if max_temporal_mask_prob <= 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.temporal_mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans

        self.mask_factory = compute_row_mask

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.temporal_mask_embed)

    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> tuple[Tensor, Tensor]:
        if seqs_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        batch_size, seq_len, model_dim = seqs.shape

        # Temporal mask over time steps.
        temporal_mask = self.mask_factory(
            shape=(batch_size, seq_len),
            span_len=self.temporal_span_len,
            max_mask_prob=self.max_temporal_mask_prob,
            row_lens=seqs_layout.seq_lens_pt,
            min_num_spans=self.min_num_temporal_mask_spans,
            device=seqs.device,
        )

        if temporal_mask is None:
            raise InternalError("`temporal_mask` is `None`.")

        seqs[temporal_mask] = self.temporal_mask_embed.type_as(seqs)

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = self.mask_factory(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=self.min_num_spatial_mask_spans,
                device=seqs.device,
            )

            if spatial_mask is None:
                raise InternalError("`spatial_mask` is `None`.")

            # (N, M) -> (N, S, M)
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

            seqs[spatial_mask] = 0.0

        return seqs, temporal_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"min_num_temporal_mask_spans={self.min_num_temporal_mask_spans}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}, "
            f"min_num_spatial_mask_spans={self.min_num_spatial_mask_spans}"
        )

        if self.mask_factory is not compute_row_mask:
            mask_factory = get_name_or_self(self.mask_factory)

            s = f"{s}, mask_factory={mask_factory}"

        return s

def apply_mask(
    seqs: Tensor, mask: Tensor, *, fill_value: int | float | Tensor = 0
) -> Tensor:
    """
    Applies the specified boolean mask to ``seqs``.

    :param seqs: The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N`
        is the batch size, :math:`S` is the sequence length, and :math:`*` is
        any number of sequence-specific dimensions including none.
    :param mask: The boolean mask.

    :returns: The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    mask = unsqueeze(mask, dim=-1, count=seqs.ndim - mask.ndim)

    return seqs.where(mask, fill_value)


def compute_row_mask(
    shape: tuple[int, int],
    span_len: int,
    max_mask_prob: float,
    row_lens: Tensor | None = None,
    min_num_spans: int = 0,
    device: torch.device | None = None,
) -> Tensor | None:
    """
    Computes a random row mask of the specified shape.

    Note that, due to mask span overlap, the effective mask probability will be
    lower than ``max_mask_prob``. The implementation also guarantees that there
    will be always at least one unmasked element in each row.
    """
    num_rows, max_row_len = shape

    if row_lens is None:
        # We only mask rows that are longer than the mask span length.
        if span_len >= max_row_len:
            raise ValueError(
                f"Size of the second dimension of `shape` must be greater than `span_len` ({span_len}), but is {max_row_len} instead."
            )

        # (N)
        row_lens = torch.full(
            (num_rows,), max_row_len, device=device, dtype=torch.int64
        )
    else:
        # (N)
        row_lens = row_lens.to(torch.int64).view(num_rows)

        # We only mask rows that are longer than the mask span length.
        if (span_len >= row_lens).any():
            raise ValueError(
                f"All lengths in `row_lens` must be greater than `span_len` ({span_len}), but at least one length is smaller. row_lens: {row_lens}"
            )

    # (N, M x L)
    indices = _compute_mask_spans(row_lens, span_len, max_mask_prob, min_num_spans)
    if indices is None:
        return row_lens.new_empty((0, 0))

    return _generate_mask(indices, max_row_len).to(device)


def _compute_mask_spans(
    row_lens: Tensor, span_len: int, max_mask_prob: float, min_num_spans: int
) -> Tensor | None:
    """Compute random mask spans of the specified shape."""
    device, dtype = row_lens.device, row_lens.dtype

    num_rows = len(row_lens)
    if num_rows == 0:
        return None

    # Compute the number of mask spans per row. We should always have at least
    # one unmasked element; this is why we subtract 1 from `row_lens`.
    num_spans_per_row = max_mask_prob / span_len * (row_lens - 1)

    # Require the same number of mask spans for all rows.
    num_spans = int(num_spans_per_row.to(dtype).min())

    if min_num_spans > num_spans:
        raise ValueError(
            f"`min_num_spans` is {min_num_spans}, but with the given `span_len` and `max_mask_prob` only {num_spans} mask span(s) can be generated."
        )

    if num_spans == 0:
        return None

    # The range of possible start indices for mask spans in form [0, max + 1).
    # (N)
    span_start_range = row_lens - span_len + 1

    # (N) -> (N x M)
    span_start_range = repeat_interleave(span_start_range, dim=0, repeat=num_spans)

    # Unlike the fairseq implementation, we do sample with replacement, which is
    # more consistent with the overlap strategy.
    # (N x M)
    rand_scales = torch.rand(num_rows * num_spans, device=device)

    # By random scaling we effectively pick a random start index for each mask
    # span.
    span_offsets = span_start_range * rand_scales

    # The following ops convert the mask span offsets (i.e. start indices) to
    # mask spans (i.e. index ranges).
    # (N x M) -> (N, M)
    span_offsets = span_offsets.to(dtype).view(num_rows, -1)

    # (N, M) -> (N, M x L)
    span_offsets = repeat_interleave(span_offsets, dim=-1, repeat=span_len)

    # (L)
    indices = torch.arange(span_len, device=device, dtype=dtype)

    # (L) -> (N, M x L)
    indices = indices.repeat(num_spans).unsqueeze(0).expand(num_rows, -1)

    return span_offsets + indices


def _generate_mask(indices: Tensor, max_row_len: int) -> Tensor:
    """Generate a boolean mask by setting ``indices`` to ``True``."""
    # (N, S)
    float_mask = torch.zeros((indices.size(0), max_row_len), device=indices.device)

    # Set elements corresponding to masked indices to 1.
    float_mask.scatter_(1, indices, 1.0)

    # Since mask spans may overlap, rows might have varying number of masked
    # elements; therefore, we have to randomly unmask some of the elements to
    # ensure that all rows have the same amount of masking.
    min_num_masked = int(torch.count_nonzero(float_mask, dim=-1).min())

    # (N, min(M x L))
    # We randomly pick `min_num_masked` masked elements from each row, which
    # effectively unmasks the remaining elements.
    #
    # We first make a tensor of random values and 0.001 to it to ensure the
    # minimum value is larger than 0. Then we multiply it with the float_mask so
    # that all the 0 values in `float_mask` are still 0 but the non-zero values
    # have a random value assigned to them. Then we select the top-k values,
    #  which would be basically a subset of non-zero values `float_mask`.
    random_values = torch.rand_like(float_mask) + 0.001

    random_values = random_values * float_mask

    _, indices = torch.topk(random_values, k=min_num_masked, dim=1, sorted=False)

    # (N, S)
    # Now we construct the actual boolean mask which has the same number of
    # masked elements in each row.
    bool_mask = torch.full_like(float_mask, False, dtype=torch.bool)

    return bool_mask.scatter_(1, indices, True)

def unsqueeze(x: Tensor, dim: int, count: int = 1) -> Tensor:
    for _ in range(count):
        x = x.unsqueeze(dim=dim)

    return x

