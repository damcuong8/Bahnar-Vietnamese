# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
import numbers
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from collections.abc import Sequence
from typing import ClassVar, TypeAlias, Literal, final
from typing import Any
from torch.nn.functional import log_softmax, nll_loss

TensorData: TypeAlias = int | float | Sequence[int] | Sequence[float]

def to_tensor( 
    data: TensorData, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    if device is None or device.type != "cuda":
        return torch.tensor(data, dtype=dtype, device=device)

    t = torch.tensor(data, dtype=dtype, device=torch.device("cpu"), pin_memory=True)

    return t.to(device, non_blocking=True)


def compute_neg_counts(
    seq_len: int,
    num_negatives: int = 100,
    min_same_ratio: float = 0.1,
    max_same_ratio: float = 0.8,
    k: float = 250.0,   # half-saturation constant (internal, you rarely need to change)
) -> tuple[int, int]:
    """
    return (num_utterance_negatives, num_batch_negatives)
    """

    seq_len = max(1, int(seq_len))
    total = max(1, int(num_negatives))

    # compute ratio via Michaelis–Menten style saturating function
    frac = seq_len / (seq_len + k)  # between 0 and 1, increases with seq_len
    same_ratio = float(min_same_ratio) + (float(max_same_ratio) - float(min_same_ratio)) * frac

    # number of negatives within the same utterance (rounded)
    num_utt = int(round(total * same_ratio))

    # cap to avoid requesting too many same-utterance vs possible positions
    max_unique_positions = max(1, seq_len - 1)  # exclude self index
    if num_utt > max_unique_positions:
        surplus = num_utt - max_unique_positions
        num_utt = max_unique_positions
    else:
        surplus = 0

    num_batch = total - num_utt + surplus

    # ensure non-negative
    num_batch = max(1, num_batch)
    if num_utt < 0:
        num_utt = 0

    return num_utt, num_batch


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    pad_idx: int | None,
    *,
    label_smoothing: float = 0.0,
    target_mask: Tensor | None = None,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> Tensor:
    """
    Computes the cross entropy loss.

    .. note::
        The loss smoothing implementation of this function is compatible with
        fairseq.
    """
    if logits.ndim == 3:
        batch_size = logits.size(0)

        # (N, S, T) -> (N x S, T)
        logits = logits.flatten(0, 1)
    else:
        batch_size = None

    if targets.ndim == 2:
        # (N, S) -> (N x S)
        targets = targets.flatten()

    # For numerical stability run in single precision.
    # (S, T) -> (S, T)
    log_probs = log_softmax(logits, dim=-1, dtype=torch.float32)

    if label_smoothing == 0.0:
        if pad_idx is None:
            pad_idx = -100

        # (S)
        loss = nll_loss(
            log_probs,
            targets,
            ignore_index=pad_idx,
            reduction=reduction if target_mask is None else "none",
        )

        if target_mask is None:
            if reduction == "none" and batch_size is not None:
                # (N x S) -> (N, S)
                loss = loss.unflatten(0, (batch_size, -1))

            return loss

        if target_mask.ndim == 2:
            # (N, S) -> (N x S)
            target_mask = target_mask.flatten(0, 1)

        # (S)
        loss = loss * target_mask

        if reduction == "sum":
            return loss.sum()

        if reduction == "mean":
            return loss.mean()

        if reduction == "none":
            if batch_size is not None:
                # (N x S) -> (N, S)
                loss = loss.unflatten(0, (batch_size, -1))

            return loss

        raise ValueError(
            f"`reduction` must be 'sum', 'mean' or 'none', but is '{reduction}' instead."
        )

    # (S) -> (S, 1)
    targets = targets.unsqueeze(-1)

    # (S, 1)
    loss = -log_probs.gather(dim=-1, index=targets)

    # (S, 1)
    if label_smoothing > 0.0:
        smooth_loss = -log_probs.sum(dim=-1, keepdim=True)
    else:
        smooth_loss = None

    if pad_idx is not None:
        padding_mask = targets.eq(pad_idx)

        loss.masked_fill_(padding_mask, 0.0)

        if smooth_loss is not None:
            smooth_loss.masked_fill_(padding_mask, 0.0)

    if target_mask is not None:
        if target_mask.ndim == 2:
            # (N, S) -> (N x S)
            target_mask = target_mask.flatten(0, 1)

        # (S) -> (S, 1)
        target_mask = target_mask.unsqueeze(-1)

        # (S, 1)
        loss = loss * target_mask

        # (S, 1)
        if smooth_loss is not None:
            smooth_loss = smooth_loss * target_mask

    if reduction == "sum":
        # ()
        loss = loss.sum()

        # ()
        if smooth_loss is not None:
            smooth_loss = smooth_loss.sum()
    elif reduction == "mean":
        # ()
        loss = loss.mean()

        # ()
        if smooth_loss is not None:
            smooth_loss = smooth_loss.mean()
    elif reduction != "none":
        raise ValueError(
            f"`reduction` must be 'sum', 'mean' or 'none', but is '{reduction}' instead."
        )

    if smooth_loss is not None:
        # This label smoothing implementation is identical to the one in fairseq
        # and varies slightly from PyTorch's version in `cross_entropy`.
        eps = label_smoothing / (log_probs.size(-1) - 1)

        loss = ((1.0 - label_smoothing - eps) * loss) + (eps * smooth_loss)

    if reduction == "none":
        # (S, 1) -> (S)
        loss = loss.squeeze(-1)

        if batch_size is not None:
            # (N x S) -> (N, S)
            loss = loss.unflatten(0, (batch_size, -1))

    return loss

@final
class BatchLayout:
    _width: int
    _seq_begin_indices: list[int]
    _seq_begin_indices_pt: Tensor
    _seq_lens: list[int]
    _seq_lens_pt: Tensor
    _position_indices: Tensor
    _min_seq_len: int
    _max_seq_len: int
    _packed: bool
    _padded: bool

    def __init__(
        self,
        shape: tuple[int, ...],
        seq_lens: Sequence[int] | None,
        *,
        packed: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self._packed = packed

        if packed:
            if len(shape) != 1:
                raise ValueError(
                    f"`shape` must be 1 dimensional, but is {len(shape)} dimensional instead."
                )

            batch_width = shape[0]

            if batch_width < 1:
                raise ValueError("`shape[0]` must be greater than or equal to 1.")

            if seq_lens is None:
                seq_lens = [batch_width]

            self._seq_begin_indices = [0]

            self._seq_lens = []

            self._position_indices = torch.arange(batch_width, device=device)

            self._num_elements = 0

            self._min_seq_len = batch_width
            self._max_seq_len = 0

            seq_beg = 0
            seq_end = 0

            for idx, seq_len in enumerate(seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                seq_end = seq_beg + seq_len

                if seq_end > batch_width:
                    raise ValueError(
                        f"`sum(seq_lens)` must be less than or equal to `shape[0]` ({batch_width}), but is {sum(seq_lens)} instead."
                    )

                self._seq_begin_indices.append(seq_end)

                self._seq_lens.append(seq_len)

                self._position_indices[seq_beg:seq_end] -= seq_beg

                self._min_seq_len = min(self._min_seq_len, seq_len)
                self._max_seq_len = max(self._max_seq_len, seq_len)

                seq_beg = seq_end

            self._position_indices[seq_end:] = -1  # pad

            self._padded = seq_end < batch_width
        else:
            if len(shape) != 2:
                raise ValueError(
                    f"`shape` must be 2 dimensional, but is {len(shape)} dimensional instead."
                )

            batch_size, batch_width = shape

            if batch_width < 1:
                raise ValueError("`shape[1]` must be greater than or equal to 1.")

            if seq_lens is None:
                seq_lens = [batch_width] * batch_size

            if len(seq_lens) != batch_size:
                raise ValueError(
                    f"`len(seq_lens)` must be equal to `shape[0]` ({batch_size}), but is {len(seq_lens)} instead."
                )

            self._seq_begin_indices = list(
                range(0, (batch_size * batch_width) + 1, batch_width)
            )

            self._seq_lens = []

            indices = torch.arange(batch_width, device=device)

            # (S) -> (N, S)
            self._position_indices = indices.expand(batch_size, -1).contiguous()

            self._min_seq_len = batch_width
            self._max_seq_len = 0

            self._padded = False

            for idx, seq_len in enumerate(seq_lens):
                if seq_len < 1:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be greater than or equal to 1, but the length at index {idx} is {seq_len} instead."
                    )

                if seq_len > batch_width:
                    raise ValueError(
                        f"All lengths in `seq_lens` must be less than or equal to `shape[1]` ({batch_width}), but the length at index {idx} is {seq_len} instead."
                    )

                self._seq_lens.append(seq_len)

                if seq_len < batch_width:
                    self._padded = True

                self._position_indices[idx, seq_len:] = -1  # pad

                self._min_seq_len = min(self._min_seq_len, seq_len)
                self._max_seq_len = max(self._max_seq_len, seq_len)

        self._width = batch_width

        self._seq_begin_indices_pt = to_tensor(
            self._seq_begin_indices, dtype=torch.int32, device=device
        )

        self._seq_lens_pt = to_tensor(self._seq_lens, dtype=torch.int32, device=device)

        # Both `seq_begin_indices` and `seq_lens` are inherently dynamic and
        # require to be marked so to avoid redundant recompilations.
        torch._dynamo.maybe_mark_dynamic(self._seq_begin_indices_pt, 0)
        torch._dynamo.maybe_mark_dynamic(self._seq_lens_pt, 0)

    @staticmethod
    def of(
        batch: Tensor, seq_lens: list[int] | None = None, *, packed: bool = False
    ) -> BatchLayout:
        shape = batch.shape[:1] if packed else batch.shape[:2]

        return BatchLayout(shape, seq_lens, packed=packed, device=batch.device)

    @property
    def width(self) -> int:
        return self._width

    @property
    def seq_begin_indices(self) -> Sequence[int]:
        return self._seq_begin_indices

    @property
    def seq_begin_indices_pt(self) -> Tensor:
        return self._seq_begin_indices_pt

    @property
    def seq_lens(self) -> Sequence[int]:
        return self._seq_lens

    @property
    def seq_lens_pt(self) -> Tensor:
        return self._seq_lens_pt

    @property
    def min_seq_len(self) -> int:
        return self._min_seq_len

    compiled_max_seq_len: ClassVar[int | None] = None

    @property
    def max_seq_len(self) -> int:
        # TODO: As of PyTorch 2.7, integers cannot be marked as dynamic during
        # compilation. This is a workaround till that gets fixed.
        if torch.compiler.is_compiling():
            if self.compiled_max_seq_len is not None:
                return self.compiled_max_seq_len

        return self._max_seq_len

    @property
    def position_indices(self) -> Tensor:
        return self._position_indices

    @property
    def padded(self) -> bool:
        return self._padded

    @property
    def packed(self) -> bool:
        return self._packed

    def __repr__(self) -> str:
        s = (
            f"width={self._width}, "
            f"seq_begin_indices={self._seq_begin_indices}, "
            f"seq_lens={self._seq_lens}, "
            f"min_seq_len={self._min_seq_len}, "
            f"max_seq_len={self._max_seq_len}, "
            f"padded={self._padded}, "
            f"packed={self._packed}"
        )

        return f"BatchLayout({s})"

class StandardLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        in_dtype = input.dtype
        input = input.to(torch.float32)
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        ).to(in_dtype)

def repeat_interleave(x: Tensor, dim: int, repeat: int) -> Tensor:
    """
    Repeats elements of a tensor.

    :param x: The input tensor.
    :param dim: The dimension along which to repeat values.
    :param repeat: The number of repetitions.

    :returns: The repeated tensor which has the same shape as input, except
        along the given axis.

    .. note::
        This is a lightweight version of :func:`torch.repeat_interleave` that
        is faster for repetitions along a single dimension.
    """
    if repeat == 1:
        return x

    shape = [-1] * (x.ndim + 1)

    if dim < 0:
        dim += x.ndim

    shape[dim + 1] = repeat

    return x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)



class InternalError(RuntimeError):
    """Internal error exception to replace external dependency."""
    pass


def get_name_or_self(obj: object) -> str:
    """Get name of object or return string representation."""
    if hasattr(obj, '__name__'):
        return obj.__name__
    elif hasattr(obj, '__class__'):
        return obj.__class__.__name__
    else:
        return str(obj)