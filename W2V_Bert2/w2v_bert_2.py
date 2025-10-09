from __future__ import annotations

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, final, Any, abstractmethod, TYPE_CHECKING, Protocol, override, Iterator
from torch import Tensor
from seamless_feature_extractor import SeamlessM4TFeatureExtractor
from vector_quantizer import GumbelWav2Vec2VectorQuantizer, Wav2Vec2VectorQuantizerOutput
from masker import StandardWav2Vec2Masker
from utils import BatchLayout, StandardLayerNorm, cross_entropy, compute_neg_counts, repeat_interleave


@dataclass
class Wav2Vec2Features:
    """Holds the extracted features of a wav2vec 2.0 model."""

    seqs: Tensor
    """The features. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is the
    batch size, :math:`S_{out}` is the output sequence length, and :math:`M` is
    the dimensionality of the model."""

    seqs_layout: BatchLayout

    targets: Tensor
    """The non-quantized resolver network targets that have been extracted from
    the input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    temporal_mask: Tensor
    """The temporal mask that has been used to extract the resolver network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    raw: Tensor
    """The raw features returned by the frontend. *Shape*: Same as :attr:`seqs`."""


@dataclass
class Wav2Vec2Output:
    """Holds the output of a wav2vec 2.0 model."""

    logits: Tensor
    """The logits for contrastive feature prediction. *Shape:*
    :math:`(N,S_{msk},L)`, where :math:`N` is the batch size, :math:`S_{msk}`
    is the masked sequence length, and :math:`L` is the number of candidates
    (i.e. the number of distractors plus 1 for the target)."""

    quantized_targets: Tensor
    """The quantized resolver network targets that have been extracted from the
    input sequences. *Shape:* :math:`(N,S_{msk},M)`, where :math:`N` is the
    batch size, :math:`S_{msk}` is the masked sequence length, and :math:`M` is
    the dimensionality of the model."""

    num_targets: int
    """The number of targets."""

    temporal_mask: Tensor
    """The temporal mask that has been applied to extract the resolver network
    targets. *Shape:* :math:`(N,S_{enc})`, where :math:`N` is the batch size and
    :math`S_{enc}` is the encoder output sequence length."""

    quantizer_output: Wav2Vec2VectorQuantizerOutput
    """The output of the vector quantizer."""

    encoder_output: Tensor
    """The resolver network output. *Shape:* :math:`(N,S_{enc},M)`, where
    :math:`N` is the batch size, :math:`S_{enc}` is the encoder output sequence
    length, and :math:`M` is the dimensionality of the model."""

    encoder_output_layout: BatchLayout

    raw_features: Tensor
    """The raw features returned by the frontend. *Shape*: Same as
    :attr:`encoder_output`."""


@dataclass
class W2VBertOutput:
    """Holds the output of a w2v-BERT model."""

    w2v2_output: Wav2Vec2Output
    """The output of the wav2vec 2.0 model."""

    bert_logits: Tensor
    """The logits for masked feature prediction. *Shape:*
    :math:`(NxS_{msk},V,G_{tgt})`, where :math:`N` is the batch size,
    :math:`S_{msk}` is the masked sequence length, :math:`V` is the number of
    entries per codebook, and :math:`G_{tgt}` is the number of target
    codebooks."""

    bert_targets: Tensor
    """The target entry index per target codebook. *Shape:*
    :math:`(NxS_{msk},G_{tgt})`, where :math:`N` is the batch size,
    :math:`S_{msk}` is the masked sequence length, and :math:`G_{tgt}` is the
    number of target codebooks."""

@dataclass
class Wav2Vec2Loss:
    aggregate: Tensor
    contrastive: Tensor
    diversity: Tensor
    features_penalty: Tensor

@dataclass
class W2VBertLoss:
    aggregate: Tensor
    bert: Tensor
    w2v2: Wav2Vec2Loss
    
# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeedForward with SeamlessM4T->SeamlessM4Tv2
class W2VBertConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        self.intermediate_dropout = nn.Dropout(dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        self.intermediate_act_fn = nn.SiLU()

        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states
    

class W2VBertConformerSelfAttention(nn.Module):
    """Construct a SeamlessM4Tv2ConformerSelfAttention object.
    Can be enhanced with relative position embeddings.
    """

    def __init__(self, config, use_position_embeddings=True):
        super().__init__()

        self.head_size = config.hidden_size // config.speech_encoder_attention_heads
        self.num_heads = config.speech_encoder_attention_heads
        self.position_embeddings_type = config.position_embeddings_type if use_position_embeddings else None

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.speech_encoder_dropout)

        self.left_max_position_embeddings = config.left_max_position_embeddings
        self.right_max_position_embeddings = config.right_max_position_embeddings
        num_positions = self.left_max_position_embeddings + self.right_max_position_embeddings + 1
        self.distance_embedding = nn.Embedding(num_positions, self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        # self-attention mechanism
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query_length, key_length = query.shape[2], key.shape[2]

        position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_r - position_ids_l
        distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)

        positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
        positional_embedding = positional_embedding.to(dtype=query.dtype)  # fp16 compatibility

        relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        
        attn_weights = attn_scores + (relative_position_attn_weights / math.sqrt(self.head_size))

        # apply attention_mask if necessary
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # => (batch, head, time1, time2)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # => (batch, head, time1, d_k)
        attn_output = torch.matmul(attn_weights, value)

        # => (batch, time1, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.linear_out(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class W2VBertConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block. Uses a causal depthwise convolution similar to that
    described in Section 2.1 of https://huggingface.co/papers/1609.03499
    """

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )
        self.depthwise_layer_norm = StandardLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Ensure that we do not leak padded positions in depthwise convolution.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.where(attention_mask.bool().unsqueeze(-1)==False, 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(-2, -1)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(-2, -1)).transpose(-2, -1)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)
        return hidden_states
    

class W2VBertConformerEncoderLayer(nn.Module):
    """Conformer block based on https://huggingface.co/papers/2005.08100."""

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4Tv2, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = StandardLayerNorm(embed_dim)
        self.ffn1 = W2VBertConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = StandardLayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = W2VBertConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module_layer_norm = StandardLayerNorm(embed_dim)
        self.conv_module = W2VBertConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = StandardLayerNorm(embed_dim)
        self.ffn2 = W2VBertConformerFeedForward(config)
        self.final_layer_norm = StandardLayerNorm(embed_dim)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        conv_attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = self.conv_module_layer_norm(hidden_states)
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


class TransformerEncoder(nn.Module):
    """Represents a Transformer encoder."""

    layers: nn.ModuleList

    def __init__(self) -> None:
        super().__init__()

        self._layer_hooks: dict[int, TransformerEncoderLayerHook] = OrderedDict()

    @abstractmethod
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """
        :param seqs: The sequences to encode. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        :param padding_mask: The padding mask of ``seqs``. *Shape:* :math:`(N,S)`,
            where :math:`N` is the batch size and :math:`S` is the sequence
            length.

        :returns: The encoder output. *Shape:* Same as ``seqs``.
        """

    if TYPE_CHECKING:
        __call__ = forward

    def register_layer_hook(self, hook: TransformerEncoderLayerHook) -> RemovableHandle:
        """
        Registers a layer hook on the module.

        The hook will be called every time after a layer in the encoder stack
        has computed an output.

        :param hook: The hook to register.

        :returns: A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._layer_hooks)

        self._layer_hooks[handle.id] = hook

        return handle


class TransformerEncoderLayerHook(Protocol):
    """
    Represents a hook to pass to :meth:`~TransformerEncoder.register_layer_hook`.
    """

    def __call__(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_output_layout: BatchLayout,
        num_layers: int,
    ) -> bool:
        """
        :param layer_idx: The index of the layer in the encoder stack.
        :param layer_output: The encoded output of the layer.
        :param num_layers: The number of layers in the encoder stack.

        :returns: ``True`` if the encoder should continue executing the
            remaining layers in the stack; ``False`` if the encoder should treat
            this layer as the final layer in the stack.
        """

@final
class W2VBert2Encoder(TransformerEncoder):
    def __init__(
        self,
        layers: nn.ModuleList,
        layer_norm: StandardLayerNorm | None = None,
        *,
        layer_drop_p: float = 0.0,
        generator: torch.Generator | None = None,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(layers)

        self.layer_drop_p = layer_drop_p

        self.generator = generator

        self.layer_norm: StandardLayerNorm | None

        self.register_module("layer_norm", layer_norm)

        if dropout_p > 0.0:
            dropout = nn.Dropout(dropout_p)
        else:
            dropout = None

        self.dropout: nn.Dropout | None

        self.register_module("dropout", dropout)

    @override
    def forward(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        if self._layer_hooks:
            if self.training and self.layer_drop_p > 0.0:
                raise InvalidOperationError(
                    "The layer output hooks cannot be run when LayerDrop is enabled."
                )

        num_layers = len(self.layers)

        for layer_idx, (layer, drop) in enumerate(self._drop_iter()):
            layer_output = layer(seqs, seqs_layout)

            if drop:
                seqs = _record_drop_for_backward(seqs, layer_output)

                continue

            seqs = layer_output

            for hook in self._layer_hooks.values():
                if not hook(layer_idx, seqs, seqs_layout, num_layers):
                    break

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs

    def _drop_iter(self) -> Iterator[tuple[nn.Module, bool]]:
        if self.training and self.layer_drop_p > 0.0:
            prob_dist = torch.rand(
                len(self.layers), generator=self.generator, device=torch.device("cpu")
            )
        else:
            prob_dist = None

        for idx, m in enumerate(self.layers):
            drop = prob_dist is not None and float(prob_dist[idx]) <= self.layer_drop_p

            yield m, drop

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        if self.layer_drop_p > 0.0:
            return f"layer_drop_p={self.layer_drop_p:G}"

        return ""


def _record_drop_for_backward(x: Tensor, dropped_output: Tensor) -> Tensor:
    return _RecordDropForBackwardFunction.apply(x, dropped_output)  # type: ignore[no-any-return]


class _RecordDropForBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, dropped_output: Tensor) -> Tensor:
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        return grad_output, torch.zeros_like(grad_output)


class W2VBert2Model(nn.Module):
    def __init__(
        self,
        config,
        num_target_codebooks: int = 2,
        quantizer_encoder_grad: bool = True,
        final_proj_bias: bool = True,
        num_distractors: int = 100,
        logit_temp: float = 0.1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.num_target_codebooks = num_target_codebooks
        self.quantizer_encoder_grad = quantizer_encoder_grad
        self.final_proj_bias = final_proj_bias
        self.num_distractors = num_distractors
        self.logit_temp = logit_temp

        self.encoder_frontend = Wav2VecBert2Frontend(
            config.model_dim,
            config.feature_size,
            SeamlessM4TFeatureExtractor(),
            device=device,
            dtype=dtype,
        )

        self.masker = StandardWav2Vec2Masker(
            config.model_dim,
            config.temporal_span_len,
            config.max_temporal_mask_prob,
            config.min_num_temporal_mask_spans,
            config.spatial_span_len,
            config.max_spatial_mask_prob,
            config.min_num_spatial_mask_spans,
        )
        self.quantizer = GumbelWav2Vec2VectorQuantizer(
            config.feature_size,
            config.hidden_size,
            config.num_codebooks,
            config.num_codebook_entries,
            config.codebook_sampling_temperature,
        )

        self.w2v_bert_encoder = W2VBert2Encoder(
            nn.ModuleList([W2VBertConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)])
        )

        # project hidden states to quantizer output dim
        self.final_w2v2_proj = nn.Linear(
            config.hidden_size,
            config.quantizer.output_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # output projection for quantizer
        self.final_target_quantizer_proj = nn.Linear(
            config.quantizer.output_dim,
            config.quantizer.output_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # linear head for mask prediction task
        self.final_bert_proj = nn.Linear(
            config.hidden_size,
            config.quantizer.num_codebook_entries * config.num_targets_codebooks,
            bias=True,
            device=device,
            dtype=dtype,
        )

        self.num_distractors = config.num_distractors
        self.num_targets_codebooks = config.num_targets_codebooks
        self.num_bert_encoder_layers = config.num_bert_encoder_layers

    def forward(
        self,
        raw_features: torch.Tensor,
        seqs_layout: BatchLayout,
        *,
        diversity_weight: float = 0.1,
        features_penalty_weight: float = 10.0,
    ) -> tuple[Wav2Vec2Loss, Wav2Vec2Output]:

        w2v2_features = self.run_frontend(raw_features, seqs_layout)
        # hook for contrastive learning
        def hook(
            layer_idx: int,
            layer_output: Tensor,
            layer_output_layout: BatchLayout,
            num_layers: int,
        ) -> bool:
            if layer_idx == num_layers - self.num_bert_encoder_layers - 1:
                w2v2_features.seqs = layer_output

            return True

        with self.w2v_bert_encoder.register_layer_hook(hook):
            encoder_output = self.w2v_bert_encoder(
                w2v2_features.seqs, w2v2_features.seqs_layout
            )

        w2v2_output = self.quantize_and_contrast(w2v2_features)

        seqs = Wav2Vec2Masker.extract_masked_elements(
            encoder_output, w2v2_features.temporal_mask
        )

        bert_logits = self.final_bert_proj(seqs)

        # (N, S_msk, V x G) -> (N x S_msk, V, G)
        bert_logits = bert_logits.view(
            -1,
            self.quantizer.num_codebook_entries,
            self.num_target_codebooks,
        )

        bert_targets = self._get_target_indices(w2v2_output.quantizer_output)     

        bert_output = W2VBertOutput(w2v2_output, bert_logits, bert_targets)   
    
        loss = self.compute_loss(
            bert_output,
            bert_weight=bert_weight,
            bert_label_smoothing=bert_label_smoothing,
            w2v2_weight=w2v2_weight,
            w2v2_diversity_weight=w2v2_diversity_weight,
            w2v2_features_penalty_weight=w2v2_features_penalty_weight,
        )

        return loss, bert_output

    def _get_target_indices(
        self, quantizer_output: Wav2Vec2VectorQuantizerOutput
    ) -> Tensor:
        num_codebooks = self.quantizer.num_codebooks

        batch_size, seq_len = quantizer_output.quantized_vectors.shape[:2]

        cb = quantizer_output.cb.view(batch_size * seq_len * num_codebooks, -1)

        indices = cb.argmax(dim=-1).view(-1, num_codebooks)

        indices = indices[..., :num_codebooks]

        return indices.detach()

    def run_frontend(self, seqs: Tensor, seqs_layout: BatchLayout) -> Wav2Vec2Features:
        """Run the encoder frontend in pretraining mode.

        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,*)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`*`
            is any number of sequence-specific dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        """
        frontend = self.encoder_frontend

        seqs, seqs_layout, raw_features = frontend.extract_features(seqs, seqs_layout)

        # We use the extracted features as resolver network targets after masking
        # and quantization.
        if self.quantizer_encoder_grad:
            targets = seqs.clone()
        else:
            targets = seqs.detach().clone()

        if frontend.first_pass_dropout is not None:
            targets = frontend.first_pass_dropout(targets)

        seqs, temporal_mask = frontend.process_features(seqs, seqs_layout, self.masker)

        if temporal_mask is None:
            raise InternalError("`temporal_mask` is `None`.")

        targets = Wav2Vec2Masker.extract_masked_elements(targets, temporal_mask)

        return Wav2Vec2Features(seqs, seqs_layout, targets, temporal_mask, raw_features)

    def quantize_and_contrast(self, features: Wav2Vec2Features) -> Wav2Vec2Output:
        """Quantize targets and produce logits for contrastive prediction.

        :param features:
            The extracted features from the encoder.
        """
        encoder_output, encoder_output_layout, targets, temporal_mask = (
            features.seqs,
            features.seqs_layout,
            features.targets,
            features.temporal_mask,
        )

        masked_encoder_features = Wav2Vec2Masker.extract_masked_elements(encoder_output, temporal_mask)

        masked_proj_features = self.final_w2v2_proj(masked_encoder_features)

        quantizer_output = self.quantizer(targets)

        targets = self.final_target_quantizer_proj(targets)

        negatives = self._sample_negatives(targets)

        logits = self._compute_logits(masked_proj_features, targets, negatives)

        batch_size, seq_len = logits.shape[:2]

        num_targets = batch_size * seq_len

        return Wav2Vec2Output(
            logits,
            targets,
            num_targets,
            temporal_mask,
            quantizer_output,
            encoder_output,
            encoder_output_layout,
            features.raw,
        )

    def _sample_negatives(self, targets: Tensor) -> Tensor:
        """Sample negatives from the target quantization vectors."""
        batch_size, seq_len, model_dim = targets.shape

        device = targets.device

        num_utterance_negatives, num_batch_negatives = compute_neg_counts(seq_len, self.num_negatives)

        # (N, S, M) -> (N x S, M)
        targets = targets.view(-1, model_dim)
        # (S)
        indices = torch.arange(seq_len, device=device)

        with torch.no_grad():
            # sample negatives from the same utterance
            if num_utterance_negatives > 0:
                # (S) -> (S x L)
                utterance_indices = repeat_interleave(indices, dim=0, repeat=num_utterance_negatives)
                # (N, S x L)
                utterance_rand_indices = torch.randint(
                    low=0,
                    high=seq_len - 1,
                    size=(batch_size, seq_len * num_utterance_negatives),
                    device=device,
                )
                # (N, S x L)
                utterance_rand_indices[utterance_rand_indices >= indices] += 1
            # sample negatives from batch
            if num_batch_negatives > 0:
                # (S) -> (S x L)
                batch_indices = repeat_interleave(indices, dim=0, repeat=num_batch_negatives)
                # (N, S x L)
                batch_rand_indices = torch.randint(
                    low=0,
                    high=batch_size * seq_len - 1,
                    size=(batch_size, seq_len * num_batch_negatives),
                    device=device,
                )
                # (N, S x L)
                batch_rand_indices[batch_rand_indices >= indices] += 1
        
        if num_utterance_negatives > 0:
            # (N, 1)
            k = torch.arange(batch_size, device=device).unsqueeze(1) * seq_len
            # (N, S x L)
            neg_indices = utterance_rand_indices + k
        else:
            neg_indices = batch_rand_indices

        # (N, S x L) -> (N x S x L)
        neg_indices = neg_indices.view(-1)

        # (N x S x L, M)
        negatives = targets[neg_indices]

        # (N x S x L) -> (N, S, L, M)
        negatives = negatives.view(
            batch_size, seq_len, num_utterance_negatives + num_batch_negatives, model_dim
        )

        return negatives

    def _compute_logits(
            self, seqs: Tensor, targets: Tensor, negatives: Tensor
        ) -> Tensor:
        """Compute the logits for the contrastive prediction."""
        # (N, S, M) -> (N, S, 1, M)
        seqs, targets = seqs.unsqueeze(2), targets.unsqueeze(2)

        # The target will be always at index 0 in the candidate list.
        # (N, S, 1, M) + (N, S, L, M) -> (N, S, L + 1, M)
        candidates = torch.cat([targets, negatives], dim=2)

        # Perform in fp32.
        # (N, S, L + 1, M) -> (N, S, L + 1)
        logits = torch.cosine_similarity(seqs.float(), candidates.float(), dim=-1)

        if self.logit_temp != 1.0:
            logits = logits / self.logit_temp

        eps = 1e-6
        negative_is_target = (torch.abs(targets - negatives) <= eps).all(-1)

        # If `True`, it means codebook utilization is low. In such case we
        # mask the corresponding logits.
        if negative_is_target.any():
            logits[:, :, 1:][negative_is_target] = -torch.inf

        return logits

    def compute_loss(
        self,
        output: W2VBertOutput,
        *,
        bert_weight: float = 1.0,
        bert_label_smoothing: float = 0.0,
        w2v2_weight: float = 1.0,
        w2v2_diversity_weight: float = 0.1,
        w2v2_features_penalty_weight: float = 10.0,
    ) -> W2VBertLoss:
        bert_loss = cross_entropy(
            output.bert_logits,
            output.bert_targets,
            pad_idx=None,
            reduction="sum",
            label_smoothing=bert_label_smoothing,
        )

        w2v2_loss = self.compute_w2v2_loss(
            output.w2v2_output,
            diversity_weight=w2v2_diversity_weight,
            features_penalty_weight=w2v2_features_penalty_weight,
        )

        weighted_bert_loss = bert_weight * bert_loss
        weighted_w2v2_loss = w2v2_weight * w2v2_loss.aggregate

        return W2VBertLoss(
            weighted_bert_loss + weighted_w2v2_loss, bert_loss, w2v2_loss
        )

    def compute_w2v2_loss(
        self,
        output: Wav2Vec2Output,
        *,
        diversity_weight: float = 0.1,
        feature_penalty_weight: float = 10.0,
    ) -> Wav2Vec2Loss:
        
        contrastive_loss = self.compute_contrastive_loss(output.logits)

        diversity_loss = self.compute_diversity_loss(
            output.logits, output.quantizer_output.prob_perplexity
        )

        features_penalty = self.compute_features_penalty(
            output.logits, output.raw_features
        )

        weighted_diversity_loss = diversity_weight * diversity_loss

        weighted_features_penalty = features_penalty_weight * features_penalty

        aggregate_loss = (
            contrastive_loss + weighted_diversity_loss + weighted_features_penalty
        )

        return Wav2Vec2Loss(
            aggregate_loss, contrastive_loss, diversity_loss, features_penalty
        )

    def compute_contrastive_loss(
        self, logits: Tensor
    ) -> Tensor:
        batch_size, seq_len, num_logits = logits.shape

        # (N, S, L) -> (S x N, L)
        logits = logits.transpose(0, 1).reshape(-1, num_logits)

        # For numerical stability in low-precision.
        logits = logits.float()

        # The target is always at index 0 in the candidate list.
        targets = logits.new_zeros((batch_size * seq_len,), dtype=torch.int64)

        return F.cross_entropy(logits, targets, reduction="sum")

    def compute_diversity_loss(self, logits: Tensor, prob_perplexity: Tensor) -> Tensor:
        num_entries = self.quantizer.num_codebooks * self.quantizer.num_codebook_entries

        quantizer_loss = (num_entries - prob_perplexity) / num_entries

        batch_size, seq_len = logits.shape[:2]

        return quantizer_loss * batch_size * seq_len

    def compute_features_penalty(self, logits: Tensor, raw_features: Tensor) -> Tensor:
        batch_size, seq_len = logits.shape[:2]

        return raw_features.float().pow(2).mean() * batch_size * seq_len
