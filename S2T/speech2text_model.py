# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SeamlessM4Tv2 model."""

import copy
import math
import logging
from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from utils import (
    compute_token_kd_loss, 
    is_fsdp_managed_module,
    ClassInstantier,
    shift_tokens_right,
    create_position_ids_from_input_ids,
    _compute_new_attention_mask,
    ACT2FN,
    GradientCheckpointingLayer,
)

from seamless_m4t_v2_config import SeamlessM4Tv2Config

# Transformers
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask


logger = logging.getLogger(__name__)

SEAMLESS_M4T_V2_COMMON_CUSTOM_ARGS = r"""
    input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
        Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
        [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.
    decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        Indices of decoder input sequence tokens in the vocabulary.

        Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
        [`PreTrainedTokenizer.__call__`] for details.

        [What are decoder input IDs?](../glossary#decoder-input-ids)

        Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
        is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

        For translation and summarization training, `decoder_input_ids` should be provided. If no
        `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
        for denoising pre-training following the paper.
    decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
        be used by default.

        If you want to change padding behavior, you should read [`modeling_bart._prepare_decoder_attention_mask`]
        and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
        information on the default strategy.
    inputs_embeds (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*):
        Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
        is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
        model's internal embedding lookup matrix.
"""


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TGenerationOutput with SeamlessM4T->SeamlessM4Tv2
@dataclass
class SeamlessM4Tv2GenerationOutput:
    r"""
    waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        The final audio waveform predicted by the model.
    waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
        The length in samples of each element in the `waveform` batch.
    sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
        The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
        early due to the `eos_token_id`.
    unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
        The generated translated unit sequences. This is the output of the text-to-units model. The second
        dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
        early due to the `t2u_eos_token_id`.
    """

    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[tuple[torch.FloatTensor]] = None


class SeamlessM4Tv2ConformerFeatureProjection(nn.Module):
    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeatureProjection.__init__
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.speech_encoder_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states.to(self.layer_norm.weight.dtype))
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerFeedForward with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerFeedForward(nn.Module):
    def __init__(self, config, act_fn=None, dropout=None):
        super().__init__()
        dropout = dropout if dropout is not None else config.speech_encoder_dropout
        act_fn = act_fn if act_fn is not None else config.speech_encoder_hidden_act

        self.intermediate_dropout = nn.Dropout(dropout)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.speech_encoder_intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.activation_function]

        self.output_dense = nn.Linear(config.speech_encoder_intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class SeamlessM4Tv2ConformerSelfAttention(nn.Module):
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



class SeamlessM4Tv2ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block. Uses a causal depthwise convolution similar to that
    described in Section 2.1 of https://huggingface.co/papers/1609.03499
    """

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
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
        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.speech_encoder_hidden_act]
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
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class SeamlessM4Tv2ConformerEncoderLayer(GradientCheckpointingLayer):
    """Conformer block based on https://huggingface.co/papers/2005.08100."""

    # Copied from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer.Wav2Vec2ConformerEncoderLayer.__init__ with Wav2Vec2->SeamlessM4Tv2, attention_dropout->speech_encoder_dropout, torch.nn->nn
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.speech_encoder_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = SeamlessM4Tv2ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = SeamlessM4Tv2ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = SeamlessM4Tv2ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

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
        hidden_states = self.conv_module(hidden_states, attention_mask=conv_attention_mask)
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


class SeamlessM4Tv2ConformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(config.speech_encoder_dropout)
        self.layers = nn.ModuleList(
            [SeamlessM4Tv2ConformerEncoderLayer(config) for _ in range(config.speech_encoder_layers)]
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def _apply_chunk_attention(self, attention_mask, hidden_states):
        """
        Creates a chunk attention mask. It creates a mask to prevent attention across chunks, ensuring that each
        position attends only to positions within its own chunk. If a left chunk overlap is specified
        (`speech_encoder_chunk_size` in the configuration), the attention mask is adjusted accordingly to allow each
        position to also attends the `speech_encoder_chunk_size - 1` previous chunks.
        """
        sequence_len = hidden_states.shape[1]

        chunk_indices = torch.arange(sequence_len, device=hidden_states.device)
        chunk_indices = torch.div(chunk_indices, self.config.speech_encoder_chunk_size).long()

        start_indices = torch.full_like(chunk_indices, 0)
        if self.config.speech_encoder_left_chunk_num >= 0:
            start_indices = (chunk_indices - self.config.speech_encoder_left_chunk_num).clamp_(min=0)
            start_indices = start_indices * self.config.speech_encoder_chunk_size
            start_indices = start_indices
        start_indices = start_indices.unsqueeze(1).expand(-1, sequence_len)

        end_indices = ((chunk_indices + 1) * self.config.speech_encoder_chunk_size).clamp_(max=sequence_len)

        end_indices = end_indices.unsqueeze(1).expand(-1, sequence_len)

        indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)

        chunk_mask = (indices < start_indices) | (indices >= end_indices)
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)

        attention_mask = chunk_mask if attention_mask is None else (attention_mask.bool() | chunk_mask)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return attention_mask

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        if self.config.speech_encoder_chunk_size is not None:
            attention_mask = self._apply_chunk_attention(attention_mask, hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min

        hidden_states = self.dropout(hidden_states)

        synced_gpus =   is_fsdp_managed_module(self)

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.speech_encoder_layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    conv_attention_mask=conv_attention_mask,
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions.append(layer_outputs[1])

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return (
            hidden_states,
            all_hidden_states,
            all_self_attentions,
        )


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapterLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.adaptor_dropout

        self.kernel_size = config.adaptor_kernel_size
        self.stride = config.adaptor_stride

        # 1. residual convolution
        self.residual_layer_norm = nn.LayerNorm(embed_dim)
        self.residual_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.activation = nn.GLU(dim=1)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_conv = nn.Conv1d(
            embed_dim,
            2 * embed_dim,
            self.kernel_size,
            stride=self.stride,
            padding=self.stride // 2,
        )
        self.self_attn = SeamlessM4Tv2ConformerSelfAttention(config, use_position_embeddings=False)
        self.self_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=dropout)

    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        pad = self.kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1

        return seq_lens.floor()

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = self.residual_layer_norm(hidden_states)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.activation(residual)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        residual = residual.transpose(1, 2)

        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Apply pooling before feeding to the multihead-attention layer.
        # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.self_attn_conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        # (batch, feature_dim, seq_len) -> (batch, seq_len, feature_dim)
        hidden_states = hidden_states.transpose(1, 2)

        if attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(attention_mask).to(
                hidden_states.device
            )
            attention_mask = _compute_new_attention_mask(hidden_states=hidden_states, seq_lens=sub_sampled_lengths)
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                hidden_states.dtype,
            )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states) + residual

        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TConformerAdapter with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2ConformerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            SeamlessM4Tv2ConformerAdapterLayer(config) for _ in range(config.num_adapter_layers)
        )

    def forward(self, hidden_states, attention_mask):
        # down project hidden_states if necessary

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


############ TEXT / UNITS related code ################


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100ScaledWordEmbedding with M2M100->SeamlessM4Tv2
class SeamlessM4Tv2ScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


# Copied from transformers.models.m2m_100.modeling_m2m_100.M2M100SinusoidalPositionalEmbedding
class SeamlessM4Tv2SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ):
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()

    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length


class SeamlessM4Tv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.bart.modeling_bart.BartAttention.__init__ with Bart->SeamlessM4Tv2
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[SeamlessM4Tv2Config] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = encoder_hidden_states is not None
        batch_size, seq_length = hidden_states.shape[:2]

        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.layers[self.layer_idx].keys
            value_states = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # using torch sdpa
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=self.is_causal, scale=self.scaling
        )
        context_states = sdpa_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.out_proj(context_states)

        return attn_output


# Copied from transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeDenseActDense with NllbMoe->SeamlessM4Tv2,DenseActDense->FeedForwardNetwork, d_model->hidden_size
class SeamlessM4Tv2FeedForwardNetwork(nn.Module):
    def __init__(self, config: SeamlessM4Tv2Config, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoderLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SeamlessM4Tv2Config, encoder_ffn_dim=None, encoder_attention_heads=None):
        super().__init__()
        encoder_ffn_dim = config.encoder_ffn_dim if encoder_ffn_dim is None else encoder_ffn_dim
        encoder_attention_heads = (
            config.encoder_attention_heads if encoder_attention_heads is None else encoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=encoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoderLayer with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2DecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self, config: SeamlessM4Tv2Config, decoder_ffn_dim=None, decoder_attention_heads=None, layer_idx=None
    ):
        super().__init__()
        decoder_ffn_dim = config.decoder_ffn_dim if decoder_ffn_dim is None else decoder_ffn_dim
        decoder_attention_heads = (
            config.decoder_attention_heads if decoder_attention_heads is None else decoder_attention_heads
        )

        self.embed_dim = config.hidden_size
        self.self_attn = SeamlessM4Tv2Attention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.attn_dropout = nn.Dropout(config.dropout)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attention = SeamlessM4Tv2Attention(
            self.embed_dim,
            decoder_attention_heads,
            config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
        )
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)

        self.ffn = SeamlessM4Tv2FeedForwardNetwork(config, ffn_dim=decoder_ffn_dim)

        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.activation_dropout)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`):
                encoder attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by
                very large negative values
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attention_layer_norm(hidden_states)

            hidden_states, cross_attn_weights = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cache_position=cache_position,
            )
            hidden_states = self.attn_dropout(hidden_states)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.ffn_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights, cross_attn_weights


class SeamlessM4Tv2PreTrainedModel(PreTrainedModel):
    config: SeamlessM4Tv2Config
    base_model_prefix = "seamless_m4t_v2"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "SeamlessM4Tv2DecoderLayer",
        "SeamlessM4Tv2ConformerEncoderLayer",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TPreTrainedModel._compute_sub_sample_lengths_from_attention_mask
    def _compute_sub_sample_lengths_from_attention_mask(self, attention_mask):
        kernel_size, stride = self.config.adaptor_kernel_size, self.config.adaptor_stride
        pad = kernel_size // 2
        seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)

        seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1

        return seq_lens.floor()

    def _indices_to_subwords(self, input_ids):
        """
        Returns the corresponding text string for each input id.
        """
        if not hasattr(self.generation_config, "id_to_text"):
            raise ValueError(
                """This model generation config doesn't have a `id_to_text` key which maps
                token ids to subwords. Make sure to load the right generation config."""
            )
        batch_size, sequence_len = input_ids.shape

        subwords_batch = []
        for batch_id in range(batch_size):
            subwords = []
            for i in range(sequence_len):
                subword = self.generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
                subwords.append(str(subword))
            subwords_batch.append(subwords)
        return subwords_batch

    def _count_character_length_in_subword(
        self,
        input_ids,
        subwords_batch,
        merge_space_with_prev_subword=False,
        pad_token_id=0,
        unk_token_id=1,
        space="▁",
    ):
        """
        Counts the number of characters per text string associated with the input token id.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`list[list[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            merge_space_with_prev_subword (`bool`, *optional*, defaults to `False`):
                Indicates if the space character is merged with the previous subword. If `False`, it will be merged
                with the next subword.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
            space (`str`, *optional*, defaults to `"▁"`):
                The space character.
        """
        batch_size, _ = input_ids.shape

        char_count_per_id = input_ids.new_zeros(input_ids.size())

        subword_lens = input_ids.ne(pad_token_id).sum(1)

        for batch_id in range(batch_size):
            # We slice out the tensor till the padding index.
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]

            is_next_start_with_space = [
                len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space if i < len(subwords) - 1 else False
                for i in range(len(subwords))
            ]
            is_punc = [
                len(subwords[i]) == 1
                and not subwords[i].isalpha()
                and not subwords[i].isnumeric()
                and subwords[i] != space
                for i in range(len(subwords))
            ]
            for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
                if subword_idx == pad_token_id:
                    break

                if subword_idx == unk_token_id:
                    # We set char_len to 1 for an unk token.
                    char_len = 1

                    if merge_space_with_prev_subword and is_next_start_with_space[i]:
                        char_len += 1
                else:
                    # By default, spaces are merged with the next subword.
                    # char_len includes the space.
                    char_len = len(subword)

                    if merge_space_with_prev_subword:
                        # Add the space for the next subword.
                        if is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the current subword.
                        if i > 0 and is_next_start_with_space[i - 1]:
                            char_len -= 1
                    else:
                        # Merge space with punctuation mark by default.
                        if is_punc[i] and is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the subword succeeding the punctuation mark.
                        elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
                            char_len -= 1

                char_count_per_id[batch_id, i] = char_len

        return char_count_per_id

    def _get_char_input_ids(self, input_ids, subwords_batch, char_count_per_id, pad_token_id=0, unk_token_id=1):
        """
        Returns the corresponding character input id for each character of `subwords_batch`.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            subwords_batch (`list[list[str]]` of shape `(batch_size, sequence_length)`):
                Corresponding text string for each input id.
            char_count_per_id (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Number of characters per input id.
            pad_token_id (`int`, *optional*, defaults to 0):
                The id of the _padding_ text token. If it is encountered when calculating the length of a subword
                sample, the lengths of subsequent subwords will be set to 0.
            unk_token_id (`int`, *optional*, defaults to 1):
                The id of the _unknown_ text token. Associated to a subword of length 1.
        Returns:
            `torch.Tensor`: Tensor of shape `(batch_size, char_sequence_length)` containing the id of each character.
        """
        if not hasattr(self.generation_config, "char_to_id"):
            raise ValueError(
                """This model generation config doesn't have a `char_to_id` key which maps
                characters to character ids. Make sure to load the right generation config."""
            )

        batch_size = input_ids.shape[0]
        max_len = int(char_count_per_id.sum(1).max().item())

        char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)

        subword_lens = input_ids.ne(pad_token_id).sum(1)

        for batch_id in range(batch_size):
            total = 0
            subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
            subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == unk_token_id:
                    char_ids = [unk_token_id]
                else:
                    # Get char token indices corresponding to the subwords.
                    char_ids = [self.generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
                char_seq_len = len(char_ids)
                char_seqs[batch_id, total : total + char_seq_len] = torch.tensor(char_ids).to(char_seqs)
                total += char_seq_len
        return char_seqs

    def _hard_upsample(self, hidden_states, durations):
        """
        Repeats the time dimension of each sample in the batch based on the corresponding duration.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, *)`, *optional*):
                The sequence to repeat, where `*` is any number of sequence-specific dimensions including none.
            durations (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indicates how many times to repeat time segments.
        """
        if hidden_states.size(0) == 1:
            hidden_states = torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
        else:
            # if batched sample, need to interleave per sample, and pad -> loss of parallelism
            if hidden_states.shape[0] > 1 and self.training:
                logger.warning_once(
                    """`self.training=True` and you use batching. You lose parallelism during the hifigan
                               forward pass because the samples are interleaved."""
                )
            hidden_states = [
                torch.repeat_interleave(hidden_state, duration, dim=0)
                for (hidden_state, duration) in zip(hidden_states, durations)
            ]

            hidden_states = nn.utils.rnn.pad_sequence(hidden_states, batch_first=True)

        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TSpeechEncoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2SpeechEncoder(SeamlessM4Tv2PreTrainedModel):
    main_input_name = "input_features"

    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)

        self.feature_projection = SeamlessM4Tv2ConformerFeatureProjection(config)
        self.encoder = SeamlessM4Tv2ConformerEncoder(config)
        self.intermediate_ffn = SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu", dropout=0.0)
        self.adapter = SeamlessM4Tv2ConformerAdapter(config) if config.add_adapter else None
        self.inner_layer_norm = nn.LayerNorm(config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, torch.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError(
                """Both `input_features` and `inputs_embeds` are `None` in `SeamlessM4Tv2SpeechEncoder.forward`.
                Make sure one of them is not `None`."""
            )

        hidden_states = self.feature_projection(input_features)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        expanded_hidden_states = self.intermediate_ffn(hidden_states)
        hidden_states = hidden_states + 0.5 * expanded_hidden_states

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states, attention_mask=attention_mask)

        hidden_states = self.inner_layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return hidden_states


# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TEncoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2Encoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
        is_t2u_encoder: bool = False,
    ):
        r"""
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
        is_t2u_encoder (`bool`, *optional*, defaults to `False`):
            indicates if it belongs to the text-to-units model, in which case it won't have input embeddings
        """
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        embed_dim = config.hidden_size

        self.is_t2u_encoder = is_t2u_encoder
        self.max_source_positions = config.max_position_embeddings

        if not self.is_t2u_encoder:
            embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            self.embed_tokens = SeamlessM4Tv2ScaledWordEmbedding(
                config.vocab_size, embed_dim, self.padding_idx, embed_scale=embed_scale
            )

            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight

            self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
            )

        layers = []
        for _ in range(config.encoder_layers):
            layers.append(
                SeamlessM4Tv2EncoderLayer(
                    config,
                    encoder_attention_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, torch.Tensor]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and self.is_t2u_encoder:
            raise ValueError(
                "You cannot pass input_ids to the encoder of the text_to_units model. Pass inputs_embeds instead."
            )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.is_t2u_encoder:
            embed_pos = self.embed_positions(input)

            hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.device)
        else:
            hidden_states = inputs_embeds

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                )

                hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states



# Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TDecoder with SeamlessM4T->SeamlessM4Tv2
class SeamlessM4Tv2Decoder(SeamlessM4Tv2PreTrainedModel):
    def __init__(
        self,
        config: SeamlessM4Tv2Config,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        r"""
        embed_tokens (`nn.Embedding`, *optional*):
            Input embedding
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # if embed_tokens defined, use its shape instead
            self.embed_tokens = SeamlessM4Tv2ScaledWordEmbedding(
                embed_tokens.num_embeddings, embed_tokens.embedding_dim, self.padding_idx, embed_scale=embed_scale
            )
            self.embed_tokens.weight = embed_tokens.weight
        else:
            self.embed_tokens = SeamlessM4Tv2ScaledWordEmbedding(
                self.vocab_size, config.hidden_size, self.padding_idx, embed_scale=embed_scale
            )

        self.embed_positions = SeamlessM4Tv2SinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        layers = []
        for i in range(config.decoder_layers):
            layers.append(
                SeamlessM4Tv2DecoderLayer(
                    config,
                    decoder_attention_heads=config.decoder_attention_heads,
                    decoder_ffn_dim=config.decoder_ffn_dim,
                    layer_idx=i,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # retrieve input_ids
        if input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            raise ValueError("You have to specify decoder_input_ids")

        inputs_embeds = self.embed_tokens(input_ids)

        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, 0
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
            )
            hidden_states = layer_outputs

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class SeamlessM4Tv2ForSpeechToTextTrain_Pivot(SeamlessM4Tv2PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_missing = ["text_encoder", "t2u_model", "vocoder"]
    main_input_name = "input_features"

    _tied_weights_keys = [
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_decoder.embed_tokens.weight",
    ]

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.__init__ with SeamlessM4T->SeamlessM4Tv2
    def __init__(self, config: SeamlessM4Tv2Config):
        super().__init__(config)
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.speech_encoder = SeamlessM4Tv2SpeechEncoder(config)
        self.text_encoder = SeamlessM4Tv2Encoder(config, self.shared)  # Add text encoder for KD
        self.text_decoder = SeamlessM4Tv2Decoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # No post_init needed for nn.Module (not PreTrainedModel)

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_encoder
    def get_encoder(self):
        return self.speech_encoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_decoder
    def get_decoder(self):
        return self.text_decoder

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.get_input_embeddings
    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.set_input_embeddings
    def set_input_embeddings(self, value):
        self.text_decoder.embed_tokens = value

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText._tie_weights
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    def load_pretrained_weights(self, model_name_or_path: str, cache_dir: Optional[str] = None):
        """
        Load pretrained weights from HuggingFace model or local checkpoint.
        
        Args:
            model_name_or_path: HuggingFace model name (e.g., "facebook/seamless-m4t-v2-large") 
                               or path to local checkpoint
            cache_dir: Directory to cache downloaded model (optional, for HF models)
        
        Returns:
            Dict with loading statistics (missing_keys, unexpected_keys, etc.)
        """
        import os
        
        logger.info(f"Loading pretrained weights from {model_name_or_path}")
        
        # Check if it's a local path or HuggingFace model
        if os.path.exists(model_name_or_path):
            return self._load_from_local_checkpoint(model_name_or_path)
        else:
            return self._load_from_huggingface(model_name_or_path, cache_dir)
    
    def _load_from_huggingface(self, model_name: str, cache_dir: Optional[str] = None):
        """Load weights from HuggingFace pretrained model"""
        try:
            from transformers import SeamlessM4Tv2Model
        except ImportError:
            raise ImportError(
                "transformers library required to load HuggingFace models. "
                "Install with: pip install transformers"
            )
        
        logger.info(f"Downloading model from HuggingFace: {model_name}")
        hf_model = SeamlessM4Tv2Model.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        stats = {
            'speech_encoder': {'missing': [], 'unexpected': []},
            'text_encoder': {'missing': [], 'unexpected': []},
            'text_decoder': {'missing': [], 'unexpected': []},
            'other': {'missing': [], 'unexpected': []}
        }
        
        # Load speech encoder
        logger.info("Loading speech_encoder weights...")
        missing, unexpected = self.speech_encoder.load_state_dict(
            hf_model.speech_encoder.state_dict(), 
            strict=False
        )
        stats['speech_encoder']['missing'] = missing
        stats['speech_encoder']['unexpected'] = unexpected
        logger.info(f"✓ Speech encoder loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
        
        # Load text encoder
        logger.info("Loading text_encoder weights...")
        missing, unexpected = self.text_encoder.load_state_dict(
            hf_model.text_encoder.state_dict(), 
            strict=False
        )
        stats['text_encoder']['missing'] = missing
        stats['text_encoder']['unexpected'] = unexpected
        logger.info(f"✓ Text encoder loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
        
        # Load text decoder
        logger.info("Loading text_decoder weights...")
        missing, unexpected = self.text_decoder.load_state_dict(
            hf_model.text_decoder.state_dict(), 
            strict=False
        )
        stats['text_decoder']['missing'] = missing
        stats['text_decoder']['unexpected'] = unexpected
        logger.info(f"✓ Text decoder loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
        
        # Load shared embeddings & LM head
        logger.info("Loading shared embeddings and LM head...")
        self.shared.load_state_dict(hf_model.shared.state_dict())
        self.lm_head.weight.data = hf_model.lm_head.weight.data.clone()
        logger.info("✓ Embeddings and LM head loaded")
        
        # Free memory
        del hf_model
        torch.cuda.empty_cache()
        
        logger.info("✓ All pretrained weights loaded successfully from HuggingFace!")
        return stats
    
    def _load_from_local_checkpoint(self, checkpoint_path: str):
        """Load weights from local checkpoint file"""
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading from local checkpoint: {checkpoint_path}")
        
        # Check if it's a directory or file
        if os.path.isdir(checkpoint_path):
            # Assume pytorch_model.bin inside directory
            model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"pytorch_model.bin not found in {checkpoint_path}"
                )
        else:
            model_file = checkpoint_path
        
        state_dict = torch.load(model_file, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            # Training checkpoint format
            state_dict = state_dict['model']
        
        # Load with strict=False to handle missing/extra keys
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        logger.info(f"✓ Loaded from local checkpoint")
        if missing_keys:
            logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        
        return {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys
        }

    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.forward
    def forward(
        self,
        audio_input_features: Optional[torch.FloatTensor] = None,
        text_input_pivot_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        text_pivot_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        text_past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        audio_past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        audio_encoder_outputs = self.speech_encoder(
            input_features=audio_input_features,
            attention_mask=audio_attention_mask,
        )

        audio_encoder_attention_mask = audio_attention_mask
        if audio_attention_mask is not None:
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(audio_attention_mask).to(
                audio_encoder_outputs.device
            )
            audio_encoder_attention_mask = _compute_new_attention_mask(
                hidden_states=audio_encoder_outputs, seq_lens=sub_sampled_lengths
            )

        # decoder for audio
        audio_decoder_outputs = self.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=audio_encoder_outputs,
            encoder_attention_mask=audio_encoder_attention_mask,
        )

        with torch.no_grad():
            text_encoder_outputs = self.text_encoder(
                input_ids=text_input_pivot_ids,
                attention_mask=text_pivot_attention_mask,
            )
        
            # decoder for text
            text_decoder_outputs = self.text_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=text_encoder_outputs,
                encoder_attention_mask=text_pivot_attention_mask,
            )

            text_pivot_logits = self.lm_head(text_decoder_outputs)

        text_logits = self.lm_head(audio_decoder_outputs)

        kd_loss, n_valid_tokens = compute_token_kd_loss(text_pivot_logits, text_logits, labels)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(text_logits.device)
            masked_lm_loss = loss_fct(text_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return (
            masked_lm_loss,
            kd_loss,
            n_valid_tokens,
            text_logits,
            text_pivot_logits,
        )


    # Copied from transformers.models.seamless_m4t.modeling_seamless_m4t.SeamlessM4TForSpeechToText.generate
    def generate(
        self,
        input_features=None,
        tgt_lang=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs,
    ):
        """
        Generates sequences of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`):
                Input audio features. This should be returned by the [`SeamlessM4TFeatureExtractor`] class or the
                [`SeamlessM4TProcessor`] class. See [`SeamlessM4TFeatureExtractor.__call__`] for details.

            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://huggingface.co/papers/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
            [`~utils.ModelOutput`] types are:
                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        text_decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        # overwrite text_decoder_input_ids if tgt_lang is passed. The latter gets priority over decoder_input_ids.
        input_features = input_features if input_features is not None else kwargs.pop("inputs")
        if tgt_lang is not None:
            inputs = kwargs.get("input_embeds") if input_features is None else input_features
            inputs = (
                inputs
                if inputs is not None
                else kwargs.get("encoder_outputs", {"last_hidden_state": None})["last_hidden_state"]
            )
            batch_size = len(inputs)

            if hasattr(self.generation_config, "text_decoder_lang_to_code_id"):
                # also accept __xxx__
                tgt_lang = tgt_lang.replace("__", "")
                if tgt_lang not in self.generation_config.text_decoder_lang_to_code_id:
                    raise ValueError(
                        f"""`tgt_lang={tgt_lang}` is not supported by this model. Please specify a `tgt_lang` in
                        {", ".join(self.generation_config.text_decoder_lang_to_code_id.keys())}"""
                    )
                # tgt_lang gets priority over decoder input ids
                text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
                device = input_features.device if input_features is not None else inputs.device
                text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size, device=device)
            else:
                raise ValueError(
                    """This model generation config doesn't have a `text_decoder_lang_to_code_id` key which maps
                    the target language to the right token id. Make sure to load the right generation config."""
                )
        else:
            # only a warning, otherwise errors appear in the tests
            logger.warning(
                """You must either specify a `tgt_lang` or pass a correct `text_decoder_input_ids` to get
                a correct generation, otherwise the generation will probably make no sense."""
            )
        return super().generate(
            input_features,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            decoder_input_ids=text_decoder_input_ids,
            **kwargs,
        )



__all__ = [
    "SeamlessM4Tv2ForSpeechToText",
]