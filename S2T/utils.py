import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.where(shifted_input_ids == -100, pad_token_id, inplace=True)

    return shifted_input_ids


def compute_token_kd_loss(
    text_pivot_logits: torch.Tensor,
    text_logits: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 2.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, int]:
    """
    Compute token-level Knowledge Distillation (KL) loss between teacher logits
    (text_pivot_logits) and student logits (audio_pivot_logits), aligned by labels.
    
    Args:
        text_pivot_logits: Teacher logits, shape (B, L, V).
        audio_pivot_logits: Student logits, shape (B, L, V).
        labels: Target token ids, shape (B, L). Positions with value == ignore_index are masked out.
        tau: Temperature for softening distributions.
        ignore_index: Value in labels used to ignore positions (commonly -100).
        reduction: "mean" or "sum". "mean" divides by number of valid tokens.
    
    Returns:
        kd_loss: scalar tensor (the KL loss, scaled by tau^2).
        n_valid_tokens: int, number of tokens used in the loss (for logging).
    """
    # Basic checks
    assert text_pivot_logits.dim() == 3 and audio_pivot_logits.dim() == 3, "Logits must be (B,L,V)"
    assert text_pivot_logits.shape == audio_pivot_logits.shape, "Teacher and student logits must have same shape"
    assert labels.dim() == 2, "Labels must be (B, L)"
    B, L, V = audio_pivot_logits.shape

    # Create mask of valid positions (1 for valid tokens, 0 for ignored)
    device = labels.device
    mask = (labels != ignore_index).to(dtype=audio_pivot_logits.dtype, device=device)  # (B, L)
    n_valid_tokens = int(mask.sum().item())

    # If no valid tokens, return zero loss
    if n_valid_tokens == 0:
        return torch.tensor(0.0, device=device, dtype=audio_pivot_logits.dtype), 0

    # Compute softened distributions
    T = float(tau)
    # student: log-probs
    student_log_prob = F.log_softmax(audio_pivot_logits / T, dim=-1)  # (B, L, V)
    # teacher: probs (detach so teacher not in grad graph)
    with torch.no_grad():
        teacher_prob = F.softmax(text_pivot_logits / T, dim=-1)       # (B, L, V)

    # KL per token: F.kl_div expects input=log_prob (student) and target=prob (teacher)
    # Use reduction='none' to keep per-element values, then sum over vocab to get per-token KL
    kl_per_elem = F.kl_div(student_log_prob, teacher_prob, reduction="none")  # (B, L, V)
    kl_per_token = kl_per_elem.sum(dim=-1)  # (B, L)

    # apply mask: zero out ignored positions
    kl_per_token = kl_per_token * mask  # (B, L)

    # sum and scale by T^2 (Hinton et al.)
    kl_sum = kl_per_token.sum()  # scalar

    kd_loss = kl_sum * (T * T)

    if reduction == "mean":
        kd_loss = kd_loss / n_valid_tokens
    elif reduction == "sum":
        # already sum; keep as is
        pass
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return kd_loss, n_valid_tokens




def is_fsdp_managed_module(module: nn.Module) -> bool:

    if not torch.distributed.is_available():
        return False

    return isinstance(module, torch.distributed.fsdp.FullyShardedDataParallel) or getattr(
        module, "_is_fsdp_managed_module", False
    )




class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
}
ACT2FN = ClassInstantier(ACT2CLS)


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
    (`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
    enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            # different names for the same thing in different layers
            # TODO cyril: this one without `S` can be removed after deprection cycle
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)
