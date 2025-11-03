"""
Training stage setup utilities for curriculum learning
Extracted from train_kaggle.py for better modularity
"""

import logging
from typing import Optional

import torch.nn as nn

from configs import TrainingConfig


logger = logging.getLogger(__name__)


def freeze_module(module: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze a module.
    
    Args:
        module: Module to freeze/unfreeze
        freeze: If True, freeze the module; if False, unfreeze it
    """
    for param in module.parameters():
        param.requires_grad = not freeze
    if freeze:
        module.eval()


def setup_stage_a(model: nn.Module, config: TrainingConfig):
    """
    Stage A: Freeze decoder, train only encoder
    - text_encoder.eval() + requires_grad=False (teacher, always frozen)
    - text_decoder frozen
    - lm_head optionally frozen
    - speech_encoder trainable
    
    Args:
        model: The model to configure
        config: Training configuration
    """
    logger.info("="*70)
    logger.info("Setting up Stage A: Encoder-only training")
    logger.info("="*70)
    
    # Freeze text encoder (teacher - always frozen)
    if hasattr(model, 'text_encoder'):
        freeze_module(model.text_encoder, freeze=True)
        logger.info("✓ Text encoder frozen (teacher)")
    
    # Freeze text decoder
    freeze_module(model.text_decoder, freeze=True)
    logger.info("✓ Text decoder frozen")
    
    # Freeze lm_head
    freeze_module(model.lm_head, freeze=True)
    logger.info("✓ LM head frozen")
    
    # Unfreeze speech encoder
    freeze_module(model.speech_encoder, freeze=False)
    model.speech_encoder.train()
    logger.info("✓ Speech encoder trainable")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def setup_stage_b(
    model: nn.Module,
    config: TrainingConfig,
    round_idx: Optional[int] = None,
    total_rounds: int = 6,
    num_layers: Optional[int] = None
):
    """
    Stage B: Unfreeze top-k decoder layers in multiple rounds.

    Args:
        model: model with attribute text_decoder.layers and lm_head
        config: TrainingConfig (expects config.unfreeze_top_k)
        round_idx: which round to apply (0-based). If None -> unfreeze all top_k (old behavior).
        total_rounds: total number of rounds to split the unfreezing into (default 6).
        num_layers: number of decoder layers (if None, inferred from model).
    """
    logger.info("="*70)
    if round_idx is not None:
        logger.info(f"Setting up Stage B (round {round_idx+1} of {total_rounds}) - Unfreezing top layers")
    else:
        logger.info("Setting up Stage B - Unfreezing top layers")
    logger.info("="*70)

    # Determine actual number of decoder layers in the model
    actual_num_layers = len(getattr(model, "text_decoder").layers)
    if num_layers is None:
        num_layers = actual_num_layers
    elif actual_num_layers != num_layers:
        logger.warning(
            f"num_layers={num_layers} but model.text_decoder has {actual_num_layers} layers. "
            "Using actual number of layers."
        )
        num_layers = actual_num_layers

    # how many top layers we intend to unfreeze in total (respect config)
    top_k = min(getattr(config, "unfreeze_top_k", num_layers), num_layers)

    # If round_idx is None -> behave like original: unfreeze all top_k
    if round_idx is None:
        start_idx = num_layers - top_k
        for i in range(start_idx, num_layers):
            freeze_module(model.text_decoder.layers[i], freeze=False)
            logger.info(f"✓ Unfroze decoder layer {i}")
    else:
        # validate round_idx
        if not (0 <= round_idx < total_rounds):
            raise ValueError(f"round_idx must be in [0, {total_rounds-1}], got {round_idx}")

        # Split the top_k layers into `total_rounds` buckets as evenly as possible
        base = top_k // total_rounds
        rem = top_k % total_rounds  # distribute remainder among first `rem` buckets

        # Build buckets: each bucket is a (start, end) of layer indices to unfreeze for that bucket
        buckets = []
        cur = 0
        for r in range(total_rounds):
            bucket_size = base + (1 if r < rem else 0)
            buckets.append((cur, cur + bucket_size))  # relative to top_k range [0, top_k)
            cur += bucket_size

        # determine cumulative layers to unfreeze up through current round
        # relative indices of top_k range that should be unfrozen now:
        cum_end = buckets[round_idx][1]
        # If previous rounds exist, include them as well (we unfreeze up to cum_end)
        # absolute layer indices:
        abs_start = num_layers - top_k
        abs_end = abs_start + cum_end  # exclusive

        if abs_end <= abs_start:
            logger.info(f"No layers to unfreeze in round {round_idx} (bucket empty).")
        else:
            for i in range(abs_start, abs_end):
                # if model has fewer layers than planned, ensure index valid
                if i < len(model.text_decoder.layers):
                    freeze_module(model.text_decoder.layers[i], freeze=False)
                    logger.info(f"✓ Unfroze decoder layer {i} (round {round_idx})")
                else:
                    logger.warning(f"Skipping layer index {i} because model only has {len(model.text_decoder.layers)} layers.")

    # Unfreeze lm_head once we start unfreezing (i.e., in any round or when round_idx is None)
    freeze_module(model.lm_head, freeze=False)
    logger.info("✓ LM head unfrozen")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def setup_stage_c(model: nn.Module, config: TrainingConfig):
    """
    Stage C: Unfreeze all decoder layers (full fine-tune)
    - Keep speech_encoder trainable
    - Unfreeze all decoder layers + lm_head
    
    Args:
        model: The model to configure
        config: Training configuration
    """
    logger.info("="*70)
    logger.info("Setting up Stage C: Full decoder unfrozen")
    logger.info("="*70)
    
    # Unfreeze entire text decoder
    freeze_module(model.text_decoder, freeze=False)
    model.text_decoder.train()
    logger.info("✓ All decoder layers unfrozen")
    
    # Unfreeze lm_head
    freeze_module(model.lm_head, freeze=False)
    logger.info("✓ LM head unfrozen")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def get_kd_alpha(
    config: TrainingConfig,
    stage: str,
    step_in_stage: int,
    total_steps_in_stage: int
) -> float:
    """
    Get dynamic KD alpha based on current stage and progress
    
    Args:
        config: Training configuration
        stage: Current stage ("A", "B", or "C")
        step_in_stage: Current step within the stage
        total_steps_in_stage: Total steps for this stage
        
    Returns:
        KD alpha weight
    """
    if stage == "A":
        return config.kd_alpha_stage_a
    elif stage == "B":
        progress = step_in_stage / max(1, total_steps_in_stage)
        alpha = config.kd_alpha_stage_a + progress * (config.kd_alpha_stage_b - config.kd_alpha_stage_a)
        return alpha
    elif stage == "C":
        # Linear decay from start to end
        progress = step_in_stage / max(1, total_steps_in_stage)
        alpha = config.kd_alpha_stage_c_start + progress * (config.kd_alpha_stage_c_end - config.kd_alpha_stage_c_start)
        return alpha
    else:
        return 1.0

