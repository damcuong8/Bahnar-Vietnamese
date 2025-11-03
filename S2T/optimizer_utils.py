"""
Optimizer utilities for SeamlessM4T v2 training
Extracted from train_kaggle.py for better modularity
"""

import re
import logging
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from configs import TrainingConfig


logger = logging.getLogger(__name__)


def _is_no_decay_param(param_name: str) -> bool:
    """Check if parameter should not have weight decay"""
    return "bias" in param_name or "layer_norm" in param_name or "LayerNorm" in param_name


def _collect_params_by_pattern(
    model: nn.Module,
    pattern: str,
    requires_grad_only: bool = True
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Collect parameters matching a pattern, separated into decay/no_decay groups.
    
    Args:
        model: The model to collect parameters from
        pattern: String pattern to match in parameter names
        requires_grad_only: Only collect parameters with requires_grad=True
        
    Returns:
        Tuple of (decay_params, no_decay_params)
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if requires_grad_only and not param.requires_grad:
            continue
        
        if pattern not in name:
            continue
        
        if _is_no_decay_param(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return decay_params, no_decay_params


def _add_param_group(
    param_groups: List[dict],
    params: List[nn.Parameter],
    lr: float,
    weight_decay: float,
    name: str,
    initial_lr: Optional[float] = None
):
    """
    Add a parameter group to the list if params is not empty.
    
    Args:
        param_groups: List to append parameter group to
        params: List of parameters
        lr: Learning rate
        weight_decay: Weight decay value
        name: Name for logging
        initial_lr: Initial learning rate (for scheduler compatibility)
    """
    if params:
        group = {
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": name
        }
        if initial_lr is not None:
            group["initial_lr"] = initial_lr
        param_groups.append(group)


def create_optimizer_stage_a(model: nn.Module, config: TrainingConfig, lr: Optional[float] = None) -> torch.optim.AdamW:
    """
    Create optimizer for Stage A (encoder-only training).
    
    Args:
        model: The model (possibly FSDP-wrapped)
        config: Training configuration
        lr: Override learning rate (if None, uses config.encoder_lr)
        
    Returns:
        AdamW optimizer
    """
    if lr is None:
        lr = config.encoder_lr
    
    param_groups = []
    
    # Stage A: Only encoder parameters are trainable
    encoder_decay, encoder_no_decay = _collect_params_by_pattern(model, "", requires_grad_only=True)
    
    _add_param_group(param_groups, encoder_decay, lr, config.weight_decay, "encoder_decay")
    _add_param_group(param_groups, encoder_no_decay, lr, 0.0, "encoder_no_decay")
    
    logger.info(f"Stage A optimizer: encoder_lr={lr:.2e}")
    logger.info(f"Created optimizer with {len(param_groups)} parameter groups")
    
    return torch.optim.AdamW(param_groups)


def create_optimizer_stage_b(model: nn.Module, config: TrainingConfig, lr: Optional[float] = None) -> torch.optim.AdamW:
    """
    Create optimizer for Stage B (encoder + top decoder layers).
    
    Args:
        model: The model (possibly FSDP-wrapped)
        config: Training configuration
        lr: Override decoder learning rate (if None, uses config.decoder_lr)
        
    Returns:
        AdamW optimizer
    """
    if lr is None:
        lr = config.decoder_lr
    
    param_groups = []
    
    # Separate encoder and decoder parameters
    encoder_decay, encoder_no_decay = _collect_params_by_pattern(model, "speech_encoder")
    decoder_decay, decoder_no_decay = _collect_params_by_pattern(model, "")
    
    # Remove encoder params from decoder lists
    encoder_params_set = set(encoder_decay + encoder_no_decay)
    decoder_decay = [p for p in decoder_decay if p not in encoder_params_set]
    decoder_no_decay = [p for p in decoder_no_decay if p not in encoder_params_set]
    
    # Add encoder params with encoder_lr
    _add_param_group(param_groups, encoder_decay, config.encoder_lr, config.weight_decay, "encoder_decay")
    _add_param_group(param_groups, encoder_no_decay, config.encoder_lr, 0.0, "encoder_no_decay")
    
    # Add decoder params with decoder_lr
    _add_param_group(param_groups, decoder_decay, lr, config.weight_decay, "decoder_top_decay")
    _add_param_group(param_groups, decoder_no_decay, lr, 0.0, "decoder_top_no_decay")
    
    logger.info(f"Stage B optimizer: encoder_lr={config.encoder_lr:.2e}, decoder_top_lr={lr:.2e}")
    logger.info(f"Created optimizer with {len(param_groups)} parameter groups")
    
    return torch.optim.AdamW(param_groups)


def create_optimizer_stage_c(
    model: nn.Module,
    config: TrainingConfig,
    lr: Optional[float] = None
) -> torch.optim.AdamW:
    """
    Create optimizer for Stage C (full model training with optional layer-wise LR decay).
    
    Args:
        model: The model (possibly FSDP-wrapped)
        config: Training configuration
        lr: Override decoder learning rate (if None, uses config.decoder_lr)
        
    Returns:
        AdamW optimizer
    """
    if lr is None:
        lr = config.decoder_lr
    
    param_groups = []
    
    if config.use_layer_wise_lr_decay:
        # Layer-wise learning rate decay (higher LR for top layers)
        param_groups = _create_layerwise_param_groups(model, config, lr)
        logger.info(f"Stage C optimizer with layer-wise LR decay: encoder_lr={config.encoder_lr:.2e}, decoder_top_lr={lr:.2e}")
    else:
        # Standard: encoder + all decoder with two different LRs
        param_groups = _create_standard_param_groups(model, config, lr)
        logger.info(f"Stage C optimizer: encoder_lr={config.encoder_lr:.2e}, decoder_full_lr={lr:.2e}")
    
    logger.info(f"Created optimizer with {len(param_groups)} parameter groups")
    
    return torch.optim.AdamW(param_groups)


def _create_layerwise_param_groups(model: nn.Module, config: TrainingConfig, decoder_lr: float) -> List[dict]:
    """Create parameter groups with layer-wise learning rate decay"""
    param_groups = []
    
    # Organize parameters by type
    encoder_params = {"decay": [], "no_decay": []}
    decoder_layer_params = {}  # key: layer_idx, value: {decay: [], no_decay: []}
    lm_head_params = {"decay": [], "no_decay": []}
    
    # Get number of decoder layers
    num_decoder_layers = len(model.text_decoder.layers) if hasattr(model, 'text_decoder') else 24
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_no_decay = _is_no_decay_param(name)
        param_key = "no_decay" if is_no_decay else "decay"
        
        if "speech_encoder" in name:
            encoder_params[param_key].append(param)
        elif "lm_head" in name:
            lm_head_params[param_key].append(param)
        elif "text_decoder.layers" in name:
            # Extract layer index
            match = re.search(r'text_decoder\.layers\.(\d+)', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in decoder_layer_params:
                    decoder_layer_params[layer_idx] = {"decay": [], "no_decay": []}
                decoder_layer_params[layer_idx][param_key].append(param)
    
    # Add encoder params
    _add_param_group(param_groups, encoder_params["decay"], config.encoder_lr, config.weight_decay, "encoder_decay")
    _add_param_group(param_groups, encoder_params["no_decay"], config.encoder_lr, 0.0, "encoder_no_decay")
    
    # Add decoder layer params with layer-wise LR
    for layer_idx in sorted(decoder_layer_params.keys()):
        # Higher layers get higher LR
        layer_lr = decoder_lr * (config.layer_wise_lr_decay_rate ** (num_decoder_layers - 1 - layer_idx))
        
        _add_param_group(
            param_groups, decoder_layer_params[layer_idx]["decay"],
            layer_lr, config.weight_decay, f"decoder_layer_{layer_idx}_decay"
        )
        _add_param_group(
            param_groups, decoder_layer_params[layer_idx]["no_decay"],
            layer_lr, 0.0, f"decoder_layer_{layer_idx}_no_decay"
        )
    
    # Add lm_head params
    _add_param_group(param_groups, lm_head_params["decay"], decoder_lr, config.weight_decay, "lm_head_decay")
    _add_param_group(param_groups, lm_head_params["no_decay"], decoder_lr, 0.0, "lm_head_no_decay")
    
    return param_groups


def _create_standard_param_groups(model: nn.Module, config: TrainingConfig, decoder_lr: float) -> List[dict]:
    """Create standard parameter groups (encoder vs decoder)"""
    param_groups = []
    
    encoder_decay, encoder_no_decay = _collect_params_by_pattern(model, "speech_encoder")
    
    # All other params are decoder/lm_head
    decoder_decay, decoder_no_decay = _collect_params_by_pattern(model, "")
    encoder_params_set = set(encoder_decay + encoder_no_decay)
    decoder_decay = [p for p in decoder_decay if p not in encoder_params_set]
    decoder_no_decay = [p for p in decoder_no_decay if p not in encoder_params_set]
    
    # Add encoder params
    _add_param_group(param_groups, encoder_decay, config.encoder_lr, config.weight_decay, "encoder_decay")
    _add_param_group(param_groups, encoder_no_decay, config.encoder_lr, 0.0, "encoder_no_decay")
    
    # Add decoder params
    _add_param_group(param_groups, decoder_decay, decoder_lr, config.weight_decay, "decoder_full_decay")
    _add_param_group(param_groups, decoder_no_decay, decoder_lr, 0.0, "decoder_full_no_decay")
    
    return param_groups


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    stage: str = "A",
    lr: Optional[float] = None
) -> torch.optim.AdamW:
    """
    Create optimizer for specific training stage.
    
    Args:
        model: The model (possibly FSDP-wrapped)
        config: Training configuration
        stage: Training stage ("A", "B", or "C")
        lr: Override learning rate (if None, uses config default for stage)
        
    Returns:
        AdamW optimizer
    """
    if stage == "A":
        return create_optimizer_stage_a(model, config, lr)
    elif stage == "B":
        return create_optimizer_stage_b(model, config, lr)
    elif stage == "C":
        return create_optimizer_stage_c(model, config, lr)
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be 'A', 'B', or 'C'")


def add_new_param_groups_to_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    config: TrainingConfig,
    stage: str,
    current_lr_multiplier: float,
    round_idx: Optional[int] = None
) -> int:
    """
    Add new trainable parameters to existing optimizer when unfreezing layers.
    
    Args:
        optimizer: Existing optimizer
        model: The model
        config: Training config
        stage: Current stage ("B" or "C")
        current_lr_multiplier: Current LR multiplier from scheduler
        round_idx: Optional round index for Stage B
    
    Returns:
        Number of new parameters added
    """
    # Get current params already in optimizer
    existing_params = set()
    for group in optimizer.param_groups:
        existing_params.update(group['params'])
    
    # Collect new params as (name, param) tuples
    new_params = [(name, param) for name, param in model.named_parameters() 
                  if param.requires_grad and param not in existing_params]
    
    if not new_params:
        logger.info("No new trainable parameters found to add to optimizer.")
        return 0
    
    added_count = 0
    
    # Stage C with layer-wise LR decay
    if stage == "C" and config.use_layer_wise_lr_decay:
        added_count = _add_layerwise_params(optimizer, model, config, new_params, current_lr_multiplier)
    else:
        # Stage B or Stage C without layer-wise LR
        added_count = _add_standard_params(optimizer, config, new_params, stage, current_lr_multiplier)
    
    logger.info(f"✓ Total new parameters added to optimizer: {added_count}")
    return added_count


def _add_layerwise_params(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    config: TrainingConfig,
    new_params: List[Tuple[str, nn.Parameter]],
    current_multiplier: float
) -> int:
    """Add new parameters with layer-wise LR decay"""
    logger.info(f"Stage C: Applying layer-wise LR decay (current multiplier: {current_multiplier:.4f})")
    
    # Separate params by type
    encoder_params = {'decay': [], 'no_decay': []}
    decoder_layer_params = {}
    other_decoder_params = {'decay': [], 'no_decay': []}
    
    num_decoder_layers = len(model.text_decoder.layers) if hasattr(model.text_decoder, 'layers') else 24
    
    for name, param in new_params:
        is_no_decay = _is_no_decay_param(name)
        key = "no_decay" if is_no_decay else "decay"
        
        if "speech_encoder" in name:
            encoder_params[key].append(param)
        elif "text_decoder.layers" in name:
            match = re.search(r'text_decoder\.layers\.(\d+)', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in decoder_layer_params:
                    decoder_layer_params[layer_idx] = {'decay': [], 'no_decay': []}
                decoder_layer_params[layer_idx][key].append(param)
            else:
                other_decoder_params[key].append(param)
        else:
            other_decoder_params[key].append(param)
    
    added_count = 0
    
    # Add encoder params
    encoder_lr_base = config.encoder_lr
    encoder_lr_adjusted = encoder_lr_base * current_multiplier
    
    for key in ['decay', 'no_decay']:
        if encoder_params[key]:
            wd = config.weight_decay if key == 'decay' else 0.0
            optimizer.add_param_group({
                'params': encoder_params[key],
                'lr': encoder_lr_adjusted,
                'initial_lr': encoder_lr_base,
                'weight_decay': wd
            })
            added_count += len(encoder_params[key])
    
    # Add decoder layer params with layer-wise LR
    base_decoder_lr = config.decoder_lr
    
    for layer_idx in sorted(decoder_layer_params.keys()):
        layer_lr_base = base_decoder_lr * (config.layer_wise_lr_decay_rate ** (num_decoder_layers - 1 - layer_idx))
        layer_lr_adjusted = layer_lr_base * current_multiplier
        
        for key in ['decay', 'no_decay']:
            if decoder_layer_params[layer_idx][key]:
                wd = config.weight_decay if key == 'decay' else 0.0
                optimizer.add_param_group({
                    'params': decoder_layer_params[layer_idx][key],
                    'lr': layer_lr_adjusted,
                    'initial_lr': layer_lr_base,
                    'weight_decay': wd
                })
                added_count += len(decoder_layer_params[layer_idx][key])
        
        logger.info(f"Added decoder layer {layer_idx} - base_lr={layer_lr_base:.2e}, current_lr={layer_lr_adjusted:.2e}")
    
    # Add other decoder params (lm_head, etc.)
    other_lr_adjusted = base_decoder_lr * current_multiplier
    
    for key in ['decay', 'no_decay']:
        if other_decoder_params[key]:
            wd = config.weight_decay if key == 'decay' else 0.0
            optimizer.add_param_group({
                'params': other_decoder_params[key],
                'lr': other_lr_adjusted,
                'initial_lr': base_decoder_lr,
                'weight_decay': wd
            })
            added_count += len(other_decoder_params[key])
    
    return added_count


def _add_standard_params(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    new_params: List[Tuple[str, nn.Parameter]],
    stage: str,
    current_multiplier: float
) -> int:
    """Add new parameters with standard (non-layerwise) LR"""
    logger.info(f"Stage {stage}: Standard param addition (current multiplier: {current_multiplier:.4f})")
    
    # Simple separation: encoder vs decoder
    encoder_params = {'decay': [], 'no_decay': []}
    decoder_params = {'decay': [], 'no_decay': []}
    
    for name, param in new_params:
        is_no_decay = _is_no_decay_param(name)
        key = "no_decay" if is_no_decay else "decay"
        
        if "speech_encoder" in name:
            encoder_params[key].append(param)
        else:
            decoder_params[key].append(param)
    
    # Determine base LR
    encoder_lr_base = config.encoder_lr
    decoder_lr_base = config.decoder_lr
    
    # Adjust with current multiplier
    encoder_lr_adjusted = encoder_lr_base * current_multiplier
    decoder_lr_adjusted = decoder_lr_base * current_multiplier
    
    added_count = 0
    
    # Add encoder params
    for key in ['decay', 'no_decay']:
        if encoder_params[key]:
            wd = config.weight_decay if key == 'decay' else 0.0
            optimizer.add_param_group({
                'params': encoder_params[key],
                'lr': encoder_lr_adjusted,
                'initial_lr': encoder_lr_base,
                'weight_decay': wd
            })
            added_count += len(encoder_params[key])
    
    # Add decoder params
    for key in ['decay', 'no_decay']:
        if decoder_params[key]:
            wd = config.weight_decay if key == 'decay' else 0.0
            optimizer.add_param_group({
                'params': decoder_params[key],
                'lr': decoder_lr_adjusted,
                'initial_lr': decoder_lr_base,
                'weight_decay': wd
            })
            added_count += len(decoder_params[key])
    
    logger.info(f"Added {added_count} params - Encoder: base={encoder_lr_base:.2e}/current={encoder_lr_adjusted:.2e}, "
                f"Decoder: base={decoder_lr_base:.2e}/current={decoder_lr_adjusted:.2e}")
    
    return added_count

