"""
Learning rate scheduler utilities for SeamlessM4T v2 training
Extracted from train_kaggle.py for better modularity
"""

import math
import logging
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from configs import TrainingConfig


logger = logging.getLogger(__name__)


def get_current_lr_multiplier(
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1
) -> float:
    """
    Calculate the current LR multiplier based on cosine schedule.
    
    Args:
        global_step: Current training step
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR ratio (default 0.1)
    
    Returns:
        Current multiplier for learning rate
    """
    # Warmup phase: linear 0 -> 1
    if global_step < warmup_steps:
        return float(global_step) / float(max(1, warmup_steps))
    
    # Cosine decay phase: 1 -> min_lr_ratio
    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int,
    warmup_steps: Optional[int] = None,
    min_lr_ratio: Optional[float] = None,
) -> LambdaLR:
    """
    Create LR scheduler: linear warmup -> cosine decay to min_lr_ratio.
    
    Args:
        optimizer: torch optimizer
        config: training config
        num_training_steps: total training steps (batches)
        warmup_steps: steps for linear warmup; if None uses config defaults
        min_lr_ratio: final LR ratio relative to base_lr; if None uses config default
        
    Returns:
        a torch.optim.lr_scheduler.LambdaLR scheduler
    """
    if warmup_steps is None:
        warmup_steps = max(config.min_warmup_steps, int(config.warmup_ratio * num_training_steps))
    
    if min_lr_ratio is None:
        min_lr_ratio = config.min_lr_ratio
    
    def lr_lambda(step: int):
        # warmup phase: linear 0 -> 1
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay phase: 1 -> min_lr_ratio
        progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(
        f"Created cosine scheduler with {warmup_steps} warmup steps out of {num_training_steps} total steps, "
        f"min_lr_ratio={min_lr_ratio}"
    )

    return scheduler


def create_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR:
    """
    Create a simple linear warmup scheduler with constant LR after warmup.
    
    Args:
        optimizer: torch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(f"Created linear warmup scheduler with {num_warmup_steps} warmup steps")
    
    return scheduler


def create_constant_scheduler(optimizer: torch.optim.Optimizer) -> LambdaLR:
    """
    Create a constant learning rate scheduler (multiplier always 1.0).
    
    Args:
        optimizer: torch optimizer
        
    Returns:
        LambdaLR scheduler
    """
    scheduler = LambdaLR(optimizer, lambda step: 1.0)
    logger.info("Created constant LR scheduler")
    return scheduler


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    num_training_steps: int,
    scheduler_type: str = "cosine",
    warmup_steps: Optional[int] = None
) -> LambdaLR:
    """
    Factory function to create different types of schedulers.
    
    Args:
        optimizer: torch optimizer
        config: Training configuration
        num_training_steps: Total training steps
        scheduler_type: Type of scheduler ("cosine", "linear", "constant")
        warmup_steps: Number of warmup steps (if None, uses config defaults)
        
    Returns:
        LambdaLR scheduler
    """
    if scheduler_type == "cosine":
        return create_cosine_scheduler(optimizer, config, num_training_steps, warmup_steps)
    elif scheduler_type == "linear":
        if warmup_steps is None:
            warmup_steps = max(config.min_warmup_steps, int(config.warmup_ratio * num_training_steps))
        return create_linear_warmup_scheduler(optimizer, warmup_steps, num_training_steps)
    elif scheduler_type == "constant":
        return create_constant_scheduler(optimizer)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Must be 'cosine', 'linear', or 'constant'")

