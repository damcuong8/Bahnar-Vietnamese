"""
Model utilities for SeamlessM4T v2 training
Handles model creation, FSDP wrapping, and distributed setup
"""

import os
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    CPUOffload,
)

from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
from seamless_m4t_v2_config import SeamlessM4Tv2Config
from configs import TrainingConfig, FSDPConfig
from memory_tracker import log_memory_stats, print_memory_summary


logger = logging.getLogger(__name__)


def setup_distributed():
    """
    Initialize distributed training.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    logger.info(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


def create_model(config: TrainingConfig):
    """
    Create and configure the SeamlessM4T v2 model.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (model, model_config)
    """
    logger.info(f"Loading model configuration from {config.model_name_or_path}")
    
    # Create model config
    model_config = SeamlessM4Tv2Config()
    
    # Get rank for logging (default to 0 if not set)
    rank = getattr(config, 'local_rank', 0)
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    
    # Create model
    logger.info("Initializing SeamlessM4Tv2ForSpeechToTextTrain_Pivot model")
    log_memory_stats("Before model creation", rank=rank)
    
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(model_config)
    log_memory_stats("After model creation", rank=rank)
    
    # Load pretrained weights if specified
    if config.is_pretrained:
        logger.info("Loading pretrained weights...")
        log_memory_stats("Before loading weights", rank=rank)
        _load_pretrained_weights(model, config)
        log_memory_stats("After loading weights", rank=rank)
    else:
        logger.info("Training from scratch (random initialization)")
    
    # Enable gradient checkpointing if specified
    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
        log_memory_stats("After enabling gradient checkpointing", rank=rank)
    
    return model, model_config


def _load_pretrained_weights(model: nn.Module, config: TrainingConfig):
    """Load pretrained weights into model"""
    # Determine cache directory (Kaggle-friendly)
    cache_dir = getattr(config, 'hf_cache_dir', None)
    if cache_dir is None:
        # Auto-detect Kaggle environment
        if os.path.exists('/kaggle'):
            cache_dir = "/kaggle/working/hf_cache"
            os.makedirs(cache_dir, exist_ok=True)
    
    # Use the new load_pretrained_weights method
    try:
        stats = model.load_pretrained_weights(
            config.model_name_or_path,
            cache_dir=cache_dir
        )
        logger.info("✓ Pretrained weights loaded successfully")
        
        # Log any issues (for debugging)
        if hasattr(stats, 'get'):
            for component, info in stats.items():
                if isinstance(info, dict):
                    if info.get('missing'):
                        logger.debug(f"{component} - Missing keys: {len(info['missing'])}")
                    if info.get('unexpected'):
                        logger.debug(f"{component} - Unexpected keys: {len(info['unexpected'])}")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        logger.warning("Continuing with random initialization")
        config.is_pretrained = False  # Update config to reflect actual state


def _enable_gradient_checkpointing(model: nn.Module):
    """Enable gradient checkpointing for memory efficiency"""
    logger.info("Enabling gradient checkpointing")
    
    if hasattr(model.speech_encoder, 'gradient_checkpointing_enable'):
        model.speech_encoder.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled for speech_encoder")
    
    if hasattr(model.text_decoder, 'gradient_checkpointing_enable'):
        model.text_decoder.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled for text_decoder")
    
    if hasattr(model.text_encoder, 'gradient_checkpointing_enable'):
        model.text_encoder.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled for text_encoder")


def wrap_model_with_fsdp(
    model: nn.Module,
    config: TrainingConfig,
    model_config,
    rank: int,
) -> nn.Module:
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: The model to wrap
        config: Training configuration
        model_config: Model configuration
        rank: Process rank
        
    Returns:
        FSDP-wrapped model (or original model if FSDP not enabled)
    """
    if not config.use_fsdp or config.world_size == 1:
        logger.info("FSDP not enabled or single GPU training, returning unwrapped model")
        return model
    
    # Get rank for logging
    rank = getattr(config, 'local_rank', 0)
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    
    logger.info("Wrapping model with FSDP")
    log_memory_stats("Before FSDP wrap", rank=rank)
    
    # Get sharding strategy
    sharding_strategy = FSDPConfig.SHARDING_STRATEGIES.get(
        config.sharding_strategy,
        FSDPConfig.SHARDING_STRATEGIES["FULL_SHARD"]
    )
    logger.info(f"Sharding strategy: {config.sharding_strategy}")
    
    # Get mixed precision policy
    mixed_precision_policy = None
    if config.use_mixed_precision:
        mixed_precision_policy = FSDPConfig.get_mixed_precision_policy(
            config.mixed_precision_dtype
        )
        logger.info(f"Using mixed precision: {config.mixed_precision_dtype}")
    
    # Get auto wrap policy
    auto_wrap_policy = FSDPConfig.get_auto_wrap_policy(model_config)
    
    # CPU offload config
    cpu_offload_config = CPUOffload(offload_params=True) if config.cpu_offload else None
    if config.cpu_offload:
        logger.info("CPU offload enabled")
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=None,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    logger.info(f"✓ Model wrapped with FSDP")
    log_memory_stats("After FSDP wrap", rank=rank)
    print_memory_summary(rank=rank)
    
    return model

