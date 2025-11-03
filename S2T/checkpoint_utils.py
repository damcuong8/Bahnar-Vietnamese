"""
Checkpoint management utilities for SeamlessM4T v2 training
Extracted from train_kaggle.py for better modularity
"""

import os
import logging
import shutil
from typing import Optional
import time

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from configs import TrainingConfig


logger = logging.getLogger(__name__)

# Check wandb availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _upload_checkpoint_to_wandb(
    checkpoint_path: str,
    step: int,
    stage: Optional[str],
    config: TrainingConfig
):
    """
    Upload checkpoint to wandb as an artifact.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        step: Training step number
        stage: Current training stage (A/B/C)
        config: Training configuration
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, skipping checkpoint upload")
        return
    
    try:
        # Get file size for logging
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f"Uploading checkpoint to wandb (size: {file_size_mb:.2f} MB)...")
        
        start_time = time.time()
        
        # Create artifact name
        stage_suffix = f"-stage-{stage}" if stage else ""
        artifact_name = f"model-checkpoint-step-{step}{stage_suffix}"
        
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Model checkpoint at step {step}" + (f" (Stage {stage})" if stage else ""),
            metadata={
                "step": step,
                "stage": stage or "unknown",
                "config": {
                    "encoder_lr": config.encoder_lr,
                    "is_pretrained": config.is_pretrained,
                    "enable_curriculum": config.enable_curriculum,
                    "unfreeze_top_k": config.unfreeze_top_k,
                }
            }
        )
        
        # Add checkpoint file to artifact
        artifact.add_file(checkpoint_path, name="pytorch_model.bin")
        
        # Log artifact to wandb
        wandb.log_artifact(artifact)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded checkpoint to wandb: {artifact_name} (took {elapsed:.1f}s)")
        
        # Also log as a simple file for quick access
        try:
            wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
        except Exception as e:
            logger.debug(f"Note: wandb.save() failed: {e} (artifact upload succeeded)")
            
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to wandb: {e}")
        raise


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    config: TrainingConfig,
    rank: int,
    scaler: Optional[object] = None,
    stage: Optional[str] = None,
):
    """
    Save model checkpoint (includes optional AMP GradScaler state) and optionally upload to wandb.
    
    Args:
        model: The model (possibly FSDP-wrapped)
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch
        step: Current training step
        config: Training configuration
        rank: Process rank (only rank 0 saves)
        scaler: Optional GradScaler for AMP (can be GradScaler object or dict)
        stage: Current training stage (A/B/C) for logging
    """
    if rank != 0:
        return

    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare common checkpoint fields
    def _add_common_fields(checkpoint_dict):
        checkpoint_dict.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        })
        # Add stage info if provided
        if stage is not None:
            checkpoint_dict["stage"] = stage
        # Add scaler state if provided (accept either GradScaler or already a dict)
        if scaler is not None:
            try:
                # If scaler is an object with state_dict()
                checkpoint_dict["scaler"] = scaler.state_dict()
            except Exception:
                # If scaler is already a dict (or cannot call state_dict), just save it
                checkpoint_dict["scaler"] = scaler

    # Save using FSDP state dict API if using FSDP
    checkpoint_path = None
    if getattr(config, "use_fsdp", False) and isinstance(model, FSDP):
        # Configure state dict settings
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()

            checkpoint = {"model": state_dict}
            _add_common_fields(checkpoint)

            checkpoint_path_tmp = os.path.join(checkpoint_dir, "pytorch_model.bin.tmp")
            checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            torch.save(checkpoint, checkpoint_path_tmp)
            os.replace(checkpoint_path_tmp, checkpoint_path)
            logger.info(f"Saved FSDP checkpoint to {checkpoint_path}")
    else:
        checkpoint = {
            "model": model.state_dict(),
        }
        _add_common_fields(checkpoint)

        checkpoint_path_tmp = os.path.join(checkpoint_dir, "pytorch_model.bin.tmp")
        checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        torch.save(checkpoint, checkpoint_path_tmp)
        os.replace(checkpoint_path_tmp, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Upload to wandb if enabled
    if config.use_wandb and config.wandb_save_checkpoints and WANDB_AVAILABLE and checkpoint_path is not None:
        try:
            _upload_checkpoint_to_wandb(checkpoint_path, step, stage, config)
        except Exception as e:
            logger.warning(f"Failed to upload checkpoint to wandb: {e}")

    # Remove old checkpoints if exceeding limit
    if getattr(config, "save_total_limit", None) is not None:
        _cleanup_old_checkpoints(config)


def _cleanup_old_checkpoints(config: TrainingConfig):
    """Remove old checkpoints exceeding save_total_limit"""
    # find checkpoint-* dirs and parse step numbers robustly
    def _try_parse_step(name):
        try:
            return int(name.split("-")[-1])
        except Exception:
            return None

    checkpoints = []
    for d in os.listdir(config.output_dir):
        if d.startswith("checkpoint-"):
            step_num = _try_parse_step(d)
            if step_num is not None:
                checkpoints.append((step_num, d))

    checkpoints = sorted(checkpoints, key=lambda x: x[0])  # sort by step number

    if len(checkpoints) > config.save_total_limit:
        to_remove = checkpoints[:-config.save_total_limit]
        for (step_num, old_checkpoint) in to_remove:
            old_path = os.path.join(config.output_dir, old_checkpoint)
            try:
                logger.info(f"Removing old checkpoint: {old_path}")
                shutil.rmtree(old_path)
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_path}: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict: bool = True
) -> dict:
    """
    Load checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        model: Model to load state dict into
        optimizer: Optional optimizer to load state dict into
        scheduler: Optional scheduler to load state dict into
        scaler: Optional GradScaler to load state dict into
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary containing checkpoint metadata (epoch, step, stage, etc.)
    """
    # If directory provided, look for pytorch_model.bin
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model state dict
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=strict)
        logger.info("✓ Model state dict loaded")
    
    # Load optimizer state dict
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("✓ Optimizer state dict loaded")
    
    # Load scheduler state dict
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("✓ Scheduler state dict loaded")
    
    # Load scaler state dict
    if scaler is not None and "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
            logger.info("✓ GradScaler state dict loaded")
        except Exception as e:
            logger.warning(f"Failed to load GradScaler state dict: {e}")
    
    # Return metadata
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "stage": checkpoint.get("stage", None),
    }
    
    logger.info(f"Loaded checkpoint: epoch={metadata['epoch']}, step={metadata['step']}, stage={metadata['stage']}")
    
    return metadata

