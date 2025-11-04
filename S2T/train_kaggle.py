"""
Fine-tuning SeamlessM4T v2 on Kaggle with FSDP (Fully Sharded Data Parallel)
This is the refactored version using modular components.

This script supports 2-GPU training with wandb monitoring and curriculum learning.
"""

import os
import logging
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import argparse

# Import refactored modules
from configs import TrainingConfig
from datasets import (
    ViBaSpeechToTextDataset,
    DataCollatorSpeechToText
)
from model_utils import (
    setup_distributed,
    cleanup_distributed,
    create_model,
    wrap_model_with_fsdp
)
from optimizer_utils import create_optimizer, add_new_param_groups_to_optimizer
from scheduler_utils import create_cosine_scheduler, get_current_lr_multiplier
from training_stages import setup_stage_a, setup_stage_b, setup_stage_c
from checkpoint_utils import save_checkpoint
from trainer import CurriculumTrainer
from transformers import (
    SeamlessM4TFeatureExtractor,
    AutoProcessor,
)

# Check wandb availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Optional YAML support for external config files
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Setup logging
def setup_logging(output_dir: str = "./output", rank: int = 0):
    """
    Setup logging to both console and file.
    
    Args:
        output_dir: Directory to save log file
        rank: Process rank (only rank 0 writes to file)
    """
    # Create output directory if it doesn't exist
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "training.log")
    else:
        log_file = None
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Console handler (for all ranks)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (only for rank 0)
    if log_file and rank == 0:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - [Rank %(process)d] - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return root_logger

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def main(config: TrainingConfig | None = None):
    """Main training function with curriculum learning support"""
    
    if config is None:
        config = TrainingConfig()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    config.local_rank = local_rank
    config.world_size = world_size
    
    # Setup logging (after getting rank)
    setup_logging(output_dir=config.output_dir, rank=rank)
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize wandb
    if rank == 0 and config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )
        logger.info(f"Initialized wandb project: {config.wandb_project}")
    
    # Create output directory
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Create model
    model, model_config = create_model(config)
    model = model.cuda()
    
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large") 
    
    # Create datasets and dataloader
    logger.info("Creating datasets")
    train_dataset = ViBaSpeechToTextDataset(
        excel_path=config.excel_path,
        audio_col=config.audio_col,
        vi_col=config.vi_col,
        en_col=config.en_col,
        target_sr=16000,
        mono=True,
        augment_fn=None,
        use_cache=False,
        processor=processor,
    )

    feature_extractor = SeamlessM4TFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
    ) 

    data_collator = DataCollatorSpeechToText(
        feature_extractor=feature_extractor,
        processor=processor,
        padding=True,
        pad_to_multiple_of=8,
        target_language="vi",
        pivot_language="en"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * config.num_epochs
    
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Total training steps: {total_training_steps}")
    
    # Calculate stage steps using config method
    stage_a_steps, stage_b_steps, stage_c_steps = config.calculate_stage_steps(total_training_steps)
    
    # Update config with calculated values
    if config.stage_a_steps is None:
        config.stage_a_steps = stage_a_steps
    
    logger.info("\n" + "="*70)
    logger.info("CURRICULUM LEARNING PLAN")
    logger.info("="*70)
    logger.info(f"Stage A (Encoder-only):           {stage_a_steps:,} steps")
    logger.info(f"Stage B (Top-{config.unfreeze_top_k} decoder layers): {stage_b_steps:,} steps")
    logger.info(f"  └─ Split into {config.num_stage_b_rounds} rounds")
    logger.info(f"Stage C (Full decoder):           {stage_c_steps:,} steps")
    logger.info(f"Total:                            {total_training_steps:,} steps")
    logger.info("="*70 + "\n")
    
    if not config.enable_curriculum:
        logger.warning("Curriculum learning DISABLED - Please enable in config")
        return
    
    # ========================================================================
    # STAGE A: Encoder-only training
    # ========================================================================
    logger.info("\n" + "#"*70)
    logger.info("# STAGE A: ENCODER-ONLY TRAINING")
    logger.info("#"*70 + "\n")
    
    # Setup Stage A
    setup_stage_a(model, config)
    
    # Sync after setup (important for FSDP)
    if config.world_size > 1:
        dist.barrier()
    
    # Wrap compile
    model = torch.compile(model)
    
    # Wrap with FSDP after stage setup
    model = wrap_model_with_fsdp(model, config, model_config, rank)
    
    # Create optimizer and scheduler
    warmup_a = max(config.stage_a_warmup_min, int(config.stage_a_warmup_pct * stage_a_steps))
    optimizer = create_optimizer(model, config, stage="A")
    scheduler = create_cosine_scheduler(optimizer, config, total_training_steps, warmup_a)
    
    logger.info(f"✓ Created optimizer and scheduler (warmup: {warmup_a} steps)")
    
    # Create trainer
    trainer = CurriculumTrainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        rank=rank,
    )
    
    # Train Stage A
    trainer.train_stage(stage="A", num_steps=stage_a_steps)
    global_step = trainer.global_step
    
    # Save Stage A checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage="A")
    
    # ========================================================================
    # STAGE B: Unfreeze top-k decoder layers (Progressive rounds)
    # ========================================================================
    logger.info("\n" + "#"*70)
    logger.info("# STAGE B: PROGRESSIVE UNFREEZE TOP-K DECODER LAYERS")
    logger.info("#"*70 + "\n")
    
    stage_b_steps_per_round = stage_b_steps // config.num_stage_b_rounds
    
    for round_idx in range(config.num_stage_b_rounds):
        logger.info(f"\n{'─'*70}")
        logger.info(f"Stage B - Round {round_idx + 1}/{config.num_stage_b_rounds}")
        logger.info(f"{'─'*70}")
        
        # Unfreeze layers progressively
        setup_stage_b(model, config, round_idx=round_idx, total_rounds=config.num_stage_b_rounds)
        
        # Sync after setup
        if config.world_size > 1:
            dist.barrier()
        
        # Add newly unfrozen parameters to optimizer
        current_multiplier = get_current_lr_multiplier(
            global_step, total_training_steps, warmup_a, config.min_lr_ratio
        )
        num_added = add_new_param_groups_to_optimizer(
            optimizer, model, config, stage="B",
            current_lr_multiplier=current_multiplier,
            round_idx=round_idx
        )
        if num_added > 0:
            logger.info(f"Round {round_idx}: Added {num_added} new parameters to optimizer")
        
        # Train this round
        steps_this_round = stage_b_steps_per_round
        if round_idx == config.num_stage_b_rounds - 1:
            # Last round: train remaining steps
            steps_this_round = stage_b_steps - (stage_b_steps_per_round * (config.num_stage_b_rounds - 1))
        
        trainer.train_stage(stage=f"B_R{round_idx}", num_steps=steps_this_round)
        global_step = trainer.global_step
        
        # Save checkpoint after each round (every 2 rounds)
        if rank == 0 and (round_idx + 1) % 2 == 0:
            save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage=f"B-round{round_idx+1}")
    
    # Save Stage B final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage="B")
    
    logger.info(f"\n✓ Completed all {config.num_stage_b_rounds} rounds of Stage B")
    
    # ========================================================================
    # STAGE C: Full decoder unfrozen
    # ========================================================================
    logger.info("\n" + "#"*70)
    logger.info("# STAGE C: FULL DECODER TRAINING")
    logger.info("#"*70 + "\n")
    
    # Setup Stage C - unfreeze all remaining decoder layers
    setup_stage_c(model, config)
    
    # Sync after setup
    if config.world_size > 1:
        dist.barrier()
    
    # Add newly unfrozen parameters to existing optimizer
    current_multiplier = get_current_lr_multiplier(
        global_step, total_training_steps, warmup_a, config.min_lr_ratio
    )
    num_added = add_new_param_groups_to_optimizer(
        optimizer, model, config, stage="C",
        current_lr_multiplier=current_multiplier
    )
    logger.info(f"Stage C: Added {num_added} new parameters to optimizer")
    
    # Train Stage C
    trainer.train_stage(stage="C", num_steps=stage_c_steps)
    global_step = trainer.global_step
    
    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage="C")
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 Training completed! Final checkpoint saved at step {global_step}")
        logger.info(f"{'='*70}\n")
    
    # Cleanup
    if config.use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SeamlessM4T v2 with curriculum learning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    if args.config is not None:
        if not YAML_AVAILABLE:
            raise ImportError("pyyaml is required to load YAML configs. Install with: pip install pyyaml")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
        # Build TrainingConfig from YAML dict (keys must match TrainingConfig fields)
        config = TrainingConfig(**cfg_dict)
        main(config)
    else:
        main()

