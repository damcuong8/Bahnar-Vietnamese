"""
Trainer class for SeamlessM4T v2 curriculum learning
Extracted from train_kaggle.py for better modularity
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from configs import TrainingConfig
from training_stages import get_kd_alpha
from checkpoint_utils import save_checkpoint
from memory_tracker import MemoryTracker, log_memory_stats, print_memory_summary


logger = logging.getLogger(__name__)

# Check wandb availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CurriculumTrainer:
    """
    Trainer for curriculum learning with SeamlessM4T v2.
    
    Args:
        model: The model to train (possibly FSDP-wrapped)
        train_dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        rank: Process rank
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        optimizer,
        scheduler,
        config: TrainingConfig,
        rank: int,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.rank = rank
        
        # Setup AMP
        self.use_amp = getattr(config, "use_amp", True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.dataloader_iter = iter(train_dataloader)
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(
            rank=rank,
            log_to_wandb=getattr(config, "use_wandb", False),
        )
        
        logger.info(f"Trainer initialized (AMP={self.use_amp}, rank={rank})")
    
    def _get_next_batch(self):
        """Get next batch from dataloader, cycling if needed"""
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.train_dataloader)
            batch = next(self.dataloader_iter)
        
        # Move batch to device
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        return batch
    
    def _training_step(self, batch, stage: str, step_in_stage: int, total_steps_in_stage: int):
        """
        Perform one training step.
        
        Returns:
            Dictionary with loss components
        """
        # Get dynamic KD alpha
        kd_alpha = get_kd_alpha(self.config, stage, step_in_stage, total_steps_in_stage)
        
        # Forward pass (with autocast if AMP enabled)
        with self.memory_tracker.track("forward"):
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(
                    audio_input_features=batch["audio_input_features"],
                    text_input_pivot_ids=batch["text_input_pivot_ids"],
                    labels=batch["labels"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    text_pivot_attention_mask=batch["text_pivot_attention_mask"],
                )
                
                # Unpack outputs
                ce_loss, kd_loss, n_valid_tokens, text_logits, text_pivot_logits = outputs
                
                # Combined loss with dynamic KD weight
                loss = self.config.weight_ce * ce_loss + kd_alpha * kd_loss
                
                # Scale loss for gradient accumulation (division before backward)
                loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass (use scaler if AMP)
        with self.memory_tracker.track("backward"):
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,  # Restore original scale
            "ce_loss": ce_loss.item() if ce_loss is not None else 0.0,
            "kd_loss": kd_loss.item() if kd_loss is not None else 0.0,
            "kd_alpha": kd_alpha,
        }
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping"""
        with self.memory_tracker.track("optimizer_step"):
            # If AMP, unscale before gradient clipping
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping (if requested)
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step (use scaler when AMP)
            if self.use_amp and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            try:
                self.scheduler.step()
            except Exception as e:
                logger.debug(f"Scheduler step failed: {e}")
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
    
    def _log_metrics(self, metrics: dict, stage: str, step_in_stage: int):
        """Log training metrics"""
        if self.rank != 0:
            return
        
        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, "get_last_lr") else None
        
        log_msg = (
            f"Stage {stage} | Step {step_in_stage} (Global: {self.global_step}) | "
            f"Loss: {metrics['loss']:.4f} | CE: {metrics['ce_loss']:.4f} | "
            f"KD: {metrics['kd_loss']:.4f} (α={metrics['kd_alpha']:.3f})"
        )
        if current_lr is not None:
            log_msg += f" | LR: {current_lr:.2e}"
        
        logger.info(log_msg)
        
        # Log current memory
        current_mem = self.memory_tracker.get_current_memory()
        logger.info(
            f"  Memory: {current_mem['allocated_gb']:.2f}GB allocated, "
            f"{current_mem['max_allocated_gb']:.2f}GB peak"
        )
        
        # Log to wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "train/loss": metrics['loss'],
                "train/ce_loss": metrics['ce_loss'],
                "train/kd_loss": metrics['kd_loss'],
                "train/kd_alpha": metrics['kd_alpha'],
                "train/stage": stage,
                "train/step_in_stage": step_in_stage,
                "train/global_step": self.global_step,
            }
            if current_lr is not None:
                log_dict["train/learning_rate"] = current_lr
            
            # Add memory metrics
            log_dict.update({
                f"memory/current_allocated_gb": current_mem['allocated_gb'],
                f"memory/current_peak_gb": current_mem['max_allocated_gb'],
            })
            
            wandb.log(log_dict)
    
    def _should_save_checkpoint(self, step_in_stage: int) -> bool:
        """Check if checkpoint should be saved"""
        return step_in_stage % self.config.save_steps == 0
    
    def train_stage(
        self,
        stage: str,
        num_steps: int,
    ) -> int:
        """
        Train for one stage of curriculum learning.
        
        Args:
            stage: Stage name (e.g., "A", "B", "C")
            num_steps: Number of steps to train
            
        Returns:
            Updated global step count
        """
        self.model.train()
        
        # Accumulators for logging
        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        
        logger.info(f"Starting Stage {stage} training for {num_steps} steps")
        
        # Start memory tracking for this stage
        self.memory_tracker.start_tracking()
        
        for step_in_stage in range(1, num_steps + 1):
            batch = self._get_next_batch()
            
            # Training step
            loss_dict = self._training_step(batch, stage, step_in_stage, num_steps)
            
            # Accumulate losses
            total_loss += loss_dict['loss']
            total_ce_loss += loss_dict['ce_loss']
            total_kd_loss += loss_dict['kd_loss']
            
            # Update weights on accumulation boundary
            if step_in_stage % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step()
            
            # Increment global step
            self.global_step += 1
            
            # Periodic logging
            if step_in_stage % self.config.logging_steps == 0:
                avg_metrics = {
                    'loss': total_loss / self.config.logging_steps,
                    'ce_loss': total_ce_loss / self.config.logging_steps,
                    'kd_loss': total_kd_loss / self.config.logging_steps,
                    'kd_alpha': loss_dict['kd_alpha'],
                }
                self._log_metrics(avg_metrics, stage, step_in_stage)
                
                # Log memory summary periodically (every 10 logging steps)
                if (step_in_stage // self.config.logging_steps) % 10 == 0:
                    self.memory_tracker.log_summary(step=self.global_step)
                
                # Reset accumulators
                total_loss = 0.0
                total_ce_loss = 0.0
                total_kd_loss = 0.0
            
            # Save checkpoint
            if self._should_save_checkpoint(step_in_stage):
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=0,
                    step=self.global_step,
                    config=self.config,
                    rank=self.rank,
                    scaler=self.scaler.state_dict() if self.scaler is not None else None,
                    stage=stage
                )
        
        logger.info(f"✓ Completed Stage {stage} ({num_steps} steps)")
        
        # Log final memory summary for this stage
        self.memory_tracker.log_summary(step=self.global_step)
        print_memory_summary(rank=self.rank)
        
        return self.global_step


def train_stage(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    rank: int,
    stage: str,
    num_steps: int,
    global_step_offset: int = 0,
) -> int:
    """
    Standalone function for training a single stage (backward compatibility).
    
    Args:
        model: The model
        train_dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: LR scheduler
        config: Training configuration
        rank: Process rank
        stage: Stage name
        num_steps: Number of steps
        global_step_offset: Starting global step
        
    Returns:
        Updated global step count
    """
    trainer = CurriculumTrainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        rank=rank,
    )
    trainer.global_step = global_step_offset
    
    trainer.train_stage(stage, num_steps)
    
    return trainer.global_step

