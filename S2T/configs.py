"""
Configuration classes for SeamlessM4T v2 training
Extracted from train_kaggle.py for better modularity
"""

from dataclasses import dataclass, field
from typing import Optional

from torch.distributed.fsdp import (
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from functools import partial
import torch


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters with curriculum learning"""
    
    # Model configuration
    model_name_or_path: str = "facebook/seamless-m4t-v2-large"
    is_pretrained: bool = True  # True if loading pretrained weights, False if from scratch
    hf_cache_dir: Optional[str] = None  # Cache directory for HuggingFace models (auto-detect Kaggle if None)
    
    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 4  # ✅ Giảm từ 8 để tiết kiệm ~50% memory
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # ✅ Tăng từ 2 để giữ effective batch size = 8
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    
    # Data configuration
    max_audio_length: int = None  # seconds
    max_text_length: int = None  # tokens
    excel_path: str = "data/ViBa_S2T_train.xlsx"
    audio_col: str = "source"
    vi_col: str = "Tiếng Việt"
    en_col: str = "Tiếng Anh"

    # Curriculum Learning - Stage Configuration
    enable_curriculum: bool = True  # Enable 3-stage training
    
    # Stage A: Encoder-only (decoder frozen) - CRITICAL for low-resource speech
    stage_a_steps: Optional[int] = None  # Auto: max(5000, 0.05 * total_steps); recommend 0.4-0.6 for low-resource
    encoder_lr: float = 1e-4  # HIGH LR: Encoder needs to learn new language from scratch
    stage_a_warmup_pct: float = 0.1 # 20% of stage A steps
    stage_a_warmup_min: int = 1000
    
    # Stage A auto-calculation constants
    min_stage_a_steps: int = 5000
    stage_a_ratio: float = 0.5  # 50% of total steps
    
    # Stage B: Unfreeze top-k decoder layers (progressive, 6 rounds)
    stage_b_steps: Optional[int] = None  # If None, compute from ratio below
    min_stage_b_steps: int = 2000
    stage_b_ratio: float = 0.3  # 30% of total steps
    num_stage_b_rounds: int = 6  # Number of rounds to progressively unfreeze
    unfreeze_top_k: int = 6  # Number of top decoder layers to unfreeze progressively
    decoder_lr: float = 1e-5  # LOW LR: ~10x lower than encoder to preserve Vietnamese knowledge

    
    # Stage C: Full decoder unfrozen - Fine-tuning phase
    use_layer_wise_lr_decay: bool = True  # ENABLE: Protect deeper layers with even lower LR
    layer_wise_lr_decay_rate: float = 0.9  # Each lower layer *= this rate (layer 0 gets lowest LR)
    
    # Knowledge Distillation Weights (dynamic across stages)
    kd_alpha_stage_a: float = 1.0 # KD weight in stage A
    kd_alpha_stage_b: float = 0.8  # KD weight in stage B
    kd_alpha_stage_c_start: float = 0.5  # KD weight at start of stage C
    kd_alpha_stage_c_end: float = 0.5  # KD weight at end of stage C
    weight_ce: float = 1.0  # CE loss weight (constant)
    
    # FSDP configuration
    use_fsdp: bool = True
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    cpu_offload: bool = False
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"  # fp16 or bf16
    
    # AMP configuration
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Torch.compile (PyTorch 2.x)
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"  # typical: inductor
    torch_compile_mode: str = "default"      # default | reduce-overhead | max-autotune
    torch_compile_fullgraph: bool = False
    
    # Checkpointing
    output_dir: str = "./output"
    save_steps: int = 1000
    save_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "seamlessm4t-v2-finetuning-S2T"
    wandb_run_name: Optional[str] = None
    wandb_save_checkpoints: bool = True  # Upload checkpoints to wandb as artifacts
    
    # Optimization
    gradient_checkpointing: bool = True  # ✅ BẬT để giảm 50-70% peak memory
    
    # Scheduler configuration
    min_warmup_steps: int = 200
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1  # Minimum LR ratio for cosine scheduler
    
    # Random seed
    seed: int = 42
    
    # Distributed
    local_rank: int = -1
    world_size: int = 2
    
    def __post_init__(self):
        """Adjust learning rates based on pretrained flag"""
        if self.is_pretrained:
            # Pretrained model: use smaller learning rates (but maintain the ratio)
            # For low-resource speech → high-resource text scenario:
            # - Encoder still needs significant learning (new acoustic features)
            # - Decoder needs very conservative updates (preserve language knowledge)
            if self.encoder_lr == 1e-4:  # If still default
                self.encoder_lr = 5e-5  # Still relatively high for encoder
            if self.decoder_lr == 1e-5:
                self.decoder_lr = 5e-6  # ~10x lower than encoder
    
    @property
    def weight_kd(self):
        """Backward compatibility - returns stage A alpha by default"""
        return self.kd_alpha_stage_a
    
    def calculate_stage_steps(self, total_training_steps: int) -> tuple[int, int, int]:
        """
        Calculate steps for each stage based on total training steps
        
        Args:
            total_training_steps: Total number of training steps
            
        Returns:
            Tuple of (stage_a_steps, stage_b_steps, stage_c_steps)
        """
        # Stage A: auto-calculate if not set
        if self.stage_a_steps is None:
            stage_a_steps = max(self.min_stage_a_steps, int(self.stage_a_ratio * total_training_steps))
        else:
            stage_a_steps = self.stage_a_steps
        
        # Stage B: auto-calculate if not set
        if self.stage_b_steps is None:
            stage_b_steps = max(self.min_stage_b_steps, int(self.stage_b_ratio * total_training_steps))
        else:
            stage_b_steps = self.stage_b_steps
        
        # Stage C: remainder
        stage_c_steps = max(0, total_training_steps - stage_a_steps - stage_b_steps)
        
        return stage_a_steps, stage_b_steps, stage_c_steps


@dataclass
class FSDPConfig:
    """FSDP-specific configuration"""
    
    # Sharding strategies
    SHARDING_STRATEGIES = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    @staticmethod
    def get_mixed_precision_policy(dtype: str = "fp16"):
        """Get mixed precision policy for FSDP"""
        if dtype == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            return None
    
    @staticmethod
    def get_auto_wrap_policy(model_config):
        """Get auto wrap policy for transformer layers"""
        from speech2text_model import (
            SeamlessM4Tv2ConformerEncoderLayer,
            SeamlessM4Tv2DecoderLayer,
            SeamlessM4Tv2EncoderLayer,
        )
        
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                SeamlessM4Tv2ConformerEncoderLayer,
                SeamlessM4Tv2DecoderLayer,
                SeamlessM4Tv2EncoderLayer,
            },
        )

