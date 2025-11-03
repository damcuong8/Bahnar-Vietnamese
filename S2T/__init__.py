"""
This package provides modular components for training SeamlessM4T v2 models
with curriculum learning, FSDP, and knowledge distillation.
"""

__version__ = "2.0.0"

# Core configurations
from .configs import TrainingConfig, FSDPConfig

# Datasets and data processing
from .datasets import (
    ViBaSpeechToTextDataset,
    DataCollatorSpeechToText,
)

# Model utilities
from .model_utils import (
    setup_distributed,
    cleanup_distributed,
    create_model,
    wrap_model_with_fsdp,
)

# Optimizer and scheduler
from .optimizer_utils import (
    create_optimizer,
    add_new_param_groups_to_optimizer,
)
from .scheduler_utils import (
    create_cosine_scheduler,
    get_current_lr_multiplier,
    get_scheduler,
)

# Training stages
from .training_stages import (
    setup_stage_a,
    setup_stage_b,
    setup_stage_c,
    get_kd_alpha,
    freeze_module,
)

# Checkpointing
from .checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
)

# Trainer
from .trainer import CurriculumTrainer, train_stage


__all__ = [
    # Configs
    "TrainingConfig",
    "FSDPConfig",
    
    # Datasets
    "ViBaSpeechToTextDataset",
    "DataCollatorSpeechToText",
    
    # Model utils
    "setup_distributed",
    "cleanup_distributed",
    "create_model",
    "wrap_model_with_fsdp",
    
    # Optimizer & Scheduler
    "create_optimizer",
    "add_new_param_groups_to_optimizer",
    "create_cosine_scheduler",
    "get_current_lr_multiplier",
    "get_scheduler",
    
    # Training stages
    "setup_stage_a",
    "setup_stage_b",
    "setup_stage_c",
    "get_kd_alpha",
    "freeze_module",
    
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    
    # Trainer
    "CurriculumTrainer",
    "train_stage",
]


