# 📦 Module Structure - SeamlessM4T v2 Training

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  train_kaggle_refactored.py                 │
│                    (Main Entry Point)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ imports & orchestrates
                 │
    ┌────────────┴────────────┬──────────────┬─────────────────┬──────────────┐
    │                         │              │                 │              │
    ▼                         ▼              ▼                 ▼              ▼
┌─────────┐            ┌──────────┐    ┌──────────┐     ┌──────────┐   ┌──────────┐
│ configs │            │ datasets │    │  model   │     │optimizer │   │scheduler │
│  .py    │            │   .py    │    │ _utils   │     │ _utils   │   │ _utils   │
│         │            │          │    │   .py    │     │   .py    │   │   .py    │
└─────────┘            └──────────┘    └──────────┘     └──────────┘   └──────────┘
    │                         │              │                 │              │
    │                         │              │                 │              │
    │                         └──────────────┴─────────────────┴──────────────┘
    │                                        │
    │                                        ▼
    │                              ┌──────────────────┐
    │                              │   trainer.py     │
    │                              │                  │
    │                              │ CurriculumTrainer│
    │                              └────────┬─────────┘
    │                                       │
    │                                       │ uses
    │                                       │
    └───────────────┬───────────────────────┴──────────────────┐
                    │                                           │
                    ▼                                           ▼
          ┌──────────────────┐                        ┌──────────────────┐
          │ training_stages  │                        │  checkpoint      │
          │      .py         │                        │   _utils.py      │
          │                  │                        │                  │
          │ - setup_stage_a  │                        │ - save_checkpoint│
          │ - setup_stage_b  │                        │ - load_checkpoint│
          │ - setup_stage_c  │                        │ - wandb_upload   │
          └──────────────────┘                        └──────────────────┘
```

## Module Dependencies

```
┌────────────────────────────────────────────────────────────────┐
│                      External Dependencies                      │
├────────────────────────────────────────────────────────────────┤
│  torch, transformers, torchaudio, pandas, wandb, numpy, etc.   │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                    Base Modules (No Dependencies)               │
├────────────────────────────────────────────────────────────────┤
│  - configs.py          (only dataclasses + torch.distributed)  │
│  - utils.py            (helper functions)                      │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                   Data Layer (depends on configs)               │
├────────────────────────────────────────────────────────────────┤
│  - datasets.py         (Dataset + DataCollator)                │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                  Model Layer (depends on configs)               │
├────────────────────────────────────────────────────────────────┤
│  - model_utils.py      (model creation, FSDP, distributed)     │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│              Optimization Layer (depends on configs)            │
├────────────────────────────────────────────────────────────────┤
│  - optimizer_utils.py  (optimizer creation)                    │
│  - scheduler_utils.py  (LR schedulers)                         │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│           Training Logic Layer (depends on all above)           │
├────────────────────────────────────────────────────────────────┤
│  - training_stages.py  (stage setup for curriculum learning)   │
│  - checkpoint_utils.py (save/load checkpoints)                 │
│  - trainer.py          (training loop orchestration)           │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                Application Layer (Main Entry Point)             │
├────────────────────────────────────────────────────────────────┤
│  - train_kaggle_refactored.py  (orchestrates everything)       │
└────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### 🔧 configs.py
**Purpose**: Central configuration management
- `TrainingConfig`: All training hyperparameters
- `FSDPConfig`: FSDP-specific settings
- Helper methods for stage calculation

**Exports**:
- `TrainingConfig`
- `FSDPConfig`

### 📊 datasets.py
**Purpose**: Data loading and preprocessing
- `DummySpeechToTextDataset`: Placeholder dataset
- `ViBaSpeechToTextDataset`: Real dataset with caching
- `DataCollatorSpeechToText`: Batching and padding

**Exports**:
- `DummySpeechToTextDataset`
- `ViBaSpeechToTextDataset`
- `DataCollatorSpeechToText`

### 🤖 model_utils.py
**Purpose**: Model creation and distributed setup
- Model initialization
- Pretrained weight loading
- FSDP wrapping
- Distributed process group setup

**Exports**:
- `create_model()`
- `wrap_model_with_fsdp()`
- `setup_distributed()`
- `cleanup_distributed()`

### ⚡ optimizer_utils.py
**Purpose**: Optimizer creation and management
- Stage-specific optimizer creation (A/B/C)
- Parameter grouping (decay/no_decay)
- Layer-wise learning rate decay
- Dynamic parameter addition

**Exports**:
- `create_optimizer()`
- `create_optimizer_stage_a()`
- `create_optimizer_stage_b()`
- `create_optimizer_stage_c()`
- `add_new_param_groups_to_optimizer()`

### 📈 scheduler_utils.py
**Purpose**: Learning rate scheduling
- Cosine annealing with warmup
- Linear warmup
- LR multiplier calculation

**Exports**:
- `create_cosine_scheduler()`
- `get_current_lr_multiplier()`
- `get_scheduler()`

### 🎓 training_stages.py
**Purpose**: Curriculum learning stage setup
- Stage A: Encoder-only training
- Stage B: Progressive decoder unfreezing
- Stage C: Full model training
- KD alpha calculation

**Exports**:
- `setup_stage_a()`
- `setup_stage_b()`
- `setup_stage_c()`
- `get_kd_alpha()`
- `freeze_module()`

### 💾 checkpoint_utils.py
**Purpose**: Checkpoint management
- Save/load checkpoints
- FSDP state dict handling
- Wandb artifact upload
- Old checkpoint cleanup

**Exports**:
- `save_checkpoint()`
- `load_checkpoint()`

### 🚂 trainer.py
**Purpose**: Training loop orchestration
- `CurriculumTrainer` class
- Training step logic
- Metrics logging
- AMP (Automatic Mixed Precision)

**Exports**:
- `CurriculumTrainer`
- `train_stage()` (standalone function)

### 🎯 train_kaggle_refactored.py
**Purpose**: Main entry point
- Orchestrates all components
- Implements 3-stage curriculum learning
- Handles distributed training
- Wandb integration

## Data Flow

```
┌─────────────┐
│   Config    │
│ (YAML/Code) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Initialization Phase                      │
│  1. Setup distributed (rank, world_size)                    │
│  2. Create model (with optional pretrained weights)         │
│  3. Load datasets                                            │
│  4. Initialize wandb                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Stage A Training                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Freeze decoder, unfreeze encoder                 │   │
│  │ 2. Create optimizer (encoder params only)           │   │
│  │ 3. Train for N steps                                │   │
│  │ 4. Save checkpoint                                   │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Stage B Training                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ For each round (1 to 6):                            │   │
│  │   1. Unfreeze more decoder layers                   │   │
│  │   2. Add params to optimizer                        │   │
│  │   3. Train for M steps                              │   │
│  │   4. Save checkpoint (every 2 rounds)               │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Stage C Training                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Unfreeze all decoder layers                      │   │
│  │ 2. Add remaining params to optimizer                │   │
│  │ 3. Train for K steps                                │   │
│  │ 4. Save final checkpoint                            │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Finalization                            │
│  1. Save final model                                        │
│  2. Upload to wandb                                         │
│  3. Cleanup distributed                                     │
│  4. Close wandb                                             │
└─────────────────────────────────────────────────────────────┘
```

## Import Graph

```
train_kaggle_refactored.py
    │
    ├── configs.py (no internal deps)
    │
    ├── datasets.py
    │   └── seamless_feature_extractor.py
    │
    ├── model_utils.py
    │   ├── configs.py
    │   ├── speech2text_model.py
    │   └── seamless_m4t_v2_config.py
    │
    ├── optimizer_utils.py
    │   └── configs.py
    │
    ├── scheduler_utils.py
    │   └── configs.py
    │
    ├── training_stages.py
    │   └── configs.py
    │
    ├── checkpoint_utils.py
    │   └── configs.py
    │
    └── trainer.py
        ├── configs.py
        ├── training_stages.py
        └── checkpoint_utils.py
```

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                       Unit Tests                             │
├─────────────────────────────────────────────────────────────┤
│  test_configs.py       - Config validation                  │
│  test_datasets.py      - Data loading & collation           │
│  test_optimizer.py     - Optimizer creation                 │
│  test_scheduler.py     - LR scheduling                      │
│  test_stages.py        - Freeze/unfreeze logic              │
│  test_checkpoint.py    - Save/load functionality            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Integration Tests                         │
├─────────────────────────────────────────────────────────────┤
│  test_trainer.py       - Training loop                      │
│  test_curriculum.py    - 3-stage pipeline                   │
│  test_distributed.py   - Multi-GPU training                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      End-to-End Tests                        │
├─────────────────────────────────────────────────────────────┤
│  test_full_training.py - Complete training run              │
└─────────────────────────────────────────────────────────────┘
```

## File Size Breakdown

| Module | Lines | % of Total | Complexity |
|--------|-------|-----------|------------|
| configs.py | 188 | 8% | Low |
| datasets.py | 300 | 13% | Medium |
| model_utils.py | 180 | 8% | Low |
| optimizer_utils.py | 440 | 19% | Medium |
| scheduler_utils.py | 150 | 6% | Low |
| training_stages.py | 200 | 9% | Low |
| checkpoint_utils.py | 240 | 10% | Medium |
| trainer.py | 260 | 11% | Medium |
| train_kaggle_refactored.py | 290 | 12% | Low |
| __init__.py | 90 | 4% | Low |
| **TOTAL** | **2,338** | **100%** | **Low-Medium** |

## Summary

✅ **Clear separation of concerns**  
✅ **Low coupling between modules**  
✅ **High cohesion within modules**  
✅ **Easy to test and maintain**  
✅ **Scalable architecture**


