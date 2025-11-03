# 🚀 Quick Start Guide - SeamlessM4T v2 Training

## Prerequisites

```bash
# Install dependencies
pip install torch transformers torchaudio pandas openpyxl wandb

# Optional: Install wandb for experiment tracking
wandb login
```

## 1. Verify Setup (IMPORTANT!)

Before training, verify your environment:

```bash
cd S2T
python test_setup.py
```

Expected output:
```
✅ PASS: Imports
✅ PASS: GPU Availability
✅ PASS: Model Creation
✅ PASS: Forward Pass
✅ PASS: Data Loader
✅ PASS: FSDP Wrapping
✅ PASS: Disk Space
======================================================================
Results: 7/7 tests passed
🎉 All tests passed! You're ready to start training.
```

## 2. Training Options

### Option A: Using Refactored Modules (Recommended)

The new modular structure makes it easy to customize:

```python
from configs import TrainingConfig
from datasets import ViBaSpeechToTextDataset, DataCollatorSpeechToText
from model_utils import create_model, setup_distributed
from trainer import CurriculumTrainer
from seamless_feature_extractor import SeamlessM4TFeatureExtractor
from transformers import AutoProcessor
from torch.utils.data import DataLoader

# 1. Setup
rank, world_size, local_rank = setup_distributed()

# 2. Configure
config = TrainingConfig(
    num_epochs=5,
    per_device_train_batch_size=2,
    encoder_lr=1e-4,
    decoder_lr=1e-5,
    enable_curriculum=True,
    use_wandb=True,
)

# 3. Load data
dataset = ViBaSpeechToTextDataset(
    excel_path="your_data.xlsx",
    audio_col="source",
    vi_col="Tiếng Việt",
    en_col="Tiếng Anh",
    use_cache=True  # Enable caching for faster training
)

feature_extractor = SeamlessM4TFeatureExtractor(...)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

collator = DataCollatorSpeechToText(
    feature_extractor=feature_extractor,
    processor=processor,
    target_language="vi",
    pivot_language="en"
)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)

# 4. Create model
model, model_config = create_model(config)

# 5. Train
trainer = CurriculumTrainer(
    model=model,
    train_dataloader=dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    rank=rank,
)

# Train Stage A (Encoder only)
trainer.train_stage(stage="A", num_steps=1000)
```

### Option B: Using Original Script

If you prefer the original monolithic script:

```bash
# Single GPU
python train_kaggle.py

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 train_kaggle.py
```

## 3. Configuration

Edit `configs.py` or create custom config:

```python
from configs import TrainingConfig

config = TrainingConfig(
    # Model
    model_name_or_path="facebook/seamless-m4t-v2-large",
    is_pretrained=True,  # Load pretrained weights
    
    # Training
    num_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    
    # Learning rates (IMPORTANT!)
    encoder_lr=1e-4,  # High LR for encoder (learning new language)
    decoder_lr=1e-5,  # Low LR for decoder (preserve knowledge)
    
    # Curriculum Learning
    enable_curriculum=True,
    stage_a_steps=5000,  # Encoder-only training
    stage_b_steps=10000,  # Progressive decoder unfreezing
    num_stage_b_rounds=6,  # Unfreeze in 6 rounds
    
    # Knowledge Distillation
    kd_alpha_stage_a=1.0,
    kd_alpha_stage_b=0.8,
    kd_alpha_stage_c_start=0.5,
    kd_alpha_stage_c_end=0.2,
    
    # FSDP (for multi-GPU)
    use_fsdp=True,
    sharding_strategy="FULL_SHARD",
    use_mixed_precision=True,
    
    # Logging
    use_wandb=True,
    wandb_project="my-seamlessm4t-project",
    logging_steps=10,
    save_steps=1000,
)
```

## 4. Data Preparation

### Format 1: Excel File

Create an Excel file with columns:
- `source`: Path to audio file
- `Tiếng Việt`: Vietnamese transcription
- `Tiếng Anh`: English translation (optional, for pivot)

Example:
```
| source              | Tiếng Việt           | Tiếng Anh          |
|---------------------|----------------------|--------------------|
| audio/sample1.wav   | Xin chào             | Hello              |
| audio/sample2.wav   | Tạm biệt             | Goodbye            |
```

### Format 2: Custom Dataset

```python
from datasets import ViBaSpeechToTextDataset

dataset = ViBaSpeechToTextDataset(
    excel_path="data.xlsx",
    audio_col="source",
    vi_col="Tiếng Việt",
    en_col="Tiếng Anh",
    target_sr=16000,
    mono=True,
    use_cache=True,  # Cache processed samples in memory
)
```

## 5. Monitoring Training

### With Wandb (Recommended)

```python
config = TrainingConfig(
    use_wandb=True,
    wandb_project="seamlessm4t-training",
    wandb_run_name="experiment-1",
    wandb_save_checkpoints=True,  # Upload checkpoints to wandb
)
```

View training at: https://wandb.ai/your-username/seamlessm4t-training

### Without Wandb

Training logs will be printed to console:
```
Stage A | Step 100/5000 (Global: 100) | Loss: 2.3456 | CE: 1.2345 | KD: 1.1111 (α=1.000) | LR: 1.00e-04
```

## 6. Checkpointing

Checkpoints are automatically saved every `save_steps`:

```
output/
├── checkpoint-1000/
│   └── pytorch_model.bin
├── checkpoint-2000/
│   └── pytorch_model.bin
└── checkpoint-3000/
    └── pytorch_model.bin
```

### Resume from Checkpoint

```python
config = TrainingConfig(
    resume_from_checkpoint="output/checkpoint-2000",
)
```

## 7. Common Issues

### Issue 1: Out of Memory (OOM)

**Solution:**
```python
config = TrainingConfig(
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=8,  # Increase accumulation
    use_mixed_precision=True,  # Use FP16
    gradient_checkpointing=True,  # Enable gradient checkpointing
)
```

### Issue 2: Slow Training

**Solution:**
```python
# Enable data caching
dataset = ViBaSpeechToTextDataset(
    excel_path="data.xlsx",
    use_cache=True,  # Cache processed samples
)

# Use more workers
dataloader = DataLoader(
    dataset,
    num_workers=4,  # Increase workers
    pin_memory=True,
)
```

### Issue 3: Import Errors

**Solution:**
```bash
# Make sure you're in the S2T directory
cd S2T

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/speech_to_text/S2T
```

## 8. Curriculum Learning Stages

The training follows 3 stages:

### Stage A: Encoder-Only (40-60% of total steps)
- **Goal**: Learn acoustic features of new language
- **Frozen**: Decoder, LM head
- **Trainable**: Speech encoder
- **LR**: High (1e-4)

### Stage B: Progressive Decoder Unfreezing (6 rounds)
- **Goal**: Gradually adapt decoder to new acoustic features
- **Frozen**: Bottom decoder layers (gradually unfrozen)
- **Trainable**: Encoder + top decoder layers
- **LR**: Encoder (1e-4), Decoder (1e-5)

### Stage C: Full Fine-tuning
- **Goal**: Fine-tune entire model
- **Frozen**: None
- **Trainable**: All parameters
- **LR**: Layer-wise decay (optional)

## 9. Tips for Best Results

1. **Start with pretrained weights**: `is_pretrained=True`
2. **Use curriculum learning**: `enable_curriculum=True`
3. **Monitor KD alpha**: Should decrease over time
4. **Check loss curves**: CE loss should decrease steadily
5. **Save frequently**: `save_steps=500` for important experiments
6. **Use wandb**: Track experiments and compare runs

## 10. Next Steps

- Read `REFACTORING_GUIDE.md` for advanced usage
- Read `CURRICULUM_LEARNING.md` for theory
- Read `BUGFIX_SUMMARY.md` if you encounter errors
- Check `test_setup.py` for environment verification

---

## Quick Commands Reference

```bash
# Verify setup
python test_setup.py

# Train (single GPU)
python train_kaggle.py

# Train (multi-GPU)
torchrun --nproc_per_node=2 train_kaggle.py

# Monitor with wandb
wandb login
python train_kaggle.py  # (with use_wandb=True in config)

# Check GPU usage
nvidia-smi -l 1
```

---

**Need help?** Check the documentation files or open an issue!

