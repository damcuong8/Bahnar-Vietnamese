# Curriculum Learning for SeamlessM4T v2

## Overview

This implementation features a **3-stage curriculum learning strategy** for fine-tuning SeamlessM4T v2 speech-to-text models. The progressive unfreezing approach helps achieve better convergence and model quality.

## Training Stages

### Stage A: Encoder-Only Training
**Goal**: Train the speech encoder to extract good audio representations while keeping the decoder frozen.

#### Configuration
- **Duration**: `max(5000, 5% of total steps)` or custom via `stage_a_steps`
- **Trainable**: Speech encoder only
- **Frozen**: Text encoder (teacher), text decoder, LM head
- **Learning Rate**: 
  - From scratch: `1e-4`
  - Pretrained: `1e-5`
- **Warmup**: 3% of stage steps (min 200)
- **KD Alpha**: `1.0` (full knowledge distillation)

#### Why This Works
- Focuses on learning audio → latent representation mapping
- Prevents decoder from adapting too quickly to noisy encoder outputs
- KD from pivot text helps encoder learn meaningful representations

### Stage B: Unfreeze Top-K Decoder Layers
**Goal**: Begin adapting the decoder to work with the newly trained encoder.

#### Configuration
- **Duration**: `3000 steps` (configurable via `stage_b_steps`)
- **Trainable**: Speech encoder + top-k decoder layers + LM head
- **Top-K**: `4 layers` by default (configurable via `unfreeze_top_k`)
- **Learning Rates**:
  - Encoder: Same as Stage A
  - Decoder top layers:
    - From scratch: `3e-4`
    - Pretrained: `5e-5`
- **Warmup**: 2% of stage steps (min 100)
- **KD Alpha**: `0.8` (still high)

#### Why This Works
- Gradual adaptation prevents catastrophic forgetting
- Top layers learn output formatting while lower layers remain stable
- Separate LRs allow encoder to continue refining while decoder adapts

### Stage C: Full Decoder Fine-Tuning
**Goal**: Fine-tune all components together with decreasing knowledge distillation.

#### Configuration
- **Duration**: Remaining steps
- **Trainable**: All parameters (encoder + full decoder + LM head)
- **Learning Rates**:
  - Encoder: Same as Stage A
  - Full decoder:
    - From scratch: `5e-5`
    - Pretrained: `2e-5`
- **Layer-wise LR Decay**: Optional
  - Decay rate: `0.9` per layer
  - Top layers get higher LR
- **Warmup**: 1% of stage steps (min 100)
- **KD Alpha**: Linear decay from `0.5` → `0.0`

#### Why This Works
- Full model coordination after encoder and top decoder are adapted
- Gradual reduction of KD allows model to learn from its own predictions
- Layer-wise LR helps fine-grained control over adaptation rate

## Knowledge Distillation Schedule

### Dynamic KD Weight (α)

```
Stage A: α = 1.0 (constant)
Stage B: α = 0.8 (constant)
Stage C: α = 0.5 → 0.0 (linear decay)
```

**Loss Formula:**
```python
total_loss = weight_ce * CE_loss + α * KD_loss
```

Where:
- `CE_loss`: Cross-entropy loss (student predictions vs labels)
- `KD_loss`: Knowledge distillation loss (student vs teacher logits)
- `α`: Dynamic KD weight
- `weight_ce`: Constant CE weight (default 1.0)

## Configuration

### Quick Start (Default Settings)

```python
from train_kaggle import TrainingConfig

# For training from scratch
config = TrainingConfig(
    is_pretrained=False,
    enable_curriculum=True,
    num_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

# For fine-tuning pretrained model
config = TrainingConfig(
    is_pretrained=True,  # Automatically adjusts LRs
    enable_curriculum=True,
    # ... other settings
)
```

### Advanced Configuration

```python
config = TrainingConfig(
    # Curriculum settings
    enable_curriculum=True,
    
    # Stage A
    stage_a_steps=8000,  # Override auto-calculation
    encoder_lr=1e-4,
    stage_a_warmup_pct=0.03,
    kd_alpha_stage_a=1.0,
    
    # Stage B
    stage_b_steps=3000,
    unfreeze_top_k=4,
    decoder_lr_unfreeze_top=3e-4,
    stage_b_warmup_pct=0.02,
    kd_alpha_stage_b=0.8,
    
    # Stage C
    decoder_lr_full=5e-5,
    stage_c_warmup_pct=0.01,
    kd_alpha_stage_c_start=0.5,
    kd_alpha_stage_c_end=0.0,
    use_layer_wise_lr_decay=True,
    layer_wise_lr_decay_rate=0.9,
    
    # General
    max_grad_norm=1.0,
    weight_decay=0.01,
)
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Initialize Model                                            │
│ ├─ Speech Encoder                                           │
│ ├─ Text Encoder (teacher - always frozen)                   │
│ ├─ Text Decoder                                             │
│ └─ LM Head                                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE A: Encoder-Only (5000+ steps)                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Freeze:    Text Encoder, Text Decoder, LM Head         │ │
│ │ Train:     Speech Encoder                              │ │
│ │ LR:        1e-4 (scratch) / 1e-5 (pretrained)          │ │
│ │ KD Alpha:  1.0                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  → Rebuild optimizer & scheduler                            │
│  → dist.barrier() for sync                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE B: Unfreeze Top-4 Decoder Layers (3000 steps)         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Freeze:    Text Encoder, Bottom Decoder Layers         │ │
│ │ Train:     Speech Encoder + Top-4 Decoder + LM Head    │ │
│ │ LR:        Encoder 1e-4, Decoder 3e-4                  │ │
│ │ KD Alpha:  0.8                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  → Rebuild optimizer & scheduler                            │
│  → dist.barrier() for sync                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE C: Full Decoder (Remaining steps)                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Freeze:    Text Encoder only                           │ │
│ │ Train:     Speech Encoder + All Decoder + LM Head      │ │
│ │ LR:        Encoder 1e-4, Decoder 5e-5                  │ │
│ │ KD Alpha:  0.5 → 0.0 (linear decay)                    │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ Training Complete│
                └─────────────────┘
```

## Key Implementation Details

### 1. Optimizer Rebuilding

After each stage transition, the optimizer **must** be rebuilt to include new parameters:

```python
# Stage A → B transition
setup_stage_b(model, config)  # Unfreeze top-k layers
dist.barrier()  # Sync all ranks
optimizer_b = create_optimizer(model, config, stage="B")  # Rebuild
scheduler_b = create_scheduler(optimizer_b, ...)
```

### 2. Distributed Synchronization

Always use `dist.barrier()` after unfreezing parameters in multi-GPU training:

```python
if config.world_size > 1:
    dist.barrier()  # Ensure all ranks see updated requires_grad
```

### 3. FSDP Compatibility

The optimizer receives FSDP-wrapped parameters automatically:

```python
# FSDP wrapping happens BEFORE creating optimizer
model = wrap_model_with_fsdp(model, config, model_config, rank)
optimizer = create_optimizer(model, config, stage="A")
```

### 4. Warmup Per Stage

Each stage gets its own warmup:

```python
warmup_a = max(200, int(0.03 * stage_a_steps))  # 3%, min 200
warmup_b = max(100, int(0.02 * stage_b_steps))  # 2%, min 100
warmup_c = max(100, int(0.01 * stage_c_steps))  # 1%, min 100
```

## Monitoring with Wandb

### Tracked Metrics

```python
wandb.log({
    "train/loss": total_loss,
    "train/ce_loss": ce_loss,
    "train/kd_loss": kd_loss,
    "train/kd_alpha": current_kd_alpha,
    "train/learning_rate": current_lr,
    "train/stage": "A" | "B" | "C",
    "train/step_in_stage": step_within_current_stage,
    "train/global_step": overall_step_count,
})
```

### Dashboard Tips

1. **Track KD Alpha**: Watch how KD weight decreases in Stage C
2. **Monitor LR per Stage**: Verify warmup and decay
3. **Compare Losses**: CE vs KD trends show learning progress
4. **Stage Transitions**: Look for smooth transitions (no spikes)

## Performance Tips

### Memory Optimization

```python
# After each stage, cleanup
del optimizer_a, scheduler_a
torch.cuda.empty_cache()
```

### Batch Size Tuning

Stage A needs less memory (encoder only):
- Start with larger batch in Stage A
- Reduce in Stage B/C if OOM

### Checkpoint Strategy

Save checkpoints at stage boundaries:
- End of Stage A (encoder ready)
- End of Stage B (partial decoder adapted)
- End of Stage C (final model)

## Troubleshooting

### Issue: Optimizer not updating new parameters

**Solution**: Ensure you rebuild the optimizer after unfreezing:

```python
setup_stage_b(model, config)  # Unfreeze
dist.barrier()  # Sync
optimizer = create_optimizer(model, config, stage="B")  # NEW optimizer
```

### Issue: FSDP gradient sync errors

**Solution**: Always call `dist.barrier()` after parameter state changes:

```python
setup_stage_c(model, config)
if config.world_size > 1:
    dist.barrier()  # Critical!
```

### Issue: Learning stagnates in Stage C

**Solutions**:
1. Increase `decoder_lr_full`
2. Reduce `kd_alpha_stage_c_start` (less reliance on teacher)
3. Enable `use_layer_wise_lr_decay`

### Issue: Catastrophic forgetting in Stage B/C

**Solutions**:
1. Reduce decoder learning rate
2. Increase `kd_alpha_stage_b` (keep teacher guidance)
3. Add more steps to Stage B before full unfreezing

## Advanced Features

### Layer-wise Learning Rate Decay

Enable differential learning rates across decoder layers:

```python
config.use_layer_wise_lr_decay = True
config.layer_wise_lr_decay_rate = 0.9

# Result: Top layer LR = decoder_lr_full
#         Layer i LR = decoder_lr_full * (0.9 ** (num_layers - i - 1))
```

### Custom Stage Durations

```python
# Total steps = 30,000
config.stage_a_steps = 10000  # 33%
config.stage_b_steps = 5000   # 17%
# Stage C gets remaining 15,000 (50%)
```

### Disable Curriculum Learning

```python
config.enable_curriculum = False  # Falls back to standard training
# (Not fully implemented - use curriculum for best results)
```

## Best Practices

### 1. Start Conservative

```python
# First run: use defaults
config = TrainingConfig(is_pretrained=False)
```

### 2. Monitor Closely

- Watch Stage A carefully - encoder quality is critical
- If Stage A loss plateaus early, may reduce steps
- If Stage B shows instability, increase warmup

### 3. Adjust Progressively

Don't change all hyperparameters at once:
1. First run: default settings
2. If needed: adjust stage durations
3. If needed: adjust learning rates
4. If needed: tune KD schedule

### 4. Validate Each Stage

- Save checkpoints at stage boundaries
- Test intermediate models on validation set
- Verify encoder quality before moving to Stage B

## Example Runs

### Small Dataset (10K samples)

```python
config = TrainingConfig(
    num_train_samples=10000,
    num_epochs=3,
    stage_a_steps=2000,  # Smaller for small dataset
    stage_b_steps=1000,
    # Rest defaults
)
```

### Large Dataset (1M samples)

```python
config = TrainingConfig(
    num_train_samples=1000000,
    num_epochs=1,  # One pass is enough
    stage_a_steps=20000,
    stage_b_steps=10000,
    # Rest defaults
)
```

### Pretrained Model Fine-tuning

```python
config = TrainingConfig(
    is_pretrained=True,  # Auto-adjusts LRs
    num_epochs=2,
    stage_a_steps=5000,
    stage_b_steps=2000,
    kd_alpha_stage_a=0.9,  # Slightly lower for pretrained
)
```

## References

- **Progressive Unfreezing**: [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146)
- **Knowledge Distillation**: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- **Discriminative Fine-tuning**: [Slanted Triangular Learning Rates](https://arxiv.org/abs/1801.06146)

---

**Version**: 1.0  
**Last Updated**: 2024  
**Compatibility**: SeamlessM4T v2, PyTorch 2.0+, FSDP

