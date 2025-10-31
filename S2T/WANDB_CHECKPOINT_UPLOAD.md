# Wandb Checkpoint Upload

## Tổng Quan

Training script hiện đã hỗ trợ **tự động upload checkpoints lên Wandb** dưới dạng **Artifacts** với version control đầy đủ.

## Tính Năng

### ✅ Tự Động Upload

Mỗi khi lưu checkpoint (mỗi `save_steps` hoặc end of stage), code sẽ:

1. **Lưu local**: `/kaggle/working/output/checkpoint-{step}/pytorch_model.bin`
2. **Upload wandb**: Tạo artifact với tên `model-checkpoint-step-{step}-stage-{stage}`
3. **Metadata**: Ghi lại step, stage, config quan trọng
4. **Version control**: Wandb tự động track versions

### 📦 Artifact Structure

Mỗi checkpoint được lưu dưới dạng wandb artifact với:

```python
Artifact Name: model-checkpoint-step-5000-stage-A
Type: model
Files: pytorch_model.bin
Metadata:
  - step: 5000
  - stage: "A"
  - config:
      - encoder_lr: 1e-4
      - is_pretrained: False
      - enable_curriculum: True
      - unfreeze_top_k: 4
```

### 🎯 Checkpoints Được Upload

| Stage | Step | Artifact Name | Khi Nào |
|-------|------|---------------|---------|
| A | 5000 | `model-checkpoint-step-5000-stage-A` | End of Stage A |
| B | 6000, 7000, 8000 | `model-checkpoint-step-{N}-stage-B-round{R}` | Every 2 rounds |
| B | 8000 | `model-checkpoint-step-8000-stage-B` | End of Stage B |
| C | 15000 | `model-checkpoint-step-15000-stage-C` | End of Stage C (final) |
| Any | Every `save_steps` | `model-checkpoint-step-{N}-stage-{S}` | During training |

## Cấu Hình

### Bật/Tắt Upload

```python
from train_kaggle import TrainingConfig

config = TrainingConfig(
    use_wandb=True,  # Bật wandb logging
    wandb_save_checkpoints=True,  # Bật checkpoint upload (mặc định: True)
    wandb_project="seamlessm4t-v2-finetuning-S2T",
    save_steps=1000,  # Upload mỗi 1000 steps
)
```

### Tắt Upload (chỉ lưu local)

```python
config = TrainingConfig(
    use_wandb=True,  # Vẫn log metrics
    wandb_save_checkpoints=False,  # KHÔNG upload checkpoints
)
```

## Sử Dụng

### 1. Xem Checkpoints Trên Wandb

Truy cập: `https://wandb.ai/{username}/{project}/artifacts`

Bạn sẽ thấy list các artifacts:
- `model-checkpoint-step-5000-stage-A`
- `model-checkpoint-step-8000-stage-B`
- `model-checkpoint-step-15000-stage-C`

### 2. Download Checkpoint Từ Wandb

```python
import wandb

# Initialize wandb
run = wandb.init(project="seamlessm4t-v2-finetuning-S2T")

# Download specific checkpoint
artifact = run.use_artifact('model-checkpoint-step-8000-stage-B:latest')
artifact_dir = artifact.download()

# Load checkpoint
import torch
checkpoint = torch.load(f"{artifact_dir}/pytorch_model.bin")

print(f"Checkpoint from step: {checkpoint['step']}")
print(f"Stage: {checkpoint['stage']}")
```

### 3. Resume Training Từ Wandb Checkpoint

```python
import wandb
import torch

# Download checkpoint
run = wandb.init(project="seamlessm4t-v2-finetuning-S2T")
artifact = run.use_artifact('model-checkpoint-step-5000-stage-A:latest')
artifact_dir = artifact.download()

# Load checkpoint
checkpoint_path = f"{artifact_dir}/pytorch_model.bin"
checkpoint = torch.load(checkpoint_path)

# Resume training
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])

print(f"✓ Resumed from step {checkpoint['step']}, stage {checkpoint['stage']}")
```

### 4. Compare Checkpoints

Wandb UI cho phép compare metadata giữa các checkpoints:
- Learning rates
- Stages
- Config settings
- File sizes

## Lưu Ý Quan Trọng

### 💰 Storage Quota

- **Wandb free tier**: 100 GB storage
- **Mỗi checkpoint**: ~2-4 GB (depending on model size)
- **Stage A → C**: ~3 checkpoints = 6-12 GB
- **Full training**: Có thể 5-10 checkpoints = 10-40 GB

**Khuyến nghị:**
- Set `save_total_limit=2` để giới hạn checkpoints local
- Wandb artifacts không tự động xóa - cần xóa thủ công nếu hết quota

### ⏱️ Upload Time

| Checkpoint Size | Upload Time (Kaggle) |
|-----------------|---------------------|
| 2 GB | ~30-60 seconds |
| 4 GB | ~1-2 minutes |

Upload chạy **đồng bộ** (blocking) - training sẽ đợi upload xong.

### 🔒 Best Practices

1. **Save important checkpoints only**:
   ```python
   save_steps=1000  # Không quá thường xuyên
   save_total_limit=2  # Giữ ít checkpoints local
   ```

2. **Tắt upload cho testing**:
   ```python
   wandb_save_checkpoints=False  # When testing code
   ```

3. **Use stage checkpoints**:
   - End of Stage A: Encoder trained
   - End of Stage B: Partial decoder adapted
   - End of Stage C: Final model

4. **Delete old artifacts**:
   - Vào wandb UI → Artifacts → Delete unused versions

### ⚠️ Failure Handling

Nếu upload thất bại:
- Training **tiếp tục** (không dừng)
- Warning log xuất hiện
- Checkpoint vẫn được lưu local

```python
try:
    _upload_checkpoint_to_wandb(...)
except Exception as e:
    logger.warning(f"Failed to upload: {e}")  # Continue training
```

## Kiểm Tra Upload

### Trong Logs

```
INFO - Uploading checkpoint to wandb (size: 2345.67 MB)...
INFO - ✓ Successfully uploaded checkpoint to wandb: model-checkpoint-step-5000-stage-A (took 45.3s)
```

### Trên Wandb UI

1. Vào run page
2. Click tab "Artifacts"
3. Xem list checkpoints
4. Click vào artifact để xem metadata và download

## Troubleshooting

### Issue: Upload quá lâu

**Solution**: 
```python
config.save_steps = 2000  # Giảm tần suất save
config.wandb_save_checkpoints = False  # Hoặc tắt upload
```

### Issue: Hết quota

**Solution**:
1. Delete old artifacts trên wandb UI
2. Hoặc upgrade wandb plan
3. Hoặc tắt `wandb_save_checkpoints`

### Issue: Upload fail liên tục

**Solution**:
```python
# Kiểm tra wandb login
!wandb login

# Check internet connection
!ping -c 3 wandb.ai

# Tắt upload nếu vẫn lỗi
config.wandb_save_checkpoints = False
```

## Examples

### Example 1: Minimal Upload

```python
config = TrainingConfig(
    use_wandb=True,
    wandb_save_checkpoints=True,
    save_steps=5000,  # Chỉ save 1 lần/stage
    save_total_limit=1,  # Chỉ giữ checkpoint mới nhất
)
```

### Example 2: Frequent Checkpoints

```python
config = TrainingConfig(
    use_wandb=True,
    wandb_save_checkpoints=True,
    save_steps=500,  # Save thường xuyên
    save_total_limit=3,  # Giữ 3 checkpoints local
)
```

### Example 3: No Upload (Local Only)

```python
config = TrainingConfig(
    use_wandb=True,  # Log metrics only
    wandb_save_checkpoints=False,  # No checkpoint upload
    save_steps=1000,
)
```

## Technical Details

### Artifact Structure

```
artifact/
└── pytorch_model.bin
    ├── model: state_dict
    ├── optimizer: state_dict
    ├── scheduler: state_dict
    ├── epoch: int
    ├── step: int
    ├── stage: str
    └── scaler: state_dict (optional)
```

### Upload Implementation

```python
def _upload_checkpoint_to_wandb(checkpoint_path, step, stage, config):
    # 1. Create artifact
    artifact = wandb.Artifact(
        name=f"model-checkpoint-step-{step}-stage-{stage}",
        type="model",
        metadata={...}
    )
    
    # 2. Add file
    artifact.add_file(checkpoint_path)
    
    # 3. Log to wandb
    wandb.log_artifact(artifact)
    
    # 4. Also save as simple file (backup)
    wandb.save(checkpoint_path)
```

## Comparison: Local vs Wandb

| Aspect | Local Only | + Wandb Artifacts |
|--------|-----------|-------------------|
| Storage | Kaggle disk (~20GB) | Wandb cloud (100GB free) |
| Persistence | Lost after session | Permanent |
| Sharing | Manual download | Share link |
| Version control | Manual | Automatic |
| Metadata | None | Full tracking |
| Download | From Kaggle UI | API or UI |
| Cost | Free | Free (with quota) |

## Summary

✅ **Tự động backup** checkpoints lên cloud  
✅ **Version control** đầy đủ với metadata  
✅ **Dễ dàng share** và resume training  
✅ **Flexible** - có thể bật/tắt  
✅ **Safe** - training tiếp tục nếu upload fail  

---

**Version**: 1.0  
**Tương thích**: SeamlessM4T v2, Wandb 0.15+  
**Recommended**: Bật cho production training, tắt cho testing

