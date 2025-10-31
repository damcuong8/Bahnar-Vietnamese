# Tóm Tắt: Upload Checkpoint Lên Wandb

## ✅ Đã Implement

Code training giờ **tự động upload checkpoints lên Wandb** dưới dạng Artifacts với đầy đủ version control.

## 🎯 Cách Hoạt Động

```python
Mỗi lần save checkpoint:
  1. Lưu local: ./output/checkpoint-{step}/pytorch_model.bin
  2. Upload wandb: Artifact "model-checkpoint-step-{step}-stage-{stage}"
  3. Track metadata: step, stage, config
```

## ⚙️ Cấu Hình

### Bật Upload (Mặc Định)

```python
config = TrainingConfig(
    use_wandb=True,
    wandb_save_checkpoints=True,  # ← Bật upload (default)
    save_steps=1000,
)
```

### Tắt Upload

```python
config = TrainingConfig(
    use_wandb=True,  # Vẫn log metrics
    wandb_save_checkpoints=False,  # ← KHÔNG upload checkpoints
)
```

## 📦 Checkpoints Được Upload

| Stage | Khi Nào | Artifact Name |
|-------|---------|---------------|
| A | End of Stage A | `model-checkpoint-step-5000-stage-A` |
| B | Every 2 rounds + end | `model-checkpoint-step-{N}-stage-B` |
| C | End of Stage C | `model-checkpoint-step-15000-stage-C` |
| Any | Every `save_steps` | `model-checkpoint-step-{N}-stage-{S}` |

## 💡 Download & Resume

```python
import wandb

# 1. Download checkpoint
run = wandb.init(project="seamlessm4t-v2-finetuning-S2T")
artifact = run.use_artifact('model-checkpoint-step-8000-stage-B:latest')
artifact_dir = artifact.download()

# 2. Load và resume
checkpoint = torch.load(f"{artifact_dir}/pytorch_model.bin")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

## 📊 Metadata Tracked

Mỗi artifact có metadata:
- ✅ `step`: Training step
- ✅ `stage`: Training stage (A/B/C)
- ✅ `encoder_lr`: Learning rate
- ✅ `is_pretrained`: Model type
- ✅ `enable_curriculum`: Training strategy
- ✅ `unfreeze_top_k`: Config

## ⚠️ Lưu Ý

### Storage Quota
- Wandb free: **100 GB**
- Mỗi checkpoint: **~2-4 GB**
- Full training: **~10-40 GB**

### Upload Time
- 2 GB checkpoint: **~30-60s**
- 4 GB checkpoint: **~1-2 min**
- Upload **blocking** (training đợi)

### Best Practices

✅ **DO**:
- Set `save_steps=1000` trở lên (không quá thường xuyên)
- Set `save_total_limit=2` để giảm storage
- Delete old artifacts khi không cần

❌ **DON'T**:
- `save_steps=100` (quá nhiều uploads)
- Giữ tất cả artifacts (hết quota)

## 🔍 Kiểm Tra

### Trong Logs
```
INFO - Uploading checkpoint to wandb (size: 2345.67 MB)...
INFO - ✓ Successfully uploaded: model-checkpoint-step-5000-stage-A (took 45.3s)
```

### Trên Wandb UI
1. Vào run page
2. Tab "Artifacts"
3. Xem list checkpoints
4. Click để download

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Upload quá lâu | `save_steps=2000` hoặc `wandb_save_checkpoints=False` |
| Hết quota | Delete old artifacts hoặc upgrade plan |
| Upload fail | Training vẫn tiếp tục, chỉ warning |

## 📝 Files Thay Đổi

1. **`train_kaggle.py`**:
   - Added `_upload_checkpoint_to_wandb()` function
   - Updated `save_checkpoint()` to upload
   - Added `wandb_save_checkpoints` config option
   - All `save_checkpoint()` calls now include `stage` parameter

2. **`TrainingConfig`**:
   - Added `wandb_save_checkpoints: bool = True`

3. **Checkpoint structure**:
   - Added `stage` field to saved checkpoints

## ✨ Ưu Điểm

✅ Tự động backup lên cloud  
✅ Version control đầy đủ  
✅ Dễ share và resume  
✅ Track metadata chi tiết  
✅ Safe: training tiếp tục nếu upload fail  
✅ Flexible: có thể bật/tắt  

## 🎯 Kết Luận

- **Default**: Tự động upload (recommended cho production)
- **Testing**: Set `wandb_save_checkpoints=False`
- **Storage**: Cẩn thận với quota (100 GB free)
- **Safe**: Upload fail không ảnh hưởng training

---

**Hoàn thành**: ✅ Đã implement đầy đủ Option 2  
**Ready to use**: Chỉ cần set `use_wandb=True`

