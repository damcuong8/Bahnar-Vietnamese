# Tóm Tắt: Curriculum Learning cho SeamlessM4T v2

## Tổng Quan

Đã triển khai đầy đủ **chiến lược học theo chương trình 3 giai đoạn (curriculum learning)** cho việc fine-tune model SeamlessM4T v2 Speech-to-Text với FSDP trên 2 GPU.

## 3 Giai Đoạn Huấn Luyện

### 🎯 Stage A: Chỉ Train Encoder (Decoder Đóng Băng)

**Mục tiêu**: Train speech encoder học biểu diễn audio tốt

**Cấu hình**:
- ✅ **Trainable**: Chỉ speech encoder
- ❄️ **Frozen**: Text encoder (teacher), text decoder, lm_head
- 📊 **Steps**: `max(5000, 5% tổng steps)`
- 🎓 **Learning Rate**: 
  - From scratch: `1e-4`
  - Pretrained: `1e-5`
- 🔥 **Warmup**: 3% steps (tối thiểu 200)
- 🎯 **KD Alpha**: `1.0` (KD đầy đủ)

**Output**: Encoder đã học được audio → latent representations tốt

---

### 🎯 Stage B: Mở Top-K Decoder Layers

**Mục tiêu**: Bắt đầu điều chỉnh decoder làm việc với encoder mới

**Cấu hình**:
- ✅ **Trainable**: Speech encoder + top-4 decoder layers + lm_head
- ❄️ **Frozen**: Text encoder, các decoder layers còn lại
- 📊 **Steps**: `3000` (có thể điều chỉnh)
- 🎓 **Learning Rates**:
  - Encoder: giữ nguyên từ Stage A
  - Top decoder layers:
    - From scratch: `3e-4`
    - Pretrained: `5e-5`
- 🔥 **Warmup**: 2% steps (tối thiểu 100)
- 🎯 **KD Alpha**: `0.8`

**Output**: Decoder layers trên cùng đã thích nghi với encoder mới

---

### 🎯 Stage C: Mở Toàn Bộ Decoder

**Mục tiêu**: Fine-tune toàn bộ model, giảm dần KD

**Cấu hình**:
- ✅ **Trainable**: Speech encoder + toàn bộ decoder + lm_head
- ❄️ **Frozen**: Chỉ text encoder (teacher)
- 📊 **Steps**: Phần còn lại
- 🎓 **Learning Rates**:
  - Encoder: giữ nguyên
  - Full decoder:
    - From scratch: `5e-5`
    - Pretrained: `2e-5`
- 🔥 **Warmup**: 1% steps (tối thiểu 100)
- 🎯 **KD Alpha**: `0.5 → 0.0` (giảm tuyến tính)

**Tùy chọn**: Layer-wise LR decay (layers trên có LR cao hơn)

**Output**: Model hoàn chỉnh đã được fine-tune

---

## Lịch Trình Knowledge Distillation

```
Stage A: α = 1.0  (không đổi - KD đầy đủ)
Stage B: α = 0.8  (không đổi - KD cao)
Stage C: α = 0.5 → 0.0 (giảm dần - model tự học)
```

**Công thức Loss**:
```python
total_loss = 1.0 * CE_loss + α * KD_loss
```

## Cách Sử Dụng

### 1. Train Từ Đầu (From Scratch)

```python
from train_kaggle import TrainingConfig

config = TrainingConfig(
    is_pretrained=False,  # Từ đầu
    enable_curriculum=True,  # Bật curriculum learning
    num_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)
```

### 2. Fine-tune Model Pretrained

```python
config = TrainingConfig(
    is_pretrained=True,  # Tự động điều chỉnh LR nhỏ hơn
    enable_curriculum=True,
    num_epochs=3,
    # Các tham số khác...
)
```

### 3. Chạy Training

```bash
# Trên Kaggle với 2 GPU
cd S2T
torchrun --nproc_per_node=2 train_kaggle.py
```

## Các Điểm Kỹ Thuật Quan Trọng

### ✅ 1. Rebuild Optimizer Sau Mỗi Stage

**BẮT BUỘC** - không làm optimizer sẽ không update params mới:

```python
# Chuyển từ Stage A → B
setup_stage_b(model, config)  # Mở các layers mới
dist.barrier()  # Sync giữa các GPU
optimizer_b = create_optimizer(model, config, stage="B")  # TẠO MỚI
scheduler_b = create_scheduler(optimizer_b, config, steps_b, warmup_b)
```

### ✅ 2. Distributed Barrier

Luôn gọi `dist.barrier()` sau khi unfreeze params:

```python
if config.world_size > 1:
    dist.barrier()  # Đảm bảo tất cả GPU đồng bộ
```

### ✅ 3. Warmup Mỗi Stage

Mỗi stage cần warmup riêng khi rebuild optimizer:

```python
Stage A: warmup = max(200, 3% stage_a_steps)
Stage B: warmup = max(100, 2% stage_b_steps)  
Stage C: warmup = max(100, 1% stage_c_steps)
```

### ✅ 4. FSDP Compatibility

FSDP wrapping xảy ra TRƯỚC khi tạo optimizer:

```python
# 1. Setup freeze/unfreeze
setup_stage_a(model, config)

# 2. Wrap với FSDP
model = wrap_model_with_fsdp(model, config, model_config, rank)

# 3. Tạo optimizer (nhận FSDP-wrapped params)
optimizer = create_optimizer(model, config, stage="A")
```

## Tham Số Đề Xuất

### From Scratch (Train từ đầu)

```python
encoder_lr = 1e-4
decoder_lr_unfreeze_top = 3e-4
decoder_lr_full = 5e-5
unfreeze_top_k = 4
kd_alpha: 1.0 → 0.8 → (0.5→0.0)
grad_clip = 1.0
```

### Pretrained (Fine-tune)

```python
encoder_lr = 1e-5  # Nhỏ hơn
decoder_lr_unfreeze_top = 5e-5  # Nhỏ hơn
decoder_lr_full = 2e-5  # Nhỏ hơn
unfreeze_top_k = 4
kd_alpha: 1.0 → 0.8 → (0.5→0.0)
grad_clip = 1.0
```

## Monitoring với Wandb

### Metrics Được Track

```python
✅ train/loss - Tổng loss
✅ train/ce_loss - Cross-entropy loss
✅ train/kd_loss - Knowledge distillation loss
✅ train/kd_alpha - KD weight (động)
✅ train/learning_rate - LR hiện tại
✅ train/stage - Stage hiện tại (A/B/C)
✅ train/step_in_stage - Bước trong stage
✅ train/global_step - Tổng số bước
```

### Quan Sát Quan Trọng

1. **Stage A**: CE loss và KD loss cần giảm đều
2. **Chuyển A→B**: Không có spike đột ngột
3. **Stage C**: KD alpha giảm dần về 0
4. **LR**: Warmup mỗi stage rồi decay

## Cấu Trúc Files

```
S2T/
├── train_kaggle.py              # Script training chính (1220 dòng)
├── CURRICULUM_LEARNING.md       # Docs tiếng Anh chi tiết
├── TOM_TAT_CURRICULUM.md        # Tóm tắt tiếng Việt (file này)
├── speech2text_model.py         # Model architecture
├── seamless_m4t_v2_config.py    # Model config
├── utils.py                     # Utilities
└── requirements.txt             # Dependencies
```

## Ví Dụ Cấu Hình

### Dataset Nhỏ (10K samples)

```python
config = TrainingConfig(
    num_train_samples=10000,
    num_epochs=3,
    stage_a_steps=2000,  # Nhỏ hơn
    stage_b_steps=1000,
    per_device_train_batch_size=2,
)
```

### Dataset Lớn (1M samples)

```python
config = TrainingConfig(
    num_train_samples=1000000,
    num_epochs=1,  # 1 epoch đủ
    stage_a_steps=20000,
    stage_b_steps=10000,
    per_device_train_batch_size=4,  # Lớn hơn nếu GPU đủ
)
```

## Troubleshooting

### ❌ Lỗi: Optimizer không update params mới

**Giải pháp**: Rebuild optimizer sau khi unfreeze

```python
setup_stage_b(model, config)
dist.barrier()
optimizer = create_optimizer(model, config, stage="B")  # MỚI
```

### ❌ Lỗi: FSDP gradient sync error

**Giải pháp**: Luôn dùng barrier

```python
setup_stage_c(model, config)
if config.world_size > 1:
    dist.barrier()  # BẮT BUỘC
```

### ❌ Lỗi: Loss không giảm ở Stage B/C

**Giải pháp**:
1. Tăng decoder LR một chút
2. Giảm KD alpha (model tự học nhiều hơn)
3. Thêm steps vào stage hiện tại

### ❌ Lỗi: CUDA OOM

**Giải pháp**:
1. Giảm `per_device_train_batch_size` 
2. Tăng `gradient_accumulation_steps`
3. Bật `cpu_offload=True`
4. Bật `gradient_checkpointing=True`

## Kết Quả Mong Đợi

### Thời Gian Training (Kaggle T4 2x GPU)

| Dataset | Stage A | Stage B | Stage C | Tổng |
|---------|---------|---------|---------|------|
| 10K | ~1h | ~30min | ~1.5h | ~3h |
| 100K | ~10h | ~3h | ~15h | ~28h |

### Output

```
output/
├── checkpoint-5000/     # End of Stage A
├── checkpoint-8000/     # End of Stage B
└── checkpoint-15000/    # Final (End of Stage C)
```

## Tips Quan Trọng

### ✅ DOs

1. **Luôn rebuild optimizer** sau mỗi stage
2. **Luôn dùng dist.barrier()** khi multi-GPU
3. **Monitor wandb** để catch vấn đề sớm
4. **Save checkpoint** ở cuối mỗi stage
5. **Test với dataset nhỏ** trước khi train lớn

### ❌ DON'Ts

1. **Không quên rebuild optimizer** - params mới sẽ không được update
2. **Không skip barrier** - sẽ có race condition
3. **Không set LR quá cao** - unstable training
4. **Không bỏ warmup** - loss spike khi bắt đầu stage
5. **Không train Stage C quá lâu với α=0** - có thể overfit

## Liên Hệ & Support

- **Documentation đầy đủ**: `CURRICULUM_LEARNING.md`
- **Code chính**: `train_kaggle.py`
- **Test script**: `test_setup.py`
- **Kaggle template**: `kaggle_training_notebook.py`

---

**Version**: 1.0  
**Ngày**: 2024  
**Tương thích**: SeamlessM4T v2, PyTorch 2.0+, FSDP, Kaggle 2xGPU

