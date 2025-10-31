# Chiến lược Fine-tuning cho Bahnar Speech → Vietnamese Text

## Bối cảnh bài toán

**Mô hình:** SeamlessM4T v2 (pretrained, đa ngôn ngữ)  
**Task:** Speech-to-Text từ tiếng Bahnar (low-resource) sang tiếng Việt (high-resource)  
**Thách thức chính:**
- `speech_encoder` chưa biết tiếng Bahnar → Cần học từ đầu
- `text_decoder` đã rất giỏi tiếng Việt → Cần bảo vệ kiến thức hiện có

## Triết lý: "Encoder Nóng, Decoder Lạnh"

### Nguyên tắc cốt lõi
1. **Encoder "Nóng" (High LR):** Để nó nhanh chóng học các đặc trưng âm học của tiếng Bahnar
2. **Decoder "Lạnh" (Low LR):** Để nó chỉ điều chỉnh nhẹ nhàng, tránh "quên" kiến thức tiếng Việt

## Kế hoạch Curriculum Learning 3 Giai đoạn

### 📊 Phân bổ Training Steps

| Giai đoạn | % Steps | Mục tiêu | Các modules trainable |
|-----------|---------|----------|----------------------|
| **Stage A** | **50%** | Encoder học tiếng Bahnar | Speech Encoder only |
| **Stage B** | **25-30%** | Tạo "cầu nối" Encoder-Decoder | Encoder + Top-k Decoder layers (progressive) |
| **Stage C** | **20-25%** | Tinh chỉnh toàn bộ mô hình | Encoder + All Decoder layers |

### 🎯 Stage A: Encoder-only Training (CRITICAL)

**Mục tiêu:** Dạy `speech_encoder` hiểu âm thanh tiếng Bahnar từ đầu

**Cấu hình:**
```python
stage_a_steps: 50% of total_training_steps  # Tăng mạnh so với mặc định 5%
encoder_lr: 1e-4                             # HIGH LR - Encoder học nhanh
```

**Lý do phân bổ nhiều steps:**
- Encoder chưa từng "nghe" tiếng Bahnar → cần thời gian học dài
- Đây là giai đoạn quyết định chất lượng cuối cùng
- Decoder đóng băng hoàn toàn → không lo "catastrophic forgetting"

**Freezing:**
- ✅ `speech_encoder`: **TRAINABLE** (requires_grad=True)
- ❌ `text_encoder`: Frozen (teacher - luôn frozen)
- ❌ `text_decoder`: Frozen
- ❌ `lm_head`: Frozen

---

### 🔗 Stage B: Progressive Top-K Decoder Unfreezing

**Mục tiêu:** Tạo lớp "adapter" giữa encoder mới và decoder cũ

**Cấu hình:**
```python
stage_b_steps: 10000 steps                   # 25-30% of total
unfreeze_top_k: 6                            # Unfreeze 6 lớp trên cùng
decoder_lr_unfreeze_top: 2e-5                # LOW LR - Bảo vệ kiến thức tiếng Việt
encoder_lr: 1e-4                             # Giữ HIGH LR cho encoder
```

**Learning Rate Ratio:**
```
decoder_lr / encoder_lr = 2e-5 / 1e-4 = 1/5 (decoder thấp hơn 5 lần)
```

**Chiến lược Progressive (6 rounds):**
```
Round 0: Unfreeze layer 23, 22  (top 2 layers)
Round 1: Unfreeze layer 21     (add 1 more)
Round 2: Unfreeze layer 20     (add 1 more)
Round 3: Unfreeze layer 19     (add 1 more)
Round 4: Unfreeze layer 18     (add 1 more)
Round 5: Unfreeze layer 17     (total 6 layers)
```

**Optimizer Strategy:**
- Ở mỗi round: `add_param_group()` cho các layers mới unfreeze
- Encoder params: LR = `1e-4`
- Decoder params: LR = `2e-5`

**Freezing:**
- ✅ `speech_encoder`: **TRAINABLE**
- ❌ `text_encoder`: Frozen
- ✅ `text_decoder.layers[18:24]`: **TRAINABLE** (progressive)
- ✅ `lm_head`: **TRAINABLE**
- ❌ `text_decoder.layers[0:18]`: Frozen (bảo vệ kiến thức lõi)

---

### 🎨 Stage C: Full Fine-tuning with Layer-wise LR Decay

**Mục tiêu:** Tinh chỉnh toàn bộ mô hình một cách nhẹ nhàng

**Cấu hình:**
```python
stage_c_steps: Remaining steps (20-25% of total)
decoder_lr_full: 1e-5                        # VERY LOW LR
encoder_lr: 1e-4                             # Giữ HIGH LR
use_layer_wise_lr_decay: True                # QUAN TRỌNG!
layer_wise_lr_decay_rate: 0.9
```

**Learning Rate Ratio:**
```
decoder_lr / encoder_lr = 1e-5 / 1e-4 = 1/10 (decoder thấp hơn 10 lần)
```

**Layer-wise LR Decay (Bảo vệ kiến thức nền tảng):**

Với `decoder_lr_full = 1e-5` và `decay_rate = 0.9`:

```python
Layer 23 (top):    LR = 1e-5 × 0.9^0  = 1.00e-5  # Cao nhất
Layer 22:          LR = 1e-5 × 0.9^1  = 9.00e-6
Layer 21:          LR = 1e-5 × 0.9^2  = 8.10e-6
...
Layer 10 (mid):    LR = 1e-5 × 0.9^13 = 2.54e-6
...
Layer 1:           LR = 1e-5 × 0.9^22 = 1.19e-6
Layer 0 (bottom):  LR = 1e-5 × 0.9^23 = 1.07e-6  # Thấp nhất - bảo vệ tối đa
```

**Lý do Layer-wise LR:**
- Các lớp sâu (layer 0-5) chứa kiến thức ngữ pháp cốt lõi của tiếng Việt
- LR cực thấp ≈ "soft freezing" → bảo vệ khỏi catastrophic forgetting
- Các lớp trên (layer 18-23) đã được "làm quen" ở Stage B → có thể update nhiều hơn

**Optimizer Strategy:**
- `add_param_group()` cho các decoder layers còn lại (0-17)
- Mỗi layer có param group riêng với LR giảm dần

**Freezing:**
- ✅ `speech_encoder`: **TRAINABLE**
- ❌ `text_encoder`: Frozen
- ✅ `text_decoder`: **ALL TRAINABLE** (với layer-wise LR)
- ✅ `lm_head`: **TRAINABLE**

---

## 📈 Tổng quan Learning Rates

### Tóm tắt LR qua các giai đoạn:

| Component | Stage A | Stage B | Stage C |
|-----------|---------|---------|---------|
| **Speech Encoder** | `1e-4` | `1e-4` | `1e-4` |
| **Decoder Top Layers** | Frozen | `2e-5` | `1e-5` (top) → `1e-6` (bottom) |
| **Decoder Deep Layers** | Frozen | Frozen | `1e-6` ~ `5e-7` (layer-wise) |
| **LM Head** | Frozen | `2e-5` | `1e-5` |

### Tỉ lệ LR Encoder/Decoder:

```
Stage B: 1e-4 / 2e-5 = 5:1   (encoder nóng gấp 5 lần)
Stage C: 1e-4 / 1e-5 = 10:1  (encoder nóng gấp 10 lần)
```

---

## 🔧 Implementation Details

### Optimizer Strategy: Tạo một lần, thêm dần parameter groups

```python
# Stage A: Tạo optimizer với encoder params
optimizer = create_optimizer(model, config, stage="A")
scheduler = create_cosine_scheduler(optimizer, config, total_training_steps)

# Stage B: Add decoder params progressively (6 rounds)
for round_idx in range(6):
    setup_stage_b(model, config, round_idx=round_idx)
    add_new_param_groups_to_optimizer(optimizer, model, config, stage="B", round_idx)
    train_stage(...)

# Stage C: Add remaining decoder params with layer-wise LR
setup_stage_c(model, config)
add_new_param_groups_to_optimizer(optimizer, model, config, stage="C")
train_stage(...)
```

### Knowledge Distillation (KD) Schedule

KD weight giảm dần qua các giai đoạn:

```python
Stage A: alpha_kd = 1.0    # Maximize KD từ text_encoder
Stage B: alpha_kd = 0.8    # Giảm nhẹ
Stage C: alpha_kd = 0.5 → 0.0  # Linear decay → cuối cùng chỉ dùng CE loss
```

---

## 🎓 Lý thuyết đằng sau chiến lược

### 1. Tại sao Encoder cần LR cao?
- Encoder đang học một nhiệm vụ hoàn toàn mới (acoustic features của tiếng Bahnar)
- Không có pretrained knowledge nào về tiếng Bahnar trong model
- Cần gradient updates mạnh để converge nhanh

### 2. Tại sao Decoder cần LR thấp?
- Decoder đã có kiến thức sâu về ngữ pháp và từ vựng tiếng Việt
- LR cao → risk of catastrophic forgetting
- Chỉ cần điều chỉnh nhẹ để "hiểu" output từ encoder mới

### 3. Tại sao Stage A chiếm 50% steps?
- Research cho thấy: trong cross-lingual adaptation, encoder learning là bottleneck
- Nếu encoder không học đủ → toàn bộ pipeline sẽ fail
- Decoder fine-tuning tương đối nhanh (đã có nền tảng tốt)

### 4. Tại sao Layer-wise LR ở Stage C?
- Inspired by "Discriminative Fine-Tuning" (Howard & Ruder, 2018)
- Các lớp sâu hơn → knowledge trừu tượng hơn → cần bảo vệ nhiều hơn
- Các lớp nông hơn → task-specific → có thể update nhiều hơn

---

## 🚀 Khuyến nghị thực hành

### Cho low-resource speech → high-resource text:

1. **Ưu tiên Stage A:**
   - Phân bổ 40-60% tổng steps cho Stage A
   - Có thể tăng `encoder_lr` lên `2e-4` nếu training chậm

2. **Theo dõi metrics:**
   - Stage A: Focus on KD loss (encoder-text_encoder alignment)
   - Stage B/C: Focus on CE loss (actual transcription quality)

3. **Điều chỉnh nếu cần:**
   - Nếu decoder "quên" tiếng Việt → giảm `decoder_lr` thêm 50%
   - Nếu encoder không converge → tăng `stage_a_steps`

4. **Pretrained model:**
   - Config tự động scale LR xuống khi `is_pretrained=True`
   - Encoder: `1e-4` → `5e-5`
   - Decoder: giữ nguyên tỉ lệ 5x/10x thấp hơn

---

## 📊 Expected Training Timeline (ví dụ)

Giả sử: 30,000 total steps

```
Stage A: 15,000 steps (50%)
├─ Warmup: 450 steps (3%)
└─ Training: 14,550 steps

Stage B: 10,000 steps (33%)
├─ Round 0: ~1,667 steps
├─ Round 1: ~1,667 steps
├─ Round 2: ~1,667 steps
├─ Round 3: ~1,667 steps
├─ Round 4: ~1,666 steps
└─ Round 5: ~1,666 steps

Stage C: 5,000 steps (17%)
├─ Warmup: 50 steps (1%)
└─ Training: 4,950 steps
```

---

## 📝 Summary

**Key Principles:**
1. ✅ Encoder "Nóng" (1e-4) - học tiếng Bahnar từ đầu
2. ✅ Decoder "Lạnh" (2e-5 → 1e-5) - bảo vệ kiến thức tiếng Việt
3. ✅ Progressive unfreezing - tránh shock
4. ✅ Layer-wise LR decay - bảo vệ kiến thức nền tảng
5. ✅ 50% steps cho Stage A - ưu tiên encoder learning

**Implementation:**
- ✅ Tạo optimizer một lần
- ✅ `add_param_group()` cho các stages sau
- ✅ Maintain LR ratio 5x-10x giữa encoder và decoder

Chiến lược này được thiết kế đặc biệt cho bài toán **low-resource speech + high-resource text**, tối ưu hóa việc học acoustic features mới trong khi bảo toàn knowledge về ngôn ngữ đích.

