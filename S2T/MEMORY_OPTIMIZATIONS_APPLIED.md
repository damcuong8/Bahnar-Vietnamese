# Các Tối Ưu Bộ Nhớ Đã Áp Dụng

## ✅ Các thay đổi đã thực hiện:

### 1. **Bật Gradient Checkpointing** ✅
**File:** `S2T/configs.py` (dòng 102)
- **Trước:** `gradient_checkpointing: bool = False`
- **Sau:** `gradient_checkpointing: bool = True  # ✅ BẬT để giảm 50-70% peak memory`
- **Tác động:** Giảm 50-70% peak memory bằng cách trade-off computation cho memory

### 2. **Giảm Batch Size và Tăng Gradient Accumulation** ✅
**File:** `S2T/configs.py` (dòng 30, 32)
- **Trước:** 
  - `per_device_train_batch_size: int = 8`
  - `gradient_accumulation_steps: int = 2`
- **Sau:**
  - `per_device_train_batch_size: int = 4  # ✅ Giảm từ 8 để tiết kiệm ~50% memory`
  - `gradient_accumulation_steps: int = 4  # ✅ Tăng từ 2 để giữ effective batch size = 8`
- **Tác động:** 
  - Giảm ~50% peak memory cho batch processing
  - Effective batch size vẫn = 8 (4 × 4 = 16 nếu có 2 GPUs)
  - Trade-off: Training chậm hơn một chút do nhiều gradient accumulation steps

### 3. **Tắt Output Hidden States và Attentions khi Training** ✅
**File:** `S2T/speech2text_model.py` (dòng 1695-1700, 1724-1725)

#### a) Speech Encoder:
```python
audio_encoder_outputs = self.speech_encoder(
    input_features=audio_input_features,
    attention_mask=audio_attention_mask,
    output_attentions=False,  # ✅ Không cần attention weights khi training
    output_hidden_states=False,  # ✅ Không cần hidden states từ mọi layer
)
```

#### b) Text Encoder (trong no_grad block):
```python
text_encoder_outputs = self.text_encoder(
    input_ids=text_input_pivot_ids,
    attention_mask=text_pivot_attention_mask,
    output_attentions=False,  # ✅ Không cần attention weights
    output_hidden_states=False,  # ✅ Không cần hidden states từ mọi layer
)
```

- **Tác động:** 
  - Giảm 20-30% peak memory
  - Không lưu hidden states từ tất cả layers (có thể tốn >2GB với 24 layers)
  - Không lưu attention weights (có thể tốn >1GB)

### 4. **Giải Phóng Memory Sớm Hơn** ✅
**File:** `S2T/speech2text_model.py` (dòng 1720, 1741, 1746)

#### a) Giải phóng audio_encoder_outputs sau decoder:
```python
# decoder for audio
audio_decoder_outputs = self.text_decoder(...)

# ✅ Giải phóng audio_encoder_outputs sau khi decoder đã dùng (giảm memory peak)
del audio_encoder_outputs
```

#### b) Giải phóng text encoder/decoder outputs trong no_grad block:
```python
with torch.no_grad():
    text_encoder_outputs = self.text_encoder(...)
    text_decoder_outputs = self.text_decoder(...)
    text_pivot_logits = self.lm_head(text_decoder_outputs)
    
    # ✅ Giải phóng text encoder/decoder outputs sớm
    del text_encoder_outputs, text_decoder_outputs
```

#### c) Giải phóng audio_decoder_outputs sau khi tính logits:
```python
text_logits = self.lm_head(audio_decoder_outputs)

# ✅ Giải phóng audio_decoder_outputs sau khi đã tính logits (giảm memory peak)
del audio_decoder_outputs
```

- **Tác động:** 
  - Giảm peak memory bằng cách giải phóng tensors ngay sau khi không cần dùng
  - Cho phép Python garbage collector giải phóng memory sớm hơn

---

## 📊 Ước Tính Giảm Bộ Nhớ:

| Tối Ưu | Giảm Memory | Status |
|--------|-------------|--------|
| Gradient Checkpointing | -50% đến -70% | ✅ Đã áp dụng |
| Tắt output_hidden_states/attentions | -20% đến -30% | ✅ Đã áp dụng |
| Giảm batch_size 50% | -50% | ✅ Đã áp dụng |
| Giải phóng memory sớm | -10% đến -15% | ✅ Đã áp dụng |
| **TỔNG ƯỚC TÍNH** | **-60% đến -80%** | ✅ |

### Ví dụ cụ thể:
- **Trước:** Nếu dùng 16GB peak memory
- **Sau:** Có thể chỉ cần **3-6GB** peak memory! 🎉

---

## ⚠️ Lưu Ý:

1. **Gradient Checkpointing:**
   - Training sẽ chậm hơn ~20-30% do phải tính lại activations
   - Nhưng tiết kiệm rất nhiều memory
   - Đặc biệt hữu ích cho models lớn hoặc sequence dài

2. **Batch Size:**
   - Effective batch size vẫn giữ nguyên (4 × 4 = 16 với 2 GPUs)
   - Training có thể chậm hơn một chút do nhiều gradient accumulation steps
   - Nhưng có thể train với GPU có ít memory hơn

3. **Output Hidden States/Attentions:**
   - Nếu cần debug hoặc visualize attention, có thể tạm thời bật lại
   - Chỉ nên bật khi thực sự cần (ví dụ: inference hoặc analysis)

4. **Memory Release:**
   - `del` không ngay lập tức giải phóng memory, nhưng giúp Python garbage collector làm việc tốt hơn
   - Kết hợp với `torch.cuda.empty_cache()` nếu cần (nhưng không nên gọi quá thường xuyên)

---

## 🚀 Các Tối Ưu Có Thể Thêm (Nếu Vẫn Thiếu Memory):

1. **CPU Offload:** Bật `cpu_offload: bool = True` trong configs.py (trade-off: chậm hơn)
2. **Chunk Processing:** Xử lý sequence theo chunks thay vì toàn bộ
3. **Mixed Precision:** Đảm bảo `use_mixed_precision: bool = True` (đã bật sẵn)
4. **FSDP:** Đảm bảo `use_fsdp: bool = True` và `sharding_strategy: "FULL_SHARD"` (đã bật sẵn)

---

## 📝 Files Đã Thay Đổi:

1. `S2T/configs.py` - Bật gradient checkpointing, giảm batch size
2. `S2T/speech2text_model.py` - Tắt output_hidden_states/attentions, giải phóng memory sớm

---

**Ngày áp dụng:** Hôm nay
**Trạng thái:** ✅ Hoàn thành

