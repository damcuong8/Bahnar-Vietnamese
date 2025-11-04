# Các Điểm Nóng Sử Dụng Bộ Nhớ (Memory Hotspots)

## 📍 Các vị trí cụ thể trong code cần tối ưu:

### 1. **Gradient Checkpointing - BẬT NGAY** 🔴
**File:** `S2T/configs.py`
**Dòng:** 102
```python
gradient_checkpointing: bool = False  # ❌ Đổi thành True
```
**Impact:** Giảm 50-70% peak memory

---

### 2. **Attention Matrices - Tốn bộ nhớ nhất**

#### a) Self-Attention Conformer
**File:** `S2T/speech2text_model.py`
**Dòng:** 201-216
```python
# Dòng 201: Einsum tạo matrix lớn (batch, head, seq_len, seq_len)
relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)

# Dòng 203: Matmul tạo attention scores
attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

# Dòng 205: Cộng thêm relative position
attn_weights = attn_scores + (relative_position_attn_weights / math.sqrt(self.head_size))

# Dòng 212: Softmax - tốn thêm memory
attn_weights = torch.softmax(attn_weights, dim=-1)

# Dòng 216: Matmul với value
attn_output = torch.matmul(attn_weights, value)
```
**Kích thước:** `(batch, num_heads, seq_len, seq_len)` 
- Với batch=8, heads=16, seq_len=1000 → **~2GB** chỉ cho attention matrix!

#### b) Chunk Attention Mask
**File:** `S2T/speech2text_model.py`
**Dòng:** 381-411 (method `_apply_chunk_attention`)
```python
# Dòng 404-407: Tạo mask lớn cho toàn bộ sequence
indices = torch.arange(sequence_len, device=hidden_states.device).unsqueeze(0).expand(sequence_len, -1)
chunk_mask = (indices < start_indices) | (indices >= end_indices)
chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)  # Tạo 4D tensor lớn
```

#### c) Attention Mask Expansion
**File:** `S2T/speech2text_model.py`
**Dòng:** 429-432
```python
attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
attention_mask = attention_mask.expand(
    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
)  # Expand tạo thêm bản copy trong memory
```

---

### 3. **Output Hidden States & Attentions - TẮT khi training**

#### a) ConformerEncoder
**File:** `S2T/speech2text_model.py`
**Dòng:** 421-422, 445-446, 465-466, 469-470
```python
# Dòng 421-422: Khởi tạo lists để lưu tất cả hidden states
all_hidden_states = [] if output_hidden_states else None
all_self_attentions = [] if output_attentions else None

# Dòng 445-446: Lưu hidden state từ mỗi layer
if output_hidden_states:
    all_hidden_states.append(hidden_states)  # ❌ Tốn rất nhiều memory!

# Dòng 465-466: Lưu attention weights từ mỗi layer
if output_attentions:
    all_self_attentions.append(layer_outputs[1])  # ❌ Tốn rất nhiều memory!
```

**Vấn đề:** 
- Với 24 layers, mỗi hidden state ~100MB → **~2.4GB** chỉ cho hidden states!
- Attention weights còn tốn hơn nữa!

**Giải pháp:** Đảm bảo `output_hidden_states=False` và `output_attentions=False` khi training

#### b) SpeechEncoder forward call
**File:** `S2T/speech2text_model.py`
**Dòng:** 1198-1204
```python
encoder_outputs = self.encoder(
    hidden_states,
    attention_mask=attention_mask,
    output_attentions=output_attentions,  # ❌ Đảm bảo = False
    output_hidden_states=output_hidden_states,  # ❌ Đảm bảo = False
    return_dict=return_dict,
)
```

#### c) Encoder forward call
**File:** `S2T/speech2text_model.py`
**Dòng:** 1320-1322
```python
output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
output_hidden_states = (
    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)
```
**Vấn đề:** Nếu config có `output_attentions=True` hoặc `output_hidden_states=True` → tốn memory!

---

### 4. **Forward Pass - Lưu nhiều outputs**

#### a) Main forward method
**File:** `S2T/speech2text_model.py`
**Dòng:** 1694-1745
```python
# Dòng 1694: Audio encoder outputs - giữ trong memory
audio_encoder_outputs = self.speech_encoder(
    input_features=audio_input_features,
    attention_mask=audio_attention_mask,
)

# Dòng 1709-1713: Audio decoder sử dụng audio_encoder_outputs
# → audio_encoder_outputs phải giữ trong memory cho đến khi decoder xong
audio_decoder_outputs = self.text_decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=audio_encoder_outputs,  # Giữ reference
    encoder_attention_mask=audio_encoder_attention_mask,
)

# Dòng 1715-1728: Text encoder/decoder trong no_grad block
with torch.no_grad():
    text_encoder_outputs = self.text_encoder(...)  # Vẫn tốn memory dù no_grad
    text_decoder_outputs = self.text_decoder(...)  # Vẫn tốn memory
    text_pivot_logits = self.lm_head(text_decoder_outputs)  # Logits lớn!

# Dòng 1730: Audio logits - kích thước (batch, seq_len, vocab_size)
text_logits = self.lm_head(audio_decoder_outputs)  # ❌ Rất lớn!

# Dòng 1732: Compute loss - giữ cả text_logits và text_pivot_logits
kd_loss, n_valid_tokens = compute_token_kd_loss(text_pivot_logits, text_logits, labels)
```

**Vấn đề:** 
- `audio_encoder_outputs`: ~100-500MB (tùy sequence length)
- `text_encoder_outputs`: ~100-500MB
- `text_pivot_logits`: ~50-200MB (batch × seq_len × vocab_size)
- `text_logits`: ~50-200MB
- **Tổng:** Có thể tốn >1GB chỉ cho các outputs này!

---

### 5. **Intermediate Tensors không được giải phóng**

#### a) Position Embeddings
**File:** `S2T/speech2text_model.py`
**Dòng:** 193-199
```python
position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
distance = position_ids_r - position_ids_l  # Tạo tensor mới
distance = torch.clamp(distance, -self.left_max_position_embeddings, self.right_max_position_embeddings)
positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
```
**Vấn đề:** Tạo nhiều intermediate tensors không được giải phóng ngay

#### b) Attention Mask Operations
**File:** `S2T/speech2text_model.py`
**Dòng:** 1356-1358
```python
# Dòng 1358: Expand mask tạo bản copy
attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
```

---

### 6. **Batch Size Configuration**
**File:** `S2T/configs.py`
**Dòng:** 30-32
```python
per_device_train_batch_size: int = 8  # ❌ Có thể giảm xuống 4
per_device_eval_batch_size: int = 2
gradient_accumulation_steps: int = 2  # ✅ Tăng lên 4 nếu giảm batch_size
```

**Giải pháp:** 
- Giảm `per_device_train_batch_size` từ 8 → 4
- Tăng `gradient_accumulation_steps` từ 2 → 4
- → Hiệu quả training giống nhau nhưng dùng ít memory hơn 50%!

---

### 7. **Mixed Precision & FSDP**
**File:** `S2T/configs.py`
**Dòng:** 81-82, 80
```python
use_mixed_precision: bool = True  # ✅ Đã bật - tốt!
mixed_precision_dtype: str = "fp16"  # ✅ OK
cpu_offload: bool = False  # ⚠️ Có thể bật nếu vẫn thiếu memory
```

---

## 🎯 Tổng kết các thay đổi cần làm NGAY:

### 1. **configs.py - Dòng 102:**
```python
gradient_checkpointing: bool = True  # ✅ BẬT
```

### 2. **configs.py - Dòng 30-32:**
```python
per_device_train_batch_size: int = 4  # Giảm từ 8
gradient_accumulation_steps: int = 4  # Tăng từ 2
```

### 3. **Đảm bảo khi gọi model.forward():**
```python
# Đảm bảo output_hidden_states=False và output_attentions=False
encoder_outputs = self.speech_encoder(
    ...,
    output_attentions=False,  # ✅
    output_hidden_states=False,  # ✅
)
```

### 4. **Nếu vẫn thiếu memory, thêm vào configs.py:**
```python
cpu_offload: bool = True  # Trade-off: chậm hơn nhưng dùng ít GPU memory hơn
```

---

## 📊 Ước tính giảm bộ nhớ sau khi tối ưu:

| Tối ưu | Giảm Memory | Thay đổi |
|--------|-------------|----------|
| Gradient Checkpointing | -50% đến -70% | configs.py:102 |
| Tắt output_hidden_states | -20% đến -30% | Đảm bảo False |
| Giảm batch_size 50% | -50% | configs.py:30-32 |
| CPU Offload | -30% đến -40% | configs.py:80 |
| **TỔNG** | **-60% đến -80%** | |

**Ví dụ:** Nếu hiện tại dùng 16GB → sau tối ưu có thể chỉ cần **3-6GB**! 🎉

