# Phân tích và Tối ưu Bộ nhớ (Memory Optimization)

## Các vấn đề liên quan đến bộ nhớ trong code

### 1. **Gradient Checkpointing bị TẮT** ⚠️ QUAN TRỌNG NHẤT
**Vị trí:** `configs.py:102`
```python
gradient_checkpointing: bool = False  # ❌ Đang TẮT
```

**Vấn đề:** 
- Gradient checkpointing giúp giảm 50-70% bộ nhớ trong quá trình training
- Thay vì lưu tất cả activations, chỉ lưu một số checkpoint và tính lại khi cần

**Giải pháp:** Bật gradient checkpointing:
```python
gradient_checkpointing: bool = True  # ✅ BẬT
```

---

### 2. **Attention Matrices - Tốn nhiều bộ nhớ nhất**
**Vị trí:** Nhiều nơi trong `speech2text_model.py`

#### a) Self-Attention trong Conformer (dòng 201-216)
```python
# Line 201-203: Tạo attention matrix lớn
relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
attn_weights = attn_scores + (relative_position_attn_weights / math.sqrt(self.head_size))
```
**Kích thước:** `(batch_size, num_heads, seq_len, seq_len)` 
- Với batch=8, heads=16, seq_len=1000 → ~2GB chỉ cho attention matrix!

#### b) Chunk Attention Mask (dòng 381-411)
```python
def _apply_chunk_attention(self, attention_mask, hidden_states):
    # Tạo mask lớn cho toàn bộ sequence
    chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)  # Line 407
```
**Vấn đề:** Tạo tensor mask lớn cho toàn bộ sequence length

#### c) Cross-Attention trong Decoder (dòng 939-944)
```python
hidden_states, cross_attn_weights = self.cross_attention(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,  # Lưu cả encoder outputs
    attention_mask=encoder_attention_mask,
)
```
**Vấn đề:** Lưu encoder_hidden_states cho cross-attention

---

### 3. **Output Hidden States & Attentions**
**Vị trí:** `speech2text_model.py` nhiều nơi

#### a) ConformerEncoder (dòng 421-470)
```python
all_hidden_states = [] if output_hidden_states else None  # Line 421
all_self_attentions = [] if output_attentions else None   # Line 422

# Line 445-446: Lưu tất cả hidden states từ mọi layer
if output_hidden_states:
    all_hidden_states.append(hidden_states)
```
**Vấn đề:** 
- Nếu `output_hidden_states=True` → lưu hidden states từ TẤT CẢ layers
- Với 24 layers, mỗi hidden state ~100MB → ~2.4GB chỉ cho hidden states!

#### b) Encoder/Decoder (dòng 1285-1322)
```python
output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
```
**Vấn đề:** Config có thể bật output_hidden_states/attentions mặc định

**Giải pháp:** Đảm bảo luôn set `output_hidden_states=False` và `output_attentions=False` khi training

---

### 4. **Forward Pass - Lưu nhiều outputs**
**Vị trí:** `speech2text_model.py:1671-1746`

#### a) Lưu cả audio và text encoder outputs (dòng 1694-1728)
```python
audio_encoder_outputs = self.speech_encoder(...)  # Line 1694 - Lưu trong memory
# ...
with torch.no_grad():
    text_encoder_outputs = self.text_encoder(...)  # Line 1716 - Lưu trong memory (no_grad nhưng vẫn tốn bộ nhớ)
    text_decoder_outputs = self.text_decoder(...)  # Line 1722
    text_pivot_logits = self.lm_head(text_decoder_outputs)  # Line 1728
```
**Vấn đề:** 
- `audio_encoder_outputs` được giữ trong memory cho đến khi decoder xong
- `text_encoder_outputs` và `text_decoder_outputs` được giữ trong no_grad block
- `text_pivot_logits` được giữ cho đến khi compute loss

#### b) Logits được giữ trong memory (dòng 1730)
```python
text_logits = self.lm_head(audio_decoder_outputs)  # Line 1730
# text_logits được giữ cho đến khi return
```
**Vấn đề:** Logits có kích thước `(batch_size, seq_len, vocab_size)` - rất lớn!

---

### 5. **Intermediate Tensors không được giải phóng**
**Vị trí:** Nhiều nơi

#### a) Position Embeddings (dòng 193-199)
```python
position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
distance = position_ids_r - position_ids_l
positional_embedding = self.distance_embedding(distance + self.left_max_position_embeddings)
```
**Vấn đề:** Tạo nhiều intermediate tensors

#### b) Attention Mask Expansion (dòng 429-432, 1356-1358)
```python
attention_mask = attention_mask.expand(
    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
)  # Line 430-432
```
**Vấn đề:** Expand mask tạo thêm bản copy trong memory

---

### 6. **Batch Processing**
**Vị trí:** Config và training code

**Vấn đề:** 
- `per_device_train_batch_size: int = 8` (dòng 30 trong configs.py)
- Batch size lớn → tăng bộ nhớ theo tuyến tính

**Giải pháp:** 
- Giảm batch size và tăng `gradient_accumulation_steps`
- Ví dụ: batch_size=4, gradient_accumulation_steps=4 → tương đương batch_size=16 nhưng dùng ít bộ nhớ hơn

---

### 7. **Mixed Precision có thể tối ưu hơn**
**Vị trí:** `configs.py:81-82`
```python
use_mixed_precision: bool = True
mixed_precision_dtype: str = "fp16"  # hoặc "bf16"
```
**Tốt:** Đã bật mixed precision (giảm ~50% bộ nhớ)
**Có thể cải thiện:** Đảm bảo tất cả operations đều dùng mixed precision

---

### 8. **FSDP Configuration**
**Vị trí:** `configs.py:78-80`
```python
use_fsdp: bool = True
sharding_strategy: str = "FULL_SHARD"  # ✅ Tốt
cpu_offload: bool = False  # Có thể bật nếu vẫn thiếu bộ nhớ
```
**Có thể cải thiện:** Bật `cpu_offload=True` nếu vẫn thiếu bộ nhớ (trade-off: chậm hơn)

---

## Tổng kết các điểm cần tối ưu theo độ ưu tiên:

### 🔴 **QUAN TRỌNG NHẤT - Phải làm ngay:**
1. **Bật Gradient Checkpointing** (`gradient_checkpointing: bool = True`)
2. **Tắt output_hidden_states và output_attentions** khi training
3. **Giảm batch size** và tăng gradient_accumulation_steps

### 🟡 **QUAN TRỌNG - Nên làm:**
4. **Giải phóng intermediate tensors** sau khi dùng (del, torch.cuda.empty_cache())
5. **Tối ưu attention mask** - không expand nếu không cần
6. **Xem xét CPU offload** nếu vẫn thiếu bộ nhớ

### 🟢 **CẢI THIỆN - Có thể làm:**
7. **Tối ưu positional embeddings** - cache thay vì tính lại
8. **Chunk processing** cho sequence dài
9. **Sử dụng torch.compile()** (PyTorch 2.0+) để tối ưu memory footprint

---

## Ước tính giảm bộ nhớ:

- **Gradient Checkpointing:** -50% đến -70% peak memory
- **Tắt output_hidden_states:** -20% đến -30% peak memory  
- **Giảm batch size 50%:** -50% peak memory (nhưng cần tăng gradient_accumulation_steps)
- **CPU Offload:** -30% đến -40% GPU memory (nhưng chậm hơn)

**Tổng có thể giảm:** 60-80% peak memory với các tối ưu trên!

