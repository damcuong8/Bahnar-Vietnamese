# Sửa Lỗi Gradient Checkpointing Không Được Gọi

## 🔴 VẤN ĐỀ PHÁT HIỆN:

1. **Gradient checkpointing được enable TRƯỚC khi wrap FSDP**
   - File: `model_utils.py` - `create_model()` enable ở dòng 85-86
   - File: `train_kaggle.py` - Model được wrap FSDP ở dòng 191
   - **Vấn đề:** FSDP wrap tạo model wrapper mới → gradient checkpointing bị mất!

2. **Không có re-enable sau khi wrap FSDP**
   - Code cũ không enable lại sau khi wrap FSDP
   - Gradient checkpointing chỉ hoạt động nếu KHÔNG dùng FSDP

3. **Code flow không đúng:**
   ```
   create_model() 
   → enable gradient checkpointing (dòng 86)
   → wrap FSDP (dòng 191) 
   → ❌ Gradient checkpointing BỊ MẤT!
   ```

## ✅ GIẢI PHÁP ĐÃ ÁP DỤNG:

### 1. **Sửa `create_model()` - KHÔNG enable ở đây nữa**

**File:** `S2T/model_utils.py` (dòng 84-86)

**Trước:**
```python
# Enable gradient checkpointing if specified
if config.gradient_checkpointing:
    _enable_gradient_checkpointing(model)
```

**Sau:**
```python
# ⚠️ KHÔNG enable gradient checkpointing ở đây nếu sẽ wrap FSDP
# Vì FSDP wrap tạo model mới, gradient checkpointing sẽ bị mất
# Sẽ enable sau khi wrap FSDP trong wrap_model_with_fsdp()
# Nếu không dùng FSDP, sẽ enable trong wrap_model_with_fsdp()
```

### 2. **Thêm re-enable trong `wrap_model_with_fsdp()`**

**File:** `S2T/model_utils.py` (dòng 178-183, 225-229)

**Thêm:**
```python
if not config.use_fsdp or config.world_size == 1:
    logger.info("FSDP not enabled or single GPU training, returning unwrapped model")
    # ✅ Đảm bảo gradient checkpointing được enable ngay cả khi không dùng FSDP
    if config.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    return model

# ... wrap FSDP ...

# ✅ QUAN TRỌNG: Re-enable gradient checkpointing SAU KHI wrap FSDP
# Vì FSDP wrap tạo model mới, gradient checkpointing có thể bị mất
if config.gradient_checkpointing:
    logger.info("Re-enabling gradient checkpointing after FSDP wrap...")
    _enable_gradient_checkpointing_after_fsdp(model, config)
```

### 3. **Thêm function `_enable_gradient_checkpointing_after_fsdp()`**

**File:** `S2T/model_utils.py` (dòng 123-139)

```python
def _enable_gradient_checkpointing_after_fsdp(model: nn.Module, config: TrainingConfig):
    """
    Enable gradient checkpointing after FSDP wrap.
    FSDP wraps model, cần access qua .module attribute.
    """
    try:
        # FSDP wraps model, cần access qua .module
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        _enable_gradient_checkpointing(actual_model)
        logger.info("✓ Gradient checkpointing re-enabled after FSDP wrap")
    except Exception as e:
        logger.warning(f"⚠️ Could not re-enable gradient checkpointing after FSDP: {e}")
        logger.warning("Gradient checkpointing may not be active. Check manually.")
```

## ✅ CODE FLOW MỚI (ĐÚNG):

```
create_model() 
  → Tạo model
  → ❌ KHÔNG enable gradient checkpointing (sẽ enable sau)
  → Return model

wrap_model_with_fsdp()
  → Nếu KHÔNG dùng FSDP:
     → ✅ Enable gradient checkpointing ngay
  → Nếu dùng FSDP:
     → Wrap với FSDP
     → ✅ Re-enable gradient checkpointing sau khi wrap
```

## ✅ KẾT QUẢ:

1. **Gradient checkpointing được enable ĐÚNG THỜI ĐIỂM:**
   - Sau khi wrap FSDP (nếu dùng FSDP)
   - Ngay trong wrap_model_with_fsdp (nếu không dùng FSDP)

2. **Không bị mất sau khi wrap FSDP:**
   - Re-enable sau khi wrap
   - Access đúng model qua `.module` attribute

3. **Logging rõ ràng:**
   - Log khi enable
   - Log khi re-enable sau FSDP
   - Warning nếu có lỗi

## 📝 LƯU Ý:

- FSDP wrap model trong wrapper, nên cần access qua `.module`
- Gradient checkpointing phải enable SAU khi wrap FSDP
- Nếu không dùng FSDP, vẫn enable trong cùng function để đảm bảo consistency

## ✅ STATUS:

- ✅ Đã sửa code flow
- ✅ Đã thêm re-enable sau FSDP wrap
- ✅ Đã thêm error handling
- ✅ Đã thêm logging

**Gradient checkpointing giờ sẽ được gọi và thực thi ĐÚNG CÁCH!**

