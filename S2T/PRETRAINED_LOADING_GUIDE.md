# Guide: Loading Pretrained Checkpoint

## 📋 Tổng Quan

Code đã được refactor để hỗ trợ load pretrained weights từ HuggingFace hoặc local checkpoint một cách dễ dàng và linh hoạt.

## ✨ Tính Năng Mới

### 1. **Method `load_pretrained_weights()` trong Model**

Model `SeamlessM4Tv2ForSpeechToTextTrain_Pivot` giờ có method tích hợp để load pretrained weights:

```python
stats = model.load_pretrained_weights(
    model_name_or_path="facebook/seamless-m4t-v2-large",
    cache_dir="/kaggle/working/hf_cache"  # Optional
)
```

### 2. **Auto-detect Kaggle Environment**

Code tự động phát hiện môi trường Kaggle và setup cache directory phù hợp.

### 3. **Support Multiple Sources**

- ✅ HuggingFace models (online)
- ✅ Local checkpoint files
- ✅ Training checkpoint format (with 'model' key)

## 🚀 Cách Sử Dụng

### Option 1: Automatic (Recommended)

```python
from train_kaggle import TrainingConfig, create_model

# Config với pretrained
config = TrainingConfig(
    model_name_or_path="facebook/seamless-m4t-v2-large",
    is_pretrained=True,  # ← Bật để load pretrained
)

# Create model - tự động load weights
model, model_config = create_model(config)
```

### Option 2: Manual Loading

```python
from seamless_m4t_v2_config import SeamlessM4Tv2Config
from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot

# Create model
config = SeamlessM4Tv2Config()
model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)

# Load pretrained weights
stats = model.load_pretrained_weights(
    "facebook/seamless-m4t-v2-large",
    cache_dir="./hf_cache"
)

print(f"Loaded! Stats: {stats}")
```

### Option 3: Load from Local Checkpoint

```python
# From local file
stats = model.load_pretrained_weights(
    "/kaggle/input/seamless-checkpoint/pytorch_model.bin"
)

# From directory
stats = model.load_pretrained_weights(
    "/kaggle/input/seamless-checkpoint/"  # Will look for pytorch_model.bin inside
)
```

## 📝 Configuration Options

### TrainingConfig Parameters

```python
@dataclass
class TrainingConfig:
    # Model loading
    model_name_or_path: str = "facebook/seamless-m4t-v2-large"
    is_pretrained: bool = True  # Load pretrained or train from scratch
    hf_cache_dir: Optional[str] = None  # Cache dir (auto-detect if None)
    
    # ... other configs
```

### Custom Cache Directory

```python
config = TrainingConfig(
    model_name_or_path="facebook/seamless-m4t-v2-large",
    is_pretrained=True,
    hf_cache_dir="/my/custom/cache"  # Custom cache location
)
```

## 🎯 Kaggle Specific Setup

### Complete Kaggle Notebook Example

```python
# Cell 1: Setup
import os
os.makedirs("/kaggle/working/hf_cache", exist_ok=True)
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_cache"

# Cell 2: Install dependencies
!pip install transformers accelerate -q

# Cell 3: Load model
from train_kaggle import TrainingConfig, create_model

config = TrainingConfig(
    model_name_or_path="facebook/seamless-m4t-v2-large",
    is_pretrained=True,
    hf_cache_dir="/kaggle/working/hf_cache",
    use_wandb=True,
    wandb_project="my-s2t-training"
)

model, model_config = create_model(config)
model = model.cuda()

print("✓ Model loaded with pretrained weights!")
```

### Using Kaggle Dataset (Faster)

1. **Upload checkpoint to Kaggle Dataset:**
   - Download `facebook/seamless-m4t-v2-large` locally
   - Upload to Kaggle Dataset (private)

2. **Add dataset to notebook:**
   - Add dataset in notebook settings

3. **Use in code:**
   ```python
   config = TrainingConfig(
       model_name_or_path="/kaggle/input/seamless-m4t-v2-large/",
       is_pretrained=True,
   )
   ```

## 🔍 Verification

### Check Weights Loaded

```python
# Check a sample weight
sample_weight = model.speech_encoder.encoder.layers[0].self_attn.linear_q.weight
print(f"Weight mean: {sample_weight.mean():.6f}")
print(f"Weight std: {sample_weight.std():.6f}")

# Random init usually has small mean (~0) and small std (~0.01-0.1)
# Pretrained has larger values with specific patterns
```

### Test Forward Pass

```python
import torch

# Create dummy input
batch_size = 2
audio_features = torch.randn(batch_size, 100, 160).cuda()
text_ids = torch.randint(0, 1000, (batch_size, 20)).cuda()
labels = torch.randint(0, 1000, (batch_size, 20)).cuda()

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(
        audio_input_features=audio_features,
        text_input_pivot_ids=text_ids,
        labels=labels
    )

ce_loss, kd_loss, n_valid, logits, pivot_logits = outputs
print(f"✓ Forward pass successful!")
print(f"  CE Loss: {ce_loss.item():.4f}")
```

## 🛠️ Troubleshooting

### Issue: "transformers not installed"

```bash
pip install transformers accelerate
```

### Issue: Download fails on Kaggle

**Solution**: Enable internet in notebook settings
- Settings → Internet → ON

### Issue: Out of memory during loading

**Solution**: Load on CPU first, then move to GPU

```python
# In create_model(), model is created on CPU by default
model, config = create_model(config)  # CPU
model = model.cuda()  # Then move to GPU
```

### Issue: Missing keys warning

This is **normal** if loading HuggingFace model into custom architecture. Key things:
- `speech_encoder` should have 0 missing keys
- `text_decoder` should have 0 missing keys  
- `text_encoder` should have 0 missing keys

If many keys missing → check model architecture compatibility.

## 📊 Loading Statistics

The `load_pretrained_weights()` method returns statistics:

```python
stats = model.load_pretrained_weights(...)

# For HuggingFace loading:
{
    'speech_encoder': {
        'missing': [],      # List of missing keys
        'unexpected': []    # List of unexpected keys
    },
    'text_encoder': {...},
    'text_decoder': {...}
}

# For local checkpoint:
{
    'missing_keys': [...],
    'unexpected_keys': [...]
}
```

## ⚡ Performance Tips

### 1. Use Kaggle Dataset for Faster Loading

Upload pretrained model to Kaggle Dataset → No download time!

### 2. Cache Management

```python
# Clear cache if needed
import shutil
shutil.rmtree("/kaggle/working/hf_cache", ignore_errors=True)
```

### 3. Mixed Precision

```python
config = TrainingConfig(
    is_pretrained=True,
    use_mixed_precision=True,  # Use FP16
    mixed_precision_dtype="fp16"
)
```

## 📚 References

- **Model Components**:
  - `speech_encoder`: SeamlessM4Tv2SpeechEncoder (Conformer-based)
  - `text_encoder`: SeamlessM4Tv2Encoder (for KD teacher)
  - `text_decoder`: SeamlessM4Tv2Decoder (Transformer decoder)
  - `lm_head`: Linear projection to vocab

- **Config**: `SeamlessM4Tv2Config` in `seamless_m4t_v2_config.py`

- **Original Model**: `facebook/seamless-m4t-v2-large` on HuggingFace

## 🎉 Summary

✅ **Automatic pretrained loading** với `is_pretrained=True`  
✅ **Flexible sources**: HuggingFace or local  
✅ **Kaggle-optimized** với auto-detect cache  
✅ **Error handling** với fallback to random init  
✅ **Detailed logging** để debug dễ dàng  

---

**Version**: 2.0  
**Last Updated**: 2024  
**Compatible**: SeamlessM4T v2, Kaggle, Local training

