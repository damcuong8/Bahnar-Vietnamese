# Quick Start Guide: SeamlessM4T v2 Fine-tuning on Kaggle

Get started with fine-tuning SeamlessM4T v2 using FSDP in under 5 minutes!

## Prerequisites

- Kaggle account with GPU access
- Basic knowledge of Python and PyTorch
- (Optional) Wandb account for monitoring

## Step-by-Step Guide

### 1. Create a New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. In the right sidebar:
   - **Accelerator**: Select "GPU" → "2 x T4" (or available 2-GPU option)
   - **Internet**: Enable (for downloading packages)
   - **Persistence**: Enable if you want to save progress

### 2. Upload Your Code

**Option A: From Git Repository**
```python
!git clone https://github.com/your-repo/speech_to_text.git
%cd speech_to_text/S2T
```

**Option B: Upload as Kaggle Dataset**
1. Zip your S2T folder
2. Create a new Kaggle dataset
3. Upload the zip file
4. In your notebook:
```python
!cp -r /kaggle/input/your-dataset/* /kaggle/working/
%cd /kaggle/working/S2T
```

**Option C: Copy Files Directly**
Use the file upload feature in Kaggle notebook (for small files).

### 3. Install Dependencies

```python
!pip install -q torch transformers accelerate wandb
```

### 4. (Optional) Setup Wandb

```python
import wandb
!wandb login
# Paste your API key from https://wandb.ai/settings
```

### 5. Verify GPU Setup

```python
import torch
print(f"GPUs available: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
if torch.cuda.device_count() > 1:
    print(f"GPU 1: {torch.cuda.get_device_name(1)}")
```

Expected output:
```
GPUs available: 2
GPU 0: Tesla T4
GPU 1: Tesla T4
```

### 6. Launch Training

**For 2 GPUs (Recommended):**
```python
!torchrun --nproc_per_node=2 train_kaggle.py
```

**For Single GPU (Testing):**
```python
!python train_kaggle.py
```

### 7. Monitor Progress

**In Notebook:**
```python
# Check GPU usage
!nvidia-smi

# View logs (if redirected)
!tail -f /kaggle/working/output/train.log
```

**With Wandb:**
Visit your dashboard at: `https://wandb.ai/your-username/seamlessm4t-v2-finetuning`

### 8. Access Checkpoints

After training starts, checkpoints will be saved to `/kaggle/working/output/checkpoint-*/`

```python
import os
import glob

checkpoints = sorted(glob.glob("/kaggle/working/output/checkpoint-*"))
print(f"Found {len(checkpoints)} checkpoints:")
for ckpt in checkpoints:
    print(f"  {ckpt}")
```

## Common Issues & Solutions

### Issue: "CUDA Out of Memory"

**Solution 1:** Reduce batch size in `train_kaggle.py`:
```python
per_device_train_batch_size: int = 1  # Was 2
```

**Solution 2:** Enable CPU offload:
```python
cpu_offload: bool = True  # Was False
```

**Solution 3:** Increase gradient accumulation:
```python
gradient_accumulation_steps: int = 8  # Was 4
```

### Issue: "NCCL initialization failed"

**Solution:** Restart the notebook and try again. Sometimes NCCL needs a clean start.

### Issue: "Wandb not logging"

**Solution 1:** Make sure you're logged in:
```python
!wandb login
```

**Solution 2:** Check if wandb is enabled:
```python
# In train_kaggle.py TrainingConfig:
use_wandb: bool = True
```

### Issue: "Training too slow"

**Tip 1:** Use mixed precision (should be enabled by default):
```python
use_mixed_precision: bool = True
mixed_precision_dtype: str = "fp16"
```

**Tip 2:** Reduce logging frequency:
```python
logging_steps: int = 50  # Was 10
```

**Tip 3:** Use gradient checkpointing:
```python
gradient_checkpointing: bool = True
```

### Issue: "Not enough disk space"

**Solution:** Reduce checkpoint frequency and limit:
```python
save_steps: int = 1000  # Was 500
save_total_limit: int = 1  # Was 3
```

## Performance Tips

### For Faster Training:
1. ✅ Use FP16 mixed precision
2. ✅ Enable gradient checkpointing
3. ✅ Use 2 GPUs with FSDP
4. ✅ Increase batch size (if memory allows)

### For Memory Optimization:
1. ✅ Enable CPU offload for parameters
2. ✅ Reduce batch size
3. ✅ Use FULL_SHARD strategy
4. ✅ Limit sequence lengths

### For Better Results:
1. ✅ Use warmup steps (500-1000)
2. ✅ Monitor with wandb
3. ✅ Save checkpoints frequently
4. ✅ Use learning rate scheduling

## Expected Training Time

| Configuration | Samples/Epoch | Time/Epoch | Total Time (3 epochs) |
|--------------|---------------|------------|----------------------|
| 1K samples, BS=2 | 1000 | ~15 min | ~45 min |
| 10K samples, BS=2 | 10000 | ~2.5 hrs | ~7.5 hrs |
| 100K samples, BS=2 | 100000 | ~25 hrs | ~75 hrs |

*Times are approximate for Kaggle T4 GPUs

## Next Steps

1. ✅ **Replace dummy data** with your actual dataset
2. ✅ **Customize hyperparameters** in `TrainingConfig`
3. ✅ **Monitor training** with wandb
4. ✅ **Evaluate** your model on validation set
5. ✅ **Share** your fine-tuned model on Hugging Face Hub

## Example: Full Training Session

```python
# Cell 1: Setup
!pip install -q torch transformers accelerate wandb
import torch
print(f"GPUs: {torch.cuda.device_count()}")

# Cell 2: Get Code
!git clone https://github.com/your-repo/speech_to_text.git
%cd speech_to_text/S2T

# Cell 3: Login to Wandb
import wandb
!wandb login

# Cell 4: Launch Training
!torchrun --nproc_per_node=2 train_kaggle.py

# Cell 5: Check Progress (run separately while training)
!nvidia-smi
!ls -lh /kaggle/working/output/checkpoint-*

# Cell 6: After Training - Download Results
!zip -r results.zip /kaggle/working/output/
# Download via Kaggle UI
```

## Resource Links

- 📚 [Full Documentation](README_TRAINING.md)
- 🎯 [Kaggle Notebook Template](kaggle_training_notebook.py)
- 🛠️ [Utility Functions](train_utils.py)
- 💬 [Troubleshooting Guide](README_TRAINING.md#troubleshooting)

## Support

If you encounter issues:
1. Check the [Troubleshooting section](README_TRAINING.md#troubleshooting)
2. Review [Kaggle-specific tips](kaggle_training_notebook.py)
3. Check wandb logs for detailed metrics
4. Verify GPU memory with `nvidia-smi`

Happy Training! 🚀

