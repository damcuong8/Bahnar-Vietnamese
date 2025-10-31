"""
Kaggle Notebook Template for SeamlessM4T v2 Fine-tuning with FSDP
Copy this code into a Kaggle notebook with 2x GPU enabled
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================

# Install required packages
!pip install -q transformers accelerate wandb
!pip install -q sentencepiece protobuf

# Verify GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ============================================================================
# CELL 2: Setup Wandb (Optional)
# ============================================================================

import wandb

# Login to wandb
# Get your API key from https://wandb.ai/settings
!wandb login

# Or login programmatically
# wandb.login(key="YOUR_API_KEY_HERE")


# ============================================================================
# CELL 3: Clone or Upload Your Code
# ============================================================================

# Option 1: Clone from repository
# !git clone https://github.com/your-repo/speech_to_text.git
# %cd speech_to_text/S2T

# Option 2: If you uploaded files as a dataset, copy them
# !cp -r /kaggle/input/your-dataset/* /kaggle/working/
# %cd /kaggle/working/S2T

# For this example, we assume files are in the current directory
import os
os.makedirs("S2T", exist_ok=True)
%cd S2T


# ============================================================================
# CELL 4: Setup Environment Variables for Distributed Training
# ============================================================================

import os

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '2'
os.environ['RANK'] = '0'

# Optional: Set CUDA device order
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


# ============================================================================
# CELL 5: Configuration (Modify as needed)
# ============================================================================

# Create a configuration file
config_code = '''
from dataclasses import dataclass
from typing import Optional

@dataclass
class KaggleTrainingConfig:
    """Training configuration optimized for Kaggle"""
    
    # Model
    model_name_or_path: str = "facebook/seamless-m4t-v2-large"
    
    # Training - Optimized for Kaggle P100
    num_epochs: int = 3
    per_device_train_batch_size: int = 2  # Adjust based on GPU memory
    gradient_accumulation_steps: int = 4   # Effective batch size = 2*4*2 = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data
    max_audio_length: int = 30
    max_text_length: int = 200
    num_train_samples: int = 1000  # Replace with actual dataset size
    
    # FSDP
    use_fsdp: bool = True
    sharding_strategy: str = "FULL_SHARD"
    cpu_offload: bool = False  # Set to True if OOM
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"
    
    # Checkpointing
    output_dir: str = "/kaggle/working/output"
    save_steps: int = 500
    save_total_limit: int = 2  # Keep only 2 checkpoints on Kaggle
    
    # Logging
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "seamlessm4t-v2-kaggle"
    wandb_run_name: str = "fsdp-2gpu-training"
    
    # Optimization
    gradient_checkpointing: bool = True
    
    # Random seed
    seed: int = 42
'''

with open("kaggle_config.py", "w") as f:
    f.write(config_code)

print("✅ Configuration saved to kaggle_config.py")


# ============================================================================
# CELL 6: Upload Your Model Files
# ============================================================================

# Make sure you have these files in your Kaggle dataset or working directory:
required_files = [
    "speech2text_model.py",
    "seamless_m4t_v2_config.py",
    "utils.py",
    "seamless_feature_extractor.py",  # If needed
]

# Check if files exist
import os
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file} found")
    else:
        print(f"❌ {file} NOT found - Please upload this file!")


# ============================================================================
# CELL 7: Quick Test - Single GPU (Optional)
# ============================================================================

# Test if the model loads correctly before distributed training
print("Testing model loading...")

try:
    from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
    from seamless_m4t_v2_config import SeamlessM4Tv2Config
    
    config = SeamlessM4Tv2Config()
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
    model = model.cuda()
    
    print(f"✅ Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass with dummy data
    dummy_audio = torch.randn(1, 100, 160).cuda()
    dummy_text = torch.randint(4, 1000, (1, 20)).cuda()
    dummy_labels = torch.randint(4, 1000, (1, 20)).cuda()
    
    with torch.no_grad():
        outputs = model(
            audio_input_features=dummy_audio,
            text_input_pivot_ids=dummy_text,
            labels=dummy_labels,
        )
    
    print(f"✅ Forward pass successful!")
    print(f"Output: {len(outputs)} elements")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# CELL 8: Launch Distributed Training
# ============================================================================

# Launch training with torchrun
!torchrun \
    --nproc_per_node=2 \
    --master_port=12355 \
    train_kaggle.py


# ============================================================================
# CELL 9: Monitor Training (Run in separate cell while training)
# ============================================================================

# Check GPU utilization
!nvidia-smi

# View latest logs
!tail -n 50 /kaggle/working/output/train.log  # If you redirect logs to file


# ============================================================================
# CELL 10: Check Checkpoints
# ============================================================================

import os
import glob

# List all checkpoints
checkpoints = sorted(glob.glob("/kaggle/working/output/checkpoint-*"))
print(f"Found {len(checkpoints)} checkpoints:")
for ckpt in checkpoints:
    size_mb = sum(os.path.getsize(os.path.join(ckpt, f)) 
                  for f in os.listdir(ckpt)) / 1024 / 1024
    print(f"  {ckpt}: {size_mb:.2f} MB")


# ============================================================================
# CELL 11: Load and Test Checkpoint
# ============================================================================

import torch
from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
from seamless_m4t_v2_config import SeamlessM4Tv2Config

# Load checkpoint
checkpoint_path = "/kaggle/working/output/checkpoint-1000/pytorch_model.bin"

config = SeamlessM4Tv2Config()
model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint['model'])
model = model.cuda()
model.eval()

print("✅ Checkpoint loaded successfully!")
print(f"Trained for {checkpoint['epoch']} epochs, {checkpoint['step']} steps")


# ============================================================================
# CELL 12: Save Final Model to Kaggle Output
# ============================================================================

# Save the final model in a format that can be downloaded
import torch
import os

def save_final_model(checkpoint_path, output_path):
    """Save model weights for easy distribution"""
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Save model weights
    model_path = os.path.join(output_path, "final_model.bin")
    torch.save(checkpoint['model'], model_path)
    
    # Save training state
    training_state_path = os.path.join(output_path, "training_state.bin")
    torch.save({
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'optimizer': checkpoint['optimizer'],
        'scheduler': checkpoint['scheduler'],
    }, training_state_path)
    
    print(f"✅ Saved final model to {model_path}")
    print(f"✅ Saved training state to {training_state_path}")

# Find the latest checkpoint
import glob
checkpoints = sorted(glob.glob("/kaggle/working/output/checkpoint-*"))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    save_final_model(
        os.path.join(latest_checkpoint, "pytorch_model.bin"),
        "/kaggle/working/final_model"
    )
else:
    print("❌ No checkpoints found!")


# ============================================================================
# CELL 13: Upload to Hugging Face Hub (Optional)
# ============================================================================

from huggingface_hub import HfApi, create_repo

# Login to Hugging Face
!huggingface-cli login

# Upload model
api = HfApi()
repo_name = "your-username/seamlessm4t-v2-finetuned"

try:
    create_repo(repo_id=repo_name, exist_ok=True)
    
    api.upload_folder(
        folder_path="/kaggle/working/final_model",
        repo_id=repo_name,
        repo_type="model",
    )
    
    print(f"✅ Model uploaded to https://huggingface.co/{repo_name}")
except Exception as e:
    print(f"❌ Upload failed: {e}")


# ============================================================================
# CELL 14: Clean Up
# ============================================================================

# Clean up to free space
import shutil
import os

def cleanup(keep_final=True):
    """Clean up intermediate files"""
    
    # Remove all checkpoints except the latest
    if keep_final:
        checkpoints = sorted(glob.glob("/kaggle/working/output/checkpoint-*"))
        for ckpt in checkpoints[:-1]:  # Keep only the last one
            print(f"Removing {ckpt}")
            shutil.rmtree(ckpt)
    
    # Clear CUDA cache
    import torch
    torch.cuda.empty_cache()
    
    print("✅ Cleanup complete!")

# cleanup(keep_final=True)


# ============================================================================
# CELL 15: Download Results (Kaggle UI)
# ============================================================================

# After training completes, download:
# 1. /kaggle/working/final_model/final_model.bin
# 2. /kaggle/working/final_model/training_state.bin
# 3. Any other outputs you need

print("""
Training complete! 🎉

To download your model:
1. Go to the right sidebar in Kaggle
2. Click on 'Output' 
3. Download the files from /kaggle/working/final_model/

Or commit the notebook to save outputs to a dataset.
""")


# ============================================================================
# ADDITIONAL TIPS
# ============================================================================

"""
KAGGLE-SPECIFIC TIPS:

1. GPU Quota:
   - Kaggle provides ~30 hours/week of GPU time
   - Enable 'GPU' in notebook settings (right sidebar)
   - Choose '2x T4 GPU' or available option

2. Session Limits:
   - Notebooks timeout after 9 hours idle
   - Save checkpoints frequently!
   - Use wandb to monitor remotely

3. Data Storage:
   - /kaggle/input - Read-only input data
   - /kaggle/working - Your working directory (max 20GB)
   - Output files are saved here

4. Internet Access:
   - Enable 'Internet' in notebook settings
   - Required for downloading models and wandb

5. Optimization for Kaggle P100:
   - Use FP16 mixed precision
   - Batch size 2-4 per GPU
   - Enable gradient checkpointing
   - Set save_total_limit=2 to save space

6. Common Issues:
   - OOM Error: Reduce batch size or enable CPU offload
   - NCCL Error: Restart notebook and try again
   - Slow Loading: Use Kaggle datasets for large files

7. Best Practices:
   - Save checkpoints every 30 minutes
   - Monitor with wandb (can close notebook)
   - Commit notebook to save outputs
   - Use Kaggle datasets for data/models

EXAMPLE FULL COMMAND:

!torchrun --nproc_per_node=2 train_kaggle.py \
    --num_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --output_dir /kaggle/working/output \
    --use_wandb True \
    --wandb_project my-project

(Note: Requires adding argparse to train_kaggle.py)
"""

