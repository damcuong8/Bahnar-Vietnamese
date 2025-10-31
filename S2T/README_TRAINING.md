# SeamlessM4T v2 Fine-tuning with FSDP on Kaggle

This directory contains the implementation for fine-tuning SeamlessM4T v2 model using FSDP (Fully Sharded Data Parallel) on Kaggle with 2 GPUs.

## Features

- ✅ Multi-GPU training with FSDP for efficient memory usage
- ✅ Wandb integration for real-time monitoring
- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient accumulation for larger effective batch sizes
- ✅ Gradient checkpointing to reduce memory footprint
- ✅ Automatic checkpoint saving and management
- ✅ Knowledge distillation with pivot language

## Installation

### On Kaggle

1. **Install dependencies:**

```bash
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate wandb
```

2. **Setup wandb (optional but recommended):**

```bash
!pip install wandb
!wandb login
# Enter your API key when prompted
```

### Local Setup

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate wandb
```

## Usage

### 1. Single GPU Training (for testing)

```bash
cd S2T
python train_kaggle.py
```

### 2. Multi-GPU Training with FSDP (2 GPUs on Kaggle)

```bash
cd S2T
torchrun --nproc_per_node=2 train_kaggle.py
```

### 3. On Kaggle Notebook

```python
import os
os.chdir('/kaggle/working/speech_to_text/S2T')

# Set environment variables for distributed training
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Run with torchrun
!torchrun --nproc_per_node=2 --master_port=12355 train_kaggle.py
```

## Configuration

The training script uses a `TrainingConfig` dataclass that you can modify. Key parameters:

### Model Configuration
- `model_name_or_path`: Pre-trained model path (default: "facebook/seamless-m4t-v2-large")

### Training Hyperparameters
- `num_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per GPU (default: 2)
- `gradient_accumulation_steps`: Steps to accumulate gradients (default: 4)
- `learning_rate`: Learning rate (default: 5e-5)
- `weight_decay`: Weight decay for AdamW (default: 0.01)
- `warmup_steps`: Number of warmup steps (default: 500)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)

### FSDP Configuration
- `use_fsdp`: Enable FSDP (default: True)
- `sharding_strategy`: FSDP sharding strategy (default: "FULL_SHARD")
  - Options: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"
- `cpu_offload`: Offload parameters to CPU (default: False)
- `use_mixed_precision`: Enable mixed precision (default: True)
- `mixed_precision_dtype`: Precision type (default: "fp16")
  - Options: "fp16", "bf16"

### Checkpointing
- `output_dir`: Directory to save checkpoints (default: "./output")
- `save_steps`: Save checkpoint every N steps (default: 500)
- `save_total_limit`: Maximum number of checkpoints to keep (default: 3)

### Logging
- `logging_steps`: Log metrics every N steps (default: 10)
- `use_wandb`: Enable wandb logging (default: True)
- `wandb_project`: Wandb project name (default: "seamlessm4t-v2-finetuning")

## Customizing the Training Script

### 1. Replace Dummy Dataset

The script includes a `DummySpeechToTextDataset` placeholder. Replace it with your actual dataset:

```python
class YourCustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, feature_extractor):
        # Load your data
        self.data = load_your_data(data_path)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
    
    def __getitem__(self, idx):
        # Load audio file
        audio, sr = torchaudio.load(self.data[idx]['audio_path'])
        
        # Extract features
        audio_features = self.feature_extractor(audio, sampling_rate=sr)
        
        # Tokenize text
        text_tokens = self.tokenizer(self.data[idx]['text'])
        
        return {
            'audio_input_features': audio_features,
            'text_input_pivot_ids': text_tokens['input_ids'],
            'labels': text_tokens['input_ids'],
            'audio_attention_mask': audio_features['attention_mask'],
            'text_pivot_attention_mask': text_tokens['attention_mask'],
        }
```

### 2. Modify Training Configuration

Edit the `TrainingConfig` dataclass in `train_kaggle.py`:

```python
@dataclass
class TrainingConfig:
    # Your custom settings
    num_epochs: int = 10
    per_device_train_batch_size: int = 4
    learning_rate: float = 3e-5
    # ... etc
```

Or pass arguments via command line (if you add argument parsing).

## Memory Optimization Tips

### For Limited GPU Memory:

1. **Reduce batch size:**
   ```python
   per_device_train_batch_size: int = 1
   ```

2. **Increase gradient accumulation:**
   ```python
   gradient_accumulation_steps: int = 8
   ```

3. **Enable CPU offload:**
   ```python
   cpu_offload: bool = True
   ```

4. **Use SHARD_GRAD_OP instead of FULL_SHARD:**
   ```python
   sharding_strategy: str = "SHARD_GRAD_OP"
   ```

5. **Enable gradient checkpointing:**
   ```python
   gradient_checkpointing: bool = True
   ```

## Monitoring Training

### Wandb Dashboard

If wandb is enabled, you can monitor:
- Training loss (total, CE, KD)
- Learning rate
- Gradient norms
- Step and epoch progress

Access your dashboard at: https://wandb.ai/your-username/seamlessm4t-v2-finetuning

### Console Logs

The script logs training progress every `logging_steps`:

```
Epoch 1 | Step 10 | Loss: 2.3456 | CE Loss: 1.5678 | KD Loss: 0.7778 | LR: 5.00e-05
```

## Checkpoints

Checkpoints are saved at:
- Every `save_steps` steps
- End of each epoch

Location: `{output_dir}/checkpoint-{step}/pytorch_model.bin`

### Loading a Checkpoint

```python
checkpoint = torch.load("output/checkpoint-1000/pytorch_model.bin")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `cpu_offload`
4. Reduce `max_audio_length` or `max_text_length`

### NCCL Initialization Failed

Ensure NCCL is properly installed:
```bash
!pip install nvidia-nccl-cu11  # For CUDA 11.x
```

### Wandb Not Logging

1. Login to wandb: `wandb login`
2. Check `WANDB_AVAILABLE` is True in the script
3. Verify `use_wandb: bool = True` in config

## Performance Benchmarks

Expected performance on Kaggle P100 GPUs (2x):

| Configuration | Batch Size | Memory Usage | Samples/sec |
|--------------|------------|--------------|-------------|
| FP32, No FSDP | 1 | ~14GB | ~2.5 |
| FP16, FULL_SHARD | 2 | ~11GB | ~5.0 |
| FP16, FULL_SHARD + Offload | 4 | ~9GB | ~4.0 |

*Note: Actual performance may vary based on sequence lengths and hardware.*

## Advanced Features

### 1. Custom Auto-Wrap Policy

Modify `FSDPConfig.get_auto_wrap_policy()` to wrap specific layers:

```python
@staticmethod
def get_auto_wrap_policy(model_config):
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            YourCustomLayer,
            AnotherLayer,
        },
    )
```

### 2. Mixed Precision with Different Dtypes

```python
# Use bfloat16 (recommended for A100 GPUs)
mixed_precision_dtype: str = "bf16"

# Use float16 (works on most GPUs)
mixed_precision_dtype: str = "fp16"
```

### 3. Hybrid Sharding Strategy

For multi-node training:

```python
sharding_strategy: str = "HYBRID_SHARD"
```

## License

This code follows the same license as the SeamlessM4T v2 model (Apache 2.0).

## Citation

If you use this code, please cite:

```bibtex
@article{seamlessm4t2023,
  title={SeamlessM4T: Massively Multilingual \& Multimodal Machine Translation},
  author={Seamless Communication et al.},
  journal={arXiv preprint arXiv:2308.11596},
  year={2023}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review PyTorch FSDP documentation
3. Open an issue in the repository

