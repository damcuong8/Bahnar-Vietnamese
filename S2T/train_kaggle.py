"""
Fine-tuning SeamlessM4T v2 on Kaggle with FSDP (Fully Sharded Data Parallel)
This script supports 2-GPU training with wandb monitoring.
"""

import os
import math
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import shutil

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Import local modules
from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
from seamless_m4t_v2_config import SeamlessM4Tv2Config

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters with curriculum learning"""
    
    # Model configuration
    model_name_or_path: str = "facebook/seamless-m4t-v2-large"
    is_pretrained: bool = False  # True if loading pretrained weights, False if from scratch
    hf_cache_dir: Optional[str] = None  # Cache directory for HuggingFace models (auto-detect Kaggle if None)
    
    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data configuration
    max_audio_length: int = 30  # seconds
    max_text_length: int = 200  # tokens
    num_train_samples: int = 1000  # Placeholder for dummy data
    num_eval_samples: int = 100   # Placeholder for dummy data
    
    # Curriculum Learning - Stage Configuration
    enable_curriculum: bool = True  # Enable 3-stage training
    
    # Stage A: Encoder-only (decoder frozen) - CRITICAL for low-resource speech
    stage_a_steps: Optional[int] = None  # Auto: max(5000, 0.05 * total_steps); recommend 0.4-0.6 for low-resource
    encoder_lr: float = 1e-4  # HIGH LR: Encoder needs to learn new language from scratch
    stage_a_warmup_pct: float = 0.2  # 20% of stage A steps
    stage_a_warmup_min: int = 1000
    
    # Stage B: Unfreeze top-k decoder layers (progressive, 6 rounds)
    stage_b_steps: int = 10000  # Moderate training for adaptation
    unfreeze_top_k: int = 6  # Number of top decoder layers to unfreeze progressively
    decoder_lr: float = 1e-5  # LOW LR: ~10x lower than encoder to preserve Vietnamese knowledge

    
    # Stage C: Full decoder unfrozen - Fine-tuning phase
    use_layer_wise_lr_decay: bool = True  # ENABLE: Protect deeper layers with even lower LR
    layer_wise_lr_decay_rate: float = 0.9  # Each lower layer *= this rate (layer 0 gets lowest LR)
    
    # Knowledge Distillation Weights (dynamic across stages)
    kd_alpha_stage_a: float = 1.0 # KD weight in stage A
    kd_alpha_stage_b: float = 0.8  # KD weight in stage B
    kd_alpha_stage_c_start: float = 0.5  # KD weight at start of stage C
    kd_alpha_stage_c_end: float = 0.2  # KD weight at end of stage C
    weight_ce: float = 1.0  # CE loss weight (constant)
    
    # FSDP configuration
    use_fsdp: bool = True
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    cpu_offload: bool = False
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"  # fp16
    
    # Checkpointing
    output_dir: str = "./output"
    save_steps: int = 1000
    save_total_limit: int = 2
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "seamlessm4t-v2-finetuning-S2T"
    wandb_run_name: Optional[str] = None
    wandb_save_checkpoints: bool = True  # Upload checkpoints to wandb as artifacts
    
    # Optimization
    gradient_checkpointing: bool = False
    
    # Random seed
    seed: int = 42
    
    # Distributed
    local_rank: int = -1
    world_size: int = 2
    
    def __post_init__(self):
        """Adjust learning rates based on pretrained flag"""
        if self.is_pretrained:
            # Pretrained model: use smaller learning rates (but maintain the ratio)
            # For low-resource speech → high-resource text scenario:
            # - Encoder still needs significant learning (new acoustic features)
            # - Decoder needs very conservative updates (preserve language knowledge)
            if self.encoder_lr == 1e-4:  # If still default
                self.encoder_lr = 5e-5  # Still relatively high for encoder
            if self.decoder_lr == 1e-5:
                self.decoder_lr = 5e-6  # ~10x lower than encoder
    
    @property
    def weight_kd(self):
        """Backward compatibility - returns stage A alpha by default"""
        return self.kd_alpha_stage_a


@dataclass
class FSDPConfig:
    """FSDP-specific configuration"""
    
    # Sharding strategies
    SHARDING_STRATEGIES = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    @staticmethod
    def get_mixed_precision_policy(dtype: str = "fp16"):
        """Get mixed precision policy for FSDP"""
        if dtype == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            return None
    
    @staticmethod
    def get_auto_wrap_policy(model_config):
        """Get auto wrap policy for transformer layers"""
        from speech2text_model import (
            SeamlessM4Tv2ConformerEncoderLayer,
            SeamlessM4Tv2DecoderLayer,
        )
        
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                SeamlessM4Tv2ConformerEncoderLayer,
                SeamlessM4Tv2DecoderLayer,
            },
        )


class DummySpeechToTextDataset(Dataset):
    """
    Placeholder dataset for demonstration purposes.
    Replace this with your actual dataset implementation.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        max_audio_length: int = 30,
        max_text_length: int = 200,
        sample_rate: int = 16000,
        num_features: int = 160,
    ):
        self.num_samples = num_samples
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        self.num_features = num_features
        
        logger.info(
            f"Created dummy dataset with {num_samples} samples "
            f"(max_audio_length={max_audio_length}s, max_text_length={max_text_length} tokens)"
        )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns a dummy sample.
        In production, replace this with actual data loading.
        
        Returns:
            dict with:
                - audio_input_features: Audio features (seq_len, num_features)
                - text_input_pivot_ids: Text input IDs for pivot (teacher) 
                - labels: Target token IDs
                - audio_attention_mask: Attention mask for audio
                - text_pivot_attention_mask: Attention mask for text pivot
        """
        # Random audio length (in frames, not seconds)
        audio_len = random.randint(100, self.max_audio_length * 50)  # ~50 frames per second
        
        # Random text length
        text_len = random.randint(10, self.max_text_length)
        
        # Generate dummy audio features
        audio_features = torch.randn(audio_len, self.num_features)
        audio_attention_mask = torch.ones(audio_len)
        
        # Generate dummy text input (pivot) and labels
        text_input_ids = torch.randint(4, 1000, (text_len,))  # Avoid special tokens 0-3
        text_attention_mask = torch.ones(text_len)
        
        # Labels (same as text for dummy data)
        labels = torch.randint(4, 1000, (text_len,))
        
        return {
            "audio_input_features": audio_features,
            "text_input_pivot_ids": text_input_ids,
            "labels": labels,
            "audio_attention_mask": audio_attention_mask,
            "text_pivot_attention_mask": text_attention_mask,
        }


class DataCollatorForSeamlessM4T:
    """
    Data collator for SeamlessM4T v2 that handles variable-length inputs.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples with proper padding.
        """
        batch = {}
        
        # Pad audio features
        audio_features = [f["audio_input_features"] for f in features]
        audio_lengths = [f.shape[0] for f in audio_features]
        max_audio_len = max(audio_lengths)
        
        batch["audio_input_features"] = torch.stack([
            torch.nn.functional.pad(
                f, (0, 0, 0, max_audio_len - f.shape[0]), value=0.0
            ) for f in audio_features
        ])
        
        # Pad audio attention masks
        audio_masks = [f["audio_attention_mask"] for f in features]
        batch["audio_attention_mask"] = torch.stack([
            torch.nn.functional.pad(
                m, (0, max_audio_len - m.shape[0]), value=0
            ) for m in audio_masks
        ])
        
        # Pad text input (pivot) IDs
        text_input_ids = [f["text_input_pivot_ids"] for f in features]
        text_lengths = [t.shape[0] for t in text_input_ids]
        max_text_len = max(text_lengths)
        
        batch["text_input_pivot_ids"] = torch.stack([
            torch.nn.functional.pad(
                t, (0, max_text_len - t.shape[0]), value=self.pad_token_id
            ) for t in text_input_ids
        ])
        
        # Pad text attention masks
        text_masks = [f["text_pivot_attention_mask"] for f in features]
        batch["text_pivot_attention_mask"] = torch.stack([
            torch.nn.functional.pad(
                m, (0, max_text_len - m.shape[0]), value=0
            ) for m in text_masks
        ])
        
        # Pad labels
        labels = [f["labels"] for f in features]
        label_lengths = [l.shape[0] for l in labels]
        max_label_len = max(label_lengths)
        
        batch["labels"] = torch.stack([
            torch.nn.functional.pad(
                l, (0, max_label_len - l.shape[0]), value=-100  # -100 is ignore index
            ) for l in labels
        ])
        
        return batch


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def freeze_module(module: nn.Module, freeze: bool = True):
    """Freeze or unfreeze a module"""
    for param in module.parameters():
        param.requires_grad = not freeze
    if freeze:
        module.eval()


def setup_stage_a(model, config: TrainingConfig):
    """
    Stage A: Freeze decoder, train only encoder
    - text_encoder.eval() + requires_grad=False (teacher, always frozen)
    - text_decoder frozen
    - lm_head optionally frozen
    - speech_encoder trainable
    """
    logger.info("="*70)
    logger.info("Setting up Stage A: Encoder-only training")
    logger.info("="*70)
    
    # Freeze text encoder (teacher - always frozen)
    if hasattr(model, 'text_encoder'):
        freeze_module(model.text_encoder, freeze=True)
        logger.info("✓ Text encoder frozen (teacher)")
    
    # Freeze text decoder
    freeze_module(model.text_decoder, freeze=True)
    logger.info("✓ Text decoder frozen")
    
    # Freeze lm_head
    freeze_module(model.lm_head, freeze=True)
    logger.info("✓ LM head frozen")
    
    # Unfreeze speech encoder
    freeze_module(model.speech_encoder, freeze=False)
    model.speech_encoder.train()
    logger.info("✓ Speech encoder trainable")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def setup_stage_b(model, config: TrainingConfig, round_idx: Optional[int] = None, total_rounds: int = 6, num_layers: int = 24):
    """
    Stage B: Unfreeze top-k decoder layers in multiple rounds.

    Args:
        model: model with attribute text_decoder.layers and lm_head
        config: TrainingConfig (expects config.unfreeze_top_k)
        round_idx: which round to apply (0-based). If None -> unfreeze all top_k (old behavior).
        total_rounds: total number of rounds to split the unfreezing into (default 6).
        num_layers: number of decoder layers (default 24).
    """
    logger.info("="*70)
    logger.info(f"Setting up Stage B (round {round_idx} of {total_rounds}) - Unfreezing top layers")
    logger.info("="*70)

    # Determine actual number of decoder layers in the model
    actual_num_layers = len(getattr(model, "text_decoder").layers)
    if actual_num_layers != num_layers:
        logger.warning(
            f"num_layers={num_layers} but model.text_decoder has {actual_num_layers} layers. "
            "Using actual number of layers."
        )
        num_layers = actual_num_layers

    # how many top layers we intend to unfreeze in total (respect config)
    top_k = min(getattr(config, "unfreeze_top_k", num_layers), num_layers)

    # If round_idx is None -> behave like original: unfreeze all top_k
    if round_idx is None:
        start_idx = num_layers - top_k
        for i in range(start_idx, num_layers):
            freeze_module(model.text_decoder.layers[i], freeze=False)
            logger.info(f"✓ Unfroze decoder layer {i}")
    else:
        # validate round_idx
        if not (0 <= round_idx < total_rounds):
            raise ValueError(f"round_idx must be in [0, {total_rounds-1}], got {round_idx}")

        # Split the top_k layers into `total_rounds` buckets as evenly as possible
        base = top_k // total_rounds
        rem = top_k % total_rounds  # distribute remainder among first `rem` buckets

        # Build buckets: each bucket is a (start, end) of layer indices to unfreeze for that bucket
        buckets = []
        cur = 0
        for r in range(total_rounds):
            bucket_size = base + (1 if r < rem else 0)
            buckets.append((cur, cur + bucket_size))  # relative to top_k range [0, top_k)
            cur += bucket_size

        # determine cumulative layers to unfreeze up through current round
        # relative indices of top_k range that should be unfrozen now:
        cum_end = buckets[round_idx][1]
        # If previous rounds exist, include them as well (we unfreeze up to cum_end)
        # absolute layer indices:
        abs_start = num_layers - top_k
        abs_end = abs_start + cum_end  # exclusive

        if abs_end <= abs_start:
            logger.info(f"No layers to unfreeze in round {round_idx} (bucket empty).")
        else:
            for i in range(abs_start, abs_end):
                # if model has fewer layers than planned, ensure index valid
                if i < len(model.text_decoder.layers):
                    freeze_module(model.text_decoder.layers[i], freeze=False)
                    logger.info(f"✓ Unfroze decoder layer {i} (round {round_idx})")
                else:
                    logger.warning(f"Skipping layer index {i} because model only has {len(model.text_decoder.layers)} layers.")

    # Unfreeze lm_head once we start unfreezing (i.e., in any round or when round_idx is None)
    # Option: only unfreeze LM head when at least one decoder bucket is unfrozen (we use simple rule: unfreeze lm_head if round_idx is None or round_idx >= 0)
    freeze_module(model.lm_head, freeze=False)
    logger.info("✓ LM head unfrozen")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def setup_stage_c(model, config: TrainingConfig):
    """
    Stage C: Unfreeze all decoder layers (full fine-tune)
    - Keep speech_encoder trainable
    - Unfreeze all decoder layers + lm_head
    """
    logger.info("="*70)
    logger.info("Setting up Stage C: Full decoder unfrozen")
    logger.info("="*70)
    
    # Unfreeze entire text decoder
    freeze_module(model.text_decoder, freeze=False)
    model.text_decoder.train()
    logger.info("✓ All decoder layers unfrozen")
    
    # Unfreeze lm_head
    freeze_module(model.lm_head, freeze=False)
    logger.info("✓ LM head unfrozen")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def create_model(config: TrainingConfig):
    """Create and configure the SeamlessM4T v2 model"""
    logger.info(f"Loading model configuration from {config.model_name_or_path}")
    
    # Create model config
    model_config = SeamlessM4Tv2Config()
    
    # Create model
    logger.info("Initializing SeamlessM4Tv2ForSpeechToTextTrain_Pivot model")
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(model_config)
    
    # Load pretrained weights if specified
    if config.is_pretrained:
        # Determine cache directory (Kaggle-friendly)
        cache_dir = getattr(config, 'hf_cache_dir', None)
        if cache_dir is None:
            # Auto-detect Kaggle environment
            import os
            if os.path.exists('/kaggle'):
                cache_dir = "/kaggle/working/hf_cache"
                os.makedirs(cache_dir, exist_ok=True)
        
        # Use the new load_pretrained_weights method
        try:
            stats = model.load_pretrained_weights(
                config.model_name_or_path,
                cache_dir=cache_dir
            )
            logger.info("✓ Pretrained weights loaded successfully")
            
            # Log any issues (for debugging)
            if hasattr(stats, 'get'):
                for component, info in stats.items():
                    if isinstance(info, dict):
                        if info.get('missing'):
                            logger.debug(f"{component} - Missing keys: {len(info['missing'])}")
                        if info.get('unexpected'):
                            logger.debug(f"{component} - Unexpected keys: {len(info['unexpected'])}")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            logger.warning("Continuing with random initialization")
            config.is_pretrained = False  # Update config to reflect actual state
    else:
        logger.info("Training from scratch (random initialization)")
    
    # Enable gradient checkpointing if specified
    if config.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        if hasattr(model.speech_encoder, 'gradient_checkpointing_enable'):
            model.speech_encoder.gradient_checkpointing_enable()
        if hasattr(model.text_decoder, 'gradient_checkpointing_enable'):
            model.text_decoder.gradient_checkpointing_enable()
        if hasattr(model.text_encoder, 'gradient_checkpointing_enable'):
            model.text_encoder.gradient_checkpointing_enable()
    
    return model, model_config


def wrap_model_with_fsdp(
    model: nn.Module,
    config: TrainingConfig,
    model_config,
    rank: int,
):
    """Wrap model with FSDP for distributed training"""
    
    if not config.use_fsdp or config.world_size == 1:
        logger.info("FSDP not enabled or single GPU training, returning unwrapped model")
        return model
    
    logger.info("Wrapping model with FSDP")
    
    # Get sharding strategy
    sharding_strategy = FSDPConfig.SHARDING_STRATEGIES.get(
        config.sharding_strategy,
        ShardingStrategy.FULL_SHARD
    )
    
    # Get mixed precision policy
    mixed_precision_policy = None
    if config.use_mixed_precision:
        mixed_precision_policy = FSDPConfig.get_mixed_precision_policy(
            config.mixed_precision_dtype
        )
        logger.info(f"Using mixed precision: {config.mixed_precision_dtype}")
    
    # Get auto wrap policy
    auto_wrap_policy = FSDPConfig.get_auto_wrap_policy(model_config)
    
    # CPU offload config
    cpu_offload_config = CPUOffload(offload_params=True) if config.cpu_offload else None
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
    )
    
    logger.info(f"Model wrapped with FSDP (strategy: {config.sharding_strategy})")
    return model


def create_optimizer(model, config: TrainingConfig, stage: str = "A", lr: float = None):
    """
    Create optimizer with weight decay for specific training stage
    
    Args:
        model: The model (possibly FSDP-wrapped)
        config: Training configuration
        stage: Training stage ("A", "B", or "C")
        lr: Override learning rate (if None, uses config default for stage)
    """
    if lr is None:
        lr = config.encoder_lr if stage == "A" else (
            config.decoder_lr if stage == "B" else config.decoder_lr
        )
    
    # Collect parameters by module type and apply weight decay rules
    param_groups = []
    
    if stage == "A":
        # Stage A: Only encoder parameters
        encoder_decay = []
        encoder_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "bias" in name or "layer_norm" in name or "LayerNorm" in name:
                encoder_no_decay.append(param)
            else:
                encoder_decay.append(param)
        
        if encoder_decay:
            param_groups.append({
                "params": encoder_decay,
                "lr": lr,
                "weight_decay": config.weight_decay,
                "name": "encoder_decay"
            })
        if encoder_no_decay:
            param_groups.append({
                "params": encoder_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "name": "encoder_no_decay"
            })
        
        logger.info(f"Stage A optimizer: encoder_lr={lr:.2e}")
        
    elif stage == "B":
        # Stage B: Encoder + top decoder layers + lm_head
        encoder_params_decay = []
        encoder_params_no_decay = []
        decoder_params_decay = []
        decoder_params_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            is_no_decay = "bias" in name or "layer_norm" in name or "LayerNorm" in name
            
            if "speech_encoder" in name:
                if is_no_decay:
                    encoder_params_no_decay.append(param)
                else:
                    encoder_params_decay.append(param)
            else:  # decoder or lm_head
                if is_no_decay:
                    decoder_params_no_decay.append(param)
                else:
                    decoder_params_decay.append(param)
        
        # Add encoder params with encoder_lr
        if encoder_params_decay:
            param_groups.append({
                "params": encoder_params_decay,
                "lr": config.encoder_lr,
                "weight_decay": config.weight_decay,
                "name": "encoder_decay"
            })
        if encoder_params_no_decay:
            param_groups.append({
                "params": encoder_params_no_decay,
                "lr": config.encoder_lr,
                "weight_decay": 0.0,
                "name": "encoder_no_decay"
            })
        
        # Add decoder params with decoder_lr
        if decoder_params_decay:
            param_groups.append({
                "params": decoder_params_decay,
                "lr": lr,
                "weight_decay": config.weight_decay,
                "name": "decoder_top_decay"
            })
        if decoder_params_no_decay:
            param_groups.append({
                "params": decoder_params_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
                "name": "decoder_top_no_decay"
            })
        
        logger.info(f"Stage B optimizer: encoder_lr={config.encoder_lr:.2e}, decoder_top_lr={lr:.2e}")
        
    elif stage == "C":
        # Stage C: Encoder + all decoder + lm_head (with optional layer-wise LR decay)
        if config.use_layer_wise_lr_decay:
            # Layer-wise learning rate decay (higher LR for top layers)
            encoder_params = {"decay": [], "no_decay": []}
            decoder_layer_params = {}  # key: layer_idx, value: {decay: [], no_decay: []}
            lm_head_params = {"decay": [], "no_decay": []}
            
            num_decoder_layers = len(model.text_decoder.layers) if hasattr(model, 'text_decoder') else 0
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                is_no_decay = "bias" in name or "layer_norm" in name or "LayerNorm" in name
                param_key = "no_decay" if is_no_decay else "decay"
                
                if "speech_encoder" in name:
                    encoder_params[param_key].append(param)
                elif "lm_head" in name:
                    lm_head_params[param_key].append(param)
                elif "text_decoder.layers" in name:
                    # Extract layer index
                    import re
                    match = re.search(r'text_decoder\.layers\.(\d+)', name)
                    if match:
                        layer_idx = int(match.group(1))
                        if layer_idx not in decoder_layer_params:
                            decoder_layer_params[layer_idx] = {"decay": [], "no_decay": []}
                        decoder_layer_params[layer_idx][param_key].append(param)
            
            # Add encoder params
            if encoder_params["decay"]:
                param_groups.append({
                    "params": encoder_params["decay"],
                    "lr": config.encoder_lr,
                    "weight_decay": config.weight_decay,
                    "name": "encoder_decay"
                })
            if encoder_params["no_decay"]:
                param_groups.append({
                    "params": encoder_params["no_decay"],
                    "lr": config.encoder_lr,
                    "weight_decay": 0.0,
                    "name": "encoder_no_decay"
                })
            
            # Add decoder layer params with layer-wise LR
            for layer_idx in sorted(decoder_layer_params.keys()):
                # Higher layers get higher LR
                layer_lr = lr * (config.layer_wise_lr_decay_rate ** (num_decoder_layers - 1 - layer_idx))
                
                if decoder_layer_params[layer_idx]["decay"]:
                    param_groups.append({
                        "params": decoder_layer_params[layer_idx]["decay"],
                        "lr": layer_lr,
                        "weight_decay": config.weight_decay,
                        "name": f"decoder_layer_{layer_idx}_decay"
                    })
                if decoder_layer_params[layer_idx]["no_decay"]:
                    param_groups.append({
                        "params": decoder_layer_params[layer_idx]["no_decay"],
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                        "name": f"decoder_layer_{layer_idx}_no_decay"
                    })
            
            # Add lm_head params
            if lm_head_params["decay"]:
                param_groups.append({
                    "params": lm_head_params["decay"],
                    "lr": lr,
                    "weight_decay": config.weight_decay,
                    "name": "lm_head_decay"
                })
            if lm_head_params["no_decay"]:
                param_groups.append({
                    "params": lm_head_params["no_decay"],
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": "lm_head_no_decay"
                })
            
            logger.info(f"Stage C optimizer with layer-wise LR decay: encoder_lr={config.encoder_lr:.2e}, decoder_top_lr={lr:.2e}")
        else:
            # Standard: encoder + all decoder with two different LRs
            encoder_params_decay = []
            encoder_params_no_decay = []
            decoder_params_decay = []
            decoder_params_no_decay = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                is_no_decay = "bias" in name or "layer_norm" in name or "LayerNorm" in name
                
                if "speech_encoder" in name:
                    if is_no_decay:
                        encoder_params_no_decay.append(param)
                    else:
                        encoder_params_decay.append(param)
                else:  # decoder or lm_head
                    if is_no_decay:
                        decoder_params_no_decay.append(param)
                    else:
                        decoder_params_decay.append(param)
            
            # Add encoder params
            if encoder_params_decay:
                param_groups.append({
                    "params": encoder_params_decay,
                    "lr": config.encoder_lr,
                    "weight_decay": config.weight_decay,
                    "name": "encoder_decay"
                })
            if encoder_params_no_decay:
                param_groups.append({
                    "params": encoder_params_no_decay,
                    "lr": config.encoder_lr,
                    "weight_decay": 0.0,
                    "name": "encoder_no_decay"
                })
            
            # Add decoder params
            if decoder_params_decay:
                param_groups.append({
                    "params": decoder_params_decay,
                    "lr": lr,
                    "weight_decay": config.weight_decay,
                    "name": "decoder_full_decay"
                })
            if decoder_params_no_decay:
                param_groups.append({
                    "params": decoder_params_no_decay,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": "decoder_full_no_decay"
                })
            
            logger.info(f"Stage C optimizer: encoder_lr={config.encoder_lr:.2e}, decoder_full_lr={lr:.2e}")
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Log parameter group info
    total_trainable = sum(len(pg["params"]) for pg in param_groups)
    logger.info(f"Created optimizer with {len(param_groups)} parameter groups, {total_trainable} trainable param tensors")
    
    return optimizer


def get_current_lr_multiplier(global_step, total_steps, warmup_steps, min_lr_ratio=0.1):
    """
    Calculate the current LR multiplier based on cosine schedule.
    Replicates the logic from create_cosine_scheduler.
    
    Args:
        global_step: Current training step
        total_steps: Total number of training steps
        warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR ratio (default 0.0)
    
    Returns:
        Current multiplier for learning rate
    """
    # Warmup phase: linear 0 -> 1
    if global_step < warmup_steps:
        return float(global_step) / float(max(1, warmup_steps))
    
    # Cosine decay phase: 1 -> min_lr_ratio
    progress = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)


def add_new_param_groups_to_optimizer(optimizer, scheduler, model, config, stage, global_step, total_steps, round_idx=None):
    """
    Add new trainable parameters to existing optimizer when unfreezing layers.
    Handles layer-wise LR for Stage C and adjusts initial_lr to prevent jumps.
    
    Args:
        optimizer: Existing optimizer
        scheduler: LR scheduler (for consistency)
        model: The model
        config: Training config
        stage: Current stage ("B" or "C")
        global_step: Current global step
        total_steps: Total training steps
        round_idx: Optional round index for Stage B
    
    Returns:
        Number of new parameters added
    """
    # Get current params already in optimizer
    existing_params = set()
    for group in optimizer.param_groups:
        existing_params.update(group['params'])
    
    # Collect new params as (name, param) tuples
    new_params = [(name, param) for name, param in model.named_parameters() 
                  if param.requires_grad and param not in existing_params]
    
    if not new_params:
        logger.info("No new trainable parameters found to add to optimizer.")
        return 0
    
    # Calculate current LR multiplier to avoid jumps
    warmup_steps = max(200, int(0.03 * total_steps))
    current_multiplier = get_current_lr_multiplier(global_step, total_steps, warmup_steps)
    
    added_count = 0
    
    # BRANCH 1: Stage C with layer-wise LR decay
    if stage == "C" and config.use_layer_wise_lr_decay:
        logger.info(f"Stage C: Applying layer-wise LR decay (current multiplier: {current_multiplier:.4f})")
        
        # Separate params by type
        encoder_params = {'decay': [], 'no_decay': []}
        decoder_layer_params = {}  # {layer_idx: {'decay': [], 'no_decay': []}}
        other_decoder_params = {'decay': [], 'no_decay': []}
        
        num_decoder_layers = len(model.text_decoder.layers) if hasattr(model.text_decoder, 'layers') else 24
        
        for name, param in new_params:
            is_no_decay = "bias" in name or "layer_norm" in name or "LayerNorm" in name
            key = "no_decay" if is_no_decay else "decay"
            
            if "speech_encoder" in name:
                encoder_params[key].append(param)
            elif "text_decoder.layers" in name:
                match = re.search(r'text_decoder\.layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in decoder_layer_params:
                        decoder_layer_params[layer_idx] = {'decay': [], 'no_decay': []}
                    decoder_layer_params[layer_idx][key].append(param)
                else:
                    other_decoder_params[key].append(param)
            else:
                other_decoder_params[key].append(param)
        
        # Add encoder params
        encoder_lr_base = config.encoder_lr
        encoder_lr_adjusted = encoder_lr_base * current_multiplier
        
        if encoder_params['decay']:
            optimizer.add_param_group({
                'params': encoder_params['decay'],
                'lr': encoder_lr_adjusted,
                'initial_lr': encoder_lr_base,
                'weight_decay': config.weight_decay
            })
            added_count += len(encoder_params['decay'])
            logger.info(f"Added {len(encoder_params['decay'])} encoder params (decay) - base_lr={encoder_lr_base:.2e}, current_lr={encoder_lr_adjusted:.2e}")
        
        if encoder_params['no_decay']:
            optimizer.add_param_group({
                'params': encoder_params['no_decay'],
                'lr': encoder_lr_adjusted,
                'initial_lr': encoder_lr_base,
                'weight_decay': 0.0
            })
            added_count += len(encoder_params['no_decay'])
            logger.info(f"Added {len(encoder_params['no_decay'])} encoder params (no decay) - base_lr={encoder_lr_base:.2e}, current_lr={encoder_lr_adjusted:.2e}")
        
        # Add decoder layer params with layer-wise LR
        base_decoder_lr = config.decoder_lr
        
        for layer_idx in sorted(decoder_layer_params.keys()):
            # Calculate layer-wise LR: higher layers get higher LR
            layer_lr_base = base_decoder_lr * (config.layer_wise_lr_decay_rate ** (num_decoder_layers - 1 - layer_idx))
            layer_lr_adjusted = layer_lr_base * current_multiplier
            
            if decoder_layer_params[layer_idx]['decay']:
                optimizer.add_param_group({
                    'params': decoder_layer_params[layer_idx]['decay'],
                    'lr': layer_lr_adjusted,
                    'initial_lr': layer_lr_base,
                    'weight_decay': config.weight_decay
                })
                added_count += len(decoder_layer_params[layer_idx]['decay'])
            
            if decoder_layer_params[layer_idx]['no_decay']:
                optimizer.add_param_group({
                    'params': decoder_layer_params[layer_idx]['no_decay'],
                    'lr': layer_lr_adjusted,
                    'initial_lr': layer_lr_base,
                    'weight_decay': 0.0
                })
                added_count += len(decoder_layer_params[layer_idx]['no_decay'])
            
            logger.info(f"Added decoder layer {layer_idx} - base_lr={layer_lr_base:.2e}, current_lr={layer_lr_adjusted:.2e}")
        
        # Add other decoder params (lm_head, etc.)
        other_lr_adjusted = base_decoder_lr * current_multiplier
        
        if other_decoder_params['decay']:
            optimizer.add_param_group({
                'params': other_decoder_params['decay'],
                'lr': other_lr_adjusted,
                'initial_lr': base_decoder_lr,
                'weight_decay': config.weight_decay
            })
            added_count += len(other_decoder_params['decay'])
        
        if other_decoder_params['no_decay']:
            optimizer.add_param_group({
                'params': other_decoder_params['no_decay'],
                'lr': other_lr_adjusted,
                'initial_lr': base_decoder_lr,
                'weight_decay': 0.0
            })
            added_count += len(other_decoder_params['no_decay'])
        
        if other_decoder_params['decay'] or other_decoder_params['no_decay']:
            logger.info(f"Added other decoder params (lm_head, etc.) - base_lr={base_decoder_lr:.2e}, current_lr={other_lr_adjusted:.2e}")
    
    # BRANCH 2: Stage B or Stage C without layer-wise LR
    else:
        stage_label = f"B_round_{round_idx}" if stage == "B" and round_idx is not None else stage
        logger.info(f"Stage {stage_label}: Standard param addition (current multiplier: {current_multiplier:.4f})")
        
        # Simple separation: encoder vs decoder
        encoder_params = {'decay': [], 'no_decay': []}
        decoder_params = {'decay': [], 'no_decay': []}
        
        for name, param in new_params:
            is_no_decay = "bias" in name or "layer_norm" in name or "LayerNorm" in name
            key = "no_decay" if is_no_decay else "decay"
            
            if "speech_encoder" in name:
                encoder_params[key].append(param)
            else:
                decoder_params[key].append(param)
        
        # Determine base LR
        encoder_lr_base = config.encoder_lr
        if stage == "B":
            decoder_lr_base = config.decoder_lr
        else:
            decoder_lr_base = config.decoder_lr
        
        # Adjust with current multiplier
        encoder_lr_adjusted = encoder_lr_base * current_multiplier
        decoder_lr_adjusted = decoder_lr_base * current_multiplier
        
        # Add encoder params
        if encoder_params['decay']:
            optimizer.add_param_group({
                'params': encoder_params['decay'],
                'lr': encoder_lr_adjusted,
                'initial_lr': encoder_lr_base,
                'weight_decay': config.weight_decay
            })
            added_count += len(encoder_params['decay'])
        
        if encoder_params['no_decay']:
            optimizer.add_param_group({
                'params': encoder_params['no_decay'],
                'lr': encoder_lr_adjusted,
                'initial_lr': encoder_lr_base,
                'weight_decay': 0.0
            })
            added_count += len(encoder_params['no_decay'])
        
        # Add decoder params
        if decoder_params['decay']:
            optimizer.add_param_group({
                'params': decoder_params['decay'],
                'lr': decoder_lr_adjusted,
                'initial_lr': decoder_lr_base,
                'weight_decay': config.weight_decay
            })
            added_count += len(decoder_params['decay'])
        
        if decoder_params['no_decay']:
            optimizer.add_param_group({
                'params': decoder_params['no_decay'],
                'lr': decoder_lr_adjusted,
                'initial_lr': decoder_lr_base,
                'weight_decay': 0.0
            })
            added_count += len(decoder_params['no_decay'])
        
        logger.info(f"Added {added_count} params - Encoder: base={encoder_lr_base:.2e}/current={encoder_lr_adjusted:.2e}, Decoder: base={decoder_lr_base:.2e}/current={decoder_lr_adjusted:.2e}")
    
    logger.info(f"✓ Total new parameters added to optimizer: {added_count}")
    return added_count


def create_cosine_scheduler(
    optimizer,
    config,
    num_training_steps: int,
    warmup_steps: Optional[int] = None,
    min_lr_ratio: float = 0.0,       # final lr = base_lr * min_lr_ratio
):
    """Create LR scheduler: linear warmup -> cosine decay to min_lr_ratio.
    
    Args:
        optimizer: torch optimizer
        config: training config (kept for API compatibility)
        num_training_steps: total training steps (batches)
        warmup_steps: steps for linear warmup; if None uses max(200, 3% of total)
        min_lr_ratio: final LR ratio relative to base_lr (0.0 -> decay to 0)
    Returns:
        a torch.optim.lr_scheduler.LambdaLR scheduler
    """
    if warmup_steps is None:
        warmup_steps = max(200, int(0.03 * num_training_steps))
        

    def lr_lambda(step: int):
        # warmup phase: linear 0 -> 1
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay phase: 1 -> min_lr_ratio
        progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    scheduler = LambdaLR(optimizer, lr_lambda)

    logger.info(f"Created cosine scheduler with {warmup_steps} warmup steps out of {num_training_steps} total steps, min_lr_ratio={min_lr_ratio}")

    return scheduler



def get_kd_alpha(config: TrainingConfig, stage: str, step_in_stage: int, total_steps_in_stage: int) -> float:
    """
    Get dynamic KD alpha based on current stage and progress
    
    Args:
        config: Training configuration
        stage: Current stage ("A", "B", or "C")
        step_in_stage: Current step within the stage
        total_steps_in_stage: Total steps for this stage
        
    Returns:
        KD alpha weight
    """
    if stage == "A":
        return config.kd_alpha_stage_a
    elif stage == "B":
        progress = step_in_stage / max(1, total_steps_in_stage)
        alpha = config.kd_alpha_stage_a + progress * (config.kd_alpha_stage_c - config.kd_alpha_stage_a)
        return alpha
    elif stage == "C":
        # Linear decay from start to end
        progress = step_in_stage / max(1, total_steps_in_stage)
        alpha = config.kd_alpha_stage_c_start + progress * (config.kd_alpha_stage_c_end - config.kd_alpha_stage_c_start)
        return alpha
    else:
        return 1.0


def _upload_checkpoint_to_wandb(checkpoint_path: str, step: int, stage: Optional[str], config: TrainingConfig):
    """
    Upload checkpoint to wandb as an artifact.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        step: Training step number
        stage: Current training stage (A/B/C)
        config: Training configuration
    """
    if not WANDB_AVAILABLE:
        return
    
    import time
    
    try:
        # Get file size for logging
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f"Uploading checkpoint to wandb (size: {file_size_mb:.2f} MB)...")
        
        start_time = time.time()
        
        # Create artifact name
        stage_suffix = f"-stage-{stage}" if stage else ""
        artifact_name = f"model-checkpoint-step-{step}{stage_suffix}"
        
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Model checkpoint at step {step}" + (f" (Stage {stage})" if stage else ""),
            metadata={
                "step": step,
                "stage": stage or "unknown",
                "config": {
                    "encoder_lr": config.encoder_lr,
                    "is_pretrained": config.is_pretrained,
                    "enable_curriculum": config.enable_curriculum,
                    "unfreeze_top_k": config.unfreeze_top_k,
                }
            }
        )
        
        # Add checkpoint file to artifact
        artifact.add_file(checkpoint_path, name="pytorch_model.bin")
        
        # Log artifact to wandb
        wandb.log_artifact(artifact)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded checkpoint to wandb: {artifact_name} (took {elapsed:.1f}s)")
        
        # Also log as a simple file for quick access
        try:
            wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
        except Exception as e:
            logger.debug(f"Note: wandb.save() failed: {e} (artifact upload succeeded)")
            
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to wandb: {e}")
        raise


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    config: TrainingConfig,
    rank: int,
    scaler: Optional[object] = None,  # torch.cuda.amp.GradScaler or dict/state
    stage: Optional[str] = None,  # Current training stage (A/B/C)
):
    """Save model checkpoint (includes optional AMP GradScaler state) and optionally upload to wandb."""
    if rank != 0:
        return

    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare checkpoint dict (scaler added below if present)
    def _add_common_fields(checkpoint_dict):
        checkpoint_dict.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        })
        # Add stage info if provided
        if stage is not None:
            checkpoint_dict["stage"] = stage
        # Add scaler state if provided (accept either GradScaler or already a dict)
        if scaler is not None:
            try:
                # If scaler is an object with state_dict()
                checkpoint_dict["scaler"] = scaler.state_dict()
            except Exception:
                # If scaler is already a dict (or cannot call state_dict), just save it
                checkpoint_dict["scaler"] = scaler

    # Save using FSDP state dict API if using FSDP
    checkpoint_path = None
    if getattr(config, "use_fsdp", False) and isinstance(model, FSDP):
        # Configure state dict settings
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()

            checkpoint = {"model": state_dict}
            _add_common_fields(checkpoint)

            checkpoint_path_tmp = os.path.join(checkpoint_dir, "pytorch_model.bin.tmp")
            checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            torch.save(checkpoint, checkpoint_path_tmp)
            os.replace(checkpoint_path_tmp, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    else:
        checkpoint = {
            "model": model.state_dict(),
        }
        _add_common_fields(checkpoint)

        checkpoint_path_tmp = os.path.join(checkpoint_dir, "pytorch_model.bin.tmp")
        checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        torch.save(checkpoint, checkpoint_path_tmp)
        os.replace(checkpoint_path_tmp, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Upload to wandb if enabled
    if config.use_wandb and config.wandb_save_checkpoints and WANDB_AVAILABLE and checkpoint_path is not None:
        try:
            _upload_checkpoint_to_wandb(checkpoint_path, step, stage, config)
        except Exception as e:
            logger.warning(f"Failed to upload checkpoint to wandb: {e}")

    # Remove old checkpoints if exceeding limit
    if getattr(config, "save_total_limit", None) is not None:
        # find checkpoint-* dirs and parse step numbers robustly
        def _try_parse_step(name):
            try:
                return int(name.split("-")[-1])
            except Exception:
                return None

        checkpoints = []
        for d in os.listdir(config.output_dir):
            if d.startswith("checkpoint-"):
                step_num = _try_parse_step(d)
                if step_num is not None:
                    checkpoints.append((step_num, d))

        checkpoints = sorted(checkpoints, key=lambda x: x[0])  # sort by step number

        if len(checkpoints) > config.save_total_limit:
            to_remove = checkpoints[:-config.save_total_limit]
            for (step_num, old_checkpoint) in to_remove:
                old_path = os.path.join(config.output_dir, old_checkpoint)
                try:
                    logger.info(f"Removing old checkpoint: {old_path}")
                    shutil.rmtree(old_path)
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {old_path}: {e}")


def train_stage(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    rank: int,
    stage: str,
    num_steps: int,
    global_step_offset: int = 0,
):
    """
    Train for one stage of curriculum learning with optional AMP + GradScaler.
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0

    steps_completed = 0
    dataloader_iter = iter(train_dataloader)

    # AMP config: use config.use_amp if present, else default True
    use_amp = getattr(config, "use_amp", True) and torch.cuda.is_available()
    scaler: Optional[torch.cuda.amp.GradScaler] = torch.cuda.amp.GradScaler() if use_amp else None

    logger.info(f"Starting Stage {stage} training for {num_steps} steps (AMP={use_amp})")

    while steps_completed < num_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            batch = next(dataloader_iter)

        # Move batch to device
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

        # Get dynamic KD alpha
        kd_alpha = get_kd_alpha(config, stage, steps_completed, num_steps)

        # Forward (with autocast if AMP enabled)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                audio_input_features=batch["audio_input_features"],
                text_input_pivot_ids=batch["text_input_pivot_ids"],
                labels=batch["labels"],
                audio_attention_mask=batch["audio_attention_mask"],
                text_pivot_attention_mask=batch["text_pivot_attention_mask"],
            )

            # Unpack outputs
            ce_loss, kd_loss, n_valid_tokens, text_logits, text_pivot_logits = outputs

            # Combined loss with dynamic KD weight
            loss = config.weight_ce * ce_loss + kd_alpha * kd_loss

            # Scale loss for gradient accumulation (division before backward)
            loss = loss / config.gradient_accumulation_steps

        # Backward pass (use scaler if AMP)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights on accumulation boundary
        if (steps_completed + 1) % config.gradient_accumulation_steps == 0:
            # If AMP, unscale before gradient clipping
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            # Gradient clipping (if requested)
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer step (use scaler when AMP)
            if use_amp and scaler is not None:
                # scaler.step will skip if inf/NaN grads; then scaler.update()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Scheduler step (keep after optimizer.step)
            try:
                scheduler.step()
            except Exception:
                # some schedulers require step per epoch or different call signature; handle gracefully
                pass

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

        # Logging accumulators (restore original scale by multiplying back)
        total_loss += loss.item() * config.gradient_accumulation_steps
        total_ce_loss += ce_loss.item() if ce_loss is not None else 0.0
        total_kd_loss += kd_loss.item() if kd_loss is not None else 0.0

        steps_completed += 1
        global_step = global_step_offset + steps_completed

        # Periodic logging
        if steps_completed % config.logging_steps == 0 and rank == 0:
            avg_loss = total_loss / config.logging_steps
            avg_ce_loss = total_ce_loss / config.logging_steps
            avg_kd_loss = total_kd_loss / config.logging_steps
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None

            logger.info(
                f"Stage {stage} | Step {steps_completed}/{num_steps} (Global: {global_step}) | "
                f"Loss: {avg_loss:.4f} | CE: {avg_ce_loss:.4f} | KD: {avg_kd_loss:.4f} (α={kd_alpha:.3f}) | "
                f"LR: {current_lr:.2e}" if current_lr is not None else ""
            )

            if config.use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/ce_loss": avg_ce_loss,
                    "train/kd_loss": avg_kd_loss,
                    "train/kd_alpha": kd_alpha,
                    "train/stage": stage,
                    "train/step_in_stage": steps_completed,
                    "train/global_step": global_step,
                }
                if current_lr is not None:
                    log_dict["train/learning_rate"] = current_lr
                wandb.log(log_dict)

            total_loss = 0.0
            total_ce_loss = 0.0
            total_kd_loss = 0.0

        # Save checkpoint
        if steps_completed % config.save_steps == 0:
            # NOTE: consider saving scaler.state_dict() too so you can resume AMP training
            save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, 
                          scaler=scaler.state_dict() if scaler is not None else None,
                          stage=stage)

    logger.info(f"Completed Stage {stage} ({steps_completed} steps)")
    return global_step_offset + steps_completed



def main():
    """Main training function with curriculum learning support"""
    
    # Create config
    config = TrainingConfig()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    config.local_rank = local_rank
    config.world_size = world_size
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize wandb
    if rank == 0 and config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )
        logger.info(f"Initialized wandb project: {config.wandb_project}")
    
    # Create output directory
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Create model
    model, model_config = create_model(config)
    model = model.cuda()
    
    # Create datasets and dataloader
    logger.info("Creating datasets")
    train_dataset = DummySpeechToTextDataset(
        num_samples=config.num_train_samples,
        max_audio_length=config.max_audio_length,
        max_text_length=config.max_text_length,
    )
    
    data_collator = DataCollatorForSeamlessM4T(pad_token_id=model_config.pad_token_id)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * config.num_epochs
    
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Steps per epoch: {num_update_steps_per_epoch}")
    logger.info(f"Total training steps: {total_training_steps}")
    
    # Determine stage steps
    # For low-resource speech → high-resource text: prioritize Stage A (encoder learning)
    if config.stage_a_steps is None:
        # Default: 50% of total steps for encoder (can be adjusted via config)
        # Minimum 5000 steps to ensure sufficient encoder training
        config.stage_a_steps = max(5000, int(0.5 * total_training_steps))
        logger.info(f"Auto-set Stage A steps to {config.stage_a_steps:,} (50% of total) for low-resource speech training")
    
    # Stage B configuration: split into multiple rounds
    num_stage_b_rounds = 6  # Number of rounds to progressively unfreeze Stage B layers
    stage_b_steps_per_round = config.stage_b_steps // num_stage_b_rounds
    
    stage_c_steps = max(0, total_training_steps - config.stage_a_steps - config.stage_b_steps)
    
    logger.info("\n" + "="*70)
    logger.info("CURRICULUM LEARNING PLAN")
    logger.info("="*70)
    logger.info(f"Stage A (Encoder-only):           {config.stage_a_steps:,} steps")
    logger.info(f"Stage B (Top-{config.unfreeze_top_k} decoder layers): {config.stage_b_steps:,} steps")
    logger.info(f"  └─ Split into {num_stage_b_rounds} rounds, ~{stage_b_steps_per_round:,} steps/round")
    logger.info(f"Stage C (Full decoder):           {stage_c_steps:,} steps")
    logger.info(f"Total:                            {total_training_steps:,} steps")
    logger.info("="*70 + "\n")
    
    if not config.enable_curriculum:
        logger.info("⚠️  Curriculum learning DISABLED - using standard training")
        # Standard training would go here
        logger.warning("Standard training not implemented in this version. Please enable curriculum learning.")
        return
    
    # ========================================================================
    # STAGE A: Encoder-only training
    # ========================================================================
    logger.info("\n" + "#"*70)
    logger.info("# STAGE A: ENCODER-ONLY TRAINING")
    logger.info("#"*70 + "\n")
    
    # Setup Stage A
    setup_stage_a(model, config)
    
    # Sync after setup (important for FSDP)
    if config.world_size > 1:
        dist.barrier()
    
    # Wrap with FSDP after stage setup
    model = wrap_model_with_fsdp(model, config, model_config, rank)
    
    # Create optimizer ONCE for ALL stages (only Stage A params initially)
    warmup_a = max(config.stage_a_warmup_min, int(config.stage_a_warmup_pct * config.stage_a_steps))
    optimizer = create_optimizer(model, config, stage="A")
    
    # Create scheduler for entire training (will be used across all stages)
    # Use total_training_steps for the scheduler
    total_warmup_steps = warmup_a  # Initial warmup for Stage A
    scheduler = create_cosine_scheduler(optimizer, config, total_training_steps, total_warmup_steps)
    
    logger.info(f"✓ Created shared optimizer with {len(optimizer.param_groups)} initial param groups")
    
    # Train Stage A
    global_step = train_stage(
        model, train_dataloader, optimizer, scheduler,
        config, rank, stage="A", num_steps=config.stage_a_steps, global_step_offset=0
    )
    
    # Save Stage A checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage="A")
    
    # ========================================================================
    # STAGE B: Unfreeze top-k decoder layers (Progressive - 6 rounds)
    # ========================================================================
    logger.info("\n" + "#"*70)
    logger.info("# STAGE B: PROGRESSIVE UNFREEZE TOP-K DECODER LAYERS")
    logger.info("#"*70 + "\n")
    
    # Train Stage B in multiple rounds
    for round_idx in range(num_stage_b_rounds):
        logger.info(f"\n{'─'*70}")
        logger.info(f"Stage B - Round {round_idx + 1}/{num_stage_b_rounds}")
        logger.info(f"{'─'*70}")
        
        # Unfreeze layers progressively
        setup_stage_b(model, config, round_idx=round_idx, total_rounds=num_stage_b_rounds)
        
        # Sync after setup
        if config.world_size > 1:
            dist.barrier()
        
        # Add newly unfrozen parameters to optimizer
        num_added = add_new_param_groups_to_optimizer(
            optimizer, scheduler, model, config,
            stage="B",
            global_step=global_step,
            total_steps=total_training_steps,
            round_idx=round_idx
        )
        if num_added > 0:
            logger.info(f"Round {round_idx}: Added {num_added} new parameters to optimizer")
        
        # Train this round
        steps_this_round = stage_b_steps_per_round
        if round_idx == num_stage_b_rounds - 1:
            # Last round: train remaining steps
            steps_this_round = config.stage_b_steps - (stage_b_steps_per_round * (num_stage_b_rounds - 1))
        
        global_step = train_stage(
            model, train_dataloader, optimizer, scheduler,
            config, rank, stage=f"B_R{round_idx}", num_steps=steps_this_round, global_step_offset=global_step
        )
        
        # Save checkpoint after each round
        if rank == 0 and (round_idx + 1) % 2 == 0:  # Save every 2 rounds
            save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage=f"B-round{round_idx+1}")
    
    # Save Stage B final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage="B")
    
    logger.info(f"\n✓ Completed all {num_stage_b_rounds} rounds of Stage B")
    
    # ========================================================================
    # STAGE C: Full decoder unfrozen
    # ========================================================================
    logger.info("\n" + "#"*70)
    logger.info("# STAGE C: FULL DECODER TRAINING")
    logger.info("#"*70 + "\n")
    
    # Setup Stage C - unfreeze all remaining decoder layers
    setup_stage_c(model, config)
    
    # Sync after setup
    if config.world_size > 1:
        dist.barrier()
    
    # Add newly unfrozen parameters to existing optimizer
    num_added = add_new_param_groups_to_optimizer(
        optimizer, scheduler, model, config,
        stage="C",
        global_step=global_step,
        total_steps=total_training_steps
    )
    logger.info(f"Stage C: Added {num_added} new parameters to optimizer")
    
    # Train Stage C
    global_step = train_stage(
        model, train_dataloader, optimizer, scheduler,
        config, rank, stage="C", num_steps=stage_c_steps, global_step_offset=global_step
    )
    
    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, 0, global_step, config, rank, stage="C")
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 Training completed! Final checkpoint saved at step {global_step}")
        logger.info(f"{'='*70}\n")
    
    # Cleanup
    if config.use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()

