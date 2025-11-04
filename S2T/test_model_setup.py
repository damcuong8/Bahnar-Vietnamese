"""
Test script để kiểm tra:
1. Tạo model
2. Load pretrained weights từ HuggingFace
3. Wrap với FSDP (shard)

Chạy với: python -m S2T.test_model_setup
"""

import os
import sys
import logging
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import modules
from configs import TrainingConfig
from model_utils import (
    setup_distributed,
    cleanup_distributed,
    create_model,
    wrap_model_with_fsdp
)
from memory_tracker import log_memory_stats, print_memory_summary


def test_model_creation():
    """Test tạo model"""
    logger.info("=" * 70)
    logger.info("TEST 1: Creating Model")
    logger.info("=" * 70)
    
    config = TrainingConfig()
    config.model_name_or_path = "facebook/seamless-m4t-v2-large"
    config.is_pretrained = True
    config.gradient_checkpointing = True
    
    log_memory_stats("Before model creation", rank=0)
    
    try:
        model, model_config = create_model(config)
        logger.info("✓ Model created successfully")
        log_memory_stats("After model creation", rank=0)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        return model, model_config, config
    except Exception as e:
        logger.error(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_load_weights(model, config):
    """Test load weights (đã được load trong create_model, nhưng test lại)"""
    logger.info("=" * 70)
    logger.info("TEST 2: Verifying Pretrained Weights")
    logger.info("=" * 70)
    
    if model is None:
        logger.warning("Skipping: Model not created")
        return False
    
    try:
        # Check if model has pretrained weights loaded
        # Check speech encoder first layer
        if hasattr(model, 'speech_encoder'):
            first_layer_weight = next(iter(model.speech_encoder.parameters()))
            logger.info(f"Speech encoder first parameter shape: {first_layer_weight.shape}")
            logger.info(f"Speech encoder first parameter mean: {first_layer_weight.mean().item():.6f}")
            
            # Check if weights are not all zeros (random init would have different stats)
            if torch.allclose(first_layer_weight, torch.zeros_like(first_layer_weight), atol=1e-6):
                logger.warning("⚠ Speech encoder weights appear to be zeros (might not be loaded)")
            else:
                logger.info("✓ Speech encoder weights appear to be loaded")
        
        # Check text decoder
        if hasattr(model, 'text_decoder'):
            first_layer_weight = next(iter(model.text_decoder.parameters()))
            logger.info(f"Text decoder first parameter shape: {first_layer_weight.shape}")
            logger.info(f"Text decoder first parameter mean: {first_layer_weight.mean().item():.6f}")
            
            if torch.allclose(first_layer_weight, torch.zeros_like(first_layer_weight), atol=1e-6):
                logger.warning("⚠ Text decoder weights appear to be zeros (might not be loaded)")
            else:
                logger.info("✓ Text decoder weights appear to be loaded")
        
        logger.info("✓ Pretrained weights verification completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to verify weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsdp_wrapping(model, model_config, config):
    """Test wrap model với FSDP"""
    logger.info("=" * 70)
    logger.info("TEST 3: Wrapping Model with FSDP")
    logger.info("=" * 70)
    
    if model is None:
        logger.warning("Skipping: Model not created")
        return None
    
    try:
        # Setup distributed (nếu chưa setup)
        if not torch.distributed.is_initialized():
            rank, world_size, local_rank = setup_distributed()
            config.local_rank = local_rank
            config.world_size = world_size
            config.use_fsdp = world_size > 1
        else:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            config.world_size = world_size
            config.use_fsdp = world_size > 1
        
        logger.info(f"Rank: {rank}, World size: {world_size}, Local rank: {config.local_rank}")
        
        if not config.use_fsdp:
            logger.warning("FSDP not enabled (single GPU or use_fsdp=False)")
            logger.info("To test FSDP, run with: torchrun --nproc_per_node=2 test_model_setup.py")
            return model
        
        # Move model to GPU
        device = torch.cuda.current_device()
        model = model.cuda(device)
        log_memory_stats("Before FSDP wrap", rank=rank)
        
        # Wrap with FSDP
        wrapped_model = wrap_model_with_fsdp(model, config, model_config, rank)
        
        logger.info("✓ Model wrapped with FSDP successfully")
        log_memory_stats("After FSDP wrap", rank=rank)
        print_memory_summary(rank=rank)
        
        # Test forward pass với dummy data
        logger.info("Testing forward pass with dummy data...")
        test_forward_pass(wrapped_model, device, rank)
        
        return wrapped_model
        
    except Exception as e:
        logger.error(f"✗ Failed to wrap with FSDP: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model, device, rank):
    """Test forward pass với dummy data"""
    logger.info("=" * 70)
    logger.info("TEST 4: Forward Pass with Dummy Data")
    logger.info("=" * 70)
    
    try:
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        seq_len_audio = 1000
        seq_len_text = 50
        feature_size = 80
        
        # Audio input features
        audio_input_features = torch.randn(
            batch_size, seq_len_audio, feature_size,
            device=device, dtype=torch.float32
        )
        audio_attention_mask = torch.ones(
            batch_size, seq_len_audio,
            device=device, dtype=torch.long
        )
        
        # Text input (pivot)
        text_input_pivot_ids = torch.randint(
            0, 1000, (batch_size, seq_len_text),
            device=device, dtype=torch.long
        )
        text_pivot_attention_mask = torch.ones(
            batch_size, seq_len_text,
            device=device, dtype=torch.long
        )
        
        # Labels
        labels = torch.randint(
            0, 1000, (batch_size, seq_len_text),
            device=device, dtype=torch.long
        )
        labels[:, 0] = -100  # Ignore first token
        
        logger.info(f"Dummy input shapes:")
        logger.info(f"  audio_input_features: {audio_input_features.shape}")
        logger.info(f"  text_input_pivot_ids: {text_input_pivot_ids.shape}")
        logger.info(f"  labels: {labels.shape}")
        
        log_memory_stats("Before forward pass", rank=rank)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                audio_input_features=audio_input_features,
                text_input_pivot_ids=text_input_pivot_ids,
                labels=labels,
                audio_attention_mask=audio_attention_mask,
                text_pivot_attention_mask=text_pivot_attention_mask,
            )
        
        log_memory_stats("After forward pass", rank=rank)
        
        # Unpack outputs
        ce_loss, kd_loss, n_valid_tokens, text_logits, text_pivot_logits = outputs
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  CE loss: {ce_loss.item():.4f}")
        logger.info(f"  KD loss: {kd_loss.item():.4f}")
        logger.info(f"  Valid tokens: {n_valid_tokens}")
        logger.info(f"  Text logits shape: {text_logits.shape}")
        logger.info(f"  Text pivot logits shape: {text_pivot_logits.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    logger.info("=" * 70)
    logger.info("MODEL SETUP TEST")
    logger.info("=" * 70)
    logger.info("This script tests:")
    logger.info("  1. Model creation")
    logger.info("  2. Loading pretrained weights from HuggingFace")
    logger.info("  3. Wrapping with FSDP")
    logger.info("  4. Forward pass with dummy data")
    logger.info("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, some tests may fail")
        return
    
    logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        # Test 1: Create model
        model, model_config, config = test_model_creation()
        if model is None:
            logger.error("Failed at model creation, stopping tests")
            return
        
        # Test 2: Verify weights
        weights_ok = test_load_weights(model, config)
        if not weights_ok:
            logger.warning("Weight verification had issues, but continuing...")
        
        # Test 3: FSDP wrapping
        wrapped_model = test_fsdp_wrapping(model, model_config, config)
        if wrapped_model is None:
            logger.warning("FSDP wrapping failed or skipped")
        
        logger.info("=" * 70)
        logger.info("✓ ALL TESTS COMPLETED")
        logger.info("=" * 70)
        
        # Final memory summary
        print_memory_summary(rank=0)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        cleanup_distributed()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    main()

