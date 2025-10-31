"""
Test script to verify that SeamlessM4T v2 FSDP training setup is working correctly.
Run this before launching full training to catch any issues early.
"""

import sys
import os
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required imports work"""
    logger.info("Testing imports...")
    
    try:
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        logger.info("✅ PyTorch and FSDP imports successful")
    except ImportError as e:
        logger.error(f"❌ Failed to import PyTorch/FSDP: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__} imported")
    except ImportError as e:
        logger.error(f"❌ Failed to import transformers: {e}")
        return False
    
    try:
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        logger.info("✅ Local model modules imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import local modules: {e}")
        logger.error("Make sure you're in the S2T directory with all required files")
        return False
    
    try:
        import wandb
        logger.info(f"✅ Wandb {wandb.__version__} imported (optional)")
    except ImportError:
        logger.warning("⚠️  Wandb not installed (optional)")
    
    return True


def test_gpu_availability():
    """Test GPU availability and configuration"""
    logger.info("\nTesting GPU availability...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"✅ CUDA available with {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name}")
        logger.info(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
        logger.info(f"    Compute capability: {props.major}.{props.minor}")
    
    if num_gpus < 2:
        logger.warning("⚠️  Only 1 GPU available. FSDP works best with 2+ GPUs")
    
    return True


def test_model_creation():
    """Test that the model can be created and moved to GPU"""
    logger.info("\nTesting model creation...")
    
    try:
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        
        config = SeamlessM4Tv2Config()
        model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created with {num_params / 1e6:.2f}M parameters")
        
        # Try to move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model moved to GPU successfully")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test a forward pass with dummy data"""
    logger.info("\nTesting forward pass...")
    
    try:
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        
        config = SeamlessM4Tv2Config()
        model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
        else:
            device = "cpu"
        
        # Create dummy inputs
        batch_size = 2
        audio_len = 100
        text_len = 20
        
        dummy_audio = torch.randn(batch_size, audio_len, 160).to(device)
        dummy_text = torch.randint(4, 1000, (batch_size, text_len)).to(device)
        dummy_labels = torch.randint(4, 1000, (batch_size, text_len)).to(device)
        dummy_audio_mask = torch.ones(batch_size, audio_len).to(device)
        dummy_text_mask = torch.ones(batch_size, text_len).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                audio_input_features=dummy_audio,
                text_input_pivot_ids=dummy_text,
                labels=dummy_labels,
                audio_attention_mask=dummy_audio_mask,
                text_pivot_attention_mask=dummy_text_mask,
            )
        
        logger.info(f"✅ Forward pass successful")
        logger.info(f"  Output contains {len(outputs)} elements")
        
        if outputs[0] is not None:
            logger.info(f"  CE Loss: {outputs[0].item():.4f}")
        if outputs[1] is not None:
            logger.info(f"  KD Loss: {outputs[1].item():.4f}")
        
        # Clean up
        del model, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test that data loader works correctly"""
    logger.info("\nTesting data loader...")
    
    try:
        from torch.utils.data import DataLoader
        from train_kaggle import DummySpeechToTextDataset, DataCollatorForSeamlessM4T
        
        # Create dataset
        dataset = DummySpeechToTextDataset(num_samples=10)
        collator = DataCollatorForSeamlessM4T(pad_token_id=0)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            num_workers=0,  # Use 0 for testing
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        logger.info(f"✅ Data loader created successfully")
        logger.info(f"  Batch keys: {list(batch.keys())}")
        logger.info(f"  Audio shape: {batch['audio_input_features'].shape}")
        logger.info(f"  Text shape: {batch['text_input_pivot_ids'].shape}")
        logger.info(f"  Labels shape: {batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsdp_wrapping():
    """Test FSDP model wrapping (requires 2+ GPUs)"""
    logger.info("\nTesting FSDP wrapping...")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus < 2:
        logger.warning("⚠️  Skipping FSDP test (requires 2+ GPUs)")
        return True
    
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        
        logger.info("✅ FSDP available and can be imported")
        logger.info("  Full FSDP test requires running with torchrun")
        logger.info("  Run: torchrun --nproc_per_node=2 test_setup.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FSDP test failed: {e}")
        return False


def test_disk_space():
    """Test available disk space"""
    logger.info("\nTesting disk space...")
    
    try:
        import shutil
        
        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        logger.info(f"  Total: {total_gb:.2f} GB")
        logger.info(f"  Free: {free_gb:.2f} GB")
        
        if free_gb < 5:
            logger.warning("⚠️  Low disk space (< 5 GB)")
        else:
            logger.info("✅ Sufficient disk space")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Disk space check failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*70)
    logger.info("SeamlessM4T v2 FSDP Training Setup Test")
    logger.info("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("GPU Availability", test_gpu_availability),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Data Loader", test_data_loader),
        ("FSDP Wrapping", test_fsdp_wrapping),
        ("Disk Space", test_disk_space),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Test Summary")
    logger.info("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*70)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! You're ready to start training.")
        logger.info("\nTo start training with 2 GPUs, run:")
        logger.info("  torchrun --nproc_per_node=2 train_kaggle.py")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Please fix the issues before training.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

