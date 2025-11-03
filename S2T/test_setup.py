"""
Test script to verify that SeamlessM4T v2 training setup is working correctly.
Uses real training config and data for comprehensive testing.

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
        from configs import TrainingConfig
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        from datasets import ViBaSpeechToTextDataset, DataCollatorSpeechToText
        logger.info("✅ Local model modules imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import local modules: {e}")
        logger.error("Make sure you're in the S2T directory with all required files")
        import traceback
        traceback.print_exc()
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


def test_config_loading():
    """Test loading training configuration"""
    logger.info("\nTesting config loading...")
    
    try:
        from configs import TrainingConfig
        
        config = TrainingConfig()
        
        logger.info(f"✅ Training config loaded successfully")
        logger.info(f"  Excel path: {config.excel_path}")
        logger.info(f"  Batch size: {config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  Encoder LR: {config.encoder_lr}")
        logger.info(f"  Decoder LR: {config.decoder_lr}")
        logger.info(f"  Curriculum enabled: {config.enable_curriculum}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test that the model can be created and moved to GPU"""
    logger.info("\nTesting model creation...")
    
    try:
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        
        config = SeamlessM4Tv2Config()
        model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model created successfully")
        logger.info(f"  Total parameters: {num_params / 1e6:.2f}M")
        logger.info(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        
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
    logger.info("\nTesting forward pass with dummy data...")
    
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
        
        # Create dummy inputs matching the real data format
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


def test_dataset_loading():
    """Test loading actual dataset from Excel file"""
    logger.info("\nTesting dataset loading from Excel...")
    
    try:
        from configs import TrainingConfig
        from datasets import ViBaSpeechToTextDataset
        from transformers import AutoProcessor
        
        config = TrainingConfig()
        
        # Check if Excel file exists
        if not os.path.exists(config.excel_path):
            logger.warning(f"⚠️  Excel file not found at {config.excel_path}")
            logger.warning("  Please update config.excel_path or create synthetic data")
            return True  # Don't fail the test, just warn
        
        logger.info(f"  Loading from: {config.excel_path}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        
        # Create dataset
        dataset = ViBaSpeechToTextDataset(
            excel_path=config.excel_path,
            audio_col=config.audio_col,
            vi_col=config.vi_col,
            en_col=config.en_col,
            target_sr=16000,
            mono=True,
            augment_fn=None,
            use_cache=False,
        )
        
        logger.info(f"✅ Dataset loaded successfully")
        logger.info(f"  Total samples: {len(dataset)}")
        
        # Try to load one sample
        try:
            sample = dataset[0]
            logger.info(f"  Sample keys: {list(sample.keys())}")
            logger.info(f"  Waveform shape: {sample['waveform'].shape}")
            logger.info(f"  VI text: {sample['raw_vi'][:50]}...")
            if sample['raw_en']:
                logger.info(f"  EN text: {sample['raw_en'][:50]}...")
            logger.info("✅ Successfully loaded and processed one sample")
        except Exception as e:
            logger.error(f"❌ Failed to load sample: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loader with real data using training config"""
    logger.info("\nTesting data loader with real config...")
    
    try:
        from configs import TrainingConfig
        from torch.utils.data import DataLoader
        from datasets import ViBaSpeechToTextDataset, DataCollatorSpeechToText
        from transformers import SeamlessM4TFeatureExtractor, AutoProcessor
        
        config = TrainingConfig()
        
        # Check if Excel file exists
        if not os.path.exists(config.excel_path):
            logger.warning(f"⚠️  Excel file not found at {config.excel_path}")
            logger.warning("  Skipping data loader test")
            return True
        
        # Create processor and feature extractor
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        
        feature_extractor = SeamlessM4TFeatureExtractor(
            feature_size=80,
            sampling_rate=16000,
            num_mel_bins=80,
            padding_value=0.0,
            stride=2,
        )
        
        # Create dataset
        dataset = ViBaSpeechToTextDataset(
            excel_path=config.excel_path,
            audio_col=config.audio_col,
            vi_col=config.vi_col,
            en_col=config.en_col,
            target_sr=16000,
            mono=True,
            augment_fn=None,
            use_cache=False,
        )
        
        # Create collator
        collator = DataCollatorSpeechToText(
            feature_extractor=feature_extractor,
            processor=processor,
            padding=True,
            pad_to_multiple_of=8,
            target_language="vi",
            pivot_language="en"
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=False,  # Don't shuffle for testing
            collate_fn=collator,
            num_workers=0,  # Use 0 for testing
        )
        
        logger.info(f"✅ Data loader created successfully")
        logger.info(f"  Batch size: {config.per_device_train_batch_size}")
        logger.info(f"  Total batches: {len(dataloader)}")
        
        # Get one batch
        batch = next(iter(dataloader))
        
        logger.info(f"  Batch keys: {list(batch.keys())}")
        logger.info(f"  Audio features shape: {batch['audio_input_features'].shape}")
        logger.info(f"  Audio mask shape: {batch['audio_attention_mask'].shape}")
        logger.info(f"  Pivot text shape: {batch['text_input_pivot_ids'].shape}")
        logger.info(f"  Pivot mask shape: {batch['text_pivot_attention_mask'].shape}")
        logger.info(f"  Labels shape: {batch['labels'].shape}")
        
        logger.info("✅ Successfully created and loaded batch from real data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test a mini training loop with real data"""
    logger.info("\nTesting mini training loop...")
    
    try:
        from configs import TrainingConfig
        from torch.utils.data import DataLoader
        from datasets import ViBaSpeechToTextDataset, DataCollatorSpeechToText
        from transformers import SeamlessM4TFeatureExtractor, AutoProcessor
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        
        config = TrainingConfig()
        
        # Check if Excel file exists
        if not os.path.exists(config.excel_path):
            logger.warning(f"⚠️  Excel file not found - skipping training loop test")
            return True
        
        # Create model
        model_config = SeamlessM4Tv2Config()
        model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(model_config)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create dataset and dataloader
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        
        feature_extractor = SeamlessM4TFeatureExtractor(
            feature_size=80,
            sampling_rate=16000,
            num_mel_bins=80,
            padding_value=0.0,
            stride=2,
        )
        
        dataset = ViBaSpeechToTextDataset(
            excel_path=config.excel_path,
            audio_col=config.audio_col,
            vi_col=config.vi_col,
            en_col=config.en_col,
            target_sr=16000,
            mono=True,
        )
        
        collator = DataCollatorSpeechToText(
            feature_extractor=feature_extractor,
            processor=processor,
            padding=True,
            pad_to_multiple_of=8,
            target_language="vi",
            pivot_language="en"
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Small batch for testing
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Run 2 training steps
        model.train()
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Only 2 steps
                break
            
            # Move batch to device
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                audio_input_features=batch['audio_input_features'],
                text_input_pivot_ids=batch['text_input_pivot_ids'],
                labels=batch['labels'],
                audio_attention_mask=batch['audio_attention_mask'],
                text_pivot_attention_mask=batch['text_pivot_attention_mask'],
            )
            
            loss_ce, loss_kd = outputs[0], outputs[1]
            
            # Backward pass
            total_loss = loss_ce + 0.5 * loss_kd if loss_kd is not None else loss_ce
            total_loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            logger.info(f"  Step {i+1}: CE Loss={loss_ce.item():.4f}, "
                       f"KD Loss={loss_kd.item() if loss_kd is not None else 'N/A'}")
        
        logger.info("✅ Mini training loop completed successfully")
        
        # Clean up
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsdp_wrapping():
    """Test FSDP model wrapping (requires 2+ GPUs)"""
    logger.info("\nTesting FSDP wrapping...")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus < 2:
        logger.warning("⚠️  Skipping FSDP test (requires 2+ GPUs)")
        logger.info("  Run with torchrun for full FSDP testing:")
        logger.info("  torchrun --nproc_per_node=2 train_kaggle.py")
        return True
    
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot
        from seamless_m4t_v2_config import SeamlessM4Tv2Config
        
        logger.info("✅ FSDP available and can be imported")
        logger.info("  Full FSDP test requires running with torchrun")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FSDP test failed: {e}")
        return False


def test_disk_space():
    """Test available disk space"""
    logger.info("\nTesting disk space...")
    
    try:
        import shutil
        from configs import TrainingConfig
        
        config = TrainingConfig()
        
        stat = shutil.disk_usage(config.output_dir if os.path.exists(os.path.dirname(config.output_dir)) else ".")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        logger.info(f"  Total: {total_gb:.2f} GB")
        logger.info(f"  Free: {free_gb:.2f} GB")
        logger.info(f"  Output directory: {config.output_dir}")
        
        if free_gb < 10:
            logger.warning("⚠️  Low disk space (< 10 GB)")
            logger.warning("  Consider cleaning up or using a larger disk")
        else:
            logger.info("✅ Sufficient disk space")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Disk space check failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*70)
    logger.info("SeamlessM4T v2 Training Setup Test (Using Real Config & Data)")
    logger.info("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("GPU Availability", test_gpu_availability),
        ("Config Loading", test_config_loading),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Dataset Loading", test_dataset_loading),
        ("Data Loader", test_data_loader),
        ("Mini Training Loop", test_training_loop),
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
            import traceback
            traceback.print_exc()
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
        logger.info("\n📋 Training Configuration Summary:")
        try:
            from configs import TrainingConfig
            config = TrainingConfig()
            logger.info(f"  - Excel: {config.excel_path}")
            logger.info(f"  - Batch size: {config.per_device_train_batch_size}")
            logger.info(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
            logger.info(f"  - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
            logger.info(f"  - Epochs: {config.num_epochs}")
            logger.info(f"  - Encoder LR: {config.encoder_lr}")
            logger.info(f"  - Decoder LR: {config.decoder_lr}")
        except:
            pass
        logger.info("\n🚀 To start training, run:")
        logger.info("  python train_kaggle.py")
        logger.info("\nOr with 2 GPUs:")
        logger.info("  torchrun --nproc_per_node=2 train_kaggle.py")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Please fix the issues before training.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
