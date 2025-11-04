"""
Script to test and track CPU memory usage when loading SeamlessM4T model.
Use this to diagnose OOM issues during model loading.

Usage:
    python test_cpu_memory_load.py
"""

import os
import sys
import logging
import torch
import gc

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from cpu_memory_tracker import CPUMemoryTracker, log_cpu_memory, check_memory_available
from seamless_m4t_v2_config import SeamlessM4Tv2Config
from speech2text_model import SeamlessM4Tv2ForSpeechToTextTrain_Pivot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cpu_memory_load_test.log')
    ]
)
logger = logging.getLogger(__name__)


def test_model_loading_memory():
    """Test CPU memory usage during model loading"""
    
    tracker = CPUMemoryTracker(log_interval=1.0)
    tracker.start_tracking()
    
    # Check if we have enough memory (estimate ~40GB needed for large model)
    logger.info("\n" + "="*70)
    logger.info("CHECKING SYSTEM REQUIREMENTS")
    logger.info("="*70)
    
    if not check_memory_available(40.0):
        logger.error("⚠️  May not have enough RAM to load the full model!")
        logger.info("Consider using a smaller model or enabling CPU offloading")
    
    try:
        # Step 1: Create config
        with tracker.track("create_config"):
            logger.info("Creating model config...")
            config = SeamlessM4Tv2Config()
            logger.info(f"Config created: hidden_size={config.hidden_size}, "
                       f"encoder_layers={config.encoder_layers}")
            tracker.check_memory("After config creation")
        
        # Step 2: Initialize model (empty weights)
        with tracker.track("initialize_model"):
            logger.info("Initializing model with random weights...")
            model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
            tracker.check_memory("After model init")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
        
        # Step 3: Load pretrained weights (this is where OOM usually happens!)
        with tracker.track("load_pretrained_weights"):
            logger.info("\n" + "="*70)
            logger.info("LOADING PRETRAINED WEIGHTS - CRITICAL PHASE")
            logger.info("="*70)
            
            model_name = "facebook/seamless-m4t-v2-large"
            logger.info(f"Loading from: {model_name}")
            
            # Check memory before loading
            tracker.check_memory("Before loading weights")
            
            try:
                # This is where the OOM typically occurs
                stats = model.load_pretrained_weights(model_name)
                
                tracker.check_memory("After loading weights")
                
                logger.info("\n✅ Successfully loaded pretrained weights!")
                logger.info(f"Loading statistics: {stats}")
                
            except Exception as e:
                logger.error(f"\n🚨 ERROR during weight loading: {e}")
                tracker.check_memory("After error")
                raise
        
        # Step 4: Move model to device (if GPU available)
        if torch.cuda.is_available():
            with tracker.track("move_to_gpu"):
                logger.info("Moving model to GPU...")
                device = torch.device("cuda:0")
                model = model.to(device)
                tracker.check_memory("After moving to GPU")
                
                # GPU memory
                from memory_tracker import log_memory_stats
                log_memory_stats("After model.to(device)", device_id=0)
        else:
            logger.info("No GPU available, keeping model on CPU")
        
        # Step 5: Test inference (optional)
        with tracker.track("test_inference"):
            logger.info("Testing inference...")
            tracker.check_memory("Before inference")
            
            # Create dummy input
            batch_size = 1
            seq_len = 1000
            num_banks = 160
            
            dummy_input = torch.randn(batch_size, seq_len, num_banks)
            if torch.cuda.is_available():
                dummy_input = dummy_input.to(device)
            
            # Forward pass
            with torch.no_grad():
                logger.info("Running forward pass...")
                tracker.check_memory("During forward pass")
                
                # You would need to provide proper inputs for your model
                # This is just a placeholder
                
            tracker.check_memory("After inference")
        
        # Log final summary
        tracker.log_summary()
        
        logger.info("\n" + "="*70)
        logger.info("✅ TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n🚨 TEST FAILED: {e}")
        logger.exception("Full traceback:")
        tracker.log_summary()
        return False
    
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        if 'model' in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tracker.force_cleanup()
        log_cpu_memory("After cleanup")


def test_checkpoint_loading_memory(checkpoint_path: str):
    """Test CPU memory when loading from local checkpoint"""
    
    tracker = CPUMemoryTracker()
    tracker.start_tracking()
    
    try:
        with tracker.track("load_from_checkpoint"):
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            tracker.check_memory("After loading checkpoint file")
            
            # Create model
            config = SeamlessM4Tv2Config()
            model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
            tracker.check_memory("After model init")
            
            # Load state dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            tracker.check_memory("After loading state dict")
            
            del checkpoint, state_dict
            gc.collect()
            tracker.check_memory("After cleanup")
        
        tracker.log_summary()
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.exception("Traceback:")
        tracker.log_summary()
        return False


def analyze_memory_bottlenecks():
    """Analyze which components use most memory"""
    
    logger.info("\n" + "="*70)
    logger.info("ANALYZING MEMORY BOTTLENECKS")
    logger.info("="*70)
    
    tracker = CPUMemoryTracker()
    tracker.start_tracking()
    
    config = SeamlessM4Tv2Config()
    
    # Test each component separately
    components = [
        ("speech_encoder", lambda: __import__('speech2text_model').SeamlessM4Tv2SpeechEncoder(config)),
        ("text_encoder", lambda: __import__('speech2text_model').SeamlessM4Tv2Encoder(config)),
        ("text_decoder", lambda: __import__('speech2text_model').SeamlessM4Tv2Decoder(config)),
    ]
    
    for name, create_fn in components:
        with tracker.track(f"create_{name}"):
            logger.info(f"\nCreating {name}...")
            component = create_fn()
            
            params = sum(p.numel() for p in component.parameters())
            logger.info(f"{name} parameters: {params:,} ({params/1e6:.1f}M)")
            
            tracker.check_memory(f"After {name}")
            
            # Cleanup
            del component
            gc.collect()
    
    tracker.log_summary()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CPU memory usage during model loading")
    parser.add_argument(
        "--test",
        type=str,
        default="full",
        choices=["full", "checkpoint", "analyze"],
        help="Type of test to run"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (for checkpoint test)"
    )
    
    args = parser.parse_args()
    
    if args.test == "full":
        logger.info("Running full model loading test...")
        success = test_model_loading_memory()
    elif args.test == "checkpoint":
        if args.checkpoint is None:
            logger.error("Please provide --checkpoint path")
            sys.exit(1)
        logger.info(f"Testing checkpoint loading from {args.checkpoint}")
        success = test_checkpoint_loading_memory(args.checkpoint)
    elif args.test == "analyze":
        logger.info("Analyzing memory bottlenecks...")
        analyze_memory_bottlenecks()
        success = True
    
    sys.exit(0 if success else 1)
