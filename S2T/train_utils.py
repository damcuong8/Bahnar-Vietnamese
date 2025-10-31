"""
Utility functions for training SeamlessM4T v2 with FSDP
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_millions": total_params / 1e6,
        "trainable_millions": trainable_params / 1e6,
    }


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return {}
    
    info = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
        
        info[f"gpu_{i}"] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "device_name": torch.cuda.get_device_name(i),
        }
    
    return info


def log_gpu_memory(prefix: str = ""):
    """Log GPU memory usage"""
    if not torch.cuda.is_available():
        return
    
    info = get_gpu_memory_info()
    for gpu_id, mem_info in info.items():
        logger.info(
            f"{prefix}{gpu_id}: {mem_info['allocated_gb']:.2f}GB allocated, "
            f"{mem_info['reserved_gb']:.2f}GB reserved"
        )


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


class MetricsTracker:
    """Track multiple metrics during training"""
    
    def __init__(self, metric_names: List[str]):
        self.meters = {name: AverageMeter(name) for name in metric_names}
    
    def update(self, metrics: Dict[str, float], n: int = 1):
        for name, value in metrics.items():
            if name in self.meters:
                self.meters[name].update(value, n)
    
    def reset(self):
        for meter in self.meters.values():
            meter.reset()
    
    def get_averages(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def get_current(self) -> Dict[str, float]:
        return {name: meter.val for name, meter in self.meters.items()}
    
    def __str__(self):
        return " | ".join(str(meter) for meter in self.meters.values())


def save_training_args(args: Any, output_dir: str):
    """Save training arguments to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert dataclass to dict if needed
    if hasattr(args, '__dataclass_fields__'):
        args_dict = {k: v for k, v in vars(args).items()}
    else:
        args_dict = vars(args)
    
    # Save to JSON
    args_path = os.path.join(output_dir, "training_args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=2, default=str)
    
    logger.info(f"Saved training arguments to {args_path}")


def load_training_args(output_dir: str) -> Dict[str, Any]:
    """Load training arguments from JSON file"""
    args_path = os.path.join(output_dir, "training_args.json")
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Training args not found at {args_path}")
    
    with open(args_path, "r") as f:
        args_dict = json.load(f)
    
    logger.info(f"Loaded training arguments from {args_path}")
    return args_dict


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in output directory"""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if not checkpoints:
        return None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(output_dir, checkpoints[-1])
    
    logger.info(f"Found latest checkpoint: {latest}")
    return latest


def cleanup_checkpoints(output_dir: str, keep_last_n: int = 3):
    """Remove old checkpoints, keeping only the last N"""
    if not os.path.exists(output_dir):
        return
    
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        checkpoint_path = os.path.join(output_dir, checkpoint)
        logger.info(f"Removing old checkpoint: {checkpoint_path}")
        import shutil
        shutil.rmtree(checkpoint_path)


def estimate_remaining_time(
    current_step: int,
    total_steps: int,
    elapsed_time: float,
) -> str:
    """Estimate remaining training time"""
    if current_step == 0:
        return "Unknown"
    
    time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    remaining_seconds = time_per_step * remaining_steps
    
    return format_time(remaining_seconds)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    return optimizer.param_groups[0]["lr"]


def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify that checkpoint can be loaded"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        required_keys = ["model", "optimizer", "scheduler", "epoch", "step"]
        for key in required_keys:
            if key not in checkpoint:
                logger.warning(f"Checkpoint missing key: {key}")
                return False
        
        logger.info(f"Checkpoint verified: {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify checkpoint: {e}")
        return False


def print_training_summary(
    num_epochs: int,
    total_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    model_params: Dict[str, int],
):
    """Print a summary of training configuration"""
    
    effective_batch_size = batch_size * gradient_accumulation_steps * world_size
    
    summary = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   Training Configuration                      ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Epochs:                    {num_epochs:>6}                         ║
    ║ Total Steps:               {total_steps:>6}                         ║
    ║ Batch Size (per device):   {batch_size:>6}                         ║
    ║ Gradient Accumulation:     {gradient_accumulation_steps:>6}                         ║
    ║ World Size (GPUs):         {world_size:>6}                         ║
    ║ Effective Batch Size:      {effective_batch_size:>6}                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Total Parameters:          {model_params['total_millions']:>6.2f}M                      ║
    ║ Trainable Parameters:      {model_params['trainable_millions']:>6.2f}M                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    print(summary)


class ProgressTracker:
    """Track training progress with ETA"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = None
        self.current_step = 0
    
    def start(self):
        import time
        self.start_time = time.time()
        self.current_step = 0
    
    def update(self, step: int):
        self.current_step = step
    
    def get_progress(self) -> Dict[str, Any]:
        if self.start_time is None:
            return {}
        
        import time
        elapsed = time.time() - self.start_time
        progress_pct = (self.current_step / self.total_steps) * 100
        
        eta = estimate_remaining_time(self.current_step, self.total_steps, elapsed)
        
        return {
            "step": self.current_step,
            "total_steps": self.total_steps,
            "progress_pct": progress_pct,
            "elapsed": format_time(elapsed),
            "eta": eta,
        }
    
    def __str__(self):
        progress = self.get_progress()
        if not progress:
            return "Not started"
        
        return (
            f"Step {progress['step']}/{progress['total_steps']} "
            f"({progress['progress_pct']:.1f}%) | "
            f"Elapsed: {progress['elapsed']} | "
            f"ETA: {progress['eta']}"
        )


def create_run_name(config: Any) -> str:
    """Create a descriptive run name for wandb"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key config values
    lr = getattr(config, 'learning_rate', 'unknown')
    bs = getattr(config, 'per_device_train_batch_size', 'unknown')
    strategy = getattr(config, 'sharding_strategy', 'unknown')
    
    return f"seamlessm4t_lr{lr}_bs{bs}_{strategy}_{timestamp}"


def log_system_info():
    """Log system information"""
    import platform
    import sys
    
    logger.info("=" * 60)
    logger.info("System Information:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}")
            logger.info(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        logger.info("  CUDA: Not available")
    
    logger.info("=" * 60)


def check_disk_space(path: str = ".", min_gb: float = 10.0) -> bool:
    """Check if there's enough disk space"""
    import shutil
    
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    
    logger.info(f"Disk space: {free_gb:.2f} GB free")
    
    if free_gb < min_gb:
        logger.warning(f"Low disk space! Only {free_gb:.2f} GB available (minimum: {min_gb} GB)")
        return False
    
    return True


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test time formatting
    print(f"Time formatting: {format_time(3665)}")
    
    # Test average meter
    meter = AverageMeter("loss")
    meter.update(2.5)
    meter.update(2.3)
    meter.update(2.1)
    print(meter)
    
    # Test metrics tracker
    tracker = MetricsTracker(["loss", "accuracy"])
    tracker.update({"loss": 2.5, "accuracy": 0.85})
    tracker.update({"loss": 2.3, "accuracy": 0.87})
    print(tracker)
    
    # Test GPU info
    if torch.cuda.is_available():
        print("\nGPU Memory Info:")
        info = get_gpu_memory_info()
        for gpu, mem in info.items():
            print(f"{gpu}: {mem}")
    
    # Test system info
    log_system_info()
    
    print("\n✅ All utility functions working!")

