"""
Memory tracking utilities for monitoring GPU memory usage during training.
Provides detailed tracking at different phases of training (forward, backward, optimizer step).
"""

import logging
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point"""
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    free_gb: float
    total_gb: float
    device_id: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "max_allocated_gb": self.max_allocated_gb,
            "free_gb": self.free_gb,
            "total_gb": self.total_gb,
        }


class MemoryTracker:
    """
    Track GPU memory usage during training.
    
    Usage:
        tracker = MemoryTracker(rank=0, log_to_wandb=True)
        tracker.start_tracking()
        
        with tracker.track("forward"):
            outputs = model(inputs)
        
        tracker.log_summary()
    """
    
    def __init__(
        self,
        rank: int = 0,
        log_to_wandb: bool = False,
        device_id: Optional[int] = None,
    ):
        """
        Initialize memory tracker.
        
        Args:
            rank: Process rank (only rank 0 logs)
            log_to_wandb: Whether to log to wandb
            device_id: GPU device ID (None = use current device)
        """
        self.rank = rank
        self.log_to_wandb = log_to_wandb
        self.device_id = device_id if device_id is not None else torch.cuda.current_device()
        
        # Memory snapshots
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.phase_deltas: Dict[str, float] = {}
        
        # Initial memory state
        self.initial_snapshot: Optional[MemorySnapshot] = None
        self.current_phase: Optional[str] = None
        self.phase_start_snapshot: Optional[MemorySnapshot] = None
        
        # Wandb import check
        self.wandb_available = False
        if log_to_wandb:
            try:
                import wandb
                self.wandb_available = True
            except ImportError:
                logger.warning("wandb not available, memory tracking won't log to wandb")
    
    def _get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot"""
        if not torch.cuda.is_available():
            return MemorySnapshot(0.0, 0.0, 0.0, 0.0, 0.0, self.device_id)
        
        torch.cuda.synchronize(self.device_id)
        
        allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1024**3  # GB
        
        # Get total memory
        props = torch.cuda.get_device_properties(self.device_id)
        total = props.total_memory / 1024**3  # GB
        free = total - reserved
        
        return MemorySnapshot(
            allocated_gb=allocated,
            reserved_gb=reserved,
            max_allocated_gb=max_allocated,
            free_gb=free,
            total_gb=total,
            device_id=self.device_id,
        )
    
    def start_tracking(self):
        """Start tracking - take initial snapshot"""
        if self.rank != 0:
            return
        
        torch.cuda.reset_peak_memory_stats(self.device_id)
        self.initial_snapshot = self._get_memory_snapshot()
        
        logger.info("=" * 70)
        logger.info("MEMORY TRACKING STARTED")
        logger.info("=" * 70)
        self._log_snapshot("Initial", self.initial_snapshot)
    
    @contextmanager
    def track(self, phase_name: str):
        """
        Context manager to track memory for a specific phase.
        
        Args:
            phase_name: Name of the phase (e.g., "forward", "backward", "optimizer_step")
        
        Usage:
            with tracker.track("forward"):
                outputs = model(inputs)
        """
        if self.rank != 0:
            yield
            return
        
        # Start of phase
        self.current_phase = phase_name
        self.phase_start_snapshot = self._get_memory_snapshot()
        
        try:
            yield
        finally:
            # End of phase
            phase_end_snapshot = self._get_memory_snapshot()
            delta = phase_end_snapshot.allocated_gb - self.phase_start_snapshot.allocated_gb
            
            self.snapshots[phase_name] = phase_end_snapshot
            self.phase_deltas[phase_name] = delta
            
            # Log phase memory
            self._log_phase(phase_name, self.phase_start_snapshot, phase_end_snapshot, delta)
            
            self.current_phase = None
            self.phase_start_snapshot = None
    
    def _log_snapshot(self, label: str, snapshot: MemorySnapshot):
        """Log a memory snapshot"""
        logger.info(
            f"[{label}] GPU {self.device_id} Memory: "
            f"Allocated={snapshot.allocated_gb:.2f}GB, "
            f"Reserved={snapshot.reserved_gb:.2f}GB, "
            f"Max={snapshot.max_allocated_gb:.2f}GB, "
            f"Free={snapshot.free_gb:.2f}GB/{snapshot.total_gb:.2f}GB"
        )
    
    def _log_phase(self, phase_name: str, start: MemorySnapshot, end: MemorySnapshot, delta: float):
        """Log memory usage for a phase"""
        delta_sign = "+" if delta >= 0 else ""
        logger.info(
            f"[{phase_name}] Memory: "
            f"{start.allocated_gb:.2f}GB → {end.allocated_gb:.2f}GB "
            f"({delta_sign}{delta:.2f}GB) | "
            f"Peak: {end.max_allocated_gb:.2f}GB"
        )
        
        # Log to wandb
        if self.log_to_wandb and self.wandb_available:
            import wandb
            wandb.log({
                f"memory/{phase_name}_allocated_gb": end.allocated_gb,
                f"memory/{phase_name}_reserved_gb": end.reserved_gb,
                f"memory/{phase_name}_peak_gb": end.max_allocated_gb,
                f"memory/{phase_name}_delta_gb": delta,
                f"memory/{phase_name}_free_gb": end.free_gb,
            })
    
    def log_summary(self, step: Optional[int] = None):
        """Log summary of all tracked phases"""
        if self.rank != 0:
            return
        
        logger.info("=" * 70)
        logger.info(f"MEMORY SUMMARY{' (Step ' + str(step) + ')' if step is not None else ''}")
        logger.info("=" * 70)
        
        if self.initial_snapshot:
            logger.info(f"Initial Memory: {self.initial_snapshot.allocated_gb:.2f}GB")
        
        # Log each phase
        for phase_name in self.snapshots:
            snapshot = self.snapshots[phase_name]
            delta = self.phase_deltas.get(phase_name, 0.0)
            delta_sign = "+" if delta >= 0 else ""
            logger.info(
                f"  {phase_name:20s}: "
                f"{snapshot.allocated_gb:6.2f}GB "
                f"({delta_sign}{delta:6.2f}GB) | "
                f"Peak: {snapshot.max_allocated_gb:6.2f}GB"
            )
        
        # Log peak memory
        if self.snapshots:
            max_snapshot = max(self.snapshots.values(), key=lambda s: s.max_allocated_gb)
            logger.info(f"\nPeak Memory Usage: {max_snapshot.max_allocated_gb:.2f}GB")
        
        # Log to wandb
        if self.log_to_wandb and self.wandb_available and step is not None:
            import wandb
            log_dict = {}
            for phase_name, snapshot in self.snapshots.items():
                log_dict[f"memory_summary/{phase_name}_allocated_gb"] = snapshot.allocated_gb
                log_dict[f"memory_summary/{phase_name}_peak_gb"] = snapshot.max_allocated_gb
                log_dict[f"memory_summary/{phase_name}_delta_gb"] = self.phase_deltas.get(phase_name, 0.0)
            
            if self.initial_snapshot:
                log_dict["memory_summary/initial_gb"] = self.initial_snapshot.allocated_gb
            
            wandb.log(log_dict, step=step)
        
        logger.info("=" * 70)
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage as dictionary"""
        snapshot = self._get_memory_snapshot()
        return snapshot.to_dict()
    
    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)


def get_memory_stats(device_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get detailed memory statistics for a GPU.
    
    Args:
        device_id: GPU device ID (None = use current device)
    
    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {}
    
    if device_id is None:
        device_id = torch.cuda.current_device()
    
    torch.cuda.synchronize(device_id)
    
    # Basic stats
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3  # GB
    
    # Device properties
    props = torch.cuda.get_device_properties(device_id)
    total = props.total_memory / 1024**3  # GB
    free = total - reserved
    
    # Detailed stats (if available)
    try:
        stats = torch.cuda.memory_stats(device_id)
        active_bytes = stats.get("active_bytes.all.current", 0) / 1024**3  # GB
        inactive_bytes = stats.get("inactive_bytes.all.current", 0) / 1024**3  # GB
        allocated_bytes = stats.get("allocated_bytes.all.current", 0) / 1024**3  # GB
    except Exception:
        active_bytes = 0.0
        inactive_bytes = 0.0
        allocated_bytes = allocated
    
    return {
        "device_id": device_id,
        "device_name": torch.cuda.get_device_name(device_id),
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "free_gb": free,
        "total_gb": total,
        "active_bytes_gb": active_bytes,
        "inactive_bytes_gb": inactive_bytes,
        "allocated_bytes_gb": allocated_bytes,
        "utilization_pct": (reserved / total * 100) if total > 0 else 0.0,
    }


def log_memory_stats(prefix: str = "", device_id: Optional[int] = None, rank: int = 0):
    """
    Log current memory statistics.
    
    Args:
        prefix: Prefix for log message
        device_id: GPU device ID (None = use current device)
        rank: Process rank (only rank 0 logs)
    """
    if rank != 0:
        return
    
    stats = get_memory_stats(device_id)
    if not stats:
        return
    
    prefix_str = f"[{prefix}] " if prefix else ""
    logger.info(
        f"{prefix_str}GPU {stats['device_id']} ({stats['device_name']}): "
        f"Allocated={stats['allocated_gb']:.2f}GB, "
        f"Reserved={stats['reserved_gb']:.2f}GB, "
        f"Max={stats['max_allocated_gb']:.2f}GB, "
        f"Free={stats['free_gb']:.2f}GB/{stats['total_gb']:.2f}GB "
        f"({stats['utilization_pct']:.1f}% used)"
    )


def print_memory_summary(device_id: Optional[int] = None, rank: int = 0):
    """
    Print detailed memory summary.
    
    Args:
        device_id: GPU device ID (None = use current device)
        rank: Process rank (only rank 0 prints)
    """
    if rank != 0:
        return
    
    stats = get_memory_stats(device_id)
    if not stats:
        logger.warning("CUDA not available, cannot print memory summary")
        return
    
    logger.info("=" * 70)
    logger.info(f"GPU {stats['device_id']} Memory Summary: {stats['device_name']}")
    logger.info("=" * 70)
    logger.info(f"Total Memory:     {stats['total_gb']:.2f} GB")
    logger.info(f"Reserved:         {stats['reserved_gb']:.2f} GB ({stats['utilization_pct']:.1f}%)")
    logger.info(f"Allocated:        {stats['allocated_gb']:.2f} GB")
    logger.info(f"Free:             {stats['free_gb']:.2f} GB")
    logger.info(f"Peak Allocated:   {stats['max_allocated_gb']:.2f} GB")
    if stats.get('active_bytes_gb', 0) > 0:
        logger.info(f"Active Bytes:     {stats['active_bytes_gb']:.2f} GB")
        logger.info(f"Inactive Bytes:   {stats['inactive_bytes_gb']:.2f} GB")
    logger.info("=" * 70)

