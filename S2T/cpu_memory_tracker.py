"""
CPU Memory tracking utilities for monitoring RAM usage during model loading.
Useful for diagnosing OOM issues when loading large models.
"""

import logging
import psutil
import os
import gc
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class CPUMemorySnapshot:
    """Snapshot of CPU memory usage at a specific point"""
    used_gb: float
    available_gb: float
    total_gb: float
    percent: float
    process_rss_gb: float  # Resident Set Size (physical memory used by process)
    process_vms_gb: float  # Virtual Memory Size
    timestamp: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            "used_gb": self.used_gb,
            "available_gb": self.available_gb,
            "total_gb": self.total_gb,
            "percent": self.percent,
            "process_rss_gb": self.process_rss_gb,
            "process_vms_gb": self.process_vms_gb,
        }


class CPUMemoryTracker:
    """
    Track CPU memory (RAM) usage during model loading and training.
    
    Usage:
        tracker = CPUMemoryTracker()
        tracker.start_tracking()
        
        with tracker.track("load_model"):
            model = load_model()
        
        tracker.log_summary()
    """
    
    def __init__(self, log_interval: float = 0.5):
        """
        Initialize CPU memory tracker.
        
        Args:
            log_interval: Minimum time interval (seconds) between logs for continuous tracking
        """
        self.log_interval = log_interval
        self.process = psutil.Process(os.getpid())
        
        # Memory snapshots
        self.snapshots: Dict[str, CPUMemorySnapshot] = {}
        self.phase_deltas: Dict[str, Dict[str, float]] = {}
        
        # Initial memory state
        self.initial_snapshot: Optional[CPUMemorySnapshot] = None
        self.current_phase: Optional[str] = None
        self.phase_start_snapshot: Optional[CPUMemorySnapshot] = None
    
    def _get_memory_snapshot(self) -> CPUMemorySnapshot:
        """Get current CPU memory snapshot"""
        # System-wide memory
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1024**3
        available_gb = mem.available / 1024**3
        total_gb = mem.total / 1024**3
        percent = mem.percent
        
        # Process-specific memory
        proc_mem = self.process.memory_info()
        process_rss_gb = proc_mem.rss / 1024**3  # Physical memory
        process_vms_gb = proc_mem.vms / 1024**3  # Virtual memory
        
        return CPUMemorySnapshot(
            used_gb=used_gb,
            available_gb=available_gb,
            total_gb=total_gb,
            percent=percent,
            process_rss_gb=process_rss_gb,
            process_vms_gb=process_vms_gb,
            timestamp=time.time(),
        )
    
    def start_tracking(self):
        """Start tracking - take initial snapshot"""
        gc.collect()  # Clean up before starting
        self.initial_snapshot = self._get_memory_snapshot()
        
        logger.info("=" * 70)
        logger.info("CPU MEMORY TRACKING STARTED")
        logger.info("=" * 70)
        self._log_snapshot("Initial", self.initial_snapshot)
    
    @contextmanager
    def track(self, phase_name: str, log_continuous: bool = False):
        """
        Context manager to track CPU memory for a specific phase.
        
        Args:
            phase_name: Name of the phase (e.g., "load_model", "forward")
            log_continuous: If True, log memory periodically during the phase
        
        Usage:
            with tracker.track("load_model", log_continuous=True):
                model = load_model()
        """
        # Start of phase
        gc.collect()  # Clean up before tracking
        self.current_phase = phase_name
        self.phase_start_snapshot = self._get_memory_snapshot()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting: {phase_name}")
        logger.info(f"{'='*70}")
        self._log_snapshot(f"{phase_name} - Start", self.phase_start_snapshot)
        
        last_log_time = time.time()
        
        try:
            # Execute the tracked code
            if log_continuous:
                # For continuous tracking, we can't yield and monitor at the same time
                # User needs to call check_memory() manually if needed
                pass
            
            yield self  # Allow user to call check_memory() if needed
            
        finally:
            # End of phase
            gc.collect()  # Clean up before final measurement
            phase_end_snapshot = self._get_memory_snapshot()
            
            # Calculate deltas
            delta_used = phase_end_snapshot.used_gb - self.phase_start_snapshot.used_gb
            delta_process = phase_end_snapshot.process_rss_gb - self.phase_start_snapshot.process_rss_gb
            delta_vms = phase_end_snapshot.process_vms_gb - self.phase_start_snapshot.process_vms_gb
            
            self.snapshots[phase_name] = phase_end_snapshot
            self.phase_deltas[phase_name] = {
                "system_used": delta_used,
                "process_rss": delta_process,
                "process_vms": delta_vms,
            }
            
            # Log phase memory
            logger.info(f"\n{'='*70}")
            logger.info(f"Finished: {phase_name}")
            logger.info(f"{'='*70}")
            self._log_phase(phase_name, self.phase_start_snapshot, phase_end_snapshot)
            
            self.current_phase = None
            self.phase_start_snapshot = None
    
    def check_memory(self, label: str = ""):
        """
        Check and log current memory during a tracked phase.
        Useful for continuous monitoring.
        
        Args:
            label: Additional label for the log
        """
        snapshot = self._get_memory_snapshot()
        phase_name = self.current_phase or "unknown"
        full_label = f"{phase_name} - {label}" if label else phase_name
        self._log_snapshot(full_label, snapshot)
        
        # Warn if memory is getting critical
        if snapshot.percent > 90:
            logger.warning(f"⚠️  WARNING: CPU memory usage is {snapshot.percent:.1f}%! OOM risk!")
        elif snapshot.percent > 80:
            logger.warning(f"⚠️  CAUTION: CPU memory usage is {snapshot.percent:.1f}%")
    
    def _log_snapshot(self, label: str, snapshot: CPUMemorySnapshot):
        """Log a memory snapshot"""
        logger.info(
            f"[{label}]\n"
            f"  System: {snapshot.used_gb:.2f}GB used / {snapshot.total_gb:.2f}GB total "
            f"({snapshot.percent:.1f}% used, {snapshot.available_gb:.2f}GB available)\n"
            f"  Process: RSS={snapshot.process_rss_gb:.2f}GB, VMS={snapshot.process_vms_gb:.2f}GB"
        )
    
    def _log_phase(self, phase_name: str, start: CPUMemorySnapshot, end: CPUMemorySnapshot):
        """Log memory usage for a phase"""
        delta_used = end.used_gb - start.used_gb
        delta_process = end.process_rss_gb - start.process_rss_gb
        delta_vms = end.process_vms_gb - start.process_vms_gb
        
        delta_sign_sys = "+" if delta_used >= 0 else ""
        delta_sign_proc = "+" if delta_process >= 0 else ""
        delta_sign_vms = "+" if delta_vms >= 0 else ""
        
        logger.info(
            f"Memory Changes:\n"
            f"  System Used:  {start.used_gb:.2f}GB → {end.used_gb:.2f}GB "
            f"({delta_sign_sys}{delta_used:.2f}GB)\n"
            f"  Process RSS:  {start.process_rss_gb:.2f}GB → {end.process_rss_gb:.2f}GB "
            f"({delta_sign_proc}{delta_process:.2f}GB)\n"
            f"  Process VMS:  {start.process_vms_gb:.2f}GB → {end.process_vms_gb:.2f}GB "
            f"({delta_sign_vms}{delta_vms:.2f}GB)\n"
            f"  System Usage: {start.percent:.1f}% → {end.percent:.1f}%"
        )
        
        # Warn if significant increase
        if delta_process > 5.0:
            logger.warning(f"⚠️  Large memory increase: +{delta_process:.2f}GB")
        
        if end.percent > 90:
            logger.error(f"🚨 CRITICAL: Memory usage at {end.percent:.1f}%! OOM imminent!")
    
    def log_summary(self):
        """Log summary of all tracked phases"""
        logger.info("\n" + "=" * 70)
        logger.info("CPU MEMORY TRACKING SUMMARY")
        logger.info("=" * 70)
        
        if self.initial_snapshot:
            logger.info(f"\nInitial State:")
            logger.info(f"  System Used:  {self.initial_snapshot.used_gb:.2f}GB ({self.initial_snapshot.percent:.1f}%)")
            logger.info(f"  Process RSS:  {self.initial_snapshot.process_rss_gb:.2f}GB")
        
        # Log each phase
        logger.info(f"\nPhase Summary:")
        for phase_name in self.snapshots:
            snapshot = self.snapshots[phase_name]
            deltas = self.phase_deltas.get(phase_name, {})
            
            delta_used = deltas.get("system_used", 0.0)
            delta_process = deltas.get("process_rss", 0.0)
            
            delta_sign_sys = "+" if delta_used >= 0 else ""
            delta_sign_proc = "+" if delta_process >= 0 else ""
            
            logger.info(
                f"  {phase_name:25s}: "
                f"System={snapshot.used_gb:6.2f}GB ({delta_sign_sys}{delta_used:6.2f}GB), "
                f"Process={snapshot.process_rss_gb:6.2f}GB ({delta_sign_proc}{delta_process:6.2f}GB)"
            )
        
        # Find peak memory
        if self.snapshots:
            max_snapshot = max(self.snapshots.values(), key=lambda s: s.process_rss_gb)
            logger.info(f"\nPeak Process Memory: {max_snapshot.process_rss_gb:.2f}GB")
            logger.info(f"Peak System Usage: {max_snapshot.percent:.1f}%")
        
        logger.info("=" * 70 + "\n")
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage as dictionary"""
        snapshot = self._get_memory_snapshot()
        return snapshot.to_dict()
    
    def force_cleanup(self):
        """Force garbage collection to free memory"""
        logger.info("Forcing garbage collection...")
        before = self._get_memory_snapshot()
        gc.collect()
        after = self._get_memory_snapshot()
        freed = before.process_rss_gb - after.process_rss_gb
        logger.info(f"Freed {freed:.2f}GB of process memory")


def log_cpu_memory(label: str = ""):
    """
    Quick utility to log current CPU memory usage.
    
    Args:
        label: Label for the log
    """
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info()
    
    label_str = f"[{label}] " if label else ""
    logger.info(
        f"{label_str}CPU Memory: "
        f"System={mem.used/1024**3:.2f}GB/{mem.total/1024**3:.2f}GB ({mem.percent:.1f}%), "
        f"Process={proc_mem.rss/1024**3:.2f}GB"
    )


def check_memory_available(required_gb: float) -> bool:
    """
    Check if enough CPU memory is available.
    
    Args:
        required_gb: Required memory in GB
    
    Returns:
        True if enough memory is available
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / 1024**3
    
    if available_gb < required_gb:
        logger.warning(
            f"Insufficient memory: {available_gb:.2f}GB available, "
            f"{required_gb:.2f}GB required"
        )
        return False
    return True


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    tracker = CPUMemoryTracker()
    tracker.start_tracking()
    
    # Simulate model loading
    with tracker.track("simulate_model_load", log_continuous=True):
        import numpy as np
        
        # Simulate loading large arrays (like model weights)
        arrays = []
        for i in range(5):
            logger.info(f"Loading chunk {i+1}/5...")
            arr = np.random.randn(100_000_000)  # ~800MB per array
            arrays.append(arr)
            tracker.check_memory(f"After chunk {i+1}")
            time.sleep(0.5)
    
    tracker.log_summary()
