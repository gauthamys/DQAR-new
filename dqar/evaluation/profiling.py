"""
Profiling Module

Tools for measuring runtime, VRAM usage, and DQAR-specific
performance metrics during inference.
"""

import torch
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import gc


@dataclass
class TimingResult:
    """Container for timing measurements."""
    total_time_ms: float
    per_step_times_ms: List[float]
    mean_step_time_ms: float
    std_step_time_ms: float


@dataclass
class MemoryResult:
    """Container for memory measurements."""
    peak_allocated_mb: float
    peak_reserved_mb: float
    current_allocated_mb: float
    kv_cache_mb: float


@dataclass
class ProfileResult:
    """Complete profiling result."""
    timing: TimingResult
    memory: MemoryResult
    dqar_stats: Dict[str, Any]
    throughput_samples_per_sec: float


class MemoryTracker:
    """
    Tracks GPU memory usage during inference.
    """

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: CUDA device to track
        """
        self.device = device
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._snapshots: List[Dict[str, float]] = []

    def reset(self):
        """Reset tracking state."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._snapshots.clear()

    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)

        self._snapshots.append({
            "label": label,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "timestamp": time.time(),
        })

        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0}

        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024 ** 2),
        }

    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage."""
        if not torch.cuda.is_available():
            return {"peak_allocated_mb": 0, "peak_reserved_mb": 0}

        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
        }

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all recorded snapshots."""
        return self._snapshots.copy()

    def get_result(self, kv_cache_bytes: int = 0) -> MemoryResult:
        """Get memory result summary."""
        peak = self.get_peak_usage()
        current = self.get_current_usage()

        return MemoryResult(
            peak_allocated_mb=peak["peak_allocated_mb"],
            peak_reserved_mb=peak["peak_reserved_mb"],
            current_allocated_mb=current["allocated_mb"],
            kv_cache_mb=kv_cache_bytes / (1024 ** 2),
        )


class DQARProfiler:
    """
    Profiler for DQAR inference performance.

    Measures:
    - Total and per-step timing
    - Memory usage (peak and average)
    - Attention reuse statistics
    - KV cache efficiency
    """

    def __init__(
        self,
        device: str = "cuda",
        warmup_runs: int = 1,
        sync_cuda: bool = True,
    ):
        """
        Args:
            device: Device to profile on
            warmup_runs: Number of warmup runs before profiling
            sync_cuda: Whether to synchronize CUDA for accurate timing
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.sync_cuda = sync_cuda

        self.memory_tracker = MemoryTracker(device)
        self._step_times: List[float] = []
        self._total_start_time: Optional[float] = None

    def reset(self):
        """Reset profiler state."""
        self.memory_tracker.reset()
        self._step_times.clear()
        self._total_start_time = None

    @contextmanager
    def profile_run(self):
        """Context manager for profiling a complete inference run."""
        self.reset()

        # Warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        self._total_start_time = time.time()
        self.memory_tracker.snapshot("start")

        try:
            yield self
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            self.memory_tracker.snapshot("end")

    @contextmanager
    def profile_step(self, step_idx: int):
        """Context manager for profiling a single diffusion step."""
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()
        self.memory_tracker.snapshot(f"step_{step_idx}_start")

        try:
            yield
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (time.time() - start_time) * 1000
            self._step_times.append(elapsed_ms)
            self.memory_tracker.snapshot(f"step_{step_idx}_end")

    def get_timing_result(self) -> TimingResult:
        """Get timing statistics."""
        if not self._step_times:
            return TimingResult(
                total_time_ms=0,
                per_step_times_ms=[],
                mean_step_time_ms=0,
                std_step_time_ms=0,
            )

        total_time = (time.time() - self._total_start_time) * 1000 if self._total_start_time else 0

        import numpy as np
        times_array = np.array(self._step_times)

        return TimingResult(
            total_time_ms=total_time,
            per_step_times_ms=self._step_times.copy(),
            mean_step_time_ms=float(times_array.mean()),
            std_step_time_ms=float(times_array.std()),
        )

    def get_result(
        self,
        num_samples: int = 1,
        dqar_stats: Optional[Dict[str, Any]] = None,
        kv_cache_bytes: int = 0,
    ) -> ProfileResult:
        """
        Get complete profiling result.

        Args:
            num_samples: Number of samples generated
            dqar_stats: Statistics from DQAR wrapper
            kv_cache_bytes: KV cache memory usage in bytes

        Returns:
            Complete profiling result
        """
        timing = self.get_timing_result()
        memory = self.memory_tracker.get_result(kv_cache_bytes)

        throughput = 0.0
        if timing.total_time_ms > 0:
            throughput = num_samples / (timing.total_time_ms / 1000)

        return ProfileResult(
            timing=timing,
            memory=memory,
            dqar_stats=dqar_stats or {},
            throughput_samples_per_sec=throughput,
        )


def profile_inference(
    sample_fn: Callable,
    num_runs: int = 3,
    warmup_runs: int = 1,
    device: str = "cuda",
    **sample_kwargs,
) -> Dict[str, Any]:
    """
    Profile inference performance over multiple runs.

    Args:
        sample_fn: Sampling function to profile
        num_runs: Number of profiling runs
        warmup_runs: Number of warmup runs
        device: Device to profile on
        **sample_kwargs: Arguments to pass to sample_fn

    Returns:
        Profiling statistics
    """
    profiler = DQARProfiler(device=device, warmup_runs=warmup_runs)

    # Warmup runs
    for _ in range(warmup_runs):
        _ = sample_fn(**sample_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Profiling runs
    results = []
    for run_idx in range(num_runs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        output = sample_fn(**sample_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - start_time) * 1000

        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results.append({
            "run_idx": run_idx,
            "time_ms": elapsed_ms,
            "peak_memory_mb": peak_memory,
        })

    # Aggregate results
    import numpy as np
    times = np.array([r["time_ms"] for r in results])
    memories = np.array([r["peak_memory_mb"] for r in results])

    return {
        "num_runs": num_runs,
        "mean_time_ms": float(times.mean()),
        "std_time_ms": float(times.std()),
        "min_time_ms": float(times.min()),
        "max_time_ms": float(times.max()),
        "mean_peak_memory_mb": float(memories.mean()),
        "std_peak_memory_mb": float(memories.std()),
        "per_run_results": results,
    }


def compare_baselines(
    baseline_fn: Callable,
    dqar_fn: Callable,
    num_runs: int = 3,
    device: str = "cuda",
    **sample_kwargs,
) -> Dict[str, Any]:
    """
    Compare DQAR performance against baseline.

    Args:
        baseline_fn: Baseline sampling function (no DQAR)
        dqar_fn: DQAR-enabled sampling function
        num_runs: Number of runs for each
        device: Device to use
        **sample_kwargs: Arguments for sampling

    Returns:
        Comparison statistics
    """
    baseline_results = profile_inference(
        baseline_fn, num_runs=num_runs, device=device, **sample_kwargs
    )
    dqar_results = profile_inference(
        dqar_fn, num_runs=num_runs, device=device, **sample_kwargs
    )

    # Compute speedup and memory savings
    speedup = baseline_results["mean_time_ms"] / max(dqar_results["mean_time_ms"], 1e-6)
    memory_reduction = (
        (baseline_results["mean_peak_memory_mb"] - dqar_results["mean_peak_memory_mb"]) /
        max(baseline_results["mean_peak_memory_mb"], 1e-6)
    ) * 100

    return {
        "baseline": baseline_results,
        "dqar": dqar_results,
        "speedup": speedup,
        "memory_reduction_pct": memory_reduction,
        "time_saved_ms": baseline_results["mean_time_ms"] - dqar_results["mean_time_ms"],
        "memory_saved_mb": baseline_results["mean_peak_memory_mb"] - dqar_results["mean_peak_memory_mb"],
    }
