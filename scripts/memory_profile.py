#!/usr/bin/env python3
"""
DQAR Memory Profiling Benchmark

Profiles GPU memory usage for DQAR vs baseline across inference steps.
Tracks peak memory, cache memory, and per-timestep memory allocation.

Usage:
    python scripts/memory_profile.py --warmup-fraction 0.4 --layer-fraction 0.33
    python scripts/memory_profile.py --warmup-fraction 0.2 --layer-fraction 1.0 --detailed
    python scripts/memory_profile.py --warmup-fraction 0.4 --layer-fraction 0.33 --output-dir results/memory
"""

import argparse
import torch
import time
import gc
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.policy.layer_scheduler import LayerScheduler
from dqar.utils import seed_everything, get_device


@dataclass
class MemorySnapshot:
    """Memory snapshot at a specific point."""
    timestamp: float
    step: int
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    cache_size_mb: float


@dataclass
class MemoryProfile:
    """Complete memory profile for a run."""
    config_name: str
    warmup_fraction: float
    layer_fraction: float
    num_steps: int

    # Summary stats
    peak_allocated_mb: float
    peak_reserved_mb: float
    avg_allocated_mb: float
    cache_memory_mb: float

    # Per-step data
    snapshots: List[Dict]

    # Timing
    total_time_s: float
    time_per_step_ms: float


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
            "peak": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    return {"allocated": 0, "reserved": 0, "peak": 0}


def reset_memory_stats():
    """Reset memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def patch_layer_schedule(warmup_fraction: float, max_layers_fraction: float):
    """Temporarily patch the layer scheduler's parameters."""
    original_linear = LayerScheduler._linear_schedule
    original_reverse = LayerScheduler._linear_reverse_schedule

    def patched_linear_schedule(self, progress: float) -> List[int]:
        # Handle 100% warmup (baseline) - never reuse any layers
        if warmup_fraction >= 1.0:
            return []
        if progress < warmup_fraction:
            return []
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        max_reusable = int(self.config.num_layers * max_layers_fraction)
        num_reusable = int(adjusted_progress * max_reusable)
        return list(range(num_reusable))

    def patched_linear_reverse_schedule(self, progress: float) -> List[int]:
        # Handle 100% warmup (baseline) - never reuse any layers
        if warmup_fraction >= 1.0:
            return []
        if progress < warmup_fraction:
            return []
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        max_reusable = int(self.config.num_layers * max_layers_fraction)
        num_reusable = int(adjusted_progress * max_reusable)
        if num_reusable == 0:
            return []
        start_idx = self.config.num_layers - num_reusable
        return list(range(start_idx, self.config.num_layers))

    LayerScheduler._linear_schedule = patched_linear_schedule
    LayerScheduler._linear_reverse_schedule = patched_linear_reverse_schedule
    return (original_linear, original_reverse)


def restore_layer_schedule(original_methods):
    """Restore the original schedule methods."""
    original_linear, original_reverse = original_methods
    LayerScheduler._linear_schedule = original_linear
    LayerScheduler._linear_reverse_schedule = original_reverse


def profile_inference(
    wrapper: DQARDiTWrapper,
    sampler: DQARDDIMSampler,
    num_steps: int,
    seed: int,
    class_label: int,
    config_name: str,
    warmup_fraction: float,
    layer_fraction: float,
    is_baseline: bool = False,
    detailed: bool = False,
) -> MemoryProfile:
    """Profile memory usage during a single inference run."""
    device = get_device()
    snapshots = []

    # Patch schedule (for baseline, use 100% warmup to disable all reuse)
    original_method = None
    if is_baseline:
        # 100% warmup = all steps are warmup = no layers reused
        original_method = patch_layer_schedule(1.0, 0.0)
    else:
        original_method = patch_layer_schedule(warmup_fraction, layer_fraction)
    if wrapper.manager and wrapper.manager.layer_scheduler:
        wrapper.manager.layer_scheduler._build_schedule()

    reset_memory_stats()
    seed_everything(seed)
    wrapper.reset()

    # Track memory at each step using a hook
    step_memory = []
    start_time = time.perf_counter()

    # Create class label tensor
    class_labels_tensor = torch.tensor([class_label], device=device)

    # Get initial memory
    initial_stats = get_memory_stats()

    with torch.no_grad():
        # Run inference with memory tracking
        latents = sampler.sample(
            batch_size=1,
            class_labels=class_labels_tensor,
            progress_bar=False,
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Get final memory stats
    final_stats = get_memory_stats()

    # Get cache memory if available
    cache_memory = 0.0
    if wrapper.manager and wrapper.manager.kv_cache:
        cache_memory = wrapper.manager.kv_cache.get_memory_usage()["total_mb"]

    # Get detailed per-step memory if requested
    if detailed and wrapper.manager:
        # Run again with step-by-step tracking
        reset_memory_stats()
        seed_everything(seed)
        wrapper.reset()

        # Manual step tracking
        scheduler = sampler.scheduler
        scheduler.set_timesteps(num_steps)
        timesteps = scheduler.timesteps

        # Initialize latents
        latent_shape = (1, 4, 32, 32)  # DiT-XL-2-256 latent shape
        latents = torch.randn(latent_shape, device=device, dtype=torch.float16)
        latents = latents * scheduler.init_noise_sigma

        step_start = time.perf_counter()
        for i, t in enumerate(timesteps):
            # Get memory before step
            pre_stats = get_memory_stats()

            # Run single step
            with torch.no_grad():
                model_output = wrapper(
                    latents,
                    t.unsqueeze(0).to(device),
                    class_labels_tensor,
                )
                latents = scheduler.step(model_output, t, latents).prev_sample

            # Get memory after step
            post_stats = get_memory_stats()

            # Get current cache size
            current_cache = 0.0
            if wrapper.manager and wrapper.manager.kv_cache:
                current_cache = wrapper.manager.kv_cache.get_memory_usage()["total_mb"]

            snapshot = MemorySnapshot(
                timestamp=time.perf_counter() - step_start,
                step=i,
                allocated_mb=post_stats["allocated"],
                reserved_mb=post_stats["reserved"],
                peak_mb=post_stats["peak"],
                cache_size_mb=current_cache,
            )
            snapshots.append(asdict(snapshot))

    # Restore original schedule
    restore_layer_schedule(original_method)

    # Calculate averages
    if snapshots:
        avg_allocated = sum(s["allocated_mb"] for s in snapshots) / len(snapshots)
    else:
        avg_allocated = final_stats["allocated"]

    profile = MemoryProfile(
        config_name=config_name,
        warmup_fraction=warmup_fraction,
        layer_fraction=layer_fraction,
        num_steps=num_steps,
        peak_allocated_mb=final_stats["peak"],
        peak_reserved_mb=final_stats["reserved"],
        avg_allocated_mb=avg_allocated,
        cache_memory_mb=cache_memory,
        snapshots=snapshots,
        total_time_s=total_time,
        time_per_step_ms=(total_time / num_steps) * 1000,
    )

    return profile


def print_memory_comparison(baseline: MemoryProfile, dqar: MemoryProfile):
    """Print memory comparison between baseline and DQAR."""
    print()
    print("=" * 70)
    print("MEMORY PROFILE COMPARISON")
    print("=" * 70)
    print()

    print(f"Configuration: {dqar.warmup_fraction*100:.0f}% warmup, {dqar.layer_fraction*100:.0f}% layers")
    print(f"Inference steps: {dqar.num_steps}")
    print()

    print(f"{'Metric':<25} {'Baseline':>12} {'DQAR':>12} {'Delta':>12}")
    print("-" * 63)

    # Peak allocated memory
    peak_delta = dqar.peak_allocated_mb - baseline.peak_allocated_mb
    print(f"{'Peak Allocated (MB)':<25} {baseline.peak_allocated_mb:>12.1f} {dqar.peak_allocated_mb:>12.1f} {peak_delta:>+12.1f}")

    # Peak reserved memory
    reserved_delta = dqar.peak_reserved_mb - baseline.peak_reserved_mb
    print(f"{'Peak Reserved (MB)':<25} {baseline.peak_reserved_mb:>12.1f} {dqar.peak_reserved_mb:>12.1f} {reserved_delta:>+12.1f}")

    # Cache memory (DQAR only)
    print(f"{'Cache Memory (MB)':<25} {0:>12.1f} {dqar.cache_memory_mb:>12.1f} {dqar.cache_memory_mb:>+12.1f}")

    # Total overhead
    total_overhead = peak_delta + dqar.cache_memory_mb
    print(f"{'Total Overhead (MB)':<25} {'-':>12} {'-':>12} {total_overhead:>+12.1f}")

    print("-" * 63)

    # Timing
    time_delta = dqar.total_time_s - baseline.total_time_s
    speedup = baseline.total_time_s / dqar.total_time_s
    print(f"{'Inference Time (s)':<25} {baseline.total_time_s:>12.2f} {dqar.total_time_s:>12.2f} {time_delta:>+12.2f}")
    print(f"{'Time per Step (ms)':<25} {baseline.time_per_step_ms:>12.1f} {dqar.time_per_step_ms:>12.1f} {dqar.time_per_step_ms - baseline.time_per_step_ms:>+12.1f}")
    print(f"{'Speedup':<25} {'1.00x':>12} {speedup:>11.2f}x {'-':>12}")

    print()

    # Memory efficiency
    print("Memory Efficiency Analysis:")
    print("-" * 40)
    overhead_pct = (total_overhead / baseline.peak_allocated_mb) * 100
    print(f"  Memory overhead: {total_overhead:.1f} MB ({overhead_pct:.1f}% of baseline)")
    print(f"  Cache utilization: {dqar.cache_memory_mb:.1f} MB for attention outputs")

    if speedup > 1.0:
        mb_per_speedup = total_overhead / (speedup - 1)
        print(f"  Memory cost per 1% speedup: {mb_per_speedup / 100:.2f} MB")

    print()


def main():
    parser = argparse.ArgumentParser(description="DQAR Memory Profiling Benchmark")

    # Required arguments
    parser.add_argument("--warmup-fraction", type=float, required=True,
                        help="Warmup fraction (0.0-1.0)")
    parser.add_argument("--layer-fraction", type=float, required=True,
                        help="Layer fraction (0.0-1.0)")

    # Optional arguments
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps (default: 50)")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="Model to benchmark")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--class-label", type=int, default=207,
                        help="ImageNet class label (default: 207, golden retriever)")
    parser.add_argument("--detailed", action="store_true",
                        help="Capture per-step memory snapshots")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for JSON results")
    parser.add_argument("--reverse", action="store_true",
                        help="Use reverse scheduling (deep layers first)")

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.warmup_fraction <= 1.0:
        parser.error("--warmup-fraction must be between 0.0 and 1.0")
    if not 0.0 <= args.layer_fraction <= 1.0:
        parser.error("--layer-fraction must be between 0.0 and 1.0")

    device = get_device()
    device_name = "Unknown"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    print()
    print("=" * 70)
    print("DQAR MEMORY PROFILER")
    print("=" * 70)
    print(f"Device: {device_name}")
    print()
    print("Configuration:")
    print(f"  Warmup Fraction:  {args.warmup_fraction:.0%}")
    print(f"  Layer Fraction:   {args.layer_fraction:.0%}")
    print(f"  Schedule Mode:    {'reverse (deep layers)' if args.reverse else 'linear (shallow layers)'}")
    print(f"  Num Steps:        {args.num_steps}")
    print(f"  Detailed Mode:    {'Yes' if args.detailed else 'No'}")
    print()

    # Load model
    print("Loading model...")
    from diffusers import DiTPipeline
    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to(device)

    model = pipe.transformer
    scheduler = pipe.scheduler
    print(f"Model loaded: {args.model}")
    print()

    # Create DQAR wrapper
    schedule_type = "linear_reverse" if args.reverse else "linear"
    dqar_config = DQARConfig(
        quantization_bits=16,
        use_layer_scheduling=True,
        schedule_type=schedule_type,
    )

    wrapper = DQARDiTWrapper(model, dqar_config)
    sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    # Profile baseline (100% warmup = no reuse)
    print("Profiling baseline (no attention reuse)...")
    baseline_profile = profile_inference(
        wrapper=wrapper,
        sampler=sampler,
        num_steps=args.num_steps,
        seed=args.seed,
        class_label=args.class_label,
        config_name="baseline",
        warmup_fraction=0.0,
        layer_fraction=0.0,
        is_baseline=True,
        detailed=args.detailed,
    )
    print(f"  Peak memory: {baseline_profile.peak_allocated_mb:.1f} MB")
    print(f"  Time: {baseline_profile.total_time_s:.2f}s")

    # Profile DQAR with attention reuse
    print(f"\nProfiling DQAR ({args.warmup_fraction*100:.0f}% warmup, {args.layer_fraction*100:.0f}% layers)...")
    wrapper.reset()
    dqar_profile = profile_inference(
        wrapper=wrapper,
        sampler=sampler,
        num_steps=args.num_steps,
        seed=args.seed,
        class_label=args.class_label,
        config_name=f"dqar_w{int(args.warmup_fraction*100)}_l{int(args.layer_fraction*100)}",
        warmup_fraction=args.warmup_fraction,
        layer_fraction=args.layer_fraction,
        is_baseline=False,
        detailed=args.detailed,
    )
    print(f"  Peak memory: {dqar_profile.peak_allocated_mb:.1f} MB")
    print(f"  Cache memory: {dqar_profile.cache_memory_mb:.1f} MB")
    print(f"  Time: {dqar_profile.total_time_s:.2f}s")

    # Print comparison
    print_memory_comparison(baseline_profile, dqar_profile)

    # Save results if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "config": {
                "warmup_fraction": args.warmup_fraction,
                "layer_fraction": args.layer_fraction,
                "num_steps": args.num_steps,
                "model": args.model,
                "seed": args.seed,
                "device": device_name,
                "schedule_type": schedule_type,
            },
            "baseline": asdict(baseline_profile),
            "dqar": asdict(dqar_profile),
            "comparison": {
                "peak_memory_delta_mb": dqar_profile.peak_allocated_mb - baseline_profile.peak_allocated_mb,
                "cache_memory_mb": dqar_profile.cache_memory_mb,
                "total_overhead_mb": (dqar_profile.peak_allocated_mb - baseline_profile.peak_allocated_mb) + dqar_profile.cache_memory_mb,
                "speedup": baseline_profile.total_time_s / dqar_profile.total_time_s,
            },
        }

        # Remove snapshots from baseline if empty (to keep JSON cleaner)
        if not results["baseline"]["snapshots"]:
            del results["baseline"]["snapshots"]
        if not results["dqar"]["snapshots"]:
            del results["dqar"]["snapshots"]

        results_path = output_dir / f"memory_profile_w{int(args.warmup_fraction*100)}_l{int(args.layer_fraction*100)}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
