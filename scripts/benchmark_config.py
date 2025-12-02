#!/usr/bin/env python3
"""
DQAR Single Configuration Benchmark

Benchmarks a specific DQAR configuration against baseline, measuring speedup and memory profile.

Usage:
    python scripts/benchmark_config.py --layer-fraction 0.66 --warmup-fraction 0.2
    python scripts/benchmark_config.py --layer-fraction 1.0 --warmup-fraction 0.2 --save-images
"""

import argparse
import torch
import time
import gc
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.policy.layer_scheduler import LayerScheduler
from dqar.utils import seed_everything, get_device

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def get_memory_stats():
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
    """Temporarily patch the layer scheduler's parameters.

    Args:
        warmup_fraction: Fraction of timesteps to skip reuse
        max_layers_fraction: Fraction of layers that can reuse
    """
    original_linear = LayerScheduler._linear_schedule
    original_reverse = LayerScheduler._linear_reverse_schedule

    def patched_linear_schedule(self, progress: float) -> List[int]:
        if progress < warmup_fraction:
            return []
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        max_reusable = int(self.config.num_layers * max_layers_fraction)
        num_reusable = int(adjusted_progress * max_reusable)
        return list(range(num_reusable))

    def patched_linear_reverse_schedule(self, progress: float) -> List[int]:
        """Reuse deep layers (closer to output) instead of shallow layers."""
        if progress < warmup_fraction:
            return []
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        max_reusable = int(self.config.num_layers * max_layers_fraction)
        num_reusable = int(adjusted_progress * max_reusable)
        if num_reusable == 0:
            return []
        # Start from deep layers (high indices)
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


def run_benchmark(
    wrapper: DQARDiTWrapper,
    sampler: DQARDDIMSampler,
    vae,
    num_samples: int,
    class_labels: List[int],
    seed: int,
    is_baseline: bool,
    warmup_fraction: float = 0.2,
    max_layers_fraction: float = 1.0,
    save_images: bool = False,
    output_dir: Optional[Path] = None,
    config_name: str = "config",
) -> Tuple[dict, List[torch.Tensor]]:
    """Run benchmark for a single configuration."""
    device = get_device()
    images = []

    # Patch schedule if not baseline
    original_method = None
    if not is_baseline:
        original_method = patch_layer_schedule(warmup_fraction, max_layers_fraction)
        if wrapper.manager and wrapper.manager.layer_scheduler:
            wrapper.manager.layer_scheduler._build_schedule()

    reset_memory_stats()

    # Warmup run (not timed)
    seed_everything(seed)
    wrapper.reset()
    with torch.no_grad():
        _ = sampler.sample(
            batch_size=1,
            class_labels=torch.tensor([class_labels[0]], device=device),
            progress_bar=False,
        )

    # Reset stats after warmup
    if wrapper.manager:
        wrapper.manager.reset_statistics()
    reset_memory_stats()

    # Timed runs
    total_time = 0.0
    for i in tqdm(range(num_samples), desc=f"  {config_name}", leave=False):
        seed_everything(seed + i)
        wrapper.reset()

        class_label = class_labels[i % len(class_labels)]

        start_time = time.perf_counter()
        with torch.no_grad():
            latent = sampler.sample(
                batch_size=1,
                class_labels=torch.tensor([class_label], device=device),
                progress_bar=False,
            )
        end_time = time.perf_counter()

        total_time += (end_time - start_time)

        # Decode latent to image using VAE
        if vae is not None:
            with torch.no_grad():
                latent_scaled = 1 / vae.config.scaling_factor * latent
                decoded = vae.decode(latent_scaled).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
            images.append(decoded.cpu())

            # Save individual images
            if save_images and output_dir and HAS_PIL:
                img_path = output_dir / f"{config_name}_sample{i}_class{class_label}.png"
                save_tensor_as_image(decoded, img_path)

    # Collect metrics
    memory_stats = get_memory_stats()

    reuse_ratio = 0.0
    cache_memory = 0.0
    if wrapper.manager and not is_baseline:
        stats = wrapper.manager.get_statistics()
        reuse_ratio = stats.get("reuse_ratio", 0.0)
        cache_stats = wrapper.manager.kv_cache.get_memory_usage()
        cache_memory = cache_stats.get("total_mb", 0.0)

    # Restore original method
    if original_method:
        restore_layer_schedule(original_method)

    result = {
        "total_time_s": total_time,
        "time_per_sample_s": total_time / num_samples,
        "reuse_ratio": reuse_ratio,
        "cache_memory_mb": cache_memory,
        "peak_memory_mb": memory_stats["peak"],
        "num_samples": num_samples,
        "is_baseline": is_baseline,
    }

    return result, images


def save_tensor_as_image(tensor: torch.Tensor, path: Path):
    """Save a tensor as an image file."""
    if not HAS_PIL:
        return

    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()

    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)

    img = Image.fromarray(tensor.cpu().numpy())
    img.save(path)


def print_results(baseline: dict, dqar: dict, layer_fraction: float, warmup_fraction: float):
    """Print comparison results table."""
    speedup = baseline["time_per_sample_s"] / dqar["time_per_sample_s"]
    time_delta = dqar["time_per_sample_s"] - baseline["time_per_sample_s"]
    memory_delta = dqar["peak_memory_mb"] - baseline["peak_memory_mb"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'DQAR':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Time/Sample':<20} {baseline['time_per_sample_s']:>11.2f}s {dqar['time_per_sample_s']:>11.2f}s {time_delta:>+11.2f}s")
    print(f"{'Speedup':<20} {'1.00x':>12} {speedup:>11.2f}x {(speedup-1)*100:>+10.1f}%")
    print(f"{'Reuse Ratio':<20} {'0%':>12} {dqar['reuse_ratio']*100:>11.1f}% {dqar['reuse_ratio']*100:>+10.1f}%")
    print(f"{'Peak Memory':<20} {baseline['peak_memory_mb']:>10.0f}MB {dqar['peak_memory_mb']:>10.0f}MB {memory_delta:>+10.0f}MB")
    print(f"{'Cache Memory':<20} {'0MB':>12} {dqar['cache_memory_mb']:>10.1f}MB {dqar['cache_memory_mb']:>+10.1f}MB")
    print("-" * 60)
    print()

    # Summary
    if speedup >= 1.15:
        print(f"Target (1.15x) achieved with {speedup:.2f}x speedup")
    else:
        print(f"Target (1.15x) not met. Current speedup: {speedup:.2f}x")
        if layer_fraction < 1.0:
            print(f"  Tip: Try increasing --layer-fraction (currently {layer_fraction:.0%})")
        if warmup_fraction > 0.2:
            print(f"  Tip: Try decreasing --warmup-fraction (currently {warmup_fraction:.0%})")


def main():
    parser = argparse.ArgumentParser(description="DQAR Single Configuration Benchmark")
    parser.add_argument("--layer-fraction", type=float, required=True,
                        help="Fraction of layers that can reuse attention (0.0-1.0)")
    parser.add_argument("--warmup-fraction", type=float, default=0.2,
                        help="Fraction of timesteps to skip reuse (default: 0.2)")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="Model to benchmark")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples per configuration")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/benchmark",
                        help="Output directory for images")
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated images")
    parser.add_argument("--reverse", action="store_true",
                        help="Use reverse scheduling (reuse deep layers instead of shallow)")
    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.layer_fraction <= 1.0:
        parser.error("--layer-fraction must be between 0.0 and 1.0")
    if not 0.0 <= args.warmup_fraction <= 1.0:
        parser.error("--warmup-fraction must be between 0.0 and 1.0")

    # Setup output directory
    output_dir = Path(args.output_dir)
    if args.save_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get device info
    device = get_device()
    device_name = str(device)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # Print header
    print()
    print("=" * 60)
    print("DQAR BENCHMARK")
    print("=" * 60)
    print(f"Device: {device_name}")
    print()
    print("Configuration:")
    print(f"  Layer Fraction:  {args.layer_fraction:.0%}")
    print(f"  Warmup Fraction: {args.warmup_fraction:.0%}")
    print(f"  Schedule Mode:   {'reverse (deep layers)' if args.reverse else 'normal (shallow layers)'}")
    print(f"  Num Steps:       {args.num_steps}")
    print(f"  Num Samples:     {args.num_samples}")
    print()

    # Class labels for diverse samples
    class_labels = [207, 360, 387, 974, 88, 417, 279, 928]

    # Load model
    print("Loading model...")
    from diffusers import DiTPipeline

    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to(device)

    model = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae
    print(f"Model: {args.model}")
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

    # Run baseline
    print("Running baseline (DQAR disabled)...")
    wrapper.manager.enabled = False
    baseline_result, baseline_images = run_benchmark(
        wrapper=wrapper,
        sampler=sampler,
        vae=vae,
        num_samples=args.num_samples,
        class_labels=class_labels,
        seed=args.seed,
        is_baseline=True,
        save_images=args.save_images,
        output_dir=output_dir,
        config_name="baseline",
    )
    print(f"  Baseline: {baseline_result['time_per_sample_s']:.2f}s/sample")

    # Run DQAR
    print(f"\nRunning DQAR (layers={args.layer_fraction:.0%}, warmup={args.warmup_fraction:.0%})...")
    wrapper.manager.enabled = True
    dqar_result, dqar_images = run_benchmark(
        wrapper=wrapper,
        sampler=sampler,
        vae=vae,
        num_samples=args.num_samples,
        class_labels=class_labels,
        seed=args.seed,
        is_baseline=False,
        warmup_fraction=args.warmup_fraction,
        max_layers_fraction=args.layer_fraction,
        save_images=args.save_images,
        output_dir=output_dir,
        config_name="dqar",
    )
    print(f"  DQAR: {dqar_result['time_per_sample_s']:.2f}s/sample")

    # Print comparison
    print_results(baseline_result, dqar_result, args.layer_fraction, args.warmup_fraction)

    if args.save_images:
        print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
