#!/usr/bin/env python3
"""
DQAR Warmup Sweep Benchmark

Tests different warmup fractions, measures speed/memory, generates sample images,
and creates comprehensive plots comparing baseline vs DQAR configurations.

Usage:
    python scripts/warmup_sweep.py --output-dir results/warmup_sweep
    python scripts/warmup_sweep.py --warmup-rates 0.1 0.2 0.3 0.4 0.5 --num-samples 4
"""

import argparse
import torch
import json
import time
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.policy.layer_scheduler import ScheduleType, SchedulerConfig, LayerScheduler
from dqar.utils import seed_everything, get_device

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plots will not be generated.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    warmup_fraction: float
    max_layers_fraction: float

    # Timing
    total_time_s: float
    time_per_sample_s: float
    speedup: float

    # DQAR metrics
    reuse_ratio: float
    cache_memory_mb: float

    # Memory
    peak_memory_mb: float

    # Config
    num_samples: int
    is_baseline: bool


def get_memory_usage() -> float:
    """Get current GPU/MPS memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, estimate from tensors
        return 0.0
    return 0.0


def reset_memory_stats():
    """Reset memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def patch_warmup_fraction(warmup_fraction: float, max_layers_fraction: float = 1.0):
    """
    Temporarily patch the layer scheduler's warmup fraction.

    Args:
        warmup_fraction: Fraction of timesteps to skip reuse (0.0-1.0)
        max_layers_fraction: Fraction of layers that can be reused (0.0-1.0)
    """
    # Monkey-patch the _linear_schedule method
    original_method = LayerScheduler._linear_schedule

    def patched_linear_schedule(self, progress: float) -> List[int]:
        if progress < warmup_fraction:
            return []
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)
        max_reusable = int(self.config.num_layers * max_layers_fraction)
        num_reusable = int(adjusted_progress * max_reusable)
        return list(range(num_reusable))

    LayerScheduler._linear_schedule = patched_linear_schedule
    return original_method


def restore_warmup_fraction(original_method):
    """Restore the original _linear_schedule method."""
    LayerScheduler._linear_schedule = original_method


def run_benchmark(
    wrapper: DQARDiTWrapper,
    sampler: DQARDDIMSampler,
    num_samples: int,
    class_labels: List[int],
    seed: int,
    is_baseline: bool = False,
    warmup_fraction: float = 0.2,
    max_layers_fraction: float = 1.0,
    save_images: bool = False,
    image_dir: Optional[Path] = None,
    config_name: str = "config",
) -> Tuple[BenchmarkResult, List[torch.Tensor]]:
    """
    Run a single benchmark configuration.

    Returns:
        Tuple of (BenchmarkResult, list of generated images)
    """
    device = get_device()
    images = []

    # Patch warmup if not baseline
    original_method = None
    if not is_baseline:
        original_method = patch_warmup_fraction(warmup_fraction, max_layers_fraction)
        # Rebuild schedule with new parameters
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
            image = sampler.sample(
                batch_size=1,
                class_labels=torch.tensor([class_label], device=device),
                progress_bar=False,
            )
        end_time = time.perf_counter()

        total_time += (end_time - start_time)
        images.append(image.cpu())

        # Save individual images
        if save_images and image_dir and HAS_PIL:
            img_path = image_dir / f"{config_name}_sample{i}_class{class_label}.png"
            save_tensor_as_image(image, img_path)

    # Collect metrics
    peak_memory = get_memory_usage()

    reuse_ratio = 0.0
    cache_memory = 0.0
    if wrapper.manager and not is_baseline:
        stats = wrapper.manager.get_statistics()
        reuse_ratio = stats.get("reuse_ratio", 0.0)
        cache_stats = wrapper.manager.kv_cache.get_memory_usage()
        cache_memory = cache_stats.get("total_mb", 0.0)

    # Restore original method
    if original_method:
        restore_warmup_fraction(original_method)

    result = BenchmarkResult(
        warmup_fraction=warmup_fraction if not is_baseline else 1.0,
        max_layers_fraction=max_layers_fraction if not is_baseline else 0.0,
        total_time_s=total_time,
        time_per_sample_s=total_time / num_samples,
        speedup=1.0,  # Will be calculated later
        reuse_ratio=reuse_ratio,
        cache_memory_mb=cache_memory,
        peak_memory_mb=peak_memory,
        num_samples=num_samples,
        is_baseline=is_baseline,
    )

    return result, images


def save_tensor_as_image(tensor: torch.Tensor, path: Path):
    """Save a tensor as an image file."""
    if not HAS_PIL:
        return

    # Assume tensor is (1, C, H, W) or (C, H, W)
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Normalize to 0-255
    tensor = (tensor + 1) / 2  # From [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()

    # Convert to PIL
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)

    img = Image.fromarray(tensor.numpy())
    img.save(path)


def create_plots(
    results: Dict[str, BenchmarkResult],
    baseline_result: BenchmarkResult,
    output_dir: Path,
    images: Dict[str, List[torch.Tensor]],
):
    """Create comprehensive benchmark plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return

    # Extract data for plotting
    warmup_rates = []
    speedups = []
    reuse_ratios = []
    times = []
    memories = []

    for name, result in sorted(results.items(), key=lambda x: x[1].warmup_fraction):
        if not result.is_baseline:
            warmup_rates.append(result.warmup_fraction * 100)
            speedups.append(result.speedup)
            reuse_ratios.append(result.reuse_ratio * 100)
            times.append(result.time_per_sample_s)
            memories.append(result.cache_memory_mb)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Speedup vs Warmup Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(warmup_rates, speedups, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    ax1.axhline(y=1.25, color='g', linestyle='--', alpha=0.5, label='Target (1.25x)')
    ax1.set_xlabel('Warmup Rate (%)', fontsize=12)
    ax1.set_ylabel('Speedup (x)', fontsize=12)
    ax1.set_title('Speedup vs Warmup Rate', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.9)

    # Plot 2: Reuse Ratio vs Warmup Rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(warmup_rates, reuse_ratios, 'g-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Warmup Rate (%)', fontsize=12)
    ax2.set_ylabel('Reuse Ratio (%)', fontsize=12)
    ax2.set_title('Attention Reuse vs Warmup Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Time per Sample vs Warmup Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(['Baseline'] + [f'{w:.0f}%' for w in warmup_rates],
            [baseline_result.time_per_sample_s] + times,
            color=['red'] + ['steelblue'] * len(times))
    ax3.set_xlabel('Configuration', fontsize=12)
    ax3.set_ylabel('Time per Sample (s)', fontsize=12)
    ax3.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Speedup vs Reuse Ratio (trade-off curve)
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = ax4.scatter(reuse_ratios, speedups, c=warmup_rates, cmap='viridis', s=100)
    ax4.set_xlabel('Reuse Ratio (%)', fontsize=12)
    ax4.set_ylabel('Speedup (x)', fontsize=12)
    ax4.set_title('Speed-Quality Trade-off', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Warmup Rate (%)')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Cache Memory vs Warmup Rate
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(warmup_rates, memories, 'm-^', linewidth=2, markersize=8)
    ax5.set_xlabel('Warmup Rate (%)', fontsize=12)
    ax5.set_ylabel('Cache Memory (MB)', fontsize=12)
    ax5.set_title('Cache Memory Usage', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary table as text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    table_data = [
        ['Config', 'Warmup', 'Speedup', 'Reuse %', 'Time (s)'],
        ['Baseline', '-', '1.00x', '0%', f'{baseline_result.time_per_sample_s:.2f}'],
    ]
    for name, result in sorted(results.items(), key=lambda x: x[1].warmup_fraction):
        if not result.is_baseline:
            table_data.append([
                name,
                f'{result.warmup_fraction*100:.0f}%',
                f'{result.speedup:.2f}x',
                f'{result.reuse_ratio*100:.1f}%',
                f'{result.time_per_sample_s:.2f}',
            ])

    table = ax6.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax6.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('DQAR Warmup Rate Sweep Results', fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.savefig(output_dir / 'warmup_sweep_plots.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'warmup_sweep_plots.pdf', bbox_inches='tight')
    plt.close()

    print(f"  Saved plots to {output_dir / 'warmup_sweep_plots.png'}")

    # Create image comparison grid if we have images
    if images and HAS_PIL:
        create_image_grid(images, output_dir)


def create_image_grid(images: Dict[str, List[torch.Tensor]], output_dir: Path):
    """Create a grid comparing images across configurations."""
    if not HAS_MATPLOTLIB or not images:
        return

    # Get configurations and samples
    configs = list(images.keys())
    if not configs:
        return

    num_configs = len(configs)
    num_samples = min(4, len(images[configs[0]]))  # Show max 4 samples

    fig, axes = plt.subplots(num_configs, num_samples, figsize=(4*num_samples, 4*num_configs))

    if num_configs == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, config in enumerate(configs):
        for j in range(num_samples):
            if j < len(images[config]):
                img = images[config][j]
                if img.dim() == 4:
                    img = img[0]
                img = (img + 1) / 2  # [-1,1] -> [0,1]
                img = img.clamp(0, 1).permute(1, 2, 0).numpy()

                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(config, fontsize=12, rotation=0, ha='right', va='center')
            else:
                axes[i, j].axis('off')

    plt.suptitle('Image Quality Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'image_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved image grid to {output_dir / 'image_comparison.png'}")


def generate_report(
    results: Dict[str, BenchmarkResult],
    baseline_result: BenchmarkResult,
    output_dir: Path,
    device_name: str,
):
    """Generate a markdown report."""
    report_path = output_dir / "WARMUP_SWEEP_REPORT.md"

    with open(report_path, "w") as f:
        f.write("# DQAR Warmup Rate Sweep Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Device**: {device_name}\n")
        f.write(f"**Model**: DiT-XL-2-256\n\n")

        f.write("---\n\n")
        f.write("## Summary\n\n")

        # Find best configuration
        best_config = None
        best_speedup = 0
        for name, result in results.items():
            if not result.is_baseline and result.speedup > best_speedup:
                best_speedup = result.speedup
                best_config = name

        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Best Speedup** | {best_speedup:.2f}x ({best_config}) |\n")
        f.write(f"| **Baseline Time** | {baseline_result.time_per_sample_s:.2f}s/sample |\n")
        f.write(f"| **Target Met** | {'Yes' if best_speedup >= 1.15 else 'No'} (>=1.15x) |\n\n")

        f.write("---\n\n")
        f.write("## Detailed Results\n\n")

        f.write("| Warmup Rate | Speedup | Reuse Ratio | Time/Sample | Cache (MB) |\n")
        f.write("|-------------|---------|-------------|-------------|------------|\n")
        f.write(f"| Baseline | 1.00x | 0% | {baseline_result.time_per_sample_s:.2f}s | 0 |\n")

        for name, result in sorted(results.items(), key=lambda x: x[1].warmup_fraction):
            if not result.is_baseline:
                f.write(f"| {result.warmup_fraction*100:.0f}% | {result.speedup:.2f}x | ")
                f.write(f"{result.reuse_ratio*100:.1f}% | {result.time_per_sample_s:.2f}s | ")
                f.write(f"{result.cache_memory_mb:.1f} |\n")

        f.write("\n---\n\n")
        f.write("## Plots\n\n")
        f.write("![Warmup Sweep Results](warmup_sweep_plots.png)\n\n")
        f.write("![Image Comparison](image_comparison.png)\n\n")

        f.write("---\n\n")
        f.write("## Recommendations\n\n")

        # Find optimal trade-off (speedup >= 1.15 with lowest reuse)
        optimal = None
        for name, result in results.items():
            if not result.is_baseline and result.speedup >= 1.15:
                if optimal is None or result.reuse_ratio < results[optimal].reuse_ratio:
                    optimal = name

        if optimal:
            opt_result = results[optimal]
            f.write(f"**Recommended Configuration**: {opt_result.warmup_fraction*100:.0f}% warmup\n")
            f.write(f"- Achieves {opt_result.speedup:.2f}x speedup\n")
            f.write(f"- Uses {opt_result.reuse_ratio*100:.1f}% attention reuse\n")
            f.write(f"- Balances speed and quality\n")
        else:
            f.write("No configuration met the 1.15x speedup target.\n")
            f.write("Consider using lower warmup rates or enabling more layers.\n")

        f.write("\n---\n\n")
        f.write("*Report generated by DQAR warmup sweep benchmark*\n")

    print(f"  Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="DQAR Warmup Rate Sweep Benchmark")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="Model to benchmark")
    parser.add_argument("--warmup-rates", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        help="Warmup fractions to test")
    parser.add_argument("--max-layers-fraction", type=float, default=0.33,
                        help="Fraction of layers that can be reused (default: 0.33 = 1/3)")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples per configuration")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/warmup_sweep",
                        help="Output directory")
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated images")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "images"
    if args.save_images:
        image_dir.mkdir(exist_ok=True)

    device = get_device()
    device_name = str(device)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    print(f"DQAR Warmup Sweep Benchmark")
    print(f"=" * 50)
    print(f"Device: {device_name}")
    print(f"Warmup rates: {args.warmup_rates}")
    print(f"Max layers fraction: {args.max_layers_fraction}")
    print(f"Samples per config: {args.num_samples}")
    print(f"Output: {output_dir}")
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
    print(f"Model loaded: {args.model}")

    dqar_config = DQARConfig(
        quantization_bits=16,  # FP16 for quality
        use_layer_scheduling=True,
        schedule_type="linear",
    )

    wrapper = DQARDiTWrapper(model, dqar_config)

    sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    results = {}
    all_images = {}

    # Run baseline
    print("\nRunning baseline...")
    wrapper.manager.enabled = False
    baseline_result, baseline_images = run_benchmark(
        wrapper=wrapper,
        sampler=sampler,
        num_samples=args.num_samples,
        class_labels=class_labels,
        seed=args.seed,
        is_baseline=True,
        save_images=args.save_images,
        image_dir=image_dir,
        config_name="baseline",
    )
    results["baseline"] = baseline_result
    all_images["baseline"] = baseline_images
    print(f"  Baseline: {baseline_result.time_per_sample_s:.2f}s/sample")

    # Run warmup sweep
    print("\nRunning warmup sweep...")
    wrapper.manager.enabled = True

    for warmup_rate in args.warmup_rates:
        config_name = f"warmup_{int(warmup_rate*100)}pct"
        print(f"\nTesting warmup={warmup_rate*100:.0f}%...")

        result, images = run_benchmark(
            wrapper=wrapper,
            sampler=sampler,
            num_samples=args.num_samples,
            class_labels=class_labels,
            seed=args.seed,
            is_baseline=False,
            warmup_fraction=warmup_rate,
            max_layers_fraction=args.max_layers_fraction,
            save_images=args.save_images,
            image_dir=image_dir,
            config_name=config_name,
        )

        # Calculate speedup
        result.speedup = baseline_result.time_per_sample_s / result.time_per_sample_s

        results[config_name] = result
        all_images[config_name] = images

        print(f"  Time: {result.time_per_sample_s:.2f}s, Speedup: {result.speedup:.2f}x, Reuse: {result.reuse_ratio*100:.1f}%")

    # Save raw results
    results_dict = {name: asdict(r) for name, r in results.items()}
    with open(output_dir / "warmup_sweep_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved results to {output_dir / 'warmup_sweep_results.json'}")

    # Generate plots
    print("\nGenerating plots...")
    create_plots(results, baseline_result, output_dir, all_images)

    # Generate report
    print("\nGenerating report...")
    generate_report(results, baseline_result, output_dir, device_name)

    print("\nDone!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
