#!/usr/bin/env python3
"""
DQAR Hyperparameter Tuning Script

Performs grid search over warmup rate, layer fraction, and schedule type
(LINEAR vs LINEAR_REVERSE) to find optimal configurations that balance
speedup and FID score.

Usage:
    python scripts/tune_hyperparameters.py --num-samples 256 --output-dir results/tuning
    python scripts/tune_hyperparameters.py --speedup-weight 0.7 --fid-weight 0.3
    python scripts/tune_hyperparameters.py --min-speedup 1.05 --max-fid 10.0
    python scripts/tune_hyperparameters.py --schedule-types linear  # Only test linear
"""

import argparse
import torch
import os
import gc
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not found. Cannot save images.")

try:
    from cleanfid import fid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False

try:
    from pytorch_fid import fid_score
    HAS_PYTORCH_FID = True
except ImportError:
    HAS_PYTORCH_FID = False

if not HAS_CLEANFID and not HAS_PYTORCH_FID:
    print("Warning: No FID library found. Install with: pip install clean-fid")


@dataclass
class TuningResult:
    """Result from a single hyperparameter configuration."""
    warmup_fraction: float
    layer_fraction: float
    schedule_type: str
    speedup: float
    fid_score: float
    time_per_sample: float
    reuse_ratio: float
    num_samples: int
    is_pareto_optimal: bool = False
    combined_score: float = 0.0


def patch_layer_schedule(warmup_fraction: float, max_layers_fraction: float):
    """Patch both linear and linear_reverse schedule methods."""
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
    """Restore original schedule methods."""
    original_linear, original_reverse = original_methods
    LayerScheduler._linear_schedule = original_linear
    LayerScheduler._linear_reverse_schedule = original_reverse


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


def generate_images(
    wrapper: DQARDiTWrapper,
    sampler: DQARDDIMSampler,
    vae,
    num_samples: int,
    output_dir: Path,
    class_labels: List[int],
    seed: int = 42,
) -> Tuple[Path, float, float]:
    """Generate images and measure timing."""
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    reuse_count = 0
    total_count = 0

    for i in tqdm(range(num_samples), desc="    Generating", leave=False):
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

            # Decode with VAE
            latent_scaled = 1 / vae.config.scaling_factor * latent
            decoded = vae.decode(latent_scaled).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
        end_time = time.perf_counter()

        total_time += (end_time - start_time)

        # Save image
        img_path = output_dir / f"sample_{i:04d}.png"
        save_tensor_as_image(decoded, img_path)

        # Collect reuse stats
        if wrapper.manager and wrapper.manager.enabled:
            stats = wrapper.manager.get_statistics()
            reuse_count += stats.get("reuse_count", 0)
            total_count += stats.get("total_count", 1)

    time_per_sample = total_time / num_samples
    reuse_ratio = reuse_count / max(1, total_count)

    return output_dir, time_per_sample, reuse_ratio


def compute_fid(path1: Path, path2: Path, device: str = "cuda") -> float:
    """Compute FID score between two directories of images."""
    if HAS_CLEANFID:
        score = fid.compute_fid(str(path1), str(path2), device=device)
        return score
    elif HAS_PYTORCH_FID:
        score = fid_score.calculate_fid_given_paths(
            [str(path1), str(path2)],
            batch_size=50,
            device=device,
            dims=2048,
        )
        return score
    else:
        return -1.0


def find_pareto_optimal(results: List[TuningResult]) -> List[TuningResult]:
    """Find Pareto-optimal configurations (maximize speedup, minimize FID)."""
    pareto_optimal = []

    for candidate in results:
        is_dominated = False
        for other in results:
            if other is candidate:
                continue
            # other dominates candidate if:
            # - other has higher or equal speedup AND lower or equal FID
            # - AND at least one is strictly better
            if (other.speedup >= candidate.speedup and
                other.fid_score <= candidate.fid_score and
                (other.speedup > candidate.speedup or other.fid_score < candidate.fid_score)):
                is_dominated = True
                break

        if not is_dominated:
            candidate.is_pareto_optimal = True
            pareto_optimal.append(candidate)

    return pareto_optimal


def compute_combined_score(
    result: TuningResult,
    speedup_weight: float,
    fid_weight: float,
    max_fid: float,
    baseline_speedup: float = 1.0,
) -> float:
    """Compute combined score for ranking configurations.

    Higher is better. Speedup is normalized by subtracting 1 (so baseline = 0).
    FID is inverted and normalized (lower FID = higher score).
    """
    # Normalize speedup: (speedup - 1) so baseline = 0, 1.2x = 0.2
    speedup_score = (result.speedup - baseline_speedup)

    # Normalize FID: (max_fid - fid) / max_fid so lower FID = higher score
    # Clamp FID to max_fid to avoid negative scores
    clamped_fid = min(result.fid_score, max_fid)
    fid_score_normalized = (max_fid - clamped_fid) / max_fid

    # Combined score
    combined = speedup_weight * speedup_score + fid_weight * fid_score_normalized
    return combined


def main():
    parser = argparse.ArgumentParser(description="DQAR Hyperparameter Tuning")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="Model to use")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples per configuration")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/tuning",
                        help="Output directory")

    # Search space
    parser.add_argument("--warmup-values", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4],
                        help="Warmup fractions to search")
    parser.add_argument("--layer-values", type=float, nargs="+",
                        default=[0.33, 0.5, 0.66, 0.75, 1.0],
                        help="Layer fractions to search")
    parser.add_argument("--schedule-types", type=str, nargs="+",
                        default=["linear", "linear_reverse"],
                        choices=["linear", "linear_reverse"],
                        help="Schedule types to search (default: both linear and linear_reverse)")

    # Optimization weights
    parser.add_argument("--speedup-weight", type=float, default=0.5,
                        help="Weight for speedup in combined score (0-1)")
    parser.add_argument("--fid-weight", type=float, default=0.5,
                        help="Weight for FID in combined score (0-1)")

    # Constraints
    parser.add_argument("--min-speedup", type=float, default=1.0,
                        help="Minimum acceptable speedup")
    parser.add_argument("--max-fid", type=float, default=50.0,
                        help="Maximum acceptable FID score")

    # Resume
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline generation (use existing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")

    args = parser.parse_args()

    if not HAS_PIL:
        print("Error: PIL is required for this script.")
        return

    if not HAS_PYTORCH_FID and not HAS_CLEANFID:
        print("Error: Either pytorch-fid or cleanfid is required.")
        print("Install with: pip install clean-fid")
        return

    # Normalize weights
    total_weight = args.speedup_weight + args.fid_weight
    speedup_weight = args.speedup_weight / total_weight
    fid_weight = args.fid_weight / total_weight

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    device_name = str(device)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    fid_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Class labels for diverse samples
    class_labels = [
        207, 360, 387, 974, 88, 417, 279, 928,
        1, 9, 18, 25, 76, 130, 291, 388,
        402, 417, 471, 574, 609, 654, 682, 717,
        755, 795, 817, 866, 895, 932, 951, 985,
    ]

    # Build search space
    schedule_types = args.schedule_types

    search_space = []
    for warmup in args.warmup_values:
        for layer_frac in args.layer_values:
            for schedule_type in schedule_types:
                search_space.append({
                    "warmup_fraction": warmup,
                    "layer_fraction": layer_frac,
                    "schedule_type": schedule_type,
                })

    # Print header
    print()
    print("=" * 70)
    print("DQAR HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Device: {device_name}")
    print(f"Model: {args.model}")
    print()
    print("Search Space:")
    print(f"  Warmup values:  {args.warmup_values}")
    print(f"  Layer values:   {args.layer_values}")
    print(f"  Schedule types: {schedule_types}")
    print(f"  Total configs:  {len(search_space)}")
    print()
    print("Optimization:")
    print(f"  Speedup weight: {speedup_weight:.2f}")
    print(f"  FID weight:     {fid_weight:.2f}")
    print(f"  Min speedup:    {args.min_speedup:.2f}x")
    print(f"  Max FID:        {args.max_fid:.1f}")
    print()

    # Load existing results if resuming
    results_path = output_dir / "tuning_results.json"
    existing_results = {}
    if args.resume and results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
            for r in data.get("results", []):
                key = f"{r['schedule_type']}_w{r['warmup_fraction']}_l{r['layer_fraction']}"
                existing_results[key] = r
        print(f"Loaded {len(existing_results)} existing results")

    # Load model
    print("Loading model...")
    from diffusers import DiTPipeline

    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to(device)

    model = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae
    print(f"Model loaded.")
    print()

    # Generate baseline images
    baseline_dir = images_dir / "baseline"
    baseline_time = None

    if not args.skip_baseline or not baseline_dir.exists():
        print("=" * 70)
        print("GENERATING BASELINE")
        print("=" * 70)

        dqar_config = DQARConfig(
            quantization_bits=16,
            use_layer_scheduling=True,
            schedule_type="linear",
        )
        wrapper = DQARDiTWrapper(model, dqar_config)
        sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
        sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

        wrapper.manager.enabled = False  # Disable DQAR for baseline

        baseline_dir, baseline_time, _ = generate_images(
            wrapper, sampler, vae, args.num_samples,
            baseline_dir, class_labels, args.seed
        )
        print(f"  Baseline time: {baseline_time:.3f}s/sample")
        print(f"  Saved to: {baseline_dir}")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("Using existing baseline images")
        # Estimate baseline time from existing results or measure
        if existing_results:
            # Find a result with speedup to back-calculate baseline time
            for r in existing_results.values():
                if r.get("speedup", 0) > 0:
                    baseline_time = r["time_per_sample"] * r["speedup"]
                    break

    # Run hyperparameter sweep
    print()
    print("=" * 70)
    print("HYPERPARAMETER SWEEP")
    print("=" * 70)
    print()

    results: List[TuningResult] = []

    for i, config in enumerate(search_space):
        warmup = config["warmup_fraction"]
        layer_frac = config["layer_fraction"]
        schedule_type = config["schedule_type"]

        config_key = f"{schedule_type}_w{warmup}_l{layer_frac}"
        config_name = f"{schedule_type}_w{int(warmup*100)}_l{int(layer_frac*100)}"

        print(f"[{i+1}/{len(search_space)}] {config_name}")

        # Check if already computed
        if config_key in existing_results:
            r = existing_results[config_key]
            result = TuningResult(
                warmup_fraction=r["warmup_fraction"],
                layer_fraction=r["layer_fraction"],
                schedule_type=r["schedule_type"],
                speedup=r["speedup"],
                fid_score=r["fid_score"],
                time_per_sample=r["time_per_sample"],
                reuse_ratio=r["reuse_ratio"],
                num_samples=r["num_samples"],
            )
            results.append(result)
            print(f"  (cached) Speedup: {result.speedup:.3f}x, FID: {result.fid_score:.2f}")
            continue

        # Patch schedule
        original_methods = patch_layer_schedule(warmup, layer_frac)

        # Create wrapper
        dqar_config = DQARConfig(
            quantization_bits=16,
            use_layer_scheduling=True,
            schedule_type=schedule_type,
        )
        wrapper = DQARDiTWrapper(model, dqar_config)
        sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
        sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

        wrapper.manager.enabled = True
        wrapper.manager.layer_scheduler._build_schedule()

        # Generate images
        config_dir = images_dir / config_name
        config_dir, time_per_sample, reuse_ratio = generate_images(
            wrapper, sampler, vae, args.num_samples,
            config_dir, class_labels, args.seed
        )

        # Restore schedule
        restore_layer_schedule(original_methods)

        # Compute FID
        print(f"  Computing FID...")
        fid_val = compute_fid(baseline_dir, config_dir, fid_device)

        # Compute speedup
        if baseline_time is None:
            # Measure baseline if not already done
            wrapper.manager.enabled = False
            _, baseline_time, _ = generate_images(
                wrapper, sampler, vae, min(4, args.num_samples),
                baseline_dir, class_labels, args.seed
            )

        speedup = baseline_time / time_per_sample

        result = TuningResult(
            warmup_fraction=warmup,
            layer_fraction=layer_frac,
            schedule_type=schedule_type,
            speedup=speedup,
            fid_score=fid_val,
            time_per_sample=time_per_sample,
            reuse_ratio=reuse_ratio,
            num_samples=args.num_samples,
        )
        results.append(result)

        print(f"  Speedup: {speedup:.3f}x, FID: {fid_val:.2f}, Reuse: {reuse_ratio*100:.1f}%")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save intermediate results
        _save_results(results, output_dir, speedup_weight, fid_weight, args)

    # Find Pareto-optimal configurations
    pareto_optimal = find_pareto_optimal(results)

    # Compute combined scores
    for result in results:
        result.combined_score = compute_combined_score(
            result, speedup_weight, fid_weight, args.max_fid
        )

    # Filter by constraints
    valid_results = [
        r for r in results
        if r.speedup >= args.min_speedup and r.fid_score <= args.max_fid
    ]

    # Sort by combined score
    valid_results.sort(key=lambda r: r.combined_score, reverse=True)
    pareto_optimal.sort(key=lambda r: r.combined_score, reverse=True)

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # All results table
    print("All Configurations (sorted by combined score):")
    print("-" * 90)
    print(f"{'Config':<30} {'Speedup':>8} {'FID':>8} {'Reuse':>8} {'Score':>8} {'Pareto':>8}")
    print("-" * 90)

    sorted_results = sorted(results, key=lambda r: r.combined_score, reverse=True)
    for r in sorted_results:
        config_name = f"{r.schedule_type}_w{int(r.warmup_fraction*100)}_l{int(r.layer_fraction*100)}"
        pareto_mark = "*" if r.is_pareto_optimal else ""
        constraint_ok = "✓" if r.speedup >= args.min_speedup and r.fid_score <= args.max_fid else "✗"
        print(f"{config_name:<30} {r.speedup:>7.3f}x {r.fid_score:>8.2f} {r.reuse_ratio*100:>7.1f}% {r.combined_score:>8.3f} {pareto_mark:>7}{constraint_ok}")
    print("-" * 90)
    print("* = Pareto-optimal, ✓ = meets constraints")
    print()

    # Pareto-optimal configurations
    print("Pareto-Optimal Configurations:")
    print("-" * 70)
    for r in pareto_optimal:
        config_name = f"{r.schedule_type}_w{int(r.warmup_fraction*100)}_l{int(r.layer_fraction*100)}"
        print(f"  {config_name}: {r.speedup:.3f}x speedup, {r.fid_score:.2f} FID")
    print()

    # Linear vs Linear Reverse Comparison (if both are in search space)
    if "linear" in schedule_types and "linear_reverse" in schedule_types:
        print("=" * 70)
        print("LINEAR vs LINEAR_REVERSE COMPARISON")
        print("=" * 70)
        print()
        print(f"{'Warmup':<8} {'Layers':<8} {'Linear Speedup':>14} {'Linear FID':>12} {'Reverse Speedup':>16} {'Reverse FID':>12} {'Winner':>10}")
        print("-" * 92)

        # Group results by warmup and layer fraction
        results_by_config = {}
        for r in results:
            key = (r.warmup_fraction, r.layer_fraction)
            if key not in results_by_config:
                results_by_config[key] = {}
            results_by_config[key][r.schedule_type] = r

        for (warmup, layer_frac) in sorted(results_by_config.keys()):
            config_results = results_by_config[(warmup, layer_frac)]
            linear_r = config_results.get("linear")
            reverse_r = config_results.get("linear_reverse")

            if linear_r and reverse_r:
                # Determine winner based on combined score or FID
                if linear_r.fid_score < reverse_r.fid_score:
                    winner = "Linear"
                elif reverse_r.fid_score < linear_r.fid_score:
                    winner = "Reverse"
                else:
                    winner = "Tie"

                print(f"{warmup*100:>5.0f}%  {layer_frac*100:>5.0f}%  "
                      f"{linear_r.speedup:>13.3f}x {linear_r.fid_score:>12.2f} "
                      f"{reverse_r.speedup:>15.3f}x {reverse_r.fid_score:>12.2f} "
                      f"{winner:>10}")
            elif linear_r:
                print(f"{warmup*100:>5.0f}%  {layer_frac*100:>5.0f}%  "
                      f"{linear_r.speedup:>13.3f}x {linear_r.fid_score:>12.2f} "
                      f"{'N/A':>15} {'N/A':>12} {'Linear':>10}")
            elif reverse_r:
                print(f"{warmup*100:>5.0f}%  {layer_frac*100:>5.0f}%  "
                      f"{'N/A':>13} {'N/A':>12} "
                      f"{reverse_r.speedup:>15.3f}x {reverse_r.fid_score:>12.2f} "
                      f"{'Reverse':>10}")

        print("-" * 92)

        # Summary statistics
        linear_wins = 0
        reverse_wins = 0
        ties = 0
        for (warmup, layer_frac) in results_by_config.keys():
            config_results = results_by_config[(warmup, layer_frac)]
            linear_r = config_results.get("linear")
            reverse_r = config_results.get("linear_reverse")
            if linear_r and reverse_r:
                if linear_r.fid_score < reverse_r.fid_score:
                    linear_wins += 1
                elif reverse_r.fid_score < linear_r.fid_score:
                    reverse_wins += 1
                else:
                    ties += 1

        print()
        print(f"Summary: Linear wins {linear_wins}, Reverse wins {reverse_wins}, Ties {ties}")

        # Best configuration per schedule type
        linear_results = [r for r in results if r.schedule_type == "linear"]
        reverse_results = [r for r in results if r.schedule_type == "linear_reverse"]

        if linear_results:
            best_linear = min(linear_results, key=lambda r: r.fid_score)
            print(f"\nBest LINEAR config (by FID): "
                  f"w{int(best_linear.warmup_fraction*100)}_l{int(best_linear.layer_fraction*100)} "
                  f"({best_linear.speedup:.3f}x, FID {best_linear.fid_score:.2f})")

        if reverse_results:
            best_reverse = min(reverse_results, key=lambda r: r.fid_score)
            print(f"Best REVERSE config (by FID): "
                  f"w{int(best_reverse.warmup_fraction*100)}_l{int(best_reverse.layer_fraction*100)} "
                  f"({best_reverse.speedup:.3f}x, FID {best_reverse.fid_score:.2f})")

        print()

    # Best configuration
    if valid_results:
        best = valid_results[0]
        print("=" * 70)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 70)
        print()
        print(f"  Schedule Type:    {best.schedule_type}")
        print(f"  Warmup Fraction:  {best.warmup_fraction:.0%}")
        print(f"  Layer Fraction:   {best.layer_fraction:.0%}")
        print()
        print(f"  Speedup:          {best.speedup:.3f}x")
        print(f"  FID Score:        {best.fid_score:.2f}")
        print(f"  Reuse Ratio:      {best.reuse_ratio*100:.1f}%")
        print(f"  Combined Score:   {best.combined_score:.3f}")
        print()
        print("Configuration code:")
        print("```python")
        print(f"warmup_fraction = {best.warmup_fraction}")
        print(f"max_layers_fraction = {best.layer_fraction}")
        print(f"schedule_type = \"{best.schedule_type}\"")
        print("```")
    else:
        print("No configurations meet the specified constraints.")
        print(f"  Min speedup: {args.min_speedup}x")
        print(f"  Max FID: {args.max_fid}")

    # Save final results
    _save_results(results, output_dir, speedup_weight, fid_weight, args)
    print()
    print(f"Results saved to: {output_dir / 'tuning_results.json'}")


def _save_results(
    results: List[TuningResult],
    output_dir: Path,
    speedup_weight: float,
    fid_weight: float,
    args,
):
    """Save results to JSON file."""
    pareto_optimal = find_pareto_optimal(results)

    # Compute combined scores
    for result in results:
        result.combined_score = compute_combined_score(
            result, speedup_weight, fid_weight, args.max_fid
        )

    valid_results = [
        r for r in results
        if r.speedup >= args.min_speedup and r.fid_score <= args.max_fid
    ]
    valid_results.sort(key=lambda r: r.combined_score, reverse=True)

    best_config = None
    if valid_results:
        best = valid_results[0]
        best_config = {
            "schedule_type": best.schedule_type,
            "warmup_fraction": best.warmup_fraction,
            "layer_fraction": best.layer_fraction,
            "speedup": best.speedup,
            "fid_score": best.fid_score,
        }

    # Build comparison data if both schedule types present
    schedule_types = args.schedule_types
    comparison = []
    if "linear" in schedule_types and "linear_reverse" in schedule_types:
        results_by_config = {}
        for r in results:
            key = (r.warmup_fraction, r.layer_fraction)
            if key not in results_by_config:
                results_by_config[key] = {}
            results_by_config[key][r.schedule_type] = r

        for (warmup, layer_frac) in sorted(results_by_config.keys()):
            config_results = results_by_config[(warmup, layer_frac)]
            linear_r = config_results.get("linear")
            reverse_r = config_results.get("linear_reverse")

            if linear_r and reverse_r:
                if linear_r.fid_score < reverse_r.fid_score:
                    winner = "linear"
                    fid_diff = reverse_r.fid_score - linear_r.fid_score
                elif reverse_r.fid_score < linear_r.fid_score:
                    winner = "linear_reverse"
                    fid_diff = linear_r.fid_score - reverse_r.fid_score
                else:
                    winner = "tie"
                    fid_diff = 0.0

                comparison.append({
                    "warmup_fraction": warmup,
                    "layer_fraction": layer_frac,
                    "linear_speedup": linear_r.speedup,
                    "linear_fid": linear_r.fid_score,
                    "reverse_speedup": reverse_r.speedup,
                    "reverse_fid": reverse_r.fid_score,
                    "winner": winner,
                    "fid_difference": fid_diff,
                })

    # Best per schedule type
    best_per_schedule = {}
    for schedule_type in schedule_types:
        schedule_results = [r for r in results if r.schedule_type == schedule_type]
        if schedule_results:
            best_by_fid = min(schedule_results, key=lambda r: r.fid_score)
            best_per_schedule[schedule_type] = {
                "warmup_fraction": best_by_fid.warmup_fraction,
                "layer_fraction": best_by_fid.layer_fraction,
                "speedup": best_by_fid.speedup,
                "fid_score": best_by_fid.fid_score,
            }

    output = {
        "config": {
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "speedup_weight": speedup_weight,
            "fid_weight": fid_weight,
            "min_speedup": args.min_speedup,
            "max_fid": args.max_fid,
            "schedule_types": schedule_types,
        },
        "results": [asdict(r) for r in results],
        "pareto_optimal": [
            {
                "schedule_type": r.schedule_type,
                "warmup_fraction": r.warmup_fraction,
                "layer_fraction": r.layer_fraction,
                "speedup": r.speedup,
                "fid_score": r.fid_score,
            }
            for r in pareto_optimal
        ],
        "linear_vs_reverse_comparison": comparison,
        "best_per_schedule_type": best_per_schedule,
        "recommended": best_config,
    }

    results_path = output_dir / "tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
