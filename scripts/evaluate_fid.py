#!/usr/bin/env python3
"""
FID Score Evaluation for DQAR Scheduling Strategies

Compares FID scores between:
- Baseline (no DQAR)
- LINEAR scheduling (shallow-first)
- LINEAR_REVERSE scheduling (deep-first)

Usage:
    python scripts/evaluate_fid.py --num-samples 256 --output-dir results/fid_eval
    python scripts/evaluate_fid.py --layer-fraction 0.5 --warmup-fraction 0.2
"""

import argparse
import torch
import os
import gc
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
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
    from pytorch_fid import fid_score
    HAS_PYTORCH_FID = True
except ImportError:
    HAS_PYTORCH_FID = False
    print("Warning: pytorch-fid not found. Install with: pip install pytorch-fid")

try:
    from cleanfid import fid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False


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
    config_name: str,
    seed: int = 42,
) -> Path:
    """Generate images for a configuration and save to directory."""
    device = get_device()
    image_dir = output_dir / config_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Use diverse ImageNet classes
    class_labels = [
        207, 360, 387, 974, 88, 417, 279, 928,
        1, 9, 18, 25, 76, 130, 291, 388,
        402, 417, 471, 574, 609, 654, 682, 717,
        755, 795, 817, 866, 895, 932, 951, 985,
    ]

    print(f"  Generating {num_samples} images for {config_name}...")

    for i in tqdm(range(num_samples), desc=f"    {config_name}", leave=False):
        seed_everything(seed + i)
        wrapper.reset()

        class_label = class_labels[i % len(class_labels)]

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

        # Save image
        img_path = image_dir / f"sample_{i:04d}.png"
        save_tensor_as_image(decoded, img_path)

    return image_dir


def compute_fid(path1: Path, path2: Path, device: str = "cuda") -> float:
    """Compute FID score between two directories of images."""
    if HAS_CLEANFID:
        # cleanfid is generally more accurate
        score = fid.compute_fid(str(path1), str(path2), device=device)
        return score
    elif HAS_PYTORCH_FID:
        # pytorch-fid fallback
        score = fid_score.calculate_fid_given_paths(
            [str(path1), str(path2)],
            batch_size=50,
            device=device,
            dims=2048,
        )
        return score
    else:
        print("Error: No FID library available. Install pytorch-fid or cleanfid.")
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="FID Evaluation for DQAR Scheduling")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="Model to use")
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of samples to generate (more = more accurate FID)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--layer-fraction", type=float, default=0.33,
                        help="Layer fraction for DQAR")
    parser.add_argument("--warmup-fraction", type=float, default=0.2,
                        help="Warmup fraction for DQAR")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/fid_eval",
                        help="Output directory")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip image generation (use existing images)")
    args = parser.parse_args()

    if not HAS_PIL:
        print("Error: PIL is required for this script.")
        return

    if not HAS_PYTORCH_FID and not HAS_CLEANFID:
        print("Error: Either pytorch-fid or cleanfid is required.")
        print("Install with: pip install pytorch-fid")
        print("         or: pip install clean-fid")
        return

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    device_name = str(device)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # Print header
    print()
    print("=" * 60)
    print("FID EVALUATION: LINEAR vs LINEAR_REVERSE")
    print("=" * 60)
    print(f"Device: {device_name}")
    print(f"Model: {args.model}")
    print()
    print("Configuration:")
    print(f"  Layer Fraction:  {args.layer_fraction:.0%}")
    print(f"  Warmup Fraction: {args.warmup_fraction:.0%}")
    print(f"  Num Samples:     {args.num_samples}")
    print(f"  Num Steps:       {args.num_steps}")
    print()

    if not args.skip_generation:
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

        # Patch schedules
        original_methods = patch_layer_schedule(args.warmup_fraction, args.layer_fraction)

        # === Generate Baseline Images ===
        print("Generating BASELINE images...")
        dqar_config = DQARConfig(
            quantization_bits=16,
            use_layer_scheduling=True,
            schedule_type="linear",
        )
        wrapper = DQARDiTWrapper(model, dqar_config)
        sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
        sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

        wrapper.manager.enabled = False  # Disable DQAR for baseline
        baseline_dir = generate_images(
            wrapper, sampler, vae, args.num_samples,
            output_dir, "baseline", args.seed
        )
        print(f"  Saved to: {baseline_dir}")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # === Generate LINEAR (shallow-first) Images ===
        print("\nGenerating LINEAR (shallow-first) images...")
        wrapper.manager.enabled = True
        wrapper.manager.layer_scheduler._build_schedule()

        linear_dir = generate_images(
            wrapper, sampler, vae, args.num_samples,
            output_dir, "linear_shallow", args.seed
        )
        print(f"  Saved to: {linear_dir}")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # === Generate LINEAR_REVERSE (deep-first) Images ===
        print("\nGenerating LINEAR_REVERSE (deep-first) images...")
        # Create new wrapper with reverse schedule
        dqar_config_reverse = DQARConfig(
            quantization_bits=16,
            use_layer_scheduling=True,
            schedule_type="linear_reverse",
        )
        wrapper_reverse = DQARDiTWrapper(model, dqar_config_reverse)
        sampler_reverse = DQARDDIMSampler(wrapper_reverse, scheduler, sampler_config)
        wrapper_reverse.manager.layer_scheduler._build_schedule()

        reverse_dir = generate_images(
            wrapper_reverse, sampler_reverse, vae, args.num_samples,
            output_dir, "linear_reverse_deep", args.seed
        )
        print(f"  Saved to: {reverse_dir}")

        # Restore original methods
        restore_layer_schedule(original_methods)

    else:
        baseline_dir = output_dir / "baseline"
        linear_dir = output_dir / "linear_shallow"
        reverse_dir = output_dir / "linear_reverse_deep"

        if not all(d.exists() for d in [baseline_dir, linear_dir, reverse_dir]):
            print("Error: Image directories not found. Run without --skip-generation first.")
            return

    # === Compute FID Scores ===
    print()
    print("=" * 60)
    print("COMPUTING FID SCORES")
    print("=" * 60)
    print()

    fid_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Computing FID: Baseline vs LINEAR (shallow-first)...")
    fid_linear = compute_fid(baseline_dir, linear_dir, fid_device)
    print(f"  FID (Linear/Shallow): {fid_linear:.2f}")

    print("\nComputing FID: Baseline vs LINEAR_REVERSE (deep-first)...")
    fid_reverse = compute_fid(baseline_dir, reverse_dir, fid_device)
    print(f"  FID (Reverse/Deep): {fid_reverse:.2f}")

    print("\nComputing FID: LINEAR vs LINEAR_REVERSE...")
    fid_cross = compute_fid(linear_dir, reverse_dir, fid_device)
    print(f"  FID (Linear vs Reverse): {fid_cross:.2f}")

    # === Print Results ===
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Configuration: {args.warmup_fraction:.0%} warmup, {args.layer_fraction:.0%} layers")
    print(f"Samples: {args.num_samples}")
    print()
    print(f"{'Comparison':<35} {'FID Score':>10}")
    print("-" * 50)
    print(f"{'Baseline vs LINEAR (shallow)':<35} {fid_linear:>10.2f}")
    print(f"{'Baseline vs LINEAR_REVERSE (deep)':<35} {fid_reverse:>10.2f}")
    print(f"{'LINEAR vs LINEAR_REVERSE':<35} {fid_cross:>10.2f}")
    print("-" * 50)

    # Determine winner
    if fid_linear < fid_reverse:
        winner = "LINEAR (shallow-first)"
        diff = fid_reverse - fid_linear
    elif fid_reverse < fid_linear:
        winner = "LINEAR_REVERSE (deep-first)"
        diff = fid_linear - fid_reverse
    else:
        winner = "Tie"
        diff = 0

    print()
    print(f"Better quality: {winner}")
    if diff > 0:
        print(f"FID difference: {diff:.2f}")

    # Save results to JSON
    results = {
        "config": {
            "warmup_fraction": args.warmup_fraction,
            "layer_fraction": args.layer_fraction,
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
            "seed": args.seed,
        },
        "fid_scores": {
            "baseline_vs_linear": fid_linear,
            "baseline_vs_reverse": fid_reverse,
            "linear_vs_reverse": fid_cross,
        },
        "winner": winner,
        "fid_difference": diff,
    }

    results_path = output_dir / "fid_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
