#!/usr/bin/env python3
"""
SNR Threshold Sweep Test

Sweeps over different SNR threshold values to find the optimal trade-off
between attention reuse ratio and image quality (FID).

Usage:
    python scripts/sweep_snr_threshold.py --num-samples 64 --output-dir results/snr_sweep
"""

import argparse
import torch
import gc
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.policy.layer_scheduler import LayerScheduler, SchedulerConfig
from dqar.utils import seed_everything, get_device

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from cleanfid import fid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False


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


def compute_fid(path1: Path, path2: Path, device: str = "cuda") -> float:
    """Compute FID score between two directories of images."""
    if HAS_CLEANFID:
        return fid.compute_fid(str(path1), str(path2), device=device)
    else:
        print("cleanfid not available, skipping FID computation")
        return -1.0


class SNRGatedScheduler(LayerScheduler):
    """Layer scheduler with SNR gating."""

    def __init__(
        self,
        config: SchedulerConfig,
        warmup_fraction: float = 0.4,
        layer_fraction: float = 0.33,
        snr_low: float = 0.1,
    ):
        self.warmup_fraction = warmup_fraction
        self.layer_fraction = layer_fraction
        self.snr_low = snr_low
        super().__init__(config)

    def _linear_schedule(self, progress: float) -> List[int]:
        if progress < self.warmup_fraction:
            return []
        adjusted = (progress - self.warmup_fraction) / (1 - self.warmup_fraction)
        max_layers = int(self.config.num_layers * self.layer_fraction)
        num_reusable = int(adjusted * max_layers)
        return list(range(num_reusable))

    def can_reuse(
        self,
        timestep_idx: int,
        layer_idx: int,
        snr: Optional[float] = None,
    ) -> bool:
        if not super().can_reuse(timestep_idx, layer_idx, snr=None):
            return False
        if snr is not None and snr < self.snr_low:
            return False
        return True


def run_with_snr_threshold(
    model,
    scheduler,
    vae,
    snr_threshold: float,
    num_samples: int,
    output_dir: Path,
    warmup_fraction: float,
    layer_fraction: float,
    seed: int,
    num_steps: int,
) -> Dict:
    """Generate images with a specific SNR threshold."""
    device = get_device()
    threshold_name = f"snr_{snr_threshold:.4f}".replace(".", "_")
    image_dir = output_dir / threshold_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create scheduler with SNR gating
    sched_config = SchedulerConfig(num_layers=28, num_timesteps=num_steps)
    snr_scheduler = SNRGatedScheduler(
        sched_config,
        warmup_fraction=warmup_fraction,
        layer_fraction=layer_fraction,
        snr_low=snr_threshold,
    )

    # Create wrapper
    dqar_config = DQARConfig(
        quantization_bits=16,
        use_layer_scheduling=True,
        schedule_type="linear",
        snr_low=snr_threshold,
        snr_high=float('inf'),
    )
    wrapper = DQARDiTWrapper(model, dqar_config)
    wrapper.manager.layer_scheduler = snr_scheduler
    wrapper.layer_scheduler = snr_scheduler

    sampler_config = SamplerConfig(num_inference_steps=num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    class_labels = [207, 360, 387, 974, 88, 417, 279, 928, 1, 9, 18, 25, 76, 130, 291, 388]

    total_reused = 0
    total_computed = 0
    snr_values = []

    for i in tqdm(range(num_samples), desc=f"    SNR>{snr_threshold:.4f}", leave=False):
        seed_everything(seed + i)
        wrapper.reset()

        class_label = class_labels[i % len(class_labels)]

        with torch.no_grad():
            latent = sampler.sample(
                batch_size=1,
                class_labels=torch.tensor([class_label], device=device),
                progress_bar=False,
            )

            latent_scaled = 1 / vae.config.scaling_factor * latent
            decoded = vae.decode(latent_scaled).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)

        stats = wrapper.manager.get_statistics()
        total_reused += stats["reused"]
        total_computed += stats["computed"]
        if wrapper.current_snr is not None:
            snr_values.append(wrapper.current_snr)

        img_path = image_dir / f"sample_{i:04d}.png"
        save_tensor_as_image(decoded, img_path)

    total_calls = total_reused + total_computed
    reuse_ratio = total_reused / total_calls if total_calls > 0 else 0

    return {
        "snr_threshold": snr_threshold,
        "image_dir": image_dir,
        "reuse_ratio": reuse_ratio,
        "total_reused": total_reused,
        "total_computed": total_computed,
        "mean_snr": sum(snr_values) / len(snr_values) if snr_values else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="SNR Threshold Sweep")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--layer-fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/snr_sweep")
    parser.add_argument("--compute-fid", action="store_true", help="Compute FID scores (slower)")
    args = parser.parse_args()

    if not HAS_PIL:
        print("Error: PIL required")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # SNR thresholds to test (log scale from very permissive to very strict)
    snr_thresholds = [
        0.0,      # No SNR gating (baseline)
        0.001,    # Very permissive
        0.01,     # Permissive
        0.05,     # Moderate
        0.1,      # Default
        0.25,     # Stricter
        0.5,      # Strict
        1.0,      # Very strict
        2.0,      # Extremely strict
        5.0,      # Almost no reuse in early steps
    ]

    print()
    print("=" * 70)
    print("SNR THRESHOLD SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Warmup: {args.warmup_fraction:.0%}, Layers: {args.layer_fraction:.0%}")
    print(f"Testing {len(snr_thresholds)} SNR thresholds")
    print(f"Samples per threshold: {args.num_samples}")
    print()

    # Load model
    print("Loading model...")
    from diffusers import DiTPipeline
    pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.to(device)
    model = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae
    print("Model loaded.\n")

    # Generate baseline (no DQAR)
    print("Generating BASELINE (no DQAR)...")
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    dqar_config = DQARConfig(quantization_bits=16, use_layer_scheduling=True)
    wrapper = DQARDiTWrapper(model, dqar_config)
    wrapper.manager.enabled = False
    sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    class_labels = [207, 360, 387, 974, 88, 417, 279, 928, 1, 9, 18, 25, 76, 130, 291, 388]

    for i in tqdm(range(args.num_samples), desc="    baseline", leave=False):
        seed_everything(args.seed + i)
        wrapper.reset()
        with torch.no_grad():
            latent = sampler.sample(
                batch_size=1,
                class_labels=torch.tensor([class_labels[i % len(class_labels)]], device=device),
                progress_bar=False,
            )
            latent_scaled = 1 / vae.config.scaling_factor * latent
            decoded = vae.decode(latent_scaled).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
        save_tensor_as_image(decoded, baseline_dir / f"sample_{i:04d}.png")

    print(f"  Baseline saved to: {baseline_dir}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # Run sweep
    results = []
    for snr_threshold in snr_thresholds:
        print(f"Testing SNR threshold > {snr_threshold}...")
        result = run_with_snr_threshold(
            model, scheduler, vae,
            snr_threshold=snr_threshold,
            num_samples=args.num_samples,
            output_dir=output_dir,
            warmup_fraction=args.warmup_fraction,
            layer_fraction=args.layer_fraction,
            seed=args.seed,
            num_steps=args.num_steps,
        )
        print(f"  Reuse ratio: {result['reuse_ratio']:.2%}")
        results.append(result)

        gc.collect()
        torch.cuda.empty_cache()

    # Compute FID if requested
    if args.compute_fid:
        print("\nComputing FID scores...")
        fid_device = "cuda" if torch.cuda.is_available() else "cpu"
        for result in results:
            fid_score = compute_fid(baseline_dir, result["image_dir"], fid_device)
            result["fid"] = fid_score
            print(f"  SNR > {result['snr_threshold']}: FID = {fid_score:.2f}")
    else:
        for result in results:
            result["fid"] = -1.0

    # Print summary
    print()
    print("=" * 70)
    print("SWEEP RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'SNR Threshold':<15} {'Reuse Ratio':>12} {'Reused':>10} {'Computed':>10} {'FID':>10}")
    print("-" * 60)
    for r in results:
        fid_str = f"{r['fid']:.2f}" if r['fid'] >= 0 else "N/A"
        print(f"{r['snr_threshold']:<15.4f} {r['reuse_ratio']*100:>11.2f}% {r['total_reused']:>10} {r['total_computed']:>10} {fid_str:>10}")
    print("-" * 60)

    # Find optimal threshold (highest reuse with acceptable FID)
    if args.compute_fid and any(r['fid'] >= 0 for r in results):
        # Sort by FID, find best quality with reasonable reuse
        valid_results = [r for r in results if r['fid'] >= 0]
        best_quality = min(valid_results, key=lambda r: r['fid'])
        best_reuse = max(valid_results, key=lambda r: r['reuse_ratio'])
        print(f"\nBest quality: SNR > {best_quality['snr_threshold']} (FID: {best_quality['fid']:.2f}, Reuse: {best_quality['reuse_ratio']:.2%})")
        print(f"Best reuse: SNR > {best_reuse['snr_threshold']} (FID: {best_reuse['fid']:.2f}, Reuse: {best_reuse['reuse_ratio']:.2%})")

    # Save results
    output_results = {
        "config": {
            "warmup_fraction": args.warmup_fraction,
            "layer_fraction": args.layer_fraction,
            "num_samples": args.num_samples,
            "num_steps": args.num_steps,
        },
        "thresholds": [
            {
                "snr_threshold": r["snr_threshold"],
                "reuse_ratio": r["reuse_ratio"],
                "total_reused": r["total_reused"],
                "total_computed": r["total_computed"],
                "fid": r["fid"],
            }
            for r in results
        ],
    }

    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate CSV for easy plotting
    csv_path = output_dir / "sweep_results.csv"
    with open(csv_path, "w") as f:
        f.write("snr_threshold,reuse_ratio,total_reused,total_computed,fid\n")
        for r in results:
            f.write(f"{r['snr_threshold']},{r['reuse_ratio']},{r['total_reused']},{r['total_computed']},{r['fid']}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
