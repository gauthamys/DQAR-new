#!/usr/bin/env python3
"""
Combined SNR + Entropy Threshold Sweep

Tests combinations of SNR and entropy thresholds to find optimal settings.

Reuse conditions:
- SNR > snr_threshold (signal is strong enough)
- Entropy < entropy_threshold (attention is focused enough)

Usage:
    python scripts/sweep_combined.py --num-samples 32 --output-dir results/combined_sweep
"""

import argparse
import torch
import gc
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
from itertools import product

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
        return -1.0


class CombinedGatedScheduler(LayerScheduler):
    """
    Layer scheduler with both SNR and entropy gating.

    Reuse only when:
    1. Progress > warmup
    2. SNR > snr_threshold
    3. Entropy < entropy_threshold (checked via gate)
    """

    def __init__(
        self,
        config: SchedulerConfig,
        warmup_fraction: float = 0.4,
        layer_fraction: float = 0.33,
        snr_threshold: float = 0.1,
    ):
        self.warmup_fraction = warmup_fraction
        self.layer_fraction = layer_fraction
        self.snr_threshold = snr_threshold
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
        if snr is not None and snr < self.snr_threshold:
            return False
        return True


def run_with_thresholds(
    model,
    scheduler,
    vae,
    snr_threshold: float,
    entropy_threshold: float,
    num_samples: int,
    output_dir: Path,
    warmup_fraction: float,
    layer_fraction: float,
    seed: int,
    num_steps: int,
) -> Dict:
    """Generate images with specific SNR and entropy thresholds."""
    device = get_device()
    threshold_name = f"snr_{snr_threshold:.3f}_ent_{entropy_threshold:.1f}".replace(".", "_")
    image_dir = output_dir / threshold_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create scheduler with SNR gating
    sched_config = SchedulerConfig(num_layers=28, num_timesteps=num_steps)
    combined_scheduler = CombinedGatedScheduler(
        sched_config,
        warmup_fraction=warmup_fraction,
        layer_fraction=layer_fraction,
        snr_threshold=snr_threshold,
    )

    # Create wrapper with entropy threshold
    dqar_config = DQARConfig(
        quantization_bits=16,
        use_layer_scheduling=True,
        schedule_type="linear",
        entropy_threshold=entropy_threshold,
        snr_low=snr_threshold,
        snr_high=float('inf'),
        adaptive_entropy=False,
    )
    wrapper = DQARDiTWrapper(model, dqar_config)
    wrapper.manager.layer_scheduler = combined_scheduler
    wrapper.layer_scheduler = combined_scheduler
    wrapper.reuse_gate.config.entropy_threshold = entropy_threshold

    sampler_config = SamplerConfig(num_inference_steps=num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    class_labels = [207, 360, 387, 974, 88, 417, 279, 928, 1, 9, 18, 25, 76, 130, 291, 388]

    total_reused = 0
    total_computed = 0

    for i in range(num_samples):
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

        img_path = image_dir / f"sample_{i:04d}.png"
        save_tensor_as_image(decoded, img_path)

    total_calls = total_reused + total_computed
    reuse_ratio = total_reused / total_calls if total_calls > 0 else 0

    return {
        "snr_threshold": snr_threshold,
        "entropy_threshold": entropy_threshold,
        "image_dir": image_dir,
        "reuse_ratio": reuse_ratio,
        "total_reused": total_reused,
        "total_computed": total_computed,
    }


def main():
    parser = argparse.ArgumentParser(description="Combined SNR + Entropy Sweep")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--layer-fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/combined_sweep")
    parser.add_argument("--compute-fid", action="store_true")
    args = parser.parse_args()

    if not HAS_PIL:
        print("Error: PIL required")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Grid of thresholds to test
    snr_thresholds = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    entropy_thresholds = [1.5, 2.0, 2.5, 3.0, 5.0, 10.0]

    combinations = list(product(snr_thresholds, entropy_thresholds))

    print()
    print("=" * 70)
    print("COMBINED SNR + ENTROPY THRESHOLD SWEEP")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Warmup: {args.warmup_fraction:.0%}, Layers: {args.layer_fraction:.0%}")
    print(f"SNR thresholds: {snr_thresholds}")
    print(f"Entropy thresholds: {entropy_thresholds}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Samples per combination: {args.num_samples}")
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

    # Generate baseline
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

    # Run grid search
    results = []
    for idx, (snr_t, ent_t) in enumerate(tqdm(combinations, desc="Grid search")):
        result = run_with_thresholds(
            model, scheduler, vae,
            snr_threshold=snr_t,
            entropy_threshold=ent_t,
            num_samples=args.num_samples,
            output_dir=output_dir,
            warmup_fraction=args.warmup_fraction,
            layer_fraction=args.layer_fraction,
            seed=args.seed,
            num_steps=args.num_steps,
        )
        results.append(result)

        if (idx + 1) % 6 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Compute FID if requested
    if args.compute_fid:
        print("\nComputing FID scores...")
        fid_device = "cuda" if torch.cuda.is_available() else "cpu"
        for result in tqdm(results, desc="FID"):
            fid_score = compute_fid(baseline_dir, result["image_dir"], fid_device)
            result["fid"] = fid_score
    else:
        for result in results:
            result["fid"] = -1.0

    # Print summary as grid
    print()
    print("=" * 70)
    print("REUSE RATIO GRID (SNR x Entropy)")
    print("=" * 70)
    print()

    # Header
    header = f"{'SNR \\ Ent':<10}"
    for ent_t in entropy_thresholds:
        header += f"{ent_t:>8.1f}"
    print(header)
    print("-" * (10 + 8 * len(entropy_thresholds)))

    # Grid
    for snr_t in snr_thresholds:
        row = f"{snr_t:<10.3f}"
        for ent_t in entropy_thresholds:
            r = next(r for r in results if r['snr_threshold'] == snr_t and r['entropy_threshold'] == ent_t)
            row += f"{r['reuse_ratio']*100:>7.1f}%"
        print(row)

    # Find best configurations
    print()
    best_reuse = max(results, key=lambda r: r['reuse_ratio'])
    print(f"Highest reuse: SNR>{best_reuse['snr_threshold']}, H<{best_reuse['entropy_threshold']} ({best_reuse['reuse_ratio']:.2%})")

    if args.compute_fid and any(r['fid'] >= 0 for r in results):
        valid = [r for r in results if r['fid'] >= 0]
        best_fid = min(valid, key=lambda r: r['fid'])
        print(f"Best FID: SNR>{best_fid['snr_threshold']}, H<{best_fid['entropy_threshold']} (FID: {best_fid['fid']:.2f}, Reuse: {best_fid['reuse_ratio']:.2%})")

        # Pareto frontier (best FID for each reuse level)
        print("\nPareto frontier (reuse vs quality):")
        sorted_by_reuse = sorted(valid, key=lambda r: r['reuse_ratio'], reverse=True)
        best_fid_so_far = float('inf')
        for r in sorted_by_reuse:
            if r['fid'] < best_fid_so_far:
                print(f"  SNR>{r['snr_threshold']}, H<{r['entropy_threshold']}: Reuse={r['reuse_ratio']:.2%}, FID={r['fid']:.2f}")
                best_fid_so_far = r['fid']

    # Save results
    output_results = {
        "config": {
            "warmup_fraction": args.warmup_fraction,
            "layer_fraction": args.layer_fraction,
            "num_samples": args.num_samples,
            "snr_thresholds": snr_thresholds,
            "entropy_thresholds": entropy_thresholds,
        },
        "results": [
            {
                "snr_threshold": r["snr_threshold"],
                "entropy_threshold": r["entropy_threshold"],
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

    # CSV
    csv_path = output_dir / "sweep_results.csv"
    with open(csv_path, "w") as f:
        f.write("snr_threshold,entropy_threshold,reuse_ratio,total_reused,total_computed,fid\n")
        for r in results:
            f.write(f"{r['snr_threshold']},{r['entropy_threshold']},{r['reuse_ratio']},{r['total_reused']},{r['total_computed']},{r['fid']}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
