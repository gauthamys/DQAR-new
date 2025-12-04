#!/usr/bin/env python3
"""
Adaptive Scheduling Experiments with Entropy and SNR

Compares:
1. Static schedule (warmup + layer fraction only)
2. SNR-gated schedule (also checks SNR range before reusing)
3. Entropy-adaptive schedule (adjusts based on attention entropy)

Usage:
    python scripts/evaluate_adaptive.py --num-samples 64 --output-dir results/adaptive
"""

import argparse
import torch
import gc
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.policy.layer_scheduler import LayerScheduler, AdaptiveLayerScheduler, SchedulerConfig, ScheduleType
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
    try:
        from pytorch_fid import fid_score
        HAS_PYTORCH_FID = True
    except ImportError:
        HAS_PYTORCH_FID = False


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
    elif HAS_PYTORCH_FID:
        return fid_score.calculate_fid_given_paths(
            [str(path1), str(path2)], batch_size=50, device=device, dims=2048
        )
    else:
        print("No FID library available")
        return -1.0


class SNRGatedScheduler(LayerScheduler):
    """
    Layer scheduler that also checks SNR before allowing reuse.

    Reuse only allowed when:
    1. Progress > warmup (base condition)
    2. SNR is in the "stable" range [snr_low, snr_high]
    """

    def __init__(
        self,
        config: SchedulerConfig,
        warmup_fraction: float = 0.4,
        layer_fraction: float = 0.33,
        snr_low: float = 0.5,
        snr_high: float = 5.0,
    ):
        self.warmup_fraction = warmup_fraction
        self.layer_fraction = layer_fraction
        self.snr_low = snr_low
        self.snr_high = snr_high
        super().__init__(config)

    def _linear_schedule(self, progress: float) -> List[int]:
        """Linear schedule with configurable warmup and layer fraction."""
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
        """Check if reuse is allowed, with SNR gating."""
        # First check base schedule
        if not super().can_reuse(timestep_idx, layer_idx, snr=None):
            return False

        # Then check SNR gate
        if snr is not None:
            if snr < self.snr_low or snr > self.snr_high:
                return False

        return True


def generate_images_with_scheduler(
    model,
    scheduler,
    vae,
    dqar_scheduler: LayerScheduler,
    num_samples: int,
    output_dir: Path,
    config_name: str,
    seed: int = 42,
    num_steps: int = 50,
    use_snr_gate: bool = False,
    snr_low: float = 0.5,
    snr_high: float = 5.0,
) -> Dict:
    """Generate images with a specific layer scheduler."""
    device = get_device()
    image_dir = output_dir / config_name
    image_dir.mkdir(parents=True, exist_ok=True)

    # Create wrapper with custom scheduler
    dqar_config = DQARConfig(
        quantization_bits=16,
        use_layer_scheduling=True,
        schedule_type="linear",
        snr_low=snr_low,
        snr_high=snr_high,
    )
    wrapper = DQARDiTWrapper(model, dqar_config)

    # Replace scheduler
    wrapper.manager.layer_scheduler = dqar_scheduler
    wrapper.layer_scheduler = dqar_scheduler

    sampler_config = SamplerConfig(num_inference_steps=num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    class_labels = [207, 360, 387, 974, 88, 417, 279, 928, 1, 9, 18, 25, 76, 130, 291, 388]

    total_reused = 0
    total_computed = 0
    snr_values = []
    debug_snr_printed = False

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

        # Collect stats
        stats = wrapper.manager.get_statistics()
        total_reused += stats["reused"]
        total_computed += stats["computed"]
        if wrapper.current_snr is not None:
            snr_values.append(wrapper.current_snr)

        # Debug: print SNR info for first sample
        if i == 0 and not debug_snr_printed:
            print(f"\n  [DEBUG] SNR at final step: {wrapper.current_snr:.6f}")
            print(f"  [DEBUG] Actual diffusion timestep: {getattr(wrapper, 'actual_diffusion_timestep', 'N/A')}")
            print(f"  [DEBUG] Step index: {wrapper.timestep_idx}")
            print(f"  [DEBUG] Manager SNR: {wrapper.manager.current_snr}")
            print(f"  [DEBUG] Stats: reused={stats['reused']}, computed={stats['computed']}")
            # Print SNR history from snr_computer
            snr_history = wrapper.snr_computer.get_history()
            if snr_history:
                print(f"  [DEBUG] SNR at step 0: {snr_history[0][1]:.6f} (should be LOW for early noisy steps)")
                if len(snr_history) > 25:
                    print(f"  [DEBUG] SNR at step 25: {snr_history[25][1]:.6f}")
                print(f"  [DEBUG] SNR at last step: {snr_history[-1][1]:.6f} (should be HIGH for clean steps)")
            debug_snr_printed = True

        # Save image
        img_path = image_dir / f"sample_{i:04d}.png"
        save_tensor_as_image(decoded, img_path)

    total_calls = total_reused + total_computed
    reuse_ratio = total_reused / total_calls if total_calls > 0 else 0

    return {
        "image_dir": image_dir,
        "reuse_ratio": reuse_ratio,
        "total_reused": total_reused,
        "total_computed": total_computed,
        "mean_snr": sum(snr_values) / len(snr_values) if snr_values else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive Scheduling Experiments")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--layer-fraction", type=float, default=0.33)
    parser.add_argument("--snr-low", type=float, default=0.1)
    parser.add_argument("--snr-high", type=float, default=float('inf'))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/adaptive")
    args = parser.parse_args()

    if not HAS_PIL:
        print("Error: PIL required")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print()
    print("=" * 70)
    print("ADAPTIVE SCHEDULING EXPERIMENTS: Static vs SNR-Gated")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Warmup: {args.warmup_fraction:.0%}, Layers: {args.layer_fraction:.0%}")
    print(f"SNR Gate: > {args.snr_low} (block early noisy steps)")
    print(f"Samples: {args.num_samples}")
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

    results = {}

    # === 1. Generate Baseline Images ===
    print("1. Generating BASELINE (no DQAR)...")
    dqar_config = DQARConfig(quantization_bits=16, use_layer_scheduling=True)
    wrapper = DQARDiTWrapper(model, dqar_config)
    wrapper.manager.enabled = False
    sampler_config = SamplerConfig(num_inference_steps=args.num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"  Saved to: {baseline_dir}")
    results["baseline"] = {"image_dir": baseline_dir}

    gc.collect()
    torch.cuda.empty_cache()

    # === 2. Static Schedule (warmup + layer fraction only) ===
    print("\n2. Generating STATIC SCHEDULE (warmup + layer fraction)...")

    static_config = SchedulerConfig(num_layers=28, num_timesteps=args.num_steps)
    static_scheduler = SNRGatedScheduler(
        static_config,
        warmup_fraction=args.warmup_fraction,
        layer_fraction=args.layer_fraction,
        snr_low=0.0,  # No SNR gating
        snr_high=float('inf'),
    )

    static_result = generate_images_with_scheduler(
        model, scheduler, vae, static_scheduler,
        args.num_samples, output_dir, "static",
        seed=args.seed, num_steps=args.num_steps,
    )
    print(f"  Reuse ratio: {static_result['reuse_ratio']:.2%}")
    results["static"] = static_result

    gc.collect()
    torch.cuda.empty_cache()

    # === 3. SNR-Gated Schedule ===
    print(f"\n3. Generating SNR-GATED SCHEDULE (SNR > {args.snr_low})...")

    snr_config = SchedulerConfig(num_layers=28, num_timesteps=args.num_steps)
    snr_scheduler = SNRGatedScheduler(
        snr_config,
        warmup_fraction=args.warmup_fraction,
        layer_fraction=args.layer_fraction,
        snr_low=args.snr_low,
        snr_high=args.snr_high,
    )

    snr_result = generate_images_with_scheduler(
        model, scheduler, vae, snr_scheduler,
        args.num_samples, output_dir, "snr_gated",
        seed=args.seed, num_steps=args.num_steps,
        use_snr_gate=True, snr_low=args.snr_low, snr_high=args.snr_high,
    )
    print(f"  Reuse ratio: {snr_result['reuse_ratio']:.2%}")
    results["snr_gated"] = snr_result

    gc.collect()
    torch.cuda.empty_cache()

    # === 4. Conservative SNR-Gated (block very early steps) ===
    print(f"\n4. Generating CONSERVATIVE SNR-GATED (SNR > 0.5 only)...")

    tight_snr_scheduler = SNRGatedScheduler(
        snr_config,
        warmup_fraction=args.warmup_fraction,
        layer_fraction=args.layer_fraction,
        snr_low=0.5,  # Block very early noisy steps
        snr_high=float('inf'),  # No upper limit
    )

    tight_result = generate_images_with_scheduler(
        model, scheduler, vae, tight_snr_scheduler,
        args.num_samples, output_dir, "snr_tight",
        seed=args.seed, num_steps=args.num_steps,
    )
    print(f"  Reuse ratio: {tight_result['reuse_ratio']:.2%}")
    results["snr_tight"] = tight_result

    gc.collect()
    torch.cuda.empty_cache()

    # === Compute FID Scores ===
    print()
    print("=" * 70)
    print("COMPUTING FID SCORES")
    print("=" * 70)

    fid_device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in ["static", "snr_gated", "snr_tight"]:
        print(f"\nComputing FID: Baseline vs {name}...")
        fid_score = compute_fid(baseline_dir, results[name]["image_dir"], fid_device)
        results[name]["fid"] = fid_score
        print(f"  FID ({name}): {fid_score:.2f}")

    # === Print Summary ===
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Configuration':<25} {'Reuse Ratio':>12} {'FID Score':>12}")
    print("-" * 55)
    print(f"{'Baseline (no DQAR)':<25} {'0%':>12} {'0.00':>12}")
    for name in ["static", "snr_gated", "snr_tight"]:
        r = results[name]
        print(f"{name:<25} {r['reuse_ratio']*100:>11.1f}% {r['fid']:>12.2f}")
    print("-" * 55)

    # Determine best
    best_name = min(["static", "snr_gated", "snr_tight"], key=lambda n: results[n]["fid"])
    print(f"\nBest quality: {best_name} (FID: {results[best_name]['fid']:.2f})")

    # Save results
    output_results = {
        "config": {
            "warmup_fraction": args.warmup_fraction,
            "layer_fraction": args.layer_fraction,
            "snr_low": args.snr_low,
            "snr_high": args.snr_high,
            "num_samples": args.num_samples,
        },
        "results": {
            name: {
                "reuse_ratio": r["reuse_ratio"],
                "fid": r.get("fid", -1),
                "total_reused": r.get("total_reused", 0),
                "total_computed": r.get("total_computed", 0),
            }
            for name, r in results.items() if name != "baseline"
        },
        "best": best_name,
    }

    results_path = output_dir / "adaptive_results.json"
    with open(results_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
