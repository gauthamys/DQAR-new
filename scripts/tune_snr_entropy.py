#!/usr/bin/env python3
"""
SNR & Entropy Threshold Tuning with Optuna

Multi-objective Bayesian optimization to find optimal SNR and entropy thresholds
that maximize speedup (reuse ratio) while minimizing quality degradation (FID).

Uses Optuna with NSGA-II for Pareto frontier discovery.

Usage:
    # Quick tuning (fewer trials)
    python scripts/tune_snr_entropy.py --n-trials 30 --num-samples 16

    # Full tuning with FID
    python scripts/tune_snr_entropy.py --n-trials 100 --num-samples 64 --compute-fid

    # Resume previous study
    python scripts/tune_snr_entropy.py --resume --study-name snr_entropy_tuning
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
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.policy.layer_scheduler import LayerScheduler, SchedulerConfig
from dqar.utils import seed_everything, get_device

try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIISampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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


def compute_fid_score(path1: Path, path2: Path, device: str = "cuda") -> float:
    """Compute FID score between two directories."""
    if HAS_CLEANFID:
        try:
            return fid.compute_fid(str(path1), str(path2), device=device)
        except Exception as e:
            print(f"FID computation failed: {e}")
            return 100.0
    return -1.0


class SNREntropyScheduler(LayerScheduler):
    """Layer scheduler with tunable SNR and entropy thresholds."""

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


class SNREntropyObjective:
    """Objective function for Optuna optimization."""

    def __init__(
        self,
        model,
        scheduler,
        vae,
        baseline_dir: Path,
        output_dir: Path,
        num_samples: int,
        num_steps: int,
        warmup_fraction: float,
        layer_fraction: float,
        seed: int,
        compute_fid: bool,
        device: str,
    ):
        self.model = model
        self.scheduler = scheduler
        self.vae = vae
        self.baseline_dir = baseline_dir
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.warmup_fraction = warmup_fraction
        self.layer_fraction = layer_fraction
        self.seed = seed
        self.compute_fid = compute_fid
        self.device = device
        self.trial_count = 0

        self.class_labels = [207, 360, 387, 974, 88, 417, 279, 928,
                            1, 9, 18, 25, 76, 130, 291, 388]

    def __call__(self, trial: "optuna.Trial") -> Tuple[float, float]:
        """
        Evaluate hyperparameters.

        Returns: (negative_reuse_ratio, fid_score)
        - We negate reuse_ratio because Optuna minimizes
        - Lower FID is better
        """
        # Sample hyperparameters
        snr_threshold = trial.suggest_float("snr_threshold", 0.0, 2.0)
        entropy_threshold = trial.suggest_float("entropy_threshold", 0.5, 5.0)

        self.trial_count += 1
        trial_dir = self.output_dir / f"trial_{self.trial_count:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create scheduler
        sched_config = SchedulerConfig(num_layers=28, num_timesteps=self.num_steps)
        custom_scheduler = SNREntropyScheduler(
            sched_config,
            warmup_fraction=self.warmup_fraction,
            layer_fraction=self.layer_fraction,
            snr_threshold=snr_threshold,
        )

        # Create wrapper
        dqar_config = DQARConfig(
            quantization_bits=16,
            use_layer_scheduling=True,
            schedule_type="linear",
            entropy_threshold=entropy_threshold,
            snr_low=snr_threshold,
            snr_high=float('inf'),
            adaptive_entropy=False,
        )
        wrapper = DQARDiTWrapper(self.model, dqar_config)
        wrapper.manager.layer_scheduler = custom_scheduler
        wrapper.layer_scheduler = custom_scheduler
        wrapper.reuse_gate.config.entropy_threshold = entropy_threshold

        sampler_config = SamplerConfig(num_inference_steps=self.num_steps)
        sampler = DQARDDIMSampler(wrapper, self.scheduler, sampler_config)

        total_reused = 0
        total_computed = 0

        # Generate samples
        for i in range(self.num_samples):
            seed_everything(self.seed + i)
            wrapper.reset()

            class_label = self.class_labels[i % len(self.class_labels)]

            with torch.no_grad():
                latent = sampler.sample(
                    batch_size=1,
                    class_labels=torch.tensor([class_label], device=self.device),
                    progress_bar=False,
                )

                latent_scaled = 1 / self.vae.config.scaling_factor * latent
                decoded = self.vae.decode(latent_scaled).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)

            stats = wrapper.manager.get_statistics()
            total_reused += stats["reused"]
            total_computed += stats["computed"]

            img_path = trial_dir / f"sample_{i:04d}.png"
            save_tensor_as_image(decoded, img_path)

        total_calls = total_reused + total_computed
        reuse_ratio = total_reused / total_calls if total_calls > 0 else 0

        # Compute FID
        if self.compute_fid and HAS_CLEANFID:
            fid_score = compute_fid_score(self.baseline_dir, trial_dir, self.device)
        else:
            # Proxy: estimate quality impact based on reuse aggressiveness
            # Higher reuse at low SNR = likely worse quality
            fid_score = max(0, reuse_ratio * 100 - 5)  # Rough proxy

        # Store metadata
        trial.set_user_attr("total_reused", total_reused)
        trial.set_user_attr("total_computed", total_computed)
        trial.set_user_attr("reuse_ratio", reuse_ratio)
        trial.set_user_attr("trial_dir", str(trial_dir))

        # Cleanup
        del wrapper, sampler
        gc.collect()
        torch.cuda.empty_cache()

        return (-reuse_ratio, fid_score)


def generate_baseline(model, scheduler, vae, baseline_dir, num_samples, num_steps, seed, device):
    """Generate baseline images."""
    baseline_dir.mkdir(parents=True, exist_ok=True)

    dqar_config = DQARConfig(quantization_bits=16, use_layer_scheduling=True)
    wrapper = DQARDiTWrapper(model, dqar_config)
    wrapper.manager.enabled = False
    sampler_config = SamplerConfig(num_inference_steps=num_steps)
    sampler = DQARDDIMSampler(wrapper, scheduler, sampler_config)

    class_labels = [207, 360, 387, 974, 88, 417, 279, 928, 1, 9, 18, 25, 76, 130, 291, 388]

    print("Generating baseline images...")
    for i in tqdm(range(num_samples), desc="Baseline"):
        seed_everything(seed + i)
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

    print(f"Baseline saved to: {baseline_dir}")


def main():
    parser = argparse.ArgumentParser(description="SNR & Entropy Threshold Tuning")
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--layer-fraction", type=float, default=0.33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/snr_entropy_tuning")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--study-name", type=str, default="snr_entropy_tuning")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not HAS_OPTUNA:
        print("Error: Optuna required. Install: pip install optuna")
        return

    if not HAS_PIL:
        print("Error: PIL required")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"

    device = get_device()

    print()
    print("=" * 70)
    print("SNR & ENTROPY THRESHOLD TUNING")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Trials: {args.n_trials}")
    print(f"Samples/trial: {args.num_samples}")
    print(f"Compute FID: {args.compute_fid}")
    print(f"Base config: warmup={args.warmup_fraction}, layers={args.layer_fraction}")
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
    if not baseline_dir.exists() or len(list(baseline_dir.glob("*.png"))) < args.num_samples:
        generate_baseline(model, scheduler, vae, baseline_dir,
                         args.num_samples, args.num_steps, args.seed, device)
    else:
        print(f"Using existing baseline: {baseline_dir}")

    gc.collect()
    torch.cuda.empty_cache()

    # Create objective
    objective = SNREntropyObjective(
        model=model,
        scheduler=scheduler,
        vae=vae,
        baseline_dir=baseline_dir,
        output_dir=output_dir / "trials",
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        warmup_fraction=args.warmup_fraction,
        layer_fraction=args.layer_fraction,
        seed=args.seed,
        compute_fid=args.compute_fid,
        device=device,
    )

    # Create study (multi-objective)
    storage = f"sqlite:///{output_dir / 'optuna_study.db'}"
    sampler = NSGAIISampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=args.resume,
        directions=["minimize", "minimize"],  # minimize -reuse (maximize reuse), minimize FID
        sampler=sampler,
    )

    # Optimize
    print("\nStarting optimization...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Print Pareto frontier
    print()
    print("=" * 70)
    print("PARETO FRONTIER (Trade-off: Reuse vs Quality)")
    print("=" * 70)
    print()
    print(f"{'SNR Thresh':<12} {'Entropy Thresh':<15} {'Reuse %':>10} {'FID':>10}")
    print("-" * 55)

    pareto_trials = study.best_trials
    pareto_results = []

    for trial in sorted(pareto_trials, key=lambda t: t.values[0]):
        snr_t = trial.params["snr_threshold"]
        ent_t = trial.params["entropy_threshold"]
        reuse = -trial.values[0] * 100
        fid_val = trial.values[1]
        print(f"{snr_t:<12.4f} {ent_t:<15.4f} {reuse:>9.2f}% {fid_val:>10.2f}")
        pareto_results.append({
            "snr_threshold": snr_t,
            "entropy_threshold": ent_t,
            "reuse_ratio": -trial.values[0],
            "fid": fid_val,
        })

    # Find recommended configuration
    if pareto_results:
        reuses = [r["reuse_ratio"] for r in pareto_results]
        fids = [r["fid"] for r in pareto_results]

        if max(reuses) > min(reuses) and max(fids) > min(fids):
            # Normalize
            norm_reuses = [(r - min(reuses)) / (max(reuses) - min(reuses) + 1e-8) for r in reuses]
            norm_fids = [(f - min(fids)) / (max(fids) - min(fids) + 1e-8) for f in fids]

            # Distance from ideal (1, 0)
            distances = [np.sqrt((1 - nr)**2 + nf**2) for nr, nf in zip(norm_reuses, norm_fids)]
            best_idx = np.argmin(distances)
            recommended = pareto_results[best_idx]

            print()
            print("=" * 70)
            print("RECOMMENDED CONFIGURATION")
            print("=" * 70)
            print(f"  SNR threshold:     > {recommended['snr_threshold']:.4f}")
            print(f"  Entropy threshold: < {recommended['entropy_threshold']:.4f}")
            print(f"  Expected reuse:    {recommended['reuse_ratio']*100:.2f}%")
            print(f"  Expected FID:      {recommended['fid']:.2f}")
            print()
            print("Python code:")
            print("```python")
            print(f"snr_threshold = {recommended['snr_threshold']:.4f}")
            print(f"entropy_threshold = {recommended['entropy_threshold']:.4f}")
            print("```")
        else:
            recommended = pareto_results[0] if pareto_results else None
    else:
        recommended = None

    # Save results
    results = {
        "config": {
            "n_trials": args.n_trials,
            "num_samples": args.num_samples,
            "warmup_fraction": args.warmup_fraction,
            "layer_fraction": args.layer_fraction,
            "compute_fid": args.compute_fid,
        },
        "pareto_frontier": pareto_results,
        "recommended": recommended,
        "all_trials": [
            {
                "trial": t.number,
                "snr_threshold": t.params.get("snr_threshold"),
                "entropy_threshold": t.params.get("entropy_threshold"),
                "reuse_ratio": -t.values[0] if t.values else None,
                "fid": t.values[1] if t.values and len(t.values) > 1 else None,
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # CSV export
    csv_path = output_dir / "all_trials.csv"
    with open(csv_path, "w") as f:
        f.write("trial,snr_threshold,entropy_threshold,reuse_ratio,fid\n")
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                snr = t.params.get("snr_threshold", 0)
                ent = t.params.get("entropy_threshold", 0)
                reuse = -t.values[0] if t.values else 0
                fid_val = t.values[1] if t.values and len(t.values) > 1 else 0
                f.write(f"{t.number},{snr},{ent},{reuse},{fid_val}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
