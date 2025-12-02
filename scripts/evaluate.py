#!/usr/bin/env python3
"""
DQAR Evaluation Script

Evaluate DQAR performance against baselines using FID, CLIP score,
runtime, and VRAM metrics.
"""

import argparse
import torch
from pathlib import Path
import json
import time
from tqdm import tqdm

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.evaluation import FIDCalculator, CLIPScoreCalculator, DQARProfiler, profile_inference
from dqar.utils import seed_everything, load_prompts, save_images, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DQAR")

    # Model settings
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="DiT model name or path")

    # Evaluation settings
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of samples to generate for evaluation")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="File containing text prompts (for CLIP score)")
    parser.add_argument("--reference-stats", type=str, default=None,
                        help="Path to reference FID statistics (.npz)")

    # Ablation configurations
    parser.add_argument("--ablations", type=str, nargs="+",
                        default=["baseline", "entropy_gate", "quant_cache", "full_dqar"],
                        help="Ablation configurations to evaluate")

    # DQAR settings for full config
    parser.add_argument("--entropy-threshold", type=float, default=2.0)
    parser.add_argument("--snr-low", type=float, default=0.1)
    parser.add_argument("--snr-high", type=float, default=10.0)

    # Output settings
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated images")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of profiling runs")

    return parser.parse_args()


def create_config(ablation: str, args) -> DQARConfig:
    """Create DQAR config for specific ablation."""
    if ablation == "baseline":
        # No DQAR - use very restrictive thresholds
        return DQARConfig(
            entropy_threshold=0.0,  # Never reuse
            snr_low=float('inf'),
            snr_high=float('inf'),
        )
    elif ablation == "entropy_gate":
        # Only entropy-based gating, no quantization
        return DQARConfig(
            entropy_threshold=args.entropy_threshold,
            snr_low=0.0,
            snr_high=float('inf'),
            quantization_bits=16,  # Effectively no quantization
        )
    elif ablation == "quant_cache":
        # Only quantized caching, always reuse
        return DQARConfig(
            entropy_threshold=float('inf'),
            snr_low=0.0,
            snr_high=float('inf'),
            quantization_bits=8,
        )
    elif ablation == "full_dqar":
        # Full DQAR with all components
        return DQARConfig(
            entropy_threshold=args.entropy_threshold,
            snr_low=args.snr_low,
            snr_high=args.snr_high,
            quantization_bits=8,
            use_layer_scheduling=True,
        )
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def evaluate_config(
    model,
    scheduler,
    vae,
    config: DQARConfig,
    config_name: str,
    args,
    device: str,
    fid_calc: FIDCalculator,
    clip_calc: CLIPScoreCalculator,
    prompts: list,
) -> dict:
    """Evaluate a single configuration."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*50}")

    # Wrap model
    wrapped_model = DQARDiTWrapper(model, config)

    # Create sampler
    sampler_config = SamplerConfig(
        num_inference_steps=args.num_steps,
        guidance_scale=4.0,
    )
    sampler = DQARDDIMSampler(wrapped_model, scheduler, sampler_config)

    # Generate samples
    all_images = []
    all_latents = []
    generation_times = []

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Generating ({config_name})"):
        current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)

        # Random class labels for ImageNet DiT
        class_labels = torch.randint(0, 1000, (current_batch_size,), device=device)

        start_time = time.time()
        latents = sampler.sample(
            batch_size=current_batch_size,
            class_labels=class_labels,
            progress_bar=False,
        )
        elapsed = time.time() - start_time
        generation_times.append(elapsed)

        all_latents.append(latents)

        # Decode with VAE
        if vae is not None:
            with torch.no_grad():
                latents_scaled = 1 / vae.config.scaling_factor * latents
                images = vae.decode(latents_scaled).sample
                images = (images / 2 + 0.5).clamp(0, 1)
                all_images.append(images)

    # Concatenate all generated images
    if all_images:
        all_images = torch.cat(all_images, dim=0)
    else:
        all_images = torch.cat(all_latents, dim=0)

    # Get DQAR statistics
    dqar_stats = sampler.get_statistics() if hasattr(sampler, 'get_statistics') else {}

    # Calculate metrics
    results = {
        "config_name": config_name,
        "num_samples": len(all_images),
    }

    # Timing metrics
    total_time = sum(generation_times)
    results["total_time_s"] = total_time
    results["time_per_sample_s"] = total_time / args.num_samples
    results["samples_per_second"] = args.num_samples / total_time

    # DQAR-specific metrics
    results["reuse_ratio"] = dqar_stats.get("overall_reuse_ratio", 0.0)
    if "kv_cache" in dqar_stats:
        results["cache_hit_rate"] = dqar_stats["kv_cache"].get("hit_rate", 0.0)
        results["cache_memory_mb"] = dqar_stats["kv_cache"].get("memory_usage", {}).get("total_mb", 0.0)

    # Memory metrics
    if torch.cuda.is_available():
        results["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)

    # FID (if reference stats available)
    if args.reference_stats and Path(args.reference_stats).exists():
        print("Computing FID...")
        try:
            fid_score = fid_calc.compute(
                all_images,
                real_stats_path=args.reference_stats,
            )
            results["fid"] = fid_score
            print(f"FID: {fid_score:.2f}")
        except Exception as e:
            print(f"FID computation failed: {e}")
            results["fid"] = None

    # CLIP score (if prompts available)
    if prompts and len(prompts) >= len(all_images):
        print("Computing CLIP score...")
        try:
            clip_score = clip_calc.compute(
                all_images[:len(prompts)],
                prompts[:len(all_images)],
            )
            results["clip_score"] = clip_score
            print(f"CLIP score: {clip_score:.2f}")
        except Exception as e:
            print(f"CLIP computation failed: {e}")
            results["clip_score"] = None

    # Save images if requested
    if args.save_images:
        output_path = Path(args.output_dir) / config_name
        save_images(all_images[:16], output_path, prefix="sample")  # Save first 16

    return results


def main():
    args = parse_args()

    # Setup
    seed_everything(args.seed)
    device = args.device or get_device()
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    try:
        from diffusers import DiTPipeline

        pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
        pipe.to(device)

        model = pipe.transformer
        scheduler = pipe.scheduler
        vae = pipe.vae
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize metric calculators
    fid_calc = FIDCalculator(device=device)
    clip_calc = CLIPScoreCalculator(device=device)

    # Load prompts if available
    prompts = []
    if args.prompts_file and Path(args.prompts_file).exists():
        prompts = load_prompts(args.prompts_file, max_prompts=args.num_samples)
        print(f"Loaded {len(prompts)} prompts")

    # Run evaluations
    all_results = []

    for ablation in args.ablations:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        config = create_config(ablation, args)
        results = evaluate_config(
            model, scheduler, vae, config, ablation,
            args, device, fid_calc, clip_calc, prompts
        )
        all_results.append(results)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Create comparison table
    print(f"\n{'Config':<20} {'Time/Sample':<12} {'Reuse%':<10} {'Memory(MB)':<12} {'FID':<10} {'CLIP':<10}")
    print("-" * 80)

    baseline_time = None
    for r in all_results:
        if r["config_name"] == "baseline":
            baseline_time = r["time_per_sample_s"]

        speedup = ""
        if baseline_time and r["time_per_sample_s"] > 0:
            speedup = f" ({baseline_time/r['time_per_sample_s']:.2f}x)"

        fid_str = f"{r.get('fid', 'N/A'):.2f}" if r.get('fid') else "N/A"
        clip_str = f"{r.get('clip_score', 'N/A'):.2f}" if r.get('clip_score') else "N/A"

        print(f"{r['config_name']:<20} "
              f"{r['time_per_sample_s']:.3f}s{speedup:<6} "
              f"{r.get('reuse_ratio', 0)*100:.1f}%      "
              f"{r.get('peak_memory_mb', 0):.1f}       "
              f"{fid_str:<10} "
              f"{clip_str:<10}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
