#!/usr/bin/env python3
"""
DQAR Inference Script

Run inference with a DiT model using DQAR optimization.
"""

import argparse
import torch
from pathlib import Path
import json
import time

from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARDPMSolverSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig
from dqar.utils import seed_everything, load_dit_model, save_images, get_device
from dqar.evaluation import DQARProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="DQAR Inference")

    # Model settings
    parser.add_argument("--model", type=str, default="facebook/DiT-XL-2-256",
                        help="DiT model name or path")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"])

    # Generation settings
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of samples to generate")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default="ddim",
                        choices=["ddim", "dpm_solver"])
    parser.add_argument("--class-label", type=int, default=None,
                        help="Class label for conditional generation")

    # DQAR settings
    parser.add_argument("--enable-dqar", action="store_true", default=True,
                        help="Enable DQAR optimization")
    parser.add_argument("--entropy-threshold", type=float, default=2.0,
                        help="Entropy threshold for reuse")
    parser.add_argument("--snr-low", type=float, default=0.1,
                        help="Lower SNR bound for reuse")
    parser.add_argument("--snr-high", type=float, default=10.0,
                        help="Upper SNR bound for reuse")
    parser.add_argument("--quantization-mode", type=str, default="per_tensor",
                        choices=["per_tensor", "per_channel", "per_head"])
    parser.add_argument("--policy-checkpoint", type=str, default=None,
                        help="Path to trained policy checkpoint")

    # Output settings
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory for generated images")
    parser.add_argument("--save-stats", action="store_true",
                        help="Save DQAR statistics to JSON")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    seed_everything(args.seed)
    device = args.device or get_device()
    dtype = getattr(torch, args.dtype)

    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Load model
    print("Loading model...")
    try:
        from diffusers import DiTPipeline

        pipe = DiTPipeline.from_pretrained(args.model, torch_dtype=dtype)
        pipe.to(device)

        model = pipe.transformer
        scheduler = pipe.scheduler
        vae = pipe.vae

    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Attempting to load transformer only...")
        model = load_dit_model(args.model, device, dtype)
        scheduler = None
        vae = None

    # Configure DQAR
    if args.enable_dqar:
        dqar_config = DQARConfig(
            entropy_threshold=args.entropy_threshold,
            snr_low=args.snr_low,
            snr_high=args.snr_high,
            quantization_mode=args.quantization_mode,
            use_learned_policy=args.policy_checkpoint is not None,
        )
        wrapped_model = DQARDiTWrapper(model, dqar_config)

        # Load policy if provided
        if args.policy_checkpoint:
            from dqar.policy import ReusePolicy
            wrapped_model.reuse_policy = ReusePolicy.load(args.policy_checkpoint, device)
    else:
        wrapped_model = model

    # Create sampler
    sampler_config = SamplerConfig(
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        enable_dqar=args.enable_dqar,
    )

    if scheduler is None:
        from dqar.utils import create_scheduler
        scheduler = create_scheduler(args.sampler)

    if args.sampler == "ddim":
        sampler = DQARDDIMSampler(wrapped_model, scheduler, sampler_config)
    else:
        sampler = DQARDPMSolverSampler(wrapped_model, scheduler, sampler_config)

    # Prepare class labels
    class_labels = None
    if args.class_label is not None:
        class_labels = torch.tensor([args.class_label] * args.num_samples, device=device)

    # Generate samples
    print(f"Generating {args.num_samples} samples with {args.num_steps} steps...")

    profiler = DQARProfiler(device=device) if args.profile else None

    start_time = time.time()

    if profiler:
        with profiler.profile_run():
            latents = sampler.sample(
                batch_size=args.num_samples,
                class_labels=class_labels,
                progress_bar=True,
            )
    else:
        latents = sampler.sample(
            batch_size=args.num_samples,
            class_labels=class_labels,
            progress_bar=True,
        )

    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f}s")

    # Decode latents if VAE is available
    if vae is not None:
        print("Decoding latents...")
        with torch.no_grad():
            latents = 1 / vae.config.scaling_factor * latents
            images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
    else:
        images = latents

    # Save images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_images(images, output_dir, prefix="dqar_sample")
    print(f"Saved images to {output_dir}")

    # Get and save statistics
    if args.enable_dqar and hasattr(sampler, 'get_statistics'):
        stats = sampler.get_statistics()

        print("\n--- DQAR Statistics ---")
        print(f"Reuse ratio: {stats.get('overall_reuse_ratio', 0):.2%}")
        if 'kv_cache' in stats:
            cache_stats = stats['kv_cache']
            print(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            mem = cache_stats.get('memory_usage', {})
            print(f"Cache memory: {mem.get('total_mb', 0):.2f} MB")

        if args.save_stats:
            stats_path = output_dir / "dqar_stats.json"
            # Convert non-serializable items
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(v) for v in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)

            with open(stats_path, 'w') as f:
                json.dump(make_serializable(stats), f, indent=2)
            print(f"Saved statistics to {stats_path}")

    if profiler:
        result = profiler.get_result(num_samples=args.num_samples)
        print(f"\n--- Profiling Results ---")
        print(f"Total time: {result.timing.total_time_ms:.2f} ms")
        print(f"Mean step time: {result.timing.mean_step_time_ms:.2f} ms")
        print(f"Peak memory: {result.memory.peak_allocated_mb:.2f} MB")
        print(f"Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")


if __name__ == "__main__":
    main()
