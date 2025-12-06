# CS 494 - Generative AI - Course Project

## Video
https://www.youtube.com/watch?v=7pKXLgNDdyk

## Report
[Project Report (PDF)](CS494-Report-Gautham-Satyanarayana.pdf)

# DQAR: Dynamic and Quantization-Aware Attention Reuse for Diffusion Transformers

A unified framework combining entropy- and SNR-based attention reuse with low-bit quantized KV caching for efficient Diffusion Transformer (DiT) inference.

## Overview

DQAR reduces the computational cost of DiT inference by:

1. **Entropy & SNR-Based Reuse Gate**: Dynamically decides when to reuse cached attention based on attention entropy and signal-to-noise ratio
2. **Quantized KV Caching**: Stores Key/Value tensors in 8-bit integer form to reduce VRAM usage
3. **Layer Scheduling**: Timestep-aware layer selection for attention reuse (linear, exponential, step, or custom schedules)
4. **Optional Learned Policy**: Lightweight MLP (<0.5M params) that predicts optimal reuse decisions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dqar.git
cd dqar

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- diffusers >= 0.25.0
- transformers >= 4.36.0

## Quick Start

### Basic Inference with DQAR

```python
import torch
from diffusers import DiTPipeline
from dqar import DQARDiTWrapper
from dqar.integration import DQARDDIMSampler, DQARConfig
from dqar.integration.samplers import SamplerConfig

# Load DiT model
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.to("cuda")

# Configure DQAR
dqar_config = DQARConfig(
    entropy_threshold=2.0,
    snr_low=0.1,
    snr_high=10.0,
    quantization_mode="per_tensor",
    quantization_bits=8,
    use_layer_scheduling=True,
    schedule_type="step",
    warmup_steps=5,
)

# Wrap model with DQAR
wrapped_model = DQARDiTWrapper(pipe.transformer, dqar_config)

# Create sampler
sampler_config = SamplerConfig(
    num_inference_steps=50,
    guidance_scale=4.0,
    enable_dqar=True,
)
sampler = DQARDDIMSampler(wrapped_model, pipe.scheduler, sampler_config)

# Generate images
latents = sampler.sample(
    batch_size=4,
    class_labels=torch.tensor([207, 360, 387, 974], device="cuda"),
    progress_bar=True,
)

# Decode with VAE
images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
```

### Using Scripts

```bash
# Run inference
python scripts/run_inference.py \
    --model facebook/DiT-XL-2-256 \
    --num-samples 4 \
    --num-steps 50 \
    --enable-dqar \
    --output-dir ./outputs

# Train reuse policy
python scripts/train_policy.py \
    --data-path ./training_data.json \
    --output-dir ./policy_checkpoints \
    --num-epochs 100

# Evaluate with ablations
python scripts/evaluate.py \
    --model facebook/DiT-XL-2-256 \
    --num-samples 256 \
    --ablations baseline entropy_gate quant_cache full_dqar

# Run hyperparameter sweeps
python scripts/layer_sweep.py --output-dir results/layer_sweep
python scripts/warmup_sweep.py --output-dir results/warmup_sweep
python scripts/sweep_entropy_threshold.py --output-dir results/entropy_sweep
python scripts/sweep_snr_threshold.py --output-dir results/snr_sweep
```

## Architecture

```
dqar/
├── core/
│   ├── entropy.py          # Attention entropy computation (AttentionEntropyComputer)
│   ├── snr.py               # Signal-to-noise ratio computation (SNRComputer)
│   ├── reuse_gate.py        # Reuse decision logic (ReuseGate, GateConfig)
│   └── quantization.py      # KV tensor quantization (KVQuantizer, MixedPrecisionKVQuantizer)
├── cache/
│   └── kv_cache.py          # Quantized KV cache management (QuantizedKVCache, LayerWiseKVCache)
├── policy/
│   ├── mlp_policy.py        # Learned reuse policy (ReusePolicy, LayerWiseReusePolicy)
│   └── layer_scheduler.py   # Layer-wise scheduling (LayerScheduler, AdaptiveLayerScheduler)
├── integration/
│   ├── dit_wrapper.py       # DiT model wrapper (DQARDiTWrapper, DQARConfig)
│   ├── manager.py           # State coordination across layers (DQARManager)
│   ├── attention_processor.py # Attention computation with reuse (DQARAttentionProcessor)
│   └── samplers.py          # DDIM/DPM-Solver samplers (DQARDDIMSampler, DQARDPMSolverSampler)
├── evaluation/
│   ├── metrics.py           # FID, CLIP score computation
│   └── profiling.py         # Runtime/memory profiling (DQARProfiler)
├── training/
│   └── policy_trainer.py    # Policy training pipeline (PolicyTrainer)
└── utils/
    └── helpers.py           # Utilities (seed_everything, load_dit_model, save_images, etc.)

scripts/
├── run_inference.py         # Main inference script
├── train_policy.py          # Policy training
├── evaluate.py              # Evaluation with ablations
├── run_ablations.py         # Ablation studies
├── layer_sweep.py           # Layer fraction hyperparameter sweep
├── warmup_sweep.py          # Warmup period hyperparameter sweep
├── sweep_entropy_threshold.py # Entropy threshold sweep
├── sweep_snr_threshold.py   # SNR threshold sweep
├── sweep_combined.py        # Combined hyperparameter sweep
├── tune_snr_entropy.py      # Joint SNR/entropy tuning
├── tune_hyperparameters.py  # General hyperparameter tuning
├── evaluate_fid.py          # FID evaluation
├── evaluate_adaptive.py     # Adaptive scheduler evaluation
├── memory_profile.py        # Memory profiling
└── benchmark_config.py      # Benchmark configurations
```

## Key Components

### 1. Entropy & SNR-Based Reuse Gate

At each timestep, DQAR computes:

- **Attention Entropy**: Measures attention distribution focus
  ```
  H_t = -1/(H*T) * Σ_h Σ_{i,j} A_t^(h)(i,j) * log(A_t^(h)(i,j) + ε)
  ```
- **Latent SNR**: Signal-to-noise ratio from diffusion schedule or prediction
  ```
  SNR_t = α_bar_t / (1 - α_bar_t)  [from schedule]
  SNR_t = ||x_0||² / (||x_t - x_0||² + ε)  [from prediction]
  ```

Attention is reused only if:
- `H_t < τ(p)` where `τ(p)` adapts to prompt length: `threshold = base * sqrt(prompt_len / 77)`
- `SNR_t ∈ [snr_low, snr_high]` (default: [0.1, 10.0])
- Current timestep > warmup_steps (default: 5)

### 2. Quantized KV Caching

Cached K/V tensors are stored in 8-bit integer form:

```
K_q = clip(round(K / s_K), -127, 127)
s_K = max|K| / 127
```

Supports three quantization modes:
- **per_tensor**: Fastest, single scale for entire tensor
- **per_channel**: Best quality, scale per channel dimension
- **per_head**: Balance, scale per attention head

### 3. Layer Scheduling

Timestep-dependent layer selection with multiple strategies:

- **linear**: Progressively enable more layers as timestep decreases
- **linear_reverse**: Start with deep layers, progress to shallow
- **exponential**: Slow start, rapid ramp-up
- **step**: Discrete phases (early/mid/late with configurable boundaries)
- **custom**: User-defined per-layer schedule

Default behavior: Early timesteps (high noise) reuse only shallow blocks; later timesteps (stabilized signal) can reuse deeper blocks.

### 4. Dynamic Reuse Policy (Optional)

A lightweight MLP predicts reuse probability:

```
p_reuse = P_θ(concat(H_t, SNR_t, ||x_t||₂, t))
```

The policy is trained offline on inference traces labeled by quality impact using quality-weighted loss.

## Configuration

See `configs/default.yaml` for full configuration options:

```yaml
model:
  name: "facebook/DiT-XL-2-256"
  num_layers: 28
  num_heads: 16
  dtype: "float16"

reuse_gate:
  entropy_threshold: 2.0
  snr_low: 0.1
  snr_high: 10.0
  adaptive_entropy: true
  base_prompt_length: 77
  warmup_steps: 2

quantization:
  mode: "per_tensor"  # per_tensor, per_channel, per_head
  bits: 8
  cache_attention_output: true

layer_scheduling:
  enabled: true
  schedule_type: "step"  # linear, exponential, step
  early_phase_end: 0.3
  mid_phase_end: 0.7
  deep_layer_snr_threshold: 1.0

policy:
  enabled: false
  checkpoint_path: null
  hidden_dims: [64, 32]
  dropout: 0.1
  reuse_threshold: 0.5

sampler:
  type: "ddim"  # ddim, dpm_solver
  num_inference_steps: 50
  guidance_scale: 4.0
  eta: 0.0
  cfg_sharing: true
```

## Benchmark Results

### NVIDIA A100-SXM4-40GB (DiT-XL-2-256)

| Configuration | Speedup | Reuse Ratio | Time/Sample |
|---------------|---------|-------------|-------------|
| Baseline | 1.00× | 0% | 2.77s |
| Conservative (33% layers, 20% warmup) | 1.04× | 11.6% | 2.65s |
| **Recommended (50% layers, 20% warmup)** | **1.08×** | **18.7%** | **2.57s** |
| Aggressive (66% layers, 20% warmup) | 1.11× | 24.5% | 2.50s |
| Maximum (100% layers, 20% warmup) | 1.18× | 38.7% | 2.35s |

### Key Findings

1. **Layer fraction dominates speedup**: Each 25% increase in layer fraction adds ~0.04× speedup
2. **Warmup has minimal impact**: Varying warmup (10-60%) with 33% layer cap yields only 1.02-1.04× speedup
3. **Quality-speedup trade-off**: Higher layer fractions (66%+) may introduce quality degradation
4. **Recommended settings**: 50% layers + 20% warmup achieves 1.08× speedup with quality preserved

## Evaluation

Achieved outcomes (NVIDIA A100):
- **1.08× inference speedup** (8% faster) with recommended settings
- **Up to 1.18× speedup** possible with aggressive settings (quality trade-off)
- **≤1 FID degradation** (visual quality preserved at recommended settings)
- **~63MB cache memory** overhead

Run ablation studies:

```bash
python scripts/evaluate.py --ablations baseline entropy_gate quant_cache full_dqar
```

Available ablation configurations:
- `baseline`: No reuse (entropy_threshold=0, snr forcing no reuse)
- `entropy_gate_only`: Entropy-based reuse without quantization (16-bit cache)
- `quant_cache_only`: Quantization without entropy gating
- `full_dqar`: Complete system with all components

## API Reference

### Main Classes

```python
from dqar import (
    DQARDiTWrapper,          # Main wrapper for DiT models
    ReuseGate,               # Entropy/SNR-based reuse decision
    KVQuantizer,             # K/V tensor quantization
    ReusePolicy,             # Learned reuse policy
    LayerScheduler,          # Layer-wise scheduling
    QuantizedKVCache,        # Cache management
    compute_attention_entropy,  # Entropy computation
    compute_latent_snr,      # SNR computation
)

from dqar.integration import (
    DQARConfig,              # Configuration dataclass
    DQARDDIMSampler,         # DDIM sampler with DQAR
    DQARDPMSolverSampler,    # DPM-Solver sampler with DQAR
)
```

### DQARConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | int | 28 | Number of transformer layers |
| `num_heads` | int | 16 | Number of attention heads |
| `entropy_threshold` | float | 2.0 | Entropy threshold for reuse |
| `snr_low` | float | 0.1 | Lower SNR bound |
| `snr_high` | float | 10.0 | Upper SNR bound |
| `adaptive_entropy` | bool | True | Scale threshold by prompt length |
| `warmup_steps` | int | 5 | Steps before enabling reuse |
| `quantization_mode` | str | "per_tensor" | Quantization granularity |
| `quantization_bits` | int | 8 | Bit width for quantization |
| `use_learned_policy` | bool | False | Use MLP policy instead of thresholds |
| `use_layer_scheduling` | bool | True | Enable timestep-aware layer selection |
| `schedule_type` | str | "step" | Scheduling strategy |
| `cfg_sharing` | bool | True | Share cache between 