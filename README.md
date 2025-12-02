# DQAR: Dynamic and Quantization-Aware Attention Reuse for Diffusion Transformers

A unified framework combining entropy- and SNR-based attention reuse with low-bit quantized KV caching for efficient Diffusion Transformer (DiT) inference.

## Overview

DQAR reduces the computational cost of DiT inference by:

1. **Entropy & SNR-Based Reuse Gate**: Dynamically decides when to reuse cached attention based on attention entropy and signal-to-noise ratio
2. **Quantized KV Caching**: Stores Key/Value tensors in 8-bit integer form to reduce VRAM usage
3. **Dynamic Reuse Policy**: Lightweight MLP (<0.5M params) that predicts optimal reuse decisions
4. **Layer Scheduling**: Timestep-aware layer selection for attention reuse

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
from dqar import DQARDiTWrapper, DQARConfig
from dqar.integration import DQARDDIMSampler, SamplerConfig
from diffusers import DiTPipeline

# Load DiT model
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.to("cuda")

# Configure DQAR
dqar_config = DQARConfig(
    entropy_threshold=2.0,
    snr_low=0.1,
    snr_high=10.0,
    quantization_mode="per_tensor",
)

# Wrap model with DQAR
wrapped_model = DQARDiTWrapper(pipe.transformer, dqar_config)

# Create sampler
sampler_config = SamplerConfig(
    num_inference_steps=50,
    guidance_scale=4.0,
)
sampler = DQARDDIMSampler(wrapped_model, pipe.scheduler, sampler_config)

# Generate images
latents = sampler.sample(
    batch_size=4,
    class_labels=torch.tensor([207, 360, 387, 974], device="cuda"),
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

# Evaluate performance
python scripts/evaluate.py \
    --model facebook/DiT-XL-2-256 \
    --num-samples 256 \
    --ablations baseline entropy_gate quant_cache full_dqar
```

## Architecture

```
dqar/
├── core/
│   ├── entropy.py        # Attention entropy computation
│   ├── snr.py            # Signal-to-noise ratio computation
│   ├── reuse_gate.py     # Reuse decision logic
│   └── quantization.py   # KV tensor quantization
├── cache/
│   └── kv_cache.py       # Quantized KV cache management
├── policy/
│   ├── mlp_policy.py     # Learned reuse policy
│   └── layer_scheduler.py # Layer-wise scheduling
├── integration/
│   ├── dit_wrapper.py    # DiT model wrapper
│   └── samplers.py       # DDIM/DPM-Solver samplers
├── evaluation/
│   ├── metrics.py        # FID, CLIP score
│   └── profiling.py      # Runtime/memory profiling
└── training/
    └── policy_trainer.py # Policy training pipeline
```

## Key Components

### 1. Entropy & SNR-Based Reuse Gate

At each timestep, DQAR computes:

- **Attention Entropy**: `H_t = -1/(HT) Σ_h Σ_{i,j} A_t^(h)(i,j) log(A_t^(h)(i,j) + ε)`
- **Latent SNR**: `SNR_t = ||x_0||² / (||x_t - x_0||² + ε)`

Attention is reused only if `H_t < τ(p)` and `SNR_t ∈ [a, b]`, where `τ(p)` adapts to prompt length.

### 2. Quantized KV Caching

Cached K/V tensors are stored in 8-bit integer form:

```
K_q = clip(round(K / s_K), -127, 127)
s_K = max|K| / 127
```

Supports per-tensor, per-channel, and per-head scaling modes.

### 3. Dynamic Reuse Policy

A lightweight MLP predicts reuse probability:

```
p_reuse = P_θ(concat(H_t, SNR_t, ||x_t||₂, t))
```

The policy is trained offline on inference traces labeled by quality impact.

### 4. Layer Scheduling

Early timesteps reuse only shallow blocks (high entropy, weak signal), while later timesteps can reuse deeper blocks (stabilized attention).

## Configuration

See `configs/default.yaml` for full configuration options:

```yaml
reuse_gate:
  entropy_threshold: 2.0
  snr_low: 0.1
  snr_high: 10.0
  adaptive_entropy: true

quantization:
  mode: "per_tensor"
  bits: 8

layer_scheduling:
  enabled: true
  schedule_type: "step"
```

## Evaluation

Expected outcomes:
- **≥25% inference speedup** or **≥20% VRAM reduction**
- **≤1 FID degradation**

Run ablation studies:

```bash
python scripts/evaluate.py --ablations baseline entropy_gate quant_cache full_dqar
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# With coverage
pytest tests/ --cov=dqar --cov-report=html
```

## Citation

If you use DQAR in your research, please cite:

```bibtex
@article{dqar2025,
  title={Dynamic and Quantization-Aware Attention Reuse for Diffusion Transformers},
  author={Satyanarayana, Gautham},
  year={2025}
}
```

## Related Work

- [Attention Compression for Diffusion Transformer Models](https://arxiv.org/abs/...) (NeurIPS 2024)
- [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005) (NeurIPS 2024)
- [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](https://arxiv.org/abs/2406.17343)

## License

MIT License
