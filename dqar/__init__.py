"""
DQAR: Dynamic and Quantization-Aware Attention Reuse for Diffusion Transformers

A unified framework combining entropy- and SNR-based attention reuse
with low-bit quantized KV caching for efficient DiT inference.
"""

__version__ = "0.1.0"

from .core.entropy import compute_attention_entropy
from .core.snr import compute_latent_snr
from .core.reuse_gate import ReuseGate
from .core.quantization import KVQuantizer
from .policy.mlp_policy import ReusePolicy
from .policy.layer_scheduler import LayerScheduler
from .cache.kv_cache import QuantizedKVCache
from .integration.dit_wrapper import DQARDiTWrapper

__all__ = [
    "compute_attention_entropy",
    "compute_latent_snr",
    "ReuseGate",
    "KVQuantizer",
    "ReusePolicy",
    "LayerScheduler",
    "QuantizedKVCache",
    "DQARDiTWrapper",
]
