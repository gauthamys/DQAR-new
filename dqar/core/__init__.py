"""Core computation modules for DQAR."""

from .entropy import compute_attention_entropy, AttentionEntropyComputer
from .snr import compute_latent_snr, SNRComputer
from .reuse_gate import ReuseGate
from .quantization import KVQuantizer, QuantizationMode

__all__ = [
    "compute_attention_entropy",
    "AttentionEntropyComputer",
    "compute_latent_snr",
    "SNRComputer",
    "ReuseGate",
    "KVQuantizer",
    "QuantizationMode",
]
