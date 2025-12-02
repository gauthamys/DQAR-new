"""
Quantized KV Cache Module

Manages the storage and retrieval of quantized Key-Value tensors
across timesteps and layers for efficient attention reuse.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

from ..core.quantization import KVQuantizer, QuantizedTensor, QuantizationMode


@dataclass
class KVCacheEntry:
    """Single cache entry for one layer at one timestep."""
    key: QuantizedTensor  # Quantized key tensor
    value: QuantizedTensor  # Quantized value tensor
    attention_output: Optional[torch.Tensor] = None  # Cached attention output
    entropy: float = 0.0  # Entropy when cached
    snr: float = 0.0  # SNR when cached
    timestep: int = 0  # Timestep when cached


class QuantizedKVCache(nn.Module):
    """
    Cache manager for quantized K/V tensors.

    Supports:
    - Per-layer caching with different quantization settings
    - Timestep-aware cache management
    - Memory usage tracking
    - Automatic cache invalidation
    """

    def __init__(
        self,
        num_layers: int,
        quantization_mode: QuantizationMode = QuantizationMode.PER_TENSOR,
        bits: int = 8,
        cache_attention_output: bool = True,
        max_cached_timesteps: int = 1,
    ):
        """
        Args:
            num_layers: Number of transformer layers to cache
            quantization_mode: Quantization granularity mode
            bits: Number of bits for quantization
            cache_attention_output: Whether to also cache attention outputs
            max_cached_timesteps: Maximum number of timesteps to keep in cache
        """
        super().__init__()
        self.num_layers = num_layers
        self.cache_attention_output = cache_attention_output
        self.max_cached_timesteps = max_cached_timesteps

        # Create quantizers for each layer
        self.quantizers = nn.ModuleList([
            KVQuantizer(mode=quantization_mode, bits=bits)
            for _ in range(num_layers)
        ])

        # Cache storage: layer_idx -> list of KVCacheEntry
        self._cache: Dict[int, List[KVCacheEntry]] = {i: [] for i in range(num_layers)}

        # Statistics tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_memory_bytes = 0

    def store(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_output: Optional[torch.Tensor] = None,
        entropy: float = 0.0,
        snr: float = 0.0,
        timestep: int = 0,
    ):
        """
        Store K/V tensors in quantized form.

        Args:
            layer_idx: Index of the transformer layer
            key: Key tensor (B, H, T, D)
            value: Value tensor (B, H, T, D)
            attention_output: Optional pre-computed attention output
            entropy: Entropy value when caching
            snr: SNR value when caching
            timestep: Current timestep
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range (max {self.num_layers - 1})")

        quantizer = self.quantizers[layer_idx]

        # Quantize K and V
        k_quantized = quantizer.quantize(key)
        v_quantized = quantizer.quantize(value)

        # Create cache entry
        entry = KVCacheEntry(
            key=k_quantized,
            value=v_quantized,
            attention_output=attention_output if self.cache_attention_output else None,
            entropy=entropy,
            snr=snr,
            timestep=timestep,
        )

        # Add to cache
        self._cache[layer_idx].append(entry)

        # Trim old entries if needed
        while len(self._cache[layer_idx]) > self.max_cached_timesteps:
            self._cache[layer_idx].pop(0)

        # Update memory tracking
        self._update_memory_stats()

    def retrieve(
        self,
        layer_idx: int,
        timestep: Optional[int] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Retrieve dequantized K/V tensors from cache.

        Args:
            layer_idx: Index of the transformer layer
            timestep: Specific timestep to retrieve (default: most recent)

        Returns:
            Tuple of (key, value, attention_output) or None if not cached
        """
        if layer_idx >= self.num_layers or not self._cache[layer_idx]:
            self._cache_misses += 1
            return None

        # Get the requested entry
        entries = self._cache[layer_idx]
        if timestep is not None:
            # Find entry for specific timestep
            entry = next((e for e in entries if e.timestep == timestep), None)
        else:
            # Get most recent
            entry = entries[-1]

        if entry is None:
            self._cache_misses += 1
            return None

        self._cache_hits += 1

        # Dequantize
        quantizer = self.quantizers[layer_idx]
        key = quantizer.dequantize(entry.key)
        value = quantizer.dequantize(entry.value)

        return key, value, entry.attention_output

    def retrieve_attention_output(
        self,
        layer_idx: int,
        timestep: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve only the cached attention output.

        Args:
            layer_idx: Index of the transformer layer
            timestep: Specific timestep to retrieve

        Returns:
            Cached attention output or None
        """
        if layer_idx >= self.num_layers or not self._cache[layer_idx]:
            return None

        entries = self._cache[layer_idx]
        if timestep is not None:
            entry = next((e for e in entries if e.timestep == timestep), None)
        else:
            entry = entries[-1]

        return entry.attention_output if entry else None

    def has_cache(self, layer_idx: int) -> bool:
        """Check if cache exists for a layer."""
        return layer_idx < self.num_layers and len(self._cache[layer_idx]) > 0

    def get_cached_entropy(self, layer_idx: int) -> Optional[float]:
        """Get the entropy value from the most recent cache entry."""
        if not self.has_cache(layer_idx):
            return None
        return self._cache[layer_idx][-1].entropy

    def get_cached_snr(self, layer_idx: int) -> Optional[float]:
        """Get the SNR value from the most recent cache entry."""
        if not self.has_cache(layer_idx):
            return None
        return self._cache[layer_idx][-1].snr

    def clear(self, layer_idx: Optional[int] = None):
        """
        Clear cache entries.

        Args:
            layer_idx: Specific layer to clear, or None for all layers
        """
        if layer_idx is not None:
            self._cache[layer_idx] = []
        else:
            for i in range(self.num_layers):
                self._cache[i] = []
        self._update_memory_stats()

    def clear_all(self):
        """Clear all cache entries and reset statistics."""
        self.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _update_memory_stats(self):
        """Update memory usage statistics."""
        total_bytes = 0
        for layer_entries in self._cache.values():
            for entry in layer_entries:
                # K and V quantized data
                total_bytes += entry.key.data.numel() * entry.key.data.element_size()
                total_bytes += entry.value.data.numel() * entry.value.data.element_size()
                # Scale factors
                total_bytes += entry.key.scale.numel() * entry.key.scale.element_size()
                total_bytes += entry.value.scale.numel() * entry.value.scale.element_size()
                # Attention output (if cached)
                if entry.attention_output is not None:
                    total_bytes += entry.attention_output.numel() * entry.attention_output.element_size()
        self._total_memory_bytes = total_bytes

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        self._update_memory_stats()
        return {
            "total_bytes": self._total_memory_bytes,
            "total_mb": self._total_memory_bytes / (1024 * 1024),
            "entries_per_layer": {i: len(self._cache[i]) for i in range(self.num_layers)},
        }

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self.get_hit_rate(),
            "memory_usage": self.get_memory_usage(),
            "num_layers": self.num_layers,
            "max_cached_timesteps": self.max_cached_timesteps,
        }


class LayerWiseKVCache(QuantizedKVCache):
    """
    Extended KV cache with per-layer quantization configuration.

    Allows different layers to use different quantization modes
    based on their sensitivity.
    """

    def __init__(
        self,
        num_layers: int,
        layer_configs: Optional[Dict[int, Dict[str, Any]]] = None,
        default_mode: QuantizationMode = QuantizationMode.PER_TENSOR,
        default_bits: int = 8,
        cache_attention_output: bool = True,
    ):
        """
        Args:
            num_layers: Number of transformer layers
            layer_configs: Per-layer configuration overrides
                          e.g., {0: {"mode": "per_channel", "bits": 8}}
            default_mode: Default quantization mode
            default_bits: Default bit width
            cache_attention_output: Whether to cache attention outputs
        """
        # Initialize parent without quantizers
        nn.Module.__init__(self)

        self.num_layers = num_layers
        self.cache_attention_output = cache_attention_output
        self.max_cached_timesteps = 1

        # Create per-layer quantizers with custom configs
        self.quantizers = nn.ModuleList()
        for i in range(num_layers):
            if layer_configs and i in layer_configs:
                cfg = layer_configs[i]
                mode = QuantizationMode(cfg.get("mode", default_mode))
                bits = cfg.get("bits", default_bits)
            else:
                mode = default_mode
                bits = default_bits
            self.quantizers.append(KVQuantizer(mode=mode, bits=bits))

        self._cache = {i: [] for i in range(num_layers)}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_memory_bytes = 0
