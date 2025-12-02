"""DQAR Manager for coordinating attention reuse across layers."""

from typing import Optional, Dict, Any, Tuple
import torch

from ..cache.kv_cache import QuantizedKVCache
from ..policy.layer_scheduler import LayerScheduler


class DQARManager:
    """
    Centralized manager for DQAR attention reuse.

    Coordinates between:
    - Layer scheduler (which layers can reuse at each timestep)
    - KV cache (stores quantized K/V tensors)
    - Per-step state (current timestep, SNR)

    This manager is shared across all DQARAttentionProcessor instances
    to enable coordinated reuse decisions.
    """

    def __init__(
        self,
        num_layers: int,
        kv_cache: QuantizedKVCache,
        layer_scheduler: LayerScheduler,
        enabled: bool = True,
    ):
        """
        Initialize DQAR manager.

        Args:
            num_layers: Number of transformer layers
            kv_cache: Quantized KV cache instance
            layer_scheduler: Layer scheduler for timestep-aware reuse
            enabled: Whether DQAR is enabled (for easy toggling)
        """
        self.num_layers = num_layers
        self.kv_cache = kv_cache
        self.layer_scheduler = layer_scheduler
        self.enabled = enabled

        # Current step state
        self.current_timestep_idx: int = 0
        self.current_snr: Optional[float] = None
        self.total_timesteps: int = 50

        # Statistics tracking
        self.stats: Dict[str, int] = {
            "reused": 0,
            "computed": 0,
            "total_calls": 0,
        }

        # Per-layer statistics
        self.layer_stats: Dict[int, Dict[str, int]] = {
            i: {"reused": 0, "computed": 0} for i in range(num_layers)
        }

    def set_timestep(self, timestep_idx: int, snr: Optional[float] = None) -> None:
        """
        Update current timestep state.

        Called at the beginning of each denoising step.

        Args:
            timestep_idx: Current timestep index (0 = first step)
            snr: Current signal-to-noise ratio (optional)
        """
        self.current_timestep_idx = timestep_idx
        self.current_snr = snr

    def set_total_timesteps(self, total: int) -> None:
        """Set total number of timesteps for normalization."""
        self.total_timesteps = total

        # Rebuild layer scheduler schedule with correct timesteps
        if self.layer_scheduler is not None:
            self.layer_scheduler.config.num_timesteps = total
            self.layer_scheduler._build_schedule()

    def should_reuse(self, layer_idx: int) -> bool:
        """
        Determine if a layer should reuse cached K/V.

        Decision based on:
        1. DQAR enabled
        2. Layer scheduler allows reuse at current timestep
        3. Cache has valid K/V for this layer

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            True if should reuse cached K/V, False otherwise
        """
        if not self.enabled:
            return False

        # Check layer scheduler (if enabled)
        if self.layer_scheduler is not None:
            if not self.layer_scheduler.can_reuse(
                self.current_timestep_idx,
                layer_idx,
                snr=self.current_snr,
            ):
                return False

        # Check cache availability
        if not self.kv_cache.has_cache(layer_idx):
            return False

        return True

    def get_cached_kv(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Retrieve cached K/V tensors for a layer.

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            Tuple of (key, value, attention_output) or None if not cached
        """
        return self.kv_cache.retrieve(layer_idx)

    def get_cached_attention_output(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Retrieve cached attention output for direct reuse.

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            Cached attention output tensor or None if not cached
        """
        return self.kv_cache.retrieve_attention_output(layer_idx)

    def cache_attention_output(self, layer_idx: int, output: torch.Tensor) -> None:
        """
        Cache attention output for future reuse.

        Uses existing kv_cache infrastructure with the attention_output field.

        Args:
            layer_idx: Index of the transformer layer
            output: Attention output tensor to cache
        """
        # Store with placeholder K/V (required by store()) and real attention output
        self.kv_cache.store(
            layer_idx=layer_idx,
            key=output,  # Placeholder - not used for output caching
            value=output,  # Placeholder - not used for output caching
            attention_output=output,  # The actual cached value
            timestep=self.current_timestep_idx,
        )

    def cache_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_output: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store K/V tensors in cache.

        Args:
            layer_idx: Index of the transformer layer
            key: Key tensor to cache
            value: Value tensor to cache
            attention_output: Optional attention output to cache
        """
        self.kv_cache.store(
            layer_idx=layer_idx,
            key=key,
            value=value,
            attention_output=attention_output,
            timestep=self.current_timestep_idx,
        )

    def record_reuse(self, layer_idx: int) -> None:
        """Record that a layer reused cached K/V."""
        self.stats["reused"] += 1
        self.stats["total_calls"] += 1
        self.layer_stats[layer_idx]["reused"] += 1

    def record_compute(self, layer_idx: int) -> None:
        """Record that a layer computed new K/V."""
        self.stats["computed"] += 1
        self.stats["total_calls"] += 1
        self.layer_stats[layer_idx]["computed"] += 1

    def get_reuse_ratio(self) -> float:
        """Get overall reuse ratio."""
        total = self.stats["total_calls"]
        if total == 0:
            return 0.0
        return self.stats["reused"] / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total = self.stats["total_calls"]
        return {
            "total_calls": total,
            "reused": self.stats["reused"],
            "computed": self.stats["computed"],
            "reuse_ratio": self.get_reuse_ratio(),
            "layer_stats": dict(self.layer_stats),
        }

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = {"reused": 0, "computed": 0, "total_calls": 0}
        self.layer_stats = {
            i: {"reused": 0, "computed": 0} for i in range(self.num_layers)
        }

    def reset_for_new_sample(self) -> None:
        """Reset state for a new sample generation."""
        self.current_timestep_idx = 0
        self.current_snr = None
        self.kv_cache.clear()

    def __repr__(self) -> str:
        return (
            f"DQARManager(num_layers={self.num_layers}, "
            f"enabled={self.enabled}, "
            f"reuse_ratio={self.get_reuse_ratio():.2%})"
        )
