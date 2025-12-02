"""
Layer Scheduling Module

Implements timestep-dependent layer scheduling for attention reuse.
Early timesteps reuse only shallow blocks (high entropy, weak signal),
while later timesteps can reuse deeper blocks (stabilized attention).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class ScheduleType(Enum):
    """Types of layer scheduling strategies."""
    LINEAR = "linear"  # Linear progression from shallow to deep
    LINEAR_REVERSE = "linear_reverse"  # Linear progression reusing deep layers first
    EXPONENTIAL = "exponential"  # Exponential ramp-up
    STEP = "step"  # Discrete steps
    CUSTOM = "custom"  # User-defined schedule


@dataclass
class SchedulerConfig:
    """Configuration for layer scheduler."""
    num_layers: int = 28  # DiT-XL default
    num_timesteps: int = 50  # DDIM default steps
    schedule_type: ScheduleType = ScheduleType.LINEAR

    # Phase boundaries (as fraction of total timesteps)
    early_phase_end: float = 0.3  # First 30% of timesteps
    mid_phase_end: float = 0.7  # 30-70% of timesteps

    # Layers to reuse in each phase
    early_phase_layers: Optional[List[int]] = None  # Shallow layers only
    mid_phase_layers: Optional[List[int]] = None  # Shallow + some mid
    late_phase_layers: Optional[List[int]] = None  # All layers

    # Minimum SNR for deep layer reuse
    deep_layer_snr_threshold: float = 1.0

    def __post_init__(self):
        # Default layer assignments based on thirds
        third = self.num_layers // 3
        if self.early_phase_layers is None:
            self.early_phase_layers = list(range(third))
        if self.mid_phase_layers is None:
            self.mid_phase_layers = list(range(2 * third))
        if self.late_phase_layers is None:
            self.late_phase_layers = list(range(self.num_layers))


class LayerScheduler(nn.Module):
    """
    Scheduler that determines which layers can reuse attention at each timestep.

    Early timesteps (high noise, unstable attention):
        - Only shallow blocks can reuse
        - Deep layers must recompute

    Later timesteps (low noise, stable attention):
        - All layers can potentially reuse
        - Convergence has stabilized attention patterns
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Args:
            config: Scheduler configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or SchedulerConfig()

        # Precompute schedule
        self._build_schedule()

    def _build_schedule(self):
        """Build the reuse schedule based on configuration."""
        self.schedule: Dict[int, Set[int]] = {}

        for t in range(self.config.num_timesteps):
            progress = t / max(1, self.config.num_timesteps - 1)

            if self.config.schedule_type == ScheduleType.LINEAR:
                layers = self._linear_schedule(progress)
            elif self.config.schedule_type == ScheduleType.LINEAR_REVERSE:
                layers = self._linear_reverse_schedule(progress)
            elif self.config.schedule_type == ScheduleType.EXPONENTIAL:
                layers = self._exponential_schedule(progress)
            elif self.config.schedule_type == ScheduleType.STEP:
                layers = self._step_schedule(progress)
            else:
                layers = self._step_schedule(progress)  # Default

            self.schedule[t] = set(layers)

    def _linear_schedule(self, progress: float) -> List[int]:
        """Linear progression of reusable layers (quality-focused).

        Changes for better quality:
        1. Increased warmup from 20% to 40% of timesteps
        2. Limited to shallow layers only (first 1/3) for quality preservation
        """
        # Delay reuse until 40% of timesteps complete to protect early structure
        warmup_fraction = 0.4
        if progress < warmup_fraction:
            return []  # No reuse during warmup

        # Number of layers that can be reused increases linearly after warmup
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)

        # Limit to shallow layers only (first third) for quality preservation
        max_reusable_layers = self.config.num_layers // 3
        num_reusable = int(adjusted_progress * max_reusable_layers)

        return list(range(num_reusable))

    def _linear_reverse_schedule(self, progress: float) -> List[int]:
        """Linear progression of reusable layers, starting from deep layers.

        Reuses deep layers (closer to output) instead of shallow layers.
        Hypothesis: Deep layers process abstract representations that may be
        more stable across timesteps than shallow layers which directly see
        the changing input noise level.
        """
        # Delay reuse until 40% of timesteps complete to protect early structure
        warmup_fraction = 0.4
        if progress < warmup_fraction:
            return []  # No reuse during warmup

        # Number of layers that can be reused increases linearly after warmup
        adjusted_progress = (progress - warmup_fraction) / (1 - warmup_fraction)

        # Limit to fraction of layers for quality preservation
        max_reusable_layers = self.config.num_layers // 3
        num_reusable = int(adjusted_progress * max_reusable_layers)

        # Start from deep layers (high indices) instead of shallow (low indices)
        if num_reusable == 0:
            return []
        start_idx = self.config.num_layers - num_reusable
        return list(range(start_idx, self.config.num_layers))

    def _exponential_schedule(self, progress: float) -> List[int]:
        """Exponential ramp-up of reusable layers."""
        # Slow start, rapid increase toward end
        exp_progress = progress ** 2
        num_reusable = int(exp_progress * self.config.num_layers)
        return list(range(num_reusable))

    def _step_schedule(self, progress: float) -> List[int]:
        """Discrete phase-based scheduling."""
        if progress < self.config.early_phase_end:
            return self.config.early_phase_layers
        elif progress < self.config.mid_phase_end:
            return self.config.mid_phase_layers
        else:
            return self.config.late_phase_layers

    def can_reuse(
        self,
        timestep_idx: int,
        layer_idx: int,
        snr: Optional[float] = None,
    ) -> bool:
        """
        Check if a specific layer can reuse attention at given timestep.

        Args:
            timestep_idx: Current timestep index
            layer_idx: Index of the transformer layer
            snr: Optional SNR value for additional gating

        Returns:
            Whether the layer can reuse cached attention
        """
        if timestep_idx not in self.schedule:
            return False

        # Check basic schedule
        if layer_idx not in self.schedule[timestep_idx]:
            return False

        # Additional SNR check for deep layers
        if snr is not None:
            is_deep = layer_idx >= 2 * self.config.num_layers // 3
            if is_deep and snr < self.config.deep_layer_snr_threshold:
                return False

        return True

    def get_reusable_layers(
        self,
        timestep_idx: int,
        snr: Optional[float] = None,
    ) -> List[int]:
        """
        Get list of layers that can reuse attention at given timestep.

        Args:
            timestep_idx: Current timestep index
            snr: Optional SNR value for filtering

        Returns:
            List of layer indices that can reuse
        """
        if timestep_idx not in self.schedule:
            return []

        layers = list(self.schedule[timestep_idx])

        # Filter by SNR if provided
        if snr is not None:
            deep_threshold = 2 * self.config.num_layers // 3
            layers = [
                l for l in layers
                if l < deep_threshold or snr >= self.config.deep_layer_snr_threshold
            ]

        return sorted(layers)

    def get_reuse_mask(
        self,
        timestep_idx: int,
        snr: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Get boolean mask indicating which layers can reuse.

        Args:
            timestep_idx: Current timestep index
            snr: Optional SNR value

        Returns:
            Boolean tensor of shape (num_layers,)
        """
        mask = torch.zeros(self.config.num_layers, dtype=torch.bool)
        reusable = self.get_reusable_layers(timestep_idx, snr)
        for layer_idx in reusable:
            mask[layer_idx] = True
        return mask

    def get_schedule_summary(self) -> Dict[str, any]:
        """Get summary of the scheduling configuration."""
        early_count = len(self.config.early_phase_layers)
        mid_count = len(self.config.mid_phase_layers)
        late_count = len(self.config.late_phase_layers)

        return {
            "schedule_type": self.config.schedule_type.value,
            "num_layers": self.config.num_layers,
            "num_timesteps": self.config.num_timesteps,
            "early_phase_layers": early_count,
            "mid_phase_layers": mid_count,
            "late_phase_layers": late_count,
            "phase_boundaries": {
                "early_end": self.config.early_phase_end,
                "mid_end": self.config.mid_phase_end,
            },
        }


class AdaptiveLayerScheduler(LayerScheduler):
    """
    Adaptive scheduler that adjusts based on observed statistics.

    Tracks entropy and SNR patterns to dynamically adjust
    which layers can safely reuse attention.
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        adaptation_rate: float = 0.1,
    ):
        """
        Args:
            config: Scheduler configuration
            adaptation_rate: How quickly to adapt based on observations
        """
        super().__init__(config)
        self.adaptation_rate = adaptation_rate

        # Per-layer statistics
        self.register_buffer(
            "layer_entropy_mean",
            torch.zeros(self.config.num_layers)
        )
        self.register_buffer(
            "layer_entropy_var",
            torch.ones(self.config.num_layers)
        )
        self.register_buffer(
            "layer_reuse_success",
            torch.zeros(self.config.num_layers)
        )
        self.register_buffer(
            "layer_reuse_total",
            torch.zeros(self.config.num_layers)
        )

    def update_statistics(
        self,
        layer_idx: int,
        entropy: float,
        reuse_success: bool,
    ):
        """
        Update layer statistics based on observation.

        Args:
            layer_idx: Layer index
            entropy: Observed entropy
            reuse_success: Whether reuse maintained quality
        """
        # Update entropy statistics with EMA
        alpha = self.adaptation_rate
        self.layer_entropy_mean[layer_idx] = (
            (1 - alpha) * self.layer_entropy_mean[layer_idx] +
            alpha * entropy
        )

        # Update success rate
        self.layer_reuse_total[layer_idx] += 1
        if reuse_success:
            self.layer_reuse_success[layer_idx] += 1

    def get_layer_success_rate(self, layer_idx: int) -> float:
        """Get reuse success rate for a layer."""
        total = self.layer_reuse_total[layer_idx].item()
        if total == 0:
            return 0.5  # Default uncertainty
        return (self.layer_reuse_success[layer_idx] / total).item()

    def can_reuse(
        self,
        timestep_idx: int,
        layer_idx: int,
        snr: Optional[float] = None,
        entropy: Optional[float] = None,
    ) -> bool:
        """
        Adaptive reuse decision incorporating learned statistics.

        Args:
            timestep_idx: Current timestep
            layer_idx: Layer index
            snr: Current SNR
            entropy: Current entropy

        Returns:
            Whether to reuse
        """
        # First check base schedule
        if not super().can_reuse(timestep_idx, layer_idx, snr):
            return False

        # Additional check based on learned success rate
        success_rate = self.get_layer_success_rate(layer_idx)
        if success_rate < 0.7:  # Require 70% historical success
            # Be more conservative - require entropy to be well below mean
            if entropy is not None:
                mean_entropy = self.layer_entropy_mean[layer_idx].item()
                if entropy > mean_entropy:
                    return False

        return True

    def reset_statistics(self):
        """Reset adaptation statistics."""
        self.layer_entropy_mean.zero_()
        self.layer_entropy_var.fill_(1.0)
        self.layer_reuse_success.zero_()
        self.layer_reuse_total.zero_()
