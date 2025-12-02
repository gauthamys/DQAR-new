"""
Attention Reuse Gate Module

Implements the decision logic for whether to reuse cached attention
based on entropy and SNR thresholds. The gate ensures attention is
only reused when the quality impact is minimal.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .entropy import compute_attention_entropy
from .snr import compute_latent_snr


@dataclass
class ReuseDecision:
    """Container for reuse gate decision and metadata."""
    should_reuse: bool
    entropy: float
    snr: float
    confidence: float = 1.0
    reason: str = ""


@dataclass
class GateConfig:
    """Configuration for the reuse gate."""
    # Entropy threshold - reuse if entropy < threshold
    entropy_threshold: float = 2.0

    # SNR range - reuse if SNR is within [snr_low, snr_high]
    snr_low: float = 0.1
    snr_high: float = 10.0

    # Adaptive threshold based on prompt length
    adaptive_entropy: bool = True
    base_prompt_length: int = 77  # CLIP default

    # Layer-wise thresholds (optional)
    layer_entropy_scales: Optional[Dict[int, float]] = None

    # Minimum timesteps before allowing reuse
    warmup_steps: int = 2

    # Confidence threshold for policy-based decisions
    policy_confidence_threshold: float = 0.5


class ReuseGate(nn.Module):
    """
    Gate module that decides whether to reuse cached attention.

    Attention is reused only if:
    1. Entropy H_t < τ(p) where τ adapts to prompt length
    2. SNR_t ∈ [a, b] indicating stable signal regime
    3. Warmup period has passed

    Can operate in two modes:
    - Threshold-based: Uses fixed entropy/SNR thresholds
    - Policy-based: Uses learned MLP policy for decisions
    """

    def __init__(self, config: Optional[GateConfig] = None):
        """
        Args:
            config: Gate configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or GateConfig()

        # Track statistics
        self.register_buffer("num_reuse_decisions", torch.tensor(0))
        self.register_buffer("num_recompute_decisions", torch.tensor(0))

        # History for analysis
        self.decision_history: list[ReuseDecision] = []

    def compute_adaptive_threshold(
        self,
        base_threshold: float,
        prompt_length: int,
    ) -> float:
        """
        Compute entropy threshold adapted to prompt length.

        Longer prompts tend to have higher entropy, so we scale
        the threshold accordingly.

        Args:
            base_threshold: Base entropy threshold
            prompt_length: Length of the conditioning prompt

        Returns:
            Adapted threshold
        """
        if not self.config.adaptive_entropy:
            return base_threshold

        # Scale factor based on prompt length ratio
        length_ratio = prompt_length / self.config.base_prompt_length
        # Use sqrt scaling to avoid over-adjustment
        scale = length_ratio ** 0.5

        return base_threshold * scale

    def get_layer_threshold(
        self,
        base_threshold: float,
        layer_idx: int,
    ) -> float:
        """
        Get entropy threshold for a specific layer.

        Args:
            base_threshold: Base threshold (possibly already adapted)
            layer_idx: Index of the transformer layer

        Returns:
            Layer-specific threshold
        """
        if self.config.layer_entropy_scales is None:
            return base_threshold

        scale = self.config.layer_entropy_scales.get(layer_idx, 1.0)
        return base_threshold * scale

    def check_entropy_condition(
        self,
        entropy: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """Check if entropy is below threshold."""
        return entropy < threshold

    def check_snr_condition(
        self,
        snr: torch.Tensor,
    ) -> torch.Tensor:
        """Check if SNR is within acceptable range."""
        return (snr >= self.config.snr_low) & (snr <= self.config.snr_high)

    def forward(
        self,
        attention_weights: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None,
        timestep_idx: int = 0,
        layer_idx: int = 0,
        prompt_length: int = 77,
        x_0: Optional[torch.Tensor] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ReuseDecision]:
        """
        Determine whether to reuse cached attention.

        Args:
            attention_weights: Attention map (B, H, T, T) - optional if entropy provided
            entropy: Pre-computed entropy (B,) - optional if attention_weights provided
            x_t: Current noisy latent - optional if snr provided
            snr: Pre-computed SNR (B,) - optional if x_t provided
            timestep_idx: Current timestep index in the diffusion process
            layer_idx: Current transformer layer index
            prompt_length: Length of conditioning prompt for adaptive threshold
            x_0: Predicted clean latent for SNR computation
            alphas_cumprod: Noise schedule for SNR computation

        Returns:
            Tuple of (should_reuse tensor of shape (B,), ReuseDecision metadata)
        """
        # Check warmup
        if timestep_idx < self.config.warmup_steps:
            return self._make_decision(
                should_reuse=False,
                entropy=0.0,
                snr=0.0,
                reason="warmup_period",
            )

        # Compute entropy if not provided
        if entropy is None:
            if attention_weights is None:
                raise ValueError("Either entropy or attention_weights must be provided")
            entropy = compute_attention_entropy(attention_weights)

        # Compute SNR if not provided
        if snr is None:
            if x_t is None:
                # Use a permissive default if no latent info
                snr = torch.tensor([1.0], device=entropy.device)
            elif alphas_cumprod is not None:
                # Compute from schedule
                from .snr import compute_snr_from_schedule
                timestep_tensor = torch.tensor([timestep_idx], device=x_t.device)
                snr = compute_snr_from_schedule(timestep_tensor, alphas_cumprod)
            else:
                snr = compute_latent_snr(x_t, x_0=x_0)

        # Ensure tensors are on same device
        if entropy.device != snr.device:
            snr = snr.to(entropy.device)

        # Compute adaptive threshold
        base_threshold = self.config.entropy_threshold
        adapted_threshold = self.compute_adaptive_threshold(base_threshold, prompt_length)
        layer_threshold = self.get_layer_threshold(adapted_threshold, layer_idx)

        # Check conditions
        entropy_ok = self.check_entropy_condition(entropy, layer_threshold)
        snr_ok = self.check_snr_condition(snr)

        # Combined decision
        should_reuse = entropy_ok & snr_ok

        # Handle batch dimension
        if should_reuse.dim() == 0:
            should_reuse = should_reuse.unsqueeze(0)

        # Create decision metadata
        mean_entropy = entropy.mean().item() if entropy.numel() > 0 else 0.0
        mean_snr = snr.mean().item() if snr.numel() > 0 else 0.0
        reuse_ratio = should_reuse.float().mean().item()

        reason = []
        if not entropy_ok.all():
            reason.append(f"entropy>{layer_threshold:.2f}")
        if not snr_ok.all():
            reason.append(f"snr_out_of_range")

        decision = ReuseDecision(
            should_reuse=should_reuse.all().item(),
            entropy=mean_entropy,
            snr=mean_snr,
            confidence=reuse_ratio,
            reason=", ".join(reason) if reason else "reuse_approved",
        )

        # Update statistics
        if decision.should_reuse:
            self.num_reuse_decisions += 1
        else:
            self.num_recompute_decisions += 1

        self.decision_history.append(decision)

        return should_reuse, decision

    def _make_decision(
        self,
        should_reuse: bool,
        entropy: float,
        snr: float,
        reason: str,
    ) -> Tuple[torch.Tensor, ReuseDecision]:
        """Helper to create a decision result."""
        decision = ReuseDecision(
            should_reuse=should_reuse,
            entropy=entropy,
            snr=snr,
            confidence=1.0 if should_reuse else 0.0,
            reason=reason,
        )
        self.decision_history.append(decision)

        if should_reuse:
            self.num_reuse_decisions += 1
        else:
            self.num_recompute_decisions += 1

        return torch.tensor([should_reuse]), decision

    def get_reuse_ratio(self) -> float:
        """Get the ratio of reuse decisions to total decisions."""
        total = self.num_reuse_decisions + self.num_recompute_decisions
        if total == 0:
            return 0.0
        return (self.num_reuse_decisions / total).item()

    def reset_statistics(self):
        """Reset decision counters and history."""
        self.num_reuse_decisions.zero_()
        self.num_recompute_decisions.zero_()
        self.decision_history.clear()

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary statistics of decisions made."""
        if not self.decision_history:
            return {"total_decisions": 0}

        entropies = [d.entropy for d in self.decision_history]
        snrs = [d.snr for d in self.decision_history]
        reuse_count = sum(1 for d in self.decision_history if d.should_reuse)

        return {
            "total_decisions": len(self.decision_history),
            "reuse_count": reuse_count,
            "recompute_count": len(self.decision_history) - reuse_count,
            "reuse_ratio": reuse_count / len(self.decision_history),
            "mean_entropy": sum(entropies) / len(entropies),
            "mean_snr": sum(snrs) / len(snrs),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
        }
