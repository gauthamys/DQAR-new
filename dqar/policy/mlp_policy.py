"""
Dynamic Reuse Policy Module

Implements a lightweight MLP that predicts the probability of attention reuse
based on entropy, SNR, latent norm, and timestep information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    """Configuration for the reuse policy network."""
    # Network architecture
    input_dim: int = 4  # [entropy, snr, latent_norm, timestep]
    hidden_dims: list = None  # Default: [64, 32]
    dropout: float = 0.1

    # Decision threshold
    reuse_threshold: float = 0.5

    # Training settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Feature normalization
    normalize_inputs: bool = True
    entropy_scale: float = 1.0
    snr_scale: float = 0.1
    norm_scale: float = 0.01

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class ReusePolicy(nn.Module):
    """
    Lightweight MLP policy for predicting attention reuse probability.

    p_reuse = P_θ(concat(H_t, SNR_t, ||x_t||_2, t))

    The policy is trained offline on cached inference traces labeled
    by performance impact, enabling data-driven decisions.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        """
        Args:
            config: Policy configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or PolicyConfig()

        # Build MLP
        layers = []
        in_dim = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            ])
            in_dim = hidden_dim

        # Output layer (probability of reuse)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Running statistics for input normalization
        self.register_buffer("running_mean", torch.zeros(self.config.input_dim))
        self.register_buffer("running_var", torch.ones(self.config.input_dim))
        self.register_buffer("num_batches", torch.tensor(0))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_inputs(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize input features using running statistics."""
        if not self.config.normalize_inputs:
            return features

        # Manual scaling based on expected ranges
        scales = torch.tensor([
            self.config.entropy_scale,
            self.config.snr_scale,
            self.config.norm_scale,
            1.0 / 1000,  # timestep scale
        ], device=features.device, dtype=features.dtype)

        return features * scales

    def forward(
        self,
        entropy: torch.Tensor,
        snr: torch.Tensor,
        latent_norm: torch.Tensor,
        timestep: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Predict reuse probability.

        Args:
            entropy: Attention entropy (B,) or scalar
            snr: Signal-to-noise ratio (B,) or scalar
            latent_norm: L2 norm of latent ||x_t||_2 (B,) or scalar
            timestep: Current timestep (B,) or scalar
            return_logits: If True, return logits instead of probabilities

        Returns:
            Reuse probability (B,) in [0, 1], or logits if return_logits=True
        """
        # Ensure all inputs are tensors with batch dimension
        def ensure_batch(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor([x], device=self.mlp[0].weight.device)
            if x.dim() == 0:
                x = x.unsqueeze(0)
            return x.float()

        entropy = ensure_batch(entropy)
        snr = ensure_batch(snr)
        latent_norm = ensure_batch(latent_norm)
        timestep = ensure_batch(timestep)

        # Concatenate features
        features = torch.stack([entropy, snr, latent_norm, timestep], dim=-1)

        # Normalize inputs
        features = self._normalize_inputs(features)

        # Forward through MLP
        logits = self.mlp(features).squeeze(-1)

        if return_logits:
            return logits

        # Apply sigmoid for probability
        return torch.sigmoid(logits)

    def predict(
        self,
        entropy: torch.Tensor,
        snr: torch.Tensor,
        latent_norm: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make reuse decision with probability.

        Args:
            entropy: Attention entropy
            snr: Signal-to-noise ratio
            latent_norm: L2 norm of latent
            timestep: Current timestep

        Returns:
            Tuple of (should_reuse bool tensor, probability tensor)
        """
        prob = self.forward(entropy, snr, latent_norm, timestep)
        should_reuse = prob > self.config.reuse_threshold
        return should_reuse, prob

    def compute_loss(
        self,
        entropy: torch.Tensor,
        snr: torch.Tensor,
        latent_norm: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        quality_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            entropy: Batch of entropy values
            snr: Batch of SNR values
            latent_norm: Batch of latent norms
            timestep: Batch of timesteps
            labels: Binary labels (1 = reuse was beneficial, 0 = reuse hurt quality)
            quality_weights: Optional per-sample weights based on quality impact

        Returns:
            Binary cross-entropy loss
        """
        logits = self.forward(entropy, snr, latent_norm, timestep, return_logits=True)

        if quality_weights is not None:
            # Weighted BCE loss
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), weight=quality_weights
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return loss

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save policy weights and config."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ReusePolicy":
        """Load policy from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        policy = cls(config=checkpoint["config"])
        policy.load_state_dict(checkpoint["state_dict"])
        return policy


class LayerWiseReusePolicy(nn.Module):
    """
    Extended policy with per-layer predictions.

    Uses a shared backbone with layer-specific heads for
    more nuanced reuse decisions.
    """

    def __init__(
        self,
        num_layers: int,
        config: Optional[PolicyConfig] = None,
    ):
        """
        Args:
            num_layers: Number of transformer layers
            config: Policy configuration
        """
        super().__init__()
        self.num_layers = num_layers
        self.config = config or PolicyConfig()

        # Shared backbone
        backbone_layers = []
        in_dim = self.config.input_dim + 1  # +1 for layer embedding
        for hidden_dim in self.config.hidden_dims[:-1]:
            backbone_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers) if backbone_layers else nn.Identity()

        # Layer embedding
        self.layer_embedding = nn.Embedding(num_layers, 8)

        # Adjust input dim for backbone output
        backbone_out_dim = self.config.hidden_dims[-2] if len(self.config.hidden_dims) > 1 else self.config.input_dim + 1

        # Per-layer heads
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out_dim + 8, self.config.hidden_dims[-1]),
                nn.GELU(),
                nn.Linear(self.config.hidden_dims[-1], 1),
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        entropy: torch.Tensor,
        snr: torch.Tensor,
        latent_norm: torch.Tensor,
        timestep: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Predict reuse probability for a specific layer.

        Args:
            entropy: Attention entropy
            snr: Signal-to-noise ratio
            latent_norm: L2 norm of latent
            timestep: Current timestep
            layer_idx: Index of the transformer layer

        Returns:
            Reuse probability
        """
        # Build features
        def ensure_batch(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor([x])
            if x.dim() == 0:
                x = x.unsqueeze(0)
            return x.float()

        entropy = ensure_batch(entropy)
        snr = ensure_batch(snr)
        latent_norm = ensure_batch(latent_norm)
        timestep = ensure_batch(timestep)

        B = entropy.shape[0]
        device = self.layer_embedding.weight.device

        # Move tensors to correct device
        entropy = entropy.to(device)
        snr = snr.to(device)
        latent_norm = latent_norm.to(device)
        timestep = timestep.to(device)

        # Concatenate base features
        layer_idx_tensor = torch.full((B,), layer_idx, dtype=torch.long, device=device)
        layer_embed = self.layer_embedding(layer_idx_tensor)

        features = torch.stack([entropy, snr, latent_norm, timestep], dim=-1)

        # Shared backbone
        if not isinstance(self.backbone, nn.Identity):
            # Add placeholder for layer info in backbone input
            features = torch.cat([features, layer_idx_tensor.unsqueeze(-1).float() / self.num_layers], dim=-1)
            features = self.backbone(features)

        # Combine with layer embedding
        features = torch.cat([features, layer_embed], dim=-1)

        # Layer-specific head
        logits = self.layer_heads[layer_idx](features).squeeze(-1)

        return torch.sigmoid(logits)
