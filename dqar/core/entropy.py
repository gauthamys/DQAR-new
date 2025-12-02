"""
Attention Entropy Computation Module

Computes the entropy of attention maps across heads to measure
attention distribution stability. Lower entropy indicates more
focused attention patterns that are good candidates for reuse.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_attention_entropy(
    attention_weights: torch.Tensor,
    eps: float = 1e-8,
    normalize_by_heads: bool = True,
) -> torch.Tensor:
    """
    Compute the entropy of attention weights.

    H_t = -1/(H*T) * sum_h sum_{i,j} A_t^{(h)}(i,j) * log(A_t^{(h)}(i,j) + eps)

    Args:
        attention_weights: Attention weights of shape (batch, heads, seq_len, seq_len)
                          or (batch, heads, seq_len_q, seq_len_k)
        eps: Small constant for numerical stability
        normalize_by_heads: Whether to normalize by number of heads and tokens

    Returns:
        Entropy tensor of shape (batch,) or scalar if single batch
    """
    # Ensure attention weights are valid probabilities
    # attention_weights shape: (B, H, T, T) or (B, H, Tq, Tk)
    B, H, Tq, Tk = attention_weights.shape

    # Compute entropy: -sum(p * log(p))
    log_attn = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_attn, dim=(-2, -1))  # (B, H)

    if normalize_by_heads:
        # Normalize by number of heads and query tokens
        # This gives us the average entropy per head per query position
        entropy = entropy / (H * Tq)
        entropy = entropy.sum(dim=-1)  # Sum across heads -> (B,)
    else:
        entropy = entropy.sum(dim=-1)  # Sum across heads -> (B,)

    return entropy


class AttentionEntropyComputer:
    """
    Stateful class for computing and tracking attention entropy across timesteps.

    Maintains a history of entropy values for analysis and visualization.
    """

    def __init__(
        self,
        num_layers: int,
        eps: float = 1e-8,
        history_size: int = 100,
    ):
        """
        Args:
            num_layers: Number of transformer layers to track
            eps: Numerical stability constant
            history_size: Maximum number of timesteps to store in history
        """
        self.num_layers = num_layers
        self.eps = eps
        self.history_size = history_size

        # History storage: list of (timestep, layer_idx, entropy_value)
        self.entropy_history: list[Tuple[int, int, float]] = []
        self.current_timestep = 0

    def compute(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute entropy for a single layer's attention weights.

        Args:
            attention_weights: Shape (B, H, Tq, Tk)
            layer_idx: Index of the transformer layer

        Returns:
            Entropy value(s) of shape (B,)
        """
        entropy = compute_attention_entropy(
            attention_weights,
            eps=self.eps,
            normalize_by_heads=True,
        )

        # Store in history
        mean_entropy = entropy.mean().item()
        self.entropy_history.append((self.current_timestep, layer_idx, mean_entropy))

        # Trim history if needed
        if len(self.entropy_history) > self.history_size * self.num_layers:
            self.entropy_history = self.entropy_history[-self.history_size * self.num_layers:]

        return entropy

    def step(self):
        """Advance to next timestep."""
        self.current_timestep += 1

    def reset(self):
        """Reset the computer for a new inference run."""
        self.entropy_history.clear()
        self.current_timestep = 0

    def get_layer_entropy_history(self, layer_idx: int) -> list[Tuple[int, float]]:
        """Get entropy history for a specific layer."""
        return [
            (t, e) for t, l, e in self.entropy_history if l == layer_idx
        ]

    def get_timestep_entropies(self, timestep: int) -> list[Tuple[int, float]]:
        """Get all layer entropies for a specific timestep."""
        return [
            (l, e) for t, l, e in self.entropy_history if t == timestep
        ]

    def get_mean_entropy(self, timestep: Optional[int] = None) -> float:
        """Get mean entropy across all layers, optionally for a specific timestep."""
        if timestep is not None:
            entropies = [e for t, l, e in self.entropy_history if t == timestep]
        else:
            entropies = [e for _, _, e in self.entropy_history]

        if not entropies:
            return 0.0
        return sum(entropies) / len(entropies)


def compute_attention_entropy_from_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute attention entropy directly from Q and K without materializing full attention.

    This is more memory-efficient for very long sequences.

    Args:
        query: Query tensor of shape (B, H, Tq, D)
        key: Key tensor of shape (B, H, Tk, D)
        scale: Attention scale factor (default: 1/sqrt(D))
        attention_mask: Optional mask of shape (B, 1, Tq, Tk) or (B, H, Tq, Tk)
        eps: Numerical stability constant

    Returns:
        Entropy of shape (B,)
    """
    B, H, Tq, D = query.shape
    Tk = key.shape[2]

    if scale is None:
        scale = D ** -0.5

    # Compute attention scores
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # (B, H, Tq, Tk)

    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask

    # Softmax to get attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)

    return compute_attention_entropy(attn_weights, eps=eps)
