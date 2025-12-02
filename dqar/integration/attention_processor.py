"""
DQAR Attention Processor

Custom attention processor that intercepts K/V computation to implement
attention reuse based on SNR + layer scheduling.

Compatible with diffusers' `model.set_attn_processor()` API.
"""

import torch
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import DQARManager


class DQARAttentionProcessor:
    """
    Custom attention processor for DQAR K/V caching and reuse.

    This processor intercepts attention computation and:
    1. Checks if K/V can be reused based on SNR + layer scheduling
    2. Retrieves cached K/V if reusable
    3. Computes and caches new K/V otherwise
    4. Performs attention with cached or fresh K/V

    Usage:
        processor = DQARAttentionProcessor(layer_idx=0, manager=manager)
        model.set_attn_processor({name: processor})
    """

    def __init__(
        self,
        layer_idx: int,
        manager: "DQARManager",
        is_cross_attention: bool = False,
    ):
        """
        Initialize the attention processor.

        Args:
            layer_idx: Index of the transformer layer this processor belongs to
            manager: Shared DQARManager instance for cache coordination
            is_cross_attention: Whether this is cross-attention (text conditioning)
        """
        self.layer_idx = layer_idx
        self.manager = manager
        self.is_cross_attention = is_cross_attention

    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention with DQAR caching/reuse logic.

        Args:
            attn: The attention module (provides to_q, to_k, to_v, to_out)
            hidden_states: Input tensor (B, seq_len, dim)
            encoder_hidden_states: Cross-attention input (for text conditioning)
            attention_mask: Optional attention mask
            temb: Optional timestep embedding

        Returns:
            Attention output tensor
        """
        residual = hidden_states

        # Handle input layer norm if present
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        # Handle group norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Always compute Query (needed regardless of reuse)
        query = attn.to_q(hidden_states)

        # Determine K/V input source
        is_cross = encoder_hidden_states is not None
        if is_cross:
            kv_input = encoder_hidden_states
        else:
            kv_input = hidden_states

        # Check reuse decision (SNR + layer scheduling)
        should_reuse = (
            not is_cross  # Only reuse self-attention, not cross-attention
            and self.manager.should_reuse(self.layer_idx)
        )

        if should_reuse:
            # Retrieve cached K/V
            cached = self.manager.get_cached_kv(self.layer_idx)
            if cached is not None:
                key, value, _ = cached
                self.manager.record_reuse(self.layer_idx)
            else:
                # Cache miss - compute new K/V
                should_reuse = False

        if not should_reuse:
            # Compute K/V projections
            key = attn.to_k(kv_input)
            value = attn.to_v(kv_input)

            # Cache K/V for future reuse (only self-attention)
            if not is_cross:
                self.manager.cache_kv(self.layer_idx, key, value)
                self.manager.record_compute(self.layer_idx)

        # Get attention dimensions
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape Q, K, V for multi-head attention
        # From (B, seq, inner_dim) to (B, heads, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Handle attention norm if present
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Compute attention using scaled dot-product attention (PyTorch 2.0+)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back: (B, heads, seq, head_dim) -> (B, seq, inner_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)  # Linear
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        # Handle 4D input case
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # Handle residual connection if configured
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # Rescale output if configured
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DQARJointAttentionProcessor:
    """
    DQAR processor for joint attention (used in some DiT variants).

    Similar to DQARAttentionProcessor but handles joint attention patterns
    where text and image tokens are concatenated.
    """

    def __init__(
        self,
        layer_idx: int,
        manager: "DQARManager",
    ):
        self.layer_idx = layer_idx
        self.manager = manager

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process joint attention with DQAR caching.

        For joint attention, we only cache the image portion of K/V,
        as text tokens are typically fixed.
        """
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # Query projection
        query = attn.to_q(hidden_states)

        # For joint attention, concatenate image and text inputs
        if encoder_hidden_states is not None:
            # Text tokens for cross-attention
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            kv_input = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        else:
            kv_input = hidden_states

        # Check reuse for image self-attention
        should_reuse = self.manager.should_reuse(self.layer_idx)

        if should_reuse:
            cached = self.manager.get_cached_kv(self.layer_idx)
            if cached is not None:
                key, value, _ = cached
                self.manager.record_reuse(self.layer_idx)
            else:
                should_reuse = False

        if not should_reuse:
            key = attn.to_k(kv_input)
            value = attn.to_v(kv_input)

            # Cache K/V
            self.manager.cache_kv(self.layer_idx, key, value)
            self.manager.record_compute(self.layer_idx)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def get_processor_class_for_model(model_type: str) -> type:
    """
    Get the appropriate processor class for a model type.

    Args:
        model_type: Type of model (e.g., "dit", "sd3", "flux")

    Returns:
        Processor class to use
    """
    # Most DiT models use standard attention
    if model_type in ["dit", "dit-xl", "pixart"]:
        return DQARAttentionProcessor
    # SD3 and Flux use joint attention
    elif model_type in ["sd3", "flux"]:
        return DQARJointAttentionProcessor
    else:
        return DQARAttentionProcessor
