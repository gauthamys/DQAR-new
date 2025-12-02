"""
DiT Wrapper Module

Wraps a Diffusion Transformer with DQAR components for
efficient inference with attention reuse and quantized caching.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import time

from ..core.entropy import AttentionEntropyComputer, compute_attention_entropy
from ..core.snr import SNRComputer
from ..core.reuse_gate import ReuseGate, GateConfig
from ..core.quantization import QuantizationMode
from ..cache.kv_cache import QuantizedKVCache
from ..policy.mlp_policy import ReusePolicy, PolicyConfig
from ..policy.layer_scheduler import LayerScheduler, SchedulerConfig
from .manager import DQARManager
from .attention_processor import DQARAttentionProcessor


@dataclass
class DQARConfig:
    """Configuration for DQAR wrapper."""
    # Model configuration
    num_layers: int = 28  # DiT-XL default
    num_heads: int = 16

    # Reuse gate configuration
    entropy_threshold: float = 2.0
    snr_low: float = 0.1
    snr_high: float = 10.0
    adaptive_entropy: bool = True
    warmup_steps: int = 2

    # Quantization configuration
    quantization_mode: str = "per_tensor"
    quantization_bits: int = 8
    cache_attention_output: bool = True

    # Policy configuration
    use_learned_policy: bool = False
    policy_threshold: float = 0.5

    # Layer scheduling
    use_layer_scheduling: bool = True
    schedule_type: str = "step"

    # CFG sharing (classifier-free guidance)
    cfg_sharing: bool = True

    # Profiling
    enable_profiling: bool = False


class DQARDiTWrapper(nn.Module):
    """
    Wrapper that adds DQAR capabilities to a DiT model.

    Intercepts attention computation to:
    1. Compute entropy and SNR
    2. Decide whether to reuse cached attention
    3. Store/retrieve quantized K/V tensors
    4. Apply layer-wise scheduling
    """

    def __init__(
        self,
        dit_model: nn.Module,
        config: Optional[DQARConfig] = None,
    ):
        """
        Args:
            dit_model: The base DiT model to wrap
            config: DQAR configuration
        """
        super().__init__()
        self.dit = dit_model
        self.config = config or DQARConfig()

        # Initialize DQAR components
        self._init_components()

        # Hook storage
        self._hooks = []
        self._attention_outputs = {}

        # Statistics
        self.stats = {
            "total_attention_calls": 0,
            "reused_attention_calls": 0,
            "layer_reuse_counts": [0] * self.config.num_layers,
            "total_time_saved_ms": 0.0,
        }

    def _init_components(self):
        """Initialize DQAR components."""
        # Reuse gate
        gate_config = GateConfig(
            entropy_threshold=self.config.entropy_threshold,
            snr_low=self.config.snr_low,
            snr_high=self.config.snr_high,
            adaptive_entropy=self.config.adaptive_entropy,
            warmup_steps=self.config.warmup_steps,
        )
        self.reuse_gate = ReuseGate(gate_config)

        # KV Cache
        quant_mode = QuantizationMode(self.config.quantization_mode)
        self.kv_cache = QuantizedKVCache(
            num_layers=self.config.num_layers,
            quantization_mode=quant_mode,
            bits=self.config.quantization_bits,
            cache_attention_output=self.config.cache_attention_output,
        )

        # Entropy and SNR computers
        self.entropy_computer = AttentionEntropyComputer(
            num_layers=self.config.num_layers
        )
        self.snr_computer = SNRComputer()

        # Learned policy (optional)
        if self.config.use_learned_policy:
            policy_config = PolicyConfig(reuse_threshold=self.config.policy_threshold)
            self.reuse_policy = ReusePolicy(policy_config)
        else:
            self.reuse_policy = None

        # Layer scheduler
        if self.config.use_layer_scheduling:
            scheduler_config = SchedulerConfig(
                num_layers=self.config.num_layers,
            )
            self.layer_scheduler = LayerScheduler(scheduler_config)
        else:
            self.layer_scheduler = None

        # Current inference state
        self.current_timestep = 0
        self.current_snr = None
        self.current_latent_norm = None

        # Create DQAR manager for attention processor coordination
        self.manager = DQARManager(
            num_layers=self.config.num_layers,
            kv_cache=self.kv_cache,
            layer_scheduler=self.layer_scheduler,
            enabled=True,
        )

        # Track if processors are installed
        self._processors_installed = False
        self._original_processors = None

    def set_timestep(self, timestep_idx: int, total_timesteps: int):
        """
        Set current timestep for scheduling decisions.

        Args:
            timestep_idx: Current timestep index (0 = first step, total-1 = last step)
            total_timesteps: Total number of timesteps
        """
        self.current_timestep = timestep_idx
        self.timestep_idx = timestep_idx

    def update_latent_info(
        self,
        x_t: torch.Tensor,
        x_0: Optional[torch.Tensor] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
    ):
        """
        Update latent information for SNR computation.

        Args:
            x_t: Current noisy latent
            x_0: Predicted clean latent (if available)
            alphas_cumprod: Noise schedule
        """
        # Compute L2 norm per sample by flattening spatial dimensions
        self.current_latent_norm = x_t.flatten(1).norm(dim=1).mean().item()

        if alphas_cumprod is not None:
            self.snr_computer.set_schedule(alphas_cumprod)

        self.current_snr = self.snr_computer.compute(
            x_t, self.current_timestep, x_0=x_0
        ).mean().item()

    def install_attention_processors(self) -> None:
        """
        Install DQAR attention processors on all transformer blocks.

        This replaces the default attention processors with DQARAttentionProcessor
        instances that implement K/V caching and reuse based on SNR + layer scheduling.
        """
        if self._processors_installed:
            return

        # Save original processors for potential restoration
        self._original_processors = {}
        installed_count = 0

        for name, module in self.dit.named_modules():
            # Check if this is an attention module with set_processor method
            if hasattr(module, 'set_processor') and hasattr(module, 'to_q'):
                # Determine if it's self-attention (attn1) or cross-attention (attn2)
                is_cross = 'attn2' in name

                # Only apply DQAR to self-attention for now
                if not is_cross:
                    # Extract layer index from name pattern like 'transformer_blocks.0.attn1'
                    layer_idx = 0
                    try:
                        parts = name.split('.')
                        for i, part in enumerate(parts):
                            if part == 'transformer_blocks' and i + 1 < len(parts):
                                layer_idx = int(parts[i + 1])
                                break
                    except (ValueError, IndexError):
                        pass

                    # Save original processor
                    if hasattr(module, 'processor'):
                        self._original_processors[name] = module.processor

                    # Create and install DQAR processor
                    processor = DQARAttentionProcessor(
                        layer_idx=layer_idx,
                        manager=self.manager,
                        is_cross_attention=is_cross,
                    )
                    module.set_processor(processor)
                    installed_count += 1

        if installed_count > 0:
            self._processors_installed = True
            print(f"[DQAR] Installed {installed_count} attention processors")
        else:
            print("[DQAR] Warning: No attention modules found to install processors")

    def uninstall_attention_processors(self) -> None:
        """Restore original attention processors."""
        if not self._processors_installed:
            return

        if self._original_processors:
            for name, module in self.dit.named_modules():
                if name in self._original_processors and hasattr(module, 'set_processor'):
                    module.set_processor(self._original_processors[name])

        self._processors_installed = False
        self._original_processors = None

    def should_reuse_attention(
        self,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
        entropy: Optional[float] = None,
    ) -> bool:
        """
        Determine if attention should be reused for a layer.

        Args:
            layer_idx: Index of the transformer layer
            attention_weights: Current attention weights (for entropy computation)
            entropy: Pre-computed entropy

        Returns:
            Whether to reuse cached attention
        """
        # Check layer scheduler first
        if self.layer_scheduler is not None:
            if not self.layer_scheduler.can_reuse(
                self.timestep_idx, layer_idx, self.current_snr
            ):
                return False

        # Check if cache exists
        if not self.kv_cache.has_cache(layer_idx):
            return False

        # Compute entropy if not provided
        if entropy is None and attention_weights is not None:
            entropy = compute_attention_entropy(attention_weights).mean().item()
        elif entropy is None:
            entropy = self.kv_cache.get_cached_entropy(layer_idx) or 0.0

        # Use learned policy if available
        if self.reuse_policy is not None:
            should_reuse, prob = self.reuse_policy.predict(
                entropy=torch.tensor([entropy]),
                snr=torch.tensor([self.current_snr or 1.0]),
                latent_norm=torch.tensor([self.current_latent_norm or 1.0]),
                timestep=torch.tensor([self.current_timestep]),
            )
            return should_reuse.item()

        # Use threshold-based gate
        snr_tensor = torch.tensor([self.current_snr]) if self.current_snr else None
        should_reuse, _ = self.reuse_gate(
            entropy=torch.tensor([entropy]),
            snr=snr_tensor,
            timestep_idx=self.timestep_idx,
            layer_idx=layer_idx,
        )
        return should_reuse.all().item()

    def cache_attention(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_output: Optional[torch.Tensor] = None,
        entropy: float = 0.0,
    ):
        """
        Cache K/V tensors for a layer.

        Args:
            layer_idx: Layer index
            key: Key tensor
            value: Value tensor
            attention_output: Pre-computed attention output
            entropy: Current entropy value
        """
        self.kv_cache.store(
            layer_idx=layer_idx,
            key=key,
            value=value,
            attention_output=attention_output,
            entropy=entropy,
            snr=self.current_snr or 0.0,
            timestep=self.current_timestep,
        )

    def get_cached_attention(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Retrieve cached K/V and attention output.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (key, value, attention_output) or None
        """
        return self.kv_cache.retrieve(layer_idx)

    def create_attention_hook(self, layer_idx: int) -> Callable:
        """
        Create a forward hook for attention interception.

        Args:
            layer_idx: Index of the layer to hook

        Returns:
            Hook function
        """
        def hook(module, input, output):
            # This hook is called after the attention forward pass
            # We can use it to cache or record attention patterns
            self._attention_outputs[layer_idx] = output
            return output

        return hook

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with DQAR optimization.

        Args:
            x: Input latent (B, C, H, W)
            timestep: Diffusion timestep
            class_labels: Optional class conditioning
            **kwargs: Additional arguments for the DiT model

        Returns:
            Model output
        """
        # Get total_steps from kwargs (sampler already set timestep via set_timestep)
        total_steps = kwargs.pop("total_timesteps", self.manager.total_timesteps)

        # Update latent info
        self.update_latent_info(x)

        # Sync manager state for attention processors
        self.manager.set_timestep(self.timestep_idx, self.current_snr)
        self.manager.set_total_timesteps(total_steps)

        # Call the underlying DiT model
        # The actual attention reuse logic would be integrated into
        # the attention layers via hooks or subclassing
        output = self.dit(x, timestep, class_labels=class_labels, **kwargs)

        # Handle diffusers output format
        if hasattr(output, 'sample'):
            output = output.sample

        # DiT outputs noise and variance prediction concatenated (8 channels for 4-channel latents)
        # Split and return only the noise prediction
        if output.shape[1] == x.shape[1] * 2:
            output, _ = output.chunk(2, dim=1)

        return output

    def reset(self):
        """Reset state for new inference run."""
        self.kv_cache.clear_all()
        self.reuse_gate.reset_statistics()
        self.entropy_computer.reset()
        self.snr_computer.reset()
        self.current_timestep = 0
        self.current_snr = None
        self.current_latent_norm = None
        self._attention_outputs.clear()

        # Reset manager
        self.manager.reset_for_new_sample()
        self.manager.reset_statistics()

        # Reset stats
        self.stats = {
            "total_attention_calls": 0,
            "reused_attention_calls": 0,
            "layer_reuse_counts": [0] * self.config.num_layers,
            "total_time_saved_ms": 0.0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about DQAR performance."""
        gate_stats = self.reuse_gate.get_decision_summary()
        cache_stats = self.kv_cache.get_statistics()
        manager_stats = self.manager.get_statistics()

        return {
            "reuse_gate": gate_stats,
            "kv_cache": cache_stats,
            "attention_stats": self.stats,
            "manager_stats": manager_stats,
            "overall_reuse_ratio": manager_stats["reuse_ratio"],
        }


class DQARAttentionWrapper(nn.Module):
    """
    Wrapper for individual attention layers that implements DQAR logic.

    This can be used to wrap attention modules within the DiT model
    for fine-grained control over attention reuse.
    """

    def __init__(
        self,
        attention_module: nn.Module,
        layer_idx: int,
        dqar_wrapper: DQARDiTWrapper,
    ):
        """
        Args:
            attention_module: The original attention module
            layer_idx: Index of this layer
            dqar_wrapper: Parent DQAR wrapper for shared state
        """
        super().__init__()
        self.attention = attention_module
        self.layer_idx = layer_idx
        self.dqar = dqar_wrapper

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with DQAR attention reuse logic.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            **kwargs: Additional arguments

        Returns:
            Attention output
        """
        self.dqar.stats["total_attention_calls"] += 1

        # Check if we should reuse cached attention
        if self.dqar.should_reuse_attention(self.layer_idx):
            cached = self.dqar.get_cached_attention(self.layer_idx)
            if cached is not None:
                _, _, attention_output = cached
                if attention_output is not None:
                    self.dqar.stats["reused_attention_calls"] += 1
                    self.dqar.stats["layer_reuse_counts"][self.layer_idx] += 1
                    return attention_output

        # Compute attention normally
        if self.dqar.config.enable_profiling:
            start_time = time.time()

        output = self.attention(hidden_states, attention_mask, **kwargs)

        if self.dqar.config.enable_profiling:
            elapsed_ms = (time.time() - start_time) * 1000

        # Cache the result
        # Note: In a full implementation, we'd also extract K, V from the attention
        # For now, we cache just the output
        self.dqar.cache_attention(
            layer_idx=self.layer_idx,
            key=hidden_states,  # Placeholder
            value=hidden_states,  # Placeholder
            attention_output=output,
        )

        return output
