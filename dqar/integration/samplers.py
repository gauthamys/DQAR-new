"""
DQAR-Enhanced Samplers

Modified DDIM and DPMSolver samplers that integrate with DQAR
for efficient attention reuse during the diffusion process.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
from dataclasses import dataclass
import math
from tqdm import tqdm

from .dit_wrapper import DQARDiTWrapper, DQARConfig


@dataclass
class SamplerConfig:
    """Configuration for DQAR samplers."""
    num_inference_steps: int = 50
    eta: float = 0.0  # DDIM eta parameter (0 = deterministic)
    guidance_scale: float = 4.0  # CFG scale
    use_cfg_sharing: bool = True  # Share attention between CFG branches

    # DQAR-specific
    enable_dqar: bool = True
    warmup_steps: int = 2  # Steps before enabling reuse


class DQARDDIMSampler:
    """
    DDIM sampler with DQAR attention reuse integration.

    At each diffusion step:
    1. Computes entropy and SNR for the current latent
    2. Consults the reuse gate or policy
    3. If approved, retrieves quantized K/V and dequantizes
    4. Otherwise, recomputes attention and updates caches
    """

    def __init__(
        self,
        model: Union[nn.Module, DQARDiTWrapper],
        scheduler: Any,  # diffusers scheduler
        config: Optional[SamplerConfig] = None,
    ):
        """
        Args:
            model: DiT model or DQAR-wrapped model
            scheduler: Diffusion noise scheduler
            config: Sampler configuration
        """
        self.config = config or SamplerConfig()

        # Wrap model with DQAR if not already wrapped
        if isinstance(model, DQARDiTWrapper):
            self.model = model
        else:
            dqar_config = DQARConfig(
                warmup_steps=self.config.warmup_steps,
                cfg_sharing=self.config.use_cfg_sharing,
            )
            self.model = DQARDiTWrapper(model, dqar_config)

        self.scheduler = scheduler
        self.device = next(model.parameters()).device

        # Get alphas from scheduler
        if hasattr(scheduler, 'alphas_cumprod'):
            self.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        else:
            self.alphas_cumprod = None

        # Install DQAR attention processors for deep hooks
        if self.config.enable_dqar:
            self.model.install_attention_processors()

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Prepare initial noise latents.

        Args:
            batch_size: Number of samples
            num_channels: Latent channels
            height: Latent height
            width: Latent width
            dtype: Data type
            generator: Random generator

        Returns:
            Initial noise tensor
        """
        shape = (batch_size, num_channels, height, width)
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=dtype,
        )
        return latents * self.scheduler.init_noise_sigma

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        num_channels: int = 4,
        height: int = 32,
        width: int = 32,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        progress_bar: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Sample from the diffusion model using DDIM with DQAR.

        Args:
            batch_size: Number of samples to generate
            num_channels: Number of latent channels
            height: Latent height
            width: Latent width
            class_labels: Optional class conditioning
            generator: Random generator for reproducibility
            return_intermediate: Whether to return intermediate latents
            progress_bar: Whether to show progress bar

        Returns:
            Generated samples, optionally with intermediate states
        """
        # Reset DQAR state
        self.model.reset()

        # Set scheduler timesteps
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare initial latents
        dtype = next(self.model.parameters()).dtype
        latents = self.prepare_latents(
            batch_size, num_channels, height, width, dtype, generator
        )

        # Handle CFG
        do_cfg = self.config.guidance_scale > 1.0
        if do_cfg and class_labels is not None:
            # Duplicate for unconditional + conditional
            # For ImageNet DiT, use class 1000 as the null class for unconditional
            class_labels_cfg = torch.cat([
                torch.full_like(class_labels, 1000),  # Unconditional (null class)
                class_labels,  # Conditional
            ])
        else:
            class_labels_cfg = class_labels

        intermediate_latents = [] if return_intermediate else None

        # Denoising loop
        iterator = tqdm(timesteps, desc="DDIM Sampling") if progress_bar else timesteps
        for i, t in enumerate(iterator):
            # Update DQAR with current state
            # Pass both step index (for scheduling) and actual timestep (for SNR computation)
            actual_t = t.item() if hasattr(t, 'item') else int(t)
            if i == 0:
                print(f"[SAMPLER DEBUG] Step {i}: actual_t={actual_t}, t={t}")
            self.model.set_timestep(i, len(timesteps), actual_timestep=actual_t)
            self.model.update_latent_info(
                latents,
                alphas_cumprod=self.alphas_cumprod,
            )

            # Expand latents for CFG
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            # Scale input
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            timestep_batch = t.expand(latent_model_input.shape[0])
            noise_pred = self.model(
                latent_model_input,
                timestep_batch,
                class_labels=class_labels_cfg,
                total_timesteps=len(timesteps),
            )

            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # DDIM step
            latents = self.scheduler.step(
                noise_pred, t, latents, eta=self.config.eta
            ).prev_sample

            if return_intermediate:
                intermediate_latents.append(latents.clone())

        if return_intermediate:
            return latents, intermediate_latents
        return latents

    def get_statistics(self) -> Dict[str, Any]:
        """Get DQAR statistics from the sampling run."""
        return self.model.get_statistics()


class DQARDPMSolverSampler:
    """
    DPM-Solver sampler with DQAR attention reuse integration.

    DPM-Solver is a fast ODE solver for diffusion models that
    requires fewer function evaluations than DDIM.
    """

    def __init__(
        self,
        model: Union[nn.Module, DQARDiTWrapper],
        scheduler: Any,  # diffusers DPMSolverMultistepScheduler
        config: Optional[SamplerConfig] = None,
    ):
        """
        Args:
            model: DiT model or DQAR-wrapped model
            scheduler: DPM-Solver scheduler
            config: Sampler configuration
        """
        self.config = config or SamplerConfig()

        # Wrap model with DQAR if not already wrapped
        if isinstance(model, DQARDiTWrapper):
            self.model = model
        else:
            dqar_config = DQARConfig(
                warmup_steps=self.config.warmup_steps,
                cfg_sharing=self.config.use_cfg_sharing,
            )
            self.model = DQARDiTWrapper(model, dqar_config)

        self.scheduler = scheduler
        self.device = next(model.parameters()).device

        if hasattr(scheduler, 'alphas_cumprod'):
            self.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
        else:
            self.alphas_cumprod = None

        # Install DQAR attention processors for deep hooks
        if self.config.enable_dqar:
            self.model.install_attention_processors()

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare initial noise latents."""
        shape = (batch_size, num_channels, height, width)
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=dtype,
        )
        return latents * self.scheduler.init_noise_sigma

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        num_channels: int = 4,
        height: int = 32,
        width: int = 32,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        progress_bar: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Sample from the diffusion model using DPM-Solver with DQAR.

        Args:
            batch_size: Number of samples to generate
            num_channels: Number of latent channels
            height: Latent height
            width: Latent width
            class_labels: Optional class conditioning
            generator: Random generator for reproducibility
            return_intermediate: Whether to return intermediate latents
            progress_bar: Whether to show progress bar

        Returns:
            Generated samples, optionally with intermediate states
        """
        # Reset DQAR state
        self.model.reset()

        # Set scheduler timesteps
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Prepare initial latents
        dtype = next(self.model.parameters()).dtype
        latents = self.prepare_latents(
            batch_size, num_channels, height, width, dtype, generator
        )

        # Handle CFG
        do_cfg = self.config.guidance_scale > 1.0
        if do_cfg and class_labels is not None:
            # For ImageNet DiT, use class 1000 as the null class for unconditional
            class_labels_cfg = torch.cat([
                torch.full_like(class_labels, 1000),  # Unconditional (null class)
                class_labels,
            ])
        else:
            class_labels_cfg = class_labels

        intermediate_latents = [] if return_intermediate else None

        # Denoising loop
        iterator = tqdm(timesteps, desc="DPM-Solver Sampling") if progress_bar else timesteps
        for i, t in enumerate(iterator):
            # Update DQAR with current state
            # Pass both step index (for scheduling) and actual timestep (for SNR computation)
            actual_t = t.item() if hasattr(t, 'item') else int(t)
            self.model.set_timestep(i, len(timesteps), actual_timestep=actual_t)
            self.model.update_latent_info(
                latents,
                alphas_cumprod=self.alphas_cumprod,
            )

            # Expand latents for CFG
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            # Scale input
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            timestep_batch = t.expand(latent_model_input.shape[0])
            noise_pred = self.model(
                latent_model_input,
                timestep_batch,
                class_labels=class_labels_cfg,
                total_timesteps=len(timesteps),
            )

            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # DPM-Solver step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if return_intermediate:
                intermediate_latents.append(latents.clone())

        if return_intermediate:
            return latents, intermediate_latents
        return latents

    def get_statistics(self) -> Dict[str, Any]:
        """Get DQAR statistics from the sampling run."""
        return self.model.get_statistics()


def create_sampler(
    model: nn.Module,
    scheduler_type: str = "ddim",
    scheduler: Optional[Any] = None,
    config: Optional[SamplerConfig] = None,
) -> Union[DQARDDIMSampler, DQARDPMSolverSampler]:
    """
    Factory function to create the appropriate DQAR sampler.

    Args:
        model: DiT model
        scheduler_type: Type of scheduler ("ddim" or "dpm_solver")
        scheduler: Pre-configured scheduler (optional)
        config: Sampler configuration

    Returns:
        Configured DQAR sampler
    """
    if scheduler is None:
        # Import diffusers schedulers
        try:
            from diffusers import DDIMScheduler, DPMSolverMultistepScheduler

            if scheduler_type == "ddim":
                scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="linear",
                    clip_sample=False,
                )
            elif scheduler_type == "dpm_solver":
                scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="linear",
                    algorithm_type="dpmsolver++",
                    solver_order=2,
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        except ImportError:
            raise ImportError("Please install diffusers: pip install diffusers")

    if scheduler_type == "ddim":
        return DQARDDIMSampler(model, scheduler, config)
    elif scheduler_type == "dpm_solver":
        return DQARDPMSolverSampler(model, scheduler, config)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
