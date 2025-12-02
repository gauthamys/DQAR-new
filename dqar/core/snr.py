"""
Latent Signal-to-Noise Ratio (SNR) Computation Module

Computes the SNR of latent representations during the diffusion process.
Higher SNR indicates that the signal dominates the noise, which typically
occurs in later timesteps where attention patterns are more stable.
"""

import torch
from typing import Optional, Tuple, Union


def compute_latent_snr(
    x_t: torch.Tensor,
    x_0: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    mode: str = "signal_noise",
) -> torch.Tensor:
    """
    Compute the latent signal-to-noise ratio.

    SNR_t = ||x_0||_2^2 / (||x_t - x_0||_2^2 + eps)

    Args:
        x_t: Noisy latent at timestep t, shape (B, C, H, W) or (B, N, D)
        x_0: Clean latent (predicted or ground truth), same shape as x_t
        noise: Noise tensor (alternative to x_0), same shape as x_t
        eps: Numerical stability constant
        mode: SNR computation mode:
            - "signal_noise": SNR = ||x_0||^2 / ||noise||^2
            - "difference": SNR = ||x_0||^2 / ||x_t - x_0||^2
            - "timestep": Compute from diffusion schedule (requires alpha values)

    Returns:
        SNR tensor of shape (B,)
    """
    # Flatten spatial/sequence dimensions
    B = x_t.shape[0]
    x_t_flat = x_t.view(B, -1)

    if mode == "signal_noise" and noise is not None:
        noise_flat = noise.view(B, -1)
        if x_0 is not None:
            x_0_flat = x_0.view(B, -1)
            signal_power = torch.sum(x_0_flat ** 2, dim=-1)
        else:
            # Estimate signal as x_t - noise
            signal = x_t_flat - noise_flat
            signal_power = torch.sum(signal ** 2, dim=-1)
        noise_power = torch.sum(noise_flat ** 2, dim=-1)
        snr = signal_power / (noise_power + eps)

    elif mode == "difference" and x_0 is not None:
        x_0_flat = x_0.view(B, -1)
        signal_power = torch.sum(x_0_flat ** 2, dim=-1)
        diff = x_t_flat - x_0_flat
        noise_power = torch.sum(diff ** 2, dim=-1)
        snr = signal_power / (noise_power + eps)

    else:
        # Fallback: estimate SNR from latent statistics
        # Use the ratio of low-frequency to high-frequency energy
        snr = estimate_snr_from_latent(x_t, eps=eps)

    return snr


def estimate_snr_from_latent(
    x_t: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate SNR from latent tensor statistics without access to x_0.

    Uses the intuition that signal tends to be smoother (low-frequency)
    while noise is high-frequency.

    Args:
        x_t: Latent tensor of shape (B, C, H, W) or (B, N, D)
        eps: Numerical stability constant

    Returns:
        Estimated SNR of shape (B,)
    """
    B = x_t.shape[0]

    if x_t.dim() == 4:
        # Image latent: (B, C, H, W)
        # Use gradient-based noise estimation
        grad_h = x_t[:, :, 1:, :] - x_t[:, :, :-1, :]
        grad_w = x_t[:, :, :, 1:] - x_t[:, :, :, :-1]

        # Total variation as noise proxy
        noise_proxy = (
            grad_h.abs().mean(dim=(1, 2, 3)) +
            grad_w.abs().mean(dim=(1, 2, 3))
        ) / 2

        # Signal as overall magnitude
        signal_proxy = x_t.abs().mean(dim=(1, 2, 3))

        snr = (signal_proxy ** 2) / (noise_proxy ** 2 + eps)

    else:
        # Sequence latent: (B, N, D) - compute variance ratio
        x_t_flat = x_t.view(B, -1)
        variance = x_t_flat.var(dim=-1)
        mean_sq = x_t_flat.mean(dim=-1) ** 2

        # SNR approximation from statistics
        snr = mean_sq / (variance + eps)

    return snr


def compute_snr_from_schedule(
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact SNR from the diffusion schedule.

    SNR = alpha_bar_t / (1 - alpha_bar_t)

    Args:
        timesteps: Current timestep indices, shape (B,) or scalar
        alphas_cumprod: Cumulative product of alphas from noise schedule, shape (T,)

    Returns:
        SNR values of shape (B,) or scalar
    """
    alpha_bar = alphas_cumprod[timesteps]
    snr = alpha_bar / (1 - alpha_bar + 1e-8)
    return snr


class SNRComputer:
    """
    Stateful class for computing and tracking SNR across timesteps.

    Can operate in different modes depending on available information:
    - With predicted x_0: Most accurate
    - With noise schedule: Fast and reliable
    - From latent statistics: No additional info needed
    """

    def __init__(
        self,
        alphas_cumprod: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
        history_size: int = 100,
    ):
        """
        Args:
            alphas_cumprod: Optional noise schedule for exact SNR computation
            eps: Numerical stability constant
            history_size: Maximum timesteps to store in history
        """
        self.alphas_cumprod = alphas_cumprod
        self.eps = eps
        self.history_size = history_size

        self.snr_history: list[Tuple[int, float]] = []
        self.current_timestep_idx = 0

    def set_schedule(self, alphas_cumprod: torch.Tensor):
        """Set or update the noise schedule."""
        self.alphas_cumprod = alphas_cumprod

    def compute(
        self,
        x_t: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        x_0: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute SNR using the best available method.

        Args:
            x_t: Current noisy latent
            timestep: Current timestep (diffusion schedule index)
            x_0: Optional predicted clean latent
            noise: Optional noise prediction

        Returns:
            SNR value(s)
        """
        B = x_t.shape[0]

        # Method 1: Use diffusion schedule if available
        if self.alphas_cumprod is not None:
            if isinstance(timestep, int):
                timestep = torch.tensor([timestep] * B, device=x_t.device)
            snr = compute_snr_from_schedule(timestep, self.alphas_cumprod.to(x_t.device))

        # Method 2: Use x_0 prediction if available
        elif x_0 is not None:
            snr = compute_latent_snr(x_t, x_0=x_0, eps=self.eps, mode="difference")

        # Method 3: Use noise prediction if available
        elif noise is not None:
            snr = compute_latent_snr(x_t, noise=noise, eps=self.eps, mode="signal_noise")

        # Method 4: Estimate from latent statistics
        else:
            snr = estimate_snr_from_latent(x_t, eps=self.eps)

        # Record history
        mean_snr = snr.mean().item()
        self.snr_history.append((self.current_timestep_idx, mean_snr))
        if len(self.snr_history) > self.history_size:
            self.snr_history = self.snr_history[-self.history_size:]

        return snr

    def step(self):
        """Advance to next timestep."""
        self.current_timestep_idx += 1

    def reset(self):
        """Reset for new inference run."""
        self.snr_history.clear()
        self.current_timestep_idx = 0

    def get_history(self) -> list[Tuple[int, float]]:
        """Get full SNR history."""
        return self.snr_history.copy()

    def is_in_range(self, snr: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Check if SNR values are within the specified range."""
        return (snr >= low) & (snr <= high)
