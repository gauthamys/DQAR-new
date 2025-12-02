"""
Quantized KV Caching Module

Implements 8-bit quantization for Key and Value tensors to reduce
VRAM usage and memory bandwidth during attention computation.
Supports per-tensor and per-channel scaling modes.
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass


class QuantizationMode(Enum):
    """Quantization granularity modes."""
    PER_TENSOR = "per_tensor"  # Single scale for entire tensor (fastest)
    PER_CHANNEL = "per_channel"  # Scale per channel (best fidelity)
    PER_HEAD = "per_head"  # Scale per attention head
    MIXED = "mixed"  # K in FP16, V quantized


@dataclass
class QuantizedTensor:
    """Container for quantized tensor data."""
    data: torch.Tensor  # int8 quantized values
    scale: torch.Tensor  # Scale factor(s) for dequantization
    zero_point: Optional[torch.Tensor] = None  # Optional zero point for asymmetric quant
    original_dtype: torch.dtype = torch.float16
    mode: QuantizationMode = QuantizationMode.PER_TENSOR


class KVQuantizer(nn.Module):
    """
    Quantizer for Key and Value tensors in attention.

    Implements symmetric 8-bit quantization:
    K_q = clip(round(K / s_K), -127, 127)
    s_K = max(|K|) / 127

    Supports multiple quantization modes for different efficiency/quality trade-offs.
    """

    def __init__(
        self,
        mode: QuantizationMode = QuantizationMode.PER_TENSOR,
        bits: int = 8,
        symmetric: bool = True,
    ):
        """
        Args:
            mode: Quantization granularity mode
            bits: Number of bits for quantization (default: 8)
            symmetric: Whether to use symmetric quantization (default: True)
        """
        super().__init__()
        self.mode = mode
        self.bits = bits
        self.symmetric = symmetric

        # Compute quantization range
        if symmetric:
            self.qmin = -(2 ** (bits - 1)) + 1  # -127 for 8-bit
            self.qmax = 2 ** (bits - 1) - 1  # 127 for 8-bit
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1  # 255 for 8-bit

    def quantize(
        self,
        tensor: torch.Tensor,
        dim: Optional[int] = None,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to int8.

        Args:
            tensor: Input tensor of shape (B, H, T, D) for attention K/V
            dim: Dimension for per-channel/per-head quantization

        Returns:
            QuantizedTensor containing int8 data and scale factors
        """
        original_dtype = tensor.dtype

        if self.mode == QuantizationMode.PER_TENSOR:
            scale = self._compute_scale_per_tensor(tensor)
            quantized = self._quantize_symmetric(tensor, scale)

        elif self.mode == QuantizationMode.PER_CHANNEL:
            # Quantize per last dimension (D)
            scale = self._compute_scale_per_channel(tensor, dim=-1)
            quantized = self._quantize_symmetric(tensor, scale)

        elif self.mode == QuantizationMode.PER_HEAD:
            # Quantize per head dimension (H)
            scale = self._compute_scale_per_channel(tensor, dim=1)
            quantized = self._quantize_symmetric(tensor, scale)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return QuantizedTensor(
            data=quantized,
            scale=scale,
            original_dtype=original_dtype,
            mode=self.mode,
        )

    def dequantize(self, qtensor: QuantizedTensor) -> torch.Tensor:
        """
        Dequantize an int8 tensor back to floating point.

        Args:
            qtensor: QuantizedTensor to dequantize

        Returns:
            Dequantized tensor in original dtype
        """
        # Convert int8 to float and scale
        dequantized = qtensor.data.float() * qtensor.scale
        return dequantized.to(qtensor.original_dtype)

    def _compute_scale_per_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute single scale factor for entire tensor."""
        max_val = tensor.abs().max()
        scale = max_val / self.qmax
        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)
        return scale

    def _compute_scale_per_channel(
        self,
        tensor: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """Compute scale factors per channel along specified dimension."""
        # Get max absolute value along all dims except the target
        dims_to_reduce = [i for i in range(tensor.dim()) if i != dim and i != dim % tensor.dim()]

        max_vals = tensor.abs()
        for d in sorted(dims_to_reduce, reverse=True):
            max_vals = max_vals.max(dim=d, keepdim=True)[0]

        scale = max_vals / self.qmax
        scale = torch.clamp(scale, min=1e-8)
        return scale

    def _quantize_symmetric(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Apply symmetric quantization."""
        scaled = tensor / scale
        quantized = torch.clamp(torch.round(scaled), self.qmin, self.qmax)
        return quantized.to(torch.int8)


class MixedPrecisionKVQuantizer(nn.Module):
    """
    Mixed-precision quantizer that keeps K in FP16 and quantizes V.

    This variant is useful when K precision is more important for
    attention score accuracy, while V can tolerate more quantization.
    """

    def __init__(
        self,
        v_mode: QuantizationMode = QuantizationMode.PER_TENSOR,
        v_bits: int = 8,
    ):
        """
        Args:
            v_mode: Quantization mode for V tensors
            v_bits: Number of bits for V quantization
        """
        super().__init__()
        self.v_quantizer = KVQuantizer(mode=v_mode, bits=v_bits)

    def quantize_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, QuantizedTensor]:
        """
        Quantize K/V pair with mixed precision.

        Args:
            key: Key tensor (B, H, T, D)
            value: Value tensor (B, H, T, D)

        Returns:
            Tuple of (K in FP16, quantized V)
        """
        # Keep K in original precision (typically FP16)
        k_out = key.to(torch.float16)
        # Quantize V
        v_quantized = self.v_quantizer.quantize(value)
        return k_out, v_quantized

    def dequantize_v(self, v_quantized: QuantizedTensor) -> torch.Tensor:
        """Dequantize V tensor."""
        return self.v_quantizer.dequantize(v_quantized)


def compute_quantization_error(
    original: torch.Tensor,
    quantized: QuantizedTensor,
    quantizer: KVQuantizer,
) -> dict:
    """
    Compute metrics for quantization error analysis.

    Args:
        original: Original tensor
        quantized: Quantized tensor
        quantizer: Quantizer used

    Returns:
        Dictionary with error metrics
    """
    dequantized = quantizer.dequantize(quantized)

    # Mean Squared Error
    mse = ((original - dequantized) ** 2).mean().item()

    # Mean Absolute Error
    mae = (original - dequantized).abs().mean().item()

    # Signal-to-Quantization-Noise Ratio
    signal_power = (original ** 2).mean().item()
    sqnr = 10 * torch.log10(torch.tensor(signal_power / (mse + 1e-10))).item()

    # Max error
    max_error = (original - dequantized).abs().max().item()

    return {
        "mse": mse,
        "mae": mae,
        "sqnr_db": sqnr,
        "max_error": max_error,
    }
