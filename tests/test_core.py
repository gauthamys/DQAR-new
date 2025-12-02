"""Tests for DQAR core modules."""

import pytest
import torch

from dqar.core.entropy import compute_attention_entropy, AttentionEntropyComputer
from dqar.core.snr import compute_latent_snr, SNRComputer, compute_snr_from_schedule
from dqar.core.quantization import KVQuantizer, QuantizationMode, compute_quantization_error
from dqar.core.reuse_gate import ReuseGate, GateConfig


class TestEntropy:
    """Tests for entropy computation."""

    def test_compute_attention_entropy_shape(self):
        """Test entropy computation output shape."""
        batch_size, num_heads, seq_len = 2, 8, 64
        attention_weights = torch.softmax(
            torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1
        )

        entropy = compute_attention_entropy(attention_weights)

        assert entropy.shape == (batch_size,)
        assert torch.all(entropy >= 0)

    def test_entropy_uniform_vs_peaked(self):
        """Test that uniform attention has higher entropy than peaked."""
        batch_size, num_heads, seq_len = 1, 4, 32

        # Uniform attention
        uniform = torch.ones(batch_size, num_heads, seq_len, seq_len) / seq_len

        # Peaked attention (mostly on diagonal)
        peaked = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        peaked[:, :, range(seq_len), range(seq_len)] = 0.9
        peaked = peaked + 0.1 / seq_len
        peaked = peaked / peaked.sum(dim=-1, keepdim=True)

        entropy_uniform = compute_attention_entropy(uniform)
        entropy_peaked = compute_attention_entropy(peaked)

        assert entropy_uniform > entropy_peaked

    def test_entropy_computer(self):
        """Test AttentionEntropyComputer stateful tracking."""
        num_layers = 4
        computer = AttentionEntropyComputer(num_layers=num_layers)

        attention = torch.softmax(torch.randn(1, 8, 32, 32), dim=-1)

        for layer_idx in range(num_layers):
            entropy = computer.compute(attention, layer_idx)
            assert entropy.shape == (1,)

        computer.step()
        assert computer.current_timestep == 1

        history = computer.get_layer_entropy_history(0)
        assert len(history) > 0


class TestSNR:
    """Tests for SNR computation."""

    def test_compute_latent_snr_shape(self):
        """Test SNR computation output shape."""
        batch_size, channels, height, width = 2, 4, 32, 32
        x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)

        snr = compute_latent_snr(x_t, x_0=x_0, mode="difference")

        assert snr.shape == (batch_size,)
        assert torch.all(snr >= 0)

    def test_snr_high_vs_low_noise(self):
        """Test that clean signal has higher SNR than noisy."""
        x_0 = torch.randn(1, 4, 32, 32)
        noise_low = torch.randn_like(x_0) * 0.1
        noise_high = torch.randn_like(x_0) * 2.0

        x_t_low_noise = x_0 + noise_low
        x_t_high_noise = x_0 + noise_high

        snr_low = compute_latent_snr(x_t_low_noise, x_0=x_0)
        snr_high = compute_latent_snr(x_t_high_noise, x_0=x_0)

        assert snr_low > snr_high

    def test_snr_from_schedule(self):
        """Test SNR computation from diffusion schedule."""
        timesteps = torch.tensor([0, 500, 999])
        alphas_cumprod = torch.linspace(0.9999, 0.0001, 1000)

        snr = compute_snr_from_schedule(timesteps, alphas_cumprod)

        assert snr.shape == (3,)
        # Early timesteps should have higher SNR
        assert snr[0] > snr[1] > snr[2]


class TestQuantization:
    """Tests for KV quantization."""

    def test_quantize_dequantize_per_tensor(self):
        """Test per-tensor quantization roundtrip."""
        quantizer = KVQuantizer(mode=QuantizationMode.PER_TENSOR, bits=8)

        original = torch.randn(2, 8, 64, 64)
        quantized = quantizer.quantize(original)
        dequantized = quantizer.dequantize(quantized)

        assert quantized.data.dtype == torch.int8
        assert dequantized.shape == original.shape

        # Check reasonable reconstruction error
        error = (original - dequantized).abs().mean()
        assert error < 0.1  # Should be small for 8-bit

    def test_quantize_dequantize_per_channel(self):
        """Test per-channel quantization."""
        quantizer = KVQuantizer(mode=QuantizationMode.PER_CHANNEL, bits=8)

        original = torch.randn(2, 8, 64, 64)
        quantized = quantizer.quantize(original)
        dequantized = quantizer.dequantize(quantized)

        assert quantized.data.dtype == torch.int8
        assert dequantized.shape == original.shape

    def test_quantization_error_metrics(self):
        """Test quantization error computation."""
        quantizer = KVQuantizer(mode=QuantizationMode.PER_TENSOR)

        original = torch.randn(2, 8, 64, 64)
        quantized = quantizer.quantize(original)

        metrics = compute_quantization_error(original, quantized, quantizer)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "sqnr_db" in metrics
        assert metrics["mse"] >= 0
        assert metrics["sqnr_db"] > 0  # Should be positive for reasonable quant


class TestReuseGate:
    """Tests for reuse gate logic."""

    def test_gate_initialization(self):
        """Test gate initialization with config."""
        config = GateConfig(
            entropy_threshold=2.0,
            snr_low=0.1,
            snr_high=10.0,
        )
        gate = ReuseGate(config)

        assert gate.config.entropy_threshold == 2.0
        assert gate.config.snr_low == 0.1
        assert gate.config.snr_high == 10.0

    def test_gate_warmup(self):
        """Test that gate blocks reuse during warmup."""
        config = GateConfig(warmup_steps=3)
        gate = ReuseGate(config)

        entropy = torch.tensor([1.0])  # Low entropy (should reuse)
        snr = torch.tensor([1.0])  # In range

        # During warmup
        should_reuse, decision = gate(
            entropy=entropy, snr=snr, timestep_idx=0
        )
        assert not should_reuse.all()
        assert "warmup" in decision.reason

    def test_gate_entropy_condition(self):
        """Test entropy-based gating."""
        config = GateConfig(entropy_threshold=2.0, warmup_steps=0)
        gate = ReuseGate(config)

        snr = torch.tensor([1.0])  # In range

        # Low entropy - should reuse
        should_reuse_low, _ = gate(entropy=torch.tensor([1.0]), snr=snr, timestep_idx=5)

        # High entropy - should not reuse
        should_reuse_high, _ = gate(entropy=torch.tensor([3.0]), snr=snr, timestep_idx=5)

        assert should_reuse_low.all()
        assert not should_reuse_high.all()

    def test_gate_snr_condition(self):
        """Test SNR-based gating."""
        config = GateConfig(
            entropy_threshold=10.0,  # High threshold (always pass entropy)
            snr_low=0.5,
            snr_high=5.0,
            warmup_steps=0,
        )
        gate = ReuseGate(config)

        entropy = torch.tensor([1.0])

        # SNR in range
        should_reuse_in, _ = gate(entropy=entropy, snr=torch.tensor([2.0]), timestep_idx=5)

        # SNR too low
        should_reuse_low, _ = gate(entropy=entropy, snr=torch.tensor([0.1]), timestep_idx=5)

        # SNR too high
        should_reuse_high, _ = gate(entropy=entropy, snr=torch.tensor([10.0]), timestep_idx=5)

        assert should_reuse_in.all()
        assert not should_reuse_low.all()
        assert not should_reuse_high.all()

    def test_gate_statistics(self):
        """Test gate statistics tracking."""
        config = GateConfig(warmup_steps=0)
        gate = ReuseGate(config)

        # Make some decisions
        for i in range(10):
            entropy = torch.tensor([1.0 if i % 2 == 0 else 3.0])
            gate(entropy=entropy, snr=torch.tensor([1.0]), timestep_idx=i)

        summary = gate.get_decision_summary()

        assert summary["total_decisions"] == 10
        assert summary["reuse_count"] + summary["recompute_count"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
