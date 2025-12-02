"""Tests for DQAR cache and policy modules."""

import pytest
import torch

from dqar.cache.kv_cache import QuantizedKVCache, LayerWiseKVCache
from dqar.core.quantization import QuantizationMode
from dqar.policy.mlp_policy import ReusePolicy, PolicyConfig, LayerWiseReusePolicy
from dqar.policy.layer_scheduler import LayerScheduler, SchedulerConfig, ScheduleType


class TestKVCache:
    """Tests for KV cache."""

    def test_cache_store_retrieve(self):
        """Test basic store and retrieve operations."""
        cache = QuantizedKVCache(
            num_layers=4,
            quantization_mode=QuantizationMode.PER_TENSOR,
        )

        key = torch.randn(2, 8, 64, 64)
        value = torch.randn(2, 8, 64, 64)

        cache.store(layer_idx=0, key=key, value=value, entropy=1.5, snr=2.0, timestep=100)

        assert cache.has_cache(0)
        assert not cache.has_cache(1)

        result = cache.retrieve(0)
        assert result is not None
        k, v, attn_out = result
        assert k.shape == key.shape
        assert v.shape == value.shape

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = QuantizedKVCache(num_layers=4)

        for i in range(4):
            cache.store(i, torch.randn(1, 8, 32, 32), torch.randn(1, 8, 32, 32))

        cache.clear(0)
        assert not cache.has_cache(0)
        assert cache.has_cache(1)

        cache.clear()
        for i in range(4):
            assert not cache.has_cache(i)

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = QuantizedKVCache(num_layers=2)

        cache.store(0, torch.randn(1, 8, 32, 32), torch.randn(1, 8, 32, 32))

        # Hit
        cache.retrieve(0)
        # Miss
        cache.retrieve(1)

        stats = cache.get_statistics()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_memory_tracking(self):
        """Test memory usage tracking."""
        cache = QuantizedKVCache(num_layers=2)

        cache.store(0, torch.randn(1, 8, 64, 64), torch.randn(1, 8, 64, 64))

        usage = cache.get_memory_usage()
        assert usage["total_bytes"] > 0
        assert usage["total_mb"] > 0


class TestReusePolicy:
    """Tests for reuse policy MLP."""

    def test_policy_forward(self):
        """Test policy forward pass."""
        config = PolicyConfig(hidden_dims=[32, 16])
        policy = ReusePolicy(config)

        entropy = torch.tensor([1.5])
        snr = torch.tensor([2.0])
        latent_norm = torch.tensor([1.0])
        timestep = torch.tensor([500])

        prob = policy(entropy, snr, latent_norm, timestep)

        assert prob.shape == (1,)
        assert 0 <= prob.item() <= 1

    def test_policy_predict(self):
        """Test policy prediction with decision."""
        policy = ReusePolicy()

        should_reuse, prob = policy.predict(
            entropy=torch.tensor([1.0]),
            snr=torch.tensor([1.0]),
            latent_norm=torch.tensor([1.0]),
            timestep=torch.tensor([500]),
        )

        assert should_reuse.dtype == torch.bool
        assert 0 <= prob.item() <= 1

    def test_policy_compute_loss(self):
        """Test policy loss computation."""
        policy = ReusePolicy()

        batch_size = 8
        entropy = torch.rand(batch_size) * 3
        snr = torch.rand(batch_size) * 10
        latent_norm = torch.rand(batch_size) * 2
        timestep = torch.randint(0, 1000, (batch_size,)).float()
        labels = torch.randint(0, 2, (batch_size,)).float()

        loss = policy.compute_loss(entropy, snr, latent_norm, timestep, labels)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_policy_num_parameters(self):
        """Test policy parameter count."""
        config = PolicyConfig(hidden_dims=[64, 32])
        policy = ReusePolicy(config)

        num_params = policy.get_num_parameters()

        # Should be less than 0.5M as specified in proposal
        assert num_params < 500000
        assert num_params > 0

    def test_policy_save_load(self, tmp_path):
        """Test policy save and load."""
        policy = ReusePolicy()

        # Forward pass to ensure weights are initialized
        _ = policy(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([100.0]),
        )

        path = tmp_path / "policy.pt"
        policy.save(str(path))

        loaded = ReusePolicy.load(str(path))

        # Check same predictions
        test_input = (
            torch.tensor([1.5]),
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            torch.tensor([500.0]),
        )
        assert torch.allclose(policy(*test_input), loaded(*test_input))


class TestLayerScheduler:
    """Tests for layer scheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        config = SchedulerConfig(
            num_layers=28,
            num_timesteps=50,
            schedule_type=ScheduleType.STEP,
        )
        scheduler = LayerScheduler(config)

        assert scheduler.config.num_layers == 28
        assert scheduler.config.num_timesteps == 50

    def test_scheduler_linear_progression(self):
        """Test linear schedule increases reusable layers."""
        config = SchedulerConfig(
            num_layers=12,
            num_timesteps=10,
            schedule_type=ScheduleType.LINEAR,
        )
        scheduler = LayerScheduler(config)

        reusable_early = scheduler.get_reusable_layers(0)
        reusable_late = scheduler.get_reusable_layers(9)

        assert len(reusable_late) >= len(reusable_early)

    def test_scheduler_step_phases(self):
        """Test step schedule has distinct phases."""
        config = SchedulerConfig(
            num_layers=12,
            num_timesteps=10,
            schedule_type=ScheduleType.STEP,
            early_phase_end=0.3,
            mid_phase_end=0.7,
        )
        scheduler = LayerScheduler(config)

        early_layers = scheduler.get_reusable_layers(1)  # Early phase
        mid_layers = scheduler.get_reusable_layers(5)  # Mid phase
        late_layers = scheduler.get_reusable_layers(9)  # Late phase

        assert len(early_layers) <= len(mid_layers) <= len(late_layers)

    def test_scheduler_can_reuse(self):
        """Test can_reuse method."""
        config = SchedulerConfig(num_layers=12, num_timesteps=10)
        scheduler = LayerScheduler(config)

        # Get reusable layers at timestep 5
        reusable = scheduler.get_reusable_layers(5)

        for layer_idx in range(12):
            can_reuse = scheduler.can_reuse(5, layer_idx)
            assert can_reuse == (layer_idx in reusable)

    def test_scheduler_mask(self):
        """Test get_reuse_mask method."""
        config = SchedulerConfig(num_layers=8, num_timesteps=10)
        scheduler = LayerScheduler(config)

        mask = scheduler.get_reuse_mask(5)

        assert mask.shape == (8,)
        assert mask.dtype == torch.bool

        reusable = scheduler.get_reusable_layers(5)
        for i in range(8):
            assert mask[i].item() == (i in reusable)


class TestLayerWisePolicy:
    """Tests for layer-wise reuse policy."""

    def test_layer_wise_forward(self):
        """Test layer-wise policy forward pass."""
        num_layers = 4
        policy = LayerWiseReusePolicy(num_layers=num_layers)

        for layer_idx in range(num_layers):
            prob = policy(
                entropy=torch.tensor([1.5]),
                snr=torch.tensor([2.0]),
                latent_norm=torch.tensor([1.0]),
                timestep=torch.tensor([500]),
                layer_idx=layer_idx,
            )
            assert 0 <= prob.item() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
