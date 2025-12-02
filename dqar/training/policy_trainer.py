"""
Policy Training Module

Trains the reuse policy MLP on cached inference traces
labeled by performance impact.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm

from ..policy.mlp_policy import ReusePolicy, PolicyConfig


@dataclass
class TrainingConfig:
    """Configuration for policy training."""
    # Training hyperparameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 100

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Data
    train_split: float = 0.8
    val_split: float = 0.1

    # Logging
    log_interval: int = 100
    save_interval: int = 10
    checkpoint_dir: str = "./checkpoints"

    # Quality threshold for labeling
    fid_threshold: float = 1.0  # FID degradation threshold


@dataclass
class TrainingSample:
    """Single training sample for policy learning."""
    entropy: float
    snr: float
    latent_norm: float
    timestep: int
    layer_idx: int
    reuse_label: int  # 1 = reuse was beneficial, 0 = reuse hurt quality
    quality_impact: float  # FID change from reusing


class PolicyTrainingDataset(Dataset):
    """Dataset for policy training."""

    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "entropy": torch.tensor(sample.entropy, dtype=torch.float32),
            "snr": torch.tensor(sample.snr, dtype=torch.float32),
            "latent_norm": torch.tensor(sample.latent_norm, dtype=torch.float32),
            "timestep": torch.tensor(sample.timestep, dtype=torch.float32),
            "label": torch.tensor(sample.reuse_label, dtype=torch.float32),
            "weight": torch.tensor(abs(sample.quality_impact) + 1.0, dtype=torch.float32),
        }


class PolicyTrainer:
    """
    Trainer for the reuse policy MLP.

    The policy is trained to predict whether reusing attention
    at a given timestep/layer will maintain image quality.
    """

    def __init__(
        self,
        policy: Optional[ReusePolicy] = None,
        config: Optional[TrainingConfig] = None,
        device: str = "cuda",
    ):
        """
        Args:
            policy: Policy network to train. Created if None.
            config: Training configuration
            device: Device for training
        """
        self.config = config or TrainingConfig()
        self.device = device

        if policy is None:
            policy_config = PolicyConfig()
            policy = ReusePolicy(policy_config)
        self.policy = policy.to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history: List[Dict[str, float]] = []

    def prepare_dataloaders(
        self,
        samples: List[TrainingSample],
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare train/val/test dataloaders.

        Args:
            samples: List of training samples

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        n = len(samples)
        n_train = int(n * self.config.train_split)
        n_val = int(n * self.config.val_split)

        # Shuffle samples
        import random
        samples = samples.copy()
        random.shuffle(samples)

        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        train_dataset = PolicyTrainingDataset(train_samples)
        val_dataset = PolicyTrainingDataset(val_samples)
        test_dataset = PolicyTrainingDataset(test_samples)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            entropy = batch["entropy"].to(self.device)
            snr = batch["snr"].to(self.device)
            latent_norm = batch["latent_norm"].to(self.device)
            timestep = batch["timestep"].to(self.device)
            labels = batch["label"].to(self.device)
            weights = batch["weight"].to(self.device)

            self.optimizer.zero_grad()

            loss = self.policy.compute_loss(
                entropy, snr, latent_norm, timestep,
                labels, quality_weights=weights,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the policy."""
        self.policy.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            entropy = batch["entropy"].to(self.device)
            snr = batch["snr"].to(self.device)
            latent_norm = batch["latent_norm"].to(self.device)
            timestep = batch["timestep"].to(self.device)
            labels = batch["label"].to(self.device)

            loss = self.policy.compute_loss(
                entropy, snr, latent_norm, timestep, labels
            )
            total_loss += loss.item() * len(labels)

            # Compute accuracy
            probs = self.policy(entropy, snr, latent_norm, timestep)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def train(
        self,
        samples: List[TrainingSample],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the policy on collected samples.

        Args:
            samples: Training samples
            verbose: Whether to print progress

        Returns:
            Training statistics
        """
        train_loader, val_loader, test_loader = self.prepare_dataloaders(samples)

        # Create checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        iterator = range(self.config.num_epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_accuracy = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Record history
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            if verbose and epoch % self.config.log_interval == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

            # Early stopping
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(checkpoint_dir / "best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Final evaluation on test set
        test_loss, test_accuracy = self.validate(test_loader)

        return {
            "final_train_loss": self.training_history[-1]["train_loss"],
            "final_val_loss": self.training_history[-1]["val_loss"],
            "final_val_accuracy": self.training_history[-1]["val_accuracy"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_val_loss": self.best_val_loss,
            "num_epochs_trained": len(self.training_history),
            "history": self.training_history,
        }

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]


def collect_training_data(
    model: nn.Module,
    sampler: Any,
    prompts: List[str],
    num_samples_per_config: int = 10,
    fid_calculator: Any = None,
    device: str = "cuda",
) -> List[TrainingSample]:
    """
    Collect training data by running inference with different reuse configurations.

    Args:
        model: DiT model
        sampler: DQAR sampler
        prompts: List of text prompts
        num_samples_per_config: Samples per configuration
        fid_calculator: FID calculator for quality measurement
        device: Device to use

    Returns:
        List of training samples
    """
    samples = []

    # For each prompt, compare reuse vs no-reuse at different configurations
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Collecting data")):
        # Generate with different reuse settings
        # This is a simplified version - full implementation would
        # systematically vary entropy thresholds and measure quality impact

        for entropy_threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for snr_range in [(0.1, 5.0), (0.5, 10.0), (1.0, 20.0)]:
                # Configure sampler
                # sampler.model.config.entropy_threshold = entropy_threshold
                # sampler.model.config.snr_low = snr_range[0]
                # sampler.model.config.snr_high = snr_range[1]

                # Generate samples and compute quality
                # ... (actual implementation would generate and evaluate)

                # Create training sample (placeholder)
                sample = TrainingSample(
                    entropy=entropy_threshold,
                    snr=(snr_range[0] + snr_range[1]) / 2,
                    latent_norm=1.0,
                    timestep=500,
                    layer_idx=0,
                    reuse_label=1 if entropy_threshold < 2.0 else 0,
                    quality_impact=0.0,
                )
                samples.append(sample)

    return samples


def load_training_data(path: str) -> List[TrainingSample]:
    """
    Load pre-collected training data from file.

    Args:
        path: Path to JSON file with training data

    Returns:
        List of training samples
    """
    with open(path, "r") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sample = TrainingSample(
            entropy=item["entropy"],
            snr=item["snr"],
            latent_norm=item["latent_norm"],
            timestep=item["timestep"],
            layer_idx=item.get("layer_idx", 0),
            reuse_label=item["reuse_label"],
            quality_impact=item.get("quality_impact", 0.0),
        )
        samples.append(sample)

    return samples


def save_training_data(samples: List[TrainingSample], path: str):
    """
    Save training data to file.

    Args:
        samples: Training samples
        path: Output file path
    """
    data = []
    for sample in samples:
        data.append({
            "entropy": sample.entropy,
            "snr": sample.snr,
            "latent_norm": sample.latent_norm,
            "timestep": sample.timestep,
            "layer_idx": sample.layer_idx,
            "reuse_label": sample.reuse_label,
            "quality_impact": sample.quality_impact,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
