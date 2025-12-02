#!/usr/bin/env python3
"""
DQAR Policy Training Script

Train the reuse policy MLP on collected inference traces.
"""

import argparse
import torch
from pathlib import Path
import json

from dqar.policy import ReusePolicy, PolicyConfig
from dqar.training import PolicyTrainer, TrainingConfig, load_training_data, save_training_data
from dqar.training.policy_trainer import TrainingSample
from dqar.utils import seed_everything, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQAR Reuse Policy")

    # Data settings
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--output-dir", type=str, default="./policy_checkpoints",
                        help="Output directory for checkpoints")

    # Model settings
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32],
                        help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training settings
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def generate_synthetic_data(num_samples: int = 10000) -> list:
    """Generate synthetic training data for demonstration."""
    import random

    samples = []
    for _ in range(num_samples):
        # Simulate different scenarios
        entropy = random.uniform(0.5, 5.0)
        snr = random.uniform(0.01, 50.0)
        timestep = random.randint(0, 999)
        latent_norm = random.uniform(0.5, 2.0)
        layer_idx = random.randint(0, 27)

        # Heuristic labeling:
        # Reuse is good when:
        # - Low entropy (< 2.0)
        # - SNR in medium range (0.1 - 10)
        # - Later timesteps (lower noise)
        # - Shallower layers

        entropy_ok = entropy < 2.0
        snr_ok = 0.1 < snr < 10.0
        timestep_ok = timestep < 700  # Earlier in sampling = later timestep index
        layer_ok = layer_idx < 20

        # Combine with some noise
        score = (
            (0.4 if entropy_ok else 0) +
            (0.3 if snr_ok else 0) +
            (0.2 if timestep_ok else 0) +
            (0.1 if layer_ok else 0)
        )
        score += random.uniform(-0.1, 0.1)

        reuse_label = 1 if score > 0.5 else 0
        quality_impact = random.uniform(-0.5, 0.5) if reuse_label else random.uniform(-1.0, 0.0)

        sample = TrainingSample(
            entropy=entropy,
            snr=snr,
            latent_norm=latent_norm,
            timestep=timestep,
            layer_idx=layer_idx,
            reuse_label=reuse_label,
            quality_impact=quality_impact,
        )
        samples.append(sample)

    return samples


def main():
    args = parse_args()

    # Setup
    seed_everything(args.seed)
    device = args.device or get_device()
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if Path(args.data_path).exists():
        print(f"Loading training data from {args.data_path}...")
        samples = load_training_data(args.data_path)
    else:
        print(f"Data file not found. Generating synthetic data...")
        samples = generate_synthetic_data(10000)
        save_training_data(samples, args.data_path)
        print(f"Saved synthetic data to {args.data_path}")

    print(f"Loaded {len(samples)} training samples")

    # Count label distribution
    num_positive = sum(1 for s in samples if s.reuse_label == 1)
    print(f"Label distribution: {num_positive} positive, {len(samples) - num_positive} negative")

    # Create policy
    policy_config = PolicyConfig(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    policy = ReusePolicy(policy_config)
    print(f"Policy parameters: {policy.get_num_parameters():,}")

    # Create trainer
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=str(output_dir),
    )
    trainer = PolicyTrainer(policy, training_config, device)

    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    results = trainer.train(samples, verbose=True)

    # Print results
    print("\n--- Training Results ---")
    print(f"Final train loss: {results['final_train_loss']:.4f}")
    print(f"Final val loss: {results['final_val_loss']:.4f}")
    print(f"Final val accuracy: {results['final_val_accuracy']:.4f}")
    print(f"Test loss: {results['test_loss']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Epochs trained: {results['num_epochs_trained']}")

    # Save final model
    final_path = output_dir / "final_policy.pt"
    policy.save(str(final_path))
    print(f"\nSaved final model to {final_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(results['history'], f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
