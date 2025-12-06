#!/usr/bin/env python3
"""
Train the Graph VAE on L-bracket graphs.

This script trains a variational autoencoder to encode face-adjacency graphs
into a latent space and reconstruct the graph features.

Usage:
    python scripts/train_vae.py --epochs 100 --latent-dim 64
    python scripts/train_vae.py --beta-strategy warmup --target-beta 1.0
    python scripts/train_vae.py --use-semantic-loss --regressor-checkpoint outputs/regressor/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from graph_cad.data import create_data_loaders
from graph_cad.models import GraphVAE, GraphVAEConfig
from graph_cad.models.losses import VAELossConfig
from graph_cad.training.vae_trainer import (
    BetaScheduleConfig,
    BetaScheduler,
    compute_latent_metrics,
    evaluate,
    load_checkpoint as load_regressor_checkpoint,
    save_checkpoint,
    train_epoch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train Graph VAE on L-bracket graphs"
    )

    # Data arguments
    parser.add_argument(
        "--train-size", type=int, default=5000, help="Training set size"
    )
    parser.add_argument(
        "--val-size", type=int, default=500, help="Validation set size"
    )
    parser.add_argument(
        "--test-size", type=int, default=500, help="Test set size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Model arguments
    parser.add_argument(
        "--latent-dim", type=int, default=64, help="Latent space dimension"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension for encoder"
    )
    parser.add_argument(
        "--num-gat-layers", type=int, default=3, help="Number of GAT layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--decoder-hidden",
        type=int,
        nargs="+",
        default=[256, 256, 128],
        help="Decoder hidden layer sizes",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Beta scheduling arguments
    parser.add_argument(
        "--beta-strategy",
        type=str,
        choices=["constant", "linear", "warmup", "cyclical"],
        default="warmup",
        help="Beta scheduling strategy",
    )
    parser.add_argument(
        "--target-beta", type=float, default=1.0, help="Target beta value"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=10, help="Epochs with beta=0"
    )
    parser.add_argument(
        "--anneal-epochs", type=int, default=20, help="Epochs to anneal beta"
    )
    parser.add_argument(
        "--free-bits", type=float, default=0.0, help="Free bits per dimension"
    )

    # Loss weighting arguments
    parser.add_argument(
        "--node-weight", type=float, default=1.0, help="Weight for node reconstruction"
    )
    parser.add_argument(
        "--edge-weight", type=float, default=1.0, help="Weight for edge reconstruction"
    )

    # Semantic loss arguments
    parser.add_argument(
        "--use-semantic-loss",
        action="store_true",
        help="Use semantic loss with pre-trained regressor",
    )
    parser.add_argument(
        "--regressor-checkpoint",
        type=Path,
        default=None,
        help="Path to regressor checkpoint for semantic loss",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.1,
        help="Weight for semantic loss",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vae"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print(f"\nCreating datasets (train={args.train_size}, val={args.val_size}, test={args.test_size})...")
    start_time = time.time()
    train_loader, val_loader, test_loader = create_data_loaders(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Dataset creation took {time.time() - start_time:.1f}s")

    # Create model
    config = GraphVAEConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        decoder_hidden_dims=tuple(args.decoder_hidden),
        encoder_dropout=args.dropout,
        decoder_dropout=args.dropout,
    )
    model = GraphVAE(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {num_params:,} trainable parameters")
    print(f"Config: latent_dim={config.latent_dim}, hidden_dim={config.hidden_dim}")
    print(f"Decoder: {config.decoder_hidden_dims}")

    # Beta scheduler
    beta_config = BetaScheduleConfig(
        strategy=args.beta_strategy,
        target_beta=args.target_beta,
        warmup_epochs=args.warmup_epochs,
        anneal_epochs=args.anneal_epochs,
    )
    beta_scheduler = BetaScheduler(beta_config)
    print(f"Beta schedule: {args.beta_strategy}, target={args.target_beta}")

    # Loss config
    loss_config = VAELossConfig(
        node_weight=args.node_weight,
        edge_weight=args.edge_weight,
    )

    # Optional: Load regressor for semantic loss
    regressor = None
    if args.use_semantic_loss:
        if args.regressor_checkpoint is None:
            # Try default path
            default_path = Path("outputs/regressor/best_model.pt")
            if default_path.exists():
                args.regressor_checkpoint = default_path
            else:
                print("WARNING: --use-semantic-loss specified but no regressor checkpoint found")
                print("         Disabling semantic loss")
                args.use_semantic_loss = False

        if args.use_semantic_loss:
            print(f"Loading regressor from {args.regressor_checkpoint}")
            from graph_cad.models import ParameterRegressor, ParameterRegressorConfig

            checkpoint = torch.load(args.regressor_checkpoint, map_location=device, weights_only=False)
            reg_config = ParameterRegressorConfig(**checkpoint["config"])
            regressor = ParameterRegressor(reg_config).to(device)
            regressor.load_state_dict(checkpoint["model_state_dict"])
            regressor.eval()
            for p in regressor.parameters():
                p.requires_grad = False
            print(f"Semantic loss enabled (weight={args.semantic_weight})")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 100)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Get current beta
        beta = beta_scheduler.get_beta(epoch)

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            beta,
            device,
            loss_config=loss_config,
            regressor=regressor,
            semantic_weight=args.semantic_weight,
            free_bits=args.free_bits,
        )

        # Validate
        val_metrics = evaluate(
            model,
            val_loader,
            beta,
            device,
            loss_config=loss_config,
            regressor=regressor,
            semantic_weight=args.semantic_weight,
            free_bits=args.free_bits,
        )

        # Update LR scheduler
        scheduler.step()

        # Record history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Log progress
        epoch_time = time.time() - epoch_start
        sem_str = f" | Sem: {val_metrics.get('semantic_loss', 0):.4f}" if regressor else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Recon: {val_metrics['recon_loss']:.4f} | "
            f"KL: {val_metrics['kl_loss']:.4f} | "
            f"Beta: {beta:.3f}{sem_str} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(args.output_dir / "best_model.pt")
            )

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(args.output_dir / f"checkpoint_epoch_{epoch}.pt")
            )

    print("-" * 100)
    print("\nTraining complete!")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, args.target_beta, device,
        loss_config=loss_config, regressor=regressor,
        semantic_weight=args.semantic_weight
    )

    print(f"\nTest Results:")
    print(f"  Total Loss: {test_metrics['loss']:.4f}")
    print(f"  Recon Loss: {test_metrics['recon_loss']:.4f}")
    print(f"  KL Loss:    {test_metrics['kl_loss']:.4f}")
    print(f"  Node MSE:   {test_metrics['node_mse']:.6f}")
    print(f"  Edge MSE:   {test_metrics['edge_mse']:.6f}")
    if regressor:
        print(f"  Semantic:   {test_metrics['semantic_loss']:.4f}")

    # Compute latent space metrics
    print("\nComputing latent space metrics...")
    latent_metrics = compute_latent_metrics(model, test_loader, device)
    print(f"  Mean ||z||:      {latent_metrics['mean_norm']:.3f}")
    print(f"  Mean std(z):     {latent_metrics['mean_std']:.3f}")
    print(f"  Active dims:     {latent_metrics['active_dims']}/{config.latent_dim} ({latent_metrics['active_dims_ratio']:.1%})")
    print(f"  KL from prior:   {latent_metrics['kl_from_prior']:.2f}")

    # Load best model and evaluate
    print("\nEvaluating best model on test set...")
    best_model, checkpoint = load_regressor_checkpoint(
        str(args.output_dir / "best_model.pt"), device
    )
    best_test_metrics = evaluate(
        best_model, test_loader, args.target_beta, device,
        loss_config=loss_config, regressor=regressor,
        semantic_weight=args.semantic_weight
    )
    best_latent_metrics = compute_latent_metrics(best_model, test_loader, device)

    print(f"\nBest Model Test Results (from epoch {checkpoint['epoch']}):")
    print(f"  Total Loss: {best_test_metrics['loss']:.4f}")
    print(f"  Recon Loss: {best_test_metrics['recon_loss']:.4f}")
    print(f"  Node MSE:   {best_test_metrics['node_mse']:.6f}")
    print(f"  Edge MSE:   {best_test_metrics['edge_mse']:.6f}")
    print(f"  Active dims: {best_latent_metrics['active_dims']}/{config.latent_dim}")

    # Save final results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": asdict(config),
        "beta_config": asdict(beta_config),
        "best_epoch": checkpoint["epoch"],
        "test_metrics": best_test_metrics,
        "latent_metrics": best_latent_metrics,
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
