#!/usr/bin/env python3
"""
Train the Feature Regressor to predict L-bracket parameters from VAE-decoded features.

This creates the final piece needed for full inference:
    Latent → VAE Decoder → Features → FeatureRegressor → Parameters → LBracket → STEP

The training uses the trained VAE to encode/decode graphs, then trains an MLP
to predict parameters from the decoded features.

Usage:
    # Basic training (uses existing VAE)
    python scripts/train_feature_regressor.py \
        --vae-checkpoint outputs/vae_16d/best_model.pt \
        --epochs 50

    # Custom settings
    python scripts/train_feature_regressor.py \
        --vae-checkpoint outputs/vae_16d/best_model.pt \
        --train-size 10000 \
        --hidden-dims 512 256 128 \
        --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data import create_data_loaders
from graph_cad.models.feature_regressor import (
    FeatureRegressor,
    FeatureRegressorConfig,
    save_feature_regressor,
)
from graph_cad.models.parameter_regressor import (
    PARAMETER_NAMES,
    denormalize_parameters,
)
from graph_cad.training.vae_trainer import load_checkpoint as load_vae_checkpoint


def prepare_dataset(
    vae,
    num_samples: int,
    batch_size: int,
    seed: int,
    device: str,
    desc: str = "Preparing",
) -> TensorDataset:
    """
    Generate dataset by encoding/decoding L-bracket graphs through VAE.

    Returns TensorDataset with (node_features, edge_features, parameters).
    """
    # Create data loaders for graph generation
    train_loader, _, _ = create_data_loaders(
        train_size=num_samples,
        val_size=1,  # Minimal, we don't need val here
        test_size=1,
        batch_size=batch_size,
        seed=seed,
    )

    all_node_features = []
    all_edge_features = []
    all_params = []

    vae.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc=desc):
            # Move to device
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            params = batch.y.to(device)

            # Get number of graphs in this batch
            num_graphs = batch_idx.max().item() + 1

            # Reshape params from flat to (num_graphs, 8)
            params = params.view(num_graphs, -1)

            # Encode to latent
            mu, logvar = vae.encode(x, edge_index, edge_attr, batch_idx)

            # Decode (use mean for deterministic output)
            node_recon, edge_recon = vae.decode(mu)

            # Collect
            all_node_features.append(node_recon.cpu())
            all_edge_features.append(edge_recon.cpu())
            all_params.append(params.cpu())

    # Stack all batches
    node_features = torch.cat(all_node_features, dim=0)
    edge_features = torch.cat(all_edge_features, dim=0)
    params = torch.cat(all_params, dim=0)

    return TensorDataset(node_features, edge_features, params)


def train_epoch(
    model: FeatureRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for node_feat, edge_feat, params in loader:
        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)
        params = params.to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_params = model(node_feat, edge_feat)

        # Loss
        loss = F.mse_loss(pred_params, params)
        mae = (pred_params - params).abs().mean()

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "mae": total_mae / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: FeatureRegressor,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    for node_feat, edge_feat, params in loader:
        node_feat = node_feat.to(device)
        edge_feat = edge_feat.to(device)
        params = params.to(device)

        # Forward pass
        pred_params = model(node_feat, edge_feat)

        # Loss
        loss = F.mse_loss(pred_params, params)
        mae = (pred_params - params).abs().mean()

        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1

        all_preds.append(pred_params.cpu())
        all_targets.append(params.cpu())

    # Per-parameter metrics
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Denormalize for meaningful error values
    preds_denorm = denormalize_parameters(preds)
    targets_denorm = denormalize_parameters(targets)

    per_param_mae = (preds_denorm - targets_denorm).abs().mean(dim=0)
    per_param_metrics = {
        f"mae_{name}": per_param_mae[i].item()
        for i, name in enumerate(PARAMETER_NAMES)
    }

    return {
        "loss": total_loss / num_batches,
        "mae": total_mae / num_batches,
        "mae_denorm": (preds_denorm - targets_denorm).abs().mean().item(),
        **per_param_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Feature Regressor for L-bracket parameter prediction"
    )

    # Required
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint",
    )

    # Data
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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    # Model
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Disable batch normalization",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_regressor",
        help="Output directory",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print(f"\nLoading VAE from {args.vae_checkpoint}...")
    vae, vae_checkpoint = load_vae_checkpoint(args.vae_checkpoint, device=device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"  VAE latent dim: {vae.config.latent_dim}")

    # Generate datasets
    print(f"\nGenerating training data ({args.train_size} samples)...")
    start_time = time.time()
    train_dataset = prepare_dataset(
        vae, args.train_size, args.batch_size, args.seed, device, "Train data"
    )
    print(f"  Generated in {time.time() - start_time:.1f}s")

    print(f"Generating validation data ({args.val_size} samples)...")
    val_dataset = prepare_dataset(
        vae, args.val_size, args.batch_size, args.seed + 1000, device, "Val data"
    )

    print(f"Generating test data ({args.test_size} samples)...")
    test_dataset = prepare_dataset(
        vae, args.test_size, args.batch_size, args.seed + 2000, device, "Test data"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create model
    config = FeatureRegressorConfig(
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        use_batch_norm=not args.no_batch_norm,
    )
    model = FeatureRegressor(config).to(device)

    print(f"\nModel config: {config}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    results = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Log
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE (mm): {val_metrics['mae_denorm']:.2f}"
        )

        # Track results
        results["train"].append({"epoch": epoch, **train_metrics})
        results["val"].append({"epoch": epoch, **val_metrics})

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_feature_regressor(
                model, optimizer, epoch, val_metrics, str(output_dir / "best_model.pt")
            )

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    results["test"] = test_metrics

    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  MAE (normalized): {test_metrics['mae']:.6f}")
    print(f"  MAE (mm): {test_metrics['mae_denorm']:.2f}")
    print("\nPer-parameter MAE (mm):")
    for name in PARAMETER_NAMES:
        print(f"  {name}: {test_metrics[f'mae_{name}']:.2f}")

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Best model saved to {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
