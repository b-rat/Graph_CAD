#!/usr/bin/env python3
"""
Train the parameter regressor on L-bracket graphs.

This script trains a GNN to predict the 8 L-bracket parameters from
face-adjacency graphs, validating that the graph representation contains
sufficient information to reconstruct the original CAD model.

Usage:
    python scripts/train_regressor.py --epochs 100 --batch-size 32
    python scripts/train_regressor.py --train-size 10000 --hidden-dim 128
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from graph_cad.data import LBracketRanges, create_data_loaders
from graph_cad.models import (
    PARAMETER_NAMES,
    ParameterRegressor,
    ParameterRegressorConfig,
    denormalize_parameters,
)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred, batch.y.view(-1, 8))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(pred, batch.y.view(-1, 8))

        total_loss += loss.item()
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.view(-1, 8).cpu())
        num_batches += 1

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    avg_loss = total_loss / num_batches

    # Per-parameter MAE (in normalized space)
    per_param_mae = (all_preds - all_targets).abs().mean(dim=0)

    # Denormalize to get errors in mm
    preds_mm = denormalize_parameters(all_preds)
    targets_mm = denormalize_parameters(all_targets)
    per_param_mae_mm = (preds_mm - targets_mm).abs().mean(dim=0)

    # Overall metrics
    overall_mae_mm = per_param_mae_mm.mean().item()
    max_param_error_mm = per_param_mae_mm.max().item()

    # Percentage error relative to parameter range
    param_ranges = torch.tensor([
        150.0, 150.0, 40.0, 9.0,  # leg1, leg2, width, thickness ranges
        176.0, 8.0, 176.0, 8.0,  # hole distances and diameters ranges
    ])
    pct_errors = (per_param_mae_mm / param_ranges * 100).tolist()

    return {
        "loss": avg_loss,
        "mae_mm": overall_mae_mm,
        "max_error_mm": max_param_error_mm,
        "per_param_mae_mm": {
            name: per_param_mae_mm[i].item()
            for i, name in enumerate(PARAMETER_NAMES)
        },
        "per_param_pct_error": {
            name: pct_errors[i] for i, name in enumerate(PARAMETER_NAMES)
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train parameter regressor on L-bracket graphs"
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

    # Model arguments
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of GAT layers"
    )
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/regressor"),
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
    config = ParameterRegressorConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = ParameterRegressor(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {num_params:,} trainable parameters")
    print(f"Config: {asdict(config)}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 80)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Log progress
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae_mm']:.2f}mm | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best_model.pt",
            )

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                },
                args.output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    print("-" * 80)
    print("\nTraining complete!")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Overall MAE: {test_metrics['mae_mm']:.2f}mm")
    print(f"  Max Parameter Error: {test_metrics['max_error_mm']:.2f}mm")
    print(f"\nPer-parameter MAE (mm):")
    for name, mae in test_metrics["per_param_mae_mm"].items():
        pct = test_metrics["per_param_pct_error"][name]
        print(f"  {name:20s}: {mae:6.2f}mm ({pct:5.2f}%)")

    # Load best model and evaluate
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(args.output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nBest Model Test Results (from epoch {checkpoint['epoch']}):")
    print(f"  Loss: {best_test_metrics['loss']:.4f}")
    print(f"  Overall MAE: {best_test_metrics['mae_mm']:.2f}mm")
    print(f"  Max Parameter Error: {best_test_metrics['max_error_mm']:.2f}mm")

    # Save final results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": asdict(config),
        "best_epoch": checkpoint["epoch"],
        "test_metrics": best_test_metrics,
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
