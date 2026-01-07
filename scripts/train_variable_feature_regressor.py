#!/usr/bin/env python3
"""
Train Feature Regressor for variable topology VAE.

Predicts L-bracket core parameters (leg1, leg2, width, thickness) from
VAE-decoded features.

Usage:
    python scripts/train_variable_feature_regressor.py \
        --vae-checkpoint outputs/vae_variable/best_model.pt \
        --epochs 100 \
        --output-dir outputs/feature_regressor_variable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data.dataset import (
    VariableLBracketDataset,
    VariableLBracketRanges,
)
from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig


# Core parameter names for variable topology
VARIABLE_PARAM_NAMES = ["leg1_length", "leg2_length", "width", "thickness"]


@dataclass
class VariableFeatureRegressorConfig:
    """Configuration for variable topology feature regressor."""
    # Input: flattened VAE decoder output
    max_nodes: int = 20
    max_edges: int = 50
    node_features: int = 13  # area, dir_xyz, centroid_xyz, curv, bbox_diagonal, bbox_center_xyz
    edge_features: int = 2
    # Architecture
    hidden_dims: tuple[int, ...] = (512, 256, 128, 64)
    dropout: float = 0.1
    use_batch_norm: bool = True
    # Output: 4 core parameters
    num_params: int = 4

    @property
    def input_dim(self) -> int:
        """Total input dimension from flattened features."""
        return self.max_nodes * self.node_features + self.max_edges * self.edge_features


class VariableFeatureRegressor(nn.Module):
    """
    MLP to predict L-bracket parameters from VAE-decoded features.

    Input: Flattened node features (max_nodes * 9) + edge features (max_edges * 2)
    Output: 4 normalized parameters [leg1, leg2, width, thickness]
    """

    def __init__(self, config: VariableFeatureRegressorConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_dim, config.num_params)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch, max_nodes, node_features)
            edge_features: (batch, max_edges, edge_features)

        Returns:
            params: (batch, 4) normalized parameters
        """
        batch_size = node_features.shape[0]

        # Flatten
        node_flat = node_features.view(batch_size, -1)
        edge_flat = edge_features.view(batch_size, -1)
        x = torch.cat([node_flat, edge_flat], dim=-1)

        # Forward
        h = self.backbone(x)
        return self.output_head(h)


def denormalize_variable_params(
    params: torch.Tensor,
    ranges: VariableLBracketRanges | None = None,
) -> torch.Tensor:
    """Convert normalized [0,1] params to actual mm values."""
    if ranges is None:
        ranges = VariableLBracketRanges()

    mins = torch.tensor([
        ranges.leg1_length[0],
        ranges.leg2_length[0],
        ranges.width[0],
        ranges.thickness[0],
    ], device=params.device)

    maxs = torch.tensor([
        ranges.leg1_length[1],
        ranges.leg2_length[1],
        ranges.width[1],
        ranges.thickness[1],
    ], device=params.device)

    return params * (maxs - mins) + mins


def load_variable_vae(checkpoint_path: str, device: str) -> tuple[VariableGraphVAE, dict]:
    """Load variable topology VAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = VariableGraphVAEConfig(**checkpoint["config"])
    model = VariableGraphVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def prepare_dataset(
    vae: VariableGraphVAE,
    num_samples: int,
    batch_size: int,
    seed: int,
    device: str,
    desc: str = "Preparing",
    cache_dir: Path | None = None,
    split: str = "train",
) -> TensorDataset:
    """
    Generate dataset by encoding/decoding variable topology L-brackets through VAE.
    """
    # Check cache
    if cache_dir is not None:
        cache_path = cache_dir / f"{split}_{num_samples}_{seed}.pt"
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            return TensorDataset(data["node_features"], data["edge_features"], data["params"])

    # Create dataset
    dataset = VariableLBracketDataset(
        num_samples=num_samples,
        max_nodes=vae.config.max_nodes,
        max_edges=vae.config.max_edges,
        seed=seed,
    )
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_node_features = []
    all_edge_features = []
    all_params = []

    vae.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            # Move to device
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            face_types = batch.face_types.to(device)
            node_mask = batch.node_mask.to(device)
            params = batch.y.to(device)

            # Get batch size
            num_graphs = batch_idx.max().item() + 1
            params = params.view(num_graphs, -1)

            # Encode
            mu, _ = vae.encode(
                x, face_types, edge_index, edge_attr, batch_idx, node_mask
            )

            # Decode
            outputs = vae.decode(mu)
            node_recon = outputs["node_features"]  # (batch, max_nodes, 9)
            edge_recon = outputs["edge_features"]  # (batch, max_edges, 2)

            all_node_features.append(node_recon.cpu())
            all_edge_features.append(edge_recon.cpu())
            all_params.append(params.cpu())

    node_features = torch.cat(all_node_features, dim=0)
    edge_features = torch.cat(all_edge_features, dim=0)
    params = torch.cat(all_params, dim=0)

    # Save to cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{split}_{num_samples}_{seed}.pt"
        torch.save({
            "node_features": node_features,
            "edge_features": edge_features,
            "params": params,
        }, cache_path)
        print(f"  Saved to cache: {cache_path}")

    return TensorDataset(node_features, edge_features, params)


def train_epoch(
    model: VariableFeatureRegressor,
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

        pred_params = model(node_feat, edge_feat)
        loss = F.mse_loss(pred_params, params)
        mae = (pred_params - params).abs().mean()

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
    model: VariableFeatureRegressor,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate model."""
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

        pred_params = model(node_feat, edge_feat)
        loss = F.mse_loss(pred_params, params)
        mae = (pred_params - params).abs().mean()

        total_loss += loss.item()
        total_mae += mae.item()
        num_batches += 1

        all_preds.append(pred_params.cpu())
        all_targets.append(params.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Denormalize for mm errors
    preds_mm = denormalize_variable_params(preds)
    targets_mm = denormalize_variable_params(targets)

    per_param_mae = (preds_mm - targets_mm).abs().mean(dim=0)
    per_param_metrics = {
        f"mae_{name}": per_param_mae[i].item()
        for i, name in enumerate(VARIABLE_PARAM_NAMES)
    }

    return {
        "loss": total_loss / num_batches,
        "mae": total_mae / num_batches,
        "mae_denorm": (preds_mm - targets_mm).abs().mean().item(),
        **per_param_metrics,
    }


def save_regressor(
    model: VariableFeatureRegressor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
):
    """Save model checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "max_nodes": model.config.max_nodes,
            "max_edges": model.config.max_edges,
            "node_features": model.config.node_features,
            "edge_features": model.config.edge_features,
            "hidden_dims": model.config.hidden_dims,
            "dropout": model.config.dropout,
            "use_batch_norm": model.config.use_batch_norm,
            "num_params": model.config.num_params,
        },
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Variable Feature Regressor"
    )

    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to trained variable topology VAE",
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128, 64],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/feature_regressor_variable")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print(f"\nLoading VAE from {args.vae_checkpoint}...")
    vae, _ = load_variable_vae(args.vae_checkpoint, device)
    for param in vae.parameters():
        param.requires_grad = False
    print(f"  Latent dim: {vae.config.latent_dim}")
    print(f"  Max nodes: {vae.config.max_nodes}")
    print(f"  Max edges: {vae.config.max_edges}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Generate datasets
    print(f"\nGenerating training data ({args.train_size} samples)...")
    start = time.time()
    train_dataset = prepare_dataset(
        vae, args.train_size, args.batch_size, args.seed, device,
        "Train", cache_dir, "train"
    )
    print(f"  Done in {time.time() - start:.1f}s")

    print(f"Generating validation data ({args.val_size} samples)...")
    val_dataset = prepare_dataset(
        vae, args.val_size, args.batch_size, args.seed + 1000, device,
        "Val", cache_dir, "val"
    )

    print(f"Generating test data ({args.test_size} samples)...")
    test_dataset = prepare_dataset(
        vae, args.test_size, args.batch_size, args.seed + 2000, device,
        "Test", cache_dir, "test"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    config = VariableFeatureRegressorConfig(
        max_nodes=vae.config.max_nodes,
        max_edges=vae.config.max_edges,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    model = VariableFeatureRegressor(config).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Input dim: {config.input_dim}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    results = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE: {val_metrics['mae_denorm']:.2f}mm"
        )

        results["train"].append({"epoch": epoch, **train_metrics})
        results["val"].append({"epoch": epoch, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_regressor(model, optimizer, epoch, val_metrics, str(output_dir / "best_model.pt"))

    # Test
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    results["test"] = test_metrics

    print("\nTest Results:")
    print(f"  MAE (mm): {test_metrics['mae_denorm']:.2f}")
    for name in VARIABLE_PARAM_NAMES:
        print(f"  {name}: {test_metrics[f'mae_{name}']:.2f}mm")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
