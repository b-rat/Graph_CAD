#!/usr/bin/env python3
"""
Train a full latent regressor that predicts ALL L-bracket parameters from z.

Unlike the simple 4-param latent regressor, this uses multiple heads to predict:
- 4 core params (leg1, leg2, width, thickness)
- Fillet (radius + exists)
- Up to 2 holes per leg (diameter, distance, exists) × 4 slots

Usage:
    python scripts/train_full_latent_regressor.py \
        --vae-checkpoint outputs/vae_variable/best_model.pt \
        --epochs 100 \
        --output-dir outputs/full_latent_regressor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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

from graph_cad.data.dataset import VariableLBracketDataset, VariableLBracketRanges
from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig


@dataclass
class FullLatentRegressorConfig:
    """Configuration for full parameter latent regressor."""
    latent_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    max_holes_per_leg: int = 2


class FullLatentRegressor(nn.Module):
    """
    Multi-head regressor predicting ALL L-bracket parameters from latent z.

    Outputs:
    - core_params: (batch, 4) - leg1, leg2, width, thickness
    - fillet_radius: (batch, 1) - normalized radius
    - fillet_exists: (batch, 1) - probability of fillet
    - hole1_params: (batch, 2, 2) - (diameter, distance) for each slot
    - hole1_exists: (batch, 2) - probability each slot is real
    - hole2_params: (batch, 2, 2) - (diameter, distance) for each slot
    - hole2_exists: (batch, 2) - probability each slot is real
    """

    def __init__(self, config: FullLatentRegressorConfig):
        super().__init__()
        self.config = config

        # Shared backbone
        layers = []
        in_dim = config.latent_dim
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Core params head (always present)
        self.core_head = nn.Linear(config.hidden_dim, 4)

        # Fillet head
        self.fillet_head = nn.Linear(config.hidden_dim, 1)  # radius
        self.fillet_exists_head = nn.Linear(config.hidden_dim, 1)  # exists prob

        # Hole heads (2 slots per leg)
        # Each slot predicts (diameter, distance)
        self.hole1_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, 2) for _ in range(config.max_holes_per_leg)
        ])
        self.hole1_exists_head = nn.Linear(config.hidden_dim, config.max_holes_per_leg)

        self.hole2_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, 2) for _ in range(config.max_holes_per_leg)
        ])
        self.hole2_exists_head = nn.Linear(config.hidden_dim, config.max_holes_per_leg)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            dict with all predictions
        """
        h = self.backbone(z)

        # Core params
        core_params = self.core_head(h)

        # Fillet
        fillet_radius = self.fillet_head(h)
        fillet_exists = torch.sigmoid(self.fillet_exists_head(h))

        # Holes on leg1
        hole1_params = torch.stack([head(h) for head in self.hole1_heads], dim=1)  # (batch, 2, 2)
        hole1_exists = torch.sigmoid(self.hole1_exists_head(h))  # (batch, 2)

        # Holes on leg2
        hole2_params = torch.stack([head(h) for head in self.hole2_heads], dim=1)  # (batch, 2, 2)
        hole2_exists = torch.sigmoid(self.hole2_exists_head(h))  # (batch, 2)

        return {
            "core_params": core_params,
            "fillet_radius": fillet_radius,
            "fillet_exists": fillet_exists,
            "hole1_params": hole1_params,
            "hole1_exists": hole1_exists,
            "hole2_params": hole2_params,
            "hole2_exists": hole2_exists,
        }


def load_variable_vae(checkpoint_path: str, device: str) -> tuple[VariableGraphVAE, dict]:
    """Load variable topology VAE."""
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
    desc: str = "Encoding",
    cache_dir: Path | None = None,
    split: str = "train",
) -> TensorDataset:
    """Encode brackets and extract full parameters from dataset."""

    # Check cache
    if cache_dir is not None:
        cache_path = cache_dir / f"full_{split}_{num_samples}_{seed}.pt"
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            return TensorDataset(
                data["z"],
                data["core"],
                data["fillet_radius"],
                data["fillet_exists"],
                data["hole1_params"],
                data["hole1_exists"],
                data["hole2_params"],
                data["hole2_exists"],
            )

    # Create dataset
    dataset = VariableLBracketDataset(
        num_samples=num_samples,
        max_nodes=vae.config.max_nodes,
        max_edges=vae.config.max_edges,
        seed=seed,
    )
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_z = []
    all_core = []
    all_fillet_radius = []
    all_fillet_exists = []
    all_hole1_params = []
    all_hole1_exists = []
    all_hole2_params = []
    all_hole2_exists = []

    vae.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            face_types = batch.face_types.to(device)
            node_mask = batch.node_mask.to(device)

            # Encode to latent
            mu, _ = vae.encode(x, face_types, edge_index, edge_attr, batch_idx, node_mask)
            all_z.append(mu.cpu())

            # Extract full params from batch (already computed by dataset)
            num_graphs = batch_idx.max().item() + 1
            core = batch.y.cpu().view(num_graphs, -1)
            all_core.append(core)

            # Fillet params
            fillet_radius = batch.fillet_radius.cpu().view(num_graphs, -1)
            fillet_exists = batch.fillet_exists.cpu().view(num_graphs, -1)
            all_fillet_radius.append(fillet_radius)
            all_fillet_exists.append(fillet_exists)

            # Hole params (already shaped correctly by PyG batching)
            hole1_params = batch.hole1_params.cpu().view(num_graphs, 2, 2)
            hole1_exists = batch.hole1_exists.cpu().view(num_graphs, 2)
            hole2_params = batch.hole2_params.cpu().view(num_graphs, 2, 2)
            hole2_exists = batch.hole2_exists.cpu().view(num_graphs, 2)
            all_hole1_params.append(hole1_params)
            all_hole1_exists.append(hole1_exists)
            all_hole2_params.append(hole2_params)
            all_hole2_exists.append(hole2_exists)

    z = torch.cat(all_z, dim=0)
    core = torch.cat(all_core, dim=0)
    fillet_radius = torch.cat(all_fillet_radius, dim=0)
    fillet_exists = torch.cat(all_fillet_exists, dim=0)
    hole1_params = torch.cat(all_hole1_params, dim=0)
    hole1_exists = torch.cat(all_hole1_exists, dim=0)
    hole2_params = torch.cat(all_hole2_params, dim=0)
    hole2_exists = torch.cat(all_hole2_exists, dim=0)

    # Save to cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"full_{split}_{num_samples}_{seed}.pt"
        torch.save({
            "z": z,
            "core": core,
            "fillet_radius": fillet_radius,
            "fillet_exists": fillet_exists,
            "hole1_params": hole1_params,
            "hole1_exists": hole1_exists,
            "hole2_params": hole2_params,
            "hole2_exists": hole2_exists,
        }, cache_path)
        print(f"  Saved to cache: {cache_path}")

    return TensorDataset(
        z, core, fillet_radius, fillet_exists,
        hole1_params, hole1_exists, hole2_params, hole2_exists
    )


def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: tuple[torch.Tensor, ...],
    exist_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute combined loss for all heads.

    Param losses are masked by existence (only compute loss for real holes).
    """
    core_target, fillet_radius_target, fillet_exists_target, \
        hole1_params_target, hole1_exists_target, \
        hole2_params_target, hole2_exists_target = targets

    # Core params - always present, MSE
    core_loss = F.mse_loss(outputs["core_params"], core_target)

    # Fillet existence - BCE
    fillet_exist_loss = F.binary_cross_entropy(
        outputs["fillet_exists"], fillet_exists_target
    )
    # Fillet radius - MSE only where fillet exists
    fillet_mask = fillet_exists_target.unsqueeze(-1)  # (batch, 1)
    if fillet_mask.sum() > 0:
        fillet_radius_loss = (
            F.mse_loss(outputs["fillet_radius"], fillet_radius_target, reduction='none')
            * fillet_mask
        ).sum() / fillet_mask.sum().clamp(min=1)
    else:
        fillet_radius_loss = torch.tensor(0.0, device=core_target.device)

    # Hole1 existence - BCE
    hole1_exist_loss = F.binary_cross_entropy(
        outputs["hole1_exists"], hole1_exists_target
    )
    # Hole1 params - MSE only where holes exist
    hole1_mask = hole1_exists_target.unsqueeze(-1)  # (batch, 2, 1)
    if hole1_mask.sum() > 0:
        hole1_params_loss = (
            F.mse_loss(outputs["hole1_params"], hole1_params_target, reduction='none')
            * hole1_mask
        ).sum() / hole1_mask.sum().clamp(min=1)
    else:
        hole1_params_loss = torch.tensor(0.0, device=core_target.device)

    # Hole2 existence - BCE
    hole2_exist_loss = F.binary_cross_entropy(
        outputs["hole2_exists"], hole2_exists_target
    )
    # Hole2 params - MSE only where holes exist
    hole2_mask = hole2_exists_target.unsqueeze(-1)
    if hole2_mask.sum() > 0:
        hole2_params_loss = (
            F.mse_loss(outputs["hole2_params"], hole2_params_target, reduction='none')
            * hole2_mask
        ).sum() / hole2_mask.sum().clamp(min=1)
    else:
        hole2_params_loss = torch.tensor(0.0, device=core_target.device)

    # Combined loss
    param_loss = core_loss + fillet_radius_loss + hole1_params_loss + hole2_params_loss
    exist_loss = fillet_exist_loss + hole1_exist_loss + hole2_exist_loss
    total_loss = param_loss + exist_weight * exist_loss

    metrics = {
        "loss": total_loss.item(),
        "core_loss": core_loss.item(),
        "fillet_radius_loss": fillet_radius_loss.item(),
        "fillet_exist_loss": fillet_exist_loss.item(),
        "hole1_params_loss": hole1_params_loss.item(),
        "hole1_exist_loss": hole1_exist_loss.item(),
        "hole2_params_loss": hole2_params_loss.item(),
        "hole2_exist_loss": hole2_exist_loss.item(),
    }

    return total_loss, metrics


def train_epoch(
    model: FullLatentRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    exist_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_metrics = {}
    num_batches = 0

    for batch in loader:
        z = batch[0].to(device)
        targets = tuple(t.to(device) for t in batch[1:])

        optimizer.zero_grad()
        outputs = model(z)
        loss, metrics = compute_loss(outputs, targets, exist_weight)
        loss.backward()
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: FullLatentRegressor,
    loader: DataLoader,
    device: str,
    exist_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate model with existence accuracy metrics."""
    model.eval()
    total_metrics = {}
    num_batches = 0

    # Accuracy tracking
    fillet_correct = 0
    fillet_total = 0
    hole1_correct = 0
    hole1_total = 0
    hole2_correct = 0
    hole2_total = 0

    for batch in loader:
        z = batch[0].to(device)
        targets = tuple(t.to(device) for t in batch[1:])

        outputs = model(z)
        _, metrics = compute_loss(outputs, targets, exist_weight)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

        # Existence accuracy
        fillet_pred = (outputs["fillet_exists"] > 0.5).float()
        fillet_correct += (fillet_pred == targets[2]).sum().item()
        fillet_total += targets[2].numel()

        hole1_pred = (outputs["hole1_exists"] > 0.5).float()
        hole1_correct += (hole1_pred == targets[4]).sum().item()
        hole1_total += targets[4].numel()

        hole2_pred = (outputs["hole2_exists"] > 0.5).float()
        hole2_correct += (hole2_pred == targets[6]).sum().item()
        hole2_total += targets[6].numel()

    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["fillet_acc"] = fillet_correct / max(fillet_total, 1)
    avg_metrics["hole1_acc"] = hole1_correct / max(hole1_total, 1)
    avg_metrics["hole2_acc"] = hole2_correct / max(hole2_total, 1)

    return avg_metrics


def denormalize_core(params: torch.Tensor) -> torch.Tensor:
    """Denormalize core params to mm."""
    ranges = VariableLBracketRanges()
    mins = torch.tensor([
        ranges.leg1_length[0], ranges.leg2_length[0],
        ranges.width[0], ranges.thickness[0],
    ], device=params.device)
    maxs = torch.tensor([
        ranges.leg1_length[1], ranges.leg2_length[1],
        ranges.width[1], ranges.thickness[1],
    ], device=params.device)
    return params * (maxs - mins) + mins


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "latent_dim": model.config.latent_dim,
            "hidden_dim": model.config.hidden_dim,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
            "max_holes_per_leg": model.config.max_holes_per_leg,
        },
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Full Latent Regressor")
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--exist-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/full_latent_regressor")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  Latent dim: {vae.config.latent_dim}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Prepare datasets
    print(f"\nEncoding training data ({args.train_size} samples)...")
    start = time.time()
    train_dataset = prepare_dataset(
        vae, args.train_size, args.batch_size, args.seed, device,
        "Train", cache_dir, "train"
    )
    print(f"  Done in {time.time() - start:.1f}s")

    print(f"Encoding validation data ({args.val_size} samples)...")
    val_dataset = prepare_dataset(
        vae, args.val_size, args.batch_size, args.seed + 1000, device,
        "Val", cache_dir, "val"
    )

    print(f"Encoding test data ({args.test_size} samples)...")
    test_dataset = prepare_dataset(
        vae, args.test_size, args.batch_size, args.seed + 2000, device,
        "Test", cache_dir, "test"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    config = FullLatentRegressorConfig(
        latent_dim=vae.config.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = FullLatentRegressor(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")
    print(f"Architecture: {config.latent_dim}D → {config.num_layers}×{config.hidden_dim} → multi-head")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    results = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.exist_weight)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, args.exist_weight)

        print(
            f"Epoch {epoch:3d} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Fillet Acc: {val_metrics['fillet_acc']:.1%} | "
            f"Hole1 Acc: {val_metrics['hole1_acc']:.1%} | "
            f"Hole2 Acc: {val_metrics['hole2_acc']:.1%}"
        )

        results["train"].append({"epoch": epoch, **train_metrics})
        results["val"].append({"epoch": epoch, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, str(output_dir / "best_model.pt"))

    # Test
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, args.exist_weight)
    results["test"] = test_metrics

    print("\nTest Results:")
    print(f"  Core Loss: {test_metrics['core_loss']:.4f}")
    print(f"  Fillet Accuracy: {test_metrics['fillet_acc']:.1%}")
    print(f"  Hole1 Accuracy: {test_metrics['hole1_acc']:.1%}")
    print(f"  Hole2 Accuracy: {test_metrics['hole2_acc']:.1%}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
