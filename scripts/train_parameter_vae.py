#!/usr/bin/env python3
"""
Train Parameter VAE: Graph encoder with parameter decoder.

Instead of reconstructing graph features, this VAE decodes directly to
L-bracket parameters. This forces the latent space to encode parameters
explicitly, making edit directions meaningful.

Usage:
    python scripts/train_parameter_vae.py \
        --train-size 5000 --val-size 500 --test-size 500 \
        --epochs 100 --latent-dim 32 \
        --output-dir outputs/parameter_vae
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data.dataset import VariableLBracketDataset
from graph_cad.models.parameter_vae import ParameterVAE, ParameterVAEConfig


def compute_param_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    exist_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute parameter prediction loss.

    Param losses are masked by existence (only compute loss for real holes/fillet).
    """
    # Core params - always present, MSE
    core_loss = F.mse_loss(outputs["core_params"], targets["core_params"])

    # Fillet existence - BCE
    fillet_exist_loss = F.binary_cross_entropy(
        outputs["fillet_exists"], targets["fillet_exists"]
    )
    # Fillet radius - MSE only where fillet exists
    fillet_mask = targets["fillet_exists"]  # (batch, 1)
    if fillet_mask.sum() > 0:
        fillet_radius_loss = (
            F.mse_loss(outputs["fillet_radius"], targets["fillet_radius"], reduction='none')
            * fillet_mask
        ).sum() / fillet_mask.sum().clamp(min=1)
    else:
        fillet_radius_loss = torch.tensor(0.0, device=core_loss.device)

    # Hole1 existence - BCE
    hole1_exist_loss = F.binary_cross_entropy(
        outputs["hole1_exists"], targets["hole1_exists"]
    )
    # Hole1 params - MSE only where holes exist
    hole1_mask = targets["hole1_exists"].unsqueeze(-1)  # (batch, 2, 1)
    if hole1_mask.sum() > 0:
        hole1_params_loss = (
            F.mse_loss(outputs["hole1_params"], targets["hole1_params"], reduction='none')
            * hole1_mask
        ).sum() / (hole1_mask.sum() * 2).clamp(min=1)  # *2 for 2 params per hole
    else:
        hole1_params_loss = torch.tensor(0.0, device=core_loss.device)

    # Hole2 existence - BCE
    hole2_exist_loss = F.binary_cross_entropy(
        outputs["hole2_exists"], targets["hole2_exists"]
    )
    # Hole2 params - MSE only where holes exist
    hole2_mask = targets["hole2_exists"].unsqueeze(-1)
    if hole2_mask.sum() > 0:
        hole2_params_loss = (
            F.mse_loss(outputs["hole2_params"], targets["hole2_params"], reduction='none')
            * hole2_mask
        ).sum() / (hole2_mask.sum() * 2).clamp(min=1)
    else:
        hole2_params_loss = torch.tensor(0.0, device=core_loss.device)

    # Combined losses
    param_loss = core_loss + fillet_radius_loss + hole1_params_loss + hole2_params_loss
    exist_loss = fillet_exist_loss + hole1_exist_loss + hole2_exist_loss
    total_param_loss = param_loss + exist_weight * exist_loss

    metrics = {
        "param_loss": total_param_loss.item(),
        "core_loss": core_loss.item(),
        "fillet_radius_loss": fillet_radius_loss.item() if isinstance(fillet_radius_loss, torch.Tensor) else fillet_radius_loss,
        "fillet_exist_loss": fillet_exist_loss.item(),
        "hole1_params_loss": hole1_params_loss.item() if isinstance(hole1_params_loss, torch.Tensor) else hole1_params_loss,
        "hole1_exist_loss": hole1_exist_loss.item(),
        "hole2_params_loss": hole2_params_loss.item() if isinstance(hole2_params_loss, torch.Tensor) else hole2_params_loss,
        "hole2_exist_loss": hole2_exist_loss.item(),
    }

    return total_param_loss, metrics


def compute_kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 2.0,
) -> torch.Tensor:
    """
    Compute KL divergence with free-bits to prevent posterior collapse.

    Args:
        mu: Latent mean, shape (batch, latent_dim)
        logvar: Latent log variance, shape (batch, latent_dim)
        free_bits: Minimum KL per dimension before loss applies

    Returns:
        KL loss (scalar)
    """
    # KL per dimension: 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)

    # Free-bits: only penalize KL above threshold per dimension
    kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0)

    # Sum over latent dims, mean over batch
    return kl_per_dim.sum(dim=1).mean()


def extract_targets(batch, num_graphs: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Extract parameter targets from batch."""
    return {
        "core_params": batch.y.view(num_graphs, -1).to(device),
        "fillet_radius": batch.fillet_radius.view(num_graphs, -1).to(device),
        "fillet_exists": batch.fillet_exists.view(num_graphs, -1).to(device),
        "hole1_params": batch.hole1_params.view(num_graphs, 2, 2).to(device),
        "hole1_exists": batch.hole1_exists.view(num_graphs, 2).to(device),
        "hole2_params": batch.hole2_params.view(num_graphs, 2, 2).to(device),
        "hole2_exists": batch.hole2_exists.view(num_graphs, 2).to(device),
    }


def train_epoch(
    model: ParameterVAE,
    loader: PyGDataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    beta: float = 0.01,
    free_bits: float = 2.0,
    exist_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_metrics = {}
    num_batches = 0

    for batch in loader:
        # Move to device
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)
        face_types = batch.face_types.to(device)
        node_mask = batch.node_mask.to(device)

        num_graphs = batch_idx.max().item() + 1
        targets = extract_targets(batch, num_graphs, device)

        # Forward
        optimizer.zero_grad()
        outputs = model(x, face_types, edge_index, edge_attr, batch_idx, node_mask)

        # Losses
        param_loss, param_metrics = compute_param_loss(outputs, targets, exist_weight)
        kl_loss = compute_kl_loss(outputs["mu"], outputs["logvar"], free_bits)

        total_loss = param_loss + beta * kl_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Accumulate metrics
        metrics = {
            "loss": total_loss.item(),
            "kl_loss": kl_loss.item(),
            **param_metrics,
        }
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

    return {k: v / num_batches for k, v in total_metrics.items()}


@torch.no_grad()
def evaluate(
    model: ParameterVAE,
    loader: PyGDataLoader,
    device: str,
    beta: float = 0.01,
    free_bits: float = 2.0,
    exist_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate model with existence accuracy."""
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
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)
        face_types = batch.face_types.to(device)
        node_mask = batch.node_mask.to(device)

        num_graphs = batch_idx.max().item() + 1
        targets = extract_targets(batch, num_graphs, device)

        outputs = model(x, face_types, edge_index, edge_attr, batch_idx, node_mask)

        param_loss, param_metrics = compute_param_loss(outputs, targets, exist_weight)
        kl_loss = compute_kl_loss(outputs["mu"], outputs["logvar"], free_bits)
        total_loss = param_loss + beta * kl_loss

        metrics = {
            "loss": total_loss.item(),
            "kl_loss": kl_loss.item(),
            **param_metrics,
        }
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        num_batches += 1

        # Existence accuracy
        fillet_pred = (outputs["fillet_exists"] > 0.5).float()
        fillet_correct += (fillet_pred == targets["fillet_exists"]).sum().item()
        fillet_total += targets["fillet_exists"].numel()

        hole1_pred = (outputs["hole1_exists"] > 0.5).float()
        hole1_correct += (hole1_pred == targets["hole1_exists"]).sum().item()
        hole1_total += targets["hole1_exists"].numel()

        hole2_pred = (outputs["hole2_exists"] > 0.5).float()
        hole2_correct += (hole2_pred == targets["hole2_exists"]).sum().item()
        hole2_total += targets["hole2_exists"].numel()

    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["fillet_acc"] = fillet_correct / max(fillet_total, 1)
    avg_metrics["hole1_acc"] = hole1_correct / max(hole1_total, 1)
    avg_metrics["hole2_acc"] = hole2_correct / max(hole2_total, 1)

    return avg_metrics


@torch.no_grad()
def analyze_latent_space(
    model: ParameterVAE,
    loader: PyGDataLoader,
    device: str,
) -> dict[str, float]:
    """Analyze latent space for dimension usage and parameter correlations."""
    model.eval()

    all_z = []
    all_core = []

    for batch in loader:
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        batch_idx = batch.batch.to(device)
        face_types = batch.face_types.to(device)
        node_mask = batch.node_mask.to(device)

        num_graphs = batch_idx.max().item() + 1

        mu, _ = model.encode(x, face_types, edge_index, edge_attr, batch_idx, node_mask)
        all_z.append(mu.cpu())
        all_core.append(batch.y.view(num_graphs, -1).cpu())

    z = torch.cat(all_z, dim=0).numpy()
    core = torch.cat(all_core, dim=0).numpy()

    # Dimension usage (variance > threshold)
    z_std = z.std(axis=0)
    active_dims = (z_std > 0.1).sum()

    # Parameter correlations
    correlations = {}
    param_names = ["leg1", "leg2", "width", "thickness"]
    for i, name in enumerate(param_names):
        # Find best correlated latent dim
        best_corr = 0
        for j in range(z.shape[1]):
            corr = abs(np.corrcoef(z[:, j], core[:, i])[0, 1])
            if not np.isnan(corr) and corr > abs(best_corr):
                best_corr = corr
        correlations[f"{name}_corr"] = best_corr

    return {
        "active_dims": int(active_dims),
        "total_dims": z.shape[1],
        **correlations,
    }


def save_checkpoint(model, optimizer, epoch, metrics, config, path):
    """Save model checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "node_features": config.node_features,
            "edge_features": config.edge_features,
            "num_face_types": config.num_face_types,
            "face_embed_dim": config.face_embed_dim,
            "max_nodes": config.max_nodes,
            "max_edges": config.max_edges,
            "hidden_dim": config.hidden_dim,
            "num_gat_layers": config.num_gat_layers,
            "num_heads": config.num_heads,
            "encoder_dropout": config.encoder_dropout,
            "latent_dim": config.latent_dim,
            "decoder_hidden_dim": config.decoder_hidden_dim,
            "decoder_num_layers": config.decoder_num_layers,
            "decoder_dropout": config.decoder_dropout,
            "max_holes_per_leg": config.max_holes_per_leg,
        },
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Parameter VAE")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)

    # Model architecture
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-num-layers", type=int, default=3)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.01, help="KL weight")
    parser.add_argument("--free-bits", type=float, default=2.0)
    parser.add_argument("--exist-weight", type=float, default=1.0)

    parser.add_argument("--output-dir", type=str, default="outputs/parameter_vae")
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

    # Create config
    config = ParameterVAEConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
    )

    # Create datasets
    print(f"\nCreating datasets...")
    train_dataset = VariableLBracketDataset(
        num_samples=args.train_size,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        seed=args.seed,
    )
    val_dataset = VariableLBracketDataset(
        num_samples=args.val_size,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        seed=args.seed + 1000,
    )
    test_dataset = VariableLBracketDataset(
        num_samples=args.test_size,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        seed=args.seed + 2000,
    )

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    model = ParameterVAE(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")
    print(f"  Encoder: GNN with {config.num_gat_layers} GAT layers")
    print(f"  Latent: {config.latent_dim}D")
    print(f"  Decoder: {config.decoder_num_layers}×{config.decoder_hidden_dim} MLP → params")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\nTraining for {args.epochs} epochs (beta={args.beta}, free_bits={args.free_bits})...")
    results = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            args.beta, args.free_bits, args.exist_weight
        )
        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, device,
            args.beta, args.free_bits, args.exist_weight
        )

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Core: {val_metrics['core_loss']:.4f} | "
            f"KL: {val_metrics['kl_loss']:.4f} | "
            f"Fillet Acc: {val_metrics['fillet_acc']:.1%}"
        )

        results["train"].append({"epoch": epoch, **train_metrics})
        results["val"].append({"epoch": epoch, **val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics, config, output_dir / "best_model.pt")

    # Test
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, device,
        args.beta, args.free_bits, args.exist_weight
    )
    results["test"] = test_metrics

    print("\nTest Results:")
    print(f"  Core Loss: {test_metrics['core_loss']:.4f}")
    print(f"  Fillet Acc: {test_metrics['fillet_acc']:.1%}")
    print(f"  Hole1 Acc: {test_metrics['hole1_acc']:.1%}")
    print(f"  Hole2 Acc: {test_metrics['hole2_acc']:.1%}")

    # Latent space analysis
    print("\nAnalyzing latent space...")
    latent_metrics = analyze_latent_space(model, test_loader, device)
    results["latent_analysis"] = latent_metrics

    print(f"  Active dims: {latent_metrics['active_dims']}/{latent_metrics['total_dims']}")
    print(f"  Correlations: leg1={latent_metrics['leg1_corr']:.3f}, leg2={latent_metrics['leg2_corr']:.3f}, "
          f"width={latent_metrics['width_corr']:.3f}, thickness={latent_metrics['thickness_corr']:.3f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
