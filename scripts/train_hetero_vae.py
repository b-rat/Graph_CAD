#!/usr/bin/env python3
"""
Train the HeteroGNN VAE (Phase 4) on multi-geometry B-Rep graphs.

This script trains the heterogeneous graph VAE that uses V/E/F message
passing for richer geometric encoding. It supports 6 geometry types:
Bracket, Tube, Channel, Block, Cylinder, BlockHole.

Usage:
    python scripts/train_hetero_vae.py --epochs 100
    python scripts/train_hetero_vae.py --samples-per-type 5000 --batch-size 16
"""

from __future__ import annotations

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from graph_cad.data.multi_geometry_dataset import (
    MultiGeometryDataset,
    create_multi_geometry_loaders,
)
from graph_cad.data.brep_types import NUM_GEOMETRY_TYPES, GEOMETRY_TYPE_NAMES
from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig
from graph_cad.models.losses import (
    MultiGeometryLossConfig,
    multi_geometry_vae_loss,
    multi_geometry_vae_loss_with_direct_latent,
)


def prepare_batch_targets(batch, device: str) -> dict[str, torch.Tensor]:
    """
    Prepare target tensors from HeteroData batch for loss computation.

    The HeteroVAE decoder outputs face-level predictions, so we need to
    prepare face features and adjacency information.
    """
    # Face features and types
    face_features = batch['face'].x
    face_types = batch['face'].face_type

    # Build adjacency matrix from edge_to_face topology
    # For now, use face-face adjacency through shared edges
    num_faces = batch.num_faces.sum().item()
    batch_size = len(batch.num_faces)

    # Simplified: each face is connected to faces that share an edge
    # This requires more complex topology handling - for now use placeholder
    max_faces = 20  # From decoder config
    adj_matrix = torch.zeros(batch_size, max_faces, max_faces, device=device)
    node_mask = torch.zeros(batch_size, max_faces, device=device)

    # Fill in real faces
    offset = 0
    for b in range(batch_size):
        n_faces = batch.num_faces[b].item()
        node_mask[b, :n_faces] = 1.0
        offset += n_faces

    # Pad face features
    face_features_padded = torch.zeros(batch_size, max_faces, face_features.shape[-1], device=device)
    face_types_padded = torch.zeros(batch_size, max_faces, dtype=torch.long, device=device)

    offset = 0
    for b in range(batch_size):
        n_faces = batch.num_faces[b].item()
        face_features_padded[b, :n_faces] = face_features[offset:offset+n_faces]
        face_types_padded[b, :n_faces] = face_types[offset:offset+n_faces]
        offset += n_faces

    # Get parameter targets
    params_normalized = batch.params_normalized
    params_mask = batch.params_mask
    geometry_type = batch.geometry_type.squeeze(-1)

    return {
        'node_features': face_features_padded,
        'face_types': face_types_padded,
        'node_mask': node_mask,
        'adj_matrix': adj_matrix,
        'geometry_type': geometry_type,
        'params_normalized': params_normalized,
        'params_mask': params_mask,
    }


def get_beta_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    start_beta: float = 0.0,
    target_beta: float = 0.1,
) -> float:
    """Linear beta annealing schedule for KL divergence."""
    if epoch <= warmup_epochs:
        progress = epoch / warmup_epochs
        return start_beta + (target_beta - start_beta) * progress
    return target_beta


def train_epoch(
    model: HeteroVAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: str,
    loss_config: MultiGeometryLossConfig,
    free_bits: float = 0.5,
    use_direct_latent: bool = False,
    aux_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch)

        # Prepare targets
        targets = prepare_batch_targets(batch, device)

        # Compute loss
        if use_direct_latent:
            loss, metrics = multi_geometry_vae_loss_with_direct_latent(
                outputs, targets, beta, aux_weight, loss_config, free_bits
            )
        else:
            loss, metrics = multi_geometry_vae_loss(
                outputs, targets, beta, loss_config, free_bits
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


@torch.no_grad()
def evaluate(
    model: HeteroVAE,
    val_loader,
    beta: float,
    device: str,
    loss_config: MultiGeometryLossConfig,
    free_bits: float = 0.5,
    use_direct_latent: bool = False,
    aux_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in val_loader:
        batch = batch.to(device)

        outputs = model(batch)
        targets = prepare_batch_targets(batch, device)

        if use_direct_latent:
            loss, metrics = multi_geometry_vae_loss_with_direct_latent(
                outputs, targets, beta, aux_weight, loss_config, free_bits
            )
        else:
            loss, metrics = multi_geometry_vae_loss(
                outputs, targets, beta, loss_config, free_bits
            )

        total_loss += loss.item()
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


@torch.no_grad()
def compute_latent_metrics(
    model: HeteroVAE,
    data_loader,
    device: str,
) -> dict[str, float]:
    """Compute latent space quality metrics."""
    model.eval()

    all_mu = []
    all_std = []

    for batch in data_loader:
        batch = batch.to(device)

        mu, logvar = model.encode(batch)
        std = torch.exp(0.5 * logvar)
        all_mu.append(mu.cpu())
        all_std.append(std.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_std = torch.cat(all_std, dim=0)

    # Compute metrics
    mean_norm = all_mu.norm(dim=-1).mean().item()
    mean_std = all_std.mean().item()

    # Active dimensions: variance > 0.01
    mu_var = all_mu.var(dim=0)
    active_dims = (mu_var > 0.01).sum().item()
    latent_dim = all_mu.shape[-1]

    # KL from prior
    kl = -0.5 * (1 + 2 * all_std.log() - all_mu.pow(2) - all_std.pow(2))
    kl_from_prior = kl.sum(dim=-1).mean().item()

    return {
        "mean_norm": mean_norm,
        "mean_std": mean_std,
        "active_dims": int(active_dims),
        "active_dims_ratio": active_dims / latent_dim,
        "kl_from_prior": kl_from_prior,
    }


def save_checkpoint(
    model: HeteroVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    path: str,
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train HeteroGNN VAE (Phase 4) on multi-geometry B-Rep graphs"
    )

    # Data arguments
    parser.add_argument("--samples-per-type", type=int, default=5000,
                        help="Training samples per geometry type")
    parser.add_argument("--val-samples-per-type", type=int, default=500,
                        help="Validation samples per geometry type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent space dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of HeteroConv layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Loss arguments
    parser.add_argument("--target-beta", type=float, default=0.1, help="Target KL weight")
    parser.add_argument("--free-bits", type=float, default=0.5, help="Free bits per dim")
    parser.add_argument("--beta-warmup-epochs", type=int, default=None,
                        help="Epochs for beta warmup (default: 30%% of epochs)")

    # Direct latent supervision
    parser.add_argument("--use-direct-latent", action="store_true",
                        help="Use direct latent supervision for parameters")
    parser.add_argument("--aux-weight", type=float, default=1.0,
                        help="Weight for direct latent/aux parameter loss")

    # Output arguments
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hetero_vae"),
                        help="Output directory")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    # Device
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    # Set seeds
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

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print(f"\nCreating multi-geometry datasets...")
    print(f"  Samples per type (train): {args.samples_per_type}")
    print(f"  Samples per type (val): {args.val_samples_per_type}")
    print(f"  Geometry types: {list(GEOMETRY_TYPE_NAMES.values())}")

    start_time = time.time()
    train_loader, val_loader, test_loader = create_multi_geometry_loaders(
        train_samples_per_type=args.samples_per_type,
        val_samples_per_type=args.val_samples_per_type,
        test_samples_per_type=args.val_samples_per_type,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Dataset creation took {time.time() - start_time:.1f}s")

    # Compute beta warmup epochs
    beta_warmup_epochs = args.beta_warmup_epochs
    if beta_warmup_epochs is None:
        beta_warmup_epochs = int(args.epochs * 0.3)

    # Create model
    config = HeteroVAEConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model = HeteroVAE(
        config,
        use_param_head=not args.use_direct_latent,
        num_params=6,  # Max params (BlockHole)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: HeteroVAE")
    print(f"  Total params: {total_params:,}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")

    # Loss config
    loss_config = MultiGeometryLossConfig()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Target beta: {args.target_beta}, Free bits: {args.free_bits}")
    print(f"  Beta warmup: {beta_warmup_epochs} epochs")
    if args.use_direct_latent:
        print(f"  Direct latent supervision enabled (aux_weight={args.aux_weight})")
    print("-" * 100)

    best_val_loss = float("inf")
    history = {"train": [], "val": [], "latent": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        beta = get_beta_schedule(
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=beta_warmup_epochs,
            start_beta=0.0,
            target_beta=args.target_beta,
        )

        train_metrics = train_epoch(
            model, train_loader, optimizer, beta, device,
            loss_config, args.free_bits, args.use_direct_latent, args.aux_weight
        )

        val_metrics = evaluate(
            model, val_loader, beta, device,
            loss_config, args.free_bits, args.use_direct_latent, args.aux_weight
        )

        latent_metrics = compute_latent_metrics(model, val_loader, device)

        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history["latent"].append(latent_metrics)

        # Log
        epoch_time = time.time() - epoch_start
        geo_acc = val_metrics.get('geometry_type_acc', 0)
        param_mae = val_metrics.get('param_mae', 0)

        log_str = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"β={beta:.3f} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"GeoAcc: {geo_acc:.1%} | "
            f"ParamMAE: {param_mae:.4f} | "
            f"Active: {latent_metrics['active_dims']}/{args.latent_dim}"
        )
        print(log_str)

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                asdict(config),
                str(args.output_dir / "best_model.pt"),
            )

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                asdict(config),
                str(args.output_dir / f"checkpoint_epoch_{epoch}.pt"),
            )

    print("-" * 100)
    print("\nTraining complete!")

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, args.target_beta, device,
        loss_config, args.free_bits, args.use_direct_latent, args.aux_weight
    )

    print(f"\nTest Results:")
    print(f"  Total Loss:        {test_metrics['loss']:.4f}")
    print(f"  Geometry Type Acc: {test_metrics.get('geometry_type_acc', 0):.1%}")
    print(f"  Parameter MAE:     {test_metrics.get('param_mae', 0):.4f}")
    print(f"  Face Type Acc:     {test_metrics.get('face_type_acc', 0):.1%}")

    # Save results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": asdict(config),
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
