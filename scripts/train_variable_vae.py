#!/usr/bin/env python3
"""
Train the Variable Topology Graph VAE on L-bracket graphs.

This script trains a variational autoencoder that handles variable topology
L-brackets with embedded face types, mask prediction, and face type classification.

Usage:
    python scripts/train_variable_vae.py --epochs 100 --latent-dim 32
    python scripts/train_variable_vae.py --train-size 10000 --batch-size 16
"""

from __future__ import annotations

# Set MPS memory limit before importing torch
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

from graph_cad.data.dataset import create_variable_data_loaders
from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig
from graph_cad.models.losses import (
    VariableVAELossConfig,
    variable_vae_loss,
    variable_vae_loss_with_aux,
)


def train_epoch(
    model: VariableGraphVAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: str,
    loss_config: VariableVAELossConfig,
    free_bits: float = 2.0,
    aux_weight: float = 0.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in train_loader:
        # Move batch to device
        batch = batch.to(device)

        # Get batch dimensions
        batch_size = batch.num_graphs
        max_nodes = batch.x.shape[0] // batch_size
        max_edges = batch.edge_attr.shape[0] // batch_size

        # Reshape tensors from PyG flat format to batched format
        node_features = batch.x.view(batch_size, max_nodes, -1)
        face_types = batch.face_types.view(batch_size, max_nodes)
        edge_features = batch.edge_attr.view(batch_size, max_edges, -1)
        node_mask = batch.node_mask.view(batch_size, max_nodes)
        edge_mask = batch.edge_mask.view(batch_size, max_edges)

        # Forward pass - use PyG format for encoder
        outputs = model(
            batch.x, batch.face_types, batch.edge_index, batch.edge_attr,
            batch=batch.batch, node_mask=batch.node_mask
        )

        # Prepare targets for loss
        targets = {
            "node_features": node_features,
            "edge_features": edge_features,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "face_types": face_types,
        }

        # Compute loss
        if aux_weight > 0 and "param_pred" in outputs:
            params = batch.y.view(batch_size, -1)
            loss, metrics = variable_vae_loss_with_aux(
                outputs, targets, params, beta, aux_weight, loss_config, free_bits
            )
        else:
            loss, metrics = variable_vae_loss(
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
    model: VariableGraphVAE,
    val_loader,
    beta: float,
    device: str,
    loss_config: VariableVAELossConfig,
    free_bits: float = 2.0,
    aux_weight: float = 0.0,
) -> dict[str, float]:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in val_loader:
        batch = batch.to(device)

        batch_size = batch.num_graphs
        max_nodes = batch.x.shape[0] // batch_size
        max_edges = batch.edge_attr.shape[0] // batch_size

        node_features = batch.x.view(batch_size, max_nodes, -1)
        face_types = batch.face_types.view(batch_size, max_nodes)
        edge_features = batch.edge_attr.view(batch_size, max_edges, -1)
        node_mask = batch.node_mask.view(batch_size, max_nodes)
        edge_mask = batch.edge_mask.view(batch_size, max_edges)

        outputs = model(
            batch.x, batch.face_types, batch.edge_index, batch.edge_attr,
            batch=batch.batch, node_mask=batch.node_mask
        )

        targets = {
            "node_features": node_features,
            "edge_features": edge_features,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "face_types": face_types,
        }

        if aux_weight > 0 and "param_pred" in outputs:
            params = batch.y.view(batch_size, -1)
            loss, metrics = variable_vae_loss_with_aux(
                outputs, targets, params, beta, aux_weight, loss_config, free_bits
            )
        else:
            loss, metrics = variable_vae_loss(
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
    model: VariableGraphVAE,
    data_loader,
    device: str,
) -> dict[str, float]:
    """Compute latent space quality metrics."""
    model.eval()

    all_mu = []
    all_std = []

    for batch in data_loader:
        batch = batch.to(device)

        mu, logvar = model.encode(
            batch.x, batch.face_types, batch.edge_index, batch.edge_attr,
            batch=batch.batch, node_mask=batch.node_mask
        )

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
    model: VariableGraphVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": asdict(model.config),
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Variable Topology Graph VAE on L-bracket graphs"
    )

    # Data arguments
    parser.add_argument("--train-size", type=int, default=5000, help="Training set size")
    parser.add_argument("--val-size", type=int, default=500, help="Validation set size")
    parser.add_argument("--test-size", type=int, default=500, help="Test set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Topology arguments
    parser.add_argument("--max-nodes", type=int, default=20, help="Maximum nodes for padding")
    parser.add_argument("--max-edges", type=int, default=50, help="Maximum edges for padding")

    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent space dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-gat-layers", type=int, default=3, help="Number of GAT layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--face-embed-dim", type=int, default=8, help="Face type embedding dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Loss arguments
    parser.add_argument("--target-beta", type=float, default=0.01, help="KL weight")
    parser.add_argument("--free-bits", type=float, default=2.0, help="Free bits per dim")
    parser.add_argument("--mask-weight", type=float, default=1.0, help="Mask prediction weight")
    parser.add_argument("--face-type-weight", type=float, default=0.5, help="Face type classification weight")
    parser.add_argument("--aux-weight", type=float, default=0.0, help="Auxiliary parameter loss weight")

    # Output arguments
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vae_variable"), help="Output directory")
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
    print(f"\nCreating variable topology datasets...")
    print(f"  Train: {args.train_size}, Val: {args.val_size}, Test: {args.test_size}")
    print(f"  Max nodes: {args.max_nodes}, Max edges: {args.max_edges}")

    start_time = time.time()
    train_loader, val_loader, test_loader = create_variable_data_loaders(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        seed=args.seed,
    )
    print(f"Dataset creation took {time.time() - start_time:.1f}s")

    # Create model
    config = VariableGraphVAEConfig(
        node_features=9,
        edge_features=2,
        num_face_types=8,
        face_embed_dim=args.face_embed_dim,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
        encoder_dropout=args.dropout,
        decoder_dropout=args.dropout,
        use_param_head=args.aux_weight > 0,
        num_params=4,  # leg1, leg2, width, thickness
    )
    model = VariableGraphVAE(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {num_params:,} trainable parameters")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Face embed dim: {config.face_embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")

    # Loss config
    loss_config = VariableVAELossConfig(
        node_mask_weight=args.mask_weight,
        edge_mask_weight=args.mask_weight,
        face_type_weight=args.face_type_weight,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Beta: {args.target_beta}, Free bits: {args.free_bits}")
    print("-" * 120)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.target_beta, device,
            loss_config, args.free_bits, args.aux_weight
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, args.target_beta, device,
            loss_config, args.free_bits, args.aux_weight
        )

        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Log
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Recon: {val_metrics['recon_loss']:.4f} | "
            f"KL: {val_metrics['kl_loss']:.4f} | "
            f"Mask: {val_metrics['mask_loss']:.4f} | "
            f"FaceType: {val_metrics['face_type_loss']:.4f} | "
            f"NodeAcc: {val_metrics['node_mask_acc']:.1%} | "
            f"FTAcc: {val_metrics['face_type_acc']:.1%} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, val_metrics,
                          str(args.output_dir / "best_model.pt"))

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics,
                          str(args.output_dir / f"checkpoint_epoch_{epoch}.pt"))

    print("-" * 120)
    print("\nTraining complete!")

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, args.target_beta, device,
        loss_config, args.free_bits, args.aux_weight
    )

    print(f"\nTest Results:")
    print(f"  Total Loss:      {test_metrics['loss']:.4f}")
    print(f"  Recon Loss:      {test_metrics['recon_loss']:.4f}")
    print(f"  KL Loss:         {test_metrics['kl_loss']:.4f}")
    print(f"  Mask Loss:       {test_metrics['mask_loss']:.4f}")
    print(f"  Face Type Loss:  {test_metrics['face_type_loss']:.4f}")
    print(f"  Node Mask Acc:   {test_metrics['node_mask_acc']:.1%}")
    print(f"  Edge Mask Acc:   {test_metrics['edge_mask_acc']:.1%}")
    print(f"  Face Type Acc:   {test_metrics['face_type_acc']:.1%}")

    # Latent metrics
    print("\nComputing latent space metrics...")
    latent_metrics = compute_latent_metrics(model, test_loader, device)
    print(f"  Mean ||z||:      {latent_metrics['mean_norm']:.3f}")
    print(f"  Mean std(z):     {latent_metrics['mean_std']:.3f}")
    print(f"  Active dims:     {latent_metrics['active_dims']}/{config.latent_dim}")
    print(f"  KL from prior:   {latent_metrics['kl_from_prior']:.2f}")

    # Save results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "config": asdict(config),
        "test_metrics": test_metrics,
        "latent_metrics": latent_metrics,
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
