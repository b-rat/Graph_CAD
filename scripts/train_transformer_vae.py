#!/usr/bin/env python3
"""
Train the Transformer Graph VAE (Phase 3) on L-bracket graphs.

This script trains the DETR-style transformer decoder VAE that uses Hungarian
matching for permutation-invariant training. This is designed to break the
64% accuracy ceiling from the Phase 2 MLP decoder.

Usage:
    python scripts/train_transformer_vae.py --epochs 100
    python scripts/train_transformer_vae.py --train-size 10000 --batch-size 16
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

from graph_cad.data.dataset import create_variable_data_loaders
from graph_cad.models.graph_vae import VariableGraphVAEConfig, VariableGraphVAEEncoder
from graph_cad.models.transformer_decoder import (
    TransformerDecoderConfig,
    TransformerGraphDecoder,
    TransformerGraphVAE,
)
from graph_cad.models.losses import (
    HungarianLossConfig,
    transformer_vae_loss,
    transformer_vae_loss_with_aux,
    kl_divergence,
)


def edge_index_to_adj_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert edge_index (COO format) to adjacency matrix.

    Args:
        edge_index: Edge indices, shape (2, num_edges)
        num_nodes: Number of nodes in the graph
        edge_mask: Optional mask for real edges, shape (num_edges,)

    Returns:
        Adjacency matrix, shape (num_nodes, num_nodes)
    """
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)

    if edge_mask is not None:
        # Only add edges where mask is 1
        mask_bool = edge_mask.bool()
        src = edge_index[0][mask_bool]
        dst = edge_index[1][mask_bool]
    else:
        src = edge_index[0]
        dst = edge_index[1]

    # Set adjacency (undirected graph)
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0

    return adj


def batch_edge_index_to_adj_matrices(
    edge_index: torch.Tensor,
    batch_size: int,
    max_nodes: int,
    max_edges: int,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert batched edge_index to batch of adjacency matrices.

    Args:
        edge_index: Batched edge indices from PyG, shape (2, batch_size * max_edges)
        batch_size: Number of graphs in batch
        max_nodes: Maximum nodes per graph
        max_edges: Maximum edges per graph
        edge_mask: Edge mask, shape (batch_size * max_edges,)

    Returns:
        Adjacency matrices, shape (batch_size, max_nodes, max_nodes)
    """
    device = edge_index.device
    adj_matrices = torch.zeros(batch_size, max_nodes, max_nodes, device=device)

    # Reshape edge_mask to (batch_size, max_edges)
    edge_mask_batched = edge_mask.view(batch_size, max_edges)

    for b in range(batch_size):
        # Get edges for this graph
        start_idx = b * max_edges
        end_idx = (b + 1) * max_edges

        edges_b = edge_index[:, start_idx:end_idx]  # (2, max_edges)
        mask_b = edge_mask_batched[b]  # (max_edges,)

        # Adjust indices (PyG adds offsets for batching)
        node_offset = b * max_nodes
        edges_b = edges_b - node_offset

        # Add edges where mask is 1
        for i in range(max_edges):
            if mask_b[i] > 0.5:
                src, dst = int(edges_b[0, i]), int(edges_b[1, i])
                if 0 <= src < max_nodes and 0 <= dst < max_nodes:
                    adj_matrices[b, src, dst] = 1.0
                    adj_matrices[b, dst, src] = 1.0

    return adj_matrices


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
    model: TransformerGraphVAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: str,
    loss_config: HungarianLossConfig,
    max_nodes: int,
    max_edges: int,
    free_bits: float = 0.5,
    aux_weight: float = 0.0,
    num_params: int = 4,
    aux_loss_type: str = "correlation",
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)

        batch_size = batch.num_graphs

        # Reshape tensors from PyG flat format to batched format
        node_features = batch.x.view(batch_size, max_nodes, -1)
        face_types = batch.face_types.view(batch_size, max_nodes)
        node_mask = batch.node_mask.view(batch_size, max_nodes)
        edge_mask = batch.edge_mask.view(batch_size * max_edges)

        # Build adjacency matrices from edge_index
        adj_matrices = batch_edge_index_to_adj_matrices(
            batch.edge_index, batch_size, max_nodes, max_edges, edge_mask
        )

        # Forward pass
        outputs = model(
            batch.x, batch.face_types, batch.edge_index, batch.edge_attr,
            batch=batch.batch, node_mask=batch.node_mask
        )

        # Prepare targets
        targets = {
            "node_features": node_features,
            "face_types": face_types,
            "node_mask": node_mask,
            "adj_matrix": adj_matrices,
        }

        # Compute loss with Hungarian matching
        if aux_weight > 0 and hasattr(batch, 'y'):
            # Get parameter targets (may be flattened by PyG DataLoader)
            y = batch.y
            if y.dim() == 1:
                y = y.view(batch_size, num_params)
            loss, metrics = transformer_vae_loss_with_aux(
                outputs, targets, y, beta, aux_weight, loss_config, free_bits,
                aux_loss_type
            )
        else:
            loss, metrics = transformer_vae_loss(
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
    model: TransformerGraphVAE,
    val_loader,
    beta: float,
    device: str,
    loss_config: HungarianLossConfig,
    max_nodes: int,
    max_edges: int,
    free_bits: float = 0.5,
    aux_weight: float = 0.0,
    num_params: int = 4,
    aux_loss_type: str = "correlation",
) -> dict[str, float]:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in val_loader:
        batch = batch.to(device)

        batch_size = batch.num_graphs

        node_features = batch.x.view(batch_size, max_nodes, -1)
        face_types = batch.face_types.view(batch_size, max_nodes)
        node_mask = batch.node_mask.view(batch_size, max_nodes)
        edge_mask = batch.edge_mask.view(batch_size * max_edges)

        adj_matrices = batch_edge_index_to_adj_matrices(
            batch.edge_index, batch_size, max_nodes, max_edges, edge_mask
        )

        outputs = model(
            batch.x, batch.face_types, batch.edge_index, batch.edge_attr,
            batch=batch.batch, node_mask=batch.node_mask
        )

        targets = {
            "node_features": node_features,
            "face_types": face_types,
            "node_mask": node_mask,
            "adj_matrix": adj_matrices,
        }

        if aux_weight > 0 and hasattr(batch, 'y'):
            y = batch.y
            if y.dim() == 1:
                y = y.view(batch_size, num_params)
            loss, metrics = transformer_vae_loss_with_aux(
                outputs, targets, y, beta, aux_weight, loss_config, free_bits,
                aux_loss_type
            )
        else:
            loss, metrics = transformer_vae_loss(
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
    model: TransformerGraphVAE,
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
    model: TransformerGraphVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    encoder_config: dict,
    decoder_config: dict,
    path: str,
    use_param_head: bool = False,
    num_params: int = 4,
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "use_param_head": use_param_head,
        "num_params": num_params,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer Graph VAE (Phase 3) on L-bracket graphs"
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

    # Encoder arguments (GAT encoder from Phase 2)
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent space dimension")
    parser.add_argument("--encoder-hidden-dim", type=int, default=64, help="Encoder hidden dimension")
    parser.add_argument("--num-gat-layers", type=int, default=3, help="Number of GAT layers")
    parser.add_argument("--num-gat-heads", type=int, default=4, help="GAT attention heads")
    parser.add_argument("--face-embed-dim", type=int, default=8, help="Face type embedding dim")
    parser.add_argument("--encoder-dropout", type=float, default=0.1, help="Encoder dropout")
    parser.add_argument("--pooling-type", type=str, default="mean",
                        choices=["mean", "attention"],
                        help="Pooling type: mean (default) or attention (multi-head)")
    parser.add_argument("--attention-heads", type=int, default=4,
                        help="Number of attention heads for attention pooling")

    # Decoder arguments (Transformer decoder - new in Phase 3)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256, help="Decoder hidden dimension")
    parser.add_argument("--num-decoder-layers", type=int, default=4, help="Transformer decoder layers")
    parser.add_argument("--num-decoder-heads", type=int, default=8, help="Decoder attention heads")
    parser.add_argument("--decoder-dropout", type=float, default=0.1, help="Decoder dropout")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Loss arguments
    parser.add_argument("--target-beta", type=float, default=0.1, help="Target KL weight")
    parser.add_argument("--free-bits", type=float, default=0.5, help="Free bits per dim")
    parser.add_argument("--beta-warmup-epochs", type=int, default=None,
                        help="Epochs for beta warmup (default: 30%% of epochs)")

    # Hungarian loss weights
    parser.add_argument("--face-type-weight", type=float, default=2.0,
                        help="Face type loss weight")
    parser.add_argument("--edge-weight", type=float, default=1.0,
                        help="Edge loss weight")

    # Auxiliary parameter prediction
    parser.add_argument("--aux-weight", type=float, default=0.0,
                        help="Auxiliary parameter prediction loss weight (0=disabled)")
    parser.add_argument("--num-params", type=int, default=4,
                        help="Number of L-bracket parameters to predict")
    parser.add_argument("--aux-loss-type", type=str, default="correlation",
                        choices=["correlation", "mse", "mse_normalized", "direct"],
                        help="Auxiliary loss type: correlation, mse, mse_normalized, or direct (latent supervision)")

    # Output arguments
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vae_transformer"),
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

    # Compute beta warmup epochs (default: 30% of total epochs for transformer)
    beta_warmup_epochs = args.beta_warmup_epochs
    if beta_warmup_epochs is None:
        beta_warmup_epochs = int(args.epochs * 0.3)

    # Create encoder config
    encoder_config = VariableGraphVAEConfig(
        node_features=13,
        edge_features=2,
        num_face_types=8,  # Expanded face types
        face_embed_dim=args.face_embed_dim,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        hidden_dim=args.encoder_hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_gat_heads,
        latent_dim=args.latent_dim,
        encoder_dropout=args.encoder_dropout,
        pooling_type=args.pooling_type,
        attention_heads=args.attention_heads,
    )

    # Create decoder config
    decoder_config = TransformerDecoderConfig(
        latent_dim=args.latent_dim,
        node_features=13,
        edge_features=2,
        num_face_types=8,
        max_nodes=args.max_nodes,
        hidden_dim=args.decoder_hidden_dim,
        num_heads=args.num_decoder_heads,
        num_layers=args.num_decoder_layers,
        dropout=args.decoder_dropout,
        predict_edge_attr=False,
    )

    # Create model
    encoder = VariableGraphVAEEncoder(encoder_config)
    # Use param_head only if aux_weight > 0 AND not using direct latent supervision
    use_param_head = args.aux_weight > 0 and args.aux_loss_type != "direct"
    model = TransformerGraphVAE(
        encoder, decoder_config,
        use_param_head=use_param_head,
        num_params=args.num_params
    ).to(device)

    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:")
    print(f"  Encoder params: {encoder_params:,}")
    print(f"  Decoder params: {decoder_params:,}")
    print(f"  Total params:   {total_params:,}")
    print(f"\n  Encoder: GAT ({args.num_gat_layers} layers, {args.encoder_hidden_dim}D)")
    pooling_info = f"{args.pooling_type}"
    if args.pooling_type == "attention":
        pooling_info += f" ({args.attention_heads} heads)"
    print(f"  Pooling: {pooling_info}")
    print(f"  Decoder: Transformer ({args.num_decoder_layers} layers, {args.decoder_hidden_dim}D)")
    print(f"  Latent dim: {args.latent_dim}")

    # Loss config
    loss_config = HungarianLossConfig(
        face_type_cost_weight=args.face_type_weight,
        face_type_loss_weight=args.face_type_weight,
        edge_loss_weight=args.edge_weight,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Target beta: {args.target_beta}, Free bits: {args.free_bits}")
    print(f"  Beta warmup: {beta_warmup_epochs} epochs")
    print(f"  Learning rate: {args.lr}")
    if args.aux_weight > 0:
        print(f"  Aux weight: {args.aux_weight}, Num params: {args.num_params}, Loss type: {args.aux_loss_type}")
        if args.aux_loss_type == "direct":
            print(f"  Direct latent supervision: mu[:, :4] will encode normalized parameters")
            print(f"  KL exclusion: First {args.num_params} dims excluded from KL (free to encode params)")
    print("-" * 140)

    best_val_loss = float("inf")
    history = {"train": [], "val": [], "latent": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Compute beta for this epoch
        beta = get_beta_schedule(
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=beta_warmup_epochs,
            start_beta=0.0,
            target_beta=args.target_beta,
        )

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, beta, device,
            loss_config, args.max_nodes, args.max_edges, args.free_bits,
            args.aux_weight, args.num_params, args.aux_loss_type
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, beta, device,
            loss_config, args.max_nodes, args.max_edges, args.free_bits,
            args.aux_weight, args.num_params, args.aux_loss_type
        )

        # Compute latent metrics
        latent_metrics = compute_latent_metrics(model, val_loader, device)

        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history["latent"].append(latent_metrics)

        # Log
        epoch_time = time.time() - epoch_start
        active_dims = latent_metrics["active_dims"]
        mean_std = latent_metrics["mean_std"]

        # Collapse warning
        collapse_warning = ""
        if active_dims < args.latent_dim * 0.5:
            collapse_warning = " ⚠️ COLLAPSE"
        elif mean_std < 0.3:
            collapse_warning = " ⚠️ LOW_STD"

        log_str = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"β={beta:.3f} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Node: {val_metrics.get('node_loss', 0):.4f} | "
            f"Edge: {val_metrics.get('edge_loss', 0):.4f} | "
            f"KL: {val_metrics['kl_loss']:.4f} | "
            f"FaceAcc: {val_metrics.get('face_type_acc', 0):.1%} | "
            f"EdgeAcc: {val_metrics.get('edge_acc', 0):.1%}"
        )
        if args.aux_weight > 0:
            log_str += f" | Aux: {val_metrics.get('aux_param_loss', 0):.4f}"
        log_str += f" | Active: {active_dims}/{args.latent_dim}{collapse_warning}"
        print(log_str)

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                asdict(encoder_config), asdict(decoder_config),
                str(args.output_dir / "best_model.pt"),
                use_param_head, args.num_params
            )

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                asdict(encoder_config), asdict(decoder_config),
                str(args.output_dir / f"checkpoint_epoch_{epoch}.pt"),
                use_param_head, args.num_params
            )

    print("-" * 140)
    print("\nTraining complete!")

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model, test_loader, args.target_beta, device,
        loss_config, args.max_nodes, args.max_edges, args.free_bits,
        args.aux_weight, args.num_params, args.aux_loss_type
    )

    print(f"\nTest Results:")
    print(f"  Total Loss:        {test_metrics['loss']:.4f}")
    print(f"  Node Loss:         {test_metrics.get('node_loss', 0):.4f}")
    print(f"  Edge Loss:         {test_metrics.get('edge_loss', 0):.4f}")
    print(f"  KL Loss:           {test_metrics['kl_loss']:.4f}")
    print(f"  Face Type Acc:     {test_metrics.get('face_type_acc', 0):.1%}")
    print(f"  Existence Acc:     {test_metrics.get('existence_acc', 0):.1%}")
    print(f"  Edge Acc:          {test_metrics.get('edge_acc', 0):.1%}")
    print(f"  Edge Precision:    {test_metrics.get('edge_precision', 0):.1%}")
    print(f"  Edge Recall:       {test_metrics.get('edge_recall', 0):.1%}")

    # Latent metrics
    print("\nLatent space metrics:")
    latent_metrics = compute_latent_metrics(model, test_loader, device)
    print(f"  Mean ||z||:        {latent_metrics['mean_norm']:.3f}")
    print(f"  Mean std(z):       {latent_metrics['mean_std']:.3f}")
    print(f"  Active dims:       {latent_metrics['active_dims']}/{args.latent_dim}")
    print(f"  KL from prior:     {latent_metrics['kl_from_prior']:.2f}")

    # Save results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "encoder_config": asdict(encoder_config),
        "decoder_config": asdict(decoder_config),
        "test_metrics": test_metrics,
        "latent_metrics": latent_metrics,
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
