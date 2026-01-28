#!/usr/bin/env python3
"""
Pre-train the Extended LLM on latent -> class + params mapping.

Stage 1 of LLM training: Learn to classify geometry type and predict
parameters from frozen VAE latent vectors. No text instructions yet.

Usage:
    python scripts/train_llm_pretrain.py \
        --vae-checkpoint outputs/hetero_vae/best_model.pt \
        --epochs 50
"""

from __future__ import annotations

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from graph_cad.data.multi_geometry_dataset import (
    create_multi_geometry_loaders,
)
from graph_cad.data.brep_types import GEOMETRY_TYPE_NAMES
from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig
from graph_cad.models.extended_latent_editor import (
    ExtendedLatentEditor,
    ExtendedLatentEditorConfig,
    compute_extended_editor_loss,
)


def load_vae(checkpoint_path: str, device: str) -> HeteroVAE:
    """Load pre-trained VAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get('config', {})
    config = HeteroVAEConfig(**config_dict)

    model = HeteroVAE(config, use_param_head=True, num_params=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Freeze VAE
    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def encode_batch(vae: HeteroVAE, batch, device: str) -> torch.Tensor:
    """Encode batch of graphs to latent vectors using frozen VAE."""
    batch = batch.to(device)
    mu, _ = vae.encode(batch)
    return mu


def train_epoch(
    llm: ExtendedLatentEditor,
    vae: HeteroVAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    class_weight: float = 1.0,
    param_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    llm.train()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in train_loader:
        # Encode to latent
        z = encode_batch(vae, batch, device)

        # Get targets
        geometry_types = batch.geometry_type.squeeze(-1).to(device)
        params_normalized = batch.params_normalized.to(device)
        params_mask = batch.params_mask.to(device)

        targets = {
            'geometry_type': geometry_types,
            'params_normalized': params_normalized,
            'params_mask': params_mask,
        }

        # Forward pass (pretrain mode)
        outputs = llm.forward_pretrain(z, geometry_types)

        # Compute loss
        loss, metrics = compute_extended_editor_loss(
            outputs, targets, class_weight, param_weight
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
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
def evaluate(
    llm: ExtendedLatentEditor,
    vae: HeteroVAE,
    val_loader,
    device: str,
    class_weight: float = 1.0,
    param_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate on validation set."""
    llm.eval()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    # Per-type accuracy tracking
    type_correct = {i: 0 for i in range(len(GEOMETRY_TYPE_NAMES))}
    type_total = {i: 0 for i in range(len(GEOMETRY_TYPE_NAMES))}

    for batch in val_loader:
        z = encode_batch(vae, batch, device)

        geometry_types = batch.geometry_type.squeeze(-1).to(device)
        params_normalized = batch.params_normalized.to(device)
        params_mask = batch.params_mask.to(device)

        targets = {
            'geometry_type': geometry_types,
            'params_normalized': params_normalized,
            'params_mask': params_mask,
        }

        outputs = llm.forward_pretrain(z, geometry_types)

        loss, metrics = compute_extended_editor_loss(
            outputs, targets, class_weight, param_weight
        )

        # Track per-type accuracy
        pred_types = outputs['class_logits'].argmax(dim=-1)
        for i in range(len(geometry_types)):
            gt_type = geometry_types[i].item()
            type_total[gt_type] += 1
            if pred_types[i].item() == gt_type:
                type_correct[gt_type] += 1

        total_loss += loss.item()
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss

    # Per-type accuracy
    for geo_type, name in GEOMETRY_TYPE_NAMES.items():
        if type_total[geo_type] > 0:
            avg_metrics[f"acc_{name}"] = type_correct[geo_type] / type_total[geo_type]

    return avg_metrics


def save_checkpoint(
    llm: ExtendedLatentEditor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    vae_checkpoint: str,
    path: str,
):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": llm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "vae_checkpoint": vae_checkpoint,
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train Extended LLM (Stage 1) on latent -> class + params"
    )

    # VAE checkpoint
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                        help="Path to pre-trained VAE checkpoint")

    # Data arguments
    parser.add_argument("--samples-per-type", type=int, default=5000,
                        help="Training samples per geometry type")
    parser.add_argument("--val-samples-per-type", type=int, default=500,
                        help="Validation samples per geometry type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent space dimension")
    parser.add_argument("--llm-hidden-dim", type=int, default=4096, help="LLM hidden dimension")
    parser.add_argument("--class-hidden-dim", type=int, default=256, help="Classification head hidden dim")
    parser.add_argument("--param-hidden-dim", type=int, default=128, help="Parameter head hidden dim")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Loss weights
    parser.add_argument("--class-weight", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--param-weight", type=float, default=1.0, help="Parameter loss weight")

    # Output arguments
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_pretrain"),
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

    # Load frozen VAE
    print(f"\nLoading frozen VAE from: {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)
    print("  VAE loaded and frozen")

    # Create data loaders
    print(f"\nCreating multi-geometry datasets...")
    start_time = time.time()
    train_loader, val_loader, test_loader = create_multi_geometry_loaders(
        train_samples_per_type=args.samples_per_type,
        val_samples_per_type=args.val_samples_per_type,
        test_samples_per_type=args.val_samples_per_type,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Dataset creation took {time.time() - start_time:.1f}s")

    # Create LLM
    config = ExtendedLatentEditorConfig(
        latent_dim=args.latent_dim,
        llm_hidden_dim=args.llm_hidden_dim,
        class_hidden_dim=args.class_hidden_dim,
        param_hidden_dim=args.param_hidden_dim,
        training_mode="pretrain",
    )

    llm = ExtendedLatentEditor(config).to(device)

    trainable_params = llm.num_trainable_params("pretrain")
    print(f"\nModel: Extended LLM (pretrain mode)")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Latent dim: {args.latent_dim}")

    # Optimizer and scheduler
    optimizer = AdamW(
        llm.get_trainable_parameters("pretrain"),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Class weight: {args.class_weight}, Param weight: {args.param_weight}")
    print("-" * 100)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            llm, vae, train_loader, optimizer, device,
            args.class_weight, args.param_weight
        )

        val_metrics = evaluate(
            llm, vae, val_loader, device,
            args.class_weight, args.param_weight
        )

        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Log
        epoch_time = time.time() - epoch_start
        class_acc = val_metrics.get('class_acc', 0)
        param_mae = val_metrics.get('param_mae', 0)

        log_str = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"ClassAcc: {class_acc:.1%} | "
            f"ParamMAE: {param_mae:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                llm, optimizer, epoch, val_metrics,
                {
                    "latent_dim": args.latent_dim,
                    "llm_hidden_dim": args.llm_hidden_dim,
                    "class_hidden_dim": args.class_hidden_dim,
                    "param_hidden_dim": args.param_hidden_dim,
                },
                args.vae_checkpoint,
                str(args.output_dir / "best_model.pt"),
            )

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(
                llm, optimizer, epoch, val_metrics,
                {
                    "latent_dim": args.latent_dim,
                    "llm_hidden_dim": args.llm_hidden_dim,
                    "class_hidden_dim": args.class_hidden_dim,
                    "param_hidden_dim": args.param_hidden_dim,
                },
                args.vae_checkpoint,
                str(args.output_dir / f"checkpoint_epoch_{epoch}.pt"),
            )

    print("-" * 100)
    print("\nTraining complete!")

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        llm, vae, test_loader, device,
        args.class_weight, args.param_weight
    )

    print(f"\nTest Results:")
    print(f"  Total Loss:     {test_metrics['loss']:.4f}")
    print(f"  Class Acc:      {test_metrics['class_acc']:.1%}")
    print(f"  Parameter MAE:  {test_metrics['param_mae']:.4f}")

    # Per-type accuracy
    print("\n  Per-type accuracy:")
    for geo_type, name in GEOMETRY_TYPE_NAMES.items():
        acc_key = f"acc_{name}"
        if acc_key in test_metrics:
            print(f"    {name:10s}: {test_metrics[acc_key]:.1%}")

    # Save results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
