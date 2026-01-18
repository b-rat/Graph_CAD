#!/usr/bin/env python3
"""
Train a regressor to predict L-bracket parameters directly from latent vectors.

This bypasses the VAE decoder entirely - predicting params from the 32D latent
rather than from the 280D decoded features.

Usage:
    python scripts/train_latent_regressor.py \
        --vae-checkpoint outputs/vae_variable/best_model.pt \
        --epochs 100 \
        --output-dir outputs/latent_regressor
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
from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig, VariableGraphVAEEncoder
from graph_cad.models.transformer_decoder import TransformerDecoderConfig, TransformerGraphVAE


PARAM_NAMES = ["leg1_length", "leg2_length", "width", "thickness"]


@dataclass
class LatentRegressorConfig:
    """Configuration for latent-to-parameter regressor."""
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    use_batch_norm: bool = True
    num_params: int = 4


class LatentRegressor(nn.Module):
    """
    MLP to predict L-bracket parameters directly from latent vectors.

    Much simpler than going through decoded features - if the latent
    encodes the parameters (even entangled), this can learn to extract them.
    """

    def __init__(self, config: LatentRegressorConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.latent_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_dim, config.num_params)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) latent vectors
        Returns:
            params: (batch, 4) normalized parameters
        """
        h = self.backbone(z)
        return self.output_head(h)


def denormalize_params(params: torch.Tensor) -> torch.Tensor:
    """Convert normalized [0,1] params to actual mm values."""
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


def load_vae(checkpoint_path: str, device: str) -> tuple[nn.Module, dict, str]:
    """Load VAE (supports both Transformer and MLP decoder variants).

    Returns:
        vae: The loaded VAE model
        checkpoint: The checkpoint dict
        vae_type: "transformer" or "variable_mlp"
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if this is a TransformerGraphVAE (has decoder_config key)
    if "decoder_config" in checkpoint:
        # TransformerGraphVAE
        encoder_config = VariableGraphVAEConfig(**checkpoint["encoder_config"])
        encoder = VariableGraphVAEEncoder(encoder_config)

        decoder_config = TransformerDecoderConfig(**checkpoint["decoder_config"])

        use_param_head = checkpoint.get("use_param_head", False)
        num_params = checkpoint.get("num_params", 4)

        model = TransformerGraphVAE(
            encoder, decoder_config,
            use_param_head=use_param_head,
            num_params=num_params,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint, "transformer"
    else:
        # VariableGraphVAE (MLP decoder)
        config = VariableGraphVAEConfig(**checkpoint["config"])
        model = VariableGraphVAE(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, checkpoint, "variable_mlp"


def prepare_dataset(
    vae: nn.Module,
    vae_type: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    device: str,
    desc: str = "Encoding",
    cache_dir: Path | None = None,
    split: str = "train",
) -> TensorDataset:
    """Encode brackets through VAE and return (z, params) dataset.

    Args:
        vae: VAE model (TransformerGraphVAE or VariableGraphVAE)
        vae_type: "transformer" or "variable_mlp"
        num_samples: Number of samples to generate
        batch_size: Batch size for encoding
        seed: Random seed
        device: Device for computation
        desc: Description for progress bar
        cache_dir: Optional cache directory
        split: Dataset split name (for cache filename)

    Returns:
        TensorDataset with (z, params) pairs
    """
    # Get max_nodes/max_edges based on VAE type
    if vae_type == "transformer":
        max_nodes = vae.decoder_config.max_nodes
        max_edges = 50  # Standard value
    else:
        max_nodes = vae.config.max_nodes
        max_edges = vae.config.max_edges

    # Check cache
    if cache_dir is not None:
        cache_path = cache_dir / f"latent_{split}_{num_samples}_{seed}.pt"
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            data = torch.load(cache_path, weights_only=True)
            return TensorDataset(data["z"], data["params"])

    # Create dataset
    dataset = VariableLBracketDataset(
        num_samples=num_samples,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed,
    )
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_z = []
    all_params = []

    vae.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            face_types = batch.face_types.to(device)
            node_mask = batch.node_mask.to(device)
            params = batch.y.to(device)

            num_graphs = batch_idx.max().item() + 1
            params = params.view(num_graphs, -1)

            # Encode to latent (use mean, not sampled)
            mu, _ = vae.encode(
                x, face_types, edge_index, edge_attr, batch_idx, node_mask
            )

            all_z.append(mu.cpu())
            all_params.append(params.cpu())

    z = torch.cat(all_z, dim=0)
    params = torch.cat(all_params, dim=0)

    # Save to cache
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"latent_{split}_{num_samples}_{seed}.pt"
        torch.save({"z": z, "params": params}, cache_path)
        print(f"  Saved to cache: {cache_path}")

    return TensorDataset(z, params)


def train_epoch(
    model: LatentRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for z, params in loader:
        z = z.to(device)
        params = params.to(device)

        optimizer.zero_grad()
        pred = model(z)
        loss = F.mse_loss(pred, params)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"loss": total_loss / num_batches}


@torch.no_grad()
def evaluate(
    model: LatentRegressor,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    for z, params in loader:
        z = z.to(device)
        params = params.to(device)

        pred = model(z)
        loss = F.mse_loss(pred, params)

        total_loss += loss.item()
        num_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(params.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Denormalize for mm errors
    preds_mm = denormalize_params(preds)
    targets_mm = denormalize_params(targets)

    per_param_mae = (preds_mm - targets_mm).abs().mean(dim=0)
    per_param_metrics = {
        f"mae_{name}": per_param_mae[i].item()
        for i, name in enumerate(PARAM_NAMES)
    }

    return {
        "loss": total_loss / num_batches,
        "mae_denorm": (preds_mm - targets_mm).abs().mean().item(),
        **per_param_metrics,
    }


def save_regressor(model, optimizer, epoch, metrics, path):
    """Save checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "latent_dim": model.config.latent_dim,
            "hidden_dims": model.config.hidden_dims,
            "dropout": model.config.dropout,
            "use_batch_norm": model.config.use_batch_norm,
            "num_params": model.config.num_params,
        },
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train Latent Regressor")
    parser.add_argument("--vae-checkpoint", type=str, default="outputs/vae_transformer/best_model.pt")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/latent_regressor")
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
    vae, _, vae_type = load_vae(args.vae_checkpoint, device)
    for p in vae.parameters():
        p.requires_grad = False

    # Get latent dim based on VAE type
    if vae_type == "transformer":
        latent_dim = vae.decoder_config.latent_dim
        print(f"  VAE type: TransformerGraphVAE")
    else:
        latent_dim = vae.config.latent_dim
        print(f"  VAE type: VariableGraphVAE (MLP decoder)")
    print(f"  Latent dim: {latent_dim}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Prepare datasets
    print(f"\nEncoding training data ({args.train_size} samples)...")
    start = time.time()
    train_dataset = prepare_dataset(
        vae, vae_type, args.train_size, args.batch_size, args.seed, device,
        "Train", cache_dir, "train"
    )
    print(f"  Done in {time.time() - start:.1f}s")

    print(f"Encoding validation data ({args.val_size} samples)...")
    val_dataset = prepare_dataset(
        vae, vae_type, args.val_size, args.batch_size, args.seed + 1000, device,
        "Val", cache_dir, "val"
    )

    print(f"Encoding test data ({args.test_size} samples)...")
    test_dataset = prepare_dataset(
        vae, vae_type, args.test_size, args.batch_size, args.seed + 2000, device,
        "Test", cache_dir, "test"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    config = LatentRegressorConfig(
        latent_dim=latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    model = LatentRegressor(config).to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Architecture: {config.latent_dim}D → {config.hidden_dims} → {config.num_params}D")

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
    print(f"  Overall MAE: {test_metrics['mae_denorm']:.2f}mm")
    for name in PARAM_NAMES:
        print(f"  {name}: {test_metrics[f'mae_{name}']:.2f}mm")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
