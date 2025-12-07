"""
Training utilities for Graph VAE.

Includes beta scheduling strategies and training/evaluation functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Batch

from graph_cad.models.losses import (
    VAELossConfig,
    vae_loss,
    vae_loss_with_semantic,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from graph_cad.models import GraphVAE, ParameterRegressor


@dataclass
class BetaScheduleConfig:
    """Configuration for beta scheduling."""

    strategy: str = "warmup"  # constant, linear, warmup, cyclical
    target_beta: float = 1.0
    warmup_epochs: int = 10
    anneal_epochs: int = 20
    cycle_epochs: int = 10  # For cyclical scheduling


class BetaScheduler:
    """
    Beta scheduling strategies for KL annealing.

    Strategies:
        - constant: beta stays fixed at target_beta
        - linear: beta increases linearly from 0 to target over anneal_epochs
        - warmup: beta=0 for warmup_epochs, then linear increase over anneal_epochs
        - cyclical: beta cycles between 0 and target (for disentanglement)

    Args:
        config: Scheduling configuration.
    """

    def __init__(self, config: BetaScheduleConfig | None = None):
        self.config = config or BetaScheduleConfig()

    def get_beta(self, epoch: int) -> float:
        """
        Get beta value for current epoch (1-indexed).

        Args:
            epoch: Current epoch number (starting from 1).

        Returns:
            beta: Beta value for this epoch.
        """
        cfg = self.config

        if cfg.strategy == "constant":
            return cfg.target_beta

        elif cfg.strategy == "linear":
            # Linear from 0 to target over anneal_epochs
            progress = min(epoch / cfg.anneal_epochs, 1.0)
            return cfg.target_beta * progress

        elif cfg.strategy == "warmup":
            # Zero for warmup_epochs, then linear increase
            if epoch <= cfg.warmup_epochs:
                return 0.0
            else:
                progress = min(
                    (epoch - cfg.warmup_epochs) / cfg.anneal_epochs, 1.0
                )
                return cfg.target_beta * progress

        elif cfg.strategy == "cyclical":
            # Cyclical annealing (useful for disentanglement)
            cycle_position = (epoch - 1) % cfg.cycle_epochs
            progress = cycle_position / cfg.cycle_epochs
            # Linear increase within each cycle
            return cfg.target_beta * progress

        else:
            raise ValueError(f"Unknown beta strategy: {cfg.strategy}")


def prepare_batch_targets(
    batch: Batch,
    num_nodes: int = 10,
    node_features: int = 8,
    num_edges: int = 22,
    edge_features: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare target tensors from PyG batch for loss computation.

    Reshapes flat batch tensors to (batch_size, num_items, features) format.

    Args:
        batch: PyG Batch object.
        num_nodes: Number of nodes per graph.
        node_features: Features per node.
        num_edges: Number of edges per graph.
        edge_features: Features per edge.

    Returns:
        node_target: Shape (batch_size, num_nodes, node_features).
        edge_target: Shape (batch_size, num_edges, edge_features).
    """
    batch_size = batch.num_graphs

    # Reshape node features: (total_nodes, features) -> (batch, nodes, features)
    node_target = batch.x.view(batch_size, num_nodes, node_features)

    # Reshape edge features: (total_edges, features) -> (batch, edges, features)
    edge_target = batch.edge_attr.view(batch_size, num_edges, edge_features)

    return node_target, edge_target


def train_epoch(
    model: GraphVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: str,
    loss_config: VAELossConfig | None = None,
    regressor: ParameterRegressor | None = None,
    semantic_weight: float = 0.1,
    free_bits: float = 0.0,
) -> dict[str, float]:
    """
    Train VAE for one epoch.

    Args:
        model: Graph VAE model.
        loader: Training data loader.
        optimizer: Optimizer.
        beta: Current beta value for KL weighting.
        device: Device to train on.
        loss_config: Loss configuration.
        regressor: Optional pre-trained regressor for semantic loss.
        semantic_weight: Weight for semantic loss if regressor provided.
        free_bits: Minimum KL per dimension.

    Returns:
        Dictionary with averaged metrics for the epoch.
    """
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_semantic = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Prepare targets
        node_target, edge_target = prepare_batch_targets(
            batch,
            num_nodes=model.config.num_nodes,
            node_features=model.config.node_features,
            num_edges=model.config.num_edges,
            edge_features=model.config.edge_features,
        )

        # Compute loss
        if regressor is not None:
            # Get single-graph edge_index for semantic loss
            single_edge_index = batch.edge_index[:, :model.config.num_edges]
            loss, loss_dict = vae_loss_with_semantic(
                outputs["node_recon"],
                node_target,
                outputs["edge_recon"],
                edge_target,
                outputs["mu"],
                outputs["logvar"],
                single_edge_index,
                regressor,
                beta=beta,
                semantic_weight=semantic_weight,
                config=loss_config,
                free_bits=free_bits,
            )
            total_semantic += loss_dict["semantic_loss"].item()
        else:
            loss, loss_dict = vae_loss(
                outputs["node_recon"],
                node_target,
                outputs["edge_recon"],
                edge_target,
                outputs["mu"],
                outputs["logvar"],
                beta=beta,
                config=loss_config,
                free_bits=free_bits,
            )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict["total_loss"].item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()
        num_batches += 1

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
        "beta": beta,
    }

    if regressor is not None:
        metrics["semantic_loss"] = total_semantic / num_batches

    return metrics


@torch.no_grad()
def evaluate(
    model: GraphVAE,
    loader: DataLoader,
    beta: float,
    device: str,
    loss_config: VAELossConfig | None = None,
    regressor: ParameterRegressor | None = None,
    semantic_weight: float = 0.1,
    free_bits: float = 0.0,
) -> dict[str, float]:
    """
    Evaluate VAE on a data loader.

    Args:
        model: Graph VAE model.
        loader: Data loader.
        beta: Beta value for KL weighting.
        device: Device to evaluate on.
        loss_config: Loss configuration.
        regressor: Optional pre-trained regressor for semantic loss.
        semantic_weight: Weight for semantic loss.
        free_bits: Minimum KL per dimension.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_semantic = 0.0
    total_node_mse = 0.0
    total_edge_mse = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Prepare targets
        node_target, edge_target = prepare_batch_targets(
            batch,
            num_nodes=model.config.num_nodes,
            node_features=model.config.node_features,
            num_edges=model.config.num_edges,
            edge_features=model.config.edge_features,
        )

        # Compute loss
        if regressor is not None:
            single_edge_index = batch.edge_index[:, :model.config.num_edges]
            loss, loss_dict = vae_loss_with_semantic(
                outputs["node_recon"],
                node_target,
                outputs["edge_recon"],
                edge_target,
                outputs["mu"],
                outputs["logvar"],
                single_edge_index,
                regressor,
                beta=beta,
                semantic_weight=semantic_weight,
                config=loss_config,
                free_bits=free_bits,
            )
            total_semantic += loss_dict["semantic_loss"].item()
        else:
            loss, loss_dict = vae_loss(
                outputs["node_recon"],
                node_target,
                outputs["edge_recon"],
                edge_target,
                outputs["mu"],
                outputs["logvar"],
                beta=beta,
                config=loss_config,
                free_bits=free_bits,
            )

        # Raw MSE (unweighted)
        node_mse = torch.nn.functional.mse_loss(
            outputs["node_recon"], node_target
        ).item()
        edge_mse = torch.nn.functional.mse_loss(
            outputs["edge_recon"], edge_target
        ).item()

        # Accumulate
        total_loss += loss_dict["total_loss"].item()
        total_recon += loss_dict["recon_loss"].item()
        total_kl += loss_dict["kl_loss"].item()
        total_node_mse += node_mse
        total_edge_mse += edge_mse
        num_batches += 1

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
        "node_mse": total_node_mse / num_batches,
        "edge_mse": total_edge_mse / num_batches,
        "beta": beta,
    }

    if regressor is not None:
        metrics["semantic_loss"] = total_semantic / num_batches

    return metrics


@torch.no_grad()
def compute_latent_metrics(
    model: GraphVAE,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    """
    Compute latent space quality metrics.

    Args:
        model: Graph VAE model.
        loader: Data loader.
        device: Device.

    Returns:
        Dictionary with:
            - mean_norm: Average L2 norm of latent vectors
            - mean_std: Average std across latent dimensions
            - active_dims: Number of dimensions with variance > 0.1
            - kl_from_prior: How close aggregated posterior is to N(0,I)
    """
    model.eval()

    all_z = []
    all_mu = []

    for batch in loader:
        batch = batch.to(device)
        mu, logvar = model.encode(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        z = model.reparameterize(mu, logvar)

        all_z.append(z.cpu())
        all_mu.append(mu.cpu())

    # Concatenate
    all_z = torch.cat(all_z, dim=0)  # (N, latent_dim)
    all_mu = torch.cat(all_mu, dim=0)

    # Metrics
    mean_norm = all_z.norm(dim=1).mean().item()
    per_dim_std = all_z.std(dim=0)
    mean_std = per_dim_std.mean().item()
    active_dims = (per_dim_std > 0.1).sum().item()

    # KL from aggregated posterior to prior
    # Approximate: compare empirical mean/var to N(0,I)
    empirical_mean = all_z.mean(dim=0)
    empirical_var = all_z.var(dim=0)
    # KL(N(mu, sigma^2) || N(0, 1)) per dim
    kl_per_dim = 0.5 * (empirical_var + empirical_mean.pow(2) - 1 - empirical_var.log())
    kl_from_prior = kl_per_dim.sum().item()

    return {
        "mean_norm": mean_norm,
        "mean_std": mean_std,
        "active_dims": active_dims,
        "active_dims_ratio": active_dims / all_z.shape[1],
        "kl_from_prior": kl_from_prior,
    }


def save_checkpoint(
    model: GraphVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,
        "metrics": metrics,
    }, path)


def load_checkpoint(
    path: str,
    device: str = "cpu",
) -> tuple[GraphVAE, dict]:
    """
    Load model from checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        model: Loaded GraphVAE model.
        checkpoint: Full checkpoint dict with metrics, epoch, etc.
    """
    from graph_cad.models import GraphVAE

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GraphVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, checkpoint
