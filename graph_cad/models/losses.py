"""
Loss functions for Graph VAE training.

Includes reconstruction loss, KL divergence, and combined VAE loss
with optional semantic loss using a pre-trained parameter regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from graph_cad.models import ParameterRegressor


@dataclass
class VAELossConfig:
    """Configuration for VAE loss computation."""

    # Overall weights
    node_weight: float = 1.0
    edge_weight: float = 1.0

    # Feature-specific weights within node features
    # Indices: [face_type(0), area(1), dir_xyz(2-4), centroid_xyz(5-7)]
    face_type_weight: float = 2.0  # Categorical-like, needs higher weight
    area_weight: float = 1.0
    direction_weight: float = 1.0
    centroid_weight: float = 1.0


def reconstruction_loss(
    node_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_pred: torch.Tensor,
    edge_target: torch.Tensor,
    config: VAELossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute reconstruction loss for graph features.

    Args:
        node_pred: Predicted node features, shape (batch, num_nodes, node_features).
        node_target: Target node features, shape (batch, num_nodes, node_features).
        edge_pred: Predicted edge features, shape (batch, num_edges, edge_features).
        edge_target: Target edge features, shape (batch, num_edges, edge_features).
        config: Loss configuration.

    Returns:
        total_loss: Scalar reconstruction loss.
        loss_dict: Component losses for logging.
    """
    if config is None:
        config = VAELossConfig()

    # Node feature losses (with per-feature weighting)
    # face_type: index 0
    face_type_loss = F.mse_loss(node_pred[..., 0], node_target[..., 0])

    # area: index 1
    area_loss = F.mse_loss(node_pred[..., 1], node_target[..., 1])

    # direction (normal/axis): indices 2-4
    direction_loss = F.mse_loss(node_pred[..., 2:5], node_target[..., 2:5])

    # centroid: indices 5-7
    centroid_loss = F.mse_loss(node_pred[..., 5:8], node_target[..., 5:8])

    # Weighted node loss
    node_loss = (
        config.face_type_weight * face_type_loss
        + config.area_weight * area_loss
        + config.direction_weight * direction_loss
        + config.centroid_weight * centroid_loss
    )

    # Edge feature loss
    edge_loss = F.mse_loss(edge_pred, edge_target)

    # Combined loss
    total_loss = config.node_weight * node_loss + config.edge_weight * edge_loss

    # Loss components for logging
    loss_dict = {
        "node_loss": node_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "face_type_loss": face_type_loss.detach(),
        "area_loss": area_loss.detach(),
        "direction_loss": direction_loss.detach(),
        "centroid_loss": centroid_loss.detach(),
    }

    return total_loss, loss_dict


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """
    Compute KL divergence KL(q(z|x) || p(z)) where p(z) = N(0, I).

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu: Latent mean, shape (batch, latent_dim).
        logvar: Latent log variance, shape (batch, latent_dim).
        free_bits: Minimum KL per dimension to prevent posterior collapse.
                   If > 0, dimensions with KL < free_bits contribute 0.

    Returns:
        kl_loss: Scalar KL divergence (averaged over batch).
    """
    # Per-dimension KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits > 0:
        # Free bits: don't penalize dimensions below threshold
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits) - free_bits

    # Sum over latent dimensions, mean over batch
    kl_loss = kl_per_dim.sum(dim=-1).mean()

    return kl_loss


def vae_loss(
    node_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_pred: torch.Tensor,
    edge_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    config: VAELossConfig | None = None,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined VAE loss = Reconstruction + beta * KL.

    Args:
        node_pred: Predicted node features.
        node_target: Target node features.
        edge_pred: Predicted edge features.
        edge_target: Target edge features.
        mu: Latent mean.
        logvar: Latent log variance.
        beta: Weight for KL divergence term.
        config: Loss configuration.
        free_bits: Minimum KL per dimension.

    Returns:
        total_loss: Combined loss.
        loss_dict: All components for logging.
    """
    # Reconstruction loss
    recon_loss, recon_dict = reconstruction_loss(
        node_pred, node_target, edge_pred, edge_target, config
    )

    # KL divergence
    kl_loss = kl_divergence(mu, logvar, free_bits)

    # Combined loss
    total_loss = recon_loss + beta * kl_loss

    # Aggregate loss dict
    loss_dict = {
        **recon_dict,
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "beta": torch.tensor(beta),
        "total_loss": total_loss.detach(),
    }

    return total_loss, loss_dict


def auxiliary_param_loss(
    param_pred: torch.Tensor,
    param_target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute auxiliary parameter prediction loss.

    Forces the VAE latent space to encode all L-bracket parameters
    by supervising a prediction head with ground truth parameters.

    Args:
        param_pred: Predicted parameters from param_head, shape (batch, 8).
        param_target: Ground truth normalized parameters, shape (batch, 8).

    Returns:
        aux_loss: MSE between predicted and target parameters.
    """
    return F.mse_loss(param_pred, param_target)


def vae_loss_with_aux(
    node_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_pred: torch.Tensor,
    edge_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    param_pred: torch.Tensor,
    param_target: torch.Tensor,
    beta: float = 1.0,
    aux_weight: float = 1.0,
    config: VAELossConfig | None = None,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    VAE loss with auxiliary parameter prediction loss.

    Loss = Reconstruction + beta * KL + aux_weight * ParamMSE

    The auxiliary loss forces the latent space to encode all 8 L-bracket
    parameters, preventing dimensionality collapse where some parameters
    (like thickness and hole diameters) are not represented.

    Args:
        node_pred: Predicted node features.
        node_target: Target node features.
        edge_pred: Predicted edge features.
        edge_target: Target edge features.
        mu: Latent mean.
        logvar: Latent log variance.
        param_pred: Predicted parameters from param_head, shape (batch, 8).
        param_target: Ground truth normalized parameters, shape (batch, 8).
        beta: Weight for KL divergence.
        aux_weight: Weight for auxiliary parameter loss.
        config: Loss configuration.
        free_bits: Minimum KL per dimension.

    Returns:
        total_loss: Combined loss.
        loss_dict: All components for logging.
    """
    # Base VAE loss
    base_loss, loss_dict = vae_loss(
        node_pred, node_target, edge_pred, edge_target,
        mu, logvar, beta, config, free_bits
    )

    # Auxiliary parameter loss
    aux_loss = auxiliary_param_loss(param_pred, param_target)

    # Combined
    total_loss = base_loss + aux_weight * aux_loss

    loss_dict["aux_param_loss"] = aux_loss.detach()
    loss_dict["aux_weight"] = torch.tensor(aux_weight)
    loss_dict["total_loss"] = total_loss.detach()

    return total_loss, loss_dict


def semantic_loss(
    node_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_pred: torch.Tensor,
    edge_target: torch.Tensor,
    edge_index: torch.Tensor,
    regressor: ParameterRegressor,
    batch_size: int,
) -> torch.Tensor:
    """
    Compute semantic loss using pre-trained parameter regressor.

    Ensures reconstructed graphs encode valid geometry by checking
    that a frozen regressor produces similar parameter predictions.

    Args:
        node_pred: Predicted node features, shape (batch, num_nodes, node_features).
        node_target: Target node features, shape (batch, num_nodes, node_features).
        edge_pred: Predicted edge features, shape (batch, num_edges, edge_features).
        edge_target: Target edge features, shape (batch, num_edges, edge_features).
        edge_index: Edge indices (same for all samples in batch).
        regressor: Pre-trained frozen ParameterRegressor.
        batch_size: Batch size for creating batch indices.

    Returns:
        semantic_loss: MSE between regressor outputs on pred vs target.
    """
    num_nodes = node_pred.shape[1]
    num_edges = edge_pred.shape[1]

    # Flatten batch dimension for regressor
    # node_pred: (batch, num_nodes, features) -> (batch * num_nodes, features)
    node_pred_flat = node_pred.view(-1, node_pred.shape[-1])
    node_target_flat = node_target.view(-1, node_target.shape[-1])

    # edge_pred: (batch, num_edges, features) -> (batch * num_edges, features)
    edge_pred_flat = edge_pred.view(-1, edge_pred.shape[-1])
    edge_target_flat = edge_target.view(-1, edge_target.shape[-1])

    # Create batch indices for pooling
    batch_indices = torch.arange(batch_size, device=node_pred.device)
    batch_indices = batch_indices.repeat_interleave(num_nodes)

    # Expand edge_index for batched graphs
    # Each graph has the same topology, offset indices by num_nodes per graph
    edge_index_batched = []
    for i in range(batch_size):
        offset = i * num_nodes
        edge_index_batched.append(edge_index + offset)
    edge_index_batched = torch.cat(edge_index_batched, dim=1)

    # Expand edge attributes
    edge_pred_batched = edge_pred_flat
    edge_target_batched = edge_target_flat

    # Run regressor on both
    with torch.no_grad():
        params_from_pred = regressor(
            node_pred_flat, edge_index_batched, edge_pred_batched, batch_indices
        )
        params_from_target = regressor(
            node_target_flat, edge_index_batched, edge_target_batched, batch_indices
        )

    # MSE between parameter predictions
    return F.mse_loss(params_from_pred, params_from_target)


def vae_loss_with_semantic(
    node_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_pred: torch.Tensor,
    edge_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    edge_index: torch.Tensor,
    regressor: ParameterRegressor,
    beta: float = 1.0,
    semantic_weight: float = 0.1,
    config: VAELossConfig | None = None,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    VAE loss with optional semantic loss from parameter regressor.

    Loss = Reconstruction + beta * KL + semantic_weight * Semantic

    Args:
        node_pred: Predicted node features.
        node_target: Target node features.
        edge_pred: Predicted edge features.
        edge_target: Target edge features.
        mu: Latent mean.
        logvar: Latent log variance.
        edge_index: Edge indices (same for all samples).
        regressor: Pre-trained frozen ParameterRegressor.
        beta: Weight for KL divergence.
        semantic_weight: Weight for semantic loss.
        config: Loss configuration.
        free_bits: Minimum KL per dimension.

    Returns:
        total_loss: Combined loss.
        loss_dict: All components for logging.
    """
    # Base VAE loss
    base_loss, loss_dict = vae_loss(
        node_pred, node_target, edge_pred, edge_target,
        mu, logvar, beta, config, free_bits
    )

    # Semantic loss
    batch_size = node_pred.shape[0]
    sem_loss = semantic_loss(
        node_pred, node_target, edge_pred, edge_target,
        edge_index, regressor, batch_size
    )

    # Combined
    total_loss = base_loss + semantic_weight * sem_loss

    loss_dict["semantic_loss"] = sem_loss.detach()
    loss_dict["total_loss"] = total_loss.detach()

    return total_loss, loss_dict
