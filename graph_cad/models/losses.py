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


# =============================================================================
# Variable Topology Loss Functions (Phase 2)
# =============================================================================


@dataclass
class VariableVAELossConfig:
    """Configuration for Variable Topology VAE loss."""

    # Reconstruction weights
    node_weight: float = 1.0
    edge_weight: float = 1.0

    # Mask prediction weights
    node_mask_weight: float = 1.0
    edge_mask_weight: float = 1.0

    # Face type classification weight
    face_type_weight: float = 0.5

    # Feature-specific weights within node features (13D)
    # [area(0), dir_xyz(1-3), centroid_xyz(4-6), curv1(7), curv2(8),
    #  bbox_diagonal(9), bbox_cx(10), bbox_cy(11), bbox_cz(12)]
    area_weight: float = 1.0
    direction_weight: float = 1.0
    centroid_weight: float = 1.0
    curvature_weight: float = 1.0
    bbox_weight: float = 1.0  # Weight for bbox features (dims 9-12)


def variable_reconstruction_loss(
    node_pred: torch.Tensor,
    node_target: torch.Tensor,
    edge_pred: torch.Tensor,
    edge_target: torch.Tensor,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    config: VariableVAELossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute masked reconstruction loss for variable topology graphs.

    Only computes loss on real nodes/edges (where mask=1).

    Args:
        node_pred: Predicted node features, shape (batch, max_nodes, 13).
        node_target: Target node features, shape (batch, max_nodes, 13).
        edge_pred: Predicted edge features, shape (batch, max_edges, 2).
        edge_target: Target edge features, shape (batch, max_edges, 2).
        node_mask: Node existence mask, shape (batch, max_nodes). 1=real, 0=padding.
        edge_mask: Edge existence mask, shape (batch, max_edges).
        config: Loss configuration.

    Returns:
        total_loss: Masked reconstruction loss.
        loss_dict: Component losses for logging.
    """
    if config is None:
        config = VariableVAELossConfig()

    # Expand masks for broadcasting
    node_mask_exp = node_mask.unsqueeze(-1)  # (batch, max_nodes, 1)
    edge_mask_exp = edge_mask.unsqueeze(-1)  # (batch, max_edges, 1)

    # Node feature losses (masked)
    # 13D features: [area, dir_xyz, centroid_xyz, curv1, curv2, bbox_d, bbox_cxyz]
    node_diff = (node_pred - node_target) ** 2

    # Per-feature losses (13D)
    area_diff = node_diff[..., 0:1]
    direction_diff = node_diff[..., 1:4]
    centroid_diff = node_diff[..., 4:7]
    curvature_diff = node_diff[..., 7:9]
    bbox_diff = node_diff[..., 9:13]  # bbox_diagonal + bbox_center_xyz

    # Masked mean (sum of masked values / number of real nodes)
    num_real_nodes = node_mask.sum().clamp(min=1)
    num_real_edges = edge_mask.sum().clamp(min=1)

    area_loss = (area_diff * node_mask_exp).sum() / num_real_nodes
    direction_loss = (direction_diff * node_mask_exp).sum() / (num_real_nodes * 3)
    centroid_loss = (centroid_diff * node_mask_exp).sum() / (num_real_nodes * 3)
    curvature_loss = (curvature_diff * node_mask_exp).sum() / (num_real_nodes * 2)
    bbox_loss = (bbox_diff * node_mask_exp).sum() / (num_real_nodes * 4)

    # Weighted node loss
    node_loss = (
        config.area_weight * area_loss
        + config.direction_weight * direction_loss
        + config.centroid_weight * centroid_loss
        + config.curvature_weight * curvature_loss
        + config.bbox_weight * bbox_loss
    )

    # Edge loss (masked)
    edge_diff = (edge_pred - edge_target) ** 2
    edge_loss = (edge_diff * edge_mask_exp).sum() / (num_real_edges * 2)

    # Combined loss
    total_loss = config.node_weight * node_loss + config.edge_weight * edge_loss

    loss_dict = {
        "node_loss": node_loss.detach(),
        "edge_loss": edge_loss.detach(),
        "area_loss": area_loss.detach(),
        "direction_loss": direction_loss.detach(),
        "centroid_loss": centroid_loss.detach(),
        "curvature_loss": curvature_loss.detach(),
        "bbox_loss": bbox_loss.detach(),
    }

    return total_loss, loss_dict


def mask_prediction_loss(
    node_mask_logits: torch.Tensor,
    edge_mask_logits: torch.Tensor,
    node_mask_target: torch.Tensor,
    edge_mask_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute BCE loss for node/edge existence prediction.

    Args:
        node_mask_logits: Predicted node existence logits, shape (batch, max_nodes).
        edge_mask_logits: Predicted edge existence logits, shape (batch, max_edges).
        node_mask_target: Target node mask, shape (batch, max_nodes). 1=exists, 0=padding.
        edge_mask_target: Target edge mask, shape (batch, max_edges).

    Returns:
        node_mask_loss: BCE for nodes.
        edge_mask_loss: BCE for edges.
        metrics: Accuracy metrics for logging.
    """
    # BCE loss
    node_mask_loss = F.binary_cross_entropy_with_logits(
        node_mask_logits, node_mask_target.float()
    )
    edge_mask_loss = F.binary_cross_entropy_with_logits(
        edge_mask_logits, edge_mask_target.float()
    )

    # Accuracy metrics
    with torch.no_grad():
        node_pred_binary = (torch.sigmoid(node_mask_logits) > 0.5).float()
        edge_pred_binary = (torch.sigmoid(edge_mask_logits) > 0.5).float()
        node_acc = (node_pred_binary == node_mask_target.float()).float().mean()
        edge_acc = (edge_pred_binary == edge_mask_target.float()).float().mean()

    metrics = {
        "node_mask_loss": node_mask_loss.detach(),
        "edge_mask_loss": edge_mask_loss.detach(),
        "node_mask_acc": node_acc,
        "edge_mask_acc": edge_acc,
    }

    return node_mask_loss, edge_mask_loss, metrics


def face_type_classification_loss(
    face_type_logits: torch.Tensor,
    face_type_target: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute cross-entropy loss for face type classification (masked).

    Only computes loss on real nodes (where mask=1).

    Args:
        face_type_logits: Predicted face type logits, shape (batch, max_nodes, num_types).
        face_type_target: Target face type indices, shape (batch, max_nodes).
        node_mask: Node existence mask, shape (batch, max_nodes).

    Returns:
        face_type_loss: Masked cross-entropy loss.
        metrics: Accuracy metrics for logging.
    """
    batch_size, max_nodes, num_types = face_type_logits.shape

    # Flatten for cross entropy
    logits_flat = face_type_logits.view(-1, num_types)  # (batch * max_nodes, num_types)
    target_flat = face_type_target.view(-1)  # (batch * max_nodes,)
    mask_flat = node_mask.view(-1)  # (batch * max_nodes,)

    # Per-element cross entropy (no reduction)
    ce_per_elem = F.cross_entropy(logits_flat, target_flat, reduction='none')

    # Masked mean
    num_real = mask_flat.sum().clamp(min=1)
    face_type_loss = (ce_per_elem * mask_flat).sum() / num_real

    # Accuracy (on real nodes only)
    with torch.no_grad():
        pred_types = logits_flat.argmax(dim=-1)
        correct = (pred_types == target_flat).float()
        face_type_acc = (correct * mask_flat).sum() / num_real

    metrics = {
        "face_type_loss": face_type_loss.detach(),
        "face_type_acc": face_type_acc,
    }

    return face_type_loss, metrics


def variable_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    beta: float = 0.01,
    config: VariableVAELossConfig | None = None,
    free_bits: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined loss for Variable Topology VAE.

    Loss = Reconstruction + beta * KL + mask_weight * MaskLoss + face_type_weight * FaceTypeLoss

    Args:
        outputs: Model outputs dict containing:
            - node_features: (batch, max_nodes, 9)
            - edge_features: (batch, max_edges, 2)
            - node_mask_logits: (batch, max_nodes)
            - edge_mask_logits: (batch, max_edges)
            - face_type_logits: (batch, max_nodes, num_types)
            - mu, logvar: latent distribution params
        targets: Target dict containing:
            - node_features: (batch, max_nodes, 9)
            - edge_features: (batch, max_edges, 2)
            - node_mask: (batch, max_nodes)
            - edge_mask: (batch, max_edges)
            - face_types: (batch, max_nodes)
        beta: KL divergence weight.
        config: Loss configuration.
        free_bits: Minimum KL per dimension.

    Returns:
        total_loss: Combined loss.
        loss_dict: All components for logging.
    """
    if config is None:
        config = VariableVAELossConfig()

    # Extract tensors
    node_pred = outputs["node_features"]
    edge_pred = outputs["edge_features"]
    node_mask_logits = outputs["node_mask_logits"]
    edge_mask_logits = outputs["edge_mask_logits"]
    face_type_logits = outputs["face_type_logits"]
    mu = outputs["mu"]
    logvar = outputs["logvar"]

    node_target = targets["node_features"]
    edge_target = targets["edge_features"]
    node_mask = targets["node_mask"]
    edge_mask = targets["edge_mask"]
    face_types = targets["face_types"]

    # 1. Masked reconstruction loss
    recon_loss, recon_metrics = variable_reconstruction_loss(
        node_pred, node_target, edge_pred, edge_target,
        node_mask, edge_mask, config
    )

    # 2. KL divergence
    kl_loss = kl_divergence(mu, logvar, free_bits)

    # 3. Mask prediction loss
    node_mask_loss, edge_mask_loss, mask_metrics = mask_prediction_loss(
        node_mask_logits, edge_mask_logits, node_mask, edge_mask
    )
    mask_loss = (
        config.node_mask_weight * node_mask_loss
        + config.edge_mask_weight * edge_mask_loss
    )

    # 4. Face type classification loss
    face_type_loss, face_type_metrics = face_type_classification_loss(
        face_type_logits, face_types, node_mask
    )

    # Combined loss
    total_loss = (
        recon_loss
        + beta * kl_loss
        + mask_loss
        + config.face_type_weight * face_type_loss
    )

    # Aggregate all metrics
    loss_dict = {
        **recon_metrics,
        **mask_metrics,
        **face_type_metrics,
        "recon_loss": recon_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "mask_loss": mask_loss.detach(),
        "beta": torch.tensor(beta),
        "total_loss": total_loss.detach(),
    }

    return total_loss, loss_dict


def variable_vae_loss_with_aux(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    param_target: torch.Tensor,
    beta: float = 0.01,
    aux_weight: float = 0.1,
    config: VariableVAELossConfig | None = None,
    free_bits: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Variable Topology VAE loss with auxiliary parameter prediction.

    Args:
        outputs: Model outputs (must include 'param_pred' if aux_weight > 0).
        targets: Target dict.
        param_target: Ground truth normalized parameters, shape (batch, num_params).
        beta: KL weight.
        aux_weight: Auxiliary parameter loss weight.
        config: Loss configuration.
        free_bits: Minimum KL per dimension.

    Returns:
        total_loss: Combined loss.
        loss_dict: All components for logging.
    """
    # Base variable VAE loss
    base_loss, loss_dict = variable_vae_loss(
        outputs, targets, beta, config, free_bits
    )

    # Auxiliary parameter loss
    if "param_pred" in outputs and aux_weight > 0:
        aux_loss = F.mse_loss(outputs["param_pred"], param_target)
        total_loss = base_loss + aux_weight * aux_loss
        loss_dict["aux_param_loss"] = aux_loss.detach()
        loss_dict["aux_weight"] = torch.tensor(aux_weight)
    else:
        total_loss = base_loss

    loss_dict["total_loss"] = total_loss.detach()

    return total_loss, loss_dict
