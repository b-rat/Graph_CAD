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
    exclude_dims: int = 0,
) -> torch.Tensor:
    """
    Compute KL divergence KL(q(z|x) || p(z)) where p(z) = N(0, I).

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu: Latent mean, shape (batch, latent_dim).
        logvar: Latent log variance, shape (batch, latent_dim).
        free_bits: Minimum KL per dimension to prevent posterior collapse.
                   If > 0, dimensions with KL < free_bits contribute 0.
        exclude_dims: Number of first dimensions to exclude from KL computation.
                      Used with direct latent supervision to let those dims
                      encode parameters freely without N(0,1) constraint.

    Returns:
        kl_loss: Scalar KL divergence (averaged over batch).
    """
    # Per-dimension KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if free_bits > 0:
        # Free bits: don't penalize dimensions below threshold
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits) - free_bits

    # Exclude first N dimensions from KL if specified
    if exclude_dims > 0:
        kl_per_dim = kl_per_dim[:, exclude_dims:]

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
    beta: float = 0.1,
    config: VariableVAELossConfig | None = None,
    free_bits: float = 0.5,
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
    beta: float = 0.1,
    aux_weight: float = 0.1,
    config: VariableVAELossConfig | None = None,
    free_bits: float = 0.5,
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


# =============================================================================
# Hungarian Matching Loss (Phase 3 - Transformer Decoder)
# =============================================================================


@dataclass
class HungarianLossConfig:
    """Configuration for Hungarian matching loss."""

    # Cost matrix weights (for matching)
    node_feature_cost_weight: float = 1.0
    face_type_cost_weight: float = 2.0  # Higher weight for topology correctness
    existence_cost_weight: float = 1.0

    # Loss weights (after matching)
    node_feature_loss_weight: float = 1.0
    face_type_loss_weight: float = 2.0
    existence_loss_weight: float = 1.0
    edge_loss_weight: float = 1.0

    # Edge class imbalance handling
    edge_positive_weight: float = 3.0  # Weight for positive edges (sparse)


@torch.no_grad()
def compute_hungarian_matching(
    pred_features: torch.Tensor,
    pred_face_types: torch.Tensor,
    pred_existence: torch.Tensor,
    target_features: torch.Tensor,
    target_face_types: torch.Tensor,
    target_mask: torch.Tensor,
    config: HungarianLossConfig | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute optimal Hungarian matching between predictions and targets.

    For each sample in the batch, finds the optimal assignment of predicted
    nodes to ground truth nodes that minimizes the total cost.

    Args:
        pred_features: Predicted node features, shape (batch, max_nodes, node_features)
        pred_face_types: Predicted face type logits, shape (batch, max_nodes, num_types)
        pred_existence: Predicted existence logits, shape (batch, max_nodes)
        target_features: Target node features, shape (batch, max_nodes, node_features)
        target_face_types: Target face type indices, shape (batch, max_nodes)
        target_mask: Target node mask (1=real, 0=padding), shape (batch, max_nodes)
        config: Loss configuration

    Returns:
        List of (pred_indices, target_indices) tuples, one per batch sample.
        Each tuple contains tensors of matched indices.
    """
    from scipy.optimize import linear_sum_assignment

    if config is None:
        config = HungarianLossConfig()

    batch_size = pred_features.shape[0]
    max_nodes = pred_features.shape[1]
    device = pred_features.device

    matchings = []

    for b in range(batch_size):
        # Get number of real nodes for this sample
        num_real = int(target_mask[b].sum().item())

        if num_real == 0:
            # No real nodes - return empty matching
            matchings.append((
                torch.tensor([], dtype=torch.long, device=device),
                torch.tensor([], dtype=torch.long, device=device)
            ))
            continue

        # Compute cost matrix: (max_nodes predictions) x (num_real targets)
        # Cost = MSE(features) + CE(face_type) + BCE(existence)

        # Node feature cost: L2 distance
        # pred: (max_nodes, features), target: (num_real, features)
        pred_feat_b = pred_features[b]  # (max_nodes, features)
        target_feat_b = target_features[b, :num_real]  # (num_real, features)

        # Pairwise L2 distance: (max_nodes, num_real)
        feat_cost = torch.cdist(pred_feat_b, target_feat_b, p=2)

        # Face type cost: negative log probability of correct class
        # pred_face_types[b]: (max_nodes, num_types) logits
        # target_face_types[b, :num_real]: (num_real,) indices
        pred_probs = F.softmax(pred_face_types[b], dim=-1)  # (max_nodes, num_types)
        target_types_b = target_face_types[b, :num_real]  # (num_real,)

        # For each prediction i and target j, cost = -log(pred_probs[i, target_types[j]])
        face_type_cost = torch.zeros(max_nodes, num_real, device=device)
        for j in range(num_real):
            target_type = target_types_b[j].long()
            face_type_cost[:, j] = -torch.log(pred_probs[:, target_type] + 1e-8)

        # Existence cost: BCE between predicted existence and 1 (real nodes)
        # For matching, we want predictions with high existence prob to match real nodes
        pred_exist_prob = torch.sigmoid(pred_existence[b])  # (max_nodes,)
        # Cost = -log(prob) for matching to real node (want high prob)
        exist_cost = -torch.log(pred_exist_prob + 1e-8).unsqueeze(1)  # (max_nodes, 1)
        exist_cost = exist_cost.expand(-1, num_real)  # (max_nodes, num_real)

        # Combined cost
        total_cost = (
            config.node_feature_cost_weight * feat_cost
            + config.face_type_cost_weight * face_type_cost
            + config.existence_cost_weight * exist_cost
        )

        # Hungarian algorithm (on CPU, as scipy doesn't support GPU)
        cost_np = total_cost.cpu().numpy()
        pred_idx, target_idx = linear_sum_assignment(cost_np)

        matchings.append((
            torch.tensor(pred_idx, dtype=torch.long, device=device),
            torch.tensor(target_idx, dtype=torch.long, device=device)
        ))

    return matchings


def hungarian_node_loss(
    pred_features: torch.Tensor,
    pred_face_types: torch.Tensor,
    pred_existence: torch.Tensor,
    target_features: torch.Tensor,
    target_face_types: torch.Tensor,
    target_mask: torch.Tensor,
    matchings: list[tuple[torch.Tensor, torch.Tensor]],
    config: HungarianLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute node-level losses using Hungarian matching.

    Args:
        pred_features: Predicted node features, shape (batch, max_nodes, node_features)
        pred_face_types: Predicted face type logits, shape (batch, max_nodes, num_types)
        pred_existence: Predicted existence logits, shape (batch, max_nodes)
        target_features: Target node features, shape (batch, max_nodes, node_features)
        target_face_types: Target face type indices, shape (batch, max_nodes)
        target_mask: Target node mask, shape (batch, max_nodes)
        matchings: List of (pred_indices, target_indices) from compute_hungarian_matching
        config: Loss configuration

    Returns:
        total_loss: Combined node loss
        loss_dict: Component losses for logging
    """
    if config is None:
        config = HungarianLossConfig()

    batch_size = pred_features.shape[0]
    max_nodes = pred_features.shape[1]
    device = pred_features.device

    # Accumulate losses
    total_feat_loss = torch.tensor(0.0, device=device)
    total_type_loss = torch.tensor(0.0, device=device)
    total_exist_loss = torch.tensor(0.0, device=device)
    total_matched = 0

    for b, (pred_idx, target_idx) in enumerate(matchings):
        num_matched = len(pred_idx)
        if num_matched == 0:
            continue

        # Node feature loss (MSE on matched pairs)
        matched_pred_feat = pred_features[b, pred_idx]  # (num_matched, features)
        matched_target_feat = target_features[b, target_idx]  # (num_matched, features)
        feat_loss = F.mse_loss(matched_pred_feat, matched_target_feat, reduction='sum')
        total_feat_loss = total_feat_loss + feat_loss

        # Face type loss (CE on matched pairs)
        matched_pred_types = pred_face_types[b, pred_idx]  # (num_matched, num_types)
        matched_target_types = target_face_types[b, target_idx]  # (num_matched,)
        type_loss = F.cross_entropy(
            matched_pred_types, matched_target_types.long(), reduction='sum'
        )
        total_type_loss = total_type_loss + type_loss

        total_matched += num_matched

    # Existence loss: all predictions vs target mask
    # Matched predictions should have high existence, unmatched should have low
    exist_target = torch.zeros_like(pred_existence)
    for b, (pred_idx, _) in enumerate(matchings):
        if len(pred_idx) > 0:
            exist_target[b, pred_idx] = 1.0

    total_exist_loss = F.binary_cross_entropy_with_logits(
        pred_existence, exist_target, reduction='mean'
    )

    # Normalize by number of matched nodes
    if total_matched > 0:
        total_feat_loss = total_feat_loss / total_matched
        total_type_loss = total_type_loss / total_matched

    # Combined loss
    total_loss = (
        config.node_feature_loss_weight * total_feat_loss
        + config.face_type_loss_weight * total_type_loss
        + config.existence_loss_weight * total_exist_loss
    )

    # Compute accuracy metrics
    with torch.no_grad():
        # Face type accuracy on matched nodes
        correct_types = 0
        for b, (pred_idx, target_idx) in enumerate(matchings):
            if len(pred_idx) > 0:
                pred_types = pred_face_types[b, pred_idx].argmax(dim=-1)
                target_types = target_face_types[b, target_idx]
                correct_types += (pred_types == target_types).sum().item()
        face_type_acc = correct_types / max(total_matched, 1)

        # Existence accuracy
        exist_pred = (torch.sigmoid(pred_existence) > 0.5).float()
        exist_acc = (exist_pred == exist_target).float().mean().item()

    loss_dict = {
        "node_feature_loss": total_feat_loss.detach(),
        "face_type_loss": total_type_loss.detach(),
        "existence_loss": total_exist_loss.detach(),
        "face_type_acc": torch.tensor(face_type_acc),
        "existence_acc": torch.tensor(exist_acc),
        "num_matched": torch.tensor(total_matched),
    }

    return total_loss, loss_dict


def hungarian_edge_loss_with_adj(
    pred_edge_logits: torch.Tensor,
    target_adj: torch.Tensor,
    target_mask: torch.Tensor,
    matchings: list[tuple[torch.Tensor, torch.Tensor]],
    config: HungarianLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute edge loss using Hungarian matching with target adjacency matrix.

    Args:
        pred_edge_logits: Predicted edge logits, shape (batch, max_nodes, max_nodes)
        target_adj: Target adjacency matrix, shape (batch, max_nodes, max_nodes)
        target_mask: Target node mask, shape (batch, max_nodes)
        matchings: List of (pred_indices, target_indices) from compute_hungarian_matching
        config: Loss configuration

    Returns:
        edge_loss: Binary cross-entropy loss for edges
        loss_dict: Metrics for logging
    """
    if config is None:
        config = HungarianLossConfig()

    batch_size = pred_edge_logits.shape[0]
    max_nodes = pred_edge_logits.shape[1]
    device = pred_edge_logits.device

    # Build matched target adjacency and validity mask
    matched_target_adj = torch.zeros_like(pred_edge_logits)
    valid_mask = torch.zeros_like(pred_edge_logits)

    for b, (pred_idx, target_idx) in enumerate(matchings):
        num_matched = len(pred_idx)
        if num_matched == 0:
            continue

        # For each pair of matched nodes, check if they're adjacent in target
        for i in range(num_matched):
            for j in range(i + 1, num_matched):
                ti, tj = int(target_idx[i]), int(target_idx[j])
                pi, pj = int(pred_idx[i]), int(pred_idx[j])

                # Mark this pair as valid (both nodes matched)
                valid_mask[b, pi, pj] = 1.0
                valid_mask[b, pj, pi] = 1.0

                # Copy adjacency from target
                if target_adj[b, ti, tj] > 0.5:
                    matched_target_adj[b, pi, pj] = 1.0
                    matched_target_adj[b, pj, pi] = 1.0

    # Compute loss only on valid pairs
    num_valid = valid_mask.sum()

    if num_valid > 0:
        # Get valid predictions and targets
        valid_pred = pred_edge_logits[valid_mask.bool()]
        valid_target = matched_target_adj[valid_mask.bool()]

        # BCE with positive weighting for class imbalance
        num_pos = valid_target.sum()
        num_neg = num_valid - num_pos
        if num_pos > 0 and num_neg > 0:
            pos_weight = torch.tensor([num_neg / num_pos], device=device)
            pos_weight = torch.clamp(pos_weight, max=config.edge_positive_weight)
        else:
            pos_weight = torch.tensor([config.edge_positive_weight], device=device)

        edge_loss = F.binary_cross_entropy_with_logits(
            valid_pred, valid_target, pos_weight=pos_weight
        )
    else:
        edge_loss = torch.tensor(0.0, device=device)

    # Metrics
    with torch.no_grad():
        if num_valid > 0:
            pred_edges = (torch.sigmoid(pred_edge_logits[valid_mask.bool()]) > 0.5).float()
            valid_target = matched_target_adj[valid_mask.bool()]
            edge_acc = (pred_edges == valid_target).float().mean().item()

            # Precision/recall for edges
            true_pos = ((pred_edges == 1) & (valid_target == 1)).sum().item()
            pred_pos = (pred_edges == 1).sum().item()
            actual_pos = (valid_target == 1).sum().item()

            edge_precision = true_pos / max(pred_pos, 1)
            edge_recall = true_pos / max(actual_pos, 1)
        else:
            edge_acc = 0.0
            edge_precision = 0.0
            edge_recall = 0.0

    loss_dict = {
        "edge_loss": edge_loss.detach(),
        "edge_acc": torch.tensor(edge_acc),
        "edge_precision": torch.tensor(edge_precision),
        "edge_recall": torch.tensor(edge_recall),
        "num_valid_edge_pairs": num_valid.detach(),
    }

    return edge_loss, loss_dict


def transformer_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    beta: float = 0.1,
    config: HungarianLossConfig | None = None,
    free_bits: float = 0.5,
    kl_exclude_dims: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined loss for Transformer Graph VAE with Hungarian matching.

    Loss = NodeLoss + EdgeLoss + beta * KL

    Where NodeLoss and EdgeLoss use Hungarian matching for permutation invariance.

    Args:
        outputs: Model outputs dict containing:
            - node_features: (batch, max_nodes, 13)
            - face_type_logits: (batch, max_nodes, num_types)
            - existence_logits: (batch, max_nodes)
            - edge_logits: (batch, max_nodes, max_nodes)
            - mu, logvar: latent distribution params
        targets: Target dict containing:
            - node_features: (batch, max_nodes, 13)
            - face_types: (batch, max_nodes)
            - node_mask: (batch, max_nodes)
            - adj_matrix: (batch, max_nodes, max_nodes)
        beta: KL divergence weight
        config: Loss configuration
        free_bits: Minimum KL per dimension
        kl_exclude_dims: Number of first latent dims to exclude from KL.
                         Used with direct latent supervision.

    Returns:
        total_loss: Combined loss
        loss_dict: All components for logging
    """
    if config is None:
        config = HungarianLossConfig()

    # Extract tensors
    pred_features = outputs["node_features"]
    pred_face_types = outputs["face_type_logits"]
    pred_existence = outputs["existence_logits"]
    pred_edges = outputs["edge_logits"]
    mu = outputs["mu"]
    logvar = outputs["logvar"]

    target_features = targets["node_features"]
    target_face_types = targets["face_types"]
    target_mask = targets["node_mask"]
    target_adj = targets["adj_matrix"]

    # 1. Compute Hungarian matching
    matchings = compute_hungarian_matching(
        pred_features, pred_face_types, pred_existence,
        target_features, target_face_types, target_mask,
        config
    )

    # 2. Node losses with matching
    node_loss, node_metrics = hungarian_node_loss(
        pred_features, pred_face_types, pred_existence,
        target_features, target_face_types, target_mask,
        matchings, config
    )

    # 3. Edge loss with matching
    edge_loss, edge_metrics = hungarian_edge_loss_with_adj(
        pred_edges, target_adj, target_mask,
        matchings, config
    )

    # 4. KL divergence (optionally excluding first N dims for direct supervision)
    kl_loss = kl_divergence(mu, logvar, free_bits, exclude_dims=kl_exclude_dims)

    # Combined loss
    total_loss = (
        node_loss
        + config.edge_loss_weight * edge_loss
        + beta * kl_loss
    )

    # Aggregate metrics
    loss_dict = {
        **node_metrics,
        **edge_metrics,
        "node_loss": node_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "kl_exclude_dims": torch.tensor(kl_exclude_dims),
        "beta": torch.tensor(beta),
        "total_loss": total_loss.detach(),
    }

    return total_loss, loss_dict


# L-bracket parameter ranges for normalization
# [leg1, leg2, width, thickness] in mm
PARAM_MINS = torch.tensor([50.0, 50.0, 20.0, 3.0])
PARAM_MAXS = torch.tensor([200.0, 200.0, 60.0, 12.0])
PARAM_RANGES = PARAM_MAXS - PARAM_MINS  # [150, 150, 40, 9]


def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute negative correlation loss for parameter prediction.

    Minimizing this loss maximizes the correlation between predictions
    and targets. Unlike MSE, this is scale/offset invariant and directly
    optimizes the metric we care about (correlation).

    Args:
        pred: Predicted parameters, shape (batch, num_params)
        target: Target parameters, shape (batch, num_params)

    Returns:
        Negative mean correlation (scalar). Lower = better correlation.
        Range: [-1, 1] where -1 means perfect correlation.
    """
    # Center the predictions and targets (subtract mean)
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    # Compute correlation per parameter using covariance formula
    # corr = cov(pred, target) / (std(pred) * std(target))
    # cov = mean((pred - mean_pred) * (target - mean_target))
    covariance = (pred_centered * target_centered).mean(dim=0)

    # Use unbiased=False for consistency with covariance calculation
    pred_std = pred_centered.std(dim=0, unbiased=False) + 1e-8
    target_std = target_centered.std(dim=0, unbiased=False) + 1e-8

    # Correlation per parameter
    correlations = covariance / (pred_std * target_std)

    # Return negative mean correlation (we minimize loss, so negative = maximize correlation)
    return -correlations.mean()


def normalize_params(params: torch.Tensor) -> torch.Tensor:
    """
    Normalize L-bracket parameters to [0, 1] range.

    This ensures all parameters contribute equally to the aux loss,
    regardless of their physical scale.

    Args:
        params: Raw parameters, shape (batch, 4) in mm
                [leg1, leg2, width, thickness]

    Returns:
        Normalized parameters in [0, 1] range
    """
    device = params.device
    mins = PARAM_MINS.to(device)
    ranges = PARAM_RANGES.to(device)
    return (params - mins) / ranges


def normalize_params_for_latent(params: torch.Tensor) -> torch.Tensor:
    """
    Normalize L-bracket parameters to latent-compatible range [-2, 2].

    For direct latent supervision, we want parameters in a range similar
    to the VAE latent space (approximately standard normal). This maps
    the parameter ranges to [-2, 2] which covers ~95% of N(0,1).

    Args:
        params: Normalized parameters in [0, 1] range, shape (batch, 4)
                [leg1, leg2, width, thickness]
                (Note: The dataset returns params already normalized to [0, 1])

    Returns:
        Parameters scaled to [-2, 2] range for latent supervision
    """
    # Dataset returns params already normalized to [0, 1]
    # Scale to [-2, 2] for latent space compatibility
    return params * 4.0 - 2.0


def denormalize_params(params_norm: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized parameters back to physical units (mm).

    Args:
        params_norm: Normalized parameters in [0, 1] range

    Returns:
        Parameters in mm
    """
    device = params_norm.device
    mins = PARAM_MINS.to(device)
    ranges = PARAM_RANGES.to(device)
    return params_norm * ranges + mins


def transformer_vae_loss_with_aux(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    param_target: torch.Tensor,
    beta: float = 0.1,
    aux_weight: float = 1.0,
    config: HungarianLossConfig | None = None,
    free_bits: float = 0.5,
    aux_loss_type: str = "correlation",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Transformer VAE loss with auxiliary parameter prediction.

    Loss = NodeLoss + EdgeLoss + beta * KL + aux_weight * AuxLoss

    The auxiliary loss forces the latent space to encode all L-bracket
    parameters (leg1, leg2, width, thickness), preventing geometric dominance
    where only large parameters (leg lengths) are encoded.

    Args:
        outputs: Model outputs dict containing:
            - node_features: (batch, max_nodes, 13)
            - face_type_logits: (batch, max_nodes, num_types)
            - existence_logits: (batch, max_nodes)
            - edge_logits: (batch, max_nodes, max_nodes)
            - mu, logvar: latent distribution params
            - param_pred: (batch, num_params) if aux_weight > 0 and not using direct
        targets: Target dict containing:
            - node_features: (batch, max_nodes, 13)
            - face_types: (batch, max_nodes)
            - node_mask: (batch, max_nodes)
            - adj_matrix: (batch, max_nodes, max_nodes)
        param_target: Ground truth parameters, shape (batch, num_params)
        beta: KL divergence weight
        aux_weight: Auxiliary parameter loss weight
        config: Loss configuration
        free_bits: Minimum KL per dimension
        aux_loss_type: Type of auxiliary loss:
            - "direct": Direct latent supervision - forces mu[:, :4] to encode
              normalized parameters. No param_head needed. KL is excluded from
              the first 4 dims so they can encode params freely.
            - "correlation": Negative correlation loss (scale/offset invariant)
            - "mse": Raw MSE loss
            - "mse_normalized": MSE on [0,1] normalized parameters

    Returns:
        total_loss: Combined loss
        loss_dict: All components for logging
    """
    # Determine KL exclusion for direct latent supervision
    num_params = param_target.shape[1]
    kl_exclude_dims = num_params if (aux_loss_type == "direct" and aux_weight > 0) else 0

    # Base transformer VAE loss (with KL exclusion if using direct supervision)
    base_loss, loss_dict = transformer_vae_loss(
        outputs, targets, beta, config, free_bits, kl_exclude_dims=kl_exclude_dims
    )

    # Auxiliary parameter loss
    if aux_loss_type == "direct" and aux_weight > 0:
        # Direct latent supervision: force first 4 dims of mu to encode parameters
        # This bypasses pooling issues by directly supervising the latent space
        mu = outputs["mu"]
        num_params = param_target.shape[1]
        mu_params = mu[:, :num_params]  # First num_params dimensions
        param_target_norm = normalize_params_for_latent(param_target)
        aux_loss = F.mse_loss(mu_params, param_target_norm)

        total_loss = base_loss + aux_weight * aux_loss
        loss_dict["aux_param_loss"] = aux_loss.detach()
        loss_dict["aux_weight"] = torch.tensor(aux_weight)
        loss_dict["direct_latent_supervision"] = torch.tensor(True)
    elif "param_pred" in outputs and aux_weight > 0:
        param_pred = outputs["param_pred"]

        if aux_loss_type == "correlation":
            # Correlation loss: scale/offset invariant, directly optimizes correlation
            aux_loss = correlation_loss(param_pred, param_target)
        elif aux_loss_type == "mse_normalized":
            # Normalize both predictions and targets to [0, 1]
            param_pred_norm = normalize_params(param_pred)
            param_target_norm = normalize_params(param_target)
            aux_loss = F.mse_loss(param_pred_norm, param_target_norm)
        else:  # "mse"
            # Raw MSE (biased toward larger parameters)
            aux_loss = F.mse_loss(param_pred, param_target)

        total_loss = base_loss + aux_weight * aux_loss
        loss_dict["aux_param_loss"] = aux_loss.detach()
        loss_dict["aux_weight"] = torch.tensor(aux_weight)
    else:
        total_loss = base_loss

    loss_dict["total_loss"] = total_loss.detach()

    return total_loss, loss_dict


# =============================================================================
# Multi-Geometry Loss Functions (Phase 4 - HeteroVAE)
# =============================================================================


@dataclass
class MultiGeometryLossConfig:
    """Configuration for multi-geometry VAE loss."""

    # Reconstruction weights (same as HungarianLossConfig)
    node_feature_loss_weight: float = 1.0
    face_type_loss_weight: float = 2.0
    existence_loss_weight: float = 1.0
    edge_loss_weight: float = 1.0

    # Geometry classification weight
    geometry_type_loss_weight: float = 2.0

    # Parameter prediction weight
    param_loss_weight: float = 1.0

    # Edge class imbalance
    edge_positive_weight: float = 3.0


def multi_geometry_vae_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    beta: float = 0.1,
    config: MultiGeometryLossConfig | None = None,
    free_bits: float = 0.5,
    kl_exclude_dims: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined loss for Multi-Geometry VAE with Hungarian matching.

    Loss = GraphReconLoss + GeometryTypeLoss + ParamLoss + beta * KL

    Args:
        outputs: Model outputs dict containing:
            - node_features: (batch, max_faces, face_features)
            - face_type_logits: (batch, max_faces, num_face_types)
            - existence_logits: (batch, max_faces)
            - edge_logits: (batch, max_faces, max_faces)
            - geometry_type_logits: (batch, num_geometry_types)
            - param_pred: (batch, max_params)
            - param_mask: (batch, max_params)
            - mu, logvar: latent distribution params
        targets: Target dict containing:
            - node_features: (batch, max_faces, face_features)
            - face_types: (batch, max_faces)
            - node_mask: (batch, max_faces)
            - adj_matrix: (batch, max_faces, max_faces)
            - geometry_type: (batch,)
            - params_normalized: (batch, max_params)
            - params_mask: (batch, max_params)
        beta: KL divergence weight
        config: Loss configuration
        free_bits: Minimum KL per dimension
        kl_exclude_dims: Number of latent dims to exclude from KL

    Returns:
        total_loss: Combined loss
        loss_dict: All components for logging
    """
    if config is None:
        config = MultiGeometryLossConfig()

    # Use existing Hungarian matching loss for graph reconstruction
    hungarian_config = HungarianLossConfig(
        node_feature_loss_weight=config.node_feature_loss_weight,
        face_type_loss_weight=config.face_type_loss_weight,
        existence_loss_weight=config.existence_loss_weight,
        edge_loss_weight=config.edge_loss_weight,
        edge_positive_weight=config.edge_positive_weight,
    )

    # Build targets dict for transformer_vae_loss
    recon_targets = {
        'node_features': targets['node_features'],
        'face_types': targets['face_types'],
        'node_mask': targets['node_mask'],
        'adj_matrix': targets['adj_matrix'],
    }

    # Graph reconstruction + KL loss
    graph_loss, loss_dict = transformer_vae_loss(
        outputs, recon_targets, beta, hungarian_config, free_bits, kl_exclude_dims
    )

    # Geometry type classification loss
    geo_type_loss = F.cross_entropy(
        outputs['geometry_type_logits'],
        targets['geometry_type'].squeeze(-1).long()
    )

    # Geometry type accuracy
    with torch.no_grad():
        pred_geo_type = outputs['geometry_type_logits'].argmax(dim=-1)
        geo_type_acc = (pred_geo_type == targets['geometry_type'].squeeze(-1)).float().mean()

    loss_dict['geometry_type_loss'] = geo_type_loss.detach()
    loss_dict['geometry_type_acc'] = geo_type_acc

    # Parameter prediction loss (masked)
    if 'param_pred' in outputs and 'params_normalized' in targets:
        param_pred = outputs['param_pred']
        param_target = targets['params_normalized']
        param_mask = targets['params_mask']

        # Masked MSE
        se = (param_pred - param_target) ** 2
        num_valid = param_mask.sum().clamp(min=1)
        param_loss = (se * param_mask).sum() / num_valid

        loss_dict['param_loss'] = param_loss.detach()

        # Per-parameter errors for analysis
        with torch.no_grad():
            abs_errors = (param_pred - param_target).abs()
            mean_abs_error = (abs_errors * param_mask).sum() / num_valid
            loss_dict['param_mae'] = mean_abs_error
    else:
        param_loss = torch.tensor(0.0, device=outputs['mu'].device)
        loss_dict['param_loss'] = param_loss

    # Combined loss
    total_loss = (
        graph_loss
        + config.geometry_type_loss_weight * geo_type_loss
        + config.param_loss_weight * param_loss
    )

    loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict


def multi_geometry_vae_loss_with_direct_latent(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    beta: float = 0.1,
    aux_weight: float = 1.0,
    config: MultiGeometryLossConfig | None = None,
    free_bits: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Multi-geometry VAE loss with direct latent supervision.

    Forces the first N dimensions of mu to encode normalized parameters,
    where N varies by geometry type. The remaining dimensions are free
    for reconstruction.

    This bypasses pooling issues by directly supervising the latent space.

    Args:
        outputs: Model outputs (must include 'mu')
        targets: Target dict (must include 'params_normalized', 'params_mask', 'geometry_type')
        beta: KL weight
        aux_weight: Direct supervision weight
        config: Loss configuration
        free_bits: Minimum KL per dimension

    Returns:
        total_loss: Combined loss
        loss_dict: All components for logging
    """
    if config is None:
        config = MultiGeometryLossConfig()

    # Get max params from targets mask
    max_used_params = int(targets['params_mask'].sum(dim=-1).max().item())

    # Exclude first max_params dims from KL
    kl_exclude_dims = max_used_params if aux_weight > 0 else 0

    # Base loss with KL exclusion
    total_loss, loss_dict = multi_geometry_vae_loss(
        outputs, targets, beta, config, free_bits, kl_exclude_dims
    )

    # Add direct latent supervision
    if aux_weight > 0:
        mu = outputs['mu']
        param_target = targets['params_normalized']
        param_mask = targets['params_mask']

        # Scale params to latent range [-2, 2]
        param_target_latent = param_target * 4.0 - 2.0

        # MSE on first N dims of mu (masked by param_mask)
        mu_params = mu[:, :param_target.shape[1]]
        se = (mu_params - param_target_latent) ** 2
        num_valid = param_mask.sum().clamp(min=1)
        direct_loss = (se * param_mask).sum() / num_valid

        total_loss = total_loss + aux_weight * direct_loss

        loss_dict['direct_latent_loss'] = direct_loss.detach()
        loss_dict['direct_latent_supervision'] = torch.tensor(True)
        loss_dict['kl_exclude_dims'] = torch.tensor(kl_exclude_dims)
        loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict