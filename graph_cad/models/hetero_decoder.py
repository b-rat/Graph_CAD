"""
Heterogeneous Graph Decoder for Phase 4 Multi-Geometry reconstruction.

This module provides a decoder that outputs face-level reconstructions
while being topology-aware. It wraps the existing TransformerGraphDecoder
and adds geometry-type-specific adaptations.

For Phase 4, face-level reconstruction is sufficient for the parameter
prediction task, as the encoder already extracts rich geometry information
from the full V/E/F B-Rep representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_cad.data.brep_types import (
    NUM_FACE_TYPES,
    NUM_GEOMETRY_TYPES,
    FACE_FEATURE_DIM,
    MAX_PARAMS,
)


@dataclass
class HeteroDecoderConfig:
    """Configuration for Hetero Graph Decoder."""

    latent_dim: int = 32
    face_features: int = FACE_FEATURE_DIM  # 13
    num_face_types: int = NUM_FACE_TYPES   # 3
    num_geometry_types: int = NUM_GEOMETRY_TYPES  # 6
    max_faces: int = 20  # Maximum number of faces to predict

    # Transformer architecture
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1


class HeteroGraphDecoder(nn.Module):
    """
    Face-level decoder with geometry type awareness.

    Wraps the TransformerGraphDecoder and adds:
    - Geometry type embedding that conditions the decoding
    - Geometry type classification head
    - Geometry-specific parameter routing

    The decoder produces face-level outputs that can be compared
    with ground truth using Hungarian matching.
    """

    def __init__(self, config: HeteroDecoderConfig | None = None):
        super().__init__()
        self.config = config or HeteroDecoderConfig()

        # Import base decoder
        from graph_cad.models.transformer_decoder import (
            TransformerGraphDecoder,
            TransformerDecoderConfig,
        )

        # Create base transformer decoder
        base_config = TransformerDecoderConfig(
            latent_dim=self.config.latent_dim,
            node_features=self.config.face_features,
            num_face_types=self.config.num_face_types,
            max_nodes=self.config.max_faces,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.base_decoder = TransformerGraphDecoder(base_config)

        # Geometry type classification head
        # Predicts which geometry type this latent represents
        self.geometry_type_head = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_geometry_types),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode latent to face-level graph and geometry type.

        Args:
            z: Latent vector, shape (batch, latent_dim)

        Returns:
            Dict containing:
                - node_features: (batch, max_faces, face_features)
                - face_type_logits: (batch, max_faces, num_face_types)
                - existence_logits: (batch, max_faces)
                - edge_logits: (batch, max_faces, max_faces)
                - geometry_type_logits: (batch, num_geometry_types)
                - node_embeddings: (batch, max_faces, hidden_dim)
        """
        # Base decoder outputs
        outputs = self.base_decoder(z)

        # Add geometry type classification
        outputs['geometry_type_logits'] = self.geometry_type_head(z)

        return outputs


class GeometryAwareParamHead(nn.Module):
    """
    Geometry-aware parameter prediction head.

    Uses separate parameter heads for each geometry type and routes
    predictions based on either ground truth or predicted geometry type.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        max_params: int = MAX_PARAMS,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_params = max_params

        # Per-type parameter heads
        # Each head outputs max_params values, unused params are masked
        self.param_heads = nn.ModuleDict({
            'bracket': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),  # leg1, leg2, width, thickness
            ),
            'tube': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),  # length, outer_dia, inner_dia
            ),
            'channel': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),  # width, height, length, thickness
            ),
            'block': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),  # length, width, height
            ),
            'cylinder': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # length, diameter
            ),
            'blockhole': nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 6),  # length, width, height, hole_dia, hole_x, hole_y
            ),
        })

        # Geometry type names in order
        self.type_names = ['bracket', 'tube', 'channel', 'block', 'cylinder', 'blockhole']

    def forward(
        self,
        z: torch.Tensor,
        geometry_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict parameters from latent vector.

        Args:
            z: Latent vector, shape (batch, latent_dim)
            geometry_types: Ground truth geometry type IDs, shape (batch,)
                If None, predictions are made for all types and returned stacked.

        Returns:
            If geometry_types is provided:
                params: Predicted parameters (padded), shape (batch, max_params)
                mask: Parameter mask, shape (batch, max_params)
            If geometry_types is None:
                all_params: Shape (batch, num_types, max_params)
                all_masks: Shape (batch, num_types, max_params)
        """
        batch_size = z.shape[0]
        device = z.device
        dtype = z.dtype

        if geometry_types is not None:
            # Route to correct head per sample
            params = torch.zeros(batch_size, self.max_params, device=device, dtype=dtype)
            masks = torch.zeros(batch_size, self.max_params, device=device, dtype=dtype)

            for geo_type, name in enumerate(self.type_names):
                # Find samples of this type
                type_mask = (geometry_types == geo_type)
                if not type_mask.any():
                    continue

                # Predict parameters
                z_type = z[type_mask]
                pred = self.param_heads[name](z_type)
                num_params = pred.shape[1]

                # Fill in padded tensor
                params[type_mask, :num_params] = pred
                masks[type_mask, :num_params] = 1.0

            return params, masks
        else:
            # Return predictions for all types
            all_params = torch.zeros(
                batch_size, len(self.type_names), self.max_params,
                device=device, dtype=dtype
            )
            all_masks = torch.zeros(
                batch_size, len(self.type_names), self.max_params,
                device=device, dtype=dtype
            )

            for geo_type, name in enumerate(self.type_names):
                pred = self.param_heads[name](z)
                num_params = pred.shape[1]
                all_params[:, geo_type, :num_params] = pred
                all_masks[:, geo_type, :num_params] = 1.0

            return all_params, all_masks


class MultiGeometryDecoder(nn.Module):
    """
    Complete decoder for multi-geometry VAE.

    Combines:
    - HeteroGraphDecoder for face-level reconstruction + geometry classification
    - GeometryAwareParamHead for per-type parameter prediction
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        param_hidden_dim: int = 64,
    ):
        super().__init__()

        config = HeteroDecoderConfig(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.graph_decoder = HeteroGraphDecoder(config)
        self.param_head = GeometryAwareParamHead(
            latent_dim=latent_dim,
            hidden_dim=param_hidden_dim,
        )

    def forward(
        self,
        z: torch.Tensor,
        geometry_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Decode latent to graph reconstruction and parameters.

        Args:
            z: Latent vector, shape (batch, latent_dim)
            geometry_types: Optional ground truth geometry types for parameter routing

        Returns:
            Dict with all decoder outputs including:
                - Graph reconstruction (face features, types, edges)
                - Geometry type logits
                - Parameter predictions (padded)
                - Parameter masks
        """
        # Graph reconstruction
        outputs = self.graph_decoder(z)

        # Parameter prediction
        if geometry_types is not None:
            params, param_mask = self.param_head(z, geometry_types)
            outputs['param_pred'] = params
            outputs['param_mask'] = param_mask
        else:
            # During inference, use predicted geometry type
            pred_type = outputs['geometry_type_logits'].argmax(dim=-1)
            params, param_mask = self.param_head(z, pred_type)
            outputs['param_pred'] = params
            outputs['param_mask'] = param_mask

        return outputs


def compute_geometry_type_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Compute cross-entropy loss for geometry type classification.

    Args:
        logits: Predicted geometry type logits, shape (batch, num_types)
        targets: Ground truth geometry type IDs, shape (batch,)

    Returns:
        loss: Cross-entropy loss
        accuracy: Classification accuracy
    """
    loss = F.cross_entropy(logits, targets.long())

    with torch.no_grad():
        pred_types = logits.argmax(dim=-1)
        accuracy = (pred_types == targets).float().mean().item()

    return loss, accuracy


def compute_param_loss(
    pred_params: torch.Tensor,
    target_params: torch.Tensor,
    param_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked MSE loss for parameter prediction.

    Args:
        pred_params: Predicted parameters (padded), shape (batch, max_params)
        target_params: Target parameters (padded), shape (batch, max_params)
        param_mask: Parameter mask, shape (batch, max_params)

    Returns:
        Masked MSE loss
    """
    # Compute squared error
    se = (pred_params - target_params) ** 2

    # Apply mask and compute mean
    num_valid = param_mask.sum().clamp(min=1)
    loss = (se * param_mask).sum() / num_valid

    return loss
