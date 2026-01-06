"""
Parameter VAE: Graph encoder with parameter decoder.

Instead of reconstructing graph features, this VAE decodes directly to
L-bracket parameters. This forces the latent space to encode parameters
explicitly, making edit directions meaningful.

Architecture:
    Graph → GNN Encoder → z (latent) → Parameter Decoder → [leg1, leg2, width, thickness, fillet, holes]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


@dataclass
class ParameterVAEConfig:
    """Configuration for Parameter VAE."""

    # Input dimensions (same as VariableGraphVAE)
    node_features: int = 9  # area, dir_xyz, centroid_xyz, curvatures
    edge_features: int = 2  # edge_length, dihedral_angle
    num_face_types: int = 3  # PLANAR=0, HOLE=1, FILLET=2
    face_embed_dim: int = 8  # Dimension of face type embeddings

    # Maximum sizes for padding (needed for batching)
    max_nodes: int = 20
    max_edges: int = 50

    # Encoder architecture (GNN)
    hidden_dim: int = 64
    num_gat_layers: int = 3
    num_heads: int = 4
    encoder_dropout: float = 0.1

    # Latent space
    latent_dim: int = 32

    # Parameter decoder architecture
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 3
    decoder_dropout: float = 0.1

    # Parameter output config
    max_holes_per_leg: int = 2  # Up to 2 holes per leg


class ParameterVAEEncoder(nn.Module):
    """
    GNN encoder for variable topology graphs.

    Same architecture as VariableGraphVAEEncoder - encodes graph to latent.
    """

    def __init__(self, config: ParameterVAEConfig):
        super().__init__()
        self.config = config

        # Face type embedding
        self.face_type_embedding = nn.Embedding(
            config.num_face_types,
            config.face_embed_dim
        )

        # Input dimension: continuous features + face embedding
        input_dim = config.node_features + config.face_embed_dim  # 9 + 8 = 17

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.encoder_dropout),
        )

        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.edge_features, config.hidden_dim),
            nn.ReLU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(config.num_gat_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.encoder_dropout,
                    edge_dim=config.hidden_dim,
                    concat=True,
                )
            )

        # Latent distribution heads
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        face_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph to distribution parameters.

        Args:
            x: Node features, shape (num_nodes, node_features).
            face_types: Face type indices, shape (num_nodes,).
            edge_index: Edge indices, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_features).
            batch: Batch assignment for nodes, shape (num_nodes,).
            node_mask: Mask for valid nodes (1=real, 0=padding).

        Returns:
            mu: Mean of latent distribution, shape (batch_size, latent_dim).
            logvar: Log variance, shape (batch_size, latent_dim).
        """
        # Embed face types
        face_embeds = self.face_type_embedding(face_types)

        # Concatenate with continuous features
        x_combined = torch.cat([x, face_embeds], dim=-1)

        # Encode
        h = self.node_encoder(x_combined)
        edge_h = self.edge_encoder(edge_attr)

        # Message passing
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_attr=edge_h)
            h = torch.relu(h)

        # Masked global pooling
        if batch is None:
            if node_mask is not None:
                # Apply mask and mean
                h = h * node_mask.unsqueeze(-1)
                h = h.sum(dim=0, keepdim=True) / node_mask.sum().clamp(min=1)
            else:
                h = h.mean(dim=0, keepdim=True)
        else:
            if node_mask is not None:
                # Masked pooling per graph
                h = h * node_mask.unsqueeze(-1)
                # Sum per graph
                h_sum = torch.zeros(
                    batch.max() + 1, h.size(-1), device=h.device, dtype=h.dtype
                )
                h_sum.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
                # Count per graph
                counts = torch.zeros(
                    batch.max() + 1, device=h.device, dtype=h.dtype
                )
                counts.scatter_add_(0, batch, node_mask.float())
                h = h_sum / counts.unsqueeze(-1).clamp(min=1)
            else:
                h = global_mean_pool(h, batch)

        # Distribution parameters
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar


class ParameterDecoder(nn.Module):
    """
    Multi-head parameter decoder.

    Decodes latent z to ALL L-bracket parameters:
    - 4 core params (leg1, leg2, width, thickness)
    - Fillet (radius + exists)
    - Up to 2 holes per leg (diameter, distance, exists) × 4 slots

    Same architecture as FullLatentRegressor.
    """

    def __init__(self, config: ParameterVAEConfig):
        super().__init__()
        self.config = config

        # Shared backbone
        layers = []
        in_dim = config.latent_dim
        for _ in range(config.decoder_num_layers):
            layers.extend([
                nn.Linear(in_dim, config.decoder_hidden_dim),
                nn.LayerNorm(config.decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.decoder_dropout),
            ])
            in_dim = config.decoder_hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Core params head (always present)
        self.core_head = nn.Linear(config.decoder_hidden_dim, 4)

        # Fillet head
        self.fillet_head = nn.Linear(config.decoder_hidden_dim, 1)  # radius
        self.fillet_exists_head = nn.Linear(config.decoder_hidden_dim, 1)

        # Hole heads (2 slots per leg)
        self.hole1_heads = nn.ModuleList([
            nn.Linear(config.decoder_hidden_dim, 2)  # diameter, distance
            for _ in range(config.max_holes_per_leg)
        ])
        self.hole1_exists_head = nn.Linear(
            config.decoder_hidden_dim, config.max_holes_per_leg
        )

        self.hole2_heads = nn.ModuleList([
            nn.Linear(config.decoder_hidden_dim, 2)
            for _ in range(config.max_holes_per_leg)
        ])
        self.hole2_exists_head = nn.Linear(
            config.decoder_hidden_dim, config.max_holes_per_leg
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode latent to parameters.

        Args:
            z: Latent vector, shape (batch, latent_dim)

        Returns:
            Dictionary with:
                - core_params: (batch, 4)
                - fillet_radius: (batch, 1)
                - fillet_exists: (batch, 1) - probability
                - hole1_params: (batch, 2, 2) - (diameter, distance) per slot
                - hole1_exists: (batch, 2) - probability per slot
                - hole2_params: (batch, 2, 2)
                - hole2_exists: (batch, 2)
        """
        h = self.backbone(z)

        # Core params
        core_params = self.core_head(h)

        # Fillet
        fillet_radius = self.fillet_head(h)
        fillet_exists = torch.sigmoid(self.fillet_exists_head(h))

        # Holes on leg1
        hole1_params = torch.stack(
            [head(h) for head in self.hole1_heads], dim=1
        )  # (batch, 2, 2)
        hole1_exists = torch.sigmoid(self.hole1_exists_head(h))  # (batch, 2)

        # Holes on leg2
        hole2_params = torch.stack(
            [head(h) for head in self.hole2_heads], dim=1
        )
        hole2_exists = torch.sigmoid(self.hole2_exists_head(h))

        return {
            "core_params": core_params,
            "fillet_radius": fillet_radius,
            "fillet_exists": fillet_exists,
            "hole1_params": hole1_params,
            "hole1_exists": hole1_exists,
            "hole2_params": hole2_params,
            "hole2_exists": hole2_exists,
        }


class ParameterVAE(nn.Module):
    """
    VAE with graph encoder and parameter decoder.

    The key insight: instead of decoding to graph features (which don't
    directly encode parameters), we decode to parameters directly. This
    forces the latent space to encode parameter information, making
    "increase leg1" a meaningful direction.

    Includes an auxiliary LINEAR head for core params that forces
    linear organization of the latent space (like aux_weight in fixed topology).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ParameterVAEConfig | None = None):
        super().__init__()
        self.config = config or ParameterVAEConfig()
        self.encoder = ParameterVAEEncoder(self.config)
        self.decoder = ParameterDecoder(self.config)

        # Auxiliary LINEAR head: z -> 4 core params (no hidden layers)
        # This forces the latent space to have linear directions for parameters
        self.aux_core_head = nn.Linear(self.config.latent_dim, 4)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(
        self,
        x: torch.Tensor,
        face_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph to distribution parameters."""
        return self.encoder(x, face_types, edge_index, edge_attr, batch, node_mask)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode latent to parameters."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        face_types: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode.

        Returns:
            Dictionary with:
                - All parameter predictions from decoder
                - aux_core_params: Linear prediction of core params (for aux loss)
                - mu, logvar, z (latent space)
        """
        mu, logvar = self.encode(
            x, face_types, edge_index, edge_attr, batch, node_mask
        )
        z = self.reparameterize(mu, logvar)
        decoder_output = self.decode(z)

        # Auxiliary linear prediction (forces linear latent organization)
        aux_core_params = self.aux_core_head(z)

        return {
            **decoder_output,
            "aux_core_params": aux_core_params,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
