"""
Graph Variational Autoencoder for L-bracket face-adjacency graphs.

Encodes face-adjacency graphs into a latent space and reconstructs
graph features for geometric representation learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


@dataclass
class GraphVAEConfig:
    """Configuration for Graph VAE model."""

    # Input dimensions (fixed for L-bracket)
    node_features: int = 8  # face_type, area, dir_xyz, centroid_xyz
    edge_features: int = 2  # edge_length, dihedral_angle
    num_nodes: int = 10  # Fixed L-bracket topology
    num_edges: int = 22  # Fixed L-bracket topology

    # Encoder architecture
    hidden_dim: int = 64
    num_gat_layers: int = 3
    num_heads: int = 4
    encoder_dropout: float = 0.1

    # Latent space
    latent_dim: int = 64

    # Decoder architecture
    decoder_hidden_dims: tuple[int, ...] = (256, 256, 128)
    decoder_dropout: float = 0.1


class GraphVAEEncoder(nn.Module):
    """
    Encodes face-adjacency graph to latent distribution parameters.

    Architecture:
        1. Node embedding (Linear -> ReLU -> Dropout)
        2. Edge embedding (Linear -> ReLU)
        3. GAT layers with edge attributes (message passing)
        4. Global mean pooling (graph-level representation)
        5. Dual heads: mu_head, logvar_head

    Args:
        config: Model configuration.
    """

    def __init__(self, config: GraphVAEConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.encoder_dropout),
        )

        # Edge feature projection (for edge_attr in GAT)
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
                    concat=True,  # Output: hidden_dim
                )
            )

        # Latent distribution heads
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph to distribution parameters.

        Args:
            x: Node features, shape (num_nodes, node_features).
            edge_index: Edge indices, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_features).
            batch: Batch assignment for nodes, shape (num_nodes,).

        Returns:
            mu: Mean of latent distribution, shape (batch_size, latent_dim).
            logvar: Log variance, shape (batch_size, latent_dim).
        """
        # Encode inputs
        h = self.node_encoder(x)
        edge_h = self.edge_encoder(edge_attr)

        # Message passing
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_attr=edge_h)
            h = torch.relu(h)

        # Global pooling
        if batch is None:
            # Single graph: pool all nodes
            h = h.mean(dim=0, keepdim=True)
        else:
            h = global_mean_pool(h, batch)

        # Distribution parameters
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar


class GraphVAEDecoder(nn.Module):
    """
    Decodes latent vector to graph features.

    For fixed topology, outputs node and edge features directly via MLP.
    The graph structure (edge_index) is fixed and not predicted.

    Architecture:
        z -> MLP backbone -> split heads
            -> node_head -> (batch, num_nodes * node_features)
            -> edge_head -> (batch, num_edges * edge_features)

    Args:
        config: Model configuration.
    """

    def __init__(self, config: GraphVAEConfig):
        super().__init__()
        self.config = config

        # Output dimensions
        self.node_output_dim = config.num_nodes * config.node_features  # 10 * 8 = 80
        self.edge_output_dim = config.num_edges * config.edge_features  # 22 * 2 = 44

        # Shared backbone
        layers = []
        in_dim = config.latent_dim
        for hidden_dim in config.decoder_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.decoder_dropout),
            ])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Split output heads
        final_hidden = config.decoder_hidden_dims[-1]
        self.node_head = nn.Linear(final_hidden, self.node_output_dim)
        self.edge_head = nn.Linear(final_hidden, self.edge_output_dim)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent vector to graph features.

        Args:
            z: Latent vector, shape (batch_size, latent_dim).

        Returns:
            node_features: Shape (batch_size, num_nodes, node_features).
            edge_features: Shape (batch_size, num_edges, edge_features).
        """
        # Shared computation
        h = self.backbone(z)

        # Predict features
        node_flat = self.node_head(h)
        edge_flat = self.edge_head(h)

        # Reshape to graph structure
        batch_size = z.shape[0]
        node_features = node_flat.view(
            batch_size, self.config.num_nodes, self.config.node_features
        )
        edge_features = edge_flat.view(
            batch_size, self.config.num_edges, self.config.edge_features
        )

        return node_features, edge_features


class GraphVAE(nn.Module):
    """
    Full Variational Autoencoder for face-adjacency graphs.

    Combines encoder and decoder with reparameterization trick.
    For L-bracket PoC, graph topology is fixed (10 nodes, 22 edges).

    Args:
        config: Model configuration. Uses defaults if None.
    """

    def __init__(self, config: GraphVAEConfig | None = None):
        super().__init__()
        self.config = config or GraphVAEConfig()
        self.encoder = GraphVAEEncoder(self.config)
        self.decoder = GraphVAEDecoder(self.config)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample z using reparameterization trick.

        z = mu + sigma * epsilon, where epsilon ~ N(0, I)

        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            z: Sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Deterministic in eval mode
            return mu

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph to distribution parameters."""
        return self.encoder(x, edge_index, edge_attr, batch)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode latent vector to graph features."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode.

        Args:
            x: Node features, shape (num_nodes, node_features).
            edge_index: Edge indices, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_features).
            batch: Batch assignment for nodes.

        Returns:
            Dictionary with keys:
                - node_recon: Reconstructed node features
                - edge_recon: Reconstructed edge features
                - mu: Latent mean
                - logvar: Latent log variance
                - z: Sampled latent vector
        """
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mu, logvar)
        node_recon, edge_recon = self.decode(z)

        return {
            "node_recon": node_recon,
            "edge_recon": edge_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def sample(
        self, num_samples: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from prior and decode.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate on.

        Returns:
            node_features: Shape (num_samples, num_nodes, node_features).
            edge_features: Shape (num_samples, num_edges, edge_features).
        """
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        return self.decode(z)

    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Linear interpolation between two latent vectors.

        Args:
            z1: Start latent vector, shape (latent_dim,) or (1, latent_dim).
            z2: End latent vector, shape (latent_dim,) or (1, latent_dim).
            num_steps: Number of interpolation steps.

        Returns:
            node_features: Shape (num_steps, num_nodes, node_features).
            edge_features: Shape (num_steps, num_edges, edge_features).
        """
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)

        # Linear interpolation
        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        z_interp = torch.stack([
            (1 - alpha) * z1 + alpha * z2 for alpha in alphas
        ]).squeeze(1)

        return self.decode(z_interp)
