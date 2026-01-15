"""
DETR-style Transformer Graph Decoder for Variable Topology VAE.

This decoder replaces the MLP decoder with a permutation-invariant transformer
that uses Hungarian matching for training. Key components:

1. Learned Node Queries: Interchangeable vectors (like DETR object queries)
2. Cross-Attention: Queries attend to latent z
3. Self-Attention: Queries attend to each other for geometric consistency
4. Hungarian Matching: Optimal assignment of predictions to ground truth

This solves the Phase 2 MLP decoder limitation where fixed output slots
couldn't handle variable face ordering from OpenCASCADE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerDecoderConfig:
    """Configuration for Transformer Graph Decoder."""

    # Input/output dimensions
    latent_dim: int = 32
    node_features: int = 13  # area, dir_xyz, centroid_xyz, curvatures, bbox
    edge_features: int = 2   # edge_length, dihedral_angle (for future use)
    num_face_types: int = 3  # PLANAR=0, HOLE=1, FILLET=2

    # Maximum graph size
    max_nodes: int = 20

    # Transformer architecture
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1

    # Edge prediction
    predict_edge_attr: bool = False  # Future extension


class TransformerGraphDecoder(nn.Module):
    """
    DETR-style Transformer Decoder for graph reconstruction.

    Architecture:
        1. Project latent z to hidden_dim (key/value for cross-attention)
        2. Learned node queries (max_nodes × hidden_dim)
        3. Transformer decoder layers:
           - Self-attention between queries
           - Cross-attention from queries to z
        4. Output heads:
           - Node features (continuous): Linear -> (max_nodes, 13)
           - Face types (categorical): Linear -> (max_nodes, num_face_types)
           - Existence mask (binary): Linear -> (max_nodes, 1)
           - Edge logits (binary): Pairwise MLP -> (max_nodes, max_nodes)

    The key insight is that node queries are interchangeable - the transformer
    doesn't know which query corresponds to which face. Hungarian matching
    during training assigns predictions to ground truth optimally.
    """

    def __init__(self, config: TransformerDecoderConfig | None = None):
        super().__init__()
        self.config = config or TransformerDecoderConfig()

        # Project latent z to hidden dimension for cross-attention
        self.z_projection = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
        )

        # Learned node queries (like DETR object queries)
        # These are permutation-equivariant - they learn to specialize
        # but can be matched to any ground truth face
        self.node_queries = nn.Parameter(
            torch.randn(self.config.max_nodes, self.config.hidden_dim) * 0.02
        )

        # Positional encoding for queries (helps with differentiation)
        self.query_pos_embed = nn.Parameter(
            torch.randn(self.config.max_nodes, self.config.hidden_dim) * 0.02
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.num_layers,
        )

        # Output heads
        # Node features: 13D continuous (area, normals, centroid, curvature, bbox)
        self.node_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.node_features),
        )

        # Face type classification: 3 classes (PLANAR, HOLE, FILLET)
        self.face_type_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_face_types),
        )

        # Node existence: binary (is this a real face or padding?)
        self.existence_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )

        # Edge prediction: pairwise MLP on concatenated node embeddings
        # Predicts adjacency matrix (binary: are these faces adjacent?)
        self.edge_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )

        # Optional: edge attribute prediction (for future extension)
        if self.config.predict_edge_attr:
            self.edge_attr_head = nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                nn.GELU(),
                nn.Linear(self.config.hidden_dim, self.config.edge_features),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode latent vector to graph predictions.

        Args:
            z: Latent vector, shape (batch_size, latent_dim)

        Returns:
            Dictionary with:
                - node_features: (batch, max_nodes, node_features)
                - face_type_logits: (batch, max_nodes, num_face_types)
                - existence_logits: (batch, max_nodes)
                - edge_logits: (batch, max_nodes, max_nodes)
                - node_embeddings: (batch, max_nodes, hidden_dim) [for debugging]
        """
        batch_size = z.shape[0]

        # Project z to hidden dimension: (batch, 1, hidden_dim)
        # This becomes the "memory" for cross-attention
        z_hidden = self.z_projection(z).unsqueeze(1)  # (batch, 1, hidden)

        # Expand node queries for batch: (batch, max_nodes, hidden_dim)
        queries = self.node_queries.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos = self.query_pos_embed.unsqueeze(0).expand(batch_size, -1, -1)

        # Add positional encoding to queries
        queries = queries + query_pos

        # Transformer decoder: queries attend to z
        # tgt=queries, memory=z_hidden
        node_embeddings = self.transformer_decoder(
            tgt=queries,
            memory=z_hidden,
        )  # (batch, max_nodes, hidden_dim)

        # Output heads
        node_features = self.node_head(node_embeddings)  # (batch, max_nodes, 13)
        face_type_logits = self.face_type_head(node_embeddings)  # (batch, max_nodes, 3)
        existence_logits = self.existence_head(node_embeddings).squeeze(-1)  # (batch, max_nodes)

        # Edge prediction: pairwise on all node pairs
        # Create all pairs: (batch, max_nodes, max_nodes, hidden*2)
        n = self.config.max_nodes
        node_i = node_embeddings.unsqueeze(2).expand(-1, -1, n, -1)  # (batch, n, n, h)
        node_j = node_embeddings.unsqueeze(1).expand(-1, n, -1, -1)  # (batch, n, n, h)
        pair_features = torch.cat([node_i, node_j], dim=-1)  # (batch, n, n, 2h)

        edge_logits = self.edge_head(pair_features).squeeze(-1)  # (batch, n, n)

        # Make edge logits symmetric (undirected graph)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2

        result = {
            "node_features": node_features,
            "face_type_logits": face_type_logits,
            "existence_logits": existence_logits,
            "edge_logits": edge_logits,
            "node_embeddings": node_embeddings,  # For debugging/analysis
        }

        # Optional edge attributes
        if self.config.predict_edge_attr:
            edge_attr = self.edge_attr_head(pair_features)  # (batch, n, n, 2)
            # Symmetrize
            edge_attr = (edge_attr + edge_attr.transpose(1, 2)) / 2
            result["edge_attr"] = edge_attr

        return result


class TransformerGraphVAE(nn.Module):
    """
    Full VAE with GAT encoder and Transformer decoder.

    Uses the existing VariableGraphVAEEncoder with the new TransformerGraphDecoder.
    This is the Phase 3 architecture designed to break the 64% accuracy ceiling.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder_config: TransformerDecoderConfig | None = None,
        use_param_head: bool = False,
        num_params: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = TransformerGraphDecoder(decoder_config)

        # Store configs for reference
        self.encoder_config = getattr(encoder, 'config', None)
        self.decoder_config = self.decoder.config

        # Auxiliary parameter prediction head
        # Forces latent space to encode all L-bracket parameters
        self.use_param_head = use_param_head
        self.num_params = num_params
        if use_param_head:
            self.param_head = nn.Sequential(
                nn.Linear(self.decoder_config.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_params),
            )

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
        """Decode latent vector to graph predictions."""
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

        Args:
            x: Node features, shape (num_nodes, node_features).
            face_types: Face type indices, shape (num_nodes,).
            edge_index: Edge indices, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_features).
            batch: Batch assignment for nodes.
            node_mask: Mask for valid nodes.

        Returns:
            Dictionary with decoder outputs plus mu, logvar, z.
        """
        mu, logvar = self.encode(
            x, face_types, edge_index, edge_attr, batch, node_mask
        )
        z = self.reparameterize(mu, logvar)
        decoder_output = self.decode(z)

        result = {
            **decoder_output,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

        # Auxiliary parameter prediction from latent
        if self.use_param_head:
            result["param_pred"] = self.param_head(mu)

        return result

    def sample(
        self, num_samples: int, device: str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Sample from prior and decode."""
        z = torch.randn(
            num_samples,
            self.decoder_config.latent_dim,
            device=device
        )
        return self.decode(z)
