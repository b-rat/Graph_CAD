"""
Heterogeneous Graph VAE for Phase 4 Multi-Geometry B-Rep encoding.

Uses PyTorch Geometric's HeteroConv for message passing between
vertex, edge, and face node types. This enables richer geometric
encoding compared to face-only graphs.

Architecture:
1. Type embeddings (edge_type, face_type)
2. Per-type linear projections to hidden_dim
3. HeteroConv layers with bidirectional V↔E, E↔F message passing
4. Per-type attention pooling
5. Concatenation and projection to latent distribution
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool


from graph_cad.data.brep_types import (
    NUM_EDGE_TYPES,
    NUM_FACE_TYPES,
    VERTEX_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    FACE_FEATURE_DIM,
)


@dataclass
class HeteroVAEConfig:
    """Configuration for Heterogeneous Graph VAE."""

    # Input dimensions
    vertex_features: int = VERTEX_FEATURE_DIM  # 3
    edge_features: int = EDGE_FEATURE_DIM      # 6
    face_features: int = FACE_FEATURE_DIM      # 13
    num_edge_types: int = NUM_EDGE_TYPES       # 4
    num_face_types: int = NUM_FACE_TYPES       # 3
    type_embed_dim: int = 8                    # Embedding dimension for types

    # Encoder architecture
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1

    # Latent space
    latent_dim: int = 32

    # Pooling
    use_attention_pooling: bool = True
    pool_heads: int = 4


class HeteroGNNEncoder(nn.Module):
    """
    Heterogeneous GNN Encoder for B-Rep graphs.

    Performs message passing between vertex, edge, and face nodes
    using HeteroConv layers. Each node type gets its own GAT layer
    for receiving messages.

    Node type feature dimensions:
    - vertex: 3 (xyz coordinates)
    - edge: 6 (length, tangent_xyz, curv_start, curv_end) + type embedding
    - face: 13 (existing features) + type embedding
    """

    def __init__(self, config: HeteroVAEConfig):
        super().__init__()
        self.config = config

        # Type embeddings
        self.edge_type_embed = nn.Embedding(config.num_edge_types, config.type_embed_dim)
        self.face_type_embed = nn.Embedding(config.num_face_types, config.type_embed_dim)

        # Input projections (combine features with embeddings)
        vertex_in_dim = config.vertex_features
        edge_in_dim = config.edge_features + config.type_embed_dim
        face_in_dim = config.face_features + config.type_embed_dim

        self.vertex_proj = nn.Sequential(
            nn.Linear(vertex_in_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.face_proj = nn.Sequential(
            nn.Linear(face_in_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # HeteroConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(config.num_layers):
            conv = HeteroConv({
                # Vertex <-> Edge message passing
                ('vertex', 'bounds', 'edge'): GATConv(
                    config.hidden_dim, config.hidden_dim // config.num_heads,
                    heads=config.num_heads, concat=True,
                    dropout=config.dropout, add_self_loops=False,
                ),
                ('edge', 'bounded_by', 'vertex'): GATConv(
                    config.hidden_dim, config.hidden_dim // config.num_heads,
                    heads=config.num_heads, concat=True,
                    dropout=config.dropout, add_self_loops=False,
                ),
                # Edge <-> Face message passing
                ('edge', 'bounds', 'face'): GATConv(
                    config.hidden_dim, config.hidden_dim // config.num_heads,
                    heads=config.num_heads, concat=True,
                    dropout=config.dropout, add_self_loops=False,
                ),
                ('face', 'bounded_by', 'edge'): GATConv(
                    config.hidden_dim, config.hidden_dim // config.num_heads,
                    heads=config.num_heads, concat=True,
                    dropout=config.dropout, add_self_loops=False,
                ),
            }, aggr='sum')
            self.convs.append(conv)

            # Per-type layer norms
            self.norms.append(nn.ModuleDict({
                'vertex': nn.LayerNorm(config.hidden_dim),
                'edge': nn.LayerNorm(config.hidden_dim),
                'face': nn.LayerNorm(config.hidden_dim),
            }))

        # Pooling layers (one per node type)
        if config.use_attention_pooling:
            self.vertex_pool = AttentionPooling(config.hidden_dim, config.pool_heads)
            self.edge_pool = AttentionPooling(config.hidden_dim, config.pool_heads)
            self.face_pool = AttentionPooling(config.hidden_dim, config.pool_heads)
        else:
            self.vertex_pool = None
            self.edge_pool = None
            self.face_pool = None

        # Final projection to latent distribution
        # Concatenate pooled V, E, F representations
        pool_dim = config.hidden_dim * 3  # One per node type
        self.mu_head = nn.Linear(pool_dim, config.latent_dim)
        self.logvar_head = nn.Linear(pool_dim, config.latent_dim)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        batch_dict: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode heterogeneous graph to latent distribution.

        Args:
            x_dict: Node features dict {'vertex': (N_v, 3), 'edge': (N_e, 6+emb), 'face': (N_f, 13+emb)}
            edge_index_dict: Edge indices dict for each relation type
            batch_dict: Batch assignment dict (for batched graphs)

        Returns:
            mu: Latent mean, shape (batch_size, latent_dim)
            logvar: Latent log variance
        """
        # Add type embeddings
        if 'edge_type' in x_dict:
            edge_type_emb = self.edge_type_embed(x_dict.pop('edge_type'))
            x_dict['edge'] = torch.cat([x_dict['edge'], edge_type_emb], dim=-1)

        if 'face_type' in x_dict:
            face_type_emb = self.face_type_embed(x_dict.pop('face_type'))
            x_dict['face'] = torch.cat([x_dict['face'], face_type_emb], dim=-1)

        # Project to hidden dimension
        h_dict = {
            'vertex': self.vertex_proj(x_dict['vertex']),
            'edge': self.edge_proj(x_dict['edge']),
            'face': self.face_proj(x_dict['face']),
        }

        # Message passing layers
        for conv, norms in zip(self.convs, self.norms):
            # Store residuals
            h_res = {k: v.clone() for k, v in h_dict.items()}

            # HeteroConv
            h_out = conv(h_dict, edge_index_dict)

            # Residual + norm + activation
            for node_type in ['vertex', 'edge', 'face']:
                if node_type in h_out and h_out[node_type] is not None:
                    h_dict[node_type] = F.relu(
                        norms[node_type](h_out[node_type] + h_res[node_type])
                    )

        # Pooling per node type
        if batch_dict is None:
            # Single graph
            if self.vertex_pool is not None:
                vertex_pooled = self.vertex_pool(h_dict['vertex'])
                edge_pooled = self.edge_pool(h_dict['edge'])
                face_pooled = self.face_pool(h_dict['face'])
            else:
                vertex_pooled = h_dict['vertex'].mean(dim=0, keepdim=True)
                edge_pooled = h_dict['edge'].mean(dim=0, keepdim=True)
                face_pooled = h_dict['face'].mean(dim=0, keepdim=True)
        else:
            # Batched graphs
            if self.vertex_pool is not None:
                vertex_pooled = self.vertex_pool(h_dict['vertex'], batch_dict.get('vertex'))
                edge_pooled = self.edge_pool(h_dict['edge'], batch_dict.get('edge'))
                face_pooled = self.face_pool(h_dict['face'], batch_dict.get('face'))
            else:
                vertex_pooled = global_mean_pool(h_dict['vertex'], batch_dict.get('vertex'))
                edge_pooled = global_mean_pool(h_dict['edge'], batch_dict.get('edge'))
                face_pooled = global_mean_pool(h_dict['face'], batch_dict.get('face'))

        # Concatenate and project to latent
        pooled = torch.cat([vertex_pooled, edge_pooled, face_pooled], dim=-1)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled)

        return mu, logvar


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for graph-level representation.

    Learns to weight nodes differently when computing the graph
    representation, breaking symmetry that causes mean pooling
    to lose information.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention score network
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads),
        )

        # Value projection
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            h: Node features, shape (num_nodes, hidden_dim)
            batch: Batch assignment, shape (num_nodes,)

        Returns:
            Pooled representation, shape (batch_size, hidden_dim)
        """
        # Compute attention scores
        attn_scores = self.attn(h)  # (num_nodes, num_heads)

        # Value projection
        values = self.value_proj(h)  # (num_nodes, hidden_dim)

        if batch is None:
            # Single graph
            attn_weights = F.softmax(attn_scores, dim=0)  # (num_nodes, num_heads)

            # Weighted sum for each head, then average
            pooled = torch.zeros(1, self.hidden_dim, device=h.device, dtype=h.dtype)
            for head in range(self.num_heads):
                weights = attn_weights[:, head:head+1]  # (num_nodes, 1)
                pooled += (weights * values).sum(dim=0, keepdim=True)
            pooled = pooled / self.num_heads
        else:
            # Batched graphs
            num_graphs = batch.max().item() + 1

            # Compute per-graph softmax using scatter
            # First, get max per graph for numerical stability
            max_scores = torch.zeros(num_graphs, self.num_heads, device=h.device)
            max_scores.fill_(float('-inf'))
            max_scores.scatter_reduce_(
                0, batch.unsqueeze(-1).expand(-1, self.num_heads),
                attn_scores, reduce='amax', include_self=False
            )

            # Subtract max and exp
            scores_shifted = attn_scores - max_scores[batch]
            exp_scores = torch.exp(scores_shifted)

            # Sum exp per graph
            sum_exp = torch.zeros(num_graphs, self.num_heads, device=h.device)
            sum_exp.scatter_add_(0, batch.unsqueeze(-1).expand(-1, self.num_heads), exp_scores)

            # Normalize
            attn_weights = exp_scores / sum_exp[batch].clamp(min=1e-10)

            # Weighted sum per graph
            pooled = torch.zeros(num_graphs, self.hidden_dim, device=h.device, dtype=h.dtype)
            for head in range(self.num_heads):
                weights = attn_weights[:, head:head+1]  # (num_nodes, 1)
                weighted = weights * values
                pooled.scatter_add_(
                    0, batch.unsqueeze(-1).expand_as(weighted), weighted
                )
            pooled = pooled / self.num_heads

        return pooled


class HeteroVAE(nn.Module):
    """
    Full Heterogeneous VAE for multi-geometry B-Rep encoding.

    Combines:
    - HeteroGNN encoder for V/E/F message passing
    - Transformer decoder for graph reconstruction (from transformer_decoder.py)
    - Optional parameter prediction head

    Note: The decoder is imported from the existing TransformerGraphDecoder
    since face-level reconstruction is sufficient for the parameter prediction
    task. The encoder benefits from the richer V/E/F representation.
    """

    def __init__(
        self,
        config: HeteroVAEConfig | None = None,
        use_param_head: bool = True,
        num_params: int = 6,  # Max params (BlockHole)
    ):
        super().__init__()
        self.config = config or HeteroVAEConfig()
        self.encoder = HeteroGNNEncoder(self.config)

        # Import and create decoder
        from graph_cad.models.transformer_decoder import (
            TransformerGraphDecoder,
            TransformerDecoderConfig,
        )

        decoder_config = TransformerDecoderConfig(
            latent_dim=self.config.latent_dim,
            node_features=FACE_FEATURE_DIM,
            num_face_types=NUM_FACE_TYPES,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
        )
        self.decoder = TransformerGraphDecoder(decoder_config)

        # Parameter prediction head
        self.use_param_head = use_param_head
        self.num_params = num_params
        if use_param_head:
            self.param_head = nn.Sequential(
                nn.Linear(self.config.latent_dim, 64),
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
        data: HeteroData,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode HeteroData to latent distribution.

        Args:
            data: PyG HeteroData batch

        Returns:
            mu, logvar: Latent distribution parameters
        """
        # Prepare input dict
        x_dict = {
            'vertex': data['vertex'].x,
            'edge': data['edge'].x,
            'face': data['face'].x,
        }

        # Add type info if present
        if hasattr(data['edge'], 'edge_type'):
            x_dict['edge_type'] = data['edge'].edge_type
        if hasattr(data['face'], 'face_type'):
            x_dict['face_type'] = data['face'].face_type

        # Prepare edge index dict
        edge_index_dict = {}
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index

        # Prepare batch dict if batched
        batch_dict = None
        if hasattr(data['vertex'], 'batch'):
            batch_dict = {
                'vertex': data['vertex'].batch,
                'edge': data['edge'].batch,
                'face': data['face'].batch,
            }

        return self.encoder(x_dict, edge_index_dict, batch_dict)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode latent to face-level graph reconstruction."""
        return self.decoder(z)

    def forward(
        self,
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            data: PyG HeteroData batch

        Returns:
            Dict with decoder outputs, mu, logvar, z, and optionally param_pred
        """
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        decoder_output = self.decode(z)

        result = {
            **decoder_output,
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }

        if self.use_param_head:
            result['param_pred'] = self.param_head(mu)

        return result

    def sample(
        self, num_samples: int, device: str = 'cpu'
    ) -> dict[str, torch.Tensor]:
        """Sample from prior and decode."""
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        return self.decode(z)


def create_hetero_vae(
    latent_dim: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 3,
    use_param_head: bool = True,
    num_params: int = 6,
) -> HeteroVAE:
    """
    Factory function to create HeteroVAE with common configurations.

    Args:
        latent_dim: Latent space dimension
        hidden_dim: Hidden dimension for message passing
        num_layers: Number of HeteroConv layers
        use_param_head: Whether to add parameter prediction head
        num_params: Number of parameters to predict

    Returns:
        Configured HeteroVAE instance
    """
    config = HeteroVAEConfig(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    return HeteroVAE(config, use_param_head=use_param_head, num_params=num_params)
