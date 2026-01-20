"""
Graph Variational Autoencoder for L-bracket face-adjacency graphs.

Supports two modes:
- Fixed topology (original): 8D node features, fixed graph size
- Variable topology (Phase 2): 9D node features + face embeddings, variable size with masks

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

    # Auxiliary parameter prediction (forces latent to encode all params)
    num_params: int = 8  # L-bracket has 8 parameters
    use_param_head: bool = False  # Enable auxiliary parameter prediction


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

        # Auxiliary parameter prediction head (z -> 8 L-bracket parameters)
        # Forces latent space to encode all parameters, not just dominant ones
        if self.config.use_param_head:
            self.param_head = nn.Sequential(
                nn.Linear(self.config.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.config.num_params),
            )
        else:
            self.param_head = None

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

        result = {
            "node_recon": node_recon,
            "edge_recon": edge_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

        # Auxiliary parameter prediction (if enabled)
        if self.param_head is not None:
            result["param_pred"] = self.param_head(z)

        return result

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


# =============================================================================
# Variable Topology VAE (Phase 2)
# =============================================================================


@dataclass
class VariableGraphVAEConfig:
    """Configuration for Variable Topology Graph VAE."""

    # Input dimensions (variable topology)
    node_features: int = 13  # area, dir_xyz, centroid_xyz, curvatures, bbox_diagonal, bbox_center_xyz
    edge_features: int = 2  # edge_length, dihedral_angle
    num_face_types: int = 3  # PLANAR=0, HOLE=1, FILLET=2 (see CLAUDE.md)
    face_embed_dim: int = 8  # Dimension of face type embeddings

    # Maximum sizes for padding
    max_nodes: int = 20  # Maximum number of faces
    max_edges: int = 50  # Maximum number of edges

    # Encoder architecture
    hidden_dim: int = 64
    num_gat_layers: int = 3
    num_heads: int = 4
    encoder_dropout: float = 0.1

    # Pooling configuration
    pooling_type: str = "mean"  # "mean" or "attention"
    attention_heads: int = 4  # Number of attention heads for attention pooling

    # Latent space (larger for variable topology)
    latent_dim: int = 32

    # Decoder architecture (reduced from 256,256,128 to prevent posterior collapse)
    decoder_hidden_dims: tuple[int, ...] = (128, 128, 64)
    decoder_dropout: float = 0.1

    # Auxiliary parameter prediction
    num_params: int = 8  # Still predict L-bracket parameters
    use_param_head: bool = False


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling for graph-level representations.

    Instead of simple mean pooling (which can lose symmetric information),
    this learns K different attention patterns over nodes. Each head can
    specialize to capture different aspects of the geometry.

    For example:
    - Head 1 might attend more to left-side faces → captures width
    - Head 2 might attend to leg1 end faces → captures leg1 length
    - Head 3 might attend to top/bottom faces → captures thickness

    This breaks the symmetry that causes mean pooling to lose width/thickness
    information in L-brackets.

    Args:
        hidden_dim: Dimension of node embeddings.
        output_dim: Total output dimension (divided among heads).
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, (
            f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Each head has its own attention mechanism
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_heads)
        ])

        # Each head projects to its portion of the output
        self.head_projections = nn.ModuleList([
            nn.Linear(hidden_dim, self.head_dim)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply multi-head attention pooling.

        Args:
            h: Node embeddings, shape (num_nodes, hidden_dim).
            batch: Batch assignment for nodes, shape (num_nodes,).
            node_mask: Mask for valid nodes (1=real, 0=padding), shape (num_nodes,).

        Returns:
            Pooled representation, shape (batch_size, output_dim).
        """
        head_outputs = []

        for attn_head, proj_head in zip(self.attention_heads, self.head_projections):
            # Compute attention scores
            scores = attn_head(h)  # (num_nodes, 1)

            # Apply node mask to scores (set padding to -inf before softmax)
            if node_mask is not None:
                mask = node_mask.unsqueeze(-1)  # (num_nodes, 1)
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Softmax over nodes within each graph
            if batch is None:
                # Single graph
                weights = torch.softmax(scores, dim=0)  # (num_nodes, 1)
                pooled = (weights * h).sum(dim=0, keepdim=True)  # (1, hidden_dim)
            else:
                # Batched graphs - need per-graph softmax
                # Use scatter_softmax equivalent
                num_graphs = batch.max().item() + 1

                # Compute softmax per graph using the log-sum-exp trick
                # First, get max per graph for numerical stability
                max_scores = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
                max_scores.fill_(float('-inf'))
                max_scores.scatter_reduce_(
                    0,
                    batch.unsqueeze(-1),
                    scores,
                    reduce='amax',
                    include_self=False,
                )

                # Subtract max and exp
                scores_shifted = scores - max_scores[batch]
                exp_scores = torch.exp(scores_shifted)

                # Handle masked nodes (they have -inf scores, so exp = 0)
                if node_mask is not None:
                    exp_scores = exp_scores * node_mask.unsqueeze(-1)

                # Sum exp per graph
                sum_exp = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
                sum_exp.scatter_add_(0, batch.unsqueeze(-1), exp_scores)

                # Normalize to get weights
                weights = exp_scores / sum_exp[batch].clamp(min=1e-10)

                # Weighted sum per graph
                weighted_h = weights * h  # (num_nodes, hidden_dim)
                pooled = torch.zeros(num_graphs, h.shape[-1], device=h.device, dtype=h.dtype)
                pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted_h), weighted_h)

            # Project to head dimension
            head_output = proj_head(pooled)  # (batch_size, head_dim)
            head_outputs.append(head_output)

        # Concatenate all heads
        return torch.cat(head_outputs, dim=-1)  # (batch_size, output_dim)


class VariableGraphVAEEncoder(nn.Module):
    """
    Encoder for variable topology graphs with face type embeddings.

    Architecture:
        1. Face type embedding (nn.Embedding)
        2. Concatenate embeddings with continuous features
        3. Node embedding (Linear -> ReLU -> Dropout)
        4. Edge embedding (Linear -> ReLU)
        5. GAT layers with edge attributes
        6. Pooling: mean pooling OR multi-head attention pooling
        7. Dual heads: mu_head, logvar_head

    Pooling Options:
        - "mean": Global mean pooling (default). Can lose symmetric information
          like width/thickness in L-brackets due to centroid cancellation.
        - "attention": Multi-head attention pooling. Each head learns different
          attention patterns over nodes, which can break symmetry and preserve
          width/thickness information.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: VariableGraphVAEConfig):
        super().__init__()
        self.config = config

        # Face type embedding
        self.face_type_embedding = nn.Embedding(
            config.num_face_types,
            config.face_embed_dim
        )

        # Input dimension: continuous features + face embedding
        input_dim = config.node_features + config.face_embed_dim  # 13 + 8 = 21

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

        # Pooling layer
        self.pooling_type = config.pooling_type
        if config.pooling_type == "attention":
            # Multi-head attention pooling
            # Output dimension is hidden_dim (same as mean pooling)
            # so mu_head and logvar_head can stay the same
            self.attention_pool = MultiHeadAttentionPooling(
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                num_heads=config.attention_heads,
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
            node_mask: Mask for valid nodes (1=real, 0=padding), shape (num_nodes,).

        Returns:
            mu: Mean of latent distribution, shape (batch_size, latent_dim).
            logvar: Log variance, shape (batch_size, latent_dim).
        """
        # Embed face types
        face_embeds = self.face_type_embedding(face_types)  # (num_nodes, embed_dim)

        # Concatenate with continuous features
        x_combined = torch.cat([x, face_embeds], dim=-1)  # (num_nodes, 9+8=17)

        # Encode
        h = self.node_encoder(x_combined)
        edge_h = self.edge_encoder(edge_attr)

        # Message passing
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_attr=edge_h)
            h = torch.relu(h)

        # Pooling: attention or mean
        if self.pooling_type == "attention":
            # Multi-head attention pooling
            h = self.attention_pool(h, batch=batch, node_mask=node_mask)
        else:
            # Masked global mean pooling (original behavior)
            if batch is None:
                if node_mask is not None:
                    # Apply mask for single graph
                    mask = node_mask.unsqueeze(-1)  # (num_nodes, 1)
                    h_masked = h * mask
                    h = h_masked.sum(dim=0, keepdim=True) / mask.sum().clamp(min=1)
                else:
                    h = h.mean(dim=0, keepdim=True)
            else:
                if node_mask is not None:
                    # Masked pooling for batched graphs
                    mask = node_mask.unsqueeze(-1)
                    h_masked = h * mask
                    # Use torch.zeros + scatter_add for masked pooling
                    num_graphs = batch.max().item() + 1
                    h_sum = torch.zeros(num_graphs, h.shape[-1], device=h.device, dtype=h.dtype)
                    h_sum.scatter_add_(0, batch.unsqueeze(-1).expand_as(h_masked), h_masked)
                    count = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
                    count.scatter_add_(0, batch.unsqueeze(-1), mask)
                    h = h_sum / count.clamp(min=1)
                else:
                    h = global_mean_pool(h, batch)

        # Distribution parameters
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        return mu, logvar


class VariableGraphVAEDecoder(nn.Module):
    """
    Decoder for variable topology graphs with mask and face type prediction.

    Outputs:
        - node_features: Continuous features (max_nodes, node_features)
        - edge_features: Edge features (max_edges, edge_features)
        - node_mask_logits: Logits for node existence (max_nodes,)
        - edge_mask_logits: Logits for edge existence (max_edges,)
        - face_type_logits: Logits for face type classification (max_nodes, num_face_types)

    Args:
        config: Model configuration.
    """

    def __init__(self, config: VariableGraphVAEConfig):
        super().__init__()
        self.config = config

        # Output dimensions (max sizes)
        self.max_node_output = config.max_nodes * config.node_features  # 20 * 9 = 180
        self.max_edge_output = config.max_edges * config.edge_features  # 50 * 2 = 100

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

        final_hidden = config.decoder_hidden_dims[-1]

        # Feature prediction heads
        self.node_head = nn.Linear(final_hidden, self.max_node_output)
        self.edge_head = nn.Linear(final_hidden, self.max_edge_output)

        # Existence mask heads
        self.node_mask_head = nn.Linear(final_hidden, config.max_nodes)
        self.edge_mask_head = nn.Linear(final_hidden, config.max_edges)

        # Face type classification head
        self.face_type_head = nn.Linear(
            final_hidden, config.max_nodes * config.num_face_types
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode latent vector to graph features and masks.

        Args:
            z: Latent vector, shape (batch_size, latent_dim).

        Returns:
            Dictionary with:
                - node_features: (batch, max_nodes, node_features)
                - edge_features: (batch, max_edges, edge_features)
                - node_mask_logits: (batch, max_nodes)
                - edge_mask_logits: (batch, max_edges)
                - face_type_logits: (batch, max_nodes, num_face_types)
        """
        batch_size = z.shape[0]
        h = self.backbone(z)

        # Continuous features
        node_flat = self.node_head(h)
        edge_flat = self.edge_head(h)
        node_features = node_flat.view(
            batch_size, self.config.max_nodes, self.config.node_features
        )
        edge_features = edge_flat.view(
            batch_size, self.config.max_edges, self.config.edge_features
        )

        # Existence masks
        node_mask_logits = self.node_mask_head(h)
        edge_mask_logits = self.edge_mask_head(h)

        # Face type logits
        face_type_logits = self.face_type_head(h).view(
            batch_size, self.config.max_nodes, self.config.num_face_types
        )

        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "node_mask_logits": node_mask_logits,
            "edge_mask_logits": edge_mask_logits,
            "face_type_logits": face_type_logits,
        }


class VariableGraphVAE(nn.Module):
    """
    Full Variable Topology VAE for face-adjacency graphs.

    Handles variable-size graphs with:
    - Face type embeddings in encoder
    - Mask prediction for node/edge existence
    - Face type classification in decoder

    Args:
        config: Model configuration. Uses defaults if None.
    """

    def __init__(self, config: VariableGraphVAEConfig | None = None):
        super().__init__()
        self.config = config or VariableGraphVAEConfig()
        self.encoder = VariableGraphVAEEncoder(self.config)
        self.decoder = VariableGraphVAEDecoder(self.config)

        # Auxiliary parameter prediction head
        if self.config.use_param_head:
            self.param_head = nn.Sequential(
                nn.Linear(self.config.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.config.num_params),
            )
        else:
            self.param_head = None

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
        """Decode latent vector to graph features and masks."""
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
            Dictionary with all outputs including:
                - node_features, edge_features (reconstructed)
                - node_mask_logits, edge_mask_logits
                - face_type_logits
                - mu, logvar, z
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

        if self.param_head is not None:
            result["param_pred"] = self.param_head(z)

        return result

    def sample(
        self, num_samples: int, device: str = "cpu"
    ) -> dict[str, torch.Tensor]:
        """Sample from prior and decode."""
        z = torch.randn(num_samples, self.config.latent_dim, device=device)
        return self.decode(z)
