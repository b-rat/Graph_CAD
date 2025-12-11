"""
Parameter regressor for L-bracket reconstruction.

Predicts the 8 L-bracket parameters from a face-adjacency graph,
enabling reconstruction of the original CAD model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

if TYPE_CHECKING:
    from graph_cad.data import BRepGraph


# L-bracket parameter names in order
PARAMETER_NAMES = [
    "leg1_length",
    "leg2_length",
    "width",
    "thickness",
    "hole1_distance",
    "hole1_diameter",
    "hole2_distance",
    "hole2_diameter",
]

# Default parameter ranges for normalization (from LBracketRanges defaults)
DEFAULT_PARAM_RANGES = {
    "leg1_length": (50.0, 200.0),
    "leg2_length": (50.0, 200.0),
    "width": (20.0, 60.0),
    "thickness": (3.0, 12.0),
    "hole1_diameter": (4.0, 12.0),
    "hole2_diameter": (4.0, 12.0),
    # hole distances have variable ranges, use conservative estimate
    "hole1_distance": (4.0, 180.0),
    "hole2_distance": (4.0, 180.0),
}


@dataclass
class ParameterRegressorConfig:
    """Configuration for ParameterRegressor model."""

    node_features: int = 8  # face_type, area, dir_xyz, centroid_xyz
    edge_features: int = 2  # edge_length, dihedral_angle
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4  # For GAT attention
    dropout: float = 0.1
    num_parameters: int = 8  # L-bracket has 8 parameters


class ParameterRegressor(nn.Module):
    """
    Graph Neural Network that predicts L-bracket parameters from face-adjacency graph.

    Architecture:
        1. Input projection (node features → hidden_dim)
        2. GAT layers with edge features (message passing)
        3. Global mean pooling (graph-level representation)
        4. MLP head (predict 8 parameters)

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ParameterRegressorConfig | None = None):
        super().__init__()

        if config is None:
            config = ParameterRegressorConfig()
        self.config = config

        # Input projection
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Edge feature projection (for edge_attr in GAT)
        self.edge_encoder = nn.Sequential(
            nn.Linear(config.edge_features, config.hidden_dim),
            nn.ReLU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(config.num_layers):
            # GATConv with edge_dim for edge features
            self.gat_layers.append(
                GATConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.dropout,
                    edge_dim=config.hidden_dim,
                    concat=True,  # Output: hidden_dim
                )
            )

        # MLP head for parameter prediction
        self.mlp_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_parameters),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features, shape (num_nodes, node_features).
            edge_index: Edge indices, shape (2, num_edges).
            edge_attr: Edge features, shape (num_edges, edge_features).
            batch: Batch assignment for nodes, shape (num_nodes,).
                   None for single graph.

        Returns:
            Predicted parameters, shape (batch_size, num_parameters).
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

        # Predict parameters
        params = self.mlp_head(h)

        return params


def brep_graph_to_pyg(graph: BRepGraph, device: str = "cpu") -> dict:
    """
    Convert BRepGraph to PyTorch Geometric tensors.

    Args:
        graph: BRepGraph from graph extraction.
        device: Device to place tensors on.

    Returns:
        Dictionary with keys: x, edge_index, edge_attr
    """
    import torch

    x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }


def normalize_parameters(params: torch.Tensor, ranges: dict | None = None) -> torch.Tensor:
    """
    Normalize parameters to [0, 1] range.

    Args:
        params: Parameter tensor, shape (..., 8).
        ranges: Dictionary of (min, max) tuples per parameter.

    Returns:
        Normalized parameters.
    """
    if ranges is None:
        ranges = DEFAULT_PARAM_RANGES

    mins = torch.tensor(
        [ranges[name][0] for name in PARAMETER_NAMES],
        dtype=params.dtype,
        device=params.device,
    )
    maxs = torch.tensor(
        [ranges[name][1] for name in PARAMETER_NAMES],
        dtype=params.dtype,
        device=params.device,
    )

    return (params - mins) / (maxs - mins)


def denormalize_parameters(
    params: torch.Tensor, ranges: dict | None = None, clamp: bool = True
) -> torch.Tensor:
    """
    Denormalize parameters from [0, 1] to original ranges.

    Args:
        params: Normalized parameter tensor, shape (..., 8).
        ranges: Dictionary of (min, max) tuples per parameter.
        clamp: If True, clamp inputs to [0, 1] before denormalizing.

    Returns:
        Denormalized parameters in mm.
    """
    if ranges is None:
        ranges = DEFAULT_PARAM_RANGES

    # Clamp to valid range to prevent impossible parameter values
    if clamp:
        params = torch.clamp(params, 0.0, 1.0)

    mins = torch.tensor(
        [ranges[name][0] for name in PARAMETER_NAMES],
        dtype=params.dtype,
        device=params.device,
    )
    maxs = torch.tensor(
        [ranges[name][1] for name in PARAMETER_NAMES],
        dtype=params.dtype,
        device=params.device,
    )

    return params * (maxs - mins) + mins
