"""
Feature-to-Parameter regressor for L-bracket reconstruction.

Takes flattened graph features (from VAE decoder output) and predicts
the 8 L-bracket parameters. This is simpler than the GNN-based regressor
since it operates on the fixed-topology decoded features directly.

Pipeline:
    VAE Decoder → (10×8 nodes, 22×2 edges) → Flatten → FeatureRegressor → 8 params
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from graph_cad.models.parameter_regressor import (
    DEFAULT_PARAM_RANGES,
    PARAMETER_NAMES,
    denormalize_parameters,
    normalize_parameters,
)


@dataclass
class FeatureRegressorConfig:
    """Configuration for FeatureRegressor model."""

    # Input dimensions (fixed for L-bracket)
    num_nodes: int = 10
    node_features: int = 8
    num_edges: int = 22
    edge_features: int = 2

    # Architecture
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.1
    use_batch_norm: bool = True

    # Output
    num_parameters: int = 8

    @property
    def input_dim(self) -> int:
        """Total flattened input dimension."""
        return self.num_nodes * self.node_features + self.num_edges * self.edge_features


class FeatureRegressor(nn.Module):
    """
    MLP that predicts L-bracket parameters from flattened graph features.

    Takes the concatenated node and edge features from VAE decoder output
    and predicts the 8 geometric parameters.

    Architecture:
        Flatten(nodes, edges) → MLP layers → 8 parameters

    Args:
        config: Model configuration.
    """

    def __init__(self, config: FeatureRegressorConfig | None = None):
        super().__init__()

        if config is None:
            config = FeatureRegressorConfig()
        self.config = config

        # Build MLP layers
        layers = []
        in_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, config.num_parameters))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: Node features from VAE decoder, shape (batch, num_nodes, node_features)
                          or (num_nodes, node_features) for single sample.
            edge_features: Edge features from VAE decoder, shape (batch, num_edges, edge_features)
                          or (num_edges, edge_features) for single sample.

        Returns:
            Predicted parameters, shape (batch, num_parameters) or (num_parameters,).
        """
        # Handle single sample case
        squeeze_output = False
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            squeeze_output = True

        batch_size = node_features.shape[0]

        # Flatten features
        node_flat = node_features.view(batch_size, -1)  # (batch, 80)
        edge_flat = edge_features.view(batch_size, -1)  # (batch, 44)

        # Concatenate
        x = torch.cat([node_flat, edge_flat], dim=1)  # (batch, 124)

        # Predict parameters
        params = self.mlp(x)  # (batch, 8)

        if squeeze_output:
            params = params.squeeze(0)

        return params

    def predict_from_latent(
        self,
        vae: nn.Module,
        z: torch.Tensor,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """
        Convenience method: predict parameters directly from latent vector.

        Args:
            vae: Trained VAE model (for decoding).
            z: Latent vector, shape (batch, latent_dim) or (latent_dim,).
            denormalize: If True, return parameters in mm (not normalized).

        Returns:
            Predicted parameters, shape (batch, 8) or (8,).
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        # Decode latent to graph features
        with torch.no_grad():
            node_features, edge_features = vae.decode(z)

        # Predict parameters
        params = self.forward(node_features, edge_features)

        if denormalize:
            params = denormalize_parameters(params)

        if squeeze:
            params = params.squeeze(0)

        return params


def create_feature_regressor(
    config: FeatureRegressorConfig | None = None,
) -> FeatureRegressor:
    """Factory function to create a FeatureRegressor."""
    return FeatureRegressor(config)


def load_feature_regressor(
    path: str,
    device: str = "cpu",
) -> tuple[FeatureRegressor, dict]:
    """
    Load FeatureRegressor from checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        (model, checkpoint_dict) tuple.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = FeatureRegressor(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_feature_regressor(
    model: FeatureRegressor,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """
    Save FeatureRegressor checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer (optional).
        epoch: Current epoch.
        metrics: Training metrics.
        path: Output path.
    """
    checkpoint = {
        "config": model.config,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
