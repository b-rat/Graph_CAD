"""
PyTorch Geometric dataset for L-bracket graphs.

Provides on-the-fly generation of L-bracket graphs with ground truth parameters
for training the parameter regressor.

Supports two modes:
- Fixed topology (original): LBracketDataset with 10 faces, 22 edges
- Variable topology (Phase 2): VariableLBracketDataset with 6-15 faces, padded to max
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from graph_cad.data.graph_extraction import (
    extract_graph_from_solid,
    extract_graph_from_solid_variable,
)
from graph_cad.data.l_bracket import (
    LBracket,
    LBracketRanges,
    VariableLBracket,
    VariableLBracketRanges,
)
from graph_cad.models.parameter_regressor import (
    PARAMETER_NAMES,
    normalize_parameters,
)

if TYPE_CHECKING:
    from numpy.random import Generator


class LBracketDataset(Dataset):
    """
    PyTorch Geometric dataset that generates L-bracket graphs on-the-fly.

    Each sample contains:
        - Node features (10 faces × 8 features)
        - Edge indices (face adjacency)
        - Edge features (edge length, dihedral angle)
        - Target parameters (8 L-bracket dimensions, normalized to [0,1])

    Args:
        num_samples: Number of samples in the dataset.
        ranges: Parameter ranges for L-bracket generation.
        seed: Random seed for reproducibility.
        normalize_targets: Whether to normalize target parameters to [0,1].
        cache_graphs: Whether to cache generated graphs in memory.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        ranges: LBracketRanges | None = None,
        seed: int = 42,
        normalize_targets: bool = True,
        cache_graphs: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.ranges = ranges or LBracketRanges()
        self.seed = seed
        self.normalize_targets = normalize_targets
        self.cache_graphs = cache_graphs

        # Create RNG for reproducible generation
        self._rng = np.random.default_rng(seed)

        # Pre-generate random states for each sample (for reproducibility)
        self._sample_seeds = self._rng.integers(0, 2**31, size=num_samples)

        # Cache for generated graphs
        self._cache: dict[int, Data] = {}

    def len(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def get(self, idx: int) -> Data:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            PyG Data object with graph and target parameters.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        # Check cache
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        # Generate bracket with deterministic seed
        rng = np.random.default_rng(self._sample_seeds[idx])
        bracket = LBracket.random(rng, self.ranges)

        # Extract graph
        graph = extract_graph_from_solid(bracket.to_solid())

        # Get ground truth parameters
        params_dict = bracket.to_dict()
        params = torch.tensor(
            [params_dict[name] for name in PARAMETER_NAMES],
            dtype=torch.float32,
        )

        # Normalize if requested
        if self.normalize_targets:
            params = normalize_parameters(params.unsqueeze(0)).squeeze(0)

        # Create PyG Data object
        data = Data(
            x=torch.tensor(graph.node_features, dtype=torch.float32),
            edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(graph.edge_features, dtype=torch.float32),
            y=params,
            # Store additional metadata
            bbox_diagonal=torch.tensor([graph.bbox_diagonal], dtype=torch.float32),
        )

        # Cache if enabled
        if self.cache_graphs:
            self._cache[idx] = data

        return data


def create_data_loaders(
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    ranges: LBracketRanges | None = None,
) -> tuple:
    """
    Create train, validation, and test data loaders.

    Uses different seed offsets for each split to ensure no overlap.

    Args:
        train_size: Number of training samples.
        val_size: Number of validation samples.
        test_size: Number of test samples.
        batch_size: Batch size for data loaders.
        seed: Base random seed.
        num_workers: Number of data loading workers.
        ranges: Parameter ranges for L-bracket generation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch_geometric.loader import DataLoader

    train_dataset = LBracketDataset(
        num_samples=train_size,
        ranges=ranges,
        seed=seed,
        cache_graphs=True,
    )
    val_dataset = LBracketDataset(
        num_samples=val_size,
        ranges=ranges,
        seed=seed + 100000,  # Different seed for validation
        cache_graphs=True,
    )
    test_dataset = LBracketDataset(
        num_samples=test_size,
        ranges=ranges,
        seed=seed + 200000,  # Different seed for test
        cache_graphs=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Variable Topology Dataset (Phase 2)
# =============================================================================


class VariableLBracketDataset(Dataset):
    """
    PyTorch Geometric dataset for variable topology L-bracket graphs.

    Each sample contains:
        - Node features (padded to max_nodes × 9 features)
        - Face types (padded to max_nodes)
        - Edge indices (padded, with self-loops for padding)
        - Edge features (padded to max_edges × 2 features)
        - Node mask (1 for real nodes, 0 for padding)
        - Edge mask (1 for real edges, 0 for padding)
        - Target: core parameters (leg lengths, width, thickness)

    Args:
        num_samples: Number of samples in the dataset.
        ranges: Parameter ranges for L-bracket generation.
        max_nodes: Maximum number of nodes (for padding).
        max_edges: Maximum number of edges (for padding).
        seed: Random seed for reproducibility.
        cache_graphs: Whether to cache generated graphs in memory.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        ranges: VariableLBracketRanges | None = None,
        max_nodes: int = 20,
        max_edges: int = 50,
        seed: int = 42,
        cache_graphs: bool = True,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.ranges = ranges or VariableLBracketRanges()
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.seed = seed
        self.cache_graphs = cache_graphs

        # Create RNG for reproducible generation
        self._rng = np.random.default_rng(seed)

        # Pre-generate random states for each sample
        self._sample_seeds = self._rng.integers(0, 2**31, size=num_samples)

        # Cache for generated graphs
        self._cache: dict[int, Data] = {}

    def len(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def get(self, idx: int) -> Data:
        """
        Get a single sample with padding.

        Args:
            idx: Sample index.

        Returns:
            PyG Data object with padded graph data and masks.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        # Check cache
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        # Generate bracket with deterministic seed
        rng = np.random.default_rng(self._sample_seeds[idx])
        bracket = VariableLBracket.random(rng, self.ranges)

        # Extract variable topology graph
        try:
            graph = extract_graph_from_solid_variable(bracket.to_solid())
        except Exception as e:
            # If extraction fails, generate a simpler bracket
            bracket = VariableLBracket(
                leg1_length=100, leg2_length=100, width=30, thickness=5
            )
            graph = extract_graph_from_solid_variable(bracket.to_solid())

        num_nodes = graph.num_faces
        num_edges = graph.num_edges

        # Validate sizes
        if num_nodes > self.max_nodes:
            raise ValueError(
                f"Graph has {num_nodes} nodes, exceeds max_nodes={self.max_nodes}"
            )
        if num_edges > self.max_edges:
            raise ValueError(
                f"Graph has {num_edges} edges, exceeds max_edges={self.max_edges}"
            )

        # Pad node features to max_nodes
        node_features = np.zeros((self.max_nodes, 9), dtype=np.float32)
        node_features[:num_nodes] = graph.node_features

        # Pad face types to max_nodes (use 0 for padding, will be masked)
        face_types = np.zeros(self.max_nodes, dtype=np.int64)
        face_types[:num_nodes] = graph.face_types

        # Create node mask
        node_mask = np.zeros(self.max_nodes, dtype=np.float32)
        node_mask[:num_nodes] = 1.0

        # Pad edge features to max_edges
        edge_features = np.zeros((self.max_edges, 2), dtype=np.float32)
        edge_features[:num_edges] = graph.edge_features

        # Create edge mask
        edge_mask = np.zeros(self.max_edges, dtype=np.float32)
        edge_mask[:num_edges] = 1.0

        # Pad edge_index - use self-loops on node 0 for padding
        # This is safe because padding edges will be masked in loss
        edge_index = np.zeros((2, self.max_edges), dtype=np.int64)
        edge_index[:, :num_edges] = graph.edge_index

        # Get core parameters for auxiliary loss
        # Core params: leg1, leg2, width, thickness (normalized)
        params_dict = bracket.to_dict()
        core_params = torch.tensor([
            params_dict["leg1_length"],
            params_dict["leg2_length"],
            params_dict["width"],
            params_dict["thickness"],
        ], dtype=torch.float32)

        # Normalize core params to [0, 1]
        core_ranges = torch.tensor([
            [self.ranges.leg1_length[0], self.ranges.leg1_length[1]],
            [self.ranges.leg2_length[0], self.ranges.leg2_length[1]],
            [self.ranges.width[0], self.ranges.width[1]],
            [self.ranges.thickness[0], self.ranges.thickness[1]],
        ], dtype=torch.float32)
        core_params_normalized = (core_params - core_ranges[:, 0]) / (
            core_ranges[:, 1] - core_ranges[:, 0]
        )

        # Full parameter extraction for multi-head regressor
        # Fillet: normalized radius and exists flag
        fillet_radius_norm = bracket.fillet_radius / self.ranges.fillet_radius[1]  # Normalize to [0, 1]
        fillet_exists = 1.0 if bracket.fillet_radius > 0 else 0.0

        # Holes on leg1 (up to 2 slots)
        hole1_params = torch.zeros(2, 2, dtype=torch.float32)  # (slot, (diam, dist))
        hole1_exists = torch.zeros(2, dtype=torch.float32)
        for i, (diam, dist) in enumerate(zip(bracket.hole1_diameters, bracket.hole1_distances)):
            if i >= 2:
                break
            hole1_params[i, 0] = (diam - self.ranges.hole_diameter[0]) / (
                self.ranges.hole_diameter[1] - self.ranges.hole_diameter[0]
            )
            hole1_params[i, 1] = dist / bracket.leg1_length  # Relative position
            hole1_exists[i] = 1.0

        # Holes on leg2 (up to 2 slots)
        hole2_params = torch.zeros(2, 2, dtype=torch.float32)
        hole2_exists = torch.zeros(2, dtype=torch.float32)
        for i, (diam, dist) in enumerate(zip(bracket.hole2_diameters, bracket.hole2_distances)):
            if i >= 2:
                break
            hole2_params[i, 0] = (diam - self.ranges.hole_diameter[0]) / (
                self.ranges.hole_diameter[1] - self.ranges.hole_diameter[0]
            )
            hole2_params[i, 1] = dist / bracket.leg2_length
            hole2_exists[i] = 1.0

        # Create PyG Data object
        data = Data(
            # Graph structure
            x=torch.tensor(node_features, dtype=torch.float32),
            face_types=torch.tensor(face_types, dtype=torch.long),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),
            # Masks
            node_mask=torch.tensor(node_mask, dtype=torch.float32),
            edge_mask=torch.tensor(edge_mask, dtype=torch.float32),
            # Actual sizes
            num_real_nodes=torch.tensor([num_nodes], dtype=torch.long),
            num_real_edges=torch.tensor([num_edges], dtype=torch.long),
            # Core parameters (normalized)
            y=core_params_normalized,
            # Full parameters for multi-head regressor
            fillet_radius=torch.tensor([fillet_radius_norm], dtype=torch.float32),
            fillet_exists=torch.tensor([fillet_exists], dtype=torch.float32),
            hole1_params=hole1_params,
            hole1_exists=hole1_exists,
            hole2_params=hole2_params,
            hole2_exists=hole2_exists,
            # Metadata
            bbox_diagonal=torch.tensor([graph.bbox_diagonal], dtype=torch.float32),
            has_fillet=torch.tensor([1.0 if bracket.has_fillet else 0.0], dtype=torch.float32),
            num_holes=torch.tensor([bracket.num_holes_leg1 + bracket.num_holes_leg2], dtype=torch.long),
        )

        # Cache if enabled
        if self.cache_graphs:
            self._cache[idx] = data

        return data


def create_variable_data_loaders(
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    batch_size: int = 32,
    max_nodes: int = 20,
    max_edges: int = 50,
    seed: int = 42,
    num_workers: int = 0,
    ranges: VariableLBracketRanges | None = None,
) -> tuple:
    """
    Create train, validation, and test data loaders for variable topology.

    Args:
        train_size: Number of training samples.
        val_size: Number of validation samples.
        test_size: Number of test samples.
        batch_size: Batch size for data loaders.
        max_nodes: Maximum nodes for padding.
        max_edges: Maximum edges for padding.
        seed: Base random seed.
        num_workers: Number of data loading workers.
        ranges: Parameter ranges for L-bracket generation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch_geometric.loader import DataLoader

    train_dataset = VariableLBracketDataset(
        num_samples=train_size,
        ranges=ranges,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed,
        cache_graphs=True,
    )
    val_dataset = VariableLBracketDataset(
        num_samples=val_size,
        ranges=ranges,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed + 100000,
        cache_graphs=True,
    )
    test_dataset = VariableLBracketDataset(
        num_samples=test_size,
        ranges=ranges,
        max_nodes=max_nodes,
        max_edges=max_edges,
        seed=seed + 200000,
        cache_graphs=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def collate_variable_batch(batch: list[Data]) -> dict[str, torch.Tensor]:
    """
    Collate a batch of variable topology graphs into tensors for VAE.

    This converts PyG batch format into the dict format expected by
    VariableGraphVAE and variable_vae_loss.

    Args:
        batch: List of PyG Data objects from VariableLBracketDataset.

    Returns:
        Dictionary with batched tensors ready for model input/loss.
    """
    from torch_geometric.data import Batch

    # Use PyG's batching
    pyg_batch = Batch.from_data_list(batch)

    # For the VAE, we need tensors in (batch, max_nodes/edges, features) format
    batch_size = len(batch)
    max_nodes = batch[0].x.shape[0]
    max_edges = batch[0].edge_attr.shape[0]

    # Reshape from flattened PyG format to batched format
    node_features = pyg_batch.x.view(batch_size, max_nodes, -1)
    face_types = pyg_batch.face_types.view(batch_size, max_nodes)
    edge_features = pyg_batch.edge_attr.view(batch_size, max_edges, -1)
    node_mask = pyg_batch.node_mask.view(batch_size, max_nodes)
    edge_mask = pyg_batch.edge_mask.view(batch_size, max_edges)
    y = pyg_batch.y.view(batch_size, -1)

    return {
        "node_features": node_features,
        "face_types": face_types,
        "edge_index": pyg_batch.edge_index,  # Keep in PyG format for encoder
        "edge_features": edge_features,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "batch": pyg_batch.batch,  # Node-to-graph assignment
        "params": y,
    }
