"""
PyTorch Geometric dataset for L-bracket graphs.

Provides on-the-fly generation of L-bracket graphs with ground truth parameters
for training the parameter regressor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from graph_cad.data.graph_extraction import extract_graph_from_solid
from graph_cad.data.l_bracket import LBracket, LBracketRanges
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
