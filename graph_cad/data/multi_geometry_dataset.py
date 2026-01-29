"""
Multi-geometry PyTorch Geometric dataset for Phase 4.

Unified dataset supporting all 6 geometry types with HeteroData format
for the HeteroGNN encoder. Each sample includes:
- Full B-Rep graph (vertices, edges, faces)
- Geometry type ID
- Normalized parameters (padded to max_params with mask)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData

from graph_cad.data.brep_extraction import (
    BRepHeteroGraph,
    extract_brep_hetero_graph_from_solid,
)
from graph_cad.data.brep_types import (
    GEOMETRY_BRACKET,
    GEOMETRY_TUBE,
    GEOMETRY_CHANNEL,
    GEOMETRY_BLOCK,
    GEOMETRY_CYLINDER,
    GEOMETRY_BLOCKHOLE,
    MAX_PARAMS,
    NUM_GEOMETRY_TYPES,
)
from graph_cad.data.geometry_generators import (
    Block,
    BlockHole,
    Channel,
    Cylinder,
    SimpleBracket,
    Tube,
)
from graph_cad.data.param_normalization import (
    get_param_count,
    normalize_params,
    pad_params,
)

if TYPE_CHECKING:
    from numpy.random import Generator


# Generator classes and their geometry type IDs
GENERATORS = {
    GEOMETRY_BRACKET: SimpleBracket,
    GEOMETRY_TUBE: Tube,
    GEOMETRY_CHANNEL: Channel,
    GEOMETRY_BLOCK: Block,
    GEOMETRY_CYLINDER: Cylinder,
    GEOMETRY_BLOCKHOLE: BlockHole,
}


def _extract_params(geometry: Any, geometry_type: int) -> torch.Tensor:
    """
    Extract parameters from a geometry instance as a tensor.

    Args:
        geometry: Geometry instance (Tube, Channel, etc.)
        geometry_type: Geometry type ID

    Returns:
        Parameter tensor, shape (num_params,)
    """
    params_dict = geometry.to_dict()

    if geometry_type == GEOMETRY_BRACKET:
        # VariableLBracket has many params, we only use core 4
        params = torch.tensor([
            params_dict["leg1_length"],
            params_dict["leg2_length"],
            params_dict["width"],
            params_dict["thickness"],
        ], dtype=torch.float32)
    elif geometry_type == GEOMETRY_TUBE:
        params = torch.tensor([
            params_dict["length"],
            params_dict["outer_dia"],
            params_dict["inner_dia"],
        ], dtype=torch.float32)
    elif geometry_type == GEOMETRY_CHANNEL:
        params = torch.tensor([
            params_dict["width"],
            params_dict["height"],
            params_dict["length"],
            params_dict["thickness"],
        ], dtype=torch.float32)
    elif geometry_type == GEOMETRY_BLOCK:
        params = torch.tensor([
            params_dict["length"],
            params_dict["width"],
            params_dict["height"],
        ], dtype=torch.float32)
    elif geometry_type == GEOMETRY_CYLINDER:
        params = torch.tensor([
            params_dict["length"],
            params_dict["diameter"],
        ], dtype=torch.float32)
    elif geometry_type == GEOMETRY_BLOCKHOLE:
        params = torch.tensor([
            params_dict["length"],
            params_dict["width"],
            params_dict["height"],
            params_dict["hole_dia"],
            params_dict["hole_x"],
            params_dict["hole_y"],
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")

    return params


def brep_graph_to_hetero_data(
    graph: BRepHeteroGraph,
    geometry_type: int,
    params: torch.Tensor,
    params_normalized: torch.Tensor,
    params_mask: torch.Tensor,
) -> HeteroData:
    """
    Convert BRepHeteroGraph to PyG HeteroData with metadata.

    Args:
        graph: BRepHeteroGraph instance
        geometry_type: Geometry type ID
        params: Raw parameters (padded)
        params_normalized: Normalized parameters (padded)
        params_mask: Parameter mask

    Returns:
        HeteroData with node features, edge indices, and metadata
    """
    data = HeteroData()

    # Node features
    data['vertex'].x = torch.tensor(graph.vertex_features, dtype=torch.float32)
    data['edge'].x = torch.tensor(graph.edge_features, dtype=torch.float32)
    data['face'].x = torch.tensor(graph.face_features, dtype=torch.float32)

    # Node types (for embeddings)
    data['edge'].edge_type = torch.tensor(graph.edge_types, dtype=torch.long)
    data['face'].face_type = torch.tensor(graph.face_types, dtype=torch.long)

    # Topology edges
    # vertex -> edge
    if graph.vertex_to_edge.shape[1] > 0:
        data['vertex', 'bounds', 'edge'].edge_index = torch.tensor(
            graph.vertex_to_edge, dtype=torch.long
        )
        data['edge', 'bounded_by', 'vertex'].edge_index = torch.tensor(
            graph.vertex_to_edge[[1, 0]], dtype=torch.long
        )
    else:
        # Empty topology
        data['vertex', 'bounds', 'edge'].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data['edge', 'bounded_by', 'vertex'].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # edge -> face
    if graph.edge_to_face.shape[1] > 0:
        data['edge', 'bounds', 'face'].edge_index = torch.tensor(
            graph.edge_to_face, dtype=torch.long
        )
        data['face', 'bounded_by', 'edge'].edge_index = torch.tensor(
            graph.edge_to_face[[1, 0]], dtype=torch.long
        )
    else:
        data['edge', 'bounds', 'face'].edge_index = torch.zeros(2, 0, dtype=torch.long)
        data['face', 'bounded_by', 'edge'].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Metadata - store params with shape (1, 6) for proper batching
    data.geometry_type = torch.tensor([geometry_type], dtype=torch.long)
    data.params = params.unsqueeze(0) if params.dim() == 1 else params
    data.params_normalized = params_normalized.unsqueeze(0) if params_normalized.dim() == 1 else params_normalized
    data.params_mask = params_mask.unsqueeze(0) if params_mask.dim() == 1 else params_mask
    data.bbox_diagonal = torch.tensor([graph.bbox_diagonal], dtype=torch.float32)
    data.bbox_center = torch.tensor(graph.bbox_center, dtype=torch.float32)
    data.num_vertices = torch.tensor([graph.num_vertices], dtype=torch.long)
    data.num_edges = torch.tensor([graph.num_edges], dtype=torch.long)
    data.num_faces = torch.tensor([graph.num_faces], dtype=torch.long)

    return data


class MultiGeometryDataset(Dataset):
    """
    PyTorch Geometric dataset for multi-geometry Phase 4 training.

    Generates samples on-the-fly from all 6 geometry types using
    full B-Rep heterogeneous graphs.

    Args:
        num_samples_per_type: Number of samples per geometry type.
        seed: Random seed for reproducibility.
        cache_graphs: Whether to cache generated graphs in memory.
        geometry_types: Optional list of geometry types to include.
            If None, includes all 6 types.
    """

    def __init__(
        self,
        num_samples_per_type: int = 5000,
        seed: int = 42,
        cache_graphs: bool = True,
        geometry_types: list[int] | None = None,
    ):
        super().__init__()

        if geometry_types is None:
            geometry_types = list(range(NUM_GEOMETRY_TYPES))

        self.geometry_types = geometry_types
        self.num_samples_per_type = num_samples_per_type
        self.seed = seed
        self.cache_graphs = cache_graphs

        # Total samples
        self.num_samples = len(geometry_types) * num_samples_per_type

        # Create RNG
        self._rng = np.random.default_rng(seed)

        # Pre-generate random states for each sample
        self._sample_seeds = self._rng.integers(0, 2**31, size=self.num_samples)

        # Map sample index to (geometry_type, within-type index)
        self._index_map = []
        for geo_type in geometry_types:
            for i in range(num_samples_per_type):
                self._index_map.append((geo_type, i))

        # Cache
        self._cache: dict[int, HeteroData] = {}

    def len(self) -> int:
        """Return total number of samples."""
        return self.num_samples

    def get(self, idx: int) -> HeteroData:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            HeteroData object with full B-Rep graph and metadata.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        # Check cache
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        # Get geometry type and within-type index
        geometry_type, _ = self._index_map[idx]

        # Generate geometry with deterministic seed
        rng = np.random.default_rng(self._sample_seeds[idx])
        geometry = self._generate_geometry(geometry_type, rng)

        # Extract B-Rep graph
        try:
            solid = geometry.to_solid()
            graph = extract_brep_hetero_graph_from_solid(solid)
        except Exception as e:
            # If extraction fails, generate simpler fallback geometry
            geometry = self._generate_fallback(geometry_type)
            solid = geometry.to_solid()
            graph = extract_brep_hetero_graph_from_solid(solid)

        # Extract and normalize parameters
        params = _extract_params(geometry, geometry_type)
        params_normalized = normalize_params(params, geometry_type)
        params_padded, params_mask = pad_params(params, geometry_type)
        params_norm_padded, _ = pad_params(params_normalized, geometry_type)

        # Convert to HeteroData
        data = brep_graph_to_hetero_data(
            graph,
            geometry_type,
            params_padded,
            params_norm_padded,
            params_mask,
        )

        # Cache
        if self.cache_graphs:
            self._cache[idx] = data

        return data

    def _generate_geometry(self, geometry_type: int, rng: Generator) -> Any:
        """Generate random geometry of specified type."""
        if geometry_type == GEOMETRY_BRACKET:
            return SimpleBracket.random(rng)
        elif geometry_type == GEOMETRY_TUBE:
            return Tube.random(rng)
        elif geometry_type == GEOMETRY_CHANNEL:
            return Channel.random(rng)
        elif geometry_type == GEOMETRY_BLOCK:
            return Block.random(rng)
        elif geometry_type == GEOMETRY_CYLINDER:
            return Cylinder.random(rng)
        elif geometry_type == GEOMETRY_BLOCKHOLE:
            return BlockHole.random(rng)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")

    def _generate_fallback(self, geometry_type: int) -> Any:
        """Generate simple fallback geometry if random generation fails."""
        if geometry_type == GEOMETRY_BRACKET:
            return SimpleBracket(
                leg1_length=100, leg2_length=100, width=30, thickness=5
            )
        elif geometry_type == GEOMETRY_TUBE:
            return Tube(length=100, outer_dia=50, inner_dia=40)
        elif geometry_type == GEOMETRY_CHANNEL:
            return Channel(width=50, height=50, length=100, thickness=5)
        elif geometry_type == GEOMETRY_BLOCK:
            return Block(length=100, width=50, height=30)
        elif geometry_type == GEOMETRY_CYLINDER:
            return Cylinder(length=100, diameter=50)
        elif geometry_type == GEOMETRY_BLOCKHOLE:
            return BlockHole(
                length=100, width=80, height=30, hole_dia=15, hole_x=0, hole_y=0
            )
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")


def create_multi_geometry_loaders(
    train_samples_per_type: int = 5000,
    val_samples_per_type: int = 500,
    test_samples_per_type: int = 500,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    geometry_types: list[int] | None = None,
) -> tuple:
    """
    Create train, validation, and test data loaders for multi-geometry dataset.

    Args:
        train_samples_per_type: Training samples per geometry type.
        val_samples_per_type: Validation samples per geometry type.
        test_samples_per_type: Test samples per geometry type.
        batch_size: Batch size for data loaders.
        seed: Base random seed.
        num_workers: Number of data loading workers.
        geometry_types: Optional list of geometry types to include.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    from torch_geometric.loader import DataLoader

    train_dataset = MultiGeometryDataset(
        num_samples_per_type=train_samples_per_type,
        seed=seed,
        cache_graphs=True,
        geometry_types=geometry_types,
    )
    val_dataset = MultiGeometryDataset(
        num_samples_per_type=val_samples_per_type,
        seed=seed + 100000,
        cache_graphs=True,
        geometry_types=geometry_types,
    )
    test_dataset = MultiGeometryDataset(
        num_samples_per_type=test_samples_per_type,
        seed=seed + 200000,
        cache_graphs=True,
        geometry_types=geometry_types,
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


def collate_hetero_batch(batch: list[HeteroData]) -> dict[str, torch.Tensor]:
    """
    Collate a batch of HeteroData into tensors for HeteroGNN.

    Note: PyG's DataLoader already handles HeteroData batching.
    This function is provided for custom collation if needed.

    Args:
        batch: List of HeteroData objects.

    Returns:
        Batched tensors dict.
    """
    from torch_geometric.data import Batch

    return Batch.from_data_list(batch)
