"""Unit tests for dataset classes."""

import numpy as np
import pytest

# Skip all tests in this module if PyTorch is not installed
torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader

from graph_cad.data.dataset import (
    LBracketDataset,
    VariableLBracketDataset,
    create_variable_data_loaders,
)
from graph_cad.data.l_bracket import VariableLBracketRanges


class TestLBracketDataset:
    """Test fixed topology L-bracket dataset."""

    def test_len(self):
        """Dataset should have correct length."""
        dataset = LBracketDataset(num_samples=100, seed=42)
        assert len(dataset) == 100

    def test_get_returns_data(self):
        """get() should return PyG Data object."""
        dataset = LBracketDataset(num_samples=10, seed=42)
        data = dataset.get(0)

        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "y")

    def test_deterministic_with_seed(self):
        """Same seed should produce same samples."""
        dataset1 = LBracketDataset(num_samples=10, seed=42)
        dataset2 = LBracketDataset(num_samples=10, seed=42)

        data1 = dataset1.get(0)
        data2 = dataset2.get(0)

        torch.testing.assert_close(data1.x, data2.x)
        torch.testing.assert_close(data1.y, data2.y)

    def test_caching(self):
        """Cached retrieval should return identical objects."""
        dataset = LBracketDataset(num_samples=10, seed=42, cache_graphs=True)

        data1 = dataset.get(0)
        data2 = dataset.get(0)

        assert data1 is data2  # Same object from cache


# =============================================================================
# Variable Topology Dataset Tests
# =============================================================================


class TestVariableLBracketDataset:
    """Test variable topology L-bracket dataset."""

    def test_len(self):
        """Dataset should have correct length."""
        dataset = VariableLBracketDataset(num_samples=50, seed=42)
        assert len(dataset) == 50

    def test_get_returns_data(self):
        """get() should return PyG Data object with expected fields."""
        dataset = VariableLBracketDataset(num_samples=10, seed=42)
        data = dataset.get(0)

        # Check required fields
        assert hasattr(data, "x")
        assert hasattr(data, "face_types")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")
        assert hasattr(data, "node_mask")
        assert hasattr(data, "edge_mask")
        assert hasattr(data, "y")
        assert hasattr(data, "num_real_nodes")
        assert hasattr(data, "num_real_edges")

    def test_padding_shape(self):
        """Data should be padded to max dimensions."""
        max_nodes = 20
        max_edges = 50
        dataset = VariableLBracketDataset(
            num_samples=10, max_nodes=max_nodes, max_edges=max_edges, seed=42
        )
        data = dataset.get(0)

        assert data.x.shape == (max_nodes, 9)
        assert data.face_types.shape == (max_nodes,)
        assert data.edge_attr.shape == (max_edges, 2)
        assert data.node_mask.shape == (max_nodes,)
        assert data.edge_mask.shape == (max_edges,)

    def test_mask_validity(self):
        """Masks should correctly indicate real vs padding."""
        dataset = VariableLBracketDataset(num_samples=10, seed=42)
        data = dataset.get(0)

        num_real_nodes = data.num_real_nodes.item()
        num_real_edges = data.num_real_edges.item()

        # First num_real_nodes should be 1, rest should be 0
        assert data.node_mask[:num_real_nodes].sum() == num_real_nodes
        assert data.node_mask[num_real_nodes:].sum() == 0

        # Same for edges
        assert data.edge_mask[:num_real_edges].sum() == num_real_edges
        assert data.edge_mask[num_real_edges:].sum() == 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same samples."""
        dataset1 = VariableLBracketDataset(num_samples=10, seed=42)
        dataset2 = VariableLBracketDataset(num_samples=10, seed=42)

        data1 = dataset1.get(0)
        data2 = dataset2.get(0)

        torch.testing.assert_close(data1.x, data2.x)
        torch.testing.assert_close(data1.y, data2.y)
        torch.testing.assert_close(data1.node_mask, data2.node_mask)

    def test_different_seeds_different_samples(self):
        """Different seeds should produce different samples."""
        dataset1 = VariableLBracketDataset(num_samples=10, seed=42)
        dataset2 = VariableLBracketDataset(num_samples=10, seed=123)

        data1 = dataset1.get(0)
        data2 = dataset2.get(0)

        # Very unlikely to be equal with different seeds
        assert not torch.allclose(data1.y, data2.y)

    def test_caching(self):
        """Cached retrieval should return identical objects."""
        dataset = VariableLBracketDataset(num_samples=10, seed=42, cache_graphs=True)

        data1 = dataset.get(0)
        data2 = dataset.get(0)

        assert data1 is data2

    def test_y_normalized(self):
        """Target parameters should be normalized to [0, 1]."""
        dataset = VariableLBracketDataset(num_samples=20, seed=42)

        for i in range(20):
            data = dataset.get(i)
            assert data.y.min() >= 0.0
            assert data.y.max() <= 1.0

    def test_topology_variety(self):
        """Dataset should contain various topologies."""
        dataset = VariableLBracketDataset(num_samples=50, seed=42)

        node_counts = set()
        for i in range(50):
            data = dataset.get(i)
            node_counts.add(data.num_real_nodes.item())

        # Should have variety in node counts
        assert len(node_counts) >= 3

    def test_respects_ranges(self):
        """Dataset should respect custom parameter ranges."""
        ranges = VariableLBracketRanges(
            leg1_length=(60, 80),
            leg2_length=(70, 90),
        )
        dataset = VariableLBracketDataset(num_samples=20, ranges=ranges, seed=42)

        # The y values are normalized, so we can't directly check ranges
        # But we can verify the dataset was created with the ranges
        assert dataset.ranges.leg1_length == (60, 80)
        assert dataset.ranges.leg2_length == (70, 90)


class TestVariableDataLoaders:
    """Test variable topology data loader creation."""

    def test_creates_three_loaders(self):
        """Should create train, val, and test loaders."""
        train_loader, val_loader, test_loader = create_variable_data_loaders(
            train_size=50,
            val_size=10,
            test_size=10,
            batch_size=8,
            seed=42,
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_correct_sizes(self):
        """Loaders should have correct number of samples."""
        train_loader, val_loader, test_loader = create_variable_data_loaders(
            train_size=48,  # Divisible by batch_size
            val_size=16,
            test_size=16,
            batch_size=8,
            seed=42,
        )

        # Count samples using num_graphs (PyG batching flattens y)
        train_count = sum(batch.num_graphs for batch in train_loader)
        val_count = sum(batch.num_graphs for batch in val_loader)
        test_count = sum(batch.num_graphs for batch in test_loader)

        assert train_count == 48
        assert val_count == 16
        assert test_count == 16

    def test_no_overlap(self):
        """Train/val/test should not overlap (different seeds internally)."""
        # This is guaranteed by the implementation using different seed offsets
        train_loader, val_loader, test_loader = create_variable_data_loaders(
            train_size=10,
            val_size=10,
            test_size=10,
            batch_size=5,
            seed=42,
        )

        # Get first batch from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        # y values should differ (different brackets)
        assert not torch.allclose(train_batch.y, val_batch.y)

    def test_batching_works(self):
        """Batches should have correct dimensions."""
        batch_size = 8
        max_nodes = 20
        max_edges = 50

        train_loader, _, _ = create_variable_data_loaders(
            train_size=24,
            val_size=8,
            test_size=8,
            batch_size=batch_size,
            max_nodes=max_nodes,
            max_edges=max_edges,
            seed=42,
        )

        batch = next(iter(train_loader))

        # In PyG batching, x is flattened: (batch_size * max_nodes, features)
        assert batch.x.shape == (batch_size * max_nodes, 9)
        assert batch.face_types.shape == (batch_size * max_nodes,)
        assert batch.node_mask.shape == (batch_size * max_nodes,)
        # y is flattened in PyG batching, need to reshape
        assert batch.y.shape[0] == batch_size * 4  # 4 params per sample
        assert batch.num_graphs == batch_size


class TestDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_index_out_of_bounds(self):
        """Should raise IndexError for invalid indices."""
        dataset = VariableLBracketDataset(num_samples=10, seed=42)

        with pytest.raises(IndexError):
            dataset.get(10)

        with pytest.raises(IndexError):
            dataset.get(-1)

    def test_small_dataset(self):
        """Should work with very small datasets."""
        dataset = VariableLBracketDataset(num_samples=1, seed=42)
        data = dataset.get(0)

        assert data.x.shape[0] > 0

    def test_max_topology(self):
        """Should handle maximum topology (15 faces)."""
        # Set probabilities to maximize holes and fillet
        ranges = VariableLBracketRanges(
            prob_fillet=1.0,  # Always have fillet
            prob_hole_configs=[0.0, 0.0, 1.0],  # Always 2 holes per leg
        )

        dataset = VariableLBracketDataset(
            num_samples=5,
            ranges=ranges,
            max_nodes=20,  # Must accommodate 15 faces
            seed=42,
        )

        for i in range(5):
            data = dataset.get(i)
            # Should have up to 15 real nodes
            assert data.num_real_nodes.item() <= 15
            assert data.num_real_nodes.item() >= 6  # Minimum is 6
