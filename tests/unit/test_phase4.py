"""
Unit tests for Phase 4: Multi-Geometry B-Rep VAE.

Tests cover:
1. Geometry generators - Valid CAD solids with expected face counts
2. B-Rep extraction - V/E/F counts and feature dimensions
3. HeteroVAE model - Forward pass shapes, gradient flow, encode/decode
4. Parameter normalization - Normalize/denormalize roundtrips
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graph_cad.data.brep_types import (
    EDGE_TYPE_LINE,
    EDGE_TYPE_ARC,
    EDGE_TYPE_CIRCLE,
    FACE_TYPE_PLANAR,
    FACE_TYPE_HOLE,
    GEOMETRY_BRACKET,
    GEOMETRY_TUBE,
    GEOMETRY_CHANNEL,
    GEOMETRY_BLOCK,
    GEOMETRY_CYLINDER,
    GEOMETRY_BLOCKHOLE,
    NUM_GEOMETRY_TYPES,
    MAX_PARAMS,
    VERTEX_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    FACE_FEATURE_DIM,
)
from graph_cad.data.geometry_generators import (
    Tube,
    TubeRanges,
    Channel,
    ChannelRanges,
    Block,
    BlockRanges,
    Cylinder,
    CylinderRanges,
    BlockHole,
    BlockHoleRanges,
)
from graph_cad.data.brep_extraction import (
    BRepHeteroGraph,
    extract_brep_hetero_graph_from_solid,
)
from graph_cad.data.param_normalization import (
    normalize_params,
    denormalize_params,
    normalize_params_to_latent,
    denormalize_params_from_latent,
    pad_params,
    unpad_params,
    get_param_count,
    MultiGeometryNormalizer,
    PARAM_RANGES,
)


# =============================================================================
# Test Geometry Generators
# =============================================================================


class TestTubeGenerator:
    """Tests for Tube geometry generator."""

    def test_valid_tube_creation(self):
        """Test creating a valid tube."""
        tube = Tube(length=100, outer_dia=50, inner_dia=40)
        assert tube.length == 100
        assert tube.outer_dia == 50
        assert tube.inner_dia == 40

    def test_tube_to_solid(self):
        """Test tube generates valid solid."""
        tube = Tube(length=100, outer_dia=50, inner_dia=40)
        solid = tube.to_solid()
        assert solid is not None
        # Tube has 4 faces: outer cylinder, inner cylinder, 2 annular ends
        faces = solid.faces().vals()
        assert len(faces) == 4

    def test_tube_invalid_inner_greater_outer(self):
        """Test tube rejects inner_dia >= outer_dia."""
        with pytest.raises(ValueError, match="inner_dia.*must be < outer_dia"):
            Tube(length=100, outer_dia=50, inner_dia=60)

    def test_tube_invalid_thin_wall(self):
        """Test tube rejects wall thickness < 1mm."""
        with pytest.raises(ValueError, match="Wall thickness"):
            Tube(length=100, outer_dia=50, inner_dia=49)

    def test_tube_random(self):
        """Test random tube generation."""
        rng = np.random.default_rng(42)
        tube = Tube.random(rng)
        assert tube.length > 0
        assert tube.outer_dia > tube.inner_dia
        # Should generate valid solid
        solid = tube.to_solid()
        assert solid is not None

    def test_tube_to_dict_from_dict(self):
        """Test tube serialization roundtrip."""
        tube = Tube(length=100, outer_dia=50, inner_dia=40)
        d = tube.to_dict()
        tube2 = Tube.from_dict(d)
        assert tube2.length == tube.length
        assert tube2.outer_dia == tube.outer_dia
        assert tube2.inner_dia == tube.inner_dia


class TestChannelGenerator:
    """Tests for Channel geometry generator."""

    def test_valid_channel_creation(self):
        """Test creating a valid channel."""
        channel = Channel(width=50, height=50, length=100, thickness=5)
        assert channel.width == 50
        assert channel.height == 50
        assert channel.length == 100
        assert channel.thickness == 5

    def test_channel_to_solid(self):
        """Test channel generates valid solid."""
        channel = Channel(width=50, height=50, length=100, thickness=5)
        solid = channel.to_solid()
        assert solid is not None
        # C-channel has 10 faces (8 from C profile + 2 ends)
        faces = solid.faces().vals()
        assert len(faces) == 10

    def test_channel_invalid_thickness(self):
        """Test channel rejects thickness >= width/2."""
        with pytest.raises(ValueError, match="thickness.*must be < width/2"):
            Channel(width=50, height=50, length=100, thickness=30)

    def test_channel_random(self):
        """Test random channel generation."""
        rng = np.random.default_rng(42)
        channel = Channel.random(rng)
        assert channel.width > 0
        assert channel.thickness < channel.width / 2
        solid = channel.to_solid()
        assert solid is not None


class TestBlockGenerator:
    """Tests for Block geometry generator."""

    def test_valid_block_creation(self):
        """Test creating a valid block."""
        block = Block(length=100, width=50, height=30)
        assert block.length == 100
        assert block.width == 50
        assert block.height == 30

    def test_block_to_solid(self):
        """Test block generates valid solid."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        assert solid is not None
        # Block has 6 faces
        faces = solid.faces().vals()
        assert len(faces) == 6

    def test_block_random(self):
        """Test random block generation."""
        rng = np.random.default_rng(42)
        block = Block.random(rng)
        assert block.length > 0
        assert block.width > 0
        assert block.height > 0
        solid = block.to_solid()
        assert solid is not None


class TestCylinderGenerator:
    """Tests for Cylinder geometry generator."""

    def test_valid_cylinder_creation(self):
        """Test creating a valid cylinder."""
        cyl = Cylinder(length=100, diameter=50)
        assert cyl.length == 100
        assert cyl.diameter == 50

    def test_cylinder_to_solid(self):
        """Test cylinder generates valid solid."""
        cyl = Cylinder(length=100, diameter=50)
        solid = cyl.to_solid()
        assert solid is not None
        # Cylinder has 3 faces: curved surface + 2 circular ends
        faces = solid.faces().vals()
        assert len(faces) == 3

    def test_cylinder_random(self):
        """Test random cylinder generation."""
        rng = np.random.default_rng(42)
        cyl = Cylinder.random(rng)
        assert cyl.length > 0
        assert cyl.diameter > 0
        solid = cyl.to_solid()
        assert solid is not None


class TestBlockHoleGenerator:
    """Tests for BlockHole geometry generator."""

    def test_valid_blockhole_creation(self):
        """Test creating a valid block with hole."""
        bh = BlockHole(length=100, width=80, height=30, hole_dia=15, hole_x=0, hole_y=0)
        assert bh.length == 100
        assert bh.width == 80
        assert bh.height == 30
        assert bh.hole_dia == 15

    def test_blockhole_to_solid(self):
        """Test blockhole generates valid solid."""
        bh = BlockHole(length=100, width=80, height=30, hole_dia=15, hole_x=0, hole_y=0)
        solid = bh.to_solid()
        assert solid is not None
        # BlockHole has 7 faces: 6 box faces + 1 hole cylinder
        faces = solid.faces().vals()
        assert len(faces) == 7

    def test_blockhole_hole_position(self):
        """Test blockhole with off-center hole."""
        bh = BlockHole(length=100, width=80, height=30, hole_dia=15, hole_x=10, hole_y=-5)
        solid = bh.to_solid()
        assert solid is not None

    def test_blockhole_invalid_hole_too_large(self):
        """Test blockhole rejects hole that doesn't fit."""
        with pytest.raises(ValueError, match="hole_dia.*too large"):
            BlockHole(length=30, width=30, height=20, hole_dia=30, hole_x=0, hole_y=0)

    def test_blockhole_invalid_hole_position(self):
        """Test blockhole rejects hole position outside bounds."""
        with pytest.raises(ValueError, match="hole_x.*out of range"):
            BlockHole(length=100, width=80, height=30, hole_dia=15, hole_x=50, hole_y=0)

    def test_blockhole_random(self):
        """Test random blockhole generation."""
        rng = np.random.default_rng(42)
        bh = BlockHole.random(rng)
        assert bh.length > 0
        assert bh.hole_dia > 0
        solid = bh.to_solid()
        assert solid is not None


# =============================================================================
# Test B-Rep Extraction
# =============================================================================


class TestBRepExtraction:
    """Tests for B-Rep heterogeneous graph extraction."""

    def test_block_extraction(self):
        """Test B-Rep extraction from block."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert isinstance(graph, BRepHeteroGraph)
        # Block: 8 vertices, 12 edges, 6 faces
        assert graph.num_vertices == 8
        assert graph.num_edges == 12
        assert graph.num_faces == 6

    def test_cylinder_extraction(self):
        """Test B-Rep extraction from cylinder."""
        cyl = Cylinder(length=100, diameter=50)
        solid = cyl.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert isinstance(graph, BRepHeteroGraph)
        # Cylinder: 2 vertices (centers), 3 edges, 3 faces
        assert graph.num_vertices == 2
        assert graph.num_edges == 3
        assert graph.num_faces == 3

    def test_tube_extraction(self):
        """Test B-Rep extraction from tube."""
        tube = Tube(length=100, outer_dia=50, inner_dia=40)
        solid = tube.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert isinstance(graph, BRepHeteroGraph)
        # Tube: 4 vertices, 6 edges, 4 faces
        assert graph.num_vertices == 4
        assert graph.num_edges == 6
        assert graph.num_faces == 4

    def test_vertex_features_shape(self):
        """Test vertex features have correct shape."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert graph.vertex_features.shape == (graph.num_vertices, VERTEX_FEATURE_DIM)
        assert graph.vertex_features.dtype == np.float32

    def test_edge_features_shape(self):
        """Test edge features have correct shape."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert graph.edge_features.shape == (graph.num_edges, EDGE_FEATURE_DIM)
        assert graph.edge_features.dtype == np.float32

    def test_face_features_shape(self):
        """Test face features have correct shape."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert graph.face_features.shape == (graph.num_faces, FACE_FEATURE_DIM)
        assert graph.face_features.dtype == np.float32

    def test_edge_types_valid(self):
        """Test edge types are valid indices."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert graph.edge_types.shape == (graph.num_edges,)
        assert all(0 <= t <= 3 for t in graph.edge_types)
        # Block should have all LINE edges
        assert all(t == EDGE_TYPE_LINE for t in graph.edge_types)

    def test_face_types_valid(self):
        """Test face types are valid indices."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert graph.face_types.shape == (graph.num_faces,)
        assert all(0 <= t <= 2 for t in graph.face_types)
        # Block should have all PLANAR faces
        assert all(t == FACE_TYPE_PLANAR for t in graph.face_types)

    def test_cylinder_has_cylindrical_face(self):
        """Test cylinder extraction identifies cylindrical face."""
        cyl = Cylinder(length=100, diameter=50)
        solid = cyl.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        # Should have at least one HOLE type (cylindrical surface)
        assert FACE_TYPE_HOLE in graph.face_types

    def test_topology_vertex_to_edge(self):
        """Test vertex-to-edge topology is valid."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        v2e = graph.vertex_to_edge
        assert v2e.shape[0] == 2
        assert v2e.shape[1] > 0
        # All vertex indices should be valid
        assert all(0 <= idx < graph.num_vertices for idx in v2e[0])
        # All edge indices should be valid
        assert all(0 <= idx < graph.num_edges for idx in v2e[1])

    def test_topology_edge_to_face(self):
        """Test edge-to-face topology is valid."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        e2f = graph.edge_to_face
        assert e2f.shape[0] == 2
        assert e2f.shape[1] > 0
        # All edge indices should be valid
        assert all(0 <= idx < graph.num_edges for idx in e2f[0])
        # All face indices should be valid
        assert all(0 <= idx < graph.num_faces for idx in e2f[1])

    def test_bbox_diagonal_positive(self):
        """Test bounding box diagonal is positive."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert graph.bbox_diagonal > 0

    def test_features_finite(self):
        """Test all features are finite (no NaN/Inf)."""
        block = Block(length=100, width=50, height=30)
        solid = block.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)

        assert np.all(np.isfinite(graph.vertex_features))
        assert np.all(np.isfinite(graph.edge_features))
        assert np.all(np.isfinite(graph.face_features))


class TestBRepExtractionAllGeometries:
    """Test B-Rep extraction works for all geometry types."""

    @pytest.fixture
    def geometries(self):
        """Create one of each geometry type."""
        return [
            ("Block", Block(100, 50, 30)),
            ("Tube", Tube(100, 50, 40)),
            ("Channel", Channel(50, 50, 100, 5)),
            ("Cylinder", Cylinder(100, 50)),
            ("BlockHole", BlockHole(100, 80, 30, 15, 0, 0)),
        ]

    def test_all_geometries_extract(self, geometries):
        """Test all geometry types can be extracted."""
        for name, geom in geometries:
            solid = geom.to_solid()
            graph = extract_brep_hetero_graph_from_solid(solid)
            assert graph.num_faces > 0, f"{name} should have faces"
            assert graph.num_edges > 0, f"{name} should have edges"
            assert graph.num_vertices > 0, f"{name} should have vertices"

    def test_all_geometries_valid_features(self, geometries):
        """Test all geometry types produce valid features."""
        for name, geom in geometries:
            solid = geom.to_solid()
            graph = extract_brep_hetero_graph_from_solid(solid)
            assert np.all(np.isfinite(graph.vertex_features)), f"{name} vertex features not finite"
            assert np.all(np.isfinite(graph.edge_features)), f"{name} edge features not finite"
            assert np.all(np.isfinite(graph.face_features)), f"{name} face features not finite"


# =============================================================================
# Test Parameter Normalization
# =============================================================================


class TestParameterNormalization:
    """Tests for parameter normalization functions."""

    def test_normalize_denormalize_roundtrip_bracket(self):
        """Test normalize/denormalize roundtrip for bracket."""
        params = torch.tensor([100.0, 150.0, 40.0, 8.0])
        normalized = normalize_params(params, GEOMETRY_BRACKET)
        denormalized = denormalize_params(normalized, GEOMETRY_BRACKET)
        torch.testing.assert_close(denormalized, params, atol=1e-5, rtol=1e-5)

    def test_normalize_denormalize_roundtrip_tube(self):
        """Test normalize/denormalize roundtrip for tube."""
        params = torch.tensor([100.0, 50.0, 40.0])
        normalized = normalize_params(params, GEOMETRY_TUBE)
        denormalized = denormalize_params(normalized, GEOMETRY_TUBE)
        torch.testing.assert_close(denormalized, params, atol=1e-5, rtol=1e-5)

    def test_normalize_denormalize_roundtrip_blockhole(self):
        """Test normalize/denormalize roundtrip for blockhole (6 params)."""
        params = torch.tensor([100.0, 80.0, 40.0, 15.0, 10.0, -5.0])
        normalized = normalize_params(params, GEOMETRY_BLOCKHOLE)
        denormalized = denormalize_params(normalized, GEOMETRY_BLOCKHOLE)
        torch.testing.assert_close(denormalized, params, atol=1e-5, rtol=1e-5)

    def test_normalized_range_is_zero_one(self):
        """Test normalized values are in [0, 1] for min/max inputs."""
        # Use min values
        min_params = torch.tensor([50.0, 50.0, 20.0, 3.0])  # Bracket mins
        normalized = normalize_params(min_params, GEOMETRY_BRACKET)
        torch.testing.assert_close(normalized, torch.zeros(4), atol=1e-5, rtol=1e-5)

        # Use max values
        max_params = torch.tensor([200.0, 200.0, 60.0, 12.0])  # Bracket maxs
        normalized = normalize_params(max_params, GEOMETRY_BRACKET)
        torch.testing.assert_close(normalized, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_latent_normalization_roundtrip(self):
        """Test latent normalization roundtrip ([-2, 2] range)."""
        params = torch.tensor([100.0, 150.0, 40.0, 8.0])
        latent = normalize_params_to_latent(params, GEOMETRY_BRACKET)
        recovered = denormalize_params_from_latent(latent, GEOMETRY_BRACKET)
        torch.testing.assert_close(recovered, params, atol=1e-4, rtol=1e-4)

    def test_latent_normalization_range(self):
        """Test latent normalized values are approximately in [-2, 2]."""
        # Mid-range values should be near 0
        mid_params = torch.tensor([125.0, 125.0, 40.0, 7.5])  # Bracket midpoints
        latent = normalize_params_to_latent(mid_params, GEOMETRY_BRACKET)
        # Should be close to 0 (within [-2, 2] for sure)
        assert all(-2.1 <= v <= 2.1 for v in latent)


class TestPadParams:
    """Tests for parameter padding functions."""

    def test_pad_params_bracket(self):
        """Test padding bracket params to max size."""
        params = torch.tensor([100.0, 150.0, 40.0, 8.0])
        padded, mask = pad_params(params, GEOMETRY_BRACKET)

        assert padded.shape == (MAX_PARAMS,)
        assert mask.shape == (MAX_PARAMS,)
        # First 4 should be real
        assert mask[:4].sum() == 4
        # Last 2 should be padding
        assert mask[4:].sum() == 0

    def test_pad_params_cylinder(self):
        """Test padding cylinder params (only 2)."""
        params = torch.tensor([100.0, 50.0])
        padded, mask = pad_params(params, GEOMETRY_CYLINDER)

        assert padded.shape == (MAX_PARAMS,)
        assert mask[:2].sum() == 2
        assert mask[2:].sum() == 0

    def test_unpad_params(self):
        """Test unpadding returns correct size."""
        padded = torch.tensor([100.0, 50.0, 0.0, 0.0, 0.0, 0.0])
        unpadded = unpad_params(padded, GEOMETRY_CYLINDER)

        assert unpadded.shape == (2,)
        torch.testing.assert_close(unpadded, torch.tensor([100.0, 50.0]))

    def test_get_param_count(self):
        """Test get_param_count returns correct values."""
        assert get_param_count(GEOMETRY_BRACKET) == 4
        assert get_param_count(GEOMETRY_TUBE) == 3
        assert get_param_count(GEOMETRY_CHANNEL) == 4
        assert get_param_count(GEOMETRY_BLOCK) == 3
        assert get_param_count(GEOMETRY_CYLINDER) == 2
        assert get_param_count(GEOMETRY_BLOCKHOLE) == 6


class TestMultiGeometryNormalizer:
    """Tests for batch normalizer."""

    def test_normalize_batch_mixed_types(self):
        """Test batch normalization with mixed geometry types."""
        normalizer = MultiGeometryNormalizer()

        # Create batch with different types
        params = torch.tensor([
            [100.0, 150.0, 40.0, 8.0, 0.0, 0.0],  # Bracket
            [100.0, 50.0, 40.0, 0.0, 0.0, 0.0],   # Tube
            [100.0, 50.0, 30.0, 0.0, 0.0, 0.0],   # Block
        ])
        types = torch.tensor([GEOMETRY_BRACKET, GEOMETRY_TUBE, GEOMETRY_BLOCK])

        normalized = normalizer.normalize_batch(params, types)
        denormalized = normalizer.denormalize_batch(normalized, types)

        # First 4 params of bracket should roundtrip
        torch.testing.assert_close(denormalized[0, :4], params[0, :4], atol=1e-5, rtol=1e-5)
        # First 3 params of tube should roundtrip
        torch.testing.assert_close(denormalized[1, :3], params[1, :3], atol=1e-5, rtol=1e-5)
        # First 3 params of block should roundtrip
        torch.testing.assert_close(denormalized[2, :3], params[2, :3], atol=1e-5, rtol=1e-5)


# =============================================================================
# Test HeteroVAE Model
# =============================================================================


class TestHeteroVAEModel:
    """Tests for HeteroVAE model."""

    @pytest.fixture
    def model(self):
        """Create a small HeteroVAE for testing."""
        from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig

        config = HeteroVAEConfig(
            latent_dim=16,
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
        )
        return HeteroVAE(config, use_param_head=True, num_params=6)

    @pytest.fixture
    def sample_hetero_data(self):
        """Create sample HeteroData for testing."""
        from torch_geometric.data import HeteroData

        data = HeteroData()

        # Node features
        data['vertex'].x = torch.randn(8, VERTEX_FEATURE_DIM)
        data['edge'].x = torch.randn(12, EDGE_FEATURE_DIM)
        data['face'].x = torch.randn(6, FACE_FEATURE_DIM)

        # Node types
        data['edge'].edge_type = torch.zeros(12, dtype=torch.long)
        data['face'].face_type = torch.zeros(6, dtype=torch.long)

        # Topology edges
        data['vertex', 'bounds', 'edge'].edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ], dtype=torch.long)
        data['edge', 'bounded_by', 'vertex'].edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
        ], dtype=torch.long)
        data['edge', 'bounds', 'face'].edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ], dtype=torch.long)
        data['face', 'bounded_by', 'edge'].edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ], dtype=torch.long)

        # Metadata
        data.num_vertices = torch.tensor([8])
        data.num_edges = torch.tensor([12])
        data.num_faces = torch.tensor([6])

        return data

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model is not None
        assert model.config.latent_dim == 16

    def test_encode_output_shapes(self, model, sample_hetero_data):
        """Test encoder output shapes."""
        mu, logvar = model.encode(sample_hetero_data)

        assert mu.shape == (1, 16)  # batch=1, latent=16
        assert logvar.shape == (1, 16)

    def test_encode_outputs_finite(self, model, sample_hetero_data):
        """Test encoder outputs are finite."""
        mu, logvar = model.encode(sample_hetero_data)

        assert torch.all(torch.isfinite(mu))
        assert torch.all(torch.isfinite(logvar))

    def test_decode_output_shapes(self, model):
        """Test decoder output shapes."""
        z = torch.randn(2, 16)  # batch=2
        outputs = model.decode(z)

        assert 'node_features' in outputs
        assert 'face_type_logits' in outputs
        assert 'existence_logits' in outputs
        assert 'edge_logits' in outputs

        # Check shapes (max_faces=20 from decoder config)
        assert outputs['node_features'].shape[0] == 2
        assert outputs['node_features'].shape[2] == FACE_FEATURE_DIM

    def test_forward_returns_expected_keys(self, model, sample_hetero_data):
        """Test forward pass returns expected keys."""
        outputs = model(sample_hetero_data)

        expected_keys = ['node_features', 'face_type_logits', 'existence_logits',
                        'edge_logits', 'mu', 'logvar', 'z', 'param_pred']
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"

    def test_forward_gradient_flow(self, model, sample_hetero_data):
        """Test gradients flow through forward pass."""
        model.train()
        outputs = model(sample_hetero_data)

        # Create dummy loss
        loss = outputs['node_features'].sum() + outputs['mu'].sum()
        loss.backward()

        # Check encoder has gradients
        for param in model.encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Encoder should have gradients"
                break

    def test_reparameterize_train_vs_eval(self, model):
        """Test reparameterization differs between train/eval."""
        mu = torch.randn(4, 16)
        logvar = torch.zeros(4, 16)  # std=1

        model.train()
        z_train = model.reparameterize(mu, logvar)

        model.eval()
        z_eval = model.reparameterize(mu, logvar)

        # In eval mode, should return mu exactly
        torch.testing.assert_close(z_eval, mu)
        # In train mode, should be different (with high probability)
        assert not torch.allclose(z_train, mu)

    def test_sample_from_prior(self, model):
        """Test sampling from prior."""
        samples = model.sample(num_samples=4, device='cpu')

        assert 'node_features' in samples
        assert samples['node_features'].shape[0] == 4


class TestHeteroVAEGradientFlow:
    """Test gradient flow through HeteroVAE components."""

    def test_encoder_gradient_flow(self):
        """Test gradients flow through encoder."""
        from graph_cad.models.hetero_vae import HeteroGNNEncoder, HeteroVAEConfig
        from torch_geometric.data import HeteroData

        config = HeteroVAEConfig(latent_dim=8, hidden_dim=16, num_layers=1)
        encoder = HeteroGNNEncoder(config)

        # Create minimal input
        x_dict = {
            'vertex': torch.randn(4, VERTEX_FEATURE_DIM, requires_grad=True),
            'edge': torch.randn(6, EDGE_FEATURE_DIM, requires_grad=True),
            'face': torch.randn(3, FACE_FEATURE_DIM, requires_grad=True),
            'edge_type': torch.zeros(6, dtype=torch.long),
            'face_type': torch.zeros(3, dtype=torch.long),
        }
        edge_index_dict = {
            ('vertex', 'bounds', 'edge'): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            ('edge', 'bounded_by', 'vertex'): torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
            ('edge', 'bounds', 'face'): torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]]),
            ('face', 'bounded_by', 'edge'): torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 5]]),
        }

        mu, logvar = encoder(x_dict, edge_index_dict)
        loss = mu.sum() + logvar.sum()
        loss.backward()

        # Check parameter gradients exist
        has_grad = False
        for param in encoder.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "Encoder should have gradients"
