"""Unit tests for graph extraction from B-Rep solids."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from graph_cad.data import (
    BRepGraph,
    LBracket,
    extract_graph,
    extract_graph_from_solid,
)
from graph_cad.data.graph_extraction import extract_graph_from_solid_variable
from graph_cad.data.l_bracket import VariableLBracket


@pytest.fixture
def sample_bracket():
    """Create a sample L-bracket for testing."""
    return LBracket(
        leg1_length=100,
        leg2_length=80,
        width=30,
        thickness=5,
        hole1_distance=20,
        hole1_diameter=8,
        hole2_distance=15,
        hole2_diameter=6,
    )


@pytest.fixture
def sample_bracket_solid(sample_bracket):
    """Create CadQuery solid from sample bracket."""
    return sample_bracket.to_solid()


@pytest.fixture
def sample_bracket_step(sample_bracket):
    """Create temporary STEP file from sample bracket."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_bracket.step"
        sample_bracket.to_step(path)
        yield path


class TestExtractGraphFromSolid:
    """Test graph extraction from CadQuery solid."""

    def test_returns_brep_graph(self, sample_bracket_solid):
        """Should return BRepGraph instance."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert isinstance(graph, BRepGraph)

    def test_has_10_faces(self, sample_bracket_solid):
        """L-bracket should have exactly 10 faces."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.num_faces == 10

    def test_node_features_shape(self, sample_bracket_solid):
        """Node features should have shape (10, 8)."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.node_features.shape == (10, 8)

    def test_edge_index_shape(self, sample_bracket_solid):
        """Edge index should have shape (2, num_edges)."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] == graph.num_edges

    def test_edge_features_shape(self, sample_bracket_solid):
        """Edge features should have shape (num_edges, 2)."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.edge_features.shape == (graph.num_edges, 2)

    def test_edge_indices_valid(self, sample_bracket_solid):
        """All edge indices should be valid face indices."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert np.all(graph.edge_index >= 0)
        assert np.all(graph.edge_index < graph.num_faces)

    def test_has_edges(self, sample_bracket_solid):
        """Graph should have some edges (faces are connected)."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.num_edges > 0

    def test_bbox_diagonal_positive(self, sample_bracket_solid):
        """Bounding box diagonal should be positive."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.bbox_diagonal > 0

    def test_bbox_center_shape(self, sample_bracket_solid):
        """Bounding box center should be 3D vector."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        assert graph.bbox_center.shape == (3,)


class TestNodeFeatures:
    """Test node feature extraction."""

    def test_face_types_correct(self, sample_bracket_solid):
        """Should have 8 planar faces and 2 cylindrical faces."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        face_types = graph.node_features[:, 0]

        # Count face types (0=planar, 1=cylindrical)
        num_planar = np.sum(face_types == 0)
        num_cylindrical = np.sum(face_types == 1)

        assert num_planar == 8, f"Expected 8 planar faces, got {num_planar}"
        assert num_cylindrical == 2, f"Expected 2 cylindrical faces, got {num_cylindrical}"

    def test_areas_positive(self, sample_bracket_solid):
        """All face areas should be positive."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        areas = graph.node_features[:, 1]
        assert np.all(areas > 0)

    def test_normals_unit_vectors(self, sample_bracket_solid):
        """Normal vectors should be approximately unit length."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        normals = graph.node_features[:, 2:5]
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

    def test_centroids_normalized(self, sample_bracket_solid):
        """Centroid coordinates should be roughly in [-1, 1] range."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        centroids = graph.node_features[:, 5:8]
        # After normalization by bbox diagonal, values should be bounded
        assert np.all(np.abs(centroids) < 2.0)


class TestEdgeFeatures:
    """Test edge feature extraction."""

    def test_edge_lengths_positive(self, sample_bracket_solid):
        """All edge lengths should be positive."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        if graph.num_edges > 0:
            edge_lengths = graph.edge_features[:, 0]
            assert np.all(edge_lengths > 0)

    def test_dihedral_angles_in_range(self, sample_bracket_solid):
        """Dihedral angles should be in [0, π]."""
        graph = extract_graph_from_solid(sample_bracket_solid)
        if graph.num_edges > 0:
            dihedral_angles = graph.edge_features[:, 1]
            assert np.all(dihedral_angles >= 0)
            assert np.all(dihedral_angles <= np.pi + 0.01)


class TestExtractGraphFromFile:
    """Test graph extraction from STEP file."""

    def test_extract_from_step_file(self, sample_bracket_step):
        """Should successfully extract graph from STEP file."""
        graph = extract_graph(sample_bracket_step)
        assert isinstance(graph, BRepGraph)
        assert graph.num_faces == 10

    def test_source_file_set(self, sample_bracket_step):
        """Source file path should be set when loading from file."""
        graph = extract_graph(sample_bracket_step)
        assert graph.source_file is not None
        assert "test_bracket.step" in graph.source_file

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            extract_graph("/nonexistent/path/file.step")


class TestGraphConsistency:
    """Test graph extraction consistency across methods."""

    def test_solid_vs_step_same_topology(self, sample_bracket, sample_bracket_step):
        """Graph from solid and STEP should have same topology."""
        solid = sample_bracket.to_solid()
        graph_from_solid = extract_graph_from_solid(solid)
        graph_from_step = extract_graph(sample_bracket_step)

        assert graph_from_solid.num_faces == graph_from_step.num_faces
        assert graph_from_solid.num_edges == graph_from_step.num_edges

    def test_deterministic_extraction(self, sample_bracket_solid):
        """Multiple extractions should produce identical results."""
        graph1 = extract_graph_from_solid(sample_bracket_solid)
        graph2 = extract_graph_from_solid(sample_bracket_solid)

        np.testing.assert_array_equal(graph1.node_features, graph2.node_features)
        np.testing.assert_array_equal(graph1.edge_index, graph2.edge_index)
        np.testing.assert_array_equal(graph1.edge_features, graph2.edge_features)


class TestMultipleBrackets:
    """Test extraction across different bracket configurations."""

    def test_different_dimensions_same_topology(self):
        """Brackets with different dimensions should have same topology."""
        bracket1 = LBracket(
            leg1_length=50,
            leg2_length=50,
            width=20,
            thickness=3,
            hole1_distance=10,
            hole1_diameter=4,
            hole2_distance=10,
            hole2_diameter=4,
        )
        bracket2 = LBracket(
            leg1_length=200,
            leg2_length=150,
            width=60,
            thickness=12,
            hole1_distance=40,
            hole1_diameter=12,
            hole2_distance=30,
            hole2_diameter=10,
        )

        graph1 = extract_graph_from_solid(bracket1.to_solid())
        graph2 = extract_graph_from_solid(bracket2.to_solid())

        # Same topology (number of faces/edges)
        assert graph1.num_faces == graph2.num_faces == 10
        # Face types should match
        np.testing.assert_array_equal(
            np.sort(graph1.node_features[:, 0]),
            np.sort(graph2.node_features[:, 0]),
        )

    def test_random_brackets_valid_graphs(self):
        """Random brackets should all produce valid graphs."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            bracket = LBracket.random(rng)
            graph = extract_graph_from_solid(bracket.to_solid())

            assert graph.num_faces == 10
            assert graph.num_edges > 0
            assert graph.node_features.shape == (10, 8)


# =============================================================================
# Variable Topology Graph Extraction Tests
# =============================================================================


@pytest.fixture
def variable_bracket_minimal():
    """Create minimal variable bracket (no holes, no fillet)."""
    return VariableLBracket(
        leg1_length=100,
        leg2_length=80,
        width=30,
        thickness=5,
    )


@pytest.fixture
def variable_bracket_with_holes():
    """Create variable bracket with holes."""
    return VariableLBracket(
        leg1_length=100,
        leg2_length=80,
        width=30,
        thickness=5,
        hole1_diameters=(8,),
        hole1_distances=(20,),
        hole2_diameters=(6, 6),
        hole2_distances=(15, 45),
    )


class TestExtractGraphFromSolidVariable:
    """Test variable topology graph extraction."""

    def test_returns_brep_graph(self, variable_bracket_minimal):
        """Should return BRepGraph instance."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        assert isinstance(graph, BRepGraph)

    def test_minimal_bracket_has_at_least_6_faces(self, variable_bracket_minimal):
        """Minimal L-bracket should have at least 6 faces."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        # CadQuery may create additional faces for L-shape geometry
        assert graph.num_faces >= 6

    def test_bracket_with_holes_has_more_faces(self, variable_bracket_with_holes):
        """Bracket with holes should have additional faces."""
        solid = variable_bracket_with_holes.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        # Should have more faces than base due to holes
        assert graph.num_faces >= 8

    def test_node_features_shape(self, variable_bracket_minimal):
        """Node features should have shape (num_faces, 9)."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        # 9 features per node
        assert graph.node_features.shape[1] == 9
        assert graph.node_features.shape[0] == graph.num_faces

    def test_node_features_include_curvature(self, variable_bracket_with_holes):
        """Node features should include curvature values."""
        solid = variable_bracket_with_holes.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        # Features: area(1), direction(3), centroid(3), curv1(1), curv2(1) = 9
        assert graph.node_features.shape[1] == 9
        # Curvatures are in columns 7 and 8
        curvatures = graph.node_features[:, 7:9]
        assert curvatures.shape == (graph.num_faces, 2)

    def test_face_types_separate_array(self, variable_bracket_minimal):
        """Face types should be stored as separate integer array."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        assert hasattr(graph, "face_types")
        assert graph.face_types.shape[0] == graph.num_faces
        assert graph.face_types.dtype in [np.int32, np.int64]

    def test_face_types_planar_for_minimal(self, variable_bracket_minimal):
        """Minimal bracket should have all planar faces (type 0)."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        # All faces should be planar (no holes or fillets)
        assert np.all(graph.face_types == 0)

    def test_face_types_cylindrical_for_holes(self, variable_bracket_with_holes):
        """Bracket with holes should have cylindrical face types."""
        solid = variable_bracket_with_holes.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        # Should have some cylindrical faces (type 1) from holes
        num_cylindrical = np.sum(graph.face_types == 1)
        # Should have at least some cylindrical faces
        assert num_cylindrical >= 3

    def test_edge_index_valid(self, variable_bracket_minimal):
        """Edge indices should be valid face indices."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        assert np.all(graph.edge_index >= 0)
        assert np.all(graph.edge_index < graph.num_faces)

    def test_edge_features_shape(self, variable_bracket_minimal):
        """Edge features should have shape (num_edges, 2)."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        assert graph.edge_features.shape == (graph.num_edges, 2)

    def test_bbox_diagonal_positive(self, variable_bracket_minimal):
        """Bounding box diagonal should be positive."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        assert graph.bbox_diagonal > 0


class TestVariableGraphCurvature:
    """Test curvature extraction for variable topology."""

    def test_planar_faces_zero_curvature(self, variable_bracket_minimal):
        """Planar faces should have approximately zero curvature."""
        solid = variable_bracket_minimal.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        curvatures = graph.node_features[:, 7:9]
        # All planar faces should have near-zero curvature
        np.testing.assert_allclose(curvatures, 0.0, atol=0.01)

    def test_cylindrical_faces_nonzero_curvature(self, variable_bracket_with_holes):
        """Cylindrical faces should have non-zero curvature."""
        solid = variable_bracket_with_holes.to_solid()
        graph = extract_graph_from_solid_variable(solid)
        cylindrical_mask = graph.face_types == 1
        curvatures = graph.node_features[cylindrical_mask, 7:9]
        # At least one principal curvature should be non-zero for cylinders
        max_curvature = np.max(np.abs(curvatures))
        assert max_curvature > 0


class TestVariableTopologyVariety:
    """Test graph extraction across different topologies."""

    def test_random_variable_brackets_valid_graphs(self):
        """Random variable brackets should produce valid graphs."""
        rng = np.random.default_rng(42)

        for _ in range(20):
            bracket = VariableLBracket.random(rng)
            solid = bracket.to_solid()
            graph = extract_graph_from_solid_variable(solid)

            # Should have at least 6 faces
            assert graph.num_faces >= 6
            # Node features should be 9D
            assert graph.node_features.shape[1] == 9
            # Face types should match num_faces
            assert graph.face_types.shape[0] == graph.num_faces
            # Should have some edges
            assert graph.num_edges > 0

    def test_topology_variety(self):
        """Different topologies should produce different graphs."""
        rng = np.random.default_rng(42)

        face_counts = set()
        for _ in range(50):
            bracket = VariableLBracket.random(rng)
            solid = bracket.to_solid()
            graph = extract_graph_from_solid_variable(solid)
            face_counts.add(graph.num_faces)

        # Should see variety in face counts
        assert len(face_counts) >= 3

    def test_deterministic_extraction_variable(self):
        """Variable graph extraction should be deterministic."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_diameters=(8,),
            hole1_distances=(20,),
        )
        solid = bracket.to_solid()

        graph1 = extract_graph_from_solid_variable(solid)
        graph2 = extract_graph_from_solid_variable(solid)

        np.testing.assert_array_equal(graph1.node_features, graph2.node_features)
        np.testing.assert_array_equal(graph1.face_types, graph2.face_types)
        np.testing.assert_array_equal(graph1.edge_index, graph2.edge_index)
