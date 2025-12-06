"""Unit tests for parameter regressor model."""

import numpy as np
import pytest

# Skip all tests in this module if PyTorch is not installed
torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from graph_cad.data import LBracket, extract_graph_from_solid
from graph_cad.models import (
    PARAMETER_NAMES,
    ParameterRegressor,
    ParameterRegressorConfig,
    brep_graph_to_pyg,
    denormalize_parameters,
    normalize_parameters,
)


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
def sample_graph(sample_bracket):
    """Extract graph from sample bracket."""
    return extract_graph_from_solid(sample_bracket.to_solid())


@pytest.fixture
def sample_pyg_data(sample_graph):
    """Convert sample graph to PyG format."""
    return brep_graph_to_pyg(sample_graph)


@pytest.fixture
def model():
    """Create a parameter regressor model."""
    return ParameterRegressor()


class TestBrepGraphToPyg:
    """Test conversion from BRepGraph to PyG tensors."""

    def test_returns_dict_with_keys(self, sample_graph):
        """Should return dict with x, edge_index, edge_attr."""
        data = brep_graph_to_pyg(sample_graph)
        assert "x" in data
        assert "edge_index" in data
        assert "edge_attr" in data

    def test_node_features_shape(self, sample_graph):
        """Node features should have correct shape."""
        data = brep_graph_to_pyg(sample_graph)
        assert data["x"].shape == (10, 8)

    def test_edge_index_shape(self, sample_graph):
        """Edge index should have shape (2, num_edges)."""
        data = brep_graph_to_pyg(sample_graph)
        assert data["edge_index"].shape[0] == 2
        assert data["edge_index"].shape[1] == sample_graph.num_edges

    def test_edge_features_shape(self, sample_graph):
        """Edge features should have shape (num_edges, 2)."""
        data = brep_graph_to_pyg(sample_graph)
        assert data["edge_attr"].shape == (sample_graph.num_edges, 2)

    def test_tensors_are_correct_dtype(self, sample_graph):
        """Tensors should have correct dtypes."""
        data = brep_graph_to_pyg(sample_graph)
        assert data["x"].dtype == torch.float32
        assert data["edge_index"].dtype == torch.long
        assert data["edge_attr"].dtype == torch.float32


class TestParameterRegressorConfig:
    """Test model configuration."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = ParameterRegressorConfig()
        assert config.node_features == 8
        assert config.edge_features == 2
        assert config.num_parameters == 8

    def test_custom_config(self):
        """Custom config values should be stored."""
        config = ParameterRegressorConfig(hidden_dim=128, num_layers=5)
        assert config.hidden_dim == 128
        assert config.num_layers == 5


class TestParameterRegressor:
    """Test the ParameterRegressor model."""

    def test_model_creation(self):
        """Model should be created without error."""
        model = ParameterRegressor()
        assert isinstance(model, torch.nn.Module)

    def test_model_with_custom_config(self):
        """Model should accept custom config."""
        config = ParameterRegressorConfig(hidden_dim=32, num_layers=2)
        model = ParameterRegressor(config)
        assert model.config.hidden_dim == 32

    def test_forward_single_graph(self, model, sample_pyg_data):
        """Forward pass on single graph should return (1, 8) tensor."""
        model.eval()
        with torch.no_grad():
            output = model(
                sample_pyg_data["x"],
                sample_pyg_data["edge_index"],
                sample_pyg_data["edge_attr"],
            )
        assert output.shape == (1, 8)

    def test_forward_batched(self, model, sample_pyg_data):
        """Forward pass with batch should return (batch_size, 8) tensor."""
        # Create a batch of 3 identical graphs
        x = sample_pyg_data["x"].repeat(3, 1)
        edge_index = torch.cat(
            [sample_pyg_data["edge_index"] + i * 10 for i in range(3)], dim=1
        )
        edge_attr = sample_pyg_data["edge_attr"].repeat(3, 1)
        batch = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)

        model.eval()
        with torch.no_grad():
            output = model(x, edge_index, edge_attr, batch)

        assert output.shape == (3, 8)

    def test_output_is_differentiable(self, model, sample_pyg_data):
        """Output should be differentiable for training."""
        output = model(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestParameterNormalization:
    """Test parameter normalization utilities."""

    def test_normalize_parameters(self):
        """Normalized parameters should be in [0, 1] range."""
        # Mid-range values
        params = torch.tensor([[125.0, 125.0, 40.0, 7.5, 92.0, 8.0, 92.0, 8.0]])
        normalized = normalize_parameters(params)

        assert normalized.shape == (1, 8)
        assert torch.all(normalized >= 0)
        assert torch.all(normalized <= 1)

    def test_denormalize_parameters(self):
        """Denormalized parameters should recover original values."""
        params = torch.tensor([[100.0, 80.0, 30.0, 5.0, 20.0, 8.0, 15.0, 6.0]])
        normalized = normalize_parameters(params)
        recovered = denormalize_parameters(normalized)

        torch.testing.assert_close(recovered, params, rtol=1e-5, atol=1e-5)

    def test_normalize_denormalize_roundtrip(self):
        """Normalize then denormalize should be identity."""
        params = torch.tensor([[75.0, 150.0, 45.0, 8.0, 50.0, 10.0, 40.0, 7.0]])
        roundtrip = denormalize_parameters(normalize_parameters(params))
        torch.testing.assert_close(roundtrip, params, rtol=1e-5, atol=1e-5)


class TestParameterNames:
    """Test parameter name constants."""

    def test_parameter_names_count(self):
        """Should have 8 parameter names."""
        assert len(PARAMETER_NAMES) == 8

    def test_parameter_names_match_lbracket(self):
        """Parameter names should match LBracket attributes."""
        bracket = LBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_distance=20,
            hole1_diameter=8,
            hole2_distance=15,
            hole2_diameter=6,
        )
        params_dict = bracket.to_dict()

        for name in PARAMETER_NAMES:
            assert name in params_dict


class TestEndToEnd:
    """End-to-end tests for the parameter regressor pipeline."""

    def test_bracket_to_prediction_pipeline(self, sample_bracket, model):
        """Full pipeline: bracket → graph → pyg → model → params."""
        # Extract graph
        graph = extract_graph_from_solid(sample_bracket.to_solid())

        # Convert to PyG format
        data = brep_graph_to_pyg(graph)

        # Run model
        model.eval()
        with torch.no_grad():
            predicted = model(data["x"], data["edge_index"], data["edge_attr"])

        # Should output 8 parameters
        assert predicted.shape == (1, 8)

        # Denormalize
        predicted_mm = denormalize_parameters(predicted)
        assert predicted_mm.shape == (1, 8)

    def test_multiple_brackets_same_topology(self, model):
        """Different brackets should produce different predictions."""
        bracket1 = LBracket(
            leg1_length=60, leg2_length=60, width=25, thickness=4,
            hole1_distance=12, hole1_diameter=5, hole2_distance=12, hole2_diameter=5,
        )
        bracket2 = LBracket(
            leg1_length=180, leg2_length=150, width=55, thickness=10,
            hole1_distance=40, hole1_diameter=10, hole2_distance=35, hole2_diameter=9,
        )

        graph1 = extract_graph_from_solid(bracket1.to_solid())
        graph2 = extract_graph_from_solid(bracket2.to_solid())

        data1 = brep_graph_to_pyg(graph1)
        data2 = brep_graph_to_pyg(graph2)

        model.eval()
        with torch.no_grad():
            pred1 = model(data1["x"], data1["edge_index"], data1["edge_attr"])
            pred2 = model(data2["x"], data2["edge_index"], data2["edge_attr"])

        # Predictions should differ (model is untrained but inputs differ)
        assert not torch.allclose(pred1, pred2)
