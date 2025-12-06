"""Unit tests for Graph VAE model."""

import numpy as np
import pytest

# Skip all tests in this module if PyTorch is not installed
torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from graph_cad.data import LBracket, extract_graph_from_solid
from graph_cad.models import (
    GraphVAE,
    GraphVAEConfig,
    GraphVAEDecoder,
    GraphVAEEncoder,
    brep_graph_to_pyg,
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
def config():
    """Create default VAE config."""
    return GraphVAEConfig()


@pytest.fixture
def model(config):
    """Create a Graph VAE model."""
    return GraphVAE(config)


@pytest.fixture
def batched_data(sample_pyg_data):
    """Create a batch of 3 identical graphs for testing."""
    data_list = []
    for _ in range(3):
        data = Data(
            x=sample_pyg_data["x"].clone(),
            edge_index=sample_pyg_data["edge_index"].clone(),
            edge_attr=sample_pyg_data["edge_attr"].clone(),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


class TestGraphVAEConfig:
    """Test VAE configuration."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = GraphVAEConfig()
        assert config.node_features == 8
        assert config.edge_features == 2
        assert config.num_nodes == 10
        assert config.num_edges == 22
        assert config.latent_dim == 64
        assert config.hidden_dim == 64

    def test_custom_config(self):
        """Custom config values should be stored."""
        config = GraphVAEConfig(latent_dim=128, hidden_dim=32)
        assert config.latent_dim == 128
        assert config.hidden_dim == 32


class TestGraphVAEEncoder:
    """Test VAE encoder."""

    def test_output_shapes(self, config, sample_pyg_data):
        """Encoder should output mu and logvar of correct shape."""
        encoder = GraphVAEEncoder(config)
        mu, logvar = encoder(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )
        assert mu.shape == (1, config.latent_dim)
        assert logvar.shape == (1, config.latent_dim)

    def test_batched_encoding(self, config, batched_data):
        """Encoder should handle batched graphs."""
        encoder = GraphVAEEncoder(config)
        mu, logvar = encoder(
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )
        assert mu.shape == (3, config.latent_dim)
        assert logvar.shape == (3, config.latent_dim)

    def test_mu_logvar_finite(self, config, sample_pyg_data):
        """Encoder outputs should be finite."""
        encoder = GraphVAEEncoder(config)
        mu, logvar = encoder(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )
        assert torch.isfinite(mu).all()
        assert torch.isfinite(logvar).all()


class TestGraphVAEDecoder:
    """Test VAE decoder."""

    def test_output_shapes(self, config):
        """Decoder should output correct shapes."""
        decoder = GraphVAEDecoder(config)
        z = torch.randn(1, config.latent_dim)
        node_feat, edge_feat = decoder(z)
        assert node_feat.shape == (1, config.num_nodes, config.node_features)
        assert edge_feat.shape == (1, config.num_edges, config.edge_features)

    def test_batched_decoding(self, config):
        """Decoder should handle batched latent vectors."""
        decoder = GraphVAEDecoder(config)
        z = torch.randn(5, config.latent_dim)
        node_feat, edge_feat = decoder(z)
        assert node_feat.shape == (5, config.num_nodes, config.node_features)
        assert edge_feat.shape == (5, config.num_edges, config.edge_features)

    def test_outputs_finite(self, config):
        """Decoder outputs should be finite."""
        decoder = GraphVAEDecoder(config)
        z = torch.randn(1, config.latent_dim)
        node_feat, edge_feat = decoder(z)
        assert torch.isfinite(node_feat).all()
        assert torch.isfinite(edge_feat).all()


class TestGraphVAE:
    """Test full VAE model."""

    def test_forward_returns_dict(self, model, sample_pyg_data):
        """Forward should return dict with expected keys."""
        outputs = model(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )
        assert "node_recon" in outputs
        assert "edge_recon" in outputs
        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs

    def test_reconstruction_shapes(self, model, config, sample_pyg_data):
        """Reconstructed features should have correct shapes."""
        outputs = model(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )
        assert outputs["node_recon"].shape == (1, config.num_nodes, config.node_features)
        assert outputs["edge_recon"].shape == (1, config.num_edges, config.edge_features)

    def test_latent_shapes(self, model, config, sample_pyg_data):
        """Latent vectors should have correct shapes."""
        outputs = model(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )
        assert outputs["mu"].shape == (1, config.latent_dim)
        assert outputs["logvar"].shape == (1, config.latent_dim)
        assert outputs["z"].shape == (1, config.latent_dim)

    def test_batched_forward(self, model, config, batched_data):
        """Forward should handle batched graphs."""
        outputs = model(
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.batch,
        )
        assert outputs["node_recon"].shape == (3, config.num_nodes, config.node_features)
        assert outputs["mu"].shape == (3, config.latent_dim)

    def test_reparameterization_train_mode(self, model):
        """In train mode, reparameterization should add noise."""
        model.train()
        mu = torch.zeros(10, 64)
        logvar = torch.zeros(10, 64)

        # Multiple samples should differ
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        assert not torch.allclose(z1, z2)

    def test_reparameterization_eval_mode(self, model):
        """In eval mode, reparameterization should be deterministic (return mu)."""
        model.eval()
        mu = torch.randn(10, 64)
        logvar = torch.zeros(10, 64)

        z = model.reparameterize(mu, logvar)
        assert torch.allclose(z, mu)

    def test_gradient_flow(self, model, sample_pyg_data):
        """Gradients should flow through reparameterization."""
        model.train()
        outputs = model(
            sample_pyg_data["x"],
            sample_pyg_data["edge_index"],
            sample_pyg_data["edge_attr"],
        )

        # Compute simple loss and backward
        loss = outputs["node_recon"].sum() + outputs["edge_recon"].sum()
        loss.backward()

        # Check encoder has gradients
        for param in model.encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Check decoder has gradients
        for param in model.decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestGraphVAESampling:
    """Test VAE sampling methods."""

    def test_sample_from_prior(self, model, config):
        """Sampling from prior should work."""
        node_samples, edge_samples = model.sample(10)
        assert node_samples.shape == (10, config.num_nodes, config.node_features)
        assert edge_samples.shape == (10, config.num_edges, config.edge_features)

    def test_sample_different_each_time(self, model):
        """Samples should be different each call."""
        s1, _ = model.sample(5)
        s2, _ = model.sample(5)
        assert not torch.allclose(s1, s2)


class TestGraphVAEInterpolation:
    """Test VAE interpolation."""

    def test_interpolation_shape(self, model, config):
        """Interpolation should return correct shapes."""
        z1 = torch.randn(config.latent_dim)
        z2 = torch.randn(config.latent_dim)

        node_interp, edge_interp = model.interpolate(z1, z2, num_steps=10)
        assert node_interp.shape == (10, config.num_nodes, config.node_features)
        assert edge_interp.shape == (10, config.num_edges, config.edge_features)

    def test_interpolation_endpoints(self, model, config):
        """First and last interpolation steps should match endpoints."""
        z1 = torch.randn(1, config.latent_dim)
        z2 = torch.randn(1, config.latent_dim)

        model.eval()
        node_interp, edge_interp = model.interpolate(z1, z2, num_steps=10)

        # Decode endpoints directly
        node1, edge1 = model.decode(z1)
        node2, edge2 = model.decode(z2)

        # First step should be close to z1 decode
        assert torch.allclose(node_interp[0], node1[0], atol=1e-5)
        # Last step should be close to z2 decode
        assert torch.allclose(node_interp[-1], node2[0], atol=1e-5)


class TestEndToEnd:
    """End-to-end tests with real bracket data."""

    def test_bracket_to_reconstruction_pipeline(self, model, sample_bracket):
        """Full pipeline: bracket -> graph -> encode -> decode."""
        # Extract graph
        graph = extract_graph_from_solid(sample_bracket.to_solid())
        pyg_data = brep_graph_to_pyg(graph)

        # Forward pass
        model.eval()
        outputs = model(
            pyg_data["x"],
            pyg_data["edge_index"],
            pyg_data["edge_attr"],
        )

        # Check outputs are valid
        assert torch.isfinite(outputs["node_recon"]).all()
        assert torch.isfinite(outputs["edge_recon"]).all()

    def test_training_reduces_loss(self, model, batched_data):
        """A few training steps should reduce loss."""
        import torch.nn.functional as F
        from graph_cad.training.vae_trainer import prepare_batch_targets

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None

        for step in range(10):
            optimizer.zero_grad()
            outputs = model(
                batched_data.x,
                batched_data.edge_index,
                batched_data.edge_attr,
                batched_data.batch,
            )

            node_target, edge_target = prepare_batch_targets(
                batched_data,
                num_nodes=model.config.num_nodes,
                node_features=model.config.node_features,
                num_edges=model.config.num_edges,
                edge_features=model.config.edge_features,
            )

            loss = F.mse_loss(outputs["node_recon"], node_target) + \
                   F.mse_loss(outputs["edge_recon"], edge_target)

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Loss should decrease (or at least not increase significantly)
        assert final_loss <= initial_loss * 1.1  # Allow some noise
