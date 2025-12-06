"""Unit tests for VAE loss functions."""

import pytest

# Skip all tests in this module if PyTorch is not installed
torch = pytest.importorskip("torch")

from graph_cad.models.losses import (
    VAELossConfig,
    kl_divergence,
    reconstruction_loss,
    vae_loss,
)


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def num_nodes():
    return 10


@pytest.fixture
def num_edges():
    return 22


@pytest.fixture
def node_features():
    return 8


@pytest.fixture
def edge_features():
    return 2


@pytest.fixture
def latent_dim():
    return 64


@pytest.fixture
def sample_node_features(batch_size, num_nodes, node_features):
    """Create sample node features."""
    return torch.randn(batch_size, num_nodes, node_features)


@pytest.fixture
def sample_edge_features(batch_size, num_edges, edge_features):
    """Create sample edge features."""
    return torch.randn(batch_size, num_edges, edge_features)


@pytest.fixture
def sample_latent(batch_size, latent_dim):
    """Create sample latent vectors."""
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)
    return mu, logvar


class TestVAELossConfig:
    """Test loss configuration."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = VAELossConfig()
        assert config.node_weight == 1.0
        assert config.edge_weight == 1.0
        assert config.face_type_weight == 2.0

    def test_custom_config(self):
        """Custom config values should be stored."""
        config = VAELossConfig(node_weight=2.0, edge_weight=0.5)
        assert config.node_weight == 2.0
        assert config.edge_weight == 0.5


class TestReconstructionLoss:
    """Test reconstruction loss computation."""

    def test_zero_loss_for_identical(self, sample_node_features, sample_edge_features):
        """Loss should be zero when prediction equals target."""
        loss, _ = reconstruction_loss(
            sample_node_features,
            sample_node_features,
            sample_edge_features,
            sample_edge_features,
        )
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_loss_for_different(
        self, sample_node_features, sample_edge_features
    ):
        """Loss should be positive when prediction differs from target."""
        pred_nodes = sample_node_features + 1.0
        pred_edges = sample_edge_features + 1.0

        loss, _ = reconstruction_loss(
            pred_nodes,
            sample_node_features,
            pred_edges,
            sample_edge_features,
        )
        assert loss > 0

    def test_loss_increases_with_error(
        self, sample_node_features, sample_edge_features
    ):
        """Loss should increase as error increases."""
        losses = []
        for scale in [0.1, 0.5, 1.0, 2.0]:
            pred_nodes = sample_node_features + scale
            pred_edges = sample_edge_features + scale
            loss, _ = reconstruction_loss(
                pred_nodes,
                sample_node_features,
                pred_edges,
                sample_edge_features,
            )
            losses.append(loss.item())

        # Losses should be monotonically increasing
        for i in range(len(losses) - 1):
            assert losses[i] < losses[i + 1]

    def test_returns_loss_dict(self, sample_node_features, sample_edge_features):
        """Should return dictionary with component losses."""
        pred_nodes = sample_node_features + 0.1
        pred_edges = sample_edge_features + 0.1

        _, loss_dict = reconstruction_loss(
            pred_nodes,
            sample_node_features,
            pred_edges,
            sample_edge_features,
        )

        assert "node_loss" in loss_dict
        assert "edge_loss" in loss_dict
        assert "face_type_loss" in loss_dict
        assert "area_loss" in loss_dict
        assert "direction_loss" in loss_dict
        assert "centroid_loss" in loss_dict

    def test_node_weight_affects_loss(
        self, sample_node_features, sample_edge_features
    ):
        """Node weight should scale node contribution to total loss."""
        pred_nodes = sample_node_features + 1.0

        config1 = VAELossConfig(node_weight=1.0, edge_weight=0.0)
        config2 = VAELossConfig(node_weight=2.0, edge_weight=0.0)

        loss1, _ = reconstruction_loss(
            pred_nodes, sample_node_features,
            sample_edge_features, sample_edge_features,
            config1,
        )
        loss2, _ = reconstruction_loss(
            pred_nodes, sample_node_features,
            sample_edge_features, sample_edge_features,
            config2,
        )

        # Loss should scale with weight
        assert torch.isclose(loss2, loss1 * 2, rtol=0.01)

    def test_edge_weight_affects_loss(
        self, sample_node_features, sample_edge_features
    ):
        """Edge weight should scale edge contribution to total loss."""
        pred_edges = sample_edge_features + 1.0

        config1 = VAELossConfig(node_weight=0.0, edge_weight=1.0)
        config2 = VAELossConfig(node_weight=0.0, edge_weight=2.0)

        loss1, _ = reconstruction_loss(
            sample_node_features, sample_node_features,
            pred_edges, sample_edge_features,
            config1,
        )
        loss2, _ = reconstruction_loss(
            sample_node_features, sample_node_features,
            pred_edges, sample_edge_features,
            config2,
        )

        assert torch.isclose(loss2, loss1 * 2, rtol=0.01)


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_zero_kl_for_standard_normal(self, batch_size, latent_dim):
        """KL(N(0,I) || N(0,I)) should be 0."""
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)

        kl = kl_divergence(mu, logvar)
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_positive_kl_for_non_standard(self, batch_size, latent_dim):
        """KL should be positive for non-standard distributions."""
        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)

        kl = kl_divergence(mu, logvar)
        assert kl > 0

    def test_kl_increases_with_mu_deviation(self, batch_size, latent_dim):
        """KL should increase as mu deviates from 0."""
        logvar = torch.zeros(batch_size, latent_dim)
        kls = []

        for mu_val in [0.0, 1.0, 2.0, 3.0]:
            mu = torch.full((batch_size, latent_dim), mu_val)
            kl = kl_divergence(mu, logvar)
            kls.append(kl.item())

        # KLs should be monotonically increasing
        for i in range(len(kls) - 1):
            assert kls[i] < kls[i + 1]

    def test_kl_increases_with_logvar_deviation(self, batch_size, latent_dim):
        """KL should increase as variance deviates from 1."""
        mu = torch.zeros(batch_size, latent_dim)

        # logvar = 0 means var = 1 (standard normal), which gives KL = 0
        # logvar != 0 increases KL
        kl_at_zero = kl_divergence(mu, torch.zeros(batch_size, latent_dim))
        kl_positive = kl_divergence(mu, torch.ones(batch_size, latent_dim))
        kl_negative = kl_divergence(mu, -torch.ones(batch_size, latent_dim))

        assert kl_positive > kl_at_zero
        assert kl_negative > kl_at_zero

    def test_free_bits_reduces_kl(self, batch_size, latent_dim):
        """Free bits should reduce effective KL."""
        mu = torch.randn(batch_size, latent_dim) * 0.1  # Small deviation
        logvar = torch.randn(batch_size, latent_dim) * 0.1

        kl_no_free = kl_divergence(mu, logvar, free_bits=0.0)
        kl_with_free = kl_divergence(mu, logvar, free_bits=0.5)

        assert kl_with_free <= kl_no_free

    def test_kl_is_scalar(self, sample_latent):
        """KL should return a scalar."""
        mu, logvar = sample_latent
        kl = kl_divergence(mu, logvar)
        assert kl.dim() == 0


class TestVAELoss:
    """Test combined VAE loss."""

    def test_combines_recon_and_kl(
        self,
        sample_node_features,
        sample_edge_features,
        sample_latent,
    ):
        """VAE loss should combine reconstruction and KL."""
        mu, logvar = sample_latent

        pred_nodes = sample_node_features + 0.5
        pred_edges = sample_edge_features + 0.5

        loss, loss_dict = vae_loss(
            pred_nodes,
            sample_node_features,
            pred_edges,
            sample_edge_features,
            mu,
            logvar,
            beta=1.0,
        )

        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert "total_loss" in loss_dict

        # Total should be approximately recon + KL
        expected = loss_dict["recon_loss"] + loss_dict["kl_loss"]
        assert torch.isclose(loss_dict["total_loss"], expected, rtol=0.01)

    def test_beta_scales_kl(
        self,
        sample_node_features,
        sample_edge_features,
        sample_latent,
    ):
        """Beta should scale KL contribution."""
        mu, logvar = sample_latent

        loss_beta1, dict1 = vae_loss(
            sample_node_features,
            sample_node_features,
            sample_edge_features,
            sample_edge_features,
            mu,
            logvar,
            beta=1.0,
        )

        loss_beta2, dict2 = vae_loss(
            sample_node_features,
            sample_node_features,
            sample_edge_features,
            sample_edge_features,
            mu,
            logvar,
            beta=2.0,
        )

        # KL contribution should double
        kl_contribution_1 = dict1["total_loss"] - dict1["recon_loss"]
        kl_contribution_2 = dict2["total_loss"] - dict2["recon_loss"]

        assert torch.isclose(kl_contribution_2, kl_contribution_1 * 2, rtol=0.01)

    def test_beta_zero_equals_recon_only(
        self,
        sample_node_features,
        sample_edge_features,
        sample_latent,
    ):
        """With beta=0, total loss should equal reconstruction loss."""
        mu, logvar = sample_latent

        pred_nodes = sample_node_features + 0.5

        loss, loss_dict = vae_loss(
            pred_nodes,
            sample_node_features,
            sample_edge_features,
            sample_edge_features,
            mu,
            logvar,
            beta=0.0,
        )

        assert torch.isclose(loss_dict["total_loss"], loss_dict["recon_loss"], rtol=0.01)

    def test_loss_is_finite(
        self,
        sample_node_features,
        sample_edge_features,
        sample_latent,
    ):
        """Loss should always be finite."""
        mu, logvar = sample_latent

        loss, _ = vae_loss(
            sample_node_features + torch.randn_like(sample_node_features),
            sample_node_features,
            sample_edge_features + torch.randn_like(sample_edge_features),
            sample_edge_features,
            mu,
            logvar,
        )

        assert torch.isfinite(loss)

    def test_loss_requires_grad(
        self,
        sample_node_features,
        sample_edge_features,
        sample_latent,
    ):
        """Loss should support backpropagation."""
        mu, logvar = sample_latent
        mu.requires_grad = True

        pred_nodes = sample_node_features.clone().requires_grad_(True)

        loss, _ = vae_loss(
            pred_nodes,
            sample_node_features,
            sample_edge_features,
            sample_edge_features,
            mu,
            logvar,
        )

        loss.backward()

        assert pred_nodes.grad is not None
        assert mu.grad is not None
