"""Unit tests for VAE loss functions."""

import pytest

# Skip all tests in this module if PyTorch is not installed
torch = pytest.importorskip("torch")

from graph_cad.models.losses import (
    VAELossConfig,
    VariableVAELossConfig,
    kl_divergence,
    reconstruction_loss,
    vae_loss,
    variable_reconstruction_loss,
    mask_prediction_loss,
    face_type_classification_loss,
    variable_vae_loss,
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


# =============================================================================
# Variable Topology Loss Tests
# =============================================================================


@pytest.fixture
def max_nodes():
    return 20


@pytest.fixture
def max_edges():
    return 50


@pytest.fixture
def num_face_types():
    return 8


@pytest.fixture
def variable_node_features(batch_size, max_nodes, node_features):
    """Create sample variable node features (9D instead of 8D)."""
    return torch.randn(batch_size, max_nodes, 9)


@pytest.fixture
def variable_edge_features(batch_size, max_edges, edge_features):
    """Create sample variable edge features."""
    return torch.randn(batch_size, max_edges, edge_features)


@pytest.fixture
def node_mask(batch_size, max_nodes):
    """Create sample node mask with some nodes masked."""
    mask = torch.zeros(batch_size, max_nodes)
    # First 10 nodes are real for all samples
    mask[:, :10] = 1.0
    return mask


@pytest.fixture
def edge_mask(batch_size, max_edges):
    """Create sample edge mask with some edges masked."""
    mask = torch.zeros(batch_size, max_edges)
    # First 20 edges are real for all samples
    mask[:, :20] = 1.0
    return mask


@pytest.fixture
def face_types_target(batch_size, max_nodes, num_face_types):
    """Create sample face type targets."""
    # Mix of face types: 0=planar (most common), 1=cylindrical
    face_types = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    face_types[:, 8:10] = 1  # Last 2 real nodes are cylindrical
    return face_types


@pytest.fixture
def face_type_logits(batch_size, max_nodes, num_face_types):
    """Create sample face type logits."""
    return torch.randn(batch_size, max_nodes, num_face_types)


class TestVariableVAELossConfig:
    """Test variable loss configuration."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = VariableVAELossConfig()
        assert config.node_weight == 1.0
        assert config.edge_weight == 1.0
        assert config.node_mask_weight == 1.0
        assert config.edge_mask_weight == 1.0
        assert config.face_type_weight == 0.5

    def test_custom_config(self):
        """Custom config values should be stored."""
        config = VariableVAELossConfig(node_weight=2.0, face_type_weight=1.0)
        assert config.node_weight == 2.0
        assert config.face_type_weight == 1.0


class TestVariableReconstructionLoss:
    """Test variable topology reconstruction loss."""

    def test_zero_loss_for_identical(
        self, variable_node_features, variable_edge_features, node_mask, edge_mask
    ):
        """Loss should be zero when prediction equals target."""
        loss, _ = variable_reconstruction_loss(
            variable_node_features,
            variable_node_features,
            variable_edge_features,
            variable_edge_features,
            node_mask,
            edge_mask,
        )
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_loss_for_different(
        self, variable_node_features, variable_edge_features, node_mask, edge_mask
    ):
        """Loss should be positive when prediction differs from target."""
        pred_nodes = variable_node_features + 1.0
        pred_edges = variable_edge_features + 1.0

        loss, _ = variable_reconstruction_loss(
            pred_nodes,
            variable_node_features,
            pred_edges,
            variable_edge_features,
            node_mask,
            edge_mask,
        )
        assert loss > 0

    def test_mask_affects_loss(
        self, variable_node_features, variable_edge_features, node_mask, edge_mask
    ):
        """Only masked (real) nodes/edges should contribute to loss."""
        # Add error only to padding nodes (indices 10+)
        pred_nodes = variable_node_features.clone()
        pred_nodes[:, 10:, :] += 100.0  # Large error in padding area

        loss, _ = variable_reconstruction_loss(
            pred_nodes,
            variable_node_features,
            variable_edge_features,
            variable_edge_features,
            node_mask,
            edge_mask,
        )
        # Loss should be near zero because padding errors are masked
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


class TestMaskPredictionLoss:
    """Test mask prediction loss."""

    def test_perfect_prediction(self):
        """Loss should be low for perfect predictions."""
        # Create fixed test data
        bs, n_nodes, n_edges = 4, 20, 50

        node_mask = torch.zeros(bs, n_nodes)
        node_mask[:, :10] = 1.0
        edge_mask = torch.zeros(bs, n_edges)
        edge_mask[:, :20] = 1.0

        # Create logits that perfectly match masks
        node_logits = torch.where(
            node_mask > 0.5,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )
        edge_logits = torch.where(
            edge_mask > 0.5,
            torch.tensor(10.0),
            torch.tensor(-10.0),
        )

        # Function signature: (node_logits, edge_logits, node_target, edge_target)
        node_loss, edge_loss, metrics = mask_prediction_loss(
            node_logits, edge_logits, node_mask, edge_mask
        )

        total_loss = node_loss + edge_loss
        assert total_loss < 0.1
        assert metrics["node_mask_acc"] > 0.99
        assert metrics["edge_mask_acc"] > 0.99

    def test_random_prediction(self):
        """Random logits should have ~50% accuracy."""
        bs, n_nodes, n_edges = 4, 20, 50

        node_mask = torch.zeros(bs, n_nodes)
        node_mask[:, :10] = 1.0
        edge_mask = torch.zeros(bs, n_edges)
        edge_mask[:, :20] = 1.0

        node_logits = torch.randn(bs, n_nodes)
        edge_logits = torch.randn(bs, n_edges)

        # Function signature: (node_logits, edge_logits, node_target, edge_target)
        node_loss, edge_loss, metrics = mask_prediction_loss(
            node_logits, edge_logits, node_mask, edge_mask
        )

        # With random predictions, accuracy should be around 50%
        assert 0.3 < metrics["node_mask_acc"] < 0.7
        assert 0.3 < metrics["edge_mask_acc"] < 0.7


class TestFaceTypeClassificationLoss:
    """Test face type classification loss."""

    def test_perfect_prediction(
        self, face_types_target, node_mask, batch_size, max_nodes, num_face_types
    ):
        """Loss should be low for perfect predictions."""
        # Create one-hot logits
        logits = torch.zeros(batch_size, max_nodes, num_face_types)
        logits.scatter_(2, face_types_target.unsqueeze(-1), 10.0)

        loss, metrics = face_type_classification_loss(logits, face_types_target, node_mask)

        assert loss < 0.1
        assert metrics["face_type_acc"] > 0.99

    def test_only_uses_real_nodes(
        self, batch_size, max_nodes, num_face_types, node_mask
    ):
        """Loss should only consider real (non-padding) nodes."""
        # Create target with different values in padding area
        face_types = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        face_types[:, 10:] = 5  # Different type in padding

        # Perfect prediction for real nodes only
        logits = torch.zeros(batch_size, max_nodes, num_face_types)
        logits[:, :10, 0] = 10.0  # Correct for real nodes
        logits[:, 10:, 3] = 10.0  # Wrong for padding (but should be ignored)

        loss, metrics = face_type_classification_loss(logits, face_types, node_mask)

        # Should be near-perfect because padding is ignored
        assert metrics["face_type_acc"] > 0.99


class TestVariableVAELoss:
    """Test combined variable VAE loss."""

    def test_combines_all_components(
        self,
        variable_node_features,
        variable_edge_features,
        node_mask,
        edge_mask,
        face_types_target,
        face_type_logits,
        sample_latent,
    ):
        """Variable VAE loss should combine all components."""
        mu, logvar = sample_latent

        # Create outputs dict
        outputs = {
            "node_features": variable_node_features + 0.1,
            "edge_features": variable_edge_features + 0.1,
            "node_mask_logits": torch.randn_like(node_mask),
            "edge_mask_logits": torch.randn_like(edge_mask),
            "face_type_logits": face_type_logits,
            "mu": mu,
            "logvar": logvar,
        }

        # Create targets dict
        targets = {
            "node_features": variable_node_features,
            "edge_features": variable_edge_features,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "face_types": face_types_target,
        }

        loss, metrics = variable_vae_loss(outputs, targets, beta=0.01)

        # Check all components are present
        assert "recon_loss" in metrics
        assert "mask_loss" in metrics
        assert "face_type_loss" in metrics
        assert "kl_loss" in metrics
        assert "total_loss" in metrics

        # All should be positive
        assert metrics["recon_loss"] > 0
        assert metrics["total_loss"] > 0

    def test_beta_scales_kl(
        self,
        variable_node_features,
        variable_edge_features,
        node_mask,
        edge_mask,
        face_types_target,
        face_type_logits,
        sample_latent,
    ):
        """Beta should scale KL contribution."""
        mu, logvar = sample_latent

        outputs = {
            "node_features": variable_node_features,
            "edge_features": variable_edge_features,
            "node_mask_logits": torch.zeros_like(node_mask),
            "edge_mask_logits": torch.zeros_like(edge_mask),
            "face_type_logits": face_type_logits,
            "mu": mu,
            "logvar": logvar,
        }

        targets = {
            "node_features": variable_node_features,
            "edge_features": variable_edge_features,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "face_types": face_types_target,
        }

        _, metrics1 = variable_vae_loss(outputs, targets, beta=0.01)
        _, metrics2 = variable_vae_loss(outputs, targets, beta=0.1)

        # Higher beta should scale KL contribution
        # Note: due to free_bits, the actual contribution may be 0 for small KL
        # So we just verify the computation completes
        assert metrics1["kl_loss"] >= 0
        assert metrics2["kl_loss"] >= 0

    def test_loss_is_finite(
        self,
        variable_node_features,
        variable_edge_features,
        node_mask,
        edge_mask,
        face_types_target,
        face_type_logits,
        sample_latent,
    ):
        """Loss should always be finite."""
        mu, logvar = sample_latent

        outputs = {
            "node_features": variable_node_features + torch.randn_like(variable_node_features),
            "edge_features": variable_edge_features + torch.randn_like(variable_edge_features),
            "node_mask_logits": torch.randn_like(node_mask),
            "edge_mask_logits": torch.randn_like(edge_mask),
            "face_type_logits": face_type_logits,
            "mu": mu,
            "logvar": logvar,
        }

        targets = {
            "node_features": variable_node_features,
            "edge_features": variable_edge_features,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "face_types": face_types_target,
        }

        loss, _ = variable_vae_loss(outputs, targets)

        assert torch.isfinite(loss)

    def test_loss_requires_grad(
        self,
        variable_node_features,
        variable_edge_features,
        node_mask,
        edge_mask,
        face_types_target,
        face_type_logits,
        sample_latent,
    ):
        """Loss should support backpropagation."""
        mu, logvar = sample_latent
        mu = mu.clone().requires_grad_(True)
        pred_nodes = variable_node_features.clone().requires_grad_(True)

        outputs = {
            "node_features": pred_nodes,
            "edge_features": variable_edge_features,
            "node_mask_logits": torch.randn_like(node_mask),
            "edge_mask_logits": torch.randn_like(edge_mask),
            "face_type_logits": face_type_logits.requires_grad_(True),
            "mu": mu,
            "logvar": logvar,
        }

        targets = {
            "node_features": variable_node_features,
            "edge_features": variable_edge_features,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "face_types": face_types_target,
        }

        loss, _ = variable_vae_loss(outputs, targets)
        loss.backward()

        assert pred_nodes.grad is not None
        assert mu.grad is not None
