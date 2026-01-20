"""
Tests for the Transformer Graph Decoder and Hungarian matching loss.

These tests verify:
1. TransformerGraphDecoder forward pass shapes
2. Hungarian matching produces valid assignments
3. Loss computation with matching works correctly
4. Full TransformerGraphVAE forward/backward
"""

import pytest
import torch
import numpy as np

from graph_cad.models.transformer_decoder import (
    TransformerDecoderConfig,
    TransformerGraphDecoder,
    TransformerGraphVAE,
)
from graph_cad.models.graph_vae import (
    VariableGraphVAEConfig,
    VariableGraphVAEEncoder,
)
from graph_cad.models.losses import (
    HungarianLossConfig,
    compute_hungarian_matching,
    hungarian_node_loss,
    hungarian_edge_loss_with_adj,
    transformer_vae_loss,
)


@pytest.fixture
def decoder_config():
    """Small config for testing."""
    return TransformerDecoderConfig(
        latent_dim=16,
        node_features=13,
        edge_features=2,
        num_face_types=3,
        max_nodes=10,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def encoder_config():
    """Matching encoder config for testing."""
    return VariableGraphVAEConfig(
        node_features=13,
        edge_features=2,
        num_face_types=3,
        face_embed_dim=8,
        max_nodes=10,
        max_edges=20,
        hidden_dim=32,
        num_gat_layers=2,
        num_heads=4,
        latent_dim=16,
        encoder_dropout=0.0,
    )


class TestTransformerGraphDecoder:
    """Tests for the TransformerGraphDecoder class."""

    def test_forward_shapes(self, decoder_config):
        """Test that forward pass produces correct output shapes."""
        decoder = TransformerGraphDecoder(decoder_config)
        batch_size = 4
        z = torch.randn(batch_size, decoder_config.latent_dim)

        outputs = decoder(z)

        assert outputs["node_features"].shape == (
            batch_size, decoder_config.max_nodes, decoder_config.node_features
        )
        assert outputs["face_type_logits"].shape == (
            batch_size, decoder_config.max_nodes, decoder_config.num_face_types
        )
        assert outputs["existence_logits"].shape == (
            batch_size, decoder_config.max_nodes
        )
        assert outputs["edge_logits"].shape == (
            batch_size, decoder_config.max_nodes, decoder_config.max_nodes
        )
        assert outputs["node_embeddings"].shape == (
            batch_size, decoder_config.max_nodes, decoder_config.hidden_dim
        )

    def test_edge_logits_symmetric(self, decoder_config):
        """Test that edge logits are symmetric (undirected graph)."""
        decoder = TransformerGraphDecoder(decoder_config)
        z = torch.randn(2, decoder_config.latent_dim)

        outputs = decoder(z)
        edge_logits = outputs["edge_logits"]

        # Check symmetry
        assert torch.allclose(edge_logits, edge_logits.transpose(1, 2), atol=1e-6)

    def test_different_batch_sizes(self, decoder_config):
        """Test decoder works with various batch sizes."""
        decoder = TransformerGraphDecoder(decoder_config)

        for batch_size in [1, 2, 8, 16]:
            z = torch.randn(batch_size, decoder_config.latent_dim)
            outputs = decoder(z)
            assert outputs["node_features"].shape[0] == batch_size

    def test_gradient_flow(self, decoder_config):
        """Test that gradients flow through the decoder."""
        decoder = TransformerGraphDecoder(decoder_config)
        z = torch.randn(2, decoder_config.latent_dim, requires_grad=True)

        outputs = decoder(z)
        loss = outputs["node_features"].sum() + outputs["edge_logits"].sum()
        loss.backward()

        assert z.grad is not None
        assert not torch.all(z.grad == 0)


class TestHungarianMatching:
    """Tests for Hungarian matching functions."""

    def test_matching_perfect_alignment(self):
        """Test matching when predictions perfectly align with targets."""
        batch_size = 2
        max_nodes = 5
        num_features = 13
        num_types = 3

        # Create targets
        target_features = torch.randn(batch_size, max_nodes, num_features)
        target_face_types = torch.randint(0, num_types, (batch_size, max_nodes))
        target_mask = torch.zeros(batch_size, max_nodes)
        target_mask[0, :3] = 1.0  # 3 real nodes in sample 0
        target_mask[1, :4] = 1.0  # 4 real nodes in sample 1

        # Use targets as predictions (perfect match)
        pred_features = target_features.clone()
        pred_face_types = torch.zeros(batch_size, max_nodes, num_types)
        for b in range(batch_size):
            for i in range(max_nodes):
                pred_face_types[b, i, target_face_types[b, i]] = 10.0  # High logit
        pred_existence = torch.ones(batch_size, max_nodes) * 10  # All high

        matchings = compute_hungarian_matching(
            pred_features, pred_face_types, pred_existence,
            target_features, target_face_types, target_mask
        )

        # Check matching lengths
        assert len(matchings) == batch_size
        assert len(matchings[0][0]) == 3  # 3 nodes matched
        assert len(matchings[1][0]) == 4  # 4 nodes matched

    def test_matching_respects_mask(self):
        """Test that matching only considers masked (real) nodes."""
        batch_size = 1
        max_nodes = 10
        num_features = 13
        num_types = 3

        target_features = torch.randn(batch_size, max_nodes, num_features)
        target_face_types = torch.randint(0, num_types, (batch_size, max_nodes))
        target_mask = torch.zeros(batch_size, max_nodes)
        target_mask[0, :5] = 1.0  # Only 5 real nodes

        pred_features = torch.randn(batch_size, max_nodes, num_features)
        pred_face_types = torch.randn(batch_size, max_nodes, num_types)
        pred_existence = torch.randn(batch_size, max_nodes)

        matchings = compute_hungarian_matching(
            pred_features, pred_face_types, pred_existence,
            target_features, target_face_types, target_mask
        )

        # Should only match 5 nodes (num_real)
        assert len(matchings[0][0]) == 5
        assert len(matchings[0][1]) == 5

        # Target indices should all be in range [0, 5)
        assert all(idx < 5 for idx in matchings[0][1].tolist())

    def test_matching_empty_graph(self):
        """Test matching handles empty graphs (no real nodes)."""
        batch_size = 1
        max_nodes = 5

        target_features = torch.randn(batch_size, max_nodes, 13)
        target_face_types = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        target_mask = torch.zeros(batch_size, max_nodes)  # No real nodes

        pred_features = torch.randn(batch_size, max_nodes, 13)
        pred_face_types = torch.randn(batch_size, max_nodes, 3)
        pred_existence = torch.randn(batch_size, max_nodes)

        matchings = compute_hungarian_matching(
            pred_features, pred_face_types, pred_existence,
            target_features, target_face_types, target_mask
        )

        assert len(matchings[0][0]) == 0
        assert len(matchings[0][1]) == 0


class TestHungarianNodeLoss:
    """Tests for hungarian_node_loss function."""

    def test_loss_decreases_with_better_match(self):
        """Test that loss is lower when predictions match targets better."""
        batch_size = 2
        max_nodes = 5
        num_features = 13
        num_types = 3

        target_features = torch.randn(batch_size, max_nodes, num_features)
        target_face_types = torch.randint(0, num_types, (batch_size, max_nodes))
        target_mask = torch.ones(batch_size, max_nodes)

        # Good predictions (close to targets)
        good_pred_features = target_features + torch.randn_like(target_features) * 0.1
        good_pred_types = torch.zeros(batch_size, max_nodes, num_types)
        for b in range(batch_size):
            for i in range(max_nodes):
                good_pred_types[b, i, target_face_types[b, i]] = 5.0
        good_pred_existence = torch.ones(batch_size, max_nodes) * 5

        # Bad predictions (random)
        bad_pred_features = torch.randn(batch_size, max_nodes, num_features)
        bad_pred_types = torch.randn(batch_size, max_nodes, num_types)
        bad_pred_existence = torch.randn(batch_size, max_nodes)

        # Compute matchings and losses
        good_matchings = compute_hungarian_matching(
            good_pred_features, good_pred_types, good_pred_existence,
            target_features, target_face_types, target_mask
        )
        bad_matchings = compute_hungarian_matching(
            bad_pred_features, bad_pred_types, bad_pred_existence,
            target_features, target_face_types, target_mask
        )

        good_loss, _ = hungarian_node_loss(
            good_pred_features, good_pred_types, good_pred_existence,
            target_features, target_face_types, target_mask, good_matchings
        )
        bad_loss, _ = hungarian_node_loss(
            bad_pred_features, bad_pred_types, bad_pred_existence,
            target_features, target_face_types, target_mask, bad_matchings
        )

        assert good_loss < bad_loss

    def test_loss_metrics_keys(self):
        """Test that loss returns expected metrics."""
        pred_features = torch.randn(2, 5, 13)
        pred_face_types = torch.randn(2, 5, 3)
        pred_existence = torch.randn(2, 5)
        target_features = torch.randn(2, 5, 13)
        target_face_types = torch.randint(0, 3, (2, 5))
        target_mask = torch.ones(2, 5)

        matchings = compute_hungarian_matching(
            pred_features, pred_face_types, pred_existence,
            target_features, target_face_types, target_mask
        )

        _, metrics = hungarian_node_loss(
            pred_features, pred_face_types, pred_existence,
            target_features, target_face_types, target_mask, matchings
        )

        expected_keys = [
            "node_feature_loss", "face_type_loss", "existence_loss",
            "face_type_acc", "existence_acc", "num_matched"
        ]
        for key in expected_keys:
            assert key in metrics


class TestHungarianEdgeLoss:
    """Tests for hungarian_edge_loss_with_adj function."""

    def test_edge_loss_with_matching(self):
        """Test edge loss computation with valid matching."""
        batch_size = 2
        max_nodes = 5

        pred_edge_logits = torch.randn(batch_size, max_nodes, max_nodes)
        target_adj = torch.zeros(batch_size, max_nodes, max_nodes)
        # Add some edges
        target_adj[0, 0, 1] = 1.0
        target_adj[0, 1, 0] = 1.0
        target_adj[0, 1, 2] = 1.0
        target_adj[0, 2, 1] = 1.0

        target_mask = torch.ones(batch_size, max_nodes)

        # Create identity matching for simplicity
        matchings = [
            (torch.arange(3), torch.arange(3)),  # Match first 3 nodes
            (torch.arange(4), torch.arange(4)),  # Match first 4 nodes
        ]

        edge_loss, metrics = hungarian_edge_loss_with_adj(
            pred_edge_logits, target_adj, target_mask, matchings
        )

        assert edge_loss >= 0
        assert "edge_loss" in metrics
        assert "edge_acc" in metrics
        assert "edge_precision" in metrics
        assert "edge_recall" in metrics

    def test_edge_loss_perfect_prediction(self):
        """Test edge loss is low when predictions match targets."""
        batch_size = 1
        max_nodes = 4

        # Create target adjacency
        target_adj = torch.zeros(batch_size, max_nodes, max_nodes)
        target_adj[0, 0, 1] = 1.0
        target_adj[0, 1, 0] = 1.0
        target_adj[0, 1, 2] = 1.0
        target_adj[0, 2, 1] = 1.0

        # Perfect prediction (high logits where adj=1, low where adj=0)
        pred_edge_logits = target_adj * 10 - 5  # 5 for edges, -5 for non-edges

        target_mask = torch.ones(batch_size, max_nodes)
        matchings = [(torch.arange(4), torch.arange(4))]

        edge_loss, metrics = hungarian_edge_loss_with_adj(
            pred_edge_logits, target_adj, target_mask, matchings
        )

        assert metrics["edge_acc"] > 0.9


class TestTransformerVAELoss:
    """Tests for the combined transformer_vae_loss function."""

    def test_full_loss_computation(self):
        """Test end-to-end loss computation."""
        batch_size = 2
        max_nodes = 5
        num_features = 13
        num_types = 3
        latent_dim = 16

        outputs = {
            "node_features": torch.randn(batch_size, max_nodes, num_features),
            "face_type_logits": torch.randn(batch_size, max_nodes, num_types),
            "existence_logits": torch.randn(batch_size, max_nodes),
            "edge_logits": torch.randn(batch_size, max_nodes, max_nodes),
            "mu": torch.randn(batch_size, latent_dim),
            "logvar": torch.randn(batch_size, latent_dim),
        }

        targets = {
            "node_features": torch.randn(batch_size, max_nodes, num_features),
            "face_types": torch.randint(0, num_types, (batch_size, max_nodes)),
            "node_mask": torch.ones(batch_size, max_nodes),
            "adj_matrix": torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        }

        loss, metrics = transformer_vae_loss(outputs, targets, beta=0.1)

        # Note: requires_grad is tested separately in test_loss_differentiable
        assert loss >= 0  # Loss should be non-negative
        assert "total_loss" in metrics
        assert "kl_loss" in metrics
        assert "node_loss" in metrics
        assert "edge_loss" in metrics

    def test_loss_differentiable(self):
        """Test that loss allows gradient computation."""
        batch_size = 2
        max_nodes = 5
        latent_dim = 16

        outputs = {
            "node_features": torch.randn(batch_size, max_nodes, 13, requires_grad=True),
            "face_type_logits": torch.randn(batch_size, max_nodes, 3, requires_grad=True),
            "existence_logits": torch.randn(batch_size, max_nodes, requires_grad=True),
            "edge_logits": torch.randn(batch_size, max_nodes, max_nodes, requires_grad=True),
            "mu": torch.randn(batch_size, latent_dim, requires_grad=True),
            "logvar": torch.randn(batch_size, latent_dim, requires_grad=True),
        }

        targets = {
            "node_features": torch.randn(batch_size, max_nodes, 13),
            "face_types": torch.randint(0, 3, (batch_size, max_nodes)),
            "node_mask": torch.ones(batch_size, max_nodes),
            "adj_matrix": torch.randint(0, 2, (batch_size, max_nodes, max_nodes)).float(),
        }

        loss, _ = transformer_vae_loss(outputs, targets)
        loss.backward()

        assert outputs["node_features"].grad is not None
        assert outputs["mu"].grad is not None


class TestTransformerGraphVAE:
    """Tests for the full TransformerGraphVAE model."""

    def test_forward_pass(self, encoder_config, decoder_config):
        """Test full forward pass through VAE."""
        encoder = VariableGraphVAEEncoder(encoder_config)
        model = TransformerGraphVAE(encoder, decoder_config)

        batch_size = 2
        num_nodes = 8
        num_edges = 15

        # Create input tensors
        x = torch.randn(batch_size * encoder_config.max_nodes, encoder_config.node_features)
        face_types = torch.randint(0, encoder_config.num_face_types, (batch_size * encoder_config.max_nodes,))

        # Create edge index (batched)
        edge_index = torch.randint(0, encoder_config.max_nodes, (2, batch_size * encoder_config.max_edges))
        # Adjust for batch offsets
        for b in range(batch_size):
            start = b * encoder_config.max_edges
            end = (b + 1) * encoder_config.max_edges
            edge_index[:, start:end] += b * encoder_config.max_nodes

        edge_attr = torch.randn(batch_size * encoder_config.max_edges, encoder_config.edge_features)
        batch = torch.repeat_interleave(
            torch.arange(batch_size), encoder_config.max_nodes
        )
        node_mask = torch.ones(batch_size * encoder_config.max_nodes)

        outputs = model(x, face_types, edge_index, edge_attr, batch, node_mask)

        assert "node_features" in outputs
        assert "face_type_logits" in outputs
        assert "existence_logits" in outputs
        assert "edge_logits" in outputs
        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs

        assert outputs["node_features"].shape == (
            batch_size, decoder_config.max_nodes, decoder_config.node_features
        )
        assert outputs["mu"].shape == (batch_size, decoder_config.latent_dim)

    def test_sample(self, encoder_config, decoder_config):
        """Test sampling from prior."""
        encoder = VariableGraphVAEEncoder(encoder_config)
        model = TransformerGraphVAE(encoder, decoder_config)

        samples = model.sample(num_samples=4)

        assert "node_features" in samples
        assert samples["node_features"].shape == (4, decoder_config.max_nodes, decoder_config.node_features)

    def test_reparameterization(self, encoder_config, decoder_config):
        """Test reparameterization trick."""
        encoder = VariableGraphVAEEncoder(encoder_config)
        model = TransformerGraphVAE(encoder, decoder_config)

        mu = torch.randn(2, decoder_config.latent_dim)
        logvar = torch.randn(2, decoder_config.latent_dim)

        # In training mode, should sample
        model.train()
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        # Different samples (with very high probability)
        assert not torch.allclose(z1, z2)

        # In eval mode, should return mu
        model.eval()
        z_eval = model.reparameterize(mu, logvar)
        assert torch.allclose(z_eval, mu)


class TestMultiHeadAttentionPooling:
    """Tests for the MultiHeadAttentionPooling class."""

    def test_forward_shape_single_graph(self):
        """Test output shape for a single graph."""
        from graph_cad.models.graph_vae import MultiHeadAttentionPooling

        hidden_dim = 64
        output_dim = 32
        num_heads = 4
        num_nodes = 10

        pooling = MultiHeadAttentionPooling(hidden_dim, output_dim, num_heads)
        h = torch.randn(num_nodes, hidden_dim)

        output = pooling(h, batch=None, node_mask=None)

        assert output.shape == (1, output_dim)

    def test_forward_shape_batched(self):
        """Test output shape for batched graphs."""
        from graph_cad.models.graph_vae import MultiHeadAttentionPooling

        hidden_dim = 64
        output_dim = 32
        num_heads = 4
        batch_size = 4
        nodes_per_graph = 8
        total_nodes = batch_size * nodes_per_graph

        pooling = MultiHeadAttentionPooling(hidden_dim, output_dim, num_heads)
        h = torch.randn(total_nodes, hidden_dim)
        batch = torch.repeat_interleave(torch.arange(batch_size), nodes_per_graph)

        output = pooling(h, batch=batch, node_mask=None)

        assert output.shape == (batch_size, output_dim)

    def test_forward_with_mask(self):
        """Test that mask is respected."""
        from graph_cad.models.graph_vae import MultiHeadAttentionPooling

        hidden_dim = 64
        output_dim = 32
        num_heads = 4
        num_nodes = 10

        pooling = MultiHeadAttentionPooling(hidden_dim, output_dim, num_heads)

        # Create input where masked nodes have very different values
        h = torch.randn(num_nodes, hidden_dim)
        h[5:] = 1000.0  # Masked nodes with extreme values

        node_mask = torch.zeros(num_nodes)
        node_mask[:5] = 1.0  # Only first 5 nodes are real

        output = pooling(h, batch=None, node_mask=node_mask)

        # Output should not be affected by the extreme masked values
        assert output.shape == (1, output_dim)
        assert not torch.any(output > 500)  # Should not include the 1000.0 values

    def test_gradient_flow(self):
        """Test that gradients flow through attention pooling."""
        from graph_cad.models.graph_vae import MultiHeadAttentionPooling

        hidden_dim = 64
        output_dim = 32
        num_heads = 4
        num_nodes = 10

        pooling = MultiHeadAttentionPooling(hidden_dim, output_dim, num_heads)
        h = torch.randn(num_nodes, hidden_dim, requires_grad=True)

        output = pooling(h, batch=None, node_mask=None)
        loss = output.sum()
        loss.backward()

        assert h.grad is not None
        assert not torch.all(h.grad == 0)

    def test_different_head_counts(self):
        """Test with different numbers of attention heads."""
        from graph_cad.models.graph_vae import MultiHeadAttentionPooling

        hidden_dim = 64
        output_dim = 32
        num_nodes = 10
        h = torch.randn(num_nodes, hidden_dim)

        for num_heads in [1, 2, 4, 8]:
            pooling = MultiHeadAttentionPooling(hidden_dim, output_dim, num_heads)
            output = pooling(h)
            assert output.shape == (1, output_dim)


class TestEncoderWithAttentionPooling:
    """Tests for VariableGraphVAEEncoder with attention pooling."""

    @pytest.fixture
    def attention_encoder_config(self):
        """Encoder config with attention pooling."""
        return VariableGraphVAEConfig(
            node_features=13,
            edge_features=2,
            num_face_types=3,
            face_embed_dim=8,
            max_nodes=10,
            max_edges=20,
            hidden_dim=32,
            num_gat_layers=2,
            num_heads=4,
            latent_dim=16,
            encoder_dropout=0.0,
            pooling_type="attention",
            attention_heads=4,
        )

    def test_encoder_forward_attention_pooling(self, attention_encoder_config):
        """Test encoder forward pass with attention pooling."""
        encoder = VariableGraphVAEEncoder(attention_encoder_config)

        batch_size = 2
        max_nodes = attention_encoder_config.max_nodes
        max_edges = attention_encoder_config.max_edges

        # Create fake graph data
        x = torch.randn(batch_size * max_nodes, attention_encoder_config.node_features)
        face_types = torch.randint(0, attention_encoder_config.num_face_types,
                                   (batch_size * max_nodes,))
        edge_index = torch.randint(0, max_nodes, (2, batch_size * max_edges))
        edge_attr = torch.randn(batch_size * max_edges, attention_encoder_config.edge_features)
        batch = torch.repeat_interleave(torch.arange(batch_size), max_nodes)
        node_mask = torch.ones(batch_size * max_nodes)

        mu, logvar = encoder(x, face_types, edge_index, edge_attr, batch, node_mask)

        assert mu.shape == (batch_size, attention_encoder_config.latent_dim)
        assert logvar.shape == (batch_size, attention_encoder_config.latent_dim)

    def test_encoder_gradient_flow_attention_pooling(self, attention_encoder_config):
        """Test gradient flow through encoder with attention pooling."""
        encoder = VariableGraphVAEEncoder(attention_encoder_config)

        max_nodes = attention_encoder_config.max_nodes
        max_edges = attention_encoder_config.max_edges

        x = torch.randn(max_nodes, attention_encoder_config.node_features, requires_grad=True)
        face_types = torch.randint(0, attention_encoder_config.num_face_types, (max_nodes,))
        edge_index = torch.randint(0, max_nodes, (2, max_edges))
        edge_attr = torch.randn(max_edges, attention_encoder_config.edge_features)

        mu, logvar = encoder(x, face_types, edge_index, edge_attr)
        loss = mu.sum() + logvar.sum()
        loss.backward()

        assert x.grad is not None

    def test_attention_vs_mean_pooling_different_outputs(self):
        """Test that attention pooling produces different outputs than mean pooling."""
        # Create two encoders with same random seed initialization but different pooling
        torch.manual_seed(42)
        mean_config = VariableGraphVAEConfig(
            node_features=13,
            edge_features=2,
            num_face_types=3,
            face_embed_dim=8,
            max_nodes=10,
            max_edges=20,
            hidden_dim=32,
            num_gat_layers=2,
            num_heads=4,
            latent_dim=16,
            encoder_dropout=0.0,
            pooling_type="mean",
        )
        mean_encoder = VariableGraphVAEEncoder(mean_config)

        torch.manual_seed(42)
        attn_config = VariableGraphVAEConfig(
            node_features=13,
            edge_features=2,
            num_face_types=3,
            face_embed_dim=8,
            max_nodes=10,
            max_edges=20,
            hidden_dim=32,
            num_gat_layers=2,
            num_heads=4,
            latent_dim=16,
            encoder_dropout=0.0,
            pooling_type="attention",
            attention_heads=4,
        )
        attn_encoder = VariableGraphVAEEncoder(attn_config)

        # Create identical input
        torch.manual_seed(123)
        max_nodes = mean_config.max_nodes
        max_edges = mean_config.max_edges
        x = torch.randn(max_nodes, mean_config.node_features)
        face_types = torch.randint(0, mean_config.num_face_types, (max_nodes,))
        edge_index = torch.randint(0, max_nodes, (2, max_edges))
        edge_attr = torch.randn(max_edges, mean_config.edge_features)

        mean_encoder.eval()
        attn_encoder.eval()

        with torch.no_grad():
            mean_mu, _ = mean_encoder(x, face_types, edge_index, edge_attr)
            attn_mu, _ = attn_encoder(x, face_types, edge_index, edge_attr)

        # Outputs should be different due to different pooling mechanisms
        # (attention has additional parameters not shared with mean pooling)
        assert not torch.allclose(mean_mu, attn_mu, atol=1e-3)
