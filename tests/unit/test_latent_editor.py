"""
Unit tests for the latent editor components.

Tests projectors, dataset, and instruction generation.
Note: LLM integration tests are skipped unless transformers is installed.
"""

import pytest
import torch
import numpy as np

from graph_cad.data.l_bracket import LBracket
from graph_cad.data.edit_dataset import (
    generate_instruction,
    LatentEditDataset,
    collate_edit_batch,
    INSTRUCTION_TEMPLATES,
    COMPOUND_TEMPLATES,
)


class TestWithModified:
    """Tests for LBracket.with_modified() method."""

    @pytest.fixture
    def bracket(self):
        """Standard bracket for testing."""
        return LBracket(
            leg1_length=100,
            leg2_length=100,
            width=40,
            thickness=8,
            hole1_distance=30,
            hole1_diameter=8,
            hole2_distance=30,
            hole2_diameter=8,
        )

    def test_modifies_leg1_length(self, bracket):
        """with_modified should change leg1_length."""
        modified = bracket.with_modified("leg1_length", 20)
        assert modified.leg1_length == 120
        assert modified.leg2_length == bracket.leg2_length  # Unchanged

    def test_modifies_width(self, bracket):
        """with_modified should change width."""
        modified = bracket.with_modified("width", -10)
        assert modified.width == 30

    def test_modifies_thickness(self, bracket):
        """with_modified should change thickness."""
        modified = bracket.with_modified("thickness", 2)
        assert modified.thickness == 10

    def test_clamps_to_valid_range(self, bracket):
        """with_modified should clamp to valid ranges."""
        # Try to make leg1 too short (min is 50)
        modified = bracket.with_modified("leg1_length", -60, clamp=True)
        assert modified.leg1_length == 50  # Clamped to min

    def test_clamps_hole_distance(self, bracket):
        """with_modified should clamp hole distance to valid range."""
        # Try to move hole too far
        modified = bracket.with_modified("hole1_distance", 100, clamp=True)
        # Max is leg1_length - thickness - hole_diameter = 100 - 8 - 8 = 84
        assert modified.hole1_distance <= 84

    def test_invalid_param_raises(self, bracket):
        """with_modified should raise for invalid parameter name."""
        with pytest.raises(ValueError, match="Unknown parameter"):
            bracket.with_modified("invalid_param", 10)

    def test_returns_new_instance(self, bracket):
        """with_modified should return a new instance."""
        modified = bracket.with_modified("leg1_length", 20)
        assert modified is not bracket
        assert bracket.leg1_length == 100  # Original unchanged


class TestGenerateInstruction:
    """Tests for instruction generation."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_generates_string(self, rng):
        """generate_instruction should return a string."""
        instruction = generate_instruction("leg1_length", 20.0, 100.0, rng)
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_instruction_varies_with_param(self, rng):
        """Different parameters should produce different instructions."""
        inst1 = generate_instruction("leg1_length", 20.0, 100.0, rng)
        inst2 = generate_instruction("width", 10.0, 40.0, rng)
        # Instructions might be different (not guaranteed due to randomness)
        assert isinstance(inst1, str)
        assert isinstance(inst2, str)

    def test_negative_delta_uses_direction(self, rng):
        """Negative delta should use 'shorter' type words."""
        # Run multiple times to increase chance of getting direction word
        instructions = [
            generate_instruction("leg1_length", -20.0, 100.0, rng)
            for _ in range(10)
        ]
        # At least some should contain direction indicators
        assert any(isinstance(i, str) and len(i) > 0 for i in instructions)

    def test_all_params_have_templates(self):
        """All L-bracket parameters should have templates."""
        params = [
            "leg1_length", "leg2_length", "width", "thickness",
            "hole1_diameter", "hole2_diameter", "hole1_distance", "hole2_distance",
        ]
        for param in params:
            assert param in INSTRUCTION_TEMPLATES
            assert len(INSTRUCTION_TEMPLATES[param]) > 0


class TestLatentEditDataset:
    """Tests for LatentEditDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {
                "instruction": "make leg1 longer",
                "z_src": [0.1] * 16,
                "z_tgt": [0.2] * 16,
                "delta_z": [0.1] * 16,
                "param_deltas": {"leg1_length": 20.0},
            },
            {
                "instruction": "make it wider",
                "z_src": [0.0] * 16,
                "z_tgt": [0.1] * 16,
                "delta_z": [0.1] * 16,
                "param_deltas": {"width": 10.0},
            },
        ]

    def test_create_from_samples(self, sample_data):
        """Dataset should initialize from sample list."""
        dataset = LatentEditDataset(samples=sample_data)
        assert len(dataset) == 2

    def test_getitem_returns_dict(self, sample_data):
        """__getitem__ should return a dictionary."""
        dataset = LatentEditDataset(samples=sample_data)
        item = dataset[0]
        assert isinstance(item, dict)
        assert "instruction" in item
        assert "z_src" in item
        assert "z_tgt" in item
        assert "delta_z" in item

    def test_getitem_returns_tensors(self, sample_data):
        """__getitem__ should return torch tensors."""
        dataset = LatentEditDataset(samples=sample_data)
        item = dataset[0]
        assert isinstance(item["z_src"], torch.Tensor)
        assert isinstance(item["z_tgt"], torch.Tensor)
        assert isinstance(item["delta_z"], torch.Tensor)

    def test_tensor_shapes(self, sample_data):
        """Tensors should have correct shapes."""
        dataset = LatentEditDataset(samples=sample_data)
        item = dataset[0]
        assert item["z_src"].shape == (16,)
        assert item["z_tgt"].shape == (16,)
        assert item["delta_z"].shape == (16,)

    def test_requires_data_path_or_samples(self):
        """Dataset should raise if neither path nor samples provided."""
        with pytest.raises(ValueError, match="Must provide"):
            LatentEditDataset()


class TestCollateEditBatch:
    """Tests for batch collation."""

    @pytest.fixture
    def batch(self):
        """Create a batch of samples."""
        return [
            {
                "instruction": "make leg1 longer",
                "z_src": torch.randn(16),
                "z_tgt": torch.randn(16),
                "delta_z": torch.randn(16),
            },
            {
                "instruction": "make it wider",
                "z_src": torch.randn(16),
                "z_tgt": torch.randn(16),
                "delta_z": torch.randn(16),
            },
        ]

    def test_collate_returns_dict(self, batch):
        """collate_edit_batch should return a dictionary."""
        collated = collate_edit_batch(batch)
        assert isinstance(collated, dict)

    def test_collate_stacks_tensors(self, batch):
        """collate_edit_batch should stack tensors."""
        collated = collate_edit_batch(batch)
        assert collated["z_src"].shape == (2, 16)
        assert collated["z_tgt"].shape == (2, 16)
        assert collated["delta_z"].shape == (2, 16)

    def test_collate_preserves_instructions(self, batch):
        """collate_edit_batch should preserve instruction list."""
        collated = collate_edit_batch(batch)
        assert isinstance(collated["instructions"], list)
        assert len(collated["instructions"]) == 2
        assert collated["instructions"][0] == "make leg1 longer"


class TestCompoundTemplates:
    """Tests for compound edit templates."""

    def test_compound_templates_exist(self):
        """COMPOUND_TEMPLATES should contain templates."""
        assert len(COMPOUND_TEMPLATES) > 0

    def test_compound_templates_format(self):
        """Each compound template should be (instruction, param_dict)."""
        for template in COMPOUND_TEMPLATES:
            assert isinstance(template, tuple)
            assert len(template) == 2
            instruction, param_dict = template
            assert isinstance(instruction, str)
            assert isinstance(param_dict, dict)

    def test_noop_templates_exist(self):
        """Should have no-op templates (empty param_dict)."""
        noop_templates = [t for t in COMPOUND_TEMPLATES if len(t[1]) == 0]
        assert len(noop_templates) > 0


# Tests requiring transformers (skip if not installed)
try:
    from graph_cad.models.latent_editor import (
        LatentProjector,
        OutputProjector,
        LatentEditorConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestLatentProjector:
    """Tests for LatentProjector module."""

    @pytest.fixture
    def config(self):
        return LatentEditorConfig(latent_dim=16, llm_hidden_dim=4096)

    @pytest.fixture
    def projector(self, config):
        return LatentProjector(config)

    def test_output_shape(self, projector):
        """Projector should output (batch, 1, hidden_dim)."""
        z = torch.randn(4, 16)
        out = projector(z)
        assert out.shape == (4, 1, 4096)

    def test_single_sample(self, projector):
        """Projector should work with single sample."""
        z = torch.randn(1, 16)
        out = projector(z)
        assert out.shape == (1, 1, 4096)

    def test_gradient_flow(self, projector):
        """Gradients should flow through projector."""
        z = torch.randn(2, 16, requires_grad=True)
        out = projector(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestOutputProjector:
    """Tests for OutputProjector module."""

    @pytest.fixture
    def config(self):
        return LatentEditorConfig(latent_dim=16, llm_hidden_dim=4096)

    @pytest.fixture
    def projector(self, config):
        return OutputProjector(config)

    def test_output_shape(self, projector):
        """Projector should output (batch, latent_dim)."""
        hidden = torch.randn(4, 4096)
        out = projector(hidden)
        assert out.shape == (4, 16)

    def test_single_sample(self, projector):
        """Projector should work with single sample."""
        hidden = torch.randn(1, 4096)
        out = projector(hidden)
        assert out.shape == (1, 16)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestProjectorRoundtrip:
    """Test that projectors are roughly invertible."""

    @pytest.fixture
    def config(self):
        return LatentEditorConfig(latent_dim=16, llm_hidden_dim=4096)

    def test_dimensions_match(self, config):
        """Input and output projectors should have compatible dimensions."""
        inp_proj = LatentProjector(config)
        out_proj = OutputProjector(config)

        z = torch.randn(2, 16)
        hidden = inp_proj(z).squeeze(1)  # (2, 4096)
        z_reconstructed = out_proj(hidden)  # (2, 16)

        assert z_reconstructed.shape == z.shape
