"""Unit tests for L-bracket generator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from graph_cad.data import LBracket, LBracketRanges


class TestLBracketValidation:
    """Test L-bracket geometry validation."""

    def test_valid_bracket(self):
        """Valid parameters should create bracket without error."""
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
        assert bracket.leg1_length == 100
        assert bracket.leg2_length == 80

    def test_negative_parameter_raises(self):
        """Negative parameters should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            LBracket(
                leg1_length=-100,
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_distance=20,
                hole1_diameter=8,
                hole2_distance=15,
                hole2_diameter=6,
            )

    def test_leg_shorter_than_thickness_raises(self):
        """Leg shorter than thickness should raise ValueError."""
        with pytest.raises(ValueError, match="must be > thickness"):
            LBracket(
                leg1_length=5,  # Same as thickness
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_distance=20,
                hole1_diameter=8,
                hole2_distance=15,
                hole2_diameter=6,
            )

    def test_hole_too_close_to_end_raises(self):
        """Hole distance less than diameter should raise ValueError."""
        with pytest.raises(ValueError, match="hole1_distance.*must be >="):
            LBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_distance=5,  # Less than diameter
                hole1_diameter=8,
                hole2_distance=15,
                hole2_diameter=6,
            )

    def test_hole_encroaches_corner_raises(self):
        """Hole too close to corner should raise ValueError."""
        with pytest.raises(ValueError, match="leg1_length - thickness - hole1_diameter"):
            LBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_distance=95,  # Too far from end, into corner
                hole1_diameter=8,
                hole2_distance=15,
                hole2_diameter=6,
            )

    def test_width_too_narrow_for_holes_raises(self):
        """Width insufficient for holes should raise ValueError."""
        with pytest.raises(ValueError, match="width.*must be >="):
            LBracket(
                leg1_length=100,
                leg2_length=80,
                width=10,  # Too narrow for 8mm hole
                thickness=5,
                hole1_distance=20,
                hole1_diameter=8,
                hole2_distance=15,
                hole2_diameter=6,
            )


class TestLBracketSerialization:
    """Test parameter serialization."""

    def test_to_dict(self):
        """to_dict should return all parameters."""
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
        d = bracket.to_dict()

        assert d["leg1_length"] == 100
        assert d["leg2_length"] == 80
        assert d["width"] == 30
        assert d["thickness"] == 5
        assert d["hole1_distance"] == 20
        assert d["hole1_diameter"] == 8
        assert d["hole2_distance"] == 15
        assert d["hole2_diameter"] == 6

    def test_from_dict_roundtrip(self):
        """from_dict should recreate identical bracket."""
        original = LBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_distance=20,
            hole1_diameter=8,
            hole2_distance=15,
            hole2_diameter=6,
        )
        recreated = LBracket.from_dict(original.to_dict())

        assert recreated.to_dict() == original.to_dict()


class TestLBracketRandom:
    """Test random bracket generation."""

    def test_random_produces_valid_bracket(self):
        """random() should produce valid bracket."""
        rng = np.random.default_rng(42)
        bracket = LBracket.random(rng)

        # If we get here without ValueError, validation passed
        assert bracket.leg1_length > 0
        assert bracket.leg2_length > 0

    def test_random_respects_ranges(self):
        """random() should respect parameter ranges."""
        rng = np.random.default_rng(42)
        ranges = LBracketRanges(
            leg1_length=(60, 80),
            leg2_length=(70, 90),
            width=(25, 35),
            thickness=(4, 6),
            hole1_diameter=(5, 7),
            hole2_diameter=(5, 7),
        )

        for _ in range(20):
            bracket = LBracket.random(rng, ranges)

            assert 60 <= bracket.leg1_length <= 80
            assert 70 <= bracket.leg2_length <= 90
            assert 25 <= bracket.width <= 35
            assert 4 <= bracket.thickness <= 6
            assert 5 <= bracket.hole1_diameter <= 7
            assert 5 <= bracket.hole2_diameter <= 7

    def test_random_deterministic_with_seed(self):
        """Same seed should produce same bracket."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        bracket1 = LBracket.random(rng1)
        bracket2 = LBracket.random(rng2)

        assert bracket1.to_dict() == bracket2.to_dict()

    def test_random_batch_all_valid(self):
        """Batch generation should produce all valid brackets."""
        rng = np.random.default_rng(42)

        # Generate 100 brackets, all should be valid
        for i in range(100):
            bracket = LBracket.random(rng)
            # Validation happens in __init__, so reaching here means valid
            assert bracket.hole1_distance >= bracket.hole1_diameter
            assert bracket.hole2_distance >= bracket.hole2_diameter


class TestLBracketGeometry:
    """Test CadQuery geometry generation."""

    def test_to_solid_returns_workplane(self):
        """to_solid should return CadQuery Workplane."""
        import cadquery as cq

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
        solid = bracket.to_solid()

        assert isinstance(solid, cq.Workplane)

    def test_to_solid_has_10_faces(self):
        """Generated solid should have exactly 10 faces."""
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
        solid = bracket.to_solid()
        faces = solid.faces().vals()

        assert len(faces) == 10

    def test_to_step_creates_file(self):
        """to_step should create a STEP file."""
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

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.step"
            bracket.to_step(path)

            assert path.exists()
            assert path.stat().st_size > 0


class TestLBracketRepr:
    """Test string representation."""

    def test_repr_contains_parameters(self):
        """repr should contain all parameters."""
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
        r = repr(bracket)

        assert "leg1_length=100.00" in r
        assert "leg2_length=80.00" in r
        assert "LBracket(" in r
