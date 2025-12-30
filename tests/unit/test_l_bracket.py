"""Unit tests for L-bracket generator."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from graph_cad.data import LBracket, LBracketRanges
from graph_cad.data.l_bracket import VariableLBracket, VariableLBracketRanges


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


# =============================================================================
# VariableLBracket Tests (Variable Topology)
# =============================================================================


class TestVariableLBracketValidation:
    """Test VariableLBracket geometry validation."""

    def test_valid_bracket_no_holes_no_fillet(self):
        """Valid minimal bracket should create without error."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
        )
        assert bracket.leg1_length == 100
        assert bracket.num_holes_leg1 == 0
        assert bracket.num_holes_leg2 == 0
        assert bracket.has_fillet is False

    def test_valid_bracket_with_holes(self):
        """Bracket with holes should create without error."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_diameters=(8,),
            hole1_distances=(20,),
            hole2_diameters=(6, 6),
            hole2_distances=(15, 40),
        )
        assert bracket.num_holes_leg1 == 1
        assert bracket.num_holes_leg2 == 2

    def test_valid_bracket_with_fillet(self):
        """Bracket with fillet should create without error."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            fillet_radius=3.0,
        )
        assert bracket.has_fillet is True
        assert bracket.fillet_radius == 3.0

    def test_negative_leg_length_raises(self):
        """Negative leg length should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            VariableLBracket(
                leg1_length=-100,
                leg2_length=80,
                width=30,
                thickness=5,
            )

    def test_leg_shorter_than_thickness_raises(self):
        """Leg shorter than thickness should raise ValueError."""
        with pytest.raises(ValueError, match="must be > thickness"):
            VariableLBracket(
                leg1_length=5,
                leg2_length=80,
                width=30,
                thickness=5,
            )

    def test_mismatched_hole_arrays_raises(self):
        """Mismatched hole diameter/distance arrays should raise ValueError."""
        with pytest.raises(ValueError, match="must have same length"):
            VariableLBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_diameters=(8, 8),
                hole1_distances=(20,),  # Wrong length
            )

    def test_hole_too_close_raises(self):
        """Hole too close to end should raise ValueError."""
        with pytest.raises(ValueError, match="must be >="):
            VariableLBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_diameters=(8,),
                hole1_distances=(5,),  # Less than diameter
            )

    def test_overlapping_holes_raises(self):
        """Overlapping holes should raise ValueError."""
        with pytest.raises(ValueError, match="too close"):
            VariableLBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=5,
                hole1_diameters=(8, 8),
                hole1_distances=(20, 22),  # Only 2mm apart, need more spacing
            )

    def test_fillet_too_large_raises(self):
        """Fillet radius too large should raise ValueError."""
        with pytest.raises(ValueError, match="fillet_radius.*too large"):
            VariableLBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=5,
                fillet_radius=50.0,  # Too large
            )

    def test_hole_interferes_with_fillet_raises(self):
        """Hole too close to fillet should raise ValueError."""
        # Fillet radius 8mm + 2mm wall = 10mm clearance needed from corner
        # Hole at distance 87mm from end of leg1 (100mm) puts hole center at X=13
        # Hole edge closest to corner at X = 13 - 4 = 9mm
        # Corner at X = thickness = 10mm, fillet extends to X = 18mm
        # So hole edge at X=9 is inside fillet zone -> should fail
        with pytest.raises(ValueError, match="accounting for fillet"):
            VariableLBracket(
                leg1_length=100,
                leg2_length=80,
                width=30,
                thickness=10,
                fillet_radius=8.0,
                hole1_diameters=(8,),
                hole1_distances=(87,),  # Too close to fillet
            )

    def test_hole_with_fillet_clearance_ok(self):
        """Hole with sufficient fillet clearance should succeed."""
        # Same setup but with hole further from corner
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=10,
            fillet_radius=8.0,
            hole1_diameters=(8,),
            hole1_distances=(20,),  # Far enough from fillet
        )
        assert bracket.fillet_radius == 8.0
        assert bracket.num_holes_leg1 == 1


class TestVariableLBracketProperties:
    """Test VariableLBracket computed properties."""

    def test_expected_num_faces_no_holes_no_fillet(self):
        """Base L-bracket should have 6 faces."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
        )
        assert bracket.expected_num_faces == 6

    def test_expected_num_faces_with_one_hole(self):
        """One hole adds 2 cylindrical faces."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_diameters=(8,),
            hole1_distances=(20,),
        )
        assert bracket.expected_num_faces == 8  # 6 + 2

    def test_expected_num_faces_with_fillet(self):
        """Fillet adds 1 torus face."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            fillet_radius=3.0,
        )
        assert bracket.expected_num_faces == 7  # 6 + 1

    def test_expected_num_faces_full_topology(self):
        """Full topology: 6 base + 2*4 holes + 1 fillet = 15."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            fillet_radius=3.0,
            hole1_diameters=(8, 8),
            hole1_distances=(20, 60),
            hole2_diameters=(6, 6),
            hole2_distances=(15, 45),
        )
        assert bracket.expected_num_faces == 15  # 6 + 8 + 1


class TestVariableLBracketSerialization:
    """Test parameter serialization."""

    def test_to_dict(self):
        """to_dict should return all parameters."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            fillet_radius=3.0,
            hole1_diameters=(8,),
            hole1_distances=(20,),
        )
        d = bracket.to_dict()

        assert d["leg1_length"] == 100
        assert d["leg2_length"] == 80
        assert d["width"] == 30
        assert d["thickness"] == 5
        assert d["fillet_radius"] == 3.0
        assert d["hole1_diameters"] == [8]
        assert d["hole1_distances"] == [20]
        assert d["num_holes_leg1"] == 1
        assert d["has_fillet"] is True

    def test_from_dict_roundtrip(self):
        """from_dict should recreate identical bracket."""
        original = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            fillet_radius=3.0,
            hole1_diameters=(8, 6),
            hole1_distances=(20, 60),
            hole2_diameters=(6,),
            hole2_distances=(15,),
        )
        recreated = VariableLBracket.from_dict(original.to_dict())

        assert recreated.leg1_length == original.leg1_length
        assert recreated.fillet_radius == original.fillet_radius
        assert recreated.hole1_diameters == original.hole1_diameters
        assert recreated.hole2_distances == original.hole2_distances


class TestVariableLBracketRandom:
    """Test random bracket generation."""

    def test_random_produces_valid_bracket(self):
        """random() should produce valid bracket."""
        rng = np.random.default_rng(42)
        bracket = VariableLBracket.random(rng)

        # If we get here without ValueError, validation passed
        assert bracket.leg1_length > 0
        assert bracket.leg2_length > 0
        assert bracket.num_holes_leg1 >= 0
        assert bracket.num_holes_leg2 >= 0

    def test_random_respects_ranges(self):
        """random() should respect parameter ranges."""
        rng = np.random.default_rng(42)
        ranges = VariableLBracketRanges(
            leg1_length=(60, 80),
            leg2_length=(70, 90),
            width=(25, 35),
            thickness=(4, 6),
        )

        for _ in range(20):
            bracket = VariableLBracket.random(rng, ranges)

            assert 60 <= bracket.leg1_length <= 80
            assert 70 <= bracket.leg2_length <= 90
            assert 25 <= bracket.width <= 35
            assert 4 <= bracket.thickness <= 6

    def test_random_deterministic_with_seed(self):
        """Same seed should produce same bracket."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        bracket1 = VariableLBracket.random(rng1)
        bracket2 = VariableLBracket.random(rng2)

        assert bracket1.leg1_length == bracket2.leg1_length
        assert bracket1.num_holes_leg1 == bracket2.num_holes_leg1
        assert bracket1.has_fillet == bracket2.has_fillet

    def test_random_batch_all_valid(self):
        """Batch generation should produce all valid brackets."""
        rng = np.random.default_rng(42)

        for _ in range(50):
            bracket = VariableLBracket.random(rng)
            # Validation happens in __init__, reaching here means valid
            assert bracket.expected_num_faces >= 6
            assert bracket.expected_num_faces <= 15

    def test_random_produces_variety(self):
        """Random generation should produce various topologies."""
        rng = np.random.default_rng(42)

        topologies = set()
        for _ in range(100):
            bracket = VariableLBracket.random(rng)
            topo = (bracket.num_holes_leg1, bracket.num_holes_leg2, bracket.has_fillet)
            topologies.add(topo)

        # Should see at least a few different configurations
        assert len(topologies) >= 5


class TestVariableLBracketGeometry:
    """Test CadQuery geometry generation."""

    def test_to_solid_returns_workplane(self):
        """to_solid should return CadQuery Workplane."""
        import cadquery as cq

        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
        )
        solid = bracket.to_solid()

        assert isinstance(solid, cq.Workplane)

    def test_to_solid_base_has_at_least_6_faces(self):
        """Base L-bracket should have at least 6 faces."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
        )
        solid = bracket.to_solid()
        faces = solid.faces().vals()

        # CadQuery may create additional faces depending on geometry
        assert len(faces) >= 6

    def test_to_solid_with_holes(self):
        """Bracket with holes should have additional faces."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_diameters=(8,),
            hole1_distances=(20,),
        )
        solid = bracket.to_solid()
        faces = solid.faces().vals()

        # Should have more faces than minimal (6) due to holes
        assert len(faces) >= 8

    def test_to_solid_random_bracket(self):
        """Random brackets should generate valid geometry."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            bracket = VariableLBracket.random(rng)
            solid = bracket.to_solid()
            faces = solid.faces().vals()

            # Should have at least 6 faces
            assert len(faces) >= 6

    def test_to_step_creates_file(self):
        """to_step should create a STEP file."""
        bracket = VariableLBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_diameters=(8,),
            hole1_distances=(20,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.step"
            bracket.to_step(path)

            assert path.exists()
            assert path.stat().st_size > 0


class TestVariableLBracketConversion:
    """Test conversion between fixed and variable L-brackets."""

    def test_from_fixed_lbracket(self):
        """Should convert fixed to variable topology bracket."""
        fixed = LBracket(
            leg1_length=100,
            leg2_length=80,
            width=30,
            thickness=5,
            hole1_distance=20,
            hole1_diameter=8,
            hole2_distance=15,
            hole2_diameter=6,
        )

        variable = VariableLBracket.from_fixed_lbracket(fixed)

        assert variable.leg1_length == fixed.leg1_length
        assert variable.leg2_length == fixed.leg2_length
        assert variable.width == fixed.width
        assert variable.thickness == fixed.thickness
        assert variable.num_holes_leg1 == 1
        assert variable.num_holes_leg2 == 1
        assert variable.hole1_diameters == (fixed.hole1_diameter,)
        assert variable.hole1_distances == (fixed.hole1_distance,)
