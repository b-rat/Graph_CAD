"""
L-Bracket CAD generator for synthetic training data.

Generates parametric L-brackets with variable topology:
- Original fixed topology: 2 holes, no fillets (10 faces)
- Variable topology: 0-2 holes per leg, optional fillet (6-15 faces)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass
class LBracketRanges:
    """Parameter ranges for L-bracket generation."""

    leg1_length: tuple[float, float] = (50.0, 200.0)
    leg2_length: tuple[float, float] = (50.0, 200.0)
    width: tuple[float, float] = (20.0, 60.0)
    thickness: tuple[float, float] = (3.0, 12.0)
    hole1_diameter: tuple[float, float] = (4.0, 12.0)
    hole2_diameter: tuple[float, float] = (4.0, 12.0)


class LBracket:
    """
    Parametric L-bracket with two through-holes.

    Coordinate system:
    - Origin at outer corner where leg 1 and leg 2 meet
    - Leg 1 on X-Y plane, extends along +X
    - Leg 2 on Y-Z plane, extends along +Z
    - Width along Y axis

    Parameters:
        leg1_length: Length of leg 1 along +X (mm)
        leg2_length: Length of leg 2 along +Z (mm)
        width: Width along Y axis (mm)
        thickness: Material thickness (mm)
        hole1_distance: Distance from end of leg 1 to hole center (mm)
        hole1_diameter: Diameter of hole in leg 1 (mm)
        hole2_distance: Distance from end of leg 2 to hole center (mm)
        hole2_diameter: Diameter of hole in leg 2 (mm)
    """

    def __init__(
        self,
        leg1_length: float,
        leg2_length: float,
        width: float,
        thickness: float,
        hole1_distance: float,
        hole1_diameter: float,
        hole2_distance: float,
        hole2_diameter: float,
    ):
        self.leg1_length = leg1_length
        self.leg2_length = leg2_length
        self.width = width
        self.thickness = thickness
        self.hole1_distance = hole1_distance
        self.hole1_diameter = hole1_diameter
        self.hole2_distance = hole2_distance
        self.hole2_diameter = hole2_diameter

        self._validate()

    def _validate(self) -> None:
        """Validate geometry constraints. Raises ValueError if invalid."""
        errors = []

        # Basic positivity
        for name in [
            "leg1_length",
            "leg2_length",
            "width",
            "thickness",
            "hole1_distance",
            "hole1_diameter",
            "hole2_distance",
            "hole2_diameter",
        ]:
            value = getattr(self, name)
            if value <= 0:
                errors.append(f"{name} must be positive, got {value}")

        # Legs must be longer than thickness
        if self.leg1_length <= self.thickness:
            errors.append(
                f"leg1_length ({self.leg1_length}) must be > thickness ({self.thickness})"
            )
        if self.leg2_length <= self.thickness:
            errors.append(
                f"leg2_length ({self.leg2_length}) must be > thickness ({self.thickness})"
            )

        # Hole distance from end (min 1 diameter)
        if self.hole1_distance < self.hole1_diameter:
            errors.append(
                f"hole1_distance ({self.hole1_distance}) must be >= "
                f"hole1_diameter ({self.hole1_diameter})"
            )
        if self.hole2_distance < self.hole2_diameter:
            errors.append(
                f"hole2_distance ({self.hole2_distance}) must be >= "
                f"hole2_diameter ({self.hole2_diameter})"
            )

        # Hole must not encroach into corner region (min 1 diameter from corner)
        max_hole1_dist = self.leg1_length - self.thickness - self.hole1_diameter
        if self.hole1_distance > max_hole1_dist:
            errors.append(
                f"hole1_distance ({self.hole1_distance}) must be <= "
                f"{max_hole1_dist:.2f} (leg1_length - thickness - hole1_diameter)"
            )
        max_hole2_dist = self.leg2_length - self.thickness - self.hole2_diameter
        if self.hole2_distance > max_hole2_dist:
            errors.append(
                f"hole2_distance ({self.hole2_distance}) must be <= "
                f"{max_hole2_dist:.2f} (leg2_length - thickness - hole2_diameter)"
            )

        # Width must accommodate holes (min 2 diameters for centerline placement)
        min_width_for_hole1 = 2 * self.hole1_diameter
        min_width_for_hole2 = 2 * self.hole2_diameter
        min_width = max(min_width_for_hole1, min_width_for_hole2)
        if self.width < min_width:
            errors.append(
                f"width ({self.width}) must be >= {min_width} "
                f"(2 × max hole diameter)"
            )

        if errors:
            raise ValueError("Invalid L-bracket geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """
        Generate CadQuery solid geometry.

        Returns:
            CadQuery Workplane containing the L-bracket solid.
        """
        # Build L-shape profile on X-Z plane, then extrude along -Y
        # Profile vertices (counterclockwise from origin):
        #   (0, 0) -> (leg1_length, 0) -> (leg1_length, thickness)
        #   -> (thickness, thickness) -> (thickness, leg2_length)
        #   -> (0, leg2_length) -> (0, 0)

        profile_points = [
            (0, 0),
            (self.leg1_length, 0),
            (self.leg1_length, self.thickness),
            (self.thickness, self.thickness),
            (self.thickness, self.leg2_length),
            (0, self.leg2_length),
        ]

        # Create L-bracket body by extruding profile along Y
        bracket = (
            cq.Workplane("XZ")
            .polyline(profile_points)
            .close()
            .extrude(self.width)
        )

        # Hole 1: through leg 1 (vertical hole, parallel to Z axis)
        # Face is at Z=thickness, spans X=[thickness, leg1_length], Y=[-width, 0]
        # Hole center: X = leg1_length - hole1_distance, Y = -width/2
        # Use ">Z[-2]" to select leg1 top face (second highest Z), not leg2 top
        face1_center_x = (self.thickness + self.leg1_length) / 2
        hole1_global_x = self.leg1_length - self.hole1_distance
        wp1_offset_x = hole1_global_x - face1_center_x

        bracket = (
            bracket
            .faces(">Z[-2]")
            .workplane(centerOption="CenterOfBoundBox")
            .moveTo(wp1_offset_x, 0)
            .hole(self.hole1_diameter, self.thickness)
        )

        # Hole 2: through leg 2 (horizontal hole, parallel to X axis)
        # Face is at X=thickness, spans Y=[-width, 0], Z=[thickness, leg2_length]
        # Hole center: Y = -width/2, Z = leg2_length - hole2_distance
        # Use ">X[-2]" to select leg2 inner face (second highest X), not outer end
        face2_center_z = (self.thickness + self.leg2_length) / 2
        hole2_global_z = self.leg2_length - self.hole2_distance
        wp2_offset_z = hole2_global_z - face2_center_z

        bracket = (
            bracket
            .faces(">X[-2]")
            .workplane(centerOption="CenterOfBoundBox")
            .moveTo(0, wp2_offset_z)
            .hole(self.hole2_diameter, self.thickness)
        )

        return bracket

    def to_step(self, path: str | Path) -> None:
        """
        Export L-bracket to STEP file.

        Args:
            path: Output file path for STEP file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict[str, float]:
        """
        Return parameters as dictionary.

        Useful for storing metadata/labels alongside generated geometry.

        Returns:
            Dictionary of parameter names to values.
        """
        return {
            "leg1_length": self.leg1_length,
            "leg2_length": self.leg2_length,
            "width": self.width,
            "thickness": self.thickness,
            "hole1_distance": self.hole1_distance,
            "hole1_diameter": self.hole1_diameter,
            "hole2_distance": self.hole2_distance,
            "hole2_diameter": self.hole2_diameter,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> LBracket:
        """
        Create LBracket from parameter dictionary.

        Args:
            params: Dictionary with parameter names and values.

        Returns:
            LBracket instance.
        """
        return cls(
            leg1_length=params["leg1_length"],
            leg2_length=params["leg2_length"],
            width=params["width"],
            thickness=params["thickness"],
            hole1_distance=params["hole1_distance"],
            hole1_diameter=params["hole1_diameter"],
            hole2_distance=params["hole2_distance"],
            hole2_diameter=params["hole2_diameter"],
        )

    @classmethod
    def random(
        cls,
        rng: Generator,
        ranges: LBracketRanges | None = None,
    ) -> LBracket:
        """
        Generate random L-bracket with valid geometry.

        Samples independent parameters uniformly, then samples dependent
        parameters (hole distances) within their valid ranges.

        Args:
            rng: NumPy random generator for reproducibility.
            ranges: Parameter ranges. Uses defaults if None.

        Returns:
            LBracket instance with randomly sampled parameters.
        """
        if ranges is None:
            ranges = LBracketRanges()

        # Sample independent parameters uniformly
        leg1_length = rng.uniform(*ranges.leg1_length)
        leg2_length = rng.uniform(*ranges.leg2_length)
        thickness = rng.uniform(*ranges.thickness)
        hole1_diameter = rng.uniform(*ranges.hole1_diameter)
        hole2_diameter = rng.uniform(*ranges.hole2_diameter)

        # Width must accommodate both holes
        min_width = max(2 * hole1_diameter, 2 * hole2_diameter, ranges.width[0])
        max_width = ranges.width[1]
        width = rng.uniform(min_width, max_width)

        # Sample hole distances within valid range
        # Min: 1 diameter from end
        # Max: 1 diameter from corner (leg_length - thickness - diameter)
        hole1_dist_min = hole1_diameter
        hole1_dist_max = leg1_length - thickness - hole1_diameter
        hole1_distance = rng.uniform(hole1_dist_min, hole1_dist_max)

        hole2_dist_min = hole2_diameter
        hole2_dist_max = leg2_length - thickness - hole2_diameter
        hole2_distance = rng.uniform(hole2_dist_min, hole2_dist_max)

        return cls(
            leg1_length=leg1_length,
            leg2_length=leg2_length,
            width=width,
            thickness=thickness,
            hole1_distance=hole1_distance,
            hole1_diameter=hole1_diameter,
            hole2_distance=hole2_distance,
            hole2_diameter=hole2_diameter,
        )

    def with_modified(
        self,
        param: str,
        delta: float,
        clamp: bool = True,
        ranges: LBracketRanges | None = None,
    ) -> LBracket:
        """
        Create new L-bracket with one parameter modified.

        Args:
            param: Parameter name to modify (e.g., 'leg1_length', 'width').
            delta: Amount to add to the parameter value.
            clamp: If True, clamp to valid ranges. If False, raise on invalid.
            ranges: Parameter ranges for clamping. Uses defaults if None.

        Returns:
            New LBracket instance with modified parameter.

        Raises:
            ValueError: If param is not a valid parameter name.
            ValueError: If clamp=False and resulting geometry is invalid.
        """
        valid_params = [
            "leg1_length",
            "leg2_length",
            "width",
            "thickness",
            "hole1_distance",
            "hole1_diameter",
            "hole2_distance",
            "hole2_diameter",
        ]
        if param not in valid_params:
            raise ValueError(
                f"Unknown parameter '{param}'. Valid: {valid_params}"
            )

        if ranges is None:
            ranges = LBracketRanges()

        # Get current parameters
        params = self.to_dict()
        new_value = params[param] + delta

        # Clamp to basic range if requested
        if clamp:
            if param == "leg1_length":
                new_value = np.clip(new_value, ranges.leg1_length[0], ranges.leg1_length[1])
            elif param == "leg2_length":
                new_value = np.clip(new_value, ranges.leg2_length[0], ranges.leg2_length[1])
            elif param == "width":
                # Must be >= 2 * max hole diameter
                min_width = max(
                    2 * params["hole1_diameter"],
                    2 * params["hole2_diameter"],
                    ranges.width[0],
                )
                new_value = np.clip(new_value, min_width, ranges.width[1])
            elif param == "thickness":
                new_value = np.clip(new_value, ranges.thickness[0], ranges.thickness[1])
            elif param == "hole1_diameter":
                new_value = np.clip(new_value, ranges.hole1_diameter[0], ranges.hole1_diameter[1])
            elif param == "hole2_diameter":
                new_value = np.clip(new_value, ranges.hole2_diameter[0], ranges.hole2_diameter[1])
            elif param == "hole1_distance":
                # Must be >= diameter and <= leg1_length - thickness - diameter
                min_dist = params["hole1_diameter"]
                max_dist = params["leg1_length"] - params["thickness"] - params["hole1_diameter"]
                new_value = np.clip(new_value, min_dist, max_dist)
            elif param == "hole2_distance":
                min_dist = params["hole2_diameter"]
                max_dist = params["leg2_length"] - params["thickness"] - params["hole2_diameter"]
                new_value = np.clip(new_value, min_dist, max_dist)

        params[param] = float(new_value)
        return LBracket.from_dict(params)

    def __repr__(self) -> str:
        return (
            f"LBracket("
            f"leg1_length={self.leg1_length:.2f}, "
            f"leg2_length={self.leg2_length:.2f}, "
            f"width={self.width:.2f}, "
            f"thickness={self.thickness:.2f}, "
            f"hole1_distance={self.hole1_distance:.2f}, "
            f"hole1_diameter={self.hole1_diameter:.2f}, "
            f"hole2_distance={self.hole2_distance:.2f}, "
            f"hole2_diameter={self.hole2_diameter:.2f})"
        )


# =============================================================================
# Variable Topology L-Bracket (Phase 2)
# =============================================================================


@dataclass
class VariableLBracketRanges:
    """Parameter ranges for variable topology L-bracket generation."""

    # Core geometry (same as fixed topology)
    leg1_length: tuple[float, float] = (50.0, 200.0)
    leg2_length: tuple[float, float] = (50.0, 200.0)
    width: tuple[float, float] = (20.0, 60.0)
    thickness: tuple[float, float] = (3.0, 12.0)

    # Hole diameters (used for all holes)
    hole_diameter: tuple[float, float] = (4.0, 12.0)

    # Fillet (0 = no fillet)
    fillet_radius: tuple[float, float] = (0.0, 8.0)

    # Topology variation probabilities
    prob_fillet: float = 0.5  # Probability of having a fillet
    prob_hole_configs: tuple[float, ...] = (0.1, 0.4, 0.5)  # P(0), P(1), P(2) holes per leg


class VariableLBracket:
    """
    Parametric L-bracket with variable topology.

    Supports:
    - 0, 1, or 2 holes per leg
    - Optional corner fillet
    - Topology varies from 6 faces (no holes, no fillet) to 15 faces (4 holes, fillet)

    Coordinate system (same as LBracket):
    - Origin at outer corner where leg 1 and leg 2 meet
    - Leg 1 on X-Y plane, extends along +X
    - Leg 2 on Y-Z plane, extends along +Z
    - Width along Y axis
    """

    def __init__(
        self,
        leg1_length: float,
        leg2_length: float,
        width: float,
        thickness: float,
        fillet_radius: float = 0.0,
        hole1_diameters: tuple[float, ...] = (),
        hole1_distances: tuple[float, ...] = (),
        hole2_diameters: tuple[float, ...] = (),
        hole2_distances: tuple[float, ...] = (),
    ):
        """
        Initialize variable topology L-bracket.

        Args:
            leg1_length: Length of leg 1 along +X (mm)
            leg2_length: Length of leg 2 along +Z (mm)
            width: Width along Y axis (mm)
            thickness: Material thickness (mm)
            fillet_radius: Corner fillet radius (0 = no fillet) (mm)
            hole1_diameters: Tuple of hole diameters in leg 1 (empty = no holes)
            hole1_distances: Tuple of hole distances from end of leg 1
            hole2_diameters: Tuple of hole diameters in leg 2
            hole2_distances: Tuple of hole distances from end of leg 2
        """
        self.leg1_length = leg1_length
        self.leg2_length = leg2_length
        self.width = width
        self.thickness = thickness
        self.fillet_radius = fillet_radius
        self.hole1_diameters = tuple(hole1_diameters)
        self.hole1_distances = tuple(hole1_distances)
        self.hole2_diameters = tuple(hole2_diameters)
        self.hole2_distances = tuple(hole2_distances)

        self._validate()

    @property
    def num_holes_leg1(self) -> int:
        """Number of holes in leg 1."""
        return len(self.hole1_diameters)

    @property
    def num_holes_leg2(self) -> int:
        """Number of holes in leg 2."""
        return len(self.hole2_diameters)

    @property
    def has_fillet(self) -> bool:
        """Whether the bracket has a corner fillet."""
        return self.fillet_radius > 0

    @property
    def expected_num_faces(self) -> int:
        """Expected number of faces based on topology."""
        # Base L-bracket: 6 planar faces
        # Each hole adds 2 cylindrical faces (inner surface of hole)
        # Fillet adds 1 torus face
        base = 6
        holes = 2 * (self.num_holes_leg1 + self.num_holes_leg2)
        fillet = 1 if self.has_fillet else 0
        return base + holes + fillet

    def _validate(self) -> None:
        """Validate geometry constraints. Raises ValueError if invalid."""
        errors = []

        # Basic positivity for core dimensions
        if self.leg1_length <= 0:
            errors.append(f"leg1_length must be positive, got {self.leg1_length}")
        if self.leg2_length <= 0:
            errors.append(f"leg2_length must be positive, got {self.leg2_length}")
        if self.width <= 0:
            errors.append(f"width must be positive, got {self.width}")
        if self.thickness <= 0:
            errors.append(f"thickness must be positive, got {self.thickness}")
        if self.fillet_radius < 0:
            errors.append(f"fillet_radius must be non-negative, got {self.fillet_radius}")

        # Legs must be longer than thickness
        if self.leg1_length <= self.thickness:
            errors.append(
                f"leg1_length ({self.leg1_length}) must be > thickness ({self.thickness})"
            )
        if self.leg2_length <= self.thickness:
            errors.append(
                f"leg2_length ({self.leg2_length}) must be > thickness ({self.thickness})"
            )

        # Fillet radius constraints
        if self.fillet_radius > 0:
            max_fillet = min(self.thickness, self.leg1_length - self.thickness,
                            self.leg2_length - self.thickness) * 0.9
            if self.fillet_radius > max_fillet:
                errors.append(
                    f"fillet_radius ({self.fillet_radius}) too large for geometry "
                    f"(max ~{max_fillet:.1f})"
                )

        # Hole count consistency
        if len(self.hole1_diameters) != len(self.hole1_distances):
            errors.append(
                f"hole1_diameters ({len(self.hole1_diameters)}) and "
                f"hole1_distances ({len(self.hole1_distances)}) must have same length"
            )
        if len(self.hole2_diameters) != len(self.hole2_distances):
            errors.append(
                f"hole2_diameters ({len(self.hole2_diameters)}) and "
                f"hole2_distances ({len(self.hole2_distances)}) must have same length"
            )

        # Minimum wall thickness between fillet and hole edge
        min_fillet_hole_wall = 2.0  # mm

        # Validate each hole in leg 1
        for i, (diam, dist) in enumerate(zip(self.hole1_diameters, self.hole1_distances)):
            if diam <= 0:
                errors.append(f"hole1[{i}] diameter must be positive, got {diam}")
            if dist <= 0:
                errors.append(f"hole1[{i}] distance must be positive, got {dist}")
            if dist < diam:
                errors.append(
                    f"hole1[{i}] distance ({dist}) must be >= diameter ({diam})"
                )
            # Max distance accounts for fillet if present
            # Hole center at X = leg1_length - dist
            # Hole edge closest to corner at X = leg1_length - dist - diam/2
            # Must be >= thickness + fillet_radius + min_wall
            fillet_clearance = self.fillet_radius + min_fillet_hole_wall if self.fillet_radius > 0 else 0
            max_dist = self.leg1_length - self.thickness - diam / 2 - fillet_clearance
            if dist > max_dist:
                if self.fillet_radius > 0:
                    errors.append(
                        f"hole1[{i}] distance ({dist:.2f}) must be <= {max_dist:.2f} "
                        f"(accounting for fillet_radius={self.fillet_radius:.1f} + {min_fillet_hole_wall}mm wall)"
                    )
                else:
                    errors.append(
                        f"hole1[{i}] distance ({dist}) must be <= {max_dist:.2f}"
                    )
            if self.width < 2 * diam:
                errors.append(
                    f"width ({self.width}) must be >= 2 × hole1[{i}] diameter ({2*diam})"
                )

        # Validate each hole in leg 2
        for i, (diam, dist) in enumerate(zip(self.hole2_diameters, self.hole2_distances)):
            if diam <= 0:
                errors.append(f"hole2[{i}] diameter must be positive, got {diam}")
            if dist <= 0:
                errors.append(f"hole2[{i}] distance must be positive, got {dist}")
            if dist < diam:
                errors.append(
                    f"hole2[{i}] distance ({dist}) must be >= diameter ({diam})"
                )
            # Max distance accounts for fillet if present
            fillet_clearance = self.fillet_radius + min_fillet_hole_wall if self.fillet_radius > 0 else 0
            max_dist = self.leg2_length - self.thickness - diam / 2 - fillet_clearance
            if dist > max_dist:
                if self.fillet_radius > 0:
                    errors.append(
                        f"hole2[{i}] distance ({dist:.2f}) must be <= {max_dist:.2f} "
                        f"(accounting for fillet_radius={self.fillet_radius:.1f} + {min_fillet_hole_wall}mm wall)"
                    )
                else:
                    errors.append(
                        f"hole2[{i}] distance ({dist}) must be <= {max_dist:.2f}"
                    )
            if self.width < 2 * diam:
                errors.append(
                    f"width ({self.width}) must be >= 2 × hole2[{i}] diameter ({2*diam})"
                )

        # Check hole spacing (holes must not overlap)
        if len(self.hole1_distances) >= 2:
            sorted_holes = sorted(zip(self.hole1_distances, self.hole1_diameters))
            for i in range(len(sorted_holes) - 1):
                dist1, diam1 = sorted_holes[i]
                dist2, diam2 = sorted_holes[i + 1]
                min_spacing = (diam1 + diam2) / 2 + 2  # 2mm min wall between holes
                if abs(dist2 - dist1) < min_spacing:
                    errors.append(
                        f"holes in leg1 too close: spacing {abs(dist2-dist1):.1f} < {min_spacing:.1f}"
                    )

        if len(self.hole2_distances) >= 2:
            sorted_holes = sorted(zip(self.hole2_distances, self.hole2_diameters))
            for i in range(len(sorted_holes) - 1):
                dist1, diam1 = sorted_holes[i]
                dist2, diam2 = sorted_holes[i + 1]
                min_spacing = (diam1 + diam2) / 2 + 2
                if abs(dist2 - dist1) < min_spacing:
                    errors.append(
                        f"holes in leg2 too close: spacing {abs(dist2-dist1):.1f} < {min_spacing:.1f}"
                    )

        if errors:
            raise ValueError("Invalid VariableLBracket geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """
        Generate CadQuery solid geometry.

        Returns:
            CadQuery Workplane containing the L-bracket solid.
        """
        # Build L-shape profile on X-Z plane
        profile_points = [
            (0, 0),
            (self.leg1_length, 0),
            (self.leg1_length, self.thickness),
            (self.thickness, self.thickness),
            (self.thickness, self.leg2_length),
            (0, self.leg2_length),
        ]

        # Create L-bracket body by extruding profile along Y
        bracket = (
            cq.Workplane("XZ")
            .polyline(profile_points)
            .close()
            .extrude(self.width)
        )

        # Add fillet to inner corner if specified
        if self.fillet_radius > 0:
            # The inner corner edge runs parallel to Y axis
            # It's at position (thickness, thickness) in X-Z plane
            try:
                bracket = (
                    bracket
                    .edges("|Y")
                    .edges(
                        cq.selectors.NearestToPointSelector(
                            (self.thickness, self.width / 2, self.thickness)
                        )
                    )
                    .fillet(self.fillet_radius)
                )
            except Exception:
                # Fillet may fail for edge cases; skip silently
                pass

        # Add holes to leg 1 (through top face, parallel to Z)
        for i, (diam, dist) in enumerate(zip(self.hole1_diameters, self.hole1_distances)):
            face1_center_x = (self.thickness + self.leg1_length) / 2
            hole_global_x = self.leg1_length - dist
            wp_offset_x = hole_global_x - face1_center_x

            bracket = (
                bracket
                .faces(">Z[-2]")
                .workplane(centerOption="CenterOfBoundBox")
                .moveTo(wp_offset_x, 0)
                .hole(diam, self.thickness)
            )

        # Add holes to leg 2 (through inner face, parallel to X)
        for i, (diam, dist) in enumerate(zip(self.hole2_diameters, self.hole2_distances)):
            face2_center_z = (self.thickness + self.leg2_length) / 2
            hole_global_z = self.leg2_length - dist
            wp_offset_z = hole_global_z - face2_center_z

            bracket = (
                bracket
                .faces(">X[-2]")
                .workplane(centerOption="CenterOfBoundBox")
                .moveTo(0, wp_offset_z)
                .hole(diam, self.thickness)
            )

        return bracket

    def to_step(self, path: str | Path) -> None:
        """Export L-bracket to STEP file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict:
        """Return parameters as dictionary."""
        return {
            "leg1_length": self.leg1_length,
            "leg2_length": self.leg2_length,
            "width": self.width,
            "thickness": self.thickness,
            "fillet_radius": self.fillet_radius,
            "hole1_diameters": list(self.hole1_diameters),
            "hole1_distances": list(self.hole1_distances),
            "hole2_diameters": list(self.hole2_diameters),
            "hole2_distances": list(self.hole2_distances),
            "num_holes_leg1": self.num_holes_leg1,
            "num_holes_leg2": self.num_holes_leg2,
            "has_fillet": self.has_fillet,
            "expected_num_faces": self.expected_num_faces,
        }

    @classmethod
    def from_dict(cls, params: dict) -> VariableLBracket:
        """Create VariableLBracket from parameter dictionary."""
        return cls(
            leg1_length=params["leg1_length"],
            leg2_length=params["leg2_length"],
            width=params["width"],
            thickness=params["thickness"],
            fillet_radius=params.get("fillet_radius", 0.0),
            hole1_diameters=tuple(params.get("hole1_diameters", [])),
            hole1_distances=tuple(params.get("hole1_distances", [])),
            hole2_diameters=tuple(params.get("hole2_diameters", [])),
            hole2_distances=tuple(params.get("hole2_distances", [])),
        )

    @classmethod
    def random(
        cls,
        rng: Generator,
        ranges: VariableLBracketRanges | None = None,
    ) -> VariableLBracket:
        """
        Generate random variable topology L-bracket with valid geometry.

        Args:
            rng: NumPy random generator for reproducibility.
            ranges: Parameter ranges. Uses defaults if None.

        Returns:
            VariableLBracket instance with randomly sampled parameters.
        """
        if ranges is None:
            ranges = VariableLBracketRanges()

        # Sample core geometry
        leg1_length = rng.uniform(*ranges.leg1_length)
        leg2_length = rng.uniform(*ranges.leg2_length)
        thickness = rng.uniform(*ranges.thickness)
        width = rng.uniform(*ranges.width)

        # Decide on fillet
        has_fillet = rng.random() < ranges.prob_fillet
        if has_fillet:
            max_fillet = min(
                thickness,
                leg1_length - thickness,
                leg2_length - thickness,
                ranges.fillet_radius[1]
            ) * 0.8  # Leave some margin
            min_fillet = max(1.0, ranges.fillet_radius[0])  # At least 1mm if having fillet
            if max_fillet > min_fillet:
                fillet_radius = rng.uniform(min_fillet, max_fillet)
            else:
                fillet_radius = 0.0
                has_fillet = False
        else:
            fillet_radius = 0.0

        # Decide on number of holes per leg
        prob_config = np.array(ranges.prob_hole_configs)
        prob_config = prob_config / prob_config.sum()  # Normalize

        num_holes_leg1 = rng.choice([0, 1, 2], p=prob_config)
        num_holes_leg2 = rng.choice([0, 1, 2], p=prob_config)

        # Minimum wall thickness between fillet and hole edge
        min_fillet_hole_wall = 2.0  # mm (must match _validate)

        # Generate holes for leg 1
        hole1_diameters = []
        hole1_distances = []
        if num_holes_leg1 > 0:
            # Available length for holes (accounting for fillet clearance)
            fillet_clearance = fillet_radius + min_fillet_hole_wall if fillet_radius > 0 else 0
            available_length = leg1_length - thickness - fillet_clearance

            for _ in range(num_holes_leg1):
                diam = rng.uniform(*ranges.hole_diameter)
                # Ensure width can accommodate
                diam = min(diam, width / 2 - 1)
                hole1_diameters.append(diam)

            # Place holes with spacing
            if num_holes_leg1 == 1:
                diam = hole1_diameters[0]
                min_dist = diam
                max_dist = available_length - diam / 2  # Account for hole radius near corner
                if max_dist > min_dist:
                    hole1_distances.append(rng.uniform(min_dist, max_dist))
                else:
                    hole1_distances.append((min_dist + max_dist) / 2)
            else:
                # Two holes - ensure they don't overlap
                d1, d2 = hole1_diameters
                min_spacing = (d1 + d2) / 2 + 3  # 3mm min wall
                total_space = available_length - d1 / 2 - d2 / 2  # Account for hole radii

                if total_space > min_spacing:
                    # First hole near end
                    dist1 = rng.uniform(d1, d1 + (total_space - min_spacing) / 2)
                    # Second hole further in (but not into fillet zone)
                    dist2 = rng.uniform(dist1 + min_spacing, available_length - d2 / 2)
                    hole1_distances = [dist1, dist2]
                else:
                    # Not enough space for 2 holes, fall back to 1
                    hole1_diameters = [hole1_diameters[0]]
                    hole1_distances = [available_length / 2]

        # Generate holes for leg 2 (similar logic)
        hole2_diameters = []
        hole2_distances = []
        if num_holes_leg2 > 0:
            fillet_clearance = fillet_radius + min_fillet_hole_wall if fillet_radius > 0 else 0
            available_length = leg2_length - thickness - fillet_clearance

            for _ in range(num_holes_leg2):
                diam = rng.uniform(*ranges.hole_diameter)
                diam = min(diam, width / 2 - 1)
                hole2_diameters.append(diam)

            if num_holes_leg2 == 1:
                diam = hole2_diameters[0]
                min_dist = diam
                max_dist = available_length - diam / 2
                if max_dist > min_dist:
                    hole2_distances.append(rng.uniform(min_dist, max_dist))
                else:
                    hole2_distances.append((min_dist + max_dist) / 2)
            else:
                d1, d2 = hole2_diameters
                min_spacing = (d1 + d2) / 2 + 3
                total_space = available_length - d1 / 2 - d2 / 2

                if total_space > min_spacing:
                    dist1 = rng.uniform(d1, d1 + (total_space - min_spacing) / 2)
                    dist2 = rng.uniform(dist1 + min_spacing, available_length - d2 / 2)
                    hole2_distances = [dist1, dist2]
                else:
                    hole2_diameters = [hole2_diameters[0]]
                    hole2_distances = [available_length / 2]

        # Ensure width accommodates all holes
        if hole1_diameters or hole2_diameters:
            max_hole_diam = max(
                max(hole1_diameters) if hole1_diameters else 0,
                max(hole2_diameters) if hole2_diameters else 0,
            )
            min_width = 2 * max_hole_diam + 2
            if width < min_width:
                width = min(min_width, ranges.width[1])

        return cls(
            leg1_length=leg1_length,
            leg2_length=leg2_length,
            width=width,
            thickness=thickness,
            fillet_radius=fillet_radius,
            hole1_diameters=tuple(hole1_diameters),
            hole1_distances=tuple(hole1_distances),
            hole2_diameters=tuple(hole2_diameters),
            hole2_distances=tuple(hole2_distances),
        )

    @classmethod
    def from_fixed_lbracket(cls, bracket: LBracket) -> VariableLBracket:
        """Convert a fixed-topology LBracket to VariableLBracket."""
        return cls(
            leg1_length=bracket.leg1_length,
            leg2_length=bracket.leg2_length,
            width=bracket.width,
            thickness=bracket.thickness,
            fillet_radius=0.0,
            hole1_diameters=(bracket.hole1_diameter,),
            hole1_distances=(bracket.hole1_distance,),
            hole2_diameters=(bracket.hole2_diameter,),
            hole2_distances=(bracket.hole2_distance,),
        )

    def __repr__(self) -> str:
        holes_leg1 = f"{self.num_holes_leg1} holes" if self.num_holes_leg1 != 1 else "1 hole"
        holes_leg2 = f"{self.num_holes_leg2} holes" if self.num_holes_leg2 != 1 else "1 hole"
        fillet_str = f", fillet={self.fillet_radius:.1f}" if self.has_fillet else ""
        return (
            f"VariableLBracket("
            f"legs={self.leg1_length:.1f}×{self.leg2_length:.1f}, "
            f"width={self.width:.1f}, thickness={self.thickness:.1f}, "
            f"leg1: {holes_leg1}, leg2: {holes_leg2}"
            f"{fillet_str}, "
            f"faces={self.expected_num_faces})"
        )
