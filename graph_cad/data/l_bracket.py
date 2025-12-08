"""
L-Bracket CAD generator for synthetic training data.

Generates parametric L-brackets with two through-holes as STEP files.
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
