"""
Parametric geometry generators for Phase 4 multi-geometry dataset.

Each generator follows the same pattern as LBracket:
- __init__: Store parameters and validate
- to_solid(): Generate CadQuery geometry
- to_step(): Export to STEP file
- to_dict(): Return parameters as dictionary
- from_dict(): Create from dictionary
- random(): Generate random valid geometry

Geometry Types:
- Tube: Hollow cylinder (pipe segment)
- Channel: C-channel structural member
- Block: Simple rectangular solid
- Cylinder: Solid cylinder
- BlockHole: Block with through-hole
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cadquery as cq
import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


# =============================================================================
# Tube (Hollow Cylinder)
# =============================================================================


@dataclass
class TubeRanges:
    """Parameter ranges for Tube generation."""

    length: tuple[float, float] = (30.0, 200.0)
    outer_dia: tuple[float, float] = (20.0, 100.0)
    inner_dia_ratio: tuple[float, float] = (0.3, 0.9)  # inner/outer ratio


class Tube:
    """
    Parametric hollow cylinder (pipe segment).

    Coordinate system:
    - Origin at center of bottom face
    - Length extends along +Z axis
    - Outer/inner radii on X-Y plane

    Parameters:
        length: Length along Z axis (mm)
        outer_dia: Outer diameter (mm)
        inner_dia: Inner diameter (mm), must be < outer_dia
    """

    def __init__(
        self,
        length: float,
        outer_dia: float,
        inner_dia: float,
    ):
        self.length = length
        self.outer_dia = outer_dia
        self.inner_dia = inner_dia
        self._validate()

    def _validate(self) -> None:
        errors = []
        if self.length <= 0:
            errors.append(f"length must be positive, got {self.length}")
        if self.outer_dia <= 0:
            errors.append(f"outer_dia must be positive, got {self.outer_dia}")
        if self.inner_dia <= 0:
            errors.append(f"inner_dia must be positive, got {self.inner_dia}")
        if self.inner_dia >= self.outer_dia:
            errors.append(
                f"inner_dia ({self.inner_dia}) must be < outer_dia ({self.outer_dia})"
            )

        # Minimum wall thickness (1mm)
        wall_thickness = (self.outer_dia - self.inner_dia) / 2
        if wall_thickness < 1.0:
            errors.append(f"Wall thickness ({wall_thickness:.2f}) must be >= 1mm")

        if errors:
            raise ValueError("Invalid Tube geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """Generate CadQuery solid geometry."""
        outer_radius = self.outer_dia / 2
        inner_radius = self.inner_dia / 2

        # Create outer cylinder and subtract inner cylinder
        tube = (
            cq.Workplane("XY")
            .circle(outer_radius)
            .extrude(self.length)
            .faces(">Z")
            .workplane()
            .circle(inner_radius)
            .cutThruAll()
        )
        return tube

    def to_step(self, path: str | Path) -> None:
        """Export to STEP file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict[str, float]:
        """Return parameters as dictionary."""
        return {
            "length": self.length,
            "outer_dia": self.outer_dia,
            "inner_dia": self.inner_dia,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> Tube:
        """Create Tube from parameter dictionary."""
        return cls(
            length=params["length"],
            outer_dia=params["outer_dia"],
            inner_dia=params["inner_dia"],
        )

    @classmethod
    def random(cls, rng: Generator, ranges: TubeRanges | None = None) -> Tube:
        """Generate random Tube with valid geometry."""
        if ranges is None:
            ranges = TubeRanges()

        length = rng.uniform(*ranges.length)
        outer_dia = rng.uniform(*ranges.outer_dia)
        inner_ratio = rng.uniform(*ranges.inner_dia_ratio)
        inner_dia = outer_dia * inner_ratio

        # Ensure minimum wall thickness
        min_inner = outer_dia - 2 * outer_dia * 0.9  # Max inner for min 10% wall
        inner_dia = max(min_inner, min(inner_dia, outer_dia - 2.0))

        return cls(length=length, outer_dia=outer_dia, inner_dia=inner_dia)

    def __repr__(self) -> str:
        return (
            f"Tube(length={self.length:.1f}, "
            f"outer_dia={self.outer_dia:.1f}, inner_dia={self.inner_dia:.1f})"
        )


# =============================================================================
# Channel (C-Channel)
# =============================================================================


@dataclass
class ChannelRanges:
    """Parameter ranges for Channel generation."""

    width: tuple[float, float] = (30.0, 100.0)
    height: tuple[float, float] = (30.0, 100.0)
    length: tuple[float, float] = (50.0, 200.0)
    thickness: tuple[float, float] = (2.0, 10.0)


class Channel:
    """
    Parametric C-channel structural member.

    Coordinate system:
    - Origin at bottom-left-back corner
    - Width along X axis (web direction)
    - Height along Z axis (flange direction)
    - Length along Y axis (extrusion direction)

    Parameters:
        width: Web width along X (mm)
        height: Total height along Z (mm)
        length: Length along Y (mm)
        thickness: Wall thickness (mm)
    """

    def __init__(
        self,
        width: float,
        height: float,
        length: float,
        thickness: float,
    ):
        self.width = width
        self.height = height
        self.length = length
        self.thickness = thickness
        self._validate()

    def _validate(self) -> None:
        errors = []
        if self.width <= 0:
            errors.append(f"width must be positive, got {self.width}")
        if self.height <= 0:
            errors.append(f"height must be positive, got {self.height}")
        if self.length <= 0:
            errors.append(f"length must be positive, got {self.length}")
        if self.thickness <= 0:
            errors.append(f"thickness must be positive, got {self.thickness}")

        # Thickness constraints
        if self.thickness >= self.width / 2:
            errors.append(
                f"thickness ({self.thickness}) must be < width/2 ({self.width/2})"
            )
        if self.thickness >= self.height / 2:
            errors.append(
                f"thickness ({self.thickness}) must be < height/2 ({self.height/2})"
            )

        if errors:
            raise ValueError("Invalid Channel geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """Generate CadQuery solid geometry."""
        # C-channel profile: outer rectangle minus inner rectangle
        # Profile in X-Z plane, extruded along Y

        # Outer profile points (counterclockwise)
        outer = [
            (0, 0),
            (self.width, 0),
            (self.width, self.thickness),
            (self.thickness, self.thickness),
            (self.thickness, self.height - self.thickness),
            (self.width, self.height - self.thickness),
            (self.width, self.height),
            (0, self.height),
        ]

        channel = (
            cq.Workplane("XZ")
            .polyline(outer)
            .close()
            .extrude(self.length)
        )
        return channel

    def to_step(self, path: str | Path) -> None:
        """Export to STEP file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict[str, float]:
        """Return parameters as dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "length": self.length,
            "thickness": self.thickness,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> Channel:
        """Create Channel from parameter dictionary."""
        return cls(
            width=params["width"],
            height=params["height"],
            length=params["length"],
            thickness=params["thickness"],
        )

    @classmethod
    def random(cls, rng: Generator, ranges: ChannelRanges | None = None) -> Channel:
        """Generate random Channel with valid geometry."""
        if ranges is None:
            ranges = ChannelRanges()

        width = rng.uniform(*ranges.width)
        height = rng.uniform(*ranges.height)
        length = rng.uniform(*ranges.length)

        # Thickness must be < min(width, height) / 2
        max_thickness = min(width, height) / 2 - 1.0
        max_thickness = min(max_thickness, ranges.thickness[1])
        min_thickness = max(ranges.thickness[0], 1.0)
        thickness = rng.uniform(min_thickness, max_thickness)

        return cls(width=width, height=height, length=length, thickness=thickness)

    def __repr__(self) -> str:
        return (
            f"Channel(width={self.width:.1f}, height={self.height:.1f}, "
            f"length={self.length:.1f}, thickness={self.thickness:.1f})"
        )


# =============================================================================
# Block (Simple Rectangular Solid)
# =============================================================================


@dataclass
class BlockRanges:
    """Parameter ranges for Block generation."""

    length: tuple[float, float] = (20.0, 150.0)
    width: tuple[float, float] = (20.0, 150.0)
    height: tuple[float, float] = (10.0, 100.0)


class Block:
    """
    Simple rectangular solid (box).

    Coordinate system:
    - Origin at bottom-left-back corner
    - Length along X axis
    - Width along Y axis
    - Height along Z axis

    Parameters:
        length: Length along X (mm)
        width: Width along Y (mm)
        height: Height along Z (mm)
    """

    def __init__(
        self,
        length: float,
        width: float,
        height: float,
    ):
        self.length = length
        self.width = width
        self.height = height
        self._validate()

    def _validate(self) -> None:
        errors = []
        if self.length <= 0:
            errors.append(f"length must be positive, got {self.length}")
        if self.width <= 0:
            errors.append(f"width must be positive, got {self.width}")
        if self.height <= 0:
            errors.append(f"height must be positive, got {self.height}")

        if errors:
            raise ValueError("Invalid Block geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """Generate CadQuery solid geometry."""
        block = cq.Workplane("XY").box(self.length, self.width, self.height)
        return block

    def to_step(self, path: str | Path) -> None:
        """Export to STEP file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict[str, float]:
        """Return parameters as dictionary."""
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> Block:
        """Create Block from parameter dictionary."""
        return cls(
            length=params["length"],
            width=params["width"],
            height=params["height"],
        )

    @classmethod
    def random(cls, rng: Generator, ranges: BlockRanges | None = None) -> Block:
        """Generate random Block with valid geometry."""
        if ranges is None:
            ranges = BlockRanges()

        length = rng.uniform(*ranges.length)
        width = rng.uniform(*ranges.width)
        height = rng.uniform(*ranges.height)

        return cls(length=length, width=width, height=height)

    def __repr__(self) -> str:
        return (
            f"Block(length={self.length:.1f}, width={self.width:.1f}, "
            f"height={self.height:.1f})"
        )


# =============================================================================
# Cylinder (Solid Cylinder)
# =============================================================================


@dataclass
class CylinderRanges:
    """Parameter ranges for Cylinder generation."""

    length: tuple[float, float] = (20.0, 200.0)
    diameter: tuple[float, float] = (10.0, 100.0)


class Cylinder:
    """
    Solid cylinder.

    Coordinate system:
    - Origin at center of bottom face
    - Length extends along +Z axis
    - Diameter on X-Y plane

    Parameters:
        length: Length along Z (mm)
        diameter: Cylinder diameter (mm)
    """

    def __init__(
        self,
        length: float,
        diameter: float,
    ):
        self.length = length
        self.diameter = diameter
        self._validate()

    def _validate(self) -> None:
        errors = []
        if self.length <= 0:
            errors.append(f"length must be positive, got {self.length}")
        if self.diameter <= 0:
            errors.append(f"diameter must be positive, got {self.diameter}")

        if errors:
            raise ValueError("Invalid Cylinder geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """Generate CadQuery solid geometry."""
        cylinder = (
            cq.Workplane("XY")
            .circle(self.diameter / 2)
            .extrude(self.length)
        )
        return cylinder

    def to_step(self, path: str | Path) -> None:
        """Export to STEP file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict[str, float]:
        """Return parameters as dictionary."""
        return {
            "length": self.length,
            "diameter": self.diameter,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> Cylinder:
        """Create Cylinder from parameter dictionary."""
        return cls(
            length=params["length"],
            diameter=params["diameter"],
        )

    @classmethod
    def random(cls, rng: Generator, ranges: CylinderRanges | None = None) -> Cylinder:
        """Generate random Cylinder with valid geometry."""
        if ranges is None:
            ranges = CylinderRanges()

        length = rng.uniform(*ranges.length)
        diameter = rng.uniform(*ranges.diameter)

        return cls(length=length, diameter=diameter)

    def __repr__(self) -> str:
        return f"Cylinder(length={self.length:.1f}, diameter={self.diameter:.1f})"


# =============================================================================
# BlockHole (Block with Through-Hole)
# =============================================================================


@dataclass
class BlockHoleRanges:
    """Parameter ranges for BlockHole generation."""

    length: tuple[float, float] = (30.0, 150.0)
    width: tuple[float, float] = (30.0, 150.0)
    height: tuple[float, float] = (15.0, 80.0)
    hole_dia: tuple[float, float] = (5.0, 30.0)


class BlockHole:
    """
    Rectangular block with a vertical through-hole.

    Coordinate system:
    - Origin at center of block (CadQuery box default)
    - Length along X axis
    - Width along Y axis
    - Height along Z axis
    - Hole is vertical (parallel to Z), positioned at (hole_x, hole_y) from center

    Parameters:
        length: Length along X (mm)
        width: Width along Y (mm)
        height: Height along Z (mm)
        hole_dia: Hole diameter (mm)
        hole_x: Hole X position relative to block center (mm)
        hole_y: Hole Y position relative to block center (mm)
    """

    def __init__(
        self,
        length: float,
        width: float,
        height: float,
        hole_dia: float,
        hole_x: float,
        hole_y: float,
    ):
        self.length = length
        self.width = width
        self.height = height
        self.hole_dia = hole_dia
        self.hole_x = hole_x
        self.hole_y = hole_y
        self._validate()

    def _validate(self) -> None:
        errors = []
        if self.length <= 0:
            errors.append(f"length must be positive, got {self.length}")
        if self.width <= 0:
            errors.append(f"width must be positive, got {self.width}")
        if self.height <= 0:
            errors.append(f"height must be positive, got {self.height}")
        if self.hole_dia <= 0:
            errors.append(f"hole_dia must be positive, got {self.hole_dia}")

        # Hole must fit within block with edge clearance
        hole_radius = self.hole_dia / 2
        min_edge_clearance = 2.0  # mm

        x_limit = self.length / 2 - hole_radius - min_edge_clearance
        y_limit = self.width / 2 - hole_radius - min_edge_clearance

        if x_limit < 0:
            errors.append(
                f"hole_dia ({self.hole_dia}) too large for length ({self.length})"
            )
        elif abs(self.hole_x) > x_limit:
            errors.append(
                f"hole_x ({self.hole_x}) out of range [-{x_limit:.1f}, {x_limit:.1f}]"
            )

        if y_limit < 0:
            errors.append(
                f"hole_dia ({self.hole_dia}) too large for width ({self.width})"
            )
        elif abs(self.hole_y) > y_limit:
            errors.append(
                f"hole_y ({self.hole_y}) out of range [-{y_limit:.1f}, {y_limit:.1f}]"
            )

        if errors:
            raise ValueError("Invalid BlockHole geometry:\n  " + "\n  ".join(errors))

    def to_solid(self) -> cq.Workplane:
        """Generate CadQuery solid geometry."""
        block = (
            cq.Workplane("XY")
            .box(self.length, self.width, self.height)
            .faces(">Z")
            .workplane()
            .moveTo(self.hole_x, self.hole_y)
            .hole(self.hole_dia, self.height)
        )
        return block

    def to_step(self, path: str | Path) -> None:
        """Export to STEP file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        solid = self.to_solid()
        cq.exporters.export(solid, str(path), cq.exporters.ExportTypes.STEP)

    def to_dict(self) -> dict[str, float]:
        """Return parameters as dictionary."""
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "hole_dia": self.hole_dia,
            "hole_x": self.hole_x,
            "hole_y": self.hole_y,
        }

    @classmethod
    def from_dict(cls, params: dict[str, float]) -> BlockHole:
        """Create BlockHole from parameter dictionary."""
        return cls(
            length=params["length"],
            width=params["width"],
            height=params["height"],
            hole_dia=params["hole_dia"],
            hole_x=params["hole_x"],
            hole_y=params["hole_y"],
        )

    @classmethod
    def random(
        cls, rng: Generator, ranges: BlockHoleRanges | None = None
    ) -> BlockHole:
        """Generate random BlockHole with valid geometry."""
        if ranges is None:
            ranges = BlockHoleRanges()

        length = rng.uniform(*ranges.length)
        width = rng.uniform(*ranges.width)
        height = rng.uniform(*ranges.height)

        # Hole diameter constrained by block size
        max_hole_dia = min(length, width) - 8.0  # 4mm clearance each side
        max_hole_dia = min(max_hole_dia, ranges.hole_dia[1])
        min_hole_dia = max(ranges.hole_dia[0], 3.0)
        hole_dia = rng.uniform(min_hole_dia, max(max_hole_dia, min_hole_dia + 1.0))

        # Hole position constrained by block size and hole diameter
        hole_radius = hole_dia / 2
        min_edge_clearance = 2.0

        x_limit = max(0.0, length / 2 - hole_radius - min_edge_clearance)
        y_limit = max(0.0, width / 2 - hole_radius - min_edge_clearance)

        hole_x = rng.uniform(-x_limit, x_limit)
        hole_y = rng.uniform(-y_limit, y_limit)

        return cls(
            length=length,
            width=width,
            height=height,
            hole_dia=hole_dia,
            hole_x=hole_x,
            hole_y=hole_y,
        )

    def __repr__(self) -> str:
        return (
            f"BlockHole(length={self.length:.1f}, width={self.width:.1f}, "
            f"height={self.height:.1f}, hole_dia={self.hole_dia:.1f}, "
            f"hole_pos=({self.hole_x:.1f}, {self.hole_y:.1f}))"
        )


# =============================================================================
# Generator Registry
# =============================================================================


GEOMETRY_GENERATORS = {
    "bracket": None,  # Use VariableLBracket from l_bracket.py
    "tube": Tube,
    "channel": Channel,
    "block": Block,
    "cylinder": Cylinder,
    "blockhole": BlockHole,
}

GEOMETRY_RANGES = {
    "tube": TubeRanges,
    "channel": ChannelRanges,
    "block": BlockRanges,
    "cylinder": CylinderRanges,
    "blockhole": BlockHoleRanges,
}
