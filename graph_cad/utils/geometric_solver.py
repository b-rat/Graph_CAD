"""
Geometric Solver: Deterministic parameter extraction from graph features.

Extracts L-bracket construction parameters directly from decoded graph features
using geometric relationships. This bypasses learned regression entirely.

The key insight is that face areas, centroids, normals, and edge lengths
encode all the information needed to reconstruct parameters:
- Core params: From planar face positions and areas
- Hole params: From hole face areas (πdh) and centroids
- Fillet radius: From fillet face area (πr/2 × width for 90° fillet)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Face type codes (must match graph_extraction.py)
FACE_TYPE_PLANAR = 0
FACE_TYPE_HOLE = 1
FACE_TYPE_FILLET = 2

# Reference scale for bbox_diagonal normalization (must match graph_extraction.py)
BBOX_REFERENCE_SCALE = 100.0


@dataclass
class SolvedParams:
    """L-bracket parameters extracted by geometric solver."""

    # Core parameters (always present)
    leg1_length: float
    leg2_length: float
    width: float
    thickness: float

    # Optional features
    fillet_radius: float = 0.0
    hole1_diameters: tuple[float, ...] = ()
    hole1_distances: tuple[float, ...] = ()
    hole2_diameters: tuple[float, ...] = ()
    hole2_distances: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        """Convert to dictionary for VariableLBracket construction."""
        result = {
            "leg1_length": self.leg1_length,
            "leg2_length": self.leg2_length,
            "width": self.width,
            "thickness": self.thickness,
        }
        if self.fillet_radius > 0:
            result["fillet_radius"] = self.fillet_radius
        if self.hole1_diameters:
            result["hole1_diameters"] = self.hole1_diameters
            result["hole1_distances"] = self.hole1_distances
        if self.hole2_diameters:
            result["hole2_diameters"] = self.hole2_diameters
            result["hole2_distances"] = self.hole2_distances
        return result


def solve_params_from_features(
    node_features: NDArray[np.float32],
    face_types: NDArray[np.int64],
    edge_index: NDArray[np.int64],
    edge_features: NDArray[np.float32],
    node_mask: NDArray[np.float32] | None = None,
) -> SolvedParams:
    """
    Extract L-bracket parameters from decoded graph features.

    Args:
        node_features: Node feature matrix, shape (num_nodes, 13).
            Columns: [area, dir_x, dir_y, dir_z, cx, cy, cz, curv1, curv2,
                      bbox_diagonal, bbox_center_x, bbox_center_y, bbox_center_z]
        face_types: Face type indices, shape (num_nodes,).
            Values: 0=PLANAR, 1=HOLE, 2=FILLET
        edge_index: Edge indices in COO format, shape (2, num_edges).
        edge_features: Edge feature matrix, shape (num_edges, 2).
            Columns: [edge_length, dihedral_angle]
        node_mask: Optional mask for valid nodes (for padded batches).

    Returns:
        SolvedParams with extracted L-bracket parameters.

    Note:
        All input features are normalized by 100mm reference scale.
        This function de-normalizes using bbox_diagonal and bbox_center
        stored in node_features[:, 9:13].
    """
    # Apply mask if provided
    if node_mask is not None:
        valid_mask = node_mask > 0.5
        node_features = node_features[valid_mask]
        face_types = face_types[valid_mask]

    num_nodes = len(node_features)
    if num_nodes == 0:
        raise ValueError("No valid nodes in graph")

    # Extract bbox info for de-normalization (same for all nodes)
    bbox_diagonal_normalized = node_features[0, 9]
    bbox_center_normalized = node_features[0, 10:13]
    bbox_diagonal = bbox_diagonal_normalized * BBOX_REFERENCE_SCALE
    bbox_center = bbox_center_normalized * BBOX_REFERENCE_SCALE

    # De-normalize features
    areas = node_features[:, 0] * (bbox_diagonal ** 2)  # area scales with length^2
    normals = node_features[:, 1:4]  # unit vectors, no scaling needed
    # Centroids: stored as (centroid - bbox_center) / bbox_diagonal
    # Recover: centroid = normalized * bbox_diagonal + bbox_center
    centroids = node_features[:, 4:7] * bbox_diagonal + bbox_center

    # Separate faces by type
    planar_mask = face_types == FACE_TYPE_PLANAR
    hole_mask = face_types == FACE_TYPE_HOLE
    fillet_mask = face_types == FACE_TYPE_FILLET

    planar_indices = np.where(planar_mask)[0]
    hole_indices = np.where(hole_mask)[0]
    fillet_indices = np.where(fillet_mask)[0]

    # Solve core parameters from planar faces
    core_params = _solve_core_params(
        areas[planar_mask],
        normals[planar_mask],
        centroids[planar_mask],
        bbox_diagonal,
    )

    # Solve hole parameters (need leg lengths to compute distances from end)
    hole1_params, hole2_params = _solve_hole_params(
        areas[hole_mask],
        normals[hole_mask],
        centroids[hole_mask],
        core_params["thickness"],
        core_params["width"],
        core_params["leg1_length"],
        core_params["leg2_length"],
    )

    # Solve fillet radius
    fillet_radius = _solve_fillet_radius(
        areas[fillet_mask],
        core_params["width"],
    )

    return SolvedParams(
        leg1_length=core_params["leg1_length"],
        leg2_length=core_params["leg2_length"],
        width=core_params["width"],
        thickness=core_params["thickness"],
        fillet_radius=fillet_radius,
        hole1_diameters=hole1_params["diameters"],
        hole1_distances=hole1_params["distances"],
        hole2_diameters=hole2_params["diameters"],
        hole2_distances=hole2_params["distances"],
    )


def _solve_core_params(
    areas: NDArray[np.float32],
    normals: NDArray[np.float32],
    centroids: NDArray[np.float32],
    bbox_diagonal: float,
) -> dict:
    """
    Solve core L-bracket parameters from planar faces.

    L-bracket geometry in standard orientation:
    - Leg1 extends along +X axis
    - Leg2 extends along +Z axis
    - Width is along Y axis
    - Thickness is the cross-section dimension

    Face identification by normal direction:
    - Y-facing faces (±Y normal): front/back of L-shape
    - X-facing faces (±X normal): end faces, inner step
    - Z-facing faces (±Z normal): end faces, inner step
    """
    if len(areas) == 0:
        return {
            "leg1_length": 100.0,
            "leg2_length": 80.0,
            "width": 30.0,
            "thickness": 10.0,
        }

    # Find faces by their normal directions
    y_facing = np.abs(normals[:, 1]) > 0.9  # Front/back faces
    x_facing = np.abs(normals[:, 0]) > 0.9  # Leg1 end faces
    z_facing = np.abs(normals[:, 2]) > 0.9  # Leg2 end faces

    # Width: from Y-facing face centroids (distance between front and back)
    if np.any(y_facing):
        y_centroids = centroids[y_facing, 1]
        width = np.max(y_centroids) - np.min(y_centroids)
        if width < 1.0:
            width = 30.0
    else:
        width = 30.0

    # Leg lengths: from centroid extents
    # Leg1 along X: max X centroid position
    # Leg2 along Z: max Z centroid position
    max_x = np.max(centroids[:, 0])
    max_z = np.max(centroids[:, 2])

    # For a face at the end of the leg, centroid is at leg_length - thickness/2
    # But we also have inner step faces, so we need to be careful

    # Use X-facing faces to find leg1 extent
    if np.any(x_facing):
        x_face_centroids = centroids[x_facing, 0]
        leg1_length = np.max(x_face_centroids)  # End face is at X = leg1_length
    else:
        leg1_length = max_x

    # Use Z-facing faces to find leg2 extent
    if np.any(z_facing):
        z_face_centroids = centroids[z_facing, 2]
        leg2_length = np.max(z_face_centroids)  # End face is at Z = leg2_length
    else:
        leg2_length = max_z

    # Thickness: from the smallest X or Z facing face (inner step)
    # Inner step face area = width × thickness
    xz_facing = x_facing | z_facing
    if np.any(xz_facing):
        xz_areas = areas[xz_facing]
        min_area = np.min(xz_areas)
        thickness = min_area / max(width, 1.0)
        thickness = np.clip(thickness, 5.0, 30.0)
    else:
        thickness = 10.0

    # Validate with the largest X-facing face (should be leg2 × width)
    if np.any(x_facing):
        x_areas = areas[x_facing]
        largest_x_area = np.max(x_areas)
        # This face has area = leg2_length × width (outer face at X=0)
        leg2_from_area = largest_x_area / max(width, 1.0)
        # Use area-based estimate if it's larger (centroid might be inner step)
        if leg2_from_area > leg2_length:
            leg2_length = leg2_from_area

    # Similarly for Z-facing faces
    if np.any(z_facing):
        z_areas = areas[z_facing]
        largest_z_area = np.max(z_areas)
        leg1_from_area = largest_z_area / max(width, 1.0)
        if leg1_from_area > leg1_length:
            leg1_length = leg1_from_area

    return {
        "leg1_length": float(np.clip(leg1_length, 50.0, 200.0)),
        "leg2_length": float(np.clip(leg2_length, 50.0, 200.0)),
        "width": float(np.clip(width, 20.0, 60.0)),
        "thickness": float(np.clip(thickness, 5.0, 30.0)),
    }


def _solve_hole_params(
    areas: NDArray[np.float32],
    normals: NDArray[np.float32],
    centroids: NDArray[np.float32],
    thickness: float,
    width: float,
    leg1_length: float,
    leg2_length: float,
) -> tuple[dict, dict]:
    """
    Solve hole parameters from cylindrical (hole) faces.

    Hole face properties:
    - Area = π × diameter × thickness (cylindrical surface through the bracket)
    - The hole axis direction tells us which leg it's on
    - Distance is measured from the END of the leg (not from origin)

    L-bracket orientation:
    - Leg1 along +X, leg2 along +Z, width along Y
    - Holes on leg1: axis along Z, distance = leg1_length - centroid_x
    - Holes on leg2: axis along X, distance = leg2_length - centroid_z

    Returns:
        Tuple of (hole1_params, hole2_params) dicts with 'diameters' and 'distances'.
    """
    hole1 = {"diameters": (), "distances": ()}
    hole2 = {"diameters": (), "distances": ()}

    if len(areas) == 0:
        return hole1, hole2

    hole1_diameters = []
    hole1_distances = []
    hole2_diameters = []
    hole2_distances = []

    for i in range(len(areas)):
        area = areas[i]
        axis = normals[i]  # For cylinders, this is the axis direction
        cx, cy, cz = centroids[i]

        # Diameter from cylindrical surface area: A = π × d × h
        # where h = thickness (hole goes through the bracket thickness)
        diameter = area / (np.pi * max(thickness, 1.0))
        diameter = np.clip(diameter, 4.0, 20.0)

        # Determine which leg this hole is on based on axis direction
        # Holes on leg1: axis is along Z (perpendicular to leg1 which is along X)
        # Holes on leg2: axis is along X (perpendicular to leg2 which is along Z)

        if np.abs(axis[2]) > 0.9:  # Z-axis hole -> on leg1
            hole1_diameters.append(float(diameter))
            # Distance from end: leg1_length - centroid_x
            distance = leg1_length - cx
            hole1_distances.append(float(max(distance, diameter)))
        elif np.abs(axis[0]) > 0.9:  # X-axis hole -> on leg2
            hole2_diameters.append(float(diameter))
            # Distance from end: leg2_length - centroid_z
            distance = leg2_length - cz
            hole2_distances.append(float(max(distance, diameter)))

    # Sort by distance (closest to end first)
    if hole1_diameters:
        sorted_idx = np.argsort(hole1_distances)
        hole1["diameters"] = tuple(hole1_diameters[i] for i in sorted_idx)
        hole1["distances"] = tuple(hole1_distances[i] for i in sorted_idx)

    if hole2_diameters:
        sorted_idx = np.argsort(hole2_distances)
        hole2["diameters"] = tuple(hole2_diameters[i] for i in sorted_idx)
        hole2["distances"] = tuple(hole2_distances[i] for i in sorted_idx)

    return hole1, hole2


def _solve_fillet_radius(
    areas: NDArray[np.float32],
    width: float,
) -> float:
    """
    Solve fillet radius from fillet (partial cylinder/torus) faces.

    For a 90° fillet at the inner corner:
    - Cylindrical fillet area = (π/2) × radius × width
    - (Quarter cylinder surface)

    Returns:
        Fillet radius, or 0.0 if no fillet.
    """
    if len(areas) == 0:
        return 0.0

    # Sum all fillet areas (might be multiple fillet faces)
    total_area = np.sum(areas)

    # For 90° fillet: A = (π/2) × r × w
    # So: r = 2A / (π × w)
    radius = (2.0 * total_area) / (np.pi * max(width, 1.0))

    # Clip to reasonable bounds
    radius = np.clip(radius, 0.0, 10.0)

    # If radius is very small, treat as no fillet
    if radius < 0.5:
        return 0.0

    return float(radius)


def solve_params_from_decoded_batch(
    node_features: NDArray[np.float32],
    face_types: NDArray[np.int64],
    edge_index: NDArray[np.int64],
    edge_features: NDArray[np.float32],
    node_mask: NDArray[np.float32],
    batch_size: int,
    max_nodes: int,
) -> list[SolvedParams]:
    """
    Solve parameters for a batch of decoded graphs.

    Args:
        node_features: Batched node features, shape (batch_size, max_nodes, 10).
        face_types: Batched face types, shape (batch_size, max_nodes).
        edge_index: Batched edge indices, shape (batch_size, 2, max_edges).
        edge_features: Batched edge features, shape (batch_size, max_edges, 2).
        node_mask: Batched node masks, shape (batch_size, max_nodes).
        batch_size: Number of samples in batch.
        max_nodes: Maximum nodes per graph.

    Returns:
        List of SolvedParams, one per sample.
    """
    results = []

    for i in range(batch_size):
        params = solve_params_from_features(
            node_features=node_features[i],
            face_types=face_types[i],
            edge_index=edge_index[i] if edge_index is not None else np.zeros((2, 0), dtype=np.int64),
            edge_features=edge_features[i] if edge_features is not None else np.zeros((0, 2), dtype=np.float32),
            node_mask=node_mask[i],
        )
        results.append(params)

    return results
