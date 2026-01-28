"""
B-Rep type constants for full topology graph extraction.

Defines edge types and face types for the HeteroGNN encoder.
These types enable richer geometric encoding compared to face-only graphs.
"""

from __future__ import annotations

# =============================================================================
# Edge Types (Geometric Classification)
# =============================================================================

EDGE_TYPE_LINE = 0      # Straight edge
EDGE_TYPE_ARC = 1       # Circular arc (partial circle)
EDGE_TYPE_CIRCLE = 2    # Full circle (closed curve)
EDGE_TYPE_OTHER = 3     # B-spline, ellipse, etc.

NUM_EDGE_TYPES = 4

EDGE_TYPE_NAMES = {
    EDGE_TYPE_LINE: "LINE",
    EDGE_TYPE_ARC: "ARC",
    EDGE_TYPE_CIRCLE: "CIRCLE",
    EDGE_TYPE_OTHER: "OTHER",
}


# =============================================================================
# Face Types (Surface Classification)
# =============================================================================

# Matches existing codes from graph_extraction.py for compatibility
FACE_TYPE_PLANAR = 0    # Flat faces
FACE_TYPE_HOLE = 1      # Cylindrical faces with arc >= 180 degrees (holes)
FACE_TYPE_FILLET = 2    # Cylindrical faces with arc < 180 degrees, or torus

NUM_FACE_TYPES = 3

FACE_TYPE_NAMES = {
    FACE_TYPE_PLANAR: "PLANAR",
    FACE_TYPE_HOLE: "HOLE",
    FACE_TYPE_FILLET: "FILLET",
}


# =============================================================================
# Geometry Type IDs (for Multi-Geometry Dataset)
# =============================================================================

GEOMETRY_BRACKET = 0
GEOMETRY_TUBE = 1
GEOMETRY_CHANNEL = 2
GEOMETRY_BLOCK = 3
GEOMETRY_CYLINDER = 4
GEOMETRY_BLOCKHOLE = 5

NUM_GEOMETRY_TYPES = 6

GEOMETRY_TYPE_NAMES = {
    GEOMETRY_BRACKET: "bracket",
    GEOMETRY_TUBE: "tube",
    GEOMETRY_CHANNEL: "channel",
    GEOMETRY_BLOCK: "block",
    GEOMETRY_CYLINDER: "cylinder",
    GEOMETRY_BLOCKHOLE: "blockhole",
}

# Reverse mapping: name -> ID
GEOMETRY_NAME_TO_ID = {v: k for k, v in GEOMETRY_TYPE_NAMES.items()}


# =============================================================================
# Parameter Counts per Geometry Type
# =============================================================================

GEOMETRY_PARAM_COUNTS = {
    GEOMETRY_BRACKET: 4,    # leg1, leg2, width, thickness
    GEOMETRY_TUBE: 3,       # length, outer_dia, inner_dia
    GEOMETRY_CHANNEL: 4,    # width, height, length, thickness
    GEOMETRY_BLOCK: 3,      # length, width, height
    GEOMETRY_CYLINDER: 2,   # length, diameter
    GEOMETRY_BLOCKHOLE: 6,  # length, width, height, hole_dia, hole_x, hole_y
}

MAX_PARAMS = max(GEOMETRY_PARAM_COUNTS.values())  # 6


# =============================================================================
# Feature Dimensions
# =============================================================================

# Vertex features: 3D coordinates (normalized)
VERTEX_FEATURE_DIM = 3

# Edge features: length, tangent (3D), curvature (2D: start, end)
# Total: 1 + 3 + 2 = 6
EDGE_FEATURE_DIM = 6

# Face features: same as existing (13D)
# [area, dir_xyz, centroid_xyz, curv1, curv2, bbox_d, bbox_cx, bbox_cy, bbox_cz]
FACE_FEATURE_DIM = 13
