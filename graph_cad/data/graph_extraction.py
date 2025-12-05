"""
Graph extraction from B-Rep CAD solids.

Extracts face-adjacency graphs with node and edge features from STEP files
or CadQuery solids for use in graph neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face


# Face type codes
FACE_TYPE_PLANAR = 0
FACE_TYPE_CYLINDRICAL = 1
FACE_TYPE_OTHER = 2


@dataclass
class BRepGraph:
    """
    Face-adjacency graph extracted from B-Rep solid.

    Attributes:
        node_features: Node feature matrix, shape (num_faces, 8).
            Columns: [face_type, area, dir_x, dir_y, dir_z,
                      centroid_x, centroid_y, centroid_z]
            - face_type: 0=planar, 1=cylindrical, 2=other
            - area: normalized by bbox_diagonal^2
            - dir_x/y/z: unit direction vector
              - For planar faces: surface normal
              - For cylindrical faces: cylinder axis direction
            - centroid_x/y/z: normalized by bbox_diagonal, centered on bbox
        edge_index: Edge indices in COO format, shape (2, num_edges).
            Each column [i, j] represents an edge between face i and face j.
        edge_features: Edge feature matrix, shape (num_edges, 2).
            Columns: [edge_length, dihedral_angle]
        bbox_diagonal: Bounding box diagonal used for normalization.
        bbox_center: Bounding box center used for normalization, shape (3,).
        num_faces: Number of faces (nodes) in the graph.
        num_edges: Number of edges in the graph.
        source_file: Path to source STEP file, if loaded from file.
    """

    node_features: NDArray[np.float32]
    edge_index: NDArray[np.int64]
    edge_features: NDArray[np.float32]
    bbox_diagonal: float
    bbox_center: NDArray[np.float32]
    num_faces: int
    num_edges: int
    source_file: str | None = None


def extract_graph(step_path: Path | str) -> BRepGraph:
    """
    Extract face-adjacency graph from STEP file.

    Args:
        step_path: Path to STEP file.

    Returns:
        BRepGraph with extracted topology and features.

    Raises:
        FileNotFoundError: If STEP file does not exist.
        ValueError: If STEP file cannot be loaded or contains no solid.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    try:
        solid = cq.importers.importStep(str(step_path))
    except Exception as e:
        raise ValueError(f"Failed to load STEP file: {step_path}") from e

    graph = extract_graph_from_solid(solid)
    graph.source_file = str(step_path)
    return graph


def extract_graph_from_solid(solid: cq.Workplane) -> BRepGraph:
    """
    Extract face-adjacency graph from CadQuery solid.

    Args:
        solid: CadQuery Workplane containing the solid.

    Returns:
        BRepGraph with extracted topology and features.

    Raises:
        ValueError: If solid contains no faces.
    """
    # Get the underlying OCC shape
    shape = solid.val().wrapped

    # Extract faces using OCC topology explorer
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        faces.append(face)
        explorer.Next()

    if not faces:
        raise ValueError("Solid contains no faces")

    num_faces = len(faces)

    # Compute bounding box for normalization
    bbox = solid.val().BoundingBox()
    bbox_min = np.array([bbox.xmin, bbox.ymin, bbox.zmin], dtype=np.float32)
    bbox_max = np.array([bbox.xmax, bbox.ymax, bbox.zmax], dtype=np.float32)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    if bbox_diagonal < 1e-10:
        bbox_diagonal = 1.0  # Avoid division by zero for degenerate geometry

    # Extract node features for each face
    node_features = np.zeros((num_faces, 8), dtype=np.float32)
    face_normals = []  # Store for dihedral angle computation

    for i, face in enumerate(faces):
        face_type, area, normal, centroid = _extract_face_features(face)

        # Normalize centroid relative to bbox
        centroid_normalized = (centroid - bbox_center) / bbox_diagonal

        # Normalize area by bbox_diagonal^2 (area scales with length^2)
        area_normalized = area / (bbox_diagonal**2)

        node_features[i, 0] = face_type
        node_features[i, 1] = area_normalized
        node_features[i, 2:5] = normal
        node_features[i, 5:8] = centroid_normalized

        face_normals.append(normal)

    # Build face adjacency from shared edges
    adjacency, edge_lengths = _build_face_adjacency(faces, shape)

    # Convert adjacency to edge_index (COO format)
    edge_list = []
    edge_feature_list = []

    for (i, j), length in edge_lengths.items():
        edge_list.append([i, j])

        # Compute dihedral angle between faces
        normal_i = face_normals[i]
        normal_j = face_normals[j]
        dihedral = _compute_dihedral_angle(normal_i, normal_j)

        # Normalize edge length
        length_normalized = length / bbox_diagonal

        edge_feature_list.append([length_normalized, dihedral])

    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T  # Shape: (2, num_edges)
        edge_features = np.array(edge_feature_list, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, 2), dtype=np.float32)

    return BRepGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        bbox_diagonal=bbox_diagonal,
        bbox_center=bbox_center,
        num_faces=num_faces,
        num_edges=edge_index.shape[1],
        source_file=None,
    )


def _extract_face_features(
    face,
) -> tuple[int, float, NDArray[np.float32], NDArray[np.float32]]:
    """
    Extract features from a single face.

    Args:
        face: OCC TopoDS_Face object.

    Returns:
        Tuple of (face_type, area, normal_or_axis, centroid).
        For planar faces, normal_or_axis is the surface normal.
        For cylindrical faces, normal_or_axis is the cylinder axis direction.
    """
    # Get surface type
    adaptor = BRepAdaptor_Surface(face)
    surface_type = adaptor.GetType()

    if surface_type == GeomAbs_Plane:
        face_type = FACE_TYPE_PLANAR
    elif surface_type == GeomAbs_Cylinder:
        face_type = FACE_TYPE_CYLINDRICAL
    else:
        face_type = FACE_TYPE_OTHER

    # Compute area
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area = props.Mass()

    # Get centroid
    center = props.CentreOfMass()
    centroid = np.array([center.X(), center.Y(), center.Z()], dtype=np.float32)

    # Get direction vector based on surface type
    # For planar faces: surface normal
    # For cylindrical faces: cylinder axis direction
    if surface_type == GeomAbs_Cylinder:
        # Extract cylinder axis direction
        cylinder = adaptor.Cylinder()
        axis = cylinder.Axis()
        direction = axis.Direction()
        normal_or_axis = np.array(
            [direction.X(), direction.Y(), direction.Z()], dtype=np.float32
        )
    elif surface_type == GeomAbs_Plane:
        # Extract plane normal
        plane = adaptor.Plane()
        axis = plane.Axis()
        direction = axis.Direction()
        normal_or_axis = np.array(
            [direction.X(), direction.Y(), direction.Z()], dtype=np.float32
        )
    else:
        # Fallback: compute normal from surface derivatives at midpoint
        u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

        surface = adaptor.Surface().Surface()
        from OCP.gp import gp_Pnt, gp_Vec

        pnt = gp_Pnt()
        d1u = gp_Vec()
        d1v = gp_Vec()

        try:
            surface.D1(u_mid, v_mid, pnt, d1u, d1v)
            normal_vec = d1u.Crossed(d1v)
            if normal_vec.Magnitude() > 1e-10:
                normal_vec.Normalize()
                normal_or_axis = np.array(
                    [normal_vec.X(), normal_vec.Y(), normal_vec.Z()], dtype=np.float32
                )
            else:
                normal_or_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        except Exception:
            normal_or_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Ensure unit length
    norm = np.linalg.norm(normal_or_axis)
    if norm > 1e-10:
        normal_or_axis = normal_or_axis / norm

    return face_type, area, normal_or_axis, centroid


def _build_face_adjacency(
    faces: list, shape
) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], float]]:
    """
    Build face adjacency graph by finding shared edges.

    Args:
        faces: List of OCC TopoDS_Face objects.
        shape: The full OCC shape (for edge extraction).

    Returns:
        Tuple of (adjacency_dict, edge_lengths_dict).
        adjacency_dict: {(face_i, face_j): True} for adjacent faces.
        edge_lengths_dict: {(face_i, face_j): total_edge_length}.
    """
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    # Build map from edges to their parent faces
    edge_to_faces_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces_map)

    # Create face index lookup
    face_to_idx = {}
    for i, face in enumerate(faces):
        # Use hash for comparison (OCC shapes are hashable)
        face_to_idx[face.__hash__()] = i

    adjacency = {}
    edge_lengths: dict[tuple[int, int], float] = {}

    # Iterate through all edges
    for edge_idx in range(1, edge_to_faces_map.Extent() + 1):
        edge = edge_to_faces_map.FindKey(edge_idx)
        face_list = edge_to_faces_map.FindFromIndex(edge_idx)

        # Get edge length
        edge_props = GProp_GProps()
        BRepGProp.LinearProperties_s(edge, edge_props)
        edge_length = edge_props.Mass()

        # Get all faces sharing this edge
        face_indices = []
        for face_shape in face_list:
            face_hash = face_shape.__hash__()
            if face_hash in face_to_idx:
                face_indices.append(face_to_idx[face_hash])

        # For manifold geometry, most edges connect exactly 2 faces
        # For seam edges (cylinder), we might see the same face twice
        unique_indices = list(set(face_indices))

        if len(unique_indices) == 2:
            i, j = sorted(unique_indices)
            key = (i, j)
            adjacency[key] = True
            edge_lengths[key] = edge_lengths.get(key, 0.0) + edge_length

    return adjacency, edge_lengths


def _compute_dihedral_angle(
    normal_i: NDArray[np.float32], normal_j: NDArray[np.float32]
) -> float:
    """
    Compute dihedral angle between two faces from their normals.

    The dihedral angle is the angle between the outward-facing normals.
    For faces meeting at 90°, the dihedral angle is π/2.
    For coplanar faces (same direction), the angle is 0.
    For opposite-facing faces, the angle is π.

    Args:
        normal_i: Unit normal vector of face i.
        normal_j: Unit normal vector of face j.

    Returns:
        Dihedral angle in radians, range [0, π].
    """
    dot = np.clip(np.dot(normal_i, normal_j), -1.0, 1.0)
    return float(np.arccos(dot))
