"""
Full B-Rep graph extraction from CAD solids.

Extracts a heterogeneous graph with three node types:
- Vertices (V): 3D corner points
- Edges (E): Curves connecting vertices
- Faces (F): Surfaces bounded by edges

This enables richer geometric encoding via HeteroGNN compared to
face-only graphs used in Phase 3.

Topology connections:
- vertex_to_edge: Which vertices bound each edge (always 2 for non-seam edges)
- edge_to_face: Which edges bound each face (varies by face complexity)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import math
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.GeomAbs import (
    GeomAbs_Circle,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_Plane,
    GeomAbs_Torus,
)
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex

from graph_cad.data.brep_types import (
    EDGE_TYPE_ARC,
    EDGE_TYPE_CIRCLE,
    EDGE_TYPE_LINE,
    EDGE_TYPE_OTHER,
    FACE_TYPE_FILLET,
    FACE_TYPE_HOLE,
    FACE_TYPE_PLANAR,
)


@dataclass
class BRepHeteroGraph:
    """
    Full B-Rep heterogeneous graph with vertices, edges, and faces.

    Attributes:
        vertex_features: Normalized vertex coordinates, shape (num_v, 3).
        edge_features: Edge features, shape (num_e, 6).
            [length_norm, tangent_x, tangent_y, tangent_z, curv_start, curv_end]
        face_features: Face features, shape (num_f, 13).
            [area_norm, dir_xyz, centroid_xyz, curv1, curv2, bbox_d, bbox_cxyz]
        edge_types: Edge type indices, shape (num_e,).
            0=LINE, 1=ARC, 2=CIRCLE, 3=OTHER
        face_types: Face type indices, shape (num_f,).
            0=PLANAR, 1=HOLE, 2=FILLET
        vertex_to_edge: Topology (2, num_v2e).
            [vertex_indices, edge_indices] - which vertices bound each edge.
        edge_to_face: Topology (2, num_e2f).
            [edge_indices, face_indices] - which edges bound each face.
        bbox_diagonal: Bounding box diagonal for denormalization.
        bbox_center: Bounding box center, shape (3,).
        num_vertices: Number of vertices.
        num_edges: Number of edges.
        num_faces: Number of faces.
        source_file: Optional path to source STEP file.
    """

    vertex_features: NDArray[np.float32]
    edge_features: NDArray[np.float32]
    face_features: NDArray[np.float32]
    edge_types: NDArray[np.int64]
    face_types: NDArray[np.int64]
    vertex_to_edge: NDArray[np.int64]
    edge_to_face: NDArray[np.int64]
    bbox_diagonal: float
    bbox_center: NDArray[np.float32]
    num_vertices: int
    num_edges: int
    num_faces: int
    source_file: str | None = None


def extract_brep_hetero_graph(step_path: Path | str) -> BRepHeteroGraph:
    """
    Extract B-Rep heterogeneous graph from STEP file.

    Args:
        step_path: Path to STEP file.

    Returns:
        BRepHeteroGraph with full V/E/F topology.

    Raises:
        FileNotFoundError: If STEP file does not exist.
        ValueError: If STEP file cannot be loaded.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    try:
        solid = cq.importers.importStep(str(step_path))
    except Exception as e:
        raise ValueError(f"Failed to load STEP file: {step_path}") from e

    graph = extract_brep_hetero_graph_from_solid(solid)
    graph.source_file = str(step_path)
    return graph


def extract_brep_hetero_graph_from_solid(solid: cq.Workplane) -> BRepHeteroGraph:
    """
    Extract B-Rep heterogeneous graph from CadQuery solid.

    Args:
        solid: CadQuery Workplane containing the solid.

    Returns:
        BRepHeteroGraph with full V/E/F topology.

    Raises:
        ValueError: If solid contains no faces.
    """
    shape = solid.val().wrapped

    # Compute bounding box for normalization
    bbox = solid.val().BoundingBox()
    bbox_min = np.array([bbox.xmin, bbox.ymin, bbox.zmin], dtype=np.float32)
    bbox_max = np.array([bbox.xmax, bbox.ymax, bbox.zmax], dtype=np.float32)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_diagonal = float(np.linalg.norm(bbox_max - bbox_min))

    if bbox_diagonal < 1e-10:
        bbox_diagonal = 1.0

    # ==========================================================================
    # Extract Vertices
    # ==========================================================================
    vertices = []
    vertex_coords = []
    vertex_hash_to_idx: dict[int, int] = {}

    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        vertex = TopoDS.Vertex_s(explorer.Current())
        vertex_hash = vertex.__hash__()

        if vertex_hash not in vertex_hash_to_idx:
            pnt = BRep_Tool.Pnt_s(vertex)
            coord = np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32)

            # Check for coordinate-based deduplication (OCC may have duplicate vertices)
            # Use a small tolerance for floating-point comparison
            is_duplicate = False
            for idx, existing_coord in enumerate(vertex_coords):
                if np.linalg.norm(coord - existing_coord) < 1e-6:
                    vertex_hash_to_idx[vertex_hash] = idx
                    is_duplicate = True
                    break

            if not is_duplicate:
                idx = len(vertices)
                vertex_hash_to_idx[vertex_hash] = idx
                vertices.append(vertex)
                vertex_coords.append(coord)

        explorer.Next()

    num_vertices = len(vertices)

    # Normalize vertex coordinates
    vertex_features = np.zeros((num_vertices, 3), dtype=np.float32)
    for i, coord in enumerate(vertex_coords):
        vertex_features[i] = (coord - bbox_center) / bbox_diagonal

    # ==========================================================================
    # Extract Edges
    # ==========================================================================
    edges = []
    edge_hash_to_idx: dict[int, int] = {}

    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        edge_hash = edge.__hash__()

        if edge_hash not in edge_hash_to_idx:
            edge_hash_to_idx[edge_hash] = len(edges)
            edges.append(edge)

        explorer.Next()

    num_edges = len(edges)

    # Extract edge features and types
    edge_features = np.zeros((num_edges, 6), dtype=np.float32)
    edge_types = np.zeros(num_edges, dtype=np.int64)

    for i, edge in enumerate(edges):
        edge_type, length, tangent, curv_start, curv_end = _extract_edge_features(edge)
        edge_types[i] = edge_type
        edge_features[i, 0] = length / bbox_diagonal  # Normalized length
        edge_features[i, 1:4] = tangent  # Tangent direction (unit vector)
        edge_features[i, 4] = curv_start * bbox_diagonal  # Normalized curvature
        edge_features[i, 5] = curv_end * bbox_diagonal

        # Clip extreme curvatures
        edge_features[i, 4:6] = np.clip(edge_features[i, 4:6], -10.0, 10.0)

    # ==========================================================================
    # Extract Faces
    # ==========================================================================
    faces = []
    face_hash_to_idx: dict[int, int] = {}

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        face_hash = face.__hash__()

        if face_hash not in face_hash_to_idx:
            face_hash_to_idx[face_hash] = len(faces)
            faces.append(face)

        explorer.Next()

    num_faces = len(faces)

    if num_faces == 0:
        raise ValueError("Solid contains no faces")

    # Normalize bbox values for consistent scale
    bbox_diagonal_normalized = bbox_diagonal / 100.0
    bbox_center_normalized = bbox_center / 100.0

    # Extract face features and types
    face_features = np.zeros((num_faces, 13), dtype=np.float32)
    face_types = np.zeros(num_faces, dtype=np.int64)

    for i, face in enumerate(faces):
        face_type, area, normal, centroid, curvatures = _extract_face_features(face)
        face_types[i] = face_type

        # Normalize
        area_normalized = area / (bbox_diagonal ** 2)
        centroid_normalized = (centroid - bbox_center) / bbox_diagonal
        curv1_normalized = np.clip(curvatures[0] * bbox_diagonal, -10.0, 10.0)
        curv2_normalized = np.clip(curvatures[1] * bbox_diagonal, -10.0, 10.0)

        face_features[i, 0] = area_normalized
        face_features[i, 1:4] = normal
        face_features[i, 4:7] = centroid_normalized
        face_features[i, 7] = curv1_normalized
        face_features[i, 8] = curv2_normalized
        face_features[i, 9] = bbox_diagonal_normalized
        face_features[i, 10:13] = bbox_center_normalized

    # ==========================================================================
    # Build Topology: Vertex -> Edge
    # ==========================================================================
    v2e_vertex_indices = []
    v2e_edge_indices = []

    for edge_idx, edge in enumerate(edges):
        # Get vertices of this edge
        v_first = TopExp.FirstVertex_s(edge)
        v_last = TopExp.LastVertex_s(edge)

        # Map to our vertex indices using hash
        v_first_hash = v_first.__hash__()
        v_last_hash = v_last.__hash__()

        if v_first_hash in vertex_hash_to_idx:
            v2e_vertex_indices.append(vertex_hash_to_idx[v_first_hash])
            v2e_edge_indices.append(edge_idx)

        if v_last_hash in vertex_hash_to_idx and v_last_hash != v_first_hash:
            v2e_vertex_indices.append(vertex_hash_to_idx[v_last_hash])
            v2e_edge_indices.append(edge_idx)

    vertex_to_edge = np.array(
        [v2e_vertex_indices, v2e_edge_indices], dtype=np.int64
    )

    # ==========================================================================
    # Build Topology: Edge -> Face
    # ==========================================================================
    e2f_edge_indices = []
    e2f_face_indices = []

    # Build map from edges to parent faces
    edge_to_faces_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces_map)

    for edge_occ_idx in range(1, edge_to_faces_map.Extent() + 1):
        edge = edge_to_faces_map.FindKey(edge_occ_idx)
        edge_hash = edge.__hash__()

        if edge_hash not in edge_hash_to_idx:
            continue

        edge_idx = edge_hash_to_idx[edge_hash]
        face_list = edge_to_faces_map.FindFromIndex(edge_occ_idx)

        for face_shape in face_list:
            face_hash = face_shape.__hash__()
            if face_hash in face_hash_to_idx:
                face_idx = face_hash_to_idx[face_hash]
                e2f_edge_indices.append(edge_idx)
                e2f_face_indices.append(face_idx)

    edge_to_face = np.array(
        [e2f_edge_indices, e2f_face_indices], dtype=np.int64
    )

    return BRepHeteroGraph(
        vertex_features=vertex_features,
        edge_features=edge_features,
        face_features=face_features,
        edge_types=edge_types,
        face_types=face_types,
        vertex_to_edge=vertex_to_edge,
        edge_to_face=edge_to_face,
        bbox_diagonal=bbox_diagonal,
        bbox_center=bbox_center,
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_faces=num_faces,
        source_file=None,
    )


def _extract_edge_features(
    edge: TopoDS_Edge,
) -> tuple[int, float, NDArray[np.float32], float, float]:
    """
    Extract features from a single edge.

    Args:
        edge: OCC TopoDS_Edge object.

    Returns:
        Tuple of (edge_type, length, tangent, curvature_start, curvature_end).
        - edge_type: 0=LINE, 1=ARC, 2=CIRCLE, 3=OTHER
        - length: Edge length in mm
        - tangent: Unit tangent vector at midpoint
        - curvature_start: Curvature at start
        - curvature_end: Curvature at end
    """
    adaptor = BRepAdaptor_Curve(edge)
    curve_type = adaptor.GetType()

    # Classify edge type
    if curve_type == GeomAbs_Line:
        edge_type = EDGE_TYPE_LINE
    elif curve_type == GeomAbs_Circle:
        # Distinguish full circle from arc
        u_min = adaptor.FirstParameter()
        u_max = adaptor.LastParameter()
        arc_extent = u_max - u_min
        if arc_extent >= 2 * math.pi - 0.01:  # ~360 degrees
            edge_type = EDGE_TYPE_CIRCLE
        else:
            edge_type = EDGE_TYPE_ARC
    else:
        edge_type = EDGE_TYPE_OTHER

    # Compute length
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    length = props.Mass()

    # Get tangent at midpoint
    u_mid = (adaptor.FirstParameter() + adaptor.LastParameter()) / 2
    pnt = gp_Pnt()
    tangent_vec = gp_Vec()

    try:
        adaptor.D1(u_mid, pnt, tangent_vec)
        if tangent_vec.Magnitude() > 1e-10:
            tangent_vec.Normalize()
            tangent = np.array(
                [tangent_vec.X(), tangent_vec.Y(), tangent_vec.Z()],
                dtype=np.float32
            )
        else:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    except Exception:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Get curvature at start and end
    curv_start = _get_edge_curvature(adaptor, adaptor.FirstParameter())
    curv_end = _get_edge_curvature(adaptor, adaptor.LastParameter())

    return edge_type, length, tangent, curv_start, curv_end


def _get_edge_curvature(adaptor: BRepAdaptor_Curve, param: float) -> float:
    """Get curvature of edge at given parameter."""
    try:
        pnt = gp_Pnt()
        d1 = gp_Vec()
        d2 = gp_Vec()
        adaptor.D2(param, pnt, d1, d2)

        # Curvature = |d1 x d2| / |d1|^3
        cross = d1.Crossed(d2)
        d1_mag = d1.Magnitude()
        if d1_mag > 1e-10:
            curvature = cross.Magnitude() / (d1_mag ** 3)
        else:
            curvature = 0.0

        return float(curvature)
    except Exception:
        return 0.0


def _extract_face_features(
    face: TopoDS_Face,
) -> tuple[int, float, NDArray[np.float32], NDArray[np.float32], tuple[float, float]]:
    """
    Extract features from a single face.

    Args:
        face: OCC TopoDS_Face object.

    Returns:
        Tuple of (face_type, area, normal, centroid, curvatures).
        Matches the existing face feature extraction in graph_extraction.py.
    """
    adaptor = BRepAdaptor_Surface(face)
    surface_type = adaptor.GetType()

    # Classify face type (matches existing logic)
    if surface_type == GeomAbs_Plane:
        face_type = FACE_TYPE_PLANAR
    elif surface_type == GeomAbs_Cylinder:
        u_min = adaptor.FirstUParameter()
        u_max = adaptor.LastUParameter()
        arc_extent = u_max - u_min
        if arc_extent >= math.pi:  # >= 180 degrees
            face_type = FACE_TYPE_HOLE
        else:
            face_type = FACE_TYPE_FILLET
    elif surface_type == GeomAbs_Torus:
        face_type = FACE_TYPE_FILLET
    else:
        face_type = FACE_TYPE_FILLET  # Default for other curved surfaces

    # Compute area
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    area = props.Mass()

    # Get centroid
    center = props.CentreOfMass()
    centroid = np.array([center.X(), center.Y(), center.Z()], dtype=np.float32)

    # Get direction vector (normal for planes, axis for cylinders)
    normal = _get_face_direction(adaptor, surface_type, centroid)

    # Get curvature
    curvatures = _get_face_curvature(adaptor)

    return face_type, area, normal, centroid, curvatures


def _get_face_direction(
    adaptor: BRepAdaptor_Surface,
    surface_type,
    centroid: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Get direction vector for face (normal or axis)."""
    if surface_type == GeomAbs_Plane:
        plane = adaptor.Plane()
        axis = plane.Axis()
        direction = axis.Direction()
    elif surface_type == GeomAbs_Cylinder:
        cylinder = adaptor.Cylinder()
        axis = cylinder.Axis()
        direction = axis.Direction()
    elif surface_type == GeomAbs_Torus:
        torus = adaptor.Torus()
        axis = torus.Axis()
        direction = axis.Direction()
    else:
        # Fallback: compute normal from surface derivatives
        u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

        try:
            pnt = gp_Pnt()
            d1u = gp_Vec()
            d1v = gp_Vec()
            adaptor.D1(u_mid, v_mid, pnt, d1u, d1v)
            normal_vec = d1u.Crossed(d1v)
            if normal_vec.Magnitude() > 1e-10:
                normal_vec.Normalize()
                return np.array(
                    [normal_vec.X(), normal_vec.Y(), normal_vec.Z()],
                    dtype=np.float32
                )
        except Exception:
            pass

        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    normal = np.array(
        [direction.X(), direction.Y(), direction.Z()],
        dtype=np.float32
    )

    # Ensure unit length
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal = normal / norm

    return normal


def _get_face_curvature(
    adaptor: BRepAdaptor_Surface,
) -> tuple[float, float]:
    """Get principal curvatures at face midpoint."""
    try:
        u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

        props = BRepLProp_SLProps(adaptor, u_mid, v_mid, 2, 1e-6)
        if props.IsCurvatureDefined():
            return props.MinCurvature(), props.MaxCurvature()
    except Exception:
        pass

    return 0.0, 0.0


def brep_hetero_to_pyg(graph: BRepHeteroGraph) -> "HeteroData":
    """
    Convert BRepHeteroGraph to PyTorch Geometric HeteroData.

    Args:
        graph: BRepHeteroGraph instance.

    Returns:
        PyG HeteroData object with node types 'vertex', 'edge', 'face'
        and edge types 'vertex_to_edge', 'edge_to_face'.
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Node features
    data['vertex'].x = torch.tensor(graph.vertex_features, dtype=torch.float32)
    data['edge'].x = torch.tensor(graph.edge_features, dtype=torch.float32)
    data['face'].x = torch.tensor(graph.face_features, dtype=torch.float32)

    # Node types (for embeddings)
    data['edge'].edge_type = torch.tensor(graph.edge_types, dtype=torch.long)
    data['face'].face_type = torch.tensor(graph.face_types, dtype=torch.long)

    # Edge indices (topology)
    # vertex -> edge (vertices bound edges)
    data['vertex', 'bounds', 'edge'].edge_index = torch.tensor(
        graph.vertex_to_edge, dtype=torch.long
    )

    # edge -> face (edges bound faces)
    data['edge', 'bounds', 'face'].edge_index = torch.tensor(
        graph.edge_to_face, dtype=torch.long
    )

    # Also add reverse edges for bidirectional message passing
    data['edge', 'bounded_by', 'vertex'].edge_index = torch.tensor(
        graph.vertex_to_edge[[1, 0]], dtype=torch.long
    )
    data['face', 'bounded_by', 'edge'].edge_index = torch.tensor(
        graph.edge_to_face[[1, 0]], dtype=torch.long
    )

    # Store metadata
    data.bbox_diagonal = graph.bbox_diagonal
    data.bbox_center = torch.tensor(graph.bbox_center, dtype=torch.float32)
    data.num_vertices = graph.num_vertices
    data.num_edges = graph.num_edges
    data.num_faces = graph.num_faces

    return data
