"""
Data loading, preprocessing, and CAD file parsers.
"""

from graph_cad.data.graph_extraction import (
    BRepGraph,
    extract_graph,
    extract_graph_from_solid,
)
from graph_cad.data.l_bracket import LBracket, LBracketRanges

# Phase 4: B-Rep extraction and geometry types
from graph_cad.data.brep_types import (
    EDGE_TYPE_LINE,
    EDGE_TYPE_ARC,
    EDGE_TYPE_CIRCLE,
    EDGE_TYPE_OTHER,
    FACE_TYPE_PLANAR,
    FACE_TYPE_HOLE,
    FACE_TYPE_FILLET,
    GEOMETRY_BRACKET,
    GEOMETRY_TUBE,
    GEOMETRY_CHANNEL,
    GEOMETRY_BLOCK,
    GEOMETRY_CYLINDER,
    GEOMETRY_BLOCKHOLE,
    NUM_GEOMETRY_TYPES,
    MAX_PARAMS,
)
from graph_cad.data.brep_extraction import (
    BRepHeteroGraph,
    extract_brep_hetero_graph,
    extract_brep_hetero_graph_from_solid,
    brep_hetero_to_pyg,
)
from graph_cad.data.geometry_generators import (
    Tube,
    TubeRanges,
    Channel,
    ChannelRanges,
    Block,
    BlockRanges,
    Cylinder,
    CylinderRanges,
    BlockHole,
    BlockHoleRanges,
)
from graph_cad.data.param_normalization import (
    normalize_params,
    denormalize_params,
    normalize_params_to_latent,
    denormalize_params_from_latent,
    pad_params,
    unpad_params,
    MultiGeometryNormalizer,
    PARAM_NAMES,
)

__all__ = [
    # Original exports
    "LBracket",
    "LBracketRanges",
    "BRepGraph",
    "extract_graph",
    "extract_graph_from_solid",
    # B-Rep types
    "EDGE_TYPE_LINE",
    "EDGE_TYPE_ARC",
    "EDGE_TYPE_CIRCLE",
    "EDGE_TYPE_OTHER",
    "FACE_TYPE_PLANAR",
    "FACE_TYPE_HOLE",
    "FACE_TYPE_FILLET",
    "GEOMETRY_BRACKET",
    "GEOMETRY_TUBE",
    "GEOMETRY_CHANNEL",
    "GEOMETRY_BLOCK",
    "GEOMETRY_CYLINDER",
    "GEOMETRY_BLOCKHOLE",
    "NUM_GEOMETRY_TYPES",
    "MAX_PARAMS",
    # B-Rep extraction
    "BRepHeteroGraph",
    "extract_brep_hetero_graph",
    "extract_brep_hetero_graph_from_solid",
    "brep_hetero_to_pyg",
    # Geometry generators
    "Tube",
    "TubeRanges",
    "Channel",
    "ChannelRanges",
    "Block",
    "BlockRanges",
    "Cylinder",
    "CylinderRanges",
    "BlockHole",
    "BlockHoleRanges",
    # Parameter normalization
    "normalize_params",
    "denormalize_params",
    "normalize_params_to_latent",
    "denormalize_params_from_latent",
    "pad_params",
    "unpad_params",
    "MultiGeometryNormalizer",
    "PARAM_NAMES",
]

# Optional imports requiring PyTorch
try:
    from graph_cad.data.dataset import LBracketDataset, create_data_loaders

    __all__.extend(["LBracketDataset", "create_data_loaders"])
except ImportError:
    pass

# Edit dataset for latent editor training
try:
    from graph_cad.data.edit_dataset import (
        LatentEditDataset,
        collate_edit_batch,
        generate_instruction,
        INSTRUCTION_TEMPLATES,
        COMPOUND_TEMPLATES,
    )

    __all__.extend([
        "LatentEditDataset",
        "collate_edit_batch",
        "generate_instruction",
        "INSTRUCTION_TEMPLATES",
        "COMPOUND_TEMPLATES",
    ])
except ImportError:
    pass

# Phase 4: Multi-geometry dataset
try:
    from graph_cad.data.multi_geometry_dataset import (
        MultiGeometryDataset,
        create_multi_geometry_loaders,
        brep_graph_to_hetero_data,
    )

    __all__.extend([
        "MultiGeometryDataset",
        "create_multi_geometry_loaders",
        "brep_graph_to_hetero_data",
    ])
except ImportError:
    pass
