"""
Data loading, preprocessing, and CAD file parsers.
"""

from graph_cad.data.graph_extraction import (
    BRepGraph,
    extract_graph,
    extract_graph_from_solid,
)
from graph_cad.data.l_bracket import LBracket, LBracketRanges

__all__ = [
    "LBracket",
    "LBracketRanges",
    "BRepGraph",
    "extract_graph",
    "extract_graph_from_solid",
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
