"""
Model definitions for graph autoencoders and generative models.
"""

try:
    from graph_cad.models.parameter_regressor import (
        PARAMETER_NAMES,
        ParameterRegressor,
        ParameterRegressorConfig,
        brep_graph_to_pyg,
        denormalize_parameters,
        normalize_parameters,
    )

    __all__ = [
        "ParameterRegressor",
        "ParameterRegressorConfig",
        "PARAMETER_NAMES",
        "brep_graph_to_pyg",
        "normalize_parameters",
        "denormalize_parameters",
    ]
except ImportError:
    # PyTorch/PyG not installed
    __all__ = []
