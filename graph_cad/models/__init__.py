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
    from graph_cad.models.graph_vae import (
        GraphVAE,
        GraphVAEConfig,
        GraphVAEEncoder,
        GraphVAEDecoder,
    )
    from graph_cad.models.feature_regressor import (
        FeatureRegressor,
        FeatureRegressorConfig,
        load_feature_regressor,
        save_feature_regressor,
    )

    __all__ = [
        # Parameter regressor (GNN-based)
        "ParameterRegressor",
        "ParameterRegressorConfig",
        "PARAMETER_NAMES",
        "brep_graph_to_pyg",
        "normalize_parameters",
        "denormalize_parameters",
        # Feature regressor (MLP-based, for VAE decoded features)
        "FeatureRegressor",
        "FeatureRegressorConfig",
        "load_feature_regressor",
        "save_feature_regressor",
        # Graph VAE
        "GraphVAE",
        "GraphVAEConfig",
        "GraphVAEEncoder",
        "GraphVAEDecoder",
    ]
except ImportError:
    # PyTorch/PyG not installed
    __all__ = []

# Optional: Latent editor (requires transformers, peft)
try:
    from graph_cad.models.latent_editor import (
        LatentEditor,
        LatentEditorConfig,
        LatentProjector,
        OutputProjector,
        load_llm_with_lora,
        create_latent_editor,
    )

    __all__.extend([
        "LatentEditor",
        "LatentEditorConfig",
        "LatentProjector",
        "OutputProjector",
        "load_llm_with_lora",
        "create_latent_editor",
    ])
except ImportError:
    # transformers/peft not installed
    pass
