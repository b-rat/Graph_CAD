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
        VariableGraphVAE,
        VariableGraphVAEConfig,
        VariableGraphVAEEncoder,
        VariableGraphVAEDecoder,
    )
    from graph_cad.models.transformer_decoder import (
        TransformerGraphDecoder,
        TransformerDecoderConfig,
        TransformerGraphVAE,
    )
    from graph_cad.models.feature_regressor import (
        FeatureRegressor,
        FeatureRegressorConfig,
        load_feature_regressor,
        save_feature_regressor,
    )
    # Phase 4: HeteroGNN VAE
    from graph_cad.models.hetero_vae import (
        HeteroVAE,
        HeteroVAEConfig,
        HeteroGNNEncoder,
        create_hetero_vae,
    )
    from graph_cad.models.hetero_decoder import (
        HeteroGraphDecoder,
        HeteroDecoderConfig,
        GeometryAwareParamHead,
        MultiGeometryDecoder,
    )
    from graph_cad.models.losses import (
        MultiGeometryLossConfig,
        multi_geometry_vae_loss,
        multi_geometry_vae_loss_with_direct_latent,
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
        # Graph VAE (fixed topology)
        "GraphVAE",
        "GraphVAEConfig",
        "GraphVAEEncoder",
        "GraphVAEDecoder",
        # Variable topology VAE (Phase 2)
        "VariableGraphVAE",
        "VariableGraphVAEConfig",
        "VariableGraphVAEEncoder",
        "VariableGraphVAEDecoder",
        # Transformer decoder VAE (Phase 3)
        "TransformerGraphDecoder",
        "TransformerDecoderConfig",
        "TransformerGraphVAE",
        # HeteroGNN VAE (Phase 4)
        "HeteroVAE",
        "HeteroVAEConfig",
        "HeteroGNNEncoder",
        "create_hetero_vae",
        "HeteroGraphDecoder",
        "HeteroDecoderConfig",
        "GeometryAwareParamHead",
        "MultiGeometryDecoder",
        "MultiGeometryLossConfig",
        "multi_geometry_vae_loss",
        "multi_geometry_vae_loss_with_direct_latent",
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
    # Phase 4: Extended latent editor
    from graph_cad.models.extended_latent_editor import (
        ExtendedLatentEditor,
        ExtendedLatentEditorConfig,
        GeometryClassificationHead,
        MultiTypeParamHeads,
        compute_extended_editor_loss,
        create_extended_latent_editor,
    )

    __all__.extend([
        "LatentEditor",
        "LatentEditorConfig",
        "LatentProjector",
        "OutputProjector",
        "load_llm_with_lora",
        "create_latent_editor",
        # Extended editor (Phase 4)
        "ExtendedLatentEditor",
        "ExtendedLatentEditorConfig",
        "GeometryClassificationHead",
        "MultiTypeParamHeads",
        "compute_extended_editor_loss",
        "create_extended_latent_editor",
    ])
except ImportError:
    # transformers/peft not installed
    pass
