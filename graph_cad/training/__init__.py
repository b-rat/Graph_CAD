"""
Training loops, loss functions, and optimization utilities.
"""

try:
    from graph_cad.training.vae_trainer import (
        BetaScheduleConfig,
        BetaScheduler,
        compute_latent_metrics,
        evaluate,
        load_checkpoint,
        prepare_batch_targets,
        save_checkpoint,
        train_epoch,
    )

    __all__ = [
        "BetaScheduleConfig",
        "BetaScheduler",
        "train_epoch",
        "evaluate",
        "compute_latent_metrics",
        "prepare_batch_targets",
        "save_checkpoint",
        "load_checkpoint",
    ]
except ImportError:
    __all__ = []
