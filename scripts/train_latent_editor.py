#!/usr/bin/env python3
"""
Train the Latent Editor (Mistral 7B + LoRA) for CAD modifications.

Usage:
    # Basic training
    python scripts/train_latent_editor.py --data-dir data/edit_data

    # With custom settings
    python scripts/train_latent_editor.py --epochs 10 --batch-size 4 --gradient-accumulation 8

    # Resume from checkpoint
    python scripts/train_latent_editor.py --resume outputs/editor/checkpoint_epoch_5.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data.edit_dataset import (
    LatentEditDataset,
    PairedLatentEditDataset,
    collate_edit_batch,
    collate_paired_edit_batch,
)
from graph_cad.models.latent_editor import (
    LatentEditor,
    LatentEditorConfig,
    load_llm_with_lora,
)
from graph_cad.training.edit_trainer import (
    EditLossConfig,
    evaluate,
    evaluate_paired,
    evaluate_with_direction,
    save_editor_checkpoint,
    train_epoch,
    train_epoch_paired,
    train_epoch_with_direction,
)


def main():
    parser = argparse.ArgumentParser(description="Train latent editor")

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/edit_data",
        help="Directory with train.json, val.json, test.json",
    )

    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Latent dimension (must match VAE; 32 for Transformer VAE, 16 for older VAEs)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable quantization (requires more VRAM)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )

    # Loss
    parser.add_argument(
        "--delta-weight",
        type=float,
        default=1.0,
        help="Weight for delta MSE loss",
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.0,
        help="Weight for graph reconstruction loss (requires VAE)",
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.0,
        help="Weight for contrastive loss (requires paired data)",
    )
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=0.0,
        help="Weight for direction classifier loss (auxiliary supervision)",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=None,
        help="VAE checkpoint for graph reconstruction loss",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/editor",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Other
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Warning: MPS does not support 4-bit quantization. Disabling.")
        args.use_4bit = False
        args.use_8bit = False
    else:
        device = "cpu"
        print("Warning: Running on CPU will be very slow")

    print(f"Using device: {device}")

    # Load datasets
    print(f"\nLoading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)

    # Check if data is paired (for contrastive learning) and get latent dim from metadata
    metadata_path = data_dir / "metadata.json"
    is_paired = False
    data_latent_dim = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            is_paired = metadata.get("paired", False)
            data_latent_dim = metadata.get("latent_dim")

    # Auto-detect latent dim from data metadata if not explicitly set and metadata exists
    if data_latent_dim is not None and args.latent_dim != data_latent_dim:
        print(f"  Note: Data was generated with latent_dim={data_latent_dim}")
        print(f"  Using --latent-dim={args.latent_dim} (override with --latent-dim if needed)")
        # Use the data's latent dim if user didn't explicitly change from default
        if args.latent_dim == 32:  # Default value
            args.latent_dim = data_latent_dim
            print(f"  Auto-detected latent_dim={args.latent_dim} from data metadata")

    if args.contrastive_weight > 0 and not is_paired:
        print("Warning: --contrastive-weight > 0 but data is not paired.")
        print("         Generate paired data with: python scripts/generate_edit_data.py --paired")
        print("         Continuing with standard training...")
        is_paired = False

    if is_paired:
        print("  Using paired dataset for contrastive learning")
        train_dataset = PairedLatentEditDataset(data_dir / "train.json")
        val_dataset = PairedLatentEditDataset(data_dir / "val.json")
        collate_fn = collate_paired_edit_batch
    else:
        train_dataset = LatentEditDataset(data_dir / "train.json")
        val_dataset = LatentEditDataset(data_dir / "val.json")
        collate_fn = collate_edit_batch

    # Print data info from metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if "parameters" in metadata:
            print(f"  Parameters: {metadata['parameters']}")
        if "edit_types" in metadata:
            print(f"  Edit types: {metadata['edit_types']}")
        if "variable_topology" in metadata:
            print(f"  Variable topology: {metadata['variable_topology']}")

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    # Create model config
    config = LatentEditorConfig(
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit and not args.no_quantization,
        use_8bit=args.use_8bit and not args.no_quantization and not args.use_4bit,
    )

    # Check if using direction classifier
    use_direction = args.direction_weight > 0

    # Create editor (without LLM initially)
    print("\nCreating LatentEditor...")
    editor = LatentEditor(config, use_direction_classifier=use_direction)
    if use_direction:
        print("  Direction classifier: ENABLED")

    # Load LLM with LoRA
    print(f"Loading {args.model_name} with LoRA...")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  4-bit: {config.use_4bit}, 8-bit: {config.use_8bit}")

    llm, tokenizer = load_llm_with_lora(config, device_map="auto")
    editor.set_llm(llm, tokenizer)

    # Move projectors to device
    editor.latent_projector.to(device)
    editor.output_projector.to(device)
    if editor.direction_classifier is not None:
        editor.direction_classifier.to(device)

    print(f"\nTrainable parameters: {editor.num_trainable_params():,}")

    # Loss config
    loss_config = EditLossConfig(
        delta_weight=args.delta_weight,
        graph_weight=args.graph_weight,
        contrastive_weight=args.contrastive_weight,
        direction_weight=args.direction_weight,
    )

    if args.contrastive_weight > 0:
        print(f"  Contrastive weight: {args.contrastive_weight}")
    if args.direction_weight > 0:
        print(f"  Direction weight: {args.direction_weight}")

    # Load VAE if using graph reconstruction loss
    vae = None
    if args.graph_weight > 0 and args.vae_checkpoint:
        from graph_cad.training.vae_trainer import load_checkpoint
        print(f"\nLoading VAE from {args.vae_checkpoint}...")
        vae, _ = load_checkpoint(args.vae_checkpoint, device=device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

    # Create optimizer
    optimizer = torch.optim.AdamW(
        editor.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    num_warmup_steps = min(args.warmup_steps, num_training_steps // 10)

    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        print(f"\nResuming from {args.resume}...")
        from graph_cad.training.edit_trainer import load_editor_checkpoint
        checkpoint = load_editor_checkpoint(
            editor, args.resume, device=device,
            load_optimizer=True, optimizer=optimizer
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["metrics"].get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")

    results = {"train": [], "val": []}

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        if is_paired:
            train_metrics = train_epoch_paired(
                editor=editor,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                config=loss_config,
                gradient_accumulation_steps=args.gradient_accumulation,
                max_grad_norm=args.max_grad_norm,
            )
        elif use_direction:
            train_metrics = train_epoch_with_direction(
                editor=editor,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                config=loss_config,
                vae=vae,
                gradient_accumulation_steps=args.gradient_accumulation,
                max_grad_norm=args.max_grad_norm,
            )
        else:
            train_metrics = train_epoch(
                editor=editor,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                config=loss_config,
                vae=vae,
                gradient_accumulation_steps=args.gradient_accumulation,
                max_grad_norm=args.max_grad_norm,
            )
        scheduler.step()

        if is_paired:
            print(f"\nTrain - Loss: {train_metrics['loss']:.6f}, "
                  f"Delta MSE: {train_metrics['delta_mse']:.6f}, "
                  f"Contrastive: {train_metrics['contrastive_loss']:.6f}, "
                  f"CosSim: {train_metrics['mean_cos_sim']:.3f}")
        elif use_direction:
            print(f"\nTrain - Loss: {train_metrics['loss']:.6f}, "
                  f"Delta MSE: {train_metrics['delta_mse']:.6f}, "
                  f"Dir Loss: {train_metrics['direction_loss']:.6f}, "
                  f"Dir Acc: {train_metrics['direction_accuracy']:.3f}")
        else:
            print(f"\nTrain - Loss: {train_metrics['loss']:.6f}, "
                  f"Delta MSE: {train_metrics['delta_mse']:.6f}")

        # Validate
        if is_paired:
            val_metrics = evaluate_paired(
                editor=editor,
                loader=val_loader,
                device=device,
                config=loss_config,
            )
            print(f"Val   - Loss: {val_metrics['loss']:.6f}, "
                  f"Delta MSE: {val_metrics['delta_mse']:.6f}, "
                  f"Delta MAE: {val_metrics['delta_mae']:.6f}, "
                  f"CosSim: {val_metrics['mean_cos_sim']:.3f}")
        elif use_direction:
            val_metrics = evaluate_with_direction(
                editor=editor,
                loader=val_loader,
                device=device,
                config=loss_config,
            )
            print(f"Val   - Loss: {val_metrics['loss']:.6f}, "
                  f"Delta MSE: {val_metrics['delta_mse']:.6f}, "
                  f"Delta MAE: {val_metrics['delta_mae']:.6f}, "
                  f"Dir Acc: {val_metrics['direction_accuracy']:.3f}")
        else:
            val_metrics = evaluate(
                editor=editor,
                loader=val_loader,
                device=device,
                config=loss_config,
                vae=vae,
            )
            print(f"Val   - Loss: {val_metrics['loss']:.6f}, "
                  f"Delta MSE: {val_metrics['delta_mse']:.6f}, "
                  f"Delta MAE: {val_metrics['delta_mae']:.6f}")

        # Track results
        results["train"].append({"epoch": epoch, **train_metrics})
        results["val"].append({"epoch": epoch, **val_metrics})

        # Save checkpoint
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            print(f"  New best validation loss!")

        if epoch % args.save_every == 0 or is_best:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_editor_checkpoint(
                editor=editor,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics, "val_loss": val_metrics["loss"]},
                path=str(checkpoint_path),
            )
            print(f"  Saved checkpoint to {checkpoint_path}")

            if is_best:
                best_path = output_dir / "best_model.pt"
                save_editor_checkpoint(
                    editor=editor,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"train": train_metrics, "val": val_metrics, "val_loss": val_metrics["loss"]},
                    path=str(best_path),
                )

    # Save final results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Results saved to: {results_path}")
    print(f"  Best model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
