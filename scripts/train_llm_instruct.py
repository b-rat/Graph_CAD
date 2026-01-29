#!/usr/bin/env python3
"""
Train the Extended LLM for instruction following (Stage 2).

Stage 2 of LLM training: Learn to edit geometry via natural language
instructions. Uses the full Mistral LLM with LoRA adapters.

Usage:
    python scripts/train_llm_instruct.py \
        --vae-checkpoint outputs/hetero_vae/best_model.pt \
        --llm-checkpoint outputs/llm_pretrain/best_model.pt \
        --epochs 30
"""

from __future__ import annotations

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from graph_cad.data.multi_geometry_dataset import MultiGeometryDataset
from graph_cad.data.brep_types import GEOMETRY_TYPE_NAMES, GEOMETRY_PARAM_COUNTS
from graph_cad.data.param_normalization import PARAM_NAMES
from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig
from graph_cad.models.extended_latent_editor import (
    ExtendedLatentEditor,
    ExtendedLatentEditorConfig,
)
from graph_cad.models.latent_editor import load_llm_with_lora


# Instruction templates for each parameter
INSTRUCTION_TEMPLATES = {
    # Length/dimension changes
    "increase_dim": [
        "make {param} +{delta}mm larger",
        "increase {param} by +{delta}mm",
        "add +{delta}mm to {param}",
        "{param} should be +{delta}mm bigger",
    ],
    "decrease_dim": [
        "make {param} -{delta}mm smaller",
        "decrease {param} by -{delta}mm",
        "reduce {param} by -{delta}mm",
        "{param} should be -{delta}mm less",
    ],
}

# Parameter names for instructions
PARAM_FRIENDLY_NAMES = {
    "leg1_length": ["leg1", "the first leg", "leg 1 length"],
    "leg2_length": ["leg2", "the second leg", "leg 2 length"],
    "width": ["width", "the width"],
    "thickness": ["thickness", "the thickness"],
    "length": ["length", "the length"],
    "outer_dia": ["outer diameter", "the outer diameter"],
    "inner_dia": ["inner diameter", "the inner diameter"],
    "diameter": ["diameter", "the diameter"],
    "height": ["height", "the height"],
    "hole_dia": ["hole diameter", "the hole diameter"],
    "hole_x": ["hole x position", "the hole x offset"],
    "hole_y": ["hole y position", "the hole y offset"],
}


def generate_instruction_and_delta(
    geometry_type: int,
    params_normalized: torch.Tensor,
    rng: random.Random,
    delta_range: tuple[float, float] = (0.05, 0.3),
) -> tuple[str, torch.Tensor]:
    """
    Generate a random instruction and the corresponding parameter delta.

    Args:
        geometry_type: Geometry type ID
        params_normalized: Normalized parameters [0, 1], shape (num_params,)
        rng: Random number generator
        delta_range: Range for normalized parameter change (fraction of range)

    Returns:
        instruction: Generated instruction string
        delta_params: Parameter deltas (padded to max_params)
    """
    param_names = PARAM_NAMES[geometry_type]
    num_params = len(param_names)

    # Choose a random parameter to edit
    param_idx = rng.randint(0, num_params - 1)
    param_name = param_names[param_idx]

    # Generate delta (in normalized space)
    delta_magnitude = rng.uniform(*delta_range)
    direction = rng.choice([-1, 1])

    # Ensure delta doesn't push value outside [0, 1]
    current_val = params_normalized[param_idx].item()
    if direction > 0:
        max_delta = 1.0 - current_val
        delta = min(delta_magnitude, max_delta)
    else:
        max_delta = current_val
        delta = -min(delta_magnitude, max_delta)

    # Convert to mm for instruction (approximate)
    # Using rough scale: normalized delta of 0.1 = ~10-15mm depending on param
    delta_mm = abs(delta) * 100  # Rough approximation

    # Generate instruction
    if delta > 0:
        template = rng.choice(INSTRUCTION_TEMPLATES["increase_dim"])
    else:
        template = rng.choice(INSTRUCTION_TEMPLATES["decrease_dim"])

    friendly_name = rng.choice(PARAM_FRIENDLY_NAMES.get(param_name, [param_name]))
    instruction = template.format(param=friendly_name, delta=int(delta_mm))

    # Create delta tensor (padded)
    delta_params = torch.zeros(6)  # MAX_PARAMS
    delta_params[param_idx] = delta

    return instruction, delta_params


def load_vae(checkpoint_path: str, device: str) -> HeteroVAE:
    """Load pre-trained VAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get('config', {})
    config = HeteroVAEConfig(**config_dict)

    model = HeteroVAE(config, use_param_head=True, num_params=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Freeze VAE
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_pretrained_llm(
    checkpoint_path: str,
    device: str,
    load_mistral: bool = True,
) -> ExtendedLatentEditor:
    """Load pre-trained LLM from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = ExtendedLatentEditorConfig(
        latent_dim=checkpoint['config']['latent_dim'],
        llm_hidden_dim=checkpoint['config']['llm_hidden_dim'],
        class_hidden_dim=checkpoint['config']['class_hidden_dim'],
        param_hidden_dim=checkpoint['config']['param_hidden_dim'],
        training_mode="instruct",
    )

    llm = ExtendedLatentEditor(config)

    # Load pre-trained weights (encoder + heads)
    # Filter out LLM-related keys if they exist
    state_dict = checkpoint['model_state_dict']
    compatible_state = {k: v for k, v in state_dict.items() if not k.startswith('llm.')}
    llm.load_state_dict(compatible_state, strict=False)

    # Load Mistral with LoRA
    if load_mistral:
        print("  Loading Mistral 7B with LoRA...")
        mistral, tokenizer = load_llm_with_lora(config, device_map="auto")
        llm.set_llm(mistral, tokenizer)
        # Move non-LLM components to device (LLM already on device via device_map)
        llm.latent_projector = llm.latent_projector.to(device)
        llm.output_projector = llm.output_projector.to(device)
        llm.class_head = llm.class_head.to(device)
        llm.param_heads = llm.param_heads.to(device)
        llm.pretrain_encoder = llm.pretrain_encoder.to(device)
    else:
        llm = llm.to(device)
    return llm


@torch.no_grad()
def encode_batch(vae: HeteroVAE, batch, device: str) -> torch.Tensor:
    """Encode batch of graphs to latent vectors."""
    batch = batch.to(device)
    mu, _ = vae.encode(batch)
    return mu


def train_epoch(
    llm: ExtendedLatentEditor,
    vae: HeteroVAE,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    rng: random.Random,
    class_weight: float = 0.5,
    param_weight: float = 1.0,
    delta_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    llm.train()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in train_loader:
        # Encode to latent
        z = encode_batch(vae, batch, device)
        batch_size = z.shape[0]

        # Get geometry types and params
        geometry_types = batch.geometry_type.squeeze(-1).to(device)
        params_normalized = batch.params_normalized.to(device)
        params_mask = batch.params_mask.to(device)

        # Generate instructions and deltas
        instructions = []
        delta_params_list = []

        for i in range(batch_size):
            geo_type = geometry_types[i].item()
            params = params_normalized[i]

            instruction, delta = generate_instruction_and_delta(geo_type, params, rng)
            instructions.append(instruction)
            delta_params_list.append(delta)

        delta_params = torch.stack(delta_params_list).to(device)

        # Target params after edit
        target_params = params_normalized + delta_params

        targets = {
            'geometry_type': geometry_types,
            'params_normalized': delta_params,  # In instruct mode, predict deltas
            'params_mask': params_mask,
        }

        # Forward pass (instruct mode)
        outputs = llm.forward_instruct(z, instructions, geometry_types)

        # Compute loss
        loss = torch.tensor(0.0, device=device)
        metrics = {}

        # Classification loss (should maintain correct type)
        class_loss = torch.nn.functional.cross_entropy(
            outputs['class_logits'], geometry_types.long()
        )
        loss = loss + class_weight * class_loss
        metrics['class_loss'] = class_loss.item()

        with torch.no_grad():
            pred_types = outputs['class_logits'].argmax(dim=-1)
            metrics['class_acc'] = (pred_types == geometry_types).float().mean().item()

        # Parameter delta loss
        se = (outputs['param_pred'] - delta_params) ** 2
        num_valid = params_mask.sum().clamp(min=1)
        param_loss = (se * params_mask).sum() / num_valid
        loss = loss + param_weight * param_loss
        metrics['param_loss'] = param_loss.item()

        with torch.no_grad():
            abs_errors = (outputs['param_pred'] - delta_params).abs()
            metrics['param_mae'] = (abs_errors * params_mask).sum().item() / num_valid.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for key, value in metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


@torch.no_grad()
def evaluate(
    llm: ExtendedLatentEditor,
    vae: HeteroVAE,
    val_loader,
    device: str,
    rng: random.Random,
    class_weight: float = 0.5,
    param_weight: float = 1.0,
) -> dict[str, float]:
    """Evaluate on validation set."""
    llm.eval()

    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    for batch in val_loader:
        z = encode_batch(vae, batch, device)
        batch_size = z.shape[0]

        geometry_types = batch.geometry_type.squeeze(-1).to(device)
        params_normalized = batch.params_normalized.to(device)
        params_mask = batch.params_mask.to(device)

        # Generate instructions
        instructions = []
        delta_params_list = []

        for i in range(batch_size):
            geo_type = geometry_types[i].item()
            params = params_normalized[i]

            instruction, delta = generate_instruction_and_delta(geo_type, params, rng)
            instructions.append(instruction)
            delta_params_list.append(delta)

        delta_params = torch.stack(delta_params_list).to(device)

        outputs = llm.forward_instruct(z, instructions, geometry_types)

        # Compute metrics
        class_loss = torch.nn.functional.cross_entropy(
            outputs['class_logits'], geometry_types.long()
        )

        se = (outputs['param_pred'] - delta_params) ** 2
        num_valid = params_mask.sum().clamp(min=1)
        param_loss = (se * params_mask).sum() / num_valid

        loss = class_weight * class_loss + param_weight * param_loss

        pred_types = outputs['class_logits'].argmax(dim=-1)
        abs_errors = (outputs['param_pred'] - delta_params).abs()

        total_loss += loss.item()
        total_metrics['class_loss'] = total_metrics.get('class_loss', 0) + class_loss.item()
        total_metrics['class_acc'] = total_metrics.get('class_acc', 0) + (pred_types == geometry_types).float().mean().item()
        total_metrics['param_loss'] = total_metrics.get('param_loss', 0) + param_loss.item()
        total_metrics['param_mae'] = total_metrics.get('param_mae', 0) + (abs_errors * params_mask).sum().item() / num_valid.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


def save_checkpoint(
    llm: ExtendedLatentEditor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    vae_checkpoint: str,
    path: str,
):
    """Save model checkpoint."""
    # Save only non-LLM weights (LoRA weights are saved separately)
    state_dict = {k: v for k, v in llm.state_dict().items() if not k.startswith('llm.')}

    torch.save({
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "vae_checkpoint": vae_checkpoint,
    }, path)

    # Save LoRA adapter separately
    if llm.llm is not None:
        lora_path = path.replace('.pt', '_lora')
        llm.llm.save_pretrained(lora_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Extended LLM for instruction following (Stage 2)"
    )

    # Checkpoints
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                        help="Path to pre-trained VAE checkpoint")
    parser.add_argument("--llm-checkpoint", type=str, required=True,
                        help="Path to pre-trained LLM checkpoint (Stage 1)")

    # Data arguments
    parser.add_argument("--samples-per-type", type=int, default=2000,
                        help="Training samples per geometry type")
    parser.add_argument("--val-samples-per-type", type=int, default=200,
                        help="Validation samples per geometry type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (smaller for LLM)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Loss weights
    parser.add_argument("--class-weight", type=float, default=0.5, help="Classification loss weight")
    parser.add_argument("--param-weight", type=float, default=1.0, help="Parameter loss weight")

    # Output arguments
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/llm_instruct"),
                        help="Output directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")

    # Device
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip loading Mistral (for testing)")

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen VAE
    print(f"\nLoading frozen VAE from: {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)

    # Load pre-trained LLM
    print(f"\nLoading pre-trained LLM from: {args.llm_checkpoint}")
    llm = load_pretrained_llm(args.llm_checkpoint, device, load_mistral=not args.skip_llm)

    # Create data loaders
    print(f"\nCreating multi-geometry datasets...")
    from torch_geometric.loader import DataLoader

    train_dataset = MultiGeometryDataset(
        num_samples_per_type=args.samples_per_type,
        seed=args.seed,
    )
    val_dataset = MultiGeometryDataset(
        num_samples_per_type=args.val_samples_per_type,
        seed=args.seed + 100000,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(
        llm.get_trainable_parameters("instruct"),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    trainable_params = llm.num_trainable_params("instruct")
    print(f"\nModel: Extended LLM (instruct mode)")
    print(f"  Trainable params: {trainable_params:,}")

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 100)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_epoch(
            llm, vae, train_loader, optimizer, device, rng,
            args.class_weight, args.param_weight
        )

        val_metrics = evaluate(
            llm, vae, val_loader, device, rng,
            args.class_weight, args.param_weight
        )

        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Log
        epoch_time = time.time() - epoch_start
        log_str = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"ClassAcc: {val_metrics['class_acc']:.1%} | "
            f"ParamMAE: {val_metrics['param_mae']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(log_str)

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                llm, optimizer, epoch, val_metrics,
                {"seed": args.seed},
                args.vae_checkpoint,
                str(args.output_dir / "best_model.pt"),
            )

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(
                llm, optimizer, epoch, val_metrics,
                {"seed": args.seed},
                args.vae_checkpoint,
                str(args.output_dir / f"checkpoint_epoch_{epoch}.pt"),
            )

    print("-" * 100)
    print("\nTraining complete!")

    # Save results
    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "history": history,
    }

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
