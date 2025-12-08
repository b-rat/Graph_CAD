"""
Training utilities for the latent editor.

Includes loss functions, training loop, and evaluation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from graph_cad.models import GraphVAE, LatentEditor


@dataclass
class EditLossConfig:
    """Configuration for latent edit loss computation."""

    # Loss weights
    delta_weight: float = 1.0  # Primary: MSE on latent delta
    graph_weight: float = 0.0  # Secondary: MSE on decoded graphs
    l1_weight: float = 0.0  # Optional: L1 regularization on delta

    # Normalization
    normalize_delta: bool = False  # Normalize delta by magnitude


def latent_delta_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    config: EditLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute loss on predicted latent delta.

    Args:
        predicted_delta: Model's predicted delta, shape (batch, latent_dim)
        target_delta: Ground truth delta, shape (batch, latent_dim)
        config: Loss configuration

    Returns:
        (total_loss, loss_dict) with individual loss components
    """
    if config is None:
        config = EditLossConfig()

    loss_dict = {}

    # Primary: MSE loss on delta
    mse_loss = F.mse_loss(predicted_delta, target_delta)
    loss_dict["delta_mse"] = mse_loss

    # Optional: L1 regularization
    if config.l1_weight > 0:
        l1_loss = predicted_delta.abs().mean()
        loss_dict["delta_l1"] = l1_loss
    else:
        l1_loss = torch.tensor(0.0, device=predicted_delta.device)
        loss_dict["delta_l1"] = l1_loss

    # Combine losses
    total_loss = config.delta_weight * mse_loss + config.l1_weight * l1_loss
    loss_dict["total_loss"] = total_loss

    return total_loss, loss_dict


def graph_reconstruction_loss(
    vae_decoder: torch.nn.Module,
    z_src: torch.Tensor,
    predicted_delta: torch.Tensor,
    z_tgt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute loss on decoded graph features.

    Ensures the edited latent decodes to similar graph as target latent.

    Args:
        vae_decoder: VAE decoder module
        z_src: Source latent, shape (batch, latent_dim)
        predicted_delta: Predicted delta, shape (batch, latent_dim)
        z_tgt: Target latent, shape (batch, latent_dim)

    Returns:
        (loss, loss_dict) with graph reconstruction components
    """
    z_pred = z_src + predicted_delta

    # Decode both latents
    node_pred, edge_pred = vae_decoder(z_pred)
    node_tgt, edge_tgt = vae_decoder(z_tgt)

    # Compute MSE
    node_loss = F.mse_loss(node_pred, node_tgt)
    edge_loss = F.mse_loss(edge_pred, edge_tgt)
    total = node_loss + edge_loss

    return total, {
        "graph_node_mse": node_loss,
        "graph_edge_mse": edge_loss,
        "graph_total": total,
    }


def edit_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    z_src: torch.Tensor | None = None,
    z_tgt: torch.Tensor | None = None,
    vae: GraphVAE | None = None,
    config: EditLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Combined loss function for latent editing.

    Args:
        predicted_delta: Model's predicted delta
        target_delta: Ground truth delta
        z_src: Source latent (for graph reconstruction loss)
        z_tgt: Target latent (for graph reconstruction loss)
        vae: VAE model (for graph reconstruction loss)
        config: Loss configuration

    Returns:
        (total_loss, loss_dict)
    """
    if config is None:
        config = EditLossConfig()

    # Primary delta loss
    total_loss, loss_dict = latent_delta_loss(predicted_delta, target_delta, config)

    # Optional graph reconstruction loss
    if config.graph_weight > 0 and vae is not None and z_src is not None and z_tgt is not None:
        graph_loss, graph_dict = graph_reconstruction_loss(
            vae.decoder, z_src, predicted_delta, z_tgt
        )
        total_loss = total_loss + config.graph_weight * graph_loss
        loss_dict.update(graph_dict)
        loss_dict["total_loss"] = total_loss

    return total_loss, loss_dict


def train_epoch(
    editor: LatentEditor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    config: EditLossConfig | None = None,
    vae: GraphVAE | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    Train editor for one epoch.

    Args:
        editor: LatentEditor model
        loader: DataLoader for edit samples
        optimizer: Optimizer
        device: Device to train on
        config: Loss configuration
        vae: Optional VAE for graph reconstruction loss
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Dictionary with averaged metrics for the epoch
    """
    editor.train()
    if vae is not None:
        vae.eval()  # VAE is frozen

    total_loss = 0.0
    total_delta_mse = 0.0
    total_delta_l1 = 0.0
    total_graph_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        z_src = batch["z_src"].to(device)
        z_tgt = batch["z_tgt"].to(device)
        delta_z = batch["delta_z"].to(device)
        instructions = batch["instructions"]

        # Forward pass
        outputs = editor(z_src, instructions)
        predicted_delta = outputs["delta_z"]

        # Compute loss
        loss, loss_dict = edit_loss(
            predicted_delta=predicted_delta,
            target_delta=delta_z,
            z_src=z_src,
            z_tgt=z_tgt,
            vae=vae,
            config=config,
        )

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Accumulate metrics (unscaled)
        total_loss += loss_dict["total_loss"].item()
        total_delta_mse += loss_dict["delta_mse"].item()
        total_delta_l1 += loss_dict["delta_l1"].item()
        if "graph_total" in loss_dict:
            total_graph_loss += loss_dict["graph_total"].item()
        num_batches += 1

        # Update weights
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                editor.get_trainable_parameters(), max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if num_batches % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            editor.get_trainable_parameters(), max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "delta_mse": total_delta_mse / num_batches,
        "delta_l1": total_delta_l1 / num_batches,
    }
    if total_graph_loss > 0:
        metrics["graph_loss"] = total_graph_loss / num_batches

    return metrics


@torch.no_grad()
def evaluate(
    editor: LatentEditor,
    loader: DataLoader,
    device: str,
    config: EditLossConfig | None = None,
    vae: GraphVAE | None = None,
) -> dict[str, float]:
    """
    Evaluate editor on a data loader.

    Args:
        editor: LatentEditor model
        loader: DataLoader for edit samples
        device: Device to evaluate on
        config: Loss configuration
        vae: Optional VAE for graph reconstruction loss

    Returns:
        Dictionary with evaluation metrics
    """
    editor.eval()
    if vae is not None:
        vae.eval()

    total_loss = 0.0
    total_delta_mse = 0.0
    total_delta_l1 = 0.0
    total_graph_loss = 0.0

    # Additional metrics
    total_delta_mae = 0.0
    total_delta_norm_error = 0.0
    num_batches = 0

    for batch in loader:
        z_src = batch["z_src"].to(device)
        z_tgt = batch["z_tgt"].to(device)
        delta_z = batch["delta_z"].to(device)
        instructions = batch["instructions"]

        # Forward pass
        outputs = editor(z_src, instructions)
        predicted_delta = outputs["delta_z"]

        # Compute loss
        loss, loss_dict = edit_loss(
            predicted_delta=predicted_delta,
            target_delta=delta_z,
            z_src=z_src,
            z_tgt=z_tgt,
            vae=vae,
            config=config,
        )

        # Accumulate metrics
        total_loss += loss_dict["total_loss"].item()
        total_delta_mse += loss_dict["delta_mse"].item()
        total_delta_l1 += loss_dict["delta_l1"].item()
        if "graph_total" in loss_dict:
            total_graph_loss += loss_dict["graph_total"].item()

        # Additional metrics
        total_delta_mae += (predicted_delta - delta_z).abs().mean().item()
        total_delta_norm_error += (
            (predicted_delta.norm(dim=1) - delta_z.norm(dim=1)).abs().mean().item()
        )
        num_batches += 1

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "delta_mse": total_delta_mse / num_batches,
        "delta_l1": total_delta_l1 / num_batches,
        "delta_mae": total_delta_mae / num_batches,
        "delta_norm_error": total_delta_norm_error / num_batches,
    }
    if total_graph_loss > 0:
        metrics["graph_loss"] = total_graph_loss / num_batches

    return metrics


def save_editor_checkpoint(
    editor: LatentEditor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """
    Save editor checkpoint.

    Saves projectors and LoRA weights (not full LLM).
    """
    checkpoint = {
        "epoch": epoch,
        "config": editor.config,
        "latent_projector_state": editor.latent_projector.state_dict(),
        "output_projector_state": editor.output_projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save LoRA weights if available
    if editor.llm is not None:
        try:
            # peft models have save_pretrained, but we save state dict
            lora_state = {}
            for name, param in editor.llm.named_parameters():
                if param.requires_grad:
                    lora_state[name] = param.data.clone()
            checkpoint["lora_state"] = lora_state
        except Exception:
            pass  # Skip if LoRA not available

    torch.save(checkpoint, path)


def load_editor_checkpoint(
    editor: LatentEditor,
    path: str,
    device: str = "cpu",
    load_optimizer: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Load editor checkpoint.

    Args:
        editor: LatentEditor to load weights into
        path: Path to checkpoint
        device: Device to load to
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into

    Returns:
        Checkpoint dictionary with epoch, metrics, etc.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load projector weights
    editor.latent_projector.load_state_dict(checkpoint["latent_projector_state"])
    editor.output_projector.load_state_dict(checkpoint["output_projector_state"])

    # Load LoRA weights if available
    if "lora_state" in checkpoint and editor.llm is not None:
        for name, param in editor.llm.named_parameters():
            if name in checkpoint["lora_state"]:
                param.data.copy_(checkpoint["lora_state"][name])

    # Load optimizer state
    if load_optimizer and optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint
