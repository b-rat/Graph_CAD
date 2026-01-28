"""
Extended LLM-based latent editor for Phase 4 multi-geometry support.

Extends the base LatentEditor with:
- Geometry type classification head
- Per-type parameter regression heads
- Two-stage training support (pre-training + instruction following)

The extended editor can:
1. Classify which geometry type a latent represents
2. Predict parameters from the latent (per-type regression)
3. Edit geometry via natural language instructions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_cad.data.brep_types import (
    NUM_GEOMETRY_TYPES,
    MAX_PARAMS,
    GEOMETRY_PARAM_COUNTS,
    GEOMETRY_TYPE_NAMES,
)
from graph_cad.models.latent_editor import (
    LatentEditorConfig,
    LatentProjector,
    OutputProjector,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class ExtendedLatentEditorConfig(LatentEditorConfig):
    """Configuration for extended latent editor with multi-geometry support."""

    # Multi-geometry settings
    num_geometry_types: int = NUM_GEOMETRY_TYPES
    max_params: int = MAX_PARAMS

    # Classification head architecture
    class_hidden_dim: int = 256

    # Parameter regression head architecture
    param_hidden_dim: int = 128

    # Training mode
    # "pretrain": latent -> class + params (no text)
    # "instruct": latent + text -> class + delta_params
    training_mode: str = "pretrain"


class GeometryClassificationHead(nn.Module):
    """
    Classifies geometry type from latent or LLM hidden state.

    Used in both pre-training (from latent directly) and instruction
    following (from LLM hidden state).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_types: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_types),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify geometry type.

        Args:
            x: Input features, shape (batch, input_dim)

        Returns:
            Logits, shape (batch, num_types)
        """
        return self.head(x)


class PerTypeParamHead(nn.Module):
    """
    Parameter regression head for a single geometry type.

    Each geometry type has its own head that outputs the correct
    number of parameters for that type.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_params: int):
        super().__init__()
        self.num_params = num_params
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict parameters.

        Args:
            x: Input features, shape (batch, input_dim)

        Returns:
            Parameters, shape (batch, num_params)
        """
        return self.head(x)


class MultiTypeParamHeads(nn.Module):
    """
    Collection of per-type parameter regression heads.

    Routes input to the appropriate head based on geometry type
    and produces padded output with mask.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_params: int = MAX_PARAMS,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_params = max_params

        # Create per-type heads
        self.heads = nn.ModuleDict()
        for geo_type, name in GEOMETRY_TYPE_NAMES.items():
            num_params = GEOMETRY_PARAM_COUNTS[geo_type]
            self.heads[name] = PerTypeParamHead(input_dim, hidden_dim, num_params)

        self.type_names = list(GEOMETRY_TYPE_NAMES.values())

    def forward(
        self,
        x: torch.Tensor,
        geometry_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict parameters using type-specific heads.

        Args:
            x: Input features, shape (batch, input_dim)
            geometry_types: Geometry type IDs, shape (batch,)

        Returns:
            params: Predicted parameters (padded), shape (batch, max_params)
            mask: Parameter mask, shape (batch, max_params)
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        params = torch.zeros(batch_size, self.max_params, device=device, dtype=dtype)
        masks = torch.zeros(batch_size, self.max_params, device=device, dtype=dtype)

        for geo_type, name in enumerate(self.type_names):
            # Find samples of this type
            type_mask = (geometry_types == geo_type)
            if not type_mask.any():
                continue

            # Get predictions from type-specific head
            x_type = x[type_mask]
            pred = self.heads[name](x_type)
            num_params = pred.shape[1]

            # Fill padded output
            params[type_mask, :num_params] = pred
            masks[type_mask, :num_params] = 1.0

        return params, masks

    def forward_all_types(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from all heads (for analysis/debugging).

        Args:
            x: Input features, shape (batch, input_dim)

        Returns:
            all_params: Shape (batch, num_types, max_params)
            all_masks: Shape (batch, num_types, max_params)
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        num_types = len(self.type_names)

        all_params = torch.zeros(
            batch_size, num_types, self.max_params, device=device, dtype=dtype
        )
        all_masks = torch.zeros(
            batch_size, num_types, self.max_params, device=device, dtype=dtype
        )

        for geo_type, name in enumerate(self.type_names):
            pred = self.heads[name](x)
            num_params = pred.shape[1]
            all_params[:, geo_type, :num_params] = pred
            all_masks[:, geo_type, :num_params] = 1.0

        return all_params, all_masks


class ExtendedLatentEditor(nn.Module):
    """
    Extended LLM-based latent editor with multi-geometry support.

    Supports two training stages:

    1. Pre-training (latent -> class + params):
       - Input: latent z from frozen VAE
       - Output: geometry class + normalized parameters
       - No LLM involvement, trains projection layers

    2. Instruction following (latent + text -> class + delta_params):
       - Input: latent z + instruction text
       - Output: geometry class + parameter deltas
       - Full LLM with LoRA

    Architecture:
        - LatentProjector: z (32D) -> LLM embedding (4096D)
        - LLM (Mistral 7B + LoRA): processes concatenated [latent, text]
        - GeometryClassificationHead: hidden -> class logits
        - MultiTypeParamHeads: hidden -> per-type parameters
        - OutputProjector: hidden -> latent delta (for instruction mode)
    """

    def __init__(
        self,
        config: ExtendedLatentEditorConfig | None = None,
        llm: "PreTrainedModel | None" = None,
        tokenizer: "PreTrainedTokenizer | None" = None,
    ):
        super().__init__()
        self.config = config or ExtendedLatentEditorConfig()

        # Latent projector (z -> LLM embedding space)
        self.latent_projector = LatentProjector(self.config)

        # Output projector (LLM hidden -> latent delta, for instruction mode)
        self.output_projector = OutputProjector(self.config)

        # Classification head
        self.class_head = GeometryClassificationHead(
            input_dim=self.config.llm_hidden_dim,
            hidden_dim=self.config.class_hidden_dim,
            num_types=self.config.num_geometry_types,
        )

        # Parameter regression heads (one per geometry type)
        self.param_heads = MultiTypeParamHeads(
            input_dim=self.config.llm_hidden_dim,
            hidden_dim=self.config.param_hidden_dim,
            max_params=self.config.max_params,
        )

        # Pre-training mode uses a simpler encoder instead of full LLM
        # This maps latent directly to the "hidden state" space
        self.pretrain_encoder = nn.Sequential(
            nn.Linear(self.config.latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.config.llm_hidden_dim),
        )

        # LLM and tokenizer (loaded separately)
        self.llm = llm
        self.tokenizer = tokenizer

    def set_llm(
        self,
        llm: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ) -> None:
        """Set the LLM and tokenizer."""
        self.llm = llm
        self.tokenizer = tokenizer

    def forward_pretrain(
        self,
        z: torch.Tensor,
        geometry_types: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Pre-training forward pass: latent -> class + params.

        No LLM involvement. Uses simple encoder to map latent to
        hidden space, then applies classification and regression heads.

        Args:
            z: Latent vectors, shape (batch, latent_dim)
            geometry_types: Ground truth geometry types, shape (batch,)

        Returns:
            Dict with class_logits, param_pred, param_mask
        """
        # Encode latent to hidden space
        hidden = self.pretrain_encoder(z)

        # Classification
        class_logits = self.class_head(hidden)

        # Parameter regression (using ground truth types for routing)
        param_pred, param_mask = self.param_heads(hidden, geometry_types)

        return {
            'class_logits': class_logits,
            'param_pred': param_pred,
            'param_mask': param_mask,
            'hidden': hidden,
        }

    def forward_instruct(
        self,
        z: torch.Tensor,
        instructions: list[str],
        geometry_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Instruction following forward pass: latent + text -> class + delta.

        Uses full LLM to process concatenated latent embedding and text.

        Args:
            z: Source latent vectors, shape (batch, latent_dim)
            instructions: List of instruction strings
            geometry_types: Optional ground truth types for parameter routing.
                If None, uses predicted class.

        Returns:
            Dict with class_logits, param_pred, delta_z, z_edited
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("LLM and tokenizer must be set for instruction mode")

        batch_size = z.shape[0]
        device = z.device

        # Project latent to embedding space
        latent_embed = self.latent_projector(z)  # (batch, 1, hidden_dim)

        # Tokenize instructions
        tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(device)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(tokens["input_ids"])

        # Concatenate: [latent_embed, text_embeds]
        combined_embeds = torch.cat([latent_embed, text_embeds], dim=1)

        # Create attention mask
        latent_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        combined_mask = torch.cat([latent_mask, tokens["attention_mask"]], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract last hidden state at final position
        last_hidden = outputs.hidden_states[-1]
        seq_lengths = combined_mask.sum(dim=1) - 1
        hidden = last_hidden[
            torch.arange(batch_size, device=device), seq_lengths
        ]

        # Classification
        class_logits = self.class_head(hidden)

        # Use predicted class if ground truth not provided
        if geometry_types is None:
            geometry_types = class_logits.argmax(dim=-1)

        # Parameter regression (predicts delta params in instruct mode)
        param_delta, param_mask = self.param_heads(hidden, geometry_types)

        # Latent delta (for residual editing)
        delta_z = self.output_projector(hidden)
        z_edited = z + delta_z

        return {
            'class_logits': class_logits,
            'param_pred': param_delta,  # These are deltas in instruct mode
            'param_mask': param_mask,
            'delta_z': delta_z,
            'z_edited': z_edited,
            'hidden': hidden,
        }

    def forward(
        self,
        z: torch.Tensor,
        geometry_types: torch.Tensor | None = None,
        instructions: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Unified forward pass that dispatches based on mode.

        Args:
            z: Latent vectors, shape (batch, latent_dim)
            geometry_types: Ground truth geometry types (required for pretrain)
            instructions: List of instruction strings (required for instruct)

        Returns:
            Dict with outputs depending on mode
        """
        if self.config.training_mode == "pretrain":
            if geometry_types is None:
                raise ValueError("geometry_types required for pretrain mode")
            return self.forward_pretrain(z, geometry_types)
        else:  # instruct
            if instructions is None:
                raise ValueError("instructions required for instruct mode")
            return self.forward_instruct(z, instructions, geometry_types)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Classify geometry type from latent.

        Args:
            z: Latent vectors, shape (batch, latent_dim) or (latent_dim,)

        Returns:
            Predicted geometry type IDs
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        with torch.no_grad():
            hidden = self.pretrain_encoder(z)
            logits = self.class_head(hidden)
            pred_types = logits.argmax(dim=-1)

        if squeeze:
            pred_types = pred_types.squeeze(0)

        return pred_types

    def predict_params(
        self,
        z: torch.Tensor,
        geometry_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict parameters from latent.

        Args:
            z: Latent vectors, shape (batch, latent_dim) or (latent_dim,)
            geometry_types: Optional geometry types. If None, classifies first.

        Returns:
            params: Predicted parameters (padded)
            mask: Parameter mask
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        with torch.no_grad():
            hidden = self.pretrain_encoder(z)

            if geometry_types is None:
                logits = self.class_head(hidden)
                geometry_types = logits.argmax(dim=-1)

            params, mask = self.param_heads(hidden, geometry_types)

        if squeeze:
            params = params.squeeze(0)
            mask = mask.squeeze(0)

        return params, mask

    def edit(
        self,
        z: torch.Tensor,
        instruction: str,
    ) -> torch.Tensor:
        """
        Edit latent via instruction.

        Args:
            z: Source latent, shape (latent_dim,) or (1, latent_dim)
            instruction: Edit instruction string

        Returns:
            Edited latent
        """
        squeeze = z.dim() == 1
        if squeeze:
            z = z.unsqueeze(0)

        with torch.no_grad():
            result = self.forward_instruct(z, [instruction])

        z_edited = result['z_edited']
        if squeeze:
            z_edited = z_edited.squeeze(0)

        return z_edited

    def get_trainable_parameters(self, mode: str | None = None) -> list[nn.Parameter]:
        """
        Get trainable parameters for specified mode.

        Args:
            mode: "pretrain", "instruct", or None (use config.training_mode)

        Returns:
            List of trainable parameters
        """
        if mode is None:
            mode = self.config.training_mode

        params = []

        if mode == "pretrain":
            # Pre-training: encoder + heads
            params.extend(self.pretrain_encoder.parameters())
            params.extend(self.class_head.parameters())
            params.extend(self.param_heads.parameters())
        else:  # instruct
            # Instruction: projectors + heads + LoRA
            params.extend(self.latent_projector.parameters())
            params.extend(self.output_projector.parameters())
            params.extend(self.class_head.parameters())
            params.extend(self.param_heads.parameters())

            # LoRA parameters from LLM
            if self.llm is not None:
                for name, param in self.llm.named_parameters():
                    if param.requires_grad:
                        params.append(param)

        return params

    def num_trainable_params(self, mode: str | None = None) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters(mode))


def compute_extended_editor_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weight: float = 1.0,
    param_weight: float = 1.0,
    delta_weight: float = 0.0,  # Only for instruct mode
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute loss for extended latent editor.

    Args:
        outputs: Model outputs from forward pass
        targets: Dict with:
            - geometry_type: (batch,)
            - params_normalized: (batch, max_params)
            - params_mask: (batch, max_params)
            - delta_z: (batch, latent_dim) [optional, for instruct mode]
        class_weight: Weight for classification loss
        param_weight: Weight for parameter regression loss
        delta_weight: Weight for latent delta loss (instruct mode)

    Returns:
        total_loss: Combined loss
        loss_dict: All components for logging
    """
    # Classification loss
    class_loss = F.cross_entropy(
        outputs['class_logits'],
        targets['geometry_type'].long()
    )

    # Classification accuracy
    with torch.no_grad():
        pred_types = outputs['class_logits'].argmax(dim=-1)
        class_acc = (pred_types == targets['geometry_type']).float().mean()

    # Parameter regression loss (masked)
    param_pred = outputs['param_pred']
    param_target = targets['params_normalized']
    param_mask = targets['params_mask']

    se = (param_pred - param_target) ** 2
    num_valid = param_mask.sum().clamp(min=1)
    param_loss = (se * param_mask).sum() / num_valid

    # Parameter MAE for logging
    with torch.no_grad():
        abs_errors = (param_pred - param_target).abs()
        param_mae = (abs_errors * param_mask).sum() / num_valid

    # Combined loss
    total_loss = class_weight * class_loss + param_weight * param_loss

    loss_dict = {
        'class_loss': class_loss.detach(),
        'class_acc': class_acc,
        'param_loss': param_loss.detach(),
        'param_mae': param_mae,
    }

    # Latent delta loss (for instruct mode)
    if delta_weight > 0 and 'delta_z' in outputs and 'delta_z' in targets:
        delta_loss = F.mse_loss(outputs['delta_z'], targets['delta_z'])
        total_loss = total_loss + delta_weight * delta_loss
        loss_dict['delta_loss'] = delta_loss.detach()

    loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict


def create_extended_latent_editor(
    latent_dim: int = 32,
    training_mode: str = "pretrain",
    load_llm: bool = False,
    device_map: str = "auto",
) -> ExtendedLatentEditor:
    """
    Factory function to create ExtendedLatentEditor.

    Args:
        latent_dim: Latent space dimension
        training_mode: "pretrain" or "instruct"
        load_llm: Whether to load LLM (only needed for instruct mode)
        device_map: Device placement for LLM

    Returns:
        Configured ExtendedLatentEditor
    """
    config = ExtendedLatentEditorConfig(
        latent_dim=latent_dim,
        training_mode=training_mode,
    )

    editor = ExtendedLatentEditor(config)

    if load_llm:
        from graph_cad.models.latent_editor import load_llm_with_lora
        llm, tokenizer = load_llm_with_lora(config, device_map)
        editor.set_llm(llm, tokenizer)

    return editor
