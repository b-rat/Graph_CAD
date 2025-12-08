"""
LLM-based latent editor for CAD modifications.

Uses Mistral 7B with LoRA adapters to predict latent space deltas
based on natural language instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class LatentEditorConfig:
    """Configuration for the latent editor model."""

    # Model selection
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Latent space dimensions
    latent_dim: int = 16
    llm_hidden_dim: int = 4096  # Mistral hidden size

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Projector architecture
    projector_hidden_dims: tuple[int, ...] = (256, 512, 1024, 2048)
    projector_dropout: float = 0.1

    # Training
    max_seq_length: int = 128
    use_4bit: bool = True  # QLoRA quantization
    use_8bit: bool = False  # Alternative to 4-bit


class LatentProjector(nn.Module):
    """
    Projects latent vector to LLM embedding space.

    Maps 16D latent to 4096D LLM hidden dimension through MLP layers.
    """

    def __init__(self, config: LatentEditorConfig):
        super().__init__()
        self.config = config

        # Build MLP layers
        dims = [config.latent_dim] + list(config.projector_hidden_dims) + [config.llm_hidden_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout on last layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.projector_dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project latent to embedding space.

        Args:
            z: Latent vector, shape (batch_size, latent_dim)

        Returns:
            Embedding, shape (batch_size, 1, llm_hidden_dim)
        """
        # Project to LLM hidden dim
        embedding = self.mlp(z)  # (batch, llm_hidden_dim)
        # Add sequence dimension
        return embedding.unsqueeze(1)  # (batch, 1, llm_hidden_dim)


class OutputProjector(nn.Module):
    """
    Projects LLM hidden state back to latent delta.

    Maps 4096D LLM output to 16D latent delta through MLP layers.
    """

    def __init__(self, config: LatentEditorConfig):
        super().__init__()
        self.config = config

        # Build MLP layers (reverse of input projector)
        dims = [config.llm_hidden_dim] + list(reversed(config.projector_hidden_dims)) + [config.latent_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation/dropout on last layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(config.projector_dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state to latent delta.

        Args:
            hidden: LLM hidden state, shape (batch_size, llm_hidden_dim)

        Returns:
            Latent delta, shape (batch_size, latent_dim)
        """
        return self.mlp(hidden)


class LatentEditor(nn.Module):
    """
    LLM-based latent editor.

    Combines Mistral 7B (with LoRA) with input/output projectors
    to predict latent deltas from text instructions.

    Architecture:
        1. Project source latent to LLM embedding space
        2. Tokenize instruction text
        3. Concatenate latent embedding with text embeddings
        4. Forward through LLM with LoRA adapters
        5. Extract last hidden state
        6. Project to latent delta
        7. Apply residual: z_edited = z_src + delta
    """

    def __init__(
        self,
        config: LatentEditorConfig,
        llm: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        super().__init__()
        self.config = config

        # Projectors (always trainable)
        self.latent_projector = LatentProjector(config)
        self.output_projector = OutputProjector(config)

        # LLM and tokenizer (loaded separately or passed in)
        self.llm = llm
        self.tokenizer = tokenizer

    def set_llm(self, llm: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        """Set the LLM and tokenizer after initialization."""
        self.llm = llm
        self.tokenizer = tokenizer

    def forward(
        self,
        z_src: torch.Tensor,
        instructions: list[str],
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: instruction + source latent -> edited latent.

        Args:
            z_src: Source latent vectors, shape (batch_size, latent_dim)
            instructions: List of instruction strings
            return_hidden: If True, also return intermediate hidden states

        Returns:
            Dictionary with:
                - delta_z: Predicted latent delta (batch_size, latent_dim)
                - z_edited: Edited latent z_src + delta_z
                - hidden: (optional) Last hidden state before projection
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("LLM and tokenizer must be set before forward pass")

        batch_size = z_src.shape[0]
        device = z_src.device

        # 1. Project latent to embedding space
        latent_embed = self.latent_projector(z_src)  # (batch, 1, hidden_dim)

        # 2. Tokenize instructions
        tokens = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(device)

        # 3. Get text embeddings from LLM
        text_embeds = self.llm.get_input_embeddings()(tokens["input_ids"])  # (batch, seq, hidden)

        # 4. Concatenate: [latent_embed, text_embeds]
        combined_embeds = torch.cat([latent_embed, text_embeds], dim=1)

        # Create attention mask for combined sequence
        latent_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        combined_mask = torch.cat([latent_mask, tokens["attention_mask"]], dim=1)

        # 5. Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 6. Extract last hidden state (at final position)
        # Get the hidden state at the last non-padding position
        last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)

        # Find last non-padding position for each sample
        seq_lengths = combined_mask.sum(dim=1) - 1  # -1 for 0-indexing
        last_token_hidden = last_hidden[
            torch.arange(batch_size, device=device), seq_lengths
        ]  # (batch, hidden)

        # 7. Project to latent delta
        delta_z = self.output_projector(last_token_hidden)

        # 8. Apply residual
        z_edited = z_src + delta_z

        result = {
            "delta_z": delta_z,
            "z_edited": z_edited,
        }

        if return_hidden:
            result["hidden"] = last_token_hidden

        return result

    def edit(
        self,
        z_src: torch.Tensor,
        instruction: str,
    ) -> torch.Tensor:
        """
        Convenience method for single-sample editing.

        Args:
            z_src: Source latent, shape (latent_dim,) or (1, latent_dim)
            instruction: Edit instruction string

        Returns:
            Edited latent, same shape as input
        """
        squeeze = z_src.dim() == 1
        if squeeze:
            z_src = z_src.unsqueeze(0)

        with torch.no_grad():
            result = self.forward(z_src, [instruction])

        z_edited = result["z_edited"]
        if squeeze:
            z_edited = z_edited.squeeze(0)

        return z_edited

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Get list of trainable parameters (projectors + LoRA)."""
        params = []

        # Projector parameters (always trainable)
        params.extend(self.latent_projector.parameters())
        params.extend(self.output_projector.parameters())

        # LoRA parameters (if LLM is set and has LoRA)
        if self.llm is not None:
            for name, param in self.llm.named_parameters():
                if param.requires_grad:
                    params.append(param)

        return params

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())


def load_llm_with_lora(
    config: LatentEditorConfig,
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load Mistral with QLoRA/LoRA configuration.

    Args:
        config: Editor configuration
        device_map: Device placement strategy

    Returns:
        (model, tokenizer) tuple
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not config.use_4bit else None,
    )

    # Prepare for k-bit training if using quantization
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def create_latent_editor(
    config: LatentEditorConfig | None = None,
    load_llm: bool = True,
    device_map: str = "auto",
) -> LatentEditor:
    """
    Factory function to create a LatentEditor.

    Args:
        config: Editor configuration (uses defaults if None)
        load_llm: Whether to load the LLM immediately
        device_map: Device placement for LLM

    Returns:
        Initialized LatentEditor
    """
    if config is None:
        config = LatentEditorConfig()

    editor = LatentEditor(config)

    if load_llm:
        llm, tokenizer = load_llm_with_lora(config, device_map)
        editor.set_llm(llm, tokenizer)

    return editor
