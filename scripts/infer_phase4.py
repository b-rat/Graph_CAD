#!/usr/bin/env python3
"""
Phase 4 Inference Script for Multi-Geometry Latent Editor.

Takes a geometry (random or specified) and a natural language instruction,
runs inference through the full pipeline:
  1. Generate/load geometry and extract B-Rep graph
  2. Encode through HeteroVAE to get latent vector
  3. Apply instruction through ExtendedLatentEditor
  4. Predict edited parameters
  5. Optionally generate edited geometry

Usage:
    # Random geometry with instruction
    python scripts/infer_phase4.py \
        --random --geometry-type bracket \
        --instruction "make leg1 +20mm longer"

    # Test all geometry types
    python scripts/infer_phase4.py --test-all

    # VAE-only mode (skip LLM)
    python scripts/infer_phase4.py \
        --random --geometry-type tube \
        --vae-only
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data.brep_types import (
    GEOMETRY_TYPE_NAMES,
    GEOMETRY_PARAM_COUNTS,
)
from graph_cad.data.brep_extraction import extract_brep_hetero_graph_from_solid
from graph_cad.data.multi_geometry_dataset import brep_graph_to_hetero_data
from graph_cad.data.param_normalization import (
    PARAM_NAMES,
    denormalize_params,
    normalize_params,
)
from graph_cad.data.geometry_generators import (
    Tube, Channel, Block, Cylinder, BlockHole,
)
from graph_cad.data.l_bracket import VariableLBracket


GEOMETRY_CLASSES = {
    0: VariableLBracket,
    1: Tube,
    2: Channel,
    3: Block,
    4: Cylinder,
    5: BlockHole,
}


def create_random_geometry(geometry_type: int, seed: int | None = None):
    """Create a random geometry of the specified type."""
    rng = random.Random(seed)

    cls = GEOMETRY_CLASSES[geometry_type]
    if geometry_type == 0:  # VariableLBracket needs rng argument
        return cls.random(rng)
    else:
        return cls.random(rng)


def geometry_to_params(geometry, geometry_type: int) -> torch.Tensor:
    """Extract parameters from a geometry object."""
    if geometry_type == 0:  # Bracket
        return torch.tensor([
            geometry.leg1_length,
            geometry.leg2_length,
            geometry.width,
            geometry.thickness,
        ], dtype=torch.float32)
    elif geometry_type == 1:  # Tube
        return torch.tensor([
            geometry.length,
            geometry.outer_diameter,
            geometry.inner_diameter,
        ], dtype=torch.float32)
    elif geometry_type == 2:  # Channel
        return torch.tensor([
            geometry.width,
            geometry.height,
            geometry.length,
            geometry.thickness,
        ], dtype=torch.float32)
    elif geometry_type == 3:  # Block
        return torch.tensor([
            geometry.length,
            geometry.width,
            geometry.height,
        ], dtype=torch.float32)
    elif geometry_type == 4:  # Cylinder
        return torch.tensor([
            geometry.length,
            geometry.diameter,
        ], dtype=torch.float32)
    elif geometry_type == 5:  # BlockHole
        return torch.tensor([
            geometry.length,
            geometry.width,
            geometry.height,
            geometry.hole_diameter,
            geometry.hole_x_offset,
            geometry.hole_y_offset,
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")


def load_vae(checkpoint_path: str, device: str):
    """Load trained HeteroVAE model."""
    from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig

    print(f"Loading HeteroVAE from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = HeteroVAEConfig(**checkpoint["config"])
    model = HeteroVAE(config, use_param_head=True, num_params=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    return model


def load_llm(checkpoint_path: str, vae_config, device: str):
    """Load trained ExtendedLatentEditor."""
    from graph_cad.models.extended_latent_editor import (
        ExtendedLatentEditor,
        ExtendedLatentEditorConfig,
    )
    from graph_cad.models.latent_editor import load_llm_with_lora, LatentEditorConfig

    print(f"Loading ExtendedLatentEditor from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create config
    config = ExtendedLatentEditorConfig(
        latent_dim=vae_config.latent_dim,
        training_mode="instruct",
    )

    # Create model without LLM first
    model = ExtendedLatentEditor(config, llm=None, tokenizer=None)

    # Load state dict (excluding LLM weights)
    state_dict = checkpoint["model_state_dict"]
    compatible_state = {k: v for k, v in state_dict.items() if not k.startswith('llm.')}
    model.load_state_dict(compatible_state, strict=False)

    # Load Mistral with LoRA
    print("  Loading Mistral 7B with LoRA...")
    lora_config = LatentEditorConfig()
    mistral, tokenizer = load_llm_with_lora(lora_config, device_map="auto")
    model.set_llm(mistral, tokenizer)

    # Move non-LLM components to device
    model.latent_projector = model.latent_projector.to(device)
    model.output_projector = model.output_projector.to(device)
    model.class_head = model.class_head.to(device)
    model.param_heads = model.param_heads.to(device)
    model.pretrain_encoder = model.pretrain_encoder.to(device)

    model.eval()
    print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    return model


def run_inference(
    vae,
    llm,
    geometry,
    geometry_type: int,
    instruction: str,
    device: str,
    verbose: bool = False,
):
    """Run full inference pipeline."""
    # Get original parameters
    original_params = geometry_to_params(geometry, geometry_type)
    original_params_norm = normalize_params(original_params, geometry_type)

    if verbose:
        param_names = PARAM_NAMES[geometry_type]
        print(f"\nOriginal parameters:")
        for name, val in zip(param_names, original_params.tolist()):
            print(f"  {name}: {val:.2f} mm")

    # Extract B-Rep graph
    solid = geometry.to_solid()
    graph = extract_brep_hetero_graph_from_solid(solid)

    # Convert to HeteroData
    params_padded = torch.zeros(6)
    params_padded[:len(original_params)] = original_params
    params_norm_padded = torch.zeros(6)
    params_norm_padded[:len(original_params_norm)] = original_params_norm
    params_mask = torch.zeros(6)
    params_mask[:len(original_params)] = 1.0

    data = brep_graph_to_hetero_data(
        graph, geometry_type, params_padded, params_norm_padded, params_mask
    )
    data = data.to(device)

    # Encode to latent
    with torch.no_grad():
        mu, logvar = vae.encode(data)
        z = mu  # Use mean for inference

    if verbose:
        print(f"\nLatent vector (first 8 dims): {z[0, :8].cpu().numpy().round(3)}")

    # Classify geometry type from latent
    with torch.no_grad():
        geo_logits = vae.geometry_type_head(mu)
        pred_type = geo_logits.argmax(dim=-1).item()

    if verbose:
        print(f"\nPredicted geometry type: {GEOMETRY_TYPE_NAMES[pred_type]} (actual: {GEOMETRY_TYPE_NAMES[geometry_type]})")

    # Apply instruction through LLM
    if llm is not None:
        print(f"\nInstruction: \"{instruction}\"")
        with torch.no_grad():
            outputs = llm.forward_instruct(
                z, [instruction],
                geometry_types=torch.tensor([geometry_type], device=device)
            )

            # Get predicted parameters
            param_pred = outputs['param_pred'][0]  # (6,)
            z_edited = outputs['z_edited'][0]  # (32,)

        # Denormalize predicted parameters
        num_params = GEOMETRY_PARAM_COUNTS[geometry_type]
        pred_params_norm = param_pred[:num_params].cpu()
        pred_params = denormalize_params(pred_params_norm, geometry_type)

        if verbose:
            print(f"\nEdited parameters:")
            param_names = PARAM_NAMES[geometry_type]
            for name, orig, pred in zip(param_names, original_params.tolist(), pred_params.tolist()):
                delta = pred - orig
                print(f"  {name}: {orig:.2f} → {pred:.2f} mm (Δ = {delta:+.2f})")

        return {
            'original_params': original_params,
            'predicted_params': pred_params,
            'z_original': z.cpu(),
            'z_edited': z_edited.cpu(),
            'geometry_type': geometry_type,
            'pred_geometry_type': pred_type,
        }
    else:
        # VAE-only mode
        with torch.no_grad():
            param_pred = vae.param_head(mu)[0]

        num_params = GEOMETRY_PARAM_COUNTS[geometry_type]
        pred_params_norm = param_pred[:num_params].cpu()
        pred_params = denormalize_params(pred_params_norm, geometry_type)

        if verbose:
            print(f"\nVAE-predicted parameters (no instruction):")
            param_names = PARAM_NAMES[geometry_type]
            for name, orig, pred in zip(param_names, original_params.tolist(), pred_params.tolist()):
                error = abs(pred - orig)
                print(f"  {name}: {orig:.2f} → {pred:.2f} mm (error: {error:.2f})")

        return {
            'original_params': original_params,
            'predicted_params': pred_params,
            'z_original': z.cpu(),
            'geometry_type': geometry_type,
            'pred_geometry_type': pred_type,
        }


def test_all_geometries(vae, device: str):
    """Test VAE encoding/decoding on all geometry types."""
    print("\n" + "="*60)
    print("Testing VAE on all geometry types")
    print("="*60)

    for geo_type in range(6):
        geo_name = GEOMETRY_TYPE_NAMES[geo_type]
        print(f"\n--- {geo_name.upper()} ---")

        geometry = create_random_geometry(geo_type, seed=42 + geo_type)
        result = run_inference(
            vae, None, geometry, geo_type, "", device, verbose=True
        )

        # Compute error
        orig = result['original_params']
        pred = result['predicted_params']
        mae = (orig - pred).abs().mean().item()
        print(f"  Mean Absolute Error: {mae:.2f} mm")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Multi-Geometry Inference")

    # Model paths
    parser.add_argument("--vae-checkpoint", type=str,
                        default="outputs/hetero_vae/best_model.pt",
                        help="Path to HeteroVAE checkpoint")
    parser.add_argument("--llm-checkpoint", type=str,
                        default="outputs/llm_instruct/best_model.pt",
                        help="Path to LLM checkpoint")

    # Geometry options
    parser.add_argument("--random", action="store_true",
                        help="Generate random geometry")
    parser.add_argument("--geometry-type", type=str, default="bracket",
                        choices=["bracket", "tube", "channel", "block", "cylinder", "blockhole"],
                        help="Geometry type to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for geometry generation")

    # Instruction
    parser.add_argument("--instruction", type=str, default="make it larger",
                        help="Natural language instruction")

    # Modes
    parser.add_argument("--vae-only", action="store_true",
                        help="Skip LLM, test VAE only")
    parser.add_argument("--test-all", action="store_true",
                        help="Test VAE on all geometry types")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load VAE
    vae = load_vae(args.vae_checkpoint, device)

    # Test all geometries mode
    if args.test_all:
        test_all_geometries(vae, device)
        return

    # Load LLM if not VAE-only
    llm = None
    if not args.vae_only:
        llm = load_llm(args.llm_checkpoint, vae.config, device)

    # Get geometry type ID
    type_name_to_id = {v: k for k, v in GEOMETRY_TYPE_NAMES.items()}
    geometry_type = type_name_to_id[args.geometry_type]

    # Create geometry
    if args.random:
        geometry = create_random_geometry(geometry_type, args.seed)
        print(f"\nGenerated random {args.geometry_type}")
    else:
        # Default: create random
        geometry = create_random_geometry(geometry_type, args.seed)
        print(f"\nGenerated random {args.geometry_type}")

    # Run inference
    result = run_inference(
        vae, llm, geometry, geometry_type,
        args.instruction, device, verbose=True
    )

    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == "__main__":
    main()
