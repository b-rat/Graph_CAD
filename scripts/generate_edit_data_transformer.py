#!/usr/bin/env python3
"""
Generate training data for the latent editor using Transformer VAE.

Creates instruction-latent pairs for all 4 core parameters:
- Increase/decrease leg1
- Increase/decrease leg2
- Increase/decrease width
- Increase/decrease thickness
- Change both legs
- No-op (zero change)

Uses absolute units (mm) for all commands.

Usage:
    python scripts/generate_edit_data_transformer.py --vae-checkpoint outputs/vae_direct_kl_exclude_v2/best_model.pt
    python scripts/generate_edit_data_transformer.py --num-samples 10000 --output data/edit_data_all_params
    python scripts/generate_edit_data_transformer.py --legs-only  # For leg-only mode (legacy)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data.graph_extraction import extract_graph_from_solid_variable
from graph_cad.data.l_bracket import VariableLBracket, VariableLBracketRanges
from graph_cad.models.graph_vae import VariableGraphVAEConfig, VariableGraphVAEEncoder
from graph_cad.models.transformer_decoder import TransformerDecoderConfig, TransformerGraphVAE


# Delta ranges for each parameter (absolute mm)
LEG_DELTA_RANGE = (-50, 50)  # Can change by -50mm to +50mm
WIDTH_DELTA_RANGE = (-20, 20)  # Width: 20-60mm range, so ±20mm
THICKNESS_DELTA_RANGE = (-5, 5)  # Thickness: 3-12mm range, so ±5mm

# Maximum padding values (must match VAE training)
MAX_NODES = 20
MAX_EDGES = 50

# Instruction templates for leg length operations
LEG1_TEMPLATES = [
    "make leg1 {delta:+.0f}mm {direction}",
    "change leg1 length by {delta:+.0f}mm",
    "{direction} the horizontal leg by {abs_delta:.0f}mm",
    "adjust leg1 by {delta:+.0f}mm",
    "leg1 {delta:+.0f}mm",
]

LEG2_TEMPLATES = [
    "make leg2 {delta:+.0f}mm {direction}",
    "change leg2 length by {delta:+.0f}mm",
    "{direction} the vertical leg by {abs_delta:.0f}mm",
    "adjust leg2 by {delta:+.0f}mm",
    "leg2 {delta:+.0f}mm",
]

BOTH_LEGS_TEMPLATES = [
    "make both legs {delta:+.0f}mm {direction}",
    "change leg1 by {delta1:+.0f}mm and leg2 by {delta2:+.0f}mm",
    "{direction} both legs by {abs_delta:.0f}mm",
    "extend both legs by {delta:+.0f}mm",
    "adjust leg1 {delta1:+.0f}mm, leg2 {delta2:+.0f}mm",
]

NOOP_TEMPLATES = [
    "keep it the same",
    "no changes",
    "leave it unchanged",
    "don't modify anything",
    "keep all dimensions",
]

# Instruction templates for width operations
WIDTH_TEMPLATES = [
    "make the bracket {delta:+.0f}mm {direction}",
    "change width by {delta:+.0f}mm",
    "{direction} the width by {abs_delta:.0f}mm",
    "adjust width by {delta:+.0f}mm",
    "width {delta:+.0f}mm",
    "make it {delta:+.0f}mm {direction} in the Y direction",
]

# Instruction templates for thickness operations
THICKNESS_TEMPLATES = [
    "make the bracket {delta:+.0f}mm {direction}",
    "change thickness by {delta:+.0f}mm",
    "{direction} the thickness by {abs_delta:.0f}mm",
    "adjust thickness by {delta:+.0f}mm",
    "thickness {delta:+.0f}mm",
    "make it {delta:+.0f}mm {direction}",
]


def generate_instruction(
    param: str,
    delta: float,
    rng: np.random.Generator,
    delta2: float | None = None,
) -> str:
    """Generate instruction for a parameter edit operation."""
    # Direction words depend on the parameter type
    if param in ("leg1_length", "leg2_length", "both"):
        direction = "longer" if delta > 0 else "shorter"
    elif param == "width":
        direction = "wider" if delta > 0 else "narrower"
    elif param == "thickness":
        direction = "thicker" if delta > 0 else "thinner"
    else:
        direction = "larger" if delta > 0 else "smaller"

    abs_delta = abs(delta)

    if param == "leg1_length":
        template = rng.choice(LEG1_TEMPLATES)
    elif param == "leg2_length":
        template = rng.choice(LEG2_TEMPLATES)
    elif param == "both":
        template = rng.choice(BOTH_LEGS_TEMPLATES)
    elif param == "width":
        template = rng.choice(WIDTH_TEMPLATES)
    elif param == "thickness":
        template = rng.choice(THICKNESS_TEMPLATES)
    else:
        return f"change {param} by {delta:+.0f}mm"

    try:
        return template.format(
            delta=delta,
            delta1=delta,
            delta2=delta2 if delta2 is not None else delta,
            abs_delta=abs_delta,
            direction=direction,
        )
    except KeyError:
        return f"change {param} by {delta:+.0f}mm"


def load_transformer_vae(checkpoint_path: str, device: str = "cpu") -> TransformerGraphVAE:
    """Load TransformerGraphVAE from checkpoint."""
    print(f"Loading Transformer VAE from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct encoder
    encoder_config = VariableGraphVAEConfig(**ckpt["encoder_config"])
    encoder = VariableGraphVAEEncoder(encoder_config)

    # Load optional param_head settings
    use_param_head = ckpt.get("use_param_head", False)
    num_params = ckpt.get("num_params", 4)

    # Reconstruct decoder config and full model
    decoder_config = TransformerDecoderConfig(**ckpt["decoder_config"])
    model = TransformerGraphVAE(
        encoder, decoder_config,
        use_param_head=use_param_head,
        num_params=num_params,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Latent dim: {decoder_config.latent_dim}")

    return model


def encode_bracket(
    vae: TransformerGraphVAE,
    bracket: VariableLBracket,
    device: str,
) -> torch.Tensor:
    """
    Encode a single bracket through the Transformer VAE.

    Returns:
        mu: Latent vector (1, latent_dim) as numpy array
    """
    # Extract graph
    graph = extract_graph_from_solid_variable(bracket.to_solid())

    num_nodes = graph.num_faces
    num_edges = graph.num_edges

    # Pad node features
    node_features = np.zeros((MAX_NODES, 13), dtype=np.float32)
    node_features[:num_nodes] = graph.node_features

    # Pad face types
    face_types = np.zeros(MAX_NODES, dtype=np.int64)
    face_types[:num_nodes] = graph.face_types

    # Create node mask
    node_mask = np.zeros(MAX_NODES, dtype=np.float32)
    node_mask[:num_nodes] = 1.0

    # Pad edge features
    edge_features = np.zeros((MAX_EDGES, 2), dtype=np.float32)
    edge_features[:num_edges] = graph.edge_features

    # Pad edge_index
    edge_index = np.zeros((2, MAX_EDGES), dtype=np.int64)
    edge_index[:, :num_edges] = graph.edge_index

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float32, device=device).unsqueeze(0)
    ft = torch.tensor(face_types, dtype=torch.long, device=device).unsqueeze(0)
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_features, dtype=torch.float32, device=device).unsqueeze(0)
    nm = torch.tensor(node_mask, dtype=torch.float32, device=device).unsqueeze(0)
    batch = torch.zeros(MAX_NODES, dtype=torch.long, device=device)

    # Encode
    with torch.no_grad():
        mu, _ = vae.encode(x.squeeze(0), ft.squeeze(0), ei, ea.squeeze(0), batch, nm.squeeze(0))

    return mu.squeeze(0).cpu().numpy()


def generate_single_leg_sample(
    vae: TransformerGraphVAE,
    rng: np.random.Generator,
    device: str,
    param: str,  # "leg1_length" or "leg2_length"
) -> dict | None:
    """Generate a single-leg edit sample."""
    try:
        # Sample source bracket with variable topology
        ranges = VariableLBracketRanges()
        bracket_src = VariableLBracket.random(rng, ranges)

        # Sample delta
        delta_min, delta_max = LEG_DELTA_RANGE
        delta = rng.uniform(delta_min, delta_max)

        # Skip very small deltas
        if abs(delta) < 1.0:
            delta = 1.0 * np.sign(delta) if delta != 0 else rng.choice([-5.0, 5.0])

        # Get old value and compute new value
        old_value = getattr(bracket_src, param)
        new_value = old_value + delta

        # Clamp to valid range
        param_range = getattr(ranges, param)
        new_value = max(param_range[0], min(param_range[1], new_value))
        actual_delta = new_value - old_value

        # Skip if clamped to near-zero
        if abs(actual_delta) < 0.5:
            return None

        # Create target bracket with modified leg length
        # Adjust hole distances proportionally when leg length changes
        new_leg1 = new_value if param == "leg1_length" else bracket_src.leg1_length
        new_leg2 = new_value if param == "leg2_length" else bracket_src.leg2_length

        # Scale hole distances proportionally to leg length change
        leg1_ratio = new_leg1 / bracket_src.leg1_length if bracket_src.leg1_length > 0 else 1.0
        leg2_ratio = new_leg2 / bracket_src.leg2_length if bracket_src.leg2_length > 0 else 1.0

        hole1_distances = tuple(d * leg1_ratio for d in bracket_src.hole1_distances)
        hole2_distances = tuple(d * leg2_ratio for d in bracket_src.hole2_distances)

        bracket_tgt = VariableLBracket(
            leg1_length=new_leg1,
            leg2_length=new_leg2,
            width=bracket_src.width,
            thickness=bracket_src.thickness,
            fillet_radius=bracket_src.fillet_radius,
            hole1_diameters=bracket_src.hole1_diameters,
            hole1_distances=hole1_distances,
            hole2_diameters=bracket_src.hole2_diameters,
            hole2_distances=hole2_distances,
        )

        # Encode both brackets
        z_src = encode_bracket(vae, bracket_src, device)
        z_tgt = encode_bracket(vae, bracket_tgt, device)
        delta_z = z_tgt - z_src

        # Generate instruction
        instruction = generate_instruction(param, actual_delta, rng)

        # Direction label
        direction = 1.0 if actual_delta > 0 else 0.0

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": {param: float(actual_delta)},
            "direction": direction,
            "edit_type": "single_leg",
        }

    except Exception as e:
        print(f"Warning: Failed to generate single leg sample: {e}")
        return None


def generate_both_legs_sample(
    vae: TransformerGraphVAE,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """Generate a sample that modifies both legs."""
    try:
        # Sample source bracket
        ranges = VariableLBracketRanges()
        bracket_src = VariableLBracket.random(rng, ranges)

        # Sample deltas for both legs (can be same or different)
        delta_min, delta_max = LEG_DELTA_RANGE

        # 50% chance of same delta for both, 50% different
        if rng.random() < 0.5:
            delta1 = rng.uniform(delta_min, delta_max)
            delta2 = delta1
        else:
            delta1 = rng.uniform(delta_min, delta_max)
            delta2 = rng.uniform(delta_min, delta_max)

        # Skip very small deltas
        if abs(delta1) < 1.0:
            delta1 = rng.choice([-5.0, 5.0])
        if abs(delta2) < 1.0:
            delta2 = rng.choice([-5.0, 5.0])

        # Compute new values with clamping
        leg1_old = bracket_src.leg1_length
        leg2_old = bracket_src.leg2_length

        leg1_new = max(ranges.leg1_length[0], min(ranges.leg1_length[1], leg1_old + delta1))
        leg2_new = max(ranges.leg2_length[0], min(ranges.leg2_length[1], leg2_old + delta2))

        actual_delta1 = leg1_new - leg1_old
        actual_delta2 = leg2_new - leg2_old

        # Skip if both clamped to near-zero
        if abs(actual_delta1) < 0.5 and abs(actual_delta2) < 0.5:
            return None

        # Create target bracket with proportionally adjusted hole distances
        leg1_ratio = leg1_new / leg1_old if leg1_old > 0 else 1.0
        leg2_ratio = leg2_new / leg2_old if leg2_old > 0 else 1.0

        hole1_distances = tuple(d * leg1_ratio for d in bracket_src.hole1_distances)
        hole2_distances = tuple(d * leg2_ratio for d in bracket_src.hole2_distances)

        bracket_tgt = VariableLBracket(
            leg1_length=leg1_new,
            leg2_length=leg2_new,
            width=bracket_src.width,
            thickness=bracket_src.thickness,
            fillet_radius=bracket_src.fillet_radius,
            hole1_diameters=bracket_src.hole1_diameters,
            hole1_distances=hole1_distances,
            hole2_diameters=bracket_src.hole2_diameters,
            hole2_distances=hole2_distances,
        )

        # Encode both brackets
        z_src = encode_bracket(vae, bracket_src, device)
        z_tgt = encode_bracket(vae, bracket_tgt, device)
        delta_z = z_tgt - z_src

        # Generate instruction
        avg_delta = (actual_delta1 + actual_delta2) / 2
        instruction = generate_instruction("both", avg_delta, rng, delta2=actual_delta2)

        # Direction based on average
        direction = 1.0 if avg_delta > 0 else 0.0

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": {
                "leg1_length": float(actual_delta1),
                "leg2_length": float(actual_delta2),
            },
            "direction": direction,
            "edit_type": "both_legs",
        }

    except Exception as e:
        print(f"Warning: Failed to generate both legs sample: {e}")
        return None


def generate_width_sample(
    vae: TransformerGraphVAE,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """Generate a width edit sample."""
    try:
        # Sample source bracket with variable topology
        ranges = VariableLBracketRanges()
        bracket_src = VariableLBracket.random(rng, ranges)

        # Sample delta
        delta_min, delta_max = WIDTH_DELTA_RANGE
        delta = rng.uniform(delta_min, delta_max)

        # Skip very small deltas
        if abs(delta) < 0.5:
            delta = 0.5 * np.sign(delta) if delta != 0 else rng.choice([-2.0, 2.0])

        # Get old value and compute new value
        old_value = bracket_src.width
        new_value = old_value + delta

        # Clamp to valid range
        new_value = max(ranges.width[0], min(ranges.width[1], new_value))
        actual_delta = new_value - old_value

        # Skip if clamped to near-zero
        if abs(actual_delta) < 0.25:
            return None

        # Create target bracket with modified width
        bracket_tgt = VariableLBracket(
            leg1_length=bracket_src.leg1_length,
            leg2_length=bracket_src.leg2_length,
            width=new_value,
            thickness=bracket_src.thickness,
            fillet_radius=bracket_src.fillet_radius,
            hole1_diameters=bracket_src.hole1_diameters,
            hole1_distances=bracket_src.hole1_distances,
            hole2_diameters=bracket_src.hole2_diameters,
            hole2_distances=bracket_src.hole2_distances,
        )

        # Encode both brackets
        z_src = encode_bracket(vae, bracket_src, device)
        z_tgt = encode_bracket(vae, bracket_tgt, device)
        delta_z = z_tgt - z_src

        # Generate instruction
        instruction = generate_instruction("width", actual_delta, rng)

        # Direction label
        direction = 1.0 if actual_delta > 0 else 0.0

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": {"width": float(actual_delta)},
            "direction": direction,
            "edit_type": "width",
        }

    except Exception as e:
        print(f"Warning: Failed to generate width sample: {e}")
        return None


def generate_thickness_sample(
    vae: TransformerGraphVAE,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """Generate a thickness edit sample."""
    try:
        # Sample source bracket with variable topology
        ranges = VariableLBracketRanges()
        bracket_src = VariableLBracket.random(rng, ranges)

        # Sample delta
        delta_min, delta_max = THICKNESS_DELTA_RANGE
        delta = rng.uniform(delta_min, delta_max)

        # Skip very small deltas
        if abs(delta) < 0.25:
            delta = 0.25 * np.sign(delta) if delta != 0 else rng.choice([-1.0, 1.0])

        # Get old value and compute new value
        old_value = bracket_src.thickness
        new_value = old_value + delta

        # Clamp to valid range
        new_value = max(ranges.thickness[0], min(ranges.thickness[1], new_value))
        actual_delta = new_value - old_value

        # Skip if clamped to near-zero
        if abs(actual_delta) < 0.1:
            return None

        # Create target bracket with modified thickness
        bracket_tgt = VariableLBracket(
            leg1_length=bracket_src.leg1_length,
            leg2_length=bracket_src.leg2_length,
            width=bracket_src.width,
            thickness=new_value,
            fillet_radius=bracket_src.fillet_radius,
            hole1_diameters=bracket_src.hole1_diameters,
            hole1_distances=bracket_src.hole1_distances,
            hole2_diameters=bracket_src.hole2_diameters,
            hole2_distances=bracket_src.hole2_distances,
        )

        # Encode both brackets
        z_src = encode_bracket(vae, bracket_src, device)
        z_tgt = encode_bracket(vae, bracket_tgt, device)
        delta_z = z_tgt - z_src

        # Generate instruction
        instruction = generate_instruction("thickness", actual_delta, rng)

        # Direction label
        direction = 1.0 if actual_delta > 0 else 0.0

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": {"thickness": float(actual_delta)},
            "direction": direction,
            "edit_type": "thickness",
        }

    except Exception as e:
        print(f"Warning: Failed to generate thickness sample: {e}")
        return None


def generate_noop_sample(
    vae: TransformerGraphVAE,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """Generate a no-op sample (identity edit)."""
    try:
        # Sample bracket
        ranges = VariableLBracketRanges()
        bracket = VariableLBracket.random(rng, ranges)

        # Encode
        z = encode_bracket(vae, bracket, device)

        # Choose no-op instruction
        instruction = rng.choice(NOOP_TEMPLATES)

        latent_dim = len(z)
        return {
            "instruction": instruction,
            "z_src": z.tolist(),
            "z_tgt": z.tolist(),
            "delta_z": [0.0] * latent_dim,
            "param_deltas": {},
            "direction": 0.5,  # No direction for noop
            "edit_type": "noop",
        }

    except Exception as e:
        print(f"Warning: Failed to generate noop sample: {e}")
        return None


def generate_dataset(
    vae: TransformerGraphVAE,
    num_samples: int,
    device: str,
    seed: int = 42,
    legs_only: bool = False,
    leg1_ratio: float = 0.20,
    leg2_ratio: float = 0.20,
    width_ratio: float = 0.20,
    thickness_ratio: float = 0.20,
    both_legs_ratio: float = 0.10,
    noop_ratio: float = 0.10,
) -> list[dict]:
    """
    Generate full dataset of edit samples for all 4 parameters.

    Args:
        vae: Transformer VAE model
        num_samples: Total number of samples
        device: Device for inference
        seed: Random seed
        legs_only: If True, only generate leg edits (legacy mode)
        leg1_ratio: Fraction of leg1-only edits
        leg2_ratio: Fraction of leg2-only edits
        width_ratio: Fraction of width edits (ignored if legs_only)
        thickness_ratio: Fraction of thickness edits (ignored if legs_only)
        both_legs_ratio: Fraction of both-legs edits
        noop_ratio: Fraction of no-op samples

    Returns:
        List of sample dictionaries
    """
    rng = np.random.default_rng(seed)
    samples = []

    # Adjust ratios for legs-only mode
    if legs_only:
        # Redistribute width/thickness ratios to legs
        extra = width_ratio + thickness_ratio
        leg1_ratio = 0.35
        leg2_ratio = 0.35
        both_legs_ratio = 0.20
        noop_ratio = 0.10
        width_ratio = 0.0
        thickness_ratio = 0.0

    # Calculate counts
    num_leg1 = int(num_samples * leg1_ratio)
    num_leg2 = int(num_samples * leg2_ratio)
    num_width = int(num_samples * width_ratio)
    num_thickness = int(num_samples * thickness_ratio)
    num_both = int(num_samples * both_legs_ratio)
    num_noop = num_samples - num_leg1 - num_leg2 - num_width - num_thickness - num_both

    if legs_only:
        print(f"LEGS-ONLY MODE: Generating {num_leg1} leg1, {num_leg2} leg2, "
              f"{num_both} both-legs, {num_noop} no-ops")
    else:
        print(f"ALL PARAMS MODE: Generating {num_leg1} leg1, {num_leg2} leg2, "
              f"{num_width} width, {num_thickness} thickness, "
              f"{num_both} both-legs, {num_noop} no-ops")

    def generate_samples_of_type(generator_fn, target_count, edit_type, desc, *args):
        """Helper to generate samples of a specific type."""
        generated = []
        pbar = tqdm(total=target_count, desc=desc)
        attempts = 0
        max_attempts = target_count * 3

        while len(generated) < target_count and attempts < max_attempts:
            sample = generator_fn(*args)
            if sample is not None:
                generated.append(sample)
                pbar.update(1)
            attempts += 1

        pbar.close()
        if len(generated) < target_count:
            print(f"Warning: Could only generate {len(generated)}/{target_count} {edit_type} samples")
        return generated

    # Generate leg1 edits
    samples.extend(generate_samples_of_type(
        lambda: generate_single_leg_sample(vae, rng, device, "leg1_length"),
        num_leg1, "leg1", "Leg1 edits"
    ))

    # Generate leg2 edits
    samples.extend(generate_samples_of_type(
        lambda: generate_single_leg_sample(vae, rng, device, "leg2_length"),
        num_leg2, "leg2", "Leg2 edits"
    ))

    # Generate width edits (if not legs-only)
    if num_width > 0:
        samples.extend(generate_samples_of_type(
            lambda: generate_width_sample(vae, rng, device),
            num_width, "width", "Width edits"
        ))

    # Generate thickness edits (if not legs-only)
    if num_thickness > 0:
        samples.extend(generate_samples_of_type(
            lambda: generate_thickness_sample(vae, rng, device),
            num_thickness, "thickness", "Thickness edits"
        ))

    # Generate both-legs edits
    samples.extend(generate_samples_of_type(
        lambda: generate_both_legs_sample(vae, rng, device),
        num_both, "both_legs", "Both-legs edits"
    ))

    # Generate no-op samples
    samples.extend(generate_samples_of_type(
        lambda: generate_noop_sample(vae, rng, device),
        num_noop, "noop", "No-op edits"
    ))

    # Shuffle
    rng.shuffle(samples)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate latent edit data for Transformer VAE (all 4 parameters)"
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_direct_kl_exclude_v2/best_model.pt",
        help="Path to Transformer VAE checkpoint",
    )
    parser.add_argument(
        "--legs-only",
        action="store_true",
        help="Only generate leg length edits (legacy mode)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/edit_data_all_params",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for VAE inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Transformer VAE
    vae = load_transformer_vae(args.vae_checkpoint, device=args.device)

    # Get latent dimension from model
    latent_dim = vae.decoder_config.latent_dim
    print(f"  Using latent dimension: {latent_dim}")

    # Generate samples
    mode_str = "leg length" if args.legs_only else "all parameter"
    print(f"\nGenerating {args.num_samples} {mode_str} edit samples...")
    samples = generate_dataset(
        vae=vae,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
        legs_only=args.legs_only,
    )

    print(f"\nGenerated {len(samples)} samples")

    # Split into train/val/test
    n_total = len(samples)
    n_val = int(n_total * args.val_ratio)
    n_test = int(n_total * args.test_ratio)
    n_train = n_total - n_val - n_test

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")

    # Save datasets
    print(f"\nSaving to {output_dir}...")

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f)

    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f)

    with open(output_dir / "test.json", "w") as f:
        json.dump(test_samples, f)

    # Save metadata
    if args.legs_only:
        edit_types_list = ["single_leg", "both_legs", "noop"]
        parameters_list = ["leg1_length", "leg2_length"]
    else:
        edit_types_list = ["single_leg", "both_legs", "width", "thickness", "noop"]
        parameters_list = ["leg1_length", "leg2_length", "width", "thickness"]

    metadata = {
        "vae_checkpoint": args.vae_checkpoint,
        "latent_dim": latent_dim,
        "num_samples": len(samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "seed": args.seed,
        "paired": False,
        "edit_types": edit_types_list,
        "parameters": parameters_list,
        "variable_topology": True,
        "legs_only": args.legs_only,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print sample distribution
    edit_types = {}
    for s in samples:
        et = s.get("edit_type", "unknown")
        edit_types[et] = edit_types.get(et, 0) + 1

    print("\nEdit type distribution:")
    for et, count in sorted(edit_types.items()):
        print(f"  {et}: {count} ({100*count/len(samples):.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
