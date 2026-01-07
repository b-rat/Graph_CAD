#!/usr/bin/env python3
"""
Generate training data for the latent editor using variable topology VAE.

Creates instruction-latent pairs by:
1. Sampling random variable topology L-brackets
2. Applying random parameter edits (core params only)
3. Encoding both through the variable topology VAE
4. Generating natural language instructions

Usage:
    python scripts/generate_variable_edit_data.py \
        --vae-checkpoint outputs/vae_variable/best_model.pt \
        --num-samples 50000 \
        --output data/edit_data_variable
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

from graph_cad.data.edit_dataset import generate_instruction
from graph_cad.data.graph_extraction import extract_graph_from_solid_variable
from graph_cad.data.l_bracket import VariableLBracket, VariableLBracketRanges
from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig
from graph_cad.models.parameter_vae import ParameterVAE, ParameterVAEConfig


# Parameter delta ranges for variable topology (core params only)
# Holes and fillets vary in count, so we don't edit them
PARAM_DELTA_RANGES = {
    "leg1_length": (-50, 50),
    "leg2_length": (-50, 50),
    "width": (-15, 15),
    "thickness": (-4, 4),
}


def load_vae(checkpoint_path: str, device: str) -> tuple[VariableGraphVAE | ParameterVAE, dict]:
    """
    Load VAE from checkpoint (auto-detects VariableGraphVAE or ParameterVAE).

    Both VAE types have identical encoders, so they produce compatible latent vectors.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    # Auto-detect VAE type from config keys
    # ParameterVAEConfig has 'max_holes_per_leg', VariableGraphVAEConfig has 'decoder_hidden_dims'
    if "max_holes_per_leg" in config_dict or "decoder_hidden_dim" in config_dict:
        # ParameterVAE
        config = ParameterVAEConfig(**config_dict)
        model = ParameterVAE(config)
        print(f"Detected ParameterVAE (latent_dim={config.latent_dim})")
    else:
        # VariableGraphVAE
        config = VariableGraphVAEConfig(**config_dict)
        model = VariableGraphVAE(config)
        print(f"Detected VariableGraphVAE (latent_dim={config.latent_dim})")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def encode_bracket(
    vae: VariableGraphVAE | ParameterVAE,
    bracket: VariableLBracket,
    device: str,
    max_nodes: int = 20,
    max_edges: int = 50,
) -> np.ndarray:
    """Encode a bracket through the variable VAE and return latent vector."""
    # Extract variable topology graph
    graph = extract_graph_from_solid_variable(bracket.to_solid())

    # Prepare tensors with padding
    num_nodes = graph.node_features.shape[0]
    num_edges = graph.edge_features.shape[0]

    # Pad node features
    x = np.zeros((max_nodes, 13), dtype=np.float32)  # 13D: area, dir, centroid, curv, bbox_diag, bbox_center
    x[:num_nodes] = graph.node_features

    # Pad face types
    face_types = np.zeros(max_nodes, dtype=np.int64)
    face_types[:num_nodes] = graph.face_types

    # Pad edge features
    edge_attr = np.zeros((max_edges, 2), dtype=np.float32)
    edge_attr[:num_edges] = graph.edge_features

    # Node mask
    node_mask = np.zeros(max_nodes, dtype=np.float32)
    node_mask[:num_nodes] = 1.0

    # Edge index - need to handle padding carefully
    # For padded edges, point to node 0 (self-loops)
    edge_index = np.zeros((2, max_edges), dtype=np.int64)
    edge_index[:, :num_edges] = graph.edge_index

    # Convert to tensors
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    face_types_t = torch.tensor(face_types, dtype=torch.long, device=device).unsqueeze(0)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device).unsqueeze(0)
    node_mask_t = torch.tensor(node_mask, dtype=torch.float32, device=device).unsqueeze(0)
    batch_t = torch.zeros(max_nodes, dtype=torch.long, device=device)

    # Flatten for encoder (it expects flattened batch format)
    x_flat = x_t.view(-1, 13)
    face_types_flat = face_types_t.view(-1)
    node_mask_flat = node_mask_t.view(-1)

    with torch.no_grad():
        mu, _ = vae.encode(
            x_flat, face_types_flat, edge_index_t, edge_attr_t.view(-1, 2),
            batch_t, node_mask_flat
        )

    return mu.squeeze(0).cpu().numpy()


def generate_single_edit_sample(
    vae: VariableGraphVAE | ParameterVAE,
    rng: np.random.Generator,
    device: str,
    max_nodes: int = 20,
    max_edges: int = 50,
) -> dict | None:
    """Generate a single-parameter edit sample."""
    try:
        # Sample source L-bracket with variable topology
        bracket_src = VariableLBracket.random(rng)

        # Choose random parameter to edit (core params only)
        param = rng.choice(list(PARAM_DELTA_RANGES.keys()))
        delta_min, delta_max = PARAM_DELTA_RANGES[param]
        delta = rng.uniform(delta_min, delta_max)

        # Skip very small deltas
        if abs(delta) < 0.5:
            delta = 0.5 * np.sign(delta) if delta != 0 else rng.choice([-0.5, 0.5])

        # Apply edit - create new bracket with modified param
        old_value = getattr(bracket_src, param)
        new_value = old_value + delta

        # Clamp to valid range
        ranges = VariableLBracketRanges()
        param_range = getattr(ranges, param)
        new_value = max(param_range[0], min(param_range[1], new_value))
        actual_delta = new_value - old_value

        # Skip if delta was clamped to near-zero
        if abs(actual_delta) < 0.1:
            return None

        # Create target bracket with same topology but modified param
        bracket_tgt = VariableLBracket(
            leg1_length=bracket_src.leg1_length if param != "leg1_length" else new_value,
            leg2_length=bracket_src.leg2_length if param != "leg2_length" else new_value,
            width=bracket_src.width if param != "width" else new_value,
            thickness=bracket_src.thickness if param != "thickness" else new_value,
            fillet_radius=bracket_src.fillet_radius,
            hole1_diameters=bracket_src.hole1_diameters,
            hole1_distances=bracket_src.hole1_distances,
            hole2_diameters=bracket_src.hole2_diameters,
            hole2_distances=bracket_src.hole2_distances,
        )

        # Encode both brackets
        z_src = encode_bracket(vae, bracket_src, device, max_nodes, max_edges)
        z_tgt = encode_bracket(vae, bracket_tgt, device, max_nodes, max_edges)
        delta_z = z_tgt - z_src

        # Generate instruction
        instruction = generate_instruction(param, actual_delta, old_value, rng)

        # Direction label: 1.0 = increase, 0.0 = decrease
        direction = 1.0 if actual_delta > 0 else 0.0

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": {param: float(actual_delta)},
            "direction": direction,
            "edit_type": "single",
            "has_fillet": bracket_src.has_fillet,
            "num_holes": bracket_src.num_holes_leg1 + bracket_src.num_holes_leg2,
        }

    except Exception as e:
        print(f"Warning: Failed to generate sample: {e}")
        return None


def generate_paired_edit_sample(
    vae: VariableGraphVAE | ParameterVAE,
    rng: np.random.Generator,
    device: str,
    max_nodes: int = 20,
    max_edges: int = 50,
) -> dict | None:
    """Generate a paired edit sample (increase + decrease for same source)."""
    try:
        # Sample source L-bracket
        bracket_src = VariableLBracket.random(rng)

        # Choose random parameter to edit
        param = rng.choice(list(PARAM_DELTA_RANGES.keys()))
        delta_min, delta_max = PARAM_DELTA_RANGES[param]

        # Generate a positive delta magnitude
        delta_magnitude = rng.uniform(1.0, max(abs(delta_min), abs(delta_max)))

        # Get old value and range
        old_value = getattr(bracket_src, param)
        ranges = VariableLBracketRanges()
        param_range = getattr(ranges, param)

        # Calculate increase value
        inc_value = min(param_range[1], old_value + delta_magnitude)
        inc_actual_delta = inc_value - old_value

        # Calculate decrease value
        dec_value = max(param_range[0], old_value - delta_magnitude)
        dec_actual_delta = dec_value - old_value

        # Skip if either delta was clamped to near-zero
        if abs(inc_actual_delta) < 0.1 or abs(dec_actual_delta) < 0.1:
            return None

        # Create target brackets with modified param
        def make_bracket(value):
            return VariableLBracket(
                leg1_length=bracket_src.leg1_length if param != "leg1_length" else value,
                leg2_length=bracket_src.leg2_length if param != "leg2_length" else value,
                width=bracket_src.width if param != "width" else value,
                thickness=bracket_src.thickness if param != "thickness" else value,
                fillet_radius=bracket_src.fillet_radius,
                hole1_diameters=bracket_src.hole1_diameters,
                hole1_distances=bracket_src.hole1_distances,
                hole2_diameters=bracket_src.hole2_diameters,
                hole2_distances=bracket_src.hole2_distances,
            )

        bracket_inc = make_bracket(inc_value)
        bracket_dec = make_bracket(dec_value)

        # Encode all three brackets
        z_src = encode_bracket(vae, bracket_src, device, max_nodes, max_edges)
        z_inc = encode_bracket(vae, bracket_inc, device, max_nodes, max_edges)
        z_dec = encode_bracket(vae, bracket_dec, device, max_nodes, max_edges)

        delta_z_inc = z_inc - z_src
        delta_z_dec = z_dec - z_src

        # Generate instructions
        instruction_inc = generate_instruction(param, inc_actual_delta, old_value, rng)
        instruction_dec = generate_instruction(param, dec_actual_delta, old_value, rng)

        return {
            "z_src": z_src.tolist(),
            "param": param,
            # Increase direction
            "instruction_inc": instruction_inc,
            "z_tgt_inc": z_inc.tolist(),
            "delta_z_inc": delta_z_inc.tolist(),
            "delta_inc": float(inc_actual_delta),
            # Decrease direction
            "instruction_dec": instruction_dec,
            "z_tgt_dec": z_dec.tolist(),
            "delta_z_dec": delta_z_dec.tolist(),
            "delta_dec": float(dec_actual_delta),
            "edit_type": "paired",
            "has_fillet": bracket_src.has_fillet,
            "num_holes": bracket_src.num_holes_leg1 + bracket_src.num_holes_leg2,
        }

    except Exception as e:
        print(f"Warning: Failed to generate paired sample: {e}")
        return None


def generate_noop_sample(
    vae: VariableGraphVAE | ParameterVAE,
    rng: np.random.Generator,
    device: str,
    max_nodes: int = 20,
    max_edges: int = 50,
) -> dict | None:
    """Generate a no-op sample (identity edit)."""
    try:
        bracket = VariableLBracket.random(rng)
        z = encode_bracket(vae, bracket, device, max_nodes, max_edges)

        noop_instructions = ["keep it the same", "no changes", "leave it unchanged"]
        instruction = rng.choice(noop_instructions)

        return {
            "instruction": instruction,
            "z_src": z.tolist(),
            "z_tgt": z.tolist(),
            "delta_z": [0.0] * len(z),
            "param_deltas": {},
            "direction": 0.5,
            "edit_type": "noop",
            "has_fillet": bracket.has_fillet,
            "num_holes": bracket.num_holes_leg1 + bracket.num_holes_leg2,
        }

    except Exception as e:
        print(f"Warning: Failed to generate noop sample: {e}")
        return None


def generate_dataset(
    vae: VariableGraphVAE | ParameterVAE,
    num_samples: int,
    device: str,
    max_nodes: int,
    max_edges: int,
    seed: int = 42,
    single_ratio: float = 0.8,
    noop_ratio: float = 0.2,
) -> list[dict]:
    """Generate full dataset of edit samples."""
    rng = np.random.default_rng(seed)
    samples = []

    num_single = int(num_samples * single_ratio)
    num_noop = num_samples - num_single

    print(f"Generating {num_single} single edits, {num_noop} no-ops")

    # Generate single-parameter edits
    print("Generating single-parameter edits...")
    pbar = tqdm(total=num_single, desc="Single edits")
    attempts = 0
    max_attempts = num_single * 3

    while len([s for s in samples if s.get("edit_type") == "single"]) < num_single:
        sample = generate_single_edit_sample(vae, rng, device, max_nodes, max_edges)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Could only generate {len(samples)} samples after {attempts} attempts")
            break
    pbar.close()

    # Generate no-op samples
    print("Generating no-op samples...")
    pbar = tqdm(total=num_noop, desc="No-op edits")
    attempts = 0
    max_attempts = num_noop * 3

    while len([s for s in samples if s.get("edit_type") == "noop"]) < num_noop:
        sample = generate_noop_sample(vae, rng, device, max_nodes, max_edges)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
        if attempts > max_attempts:
            break
    pbar.close()

    rng.shuffle(samples)
    return samples


def generate_paired_dataset(
    vae: VariableGraphVAE | ParameterVAE,
    num_pairs: int,
    device: str,
    max_nodes: int,
    max_edges: int,
    seed: int = 42,
) -> list[dict]:
    """Generate dataset of paired edit samples for direction classifier training."""
    rng = np.random.default_rng(seed)
    samples = []

    print(f"Generating {num_pairs} paired edit samples...")

    pbar = tqdm(total=num_pairs, desc="Paired edits")
    attempts = 0
    max_attempts = num_pairs * 3

    while len(samples) < num_pairs:
        sample = generate_paired_edit_sample(vae, rng, device, max_nodes, max_edges)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Could only generate {len(samples)} paired samples")
            break
    pbar.close()

    rng.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate variable topology latent edit data")
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_variable/best_model.pt",
        help="Path to trained variable topology VAE checkpoint",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/edit_data_variable",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
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
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Generate paired samples for direction classifier",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=20,
        help="Maximum nodes (must match VAE config)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=50,
        help="Maximum edges (must match VAE config)",
    )

    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print(f"Loading variable VAE from {args.vae_checkpoint}...")
    vae, checkpoint = load_vae(args.vae_checkpoint, device)
    print(f"  Latent dim: {vae.config.latent_dim}")
    print(f"  Max nodes: {vae.config.max_nodes}")
    print(f"  Max edges: {vae.config.max_edges}")

    # Generate samples
    if args.paired:
        samples = generate_paired_dataset(
            vae=vae,
            num_pairs=args.num_samples,
            device=device,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            seed=args.seed,
        )
    else:
        samples = generate_dataset(
            vae=vae,
            num_samples=args.num_samples,
            device=device,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            seed=args.seed,
        )

    print(f"\nGenerated {len(samples)} samples")

    # Split
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

    # Save
    print(f"\nSaving to {output_dir}...")

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f)

    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f)

    with open(output_dir / "test.json", "w") as f:
        json.dump(test_samples, f)

    # Metadata
    metadata = {
        "vae_checkpoint": args.vae_checkpoint,
        "latent_dim": vae.config.latent_dim,
        "max_nodes": vae.config.max_nodes,
        "max_edges": vae.config.max_edges,
        "num_samples": len(samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "seed": args.seed,
        "paired": args.paired,
        "variable_topology": True,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
