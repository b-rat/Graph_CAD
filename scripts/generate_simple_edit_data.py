#!/usr/bin/env python3
"""
Generate simplified edit data with directional-only instructions.

Instead of "increase leg1 by 17.3mm", uses simple instructions like
"make leg1 longer" with a fixed proportional change.

This removes numerical reasoning burden from the LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig
from graph_cad.data.l_bracket import VariableLBracket, VariableLBracketRanges
from graph_cad.data.graph_extraction import extract_graph_from_solid_variable


# Simple directional instructions - no numbers
SIMPLE_INSTRUCTIONS = {
    "leg1_length": {
        "increase": [
            "make leg1 longer",
            "extend the horizontal leg",
            "increase leg1 length",
            "lengthen the first leg",
            "make the horizontal leg bigger",
        ],
        "decrease": [
            "make leg1 shorter",
            "shorten the horizontal leg",
            "decrease leg1 length",
            "reduce the first leg",
            "make the horizontal leg smaller",
        ],
    },
    "leg2_length": {
        "increase": [
            "make leg2 longer",
            "extend the vertical leg",
            "increase leg2 length",
            "lengthen the second leg",
            "make the vertical leg bigger",
        ],
        "decrease": [
            "make leg2 shorter",
            "shorten the vertical leg",
            "decrease leg2 length",
            "reduce the second leg",
            "make the vertical leg smaller",
        ],
    },
    "width": {
        "increase": [
            "make it wider",
            "increase the width",
            "widen the bracket",
            "make the bracket wider",
            "increase the Y dimension",
        ],
        "decrease": [
            "make it narrower",
            "decrease the width",
            "narrow the bracket",
            "make the bracket narrower",
            "reduce the Y dimension",
        ],
    },
    "thickness": {
        "increase": [
            "make it thicker",
            "increase the thickness",
            "thicken the material",
            "use thicker stock",
            "make the walls thicker",
        ],
        "decrease": [
            "make it thinner",
            "decrease the thickness",
            "thin the material",
            "use thinner stock",
            "make the walls thinner",
        ],
    },
}

# No-op instructions
NOOP_INSTRUCTIONS = [
    "keep it the same",
    "no changes",
    "leave it unchanged",
    "don't modify anything",
    "keep current dimensions",
]

# Default parameter ranges
RANGES = VariableLBracketRanges()


def load_vae(checkpoint_path: str, device: str) -> tuple[VariableGraphVAE, dict]:
    """Load VAE from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = VariableGraphVAEConfig(**checkpoint["config"])
    model = VariableGraphVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def encode_bracket(
    vae: VariableGraphVAE,
    bracket: VariableLBracket,
    device: str,
    max_nodes: int = 20,
    max_edges: int = 50,
) -> np.ndarray:
    """Encode a bracket to latent z."""
    graph = extract_graph_from_solid_variable(bracket.to_solid())

    num_nodes = graph.node_features.shape[0]
    num_edges = graph.edge_features.shape[0]

    # Pad node features
    x = np.zeros((max_nodes, 13), dtype=np.float32)
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

    # Edge index
    edge_index = np.zeros((2, max_edges), dtype=np.int64)
    edge_index[:, :num_edges] = graph.edge_index

    # Convert to tensors
    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    face_types_t = torch.tensor(face_types, dtype=torch.long, device=device).unsqueeze(0)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device).unsqueeze(0)
    node_mask_t = torch.tensor(node_mask, dtype=torch.float32, device=device).unsqueeze(0)
    batch_t = torch.zeros(max_nodes, dtype=torch.long, device=device)

    # Flatten for encoder
    x_flat = x_t.view(-1, 13)
    face_types_flat = face_types_t.view(-1)
    node_mask_flat = node_mask_t.view(-1)

    with torch.no_grad():
        mu, _ = vae.encode(
            x_flat, face_types_flat, edge_index_t, edge_attr_t.view(-1, 2),
            batch_t, node_mask_flat
        )

    return mu.squeeze(0).cpu().numpy()


def get_param_range(param: str) -> tuple[float, float]:
    """Get valid range for a parameter."""
    if param == "leg1_length":
        return RANGES.leg1_length
    elif param == "leg2_length":
        return RANGES.leg2_length
    elif param == "width":
        return RANGES.width
    elif param == "thickness":
        return RANGES.thickness
    else:
        raise ValueError(f"Unknown parameter: {param}")


def generate_single_sample(
    vae: VariableGraphVAE,
    rng: np.random.Generator,
    device: str,
    max_nodes: int,
    max_edges: int,
    delta_fraction: float = 0.15,
    increase_only: bool = False,
) -> dict | None:
    """Generate a single edit sample with simple instruction."""
    try:
        # Create random source bracket (simple, no holes/fillets for now)
        bracket_src = VariableLBracket.random(rng)
        z_src = encode_bracket(vae, bracket_src, device, max_nodes, max_edges)

        # Pick random parameter and direction
        param = rng.choice(list(SIMPLE_INSTRUCTIONS.keys()))
        if increase_only:
            direction = "increase"
        else:
            direction = rng.choice(["increase", "decrease"])
        instruction = rng.choice(SIMPLE_INSTRUCTIONS[param][direction])

        # Get current value and compute delta
        current_value = getattr(bracket_src, param)
        param_range = get_param_range(param)

        if direction == "increase":
            delta = current_value * delta_fraction
            new_value = min(param_range[1], current_value + delta)
        else:
            delta = -current_value * delta_fraction
            new_value = max(param_range[0], current_value + delta)

        actual_delta = new_value - current_value

        # Skip if change is too small
        if abs(actual_delta) < 0.5:
            return None

        # Create target bracket with modified param
        target_params = {
            "leg1_length": bracket_src.leg1_length,
            "leg2_length": bracket_src.leg2_length,
            "width": bracket_src.width,
            "thickness": bracket_src.thickness,
        }
        target_params[param] = new_value

        # Keep topology the same
        bracket_tgt = VariableLBracket(
            leg1_length=target_params["leg1_length"],
            leg2_length=target_params["leg2_length"],
            width=target_params["width"],
            thickness=target_params["thickness"],
            fillet_radius=bracket_src.fillet_radius,
            hole1_diameters=bracket_src.hole1_diameters,
            hole1_distances=bracket_src.hole1_distances,
            hole2_diameters=bracket_src.hole2_diameters,
            hole2_distances=bracket_src.hole2_distances,
        )

        z_tgt = encode_bracket(vae, bracket_tgt, device, max_nodes, max_edges)
        delta_z = z_tgt - z_src

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param": param,
            "direction": 1.0 if direction == "increase" else 0.0,
            "current_value": float(current_value),
            "new_value": float(new_value),
            "delta_value": float(actual_delta),
            "edit_type": "simple_direction",
        }

    except Exception as e:
        return None


def generate_noop_sample(
    vae: VariableGraphVAE,
    rng: np.random.Generator,
    device: str,
    max_nodes: int,
    max_edges: int,
) -> dict | None:
    """Generate a no-op sample."""
    try:
        bracket = VariableLBracket.random(rng)
        z = encode_bracket(vae, bracket, device, max_nodes, max_edges)
        instruction = rng.choice(NOOP_INSTRUCTIONS)

        return {
            "instruction": instruction,
            "z_src": z.tolist(),
            "z_tgt": z.tolist(),
            "delta_z": [0.0] * len(z),
            "param": None,
            "direction": 0.5,
            "current_value": None,
            "new_value": None,
            "delta_value": 0.0,
            "edit_type": "noop",
        }

    except Exception as e:
        return None


def generate_dataset(
    vae: VariableGraphVAE,
    num_samples: int,
    device: str,
    max_nodes: int,
    max_edges: int,
    seed: int = 42,
    delta_fraction: float = 0.15,
    noop_ratio: float = 0.1,
    increase_only: bool = False,
) -> list[dict]:
    """Generate full dataset of simple edit samples."""
    rng = np.random.default_rng(seed)
    samples = []

    num_edit = int(num_samples * (1 - noop_ratio))
    num_noop = num_samples - num_edit

    print(f"Generating {num_edit} edit samples, {num_noop} no-ops")
    print(f"Delta fraction: {delta_fraction:.0%}")
    if increase_only:
        print("Direction: INCREASE ONLY")

    # Generate edit samples
    pbar = tqdm(total=num_edit, desc="Edit samples")
    attempts = 0
    max_attempts = num_edit * 10

    while len(samples) < num_edit and attempts < max_attempts:
        sample = generate_single_sample(
            vae, rng, device, max_nodes, max_edges, delta_fraction, increase_only
        )
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1

    pbar.close()

    if len(samples) < num_edit:
        print(f"Warning: Only generated {len(samples)}/{num_edit} edit samples")

    # Generate noop samples
    print("Generating no-op samples...")
    pbar = tqdm(total=num_noop, desc="No-op samples")
    attempts = 0

    while len(samples) < num_edit + num_noop and attempts < num_noop * 10:
        sample = generate_noop_sample(vae, rng, device, max_nodes, max_edges)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1

    pbar.close()

    # Shuffle
    rng.shuffle(samples)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate simplified edit data with directional instructions"
    )

    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        required=True,
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--delta-fraction",
        type=float,
        default=0.15,
        help="Proportional change for each edit (default: 15%%)",
    )
    parser.add_argument(
        "--noop-ratio",
        type=float,
        default=0.1,
        help="Fraction of no-op samples (default: 10%%)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--increase-only",
        action="store_true",
        help="Only generate increase instructions (no decrease)",
    )

    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load VAE
    print(f"\nLoading VAE from {args.vae_checkpoint}...")
    vae, _ = load_vae(args.vae_checkpoint, device)
    print(f"  Latent dim: {vae.config.latent_dim}")

    # Generate dataset
    print(f"\nGenerating {args.num_samples} samples...")
    samples = generate_dataset(
        vae=vae,
        num_samples=args.num_samples,
        device=device,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        seed=args.seed,
        delta_fraction=args.delta_fraction,
        noop_ratio=args.noop_ratio,
        increase_only=args.increase_only,
    )

    # Split
    n_total = len(samples)
    n_val = int(n_total * args.val_ratio)
    n_test = int(n_total * args.test_ratio)
    n_train = n_total - n_val - n_test

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")

    # Count by param
    param_counts = {}
    for s in train_samples:
        p = s.get("param")
        if p:
            param_counts[p] = param_counts.get(p, 0) + 1
    print(f"\nParameter distribution in train:")
    for p, c in sorted(param_counts.items()):
        print(f"  {p}: {c}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f)

    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f)

    with open(output_dir / "test.json", "w") as f:
        json.dump(test_samples, f)

    # Metadata
    metadata = {
        "vae_checkpoint": args.vae_checkpoint,
        "num_samples": n_total,
        "delta_fraction": args.delta_fraction,
        "noop_ratio": args.noop_ratio,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "seed": args.seed,
        "instruction_type": "simple_directional",
        "increase_only": args.increase_only,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
