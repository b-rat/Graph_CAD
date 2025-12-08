#!/usr/bin/env python3
"""
Generate training data for the latent editor.

Creates instruction-latent pairs by:
1. Sampling random L-brackets
2. Applying random parameter edits
3. Encoding both through the VAE
4. Generating natural language instructions

Usage:
    python scripts/generate_edit_data.py --vae-checkpoint outputs/vae_16d/best_model.pt
    python scripts/generate_edit_data.py --num-samples 50000 --output data/edit_data
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

from graph_cad.data.edit_dataset import (
    COMPOUND_TEMPLATES,
    generate_instruction,
)
from graph_cad.data.graph_extraction import extract_graph_from_solid
from graph_cad.data.l_bracket import LBracket, LBracketRanges
from graph_cad.training.vae_trainer import load_checkpoint


# Parameter delta ranges for single-parameter edits
PARAM_DELTA_RANGES = {
    "leg1_length": (-50, 50),
    "leg2_length": (-50, 50),
    "width": (-15, 15),
    "thickness": (-4, 4),
    "hole1_diameter": (-4, 4),
    "hole2_diameter": (-4, 4),
    "hole1_distance": (-30, 30),
    "hole2_distance": (-30, 30),
}


def generate_single_edit_sample(
    vae: torch.nn.Module,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """
    Generate a single-parameter edit sample.

    Args:
        vae: Trained VAE model
        rng: Random number generator
        device: Device for VAE inference

    Returns:
        Sample dict or None if generation fails
    """
    try:
        # Sample source L-bracket
        bracket_src = LBracket.random(rng)

        # Choose random parameter to edit
        param = rng.choice(list(PARAM_DELTA_RANGES.keys()))
        delta_min, delta_max = PARAM_DELTA_RANGES[param]
        delta = rng.uniform(delta_min, delta_max)

        # Skip very small deltas (avoid near-identity edits)
        if abs(delta) < 0.5:
            delta = 0.5 * np.sign(delta) if delta != 0 else rng.choice([-0.5, 0.5])

        # Apply edit
        old_value = getattr(bracket_src, param)
        bracket_tgt = bracket_src.with_modified(param, delta, clamp=True)
        new_value = getattr(bracket_tgt, param)
        actual_delta = new_value - old_value

        # Skip if delta was clamped to near-zero
        if abs(actual_delta) < 0.1:
            return None

        # Extract graphs
        graph_src = extract_graph_from_solid(bracket_src.to_solid())
        graph_tgt = extract_graph_from_solid(bracket_tgt.to_solid())

        # Encode through VAE
        with torch.no_grad():
            # Prepare single-graph batch
            x_src = torch.tensor(graph_src.node_features, dtype=torch.float32, device=device)
            edge_index_src = torch.tensor(graph_src.edge_index, dtype=torch.long, device=device)
            edge_attr_src = torch.tensor(graph_src.edge_features, dtype=torch.float32, device=device)
            batch_src = torch.zeros(x_src.shape[0], dtype=torch.long, device=device)

            x_tgt = torch.tensor(graph_tgt.node_features, dtype=torch.float32, device=device)
            edge_index_tgt = torch.tensor(graph_tgt.edge_index, dtype=torch.long, device=device)
            edge_attr_tgt = torch.tensor(graph_tgt.edge_features, dtype=torch.float32, device=device)
            batch_tgt = torch.zeros(x_tgt.shape[0], dtype=torch.long, device=device)

            # Encode
            mu_src, _ = vae.encode(x_src, edge_index_src, edge_attr_src, batch_src)
            mu_tgt, _ = vae.encode(x_tgt, edge_index_tgt, edge_attr_tgt, batch_tgt)

            z_src = mu_src.squeeze(0).cpu().numpy()
            z_tgt = mu_tgt.squeeze(0).cpu().numpy()
            delta_z = z_tgt - z_src

        # Generate instruction
        instruction = generate_instruction(param, actual_delta, old_value, rng)

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": {param: float(actual_delta)},
            "edit_type": "single",
        }

    except Exception as e:
        print(f"Warning: Failed to generate sample: {e}")
        return None


def generate_compound_edit_sample(
    vae: torch.nn.Module,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """
    Generate a compound (multi-parameter) edit sample.

    Args:
        vae: Trained VAE model
        rng: Random number generator
        device: Device for VAE inference

    Returns:
        Sample dict or None if generation fails
    """
    try:
        # Sample source L-bracket
        bracket_src = LBracket.random(rng)

        # Choose random compound template
        instruction, param_deltas = rng.choice(COMPOUND_TEMPLATES)

        # Apply all edits
        bracket_tgt = bracket_src
        actual_deltas = {}

        for param, delta in param_deltas.items():
            old_value = getattr(bracket_tgt, param)
            bracket_tgt = bracket_tgt.with_modified(param, delta, clamp=True)
            new_value = getattr(bracket_tgt, param)
            actual_deltas[param] = float(new_value - old_value)

        # Extract graphs
        graph_src = extract_graph_from_solid(bracket_src.to_solid())
        graph_tgt = extract_graph_from_solid(bracket_tgt.to_solid())

        # Encode through VAE
        with torch.no_grad():
            x_src = torch.tensor(graph_src.node_features, dtype=torch.float32, device=device)
            edge_index_src = torch.tensor(graph_src.edge_index, dtype=torch.long, device=device)
            edge_attr_src = torch.tensor(graph_src.edge_features, dtype=torch.float32, device=device)
            batch_src = torch.zeros(x_src.shape[0], dtype=torch.long, device=device)

            x_tgt = torch.tensor(graph_tgt.node_features, dtype=torch.float32, device=device)
            edge_index_tgt = torch.tensor(graph_tgt.edge_index, dtype=torch.long, device=device)
            edge_attr_tgt = torch.tensor(graph_tgt.edge_features, dtype=torch.float32, device=device)
            batch_tgt = torch.zeros(x_tgt.shape[0], dtype=torch.long, device=device)

            mu_src, _ = vae.encode(x_src, edge_index_src, edge_attr_src, batch_src)
            mu_tgt, _ = vae.encode(x_tgt, edge_index_tgt, edge_attr_tgt, batch_tgt)

            z_src = mu_src.squeeze(0).cpu().numpy()
            z_tgt = mu_tgt.squeeze(0).cpu().numpy()
            delta_z = z_tgt - z_src

        return {
            "instruction": instruction,
            "z_src": z_src.tolist(),
            "z_tgt": z_tgt.tolist(),
            "delta_z": delta_z.tolist(),
            "param_deltas": actual_deltas,
            "edit_type": "compound",
        }

    except Exception as e:
        print(f"Warning: Failed to generate compound sample: {e}")
        return None


def generate_noop_sample(
    vae: torch.nn.Module,
    rng: np.random.Generator,
    device: str,
) -> dict | None:
    """
    Generate a no-op sample (identity edit).

    Args:
        vae: Trained VAE model
        rng: Random number generator
        device: Device for VAE inference

    Returns:
        Sample dict or None if generation fails
    """
    try:
        # Sample L-bracket
        bracket = LBracket.random(rng)

        # Extract graph
        graph = extract_graph_from_solid(bracket.to_solid())

        # Encode through VAE
        with torch.no_grad():
            x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
            edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
            edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

            mu, _ = vae.encode(x, edge_index, edge_attr, batch)
            z = mu.squeeze(0).cpu().numpy()

        # Choose no-op instruction
        noop_instructions = ["keep it the same", "no changes", "leave it unchanged"]
        instruction = rng.choice(noop_instructions)

        return {
            "instruction": instruction,
            "z_src": z.tolist(),
            "z_tgt": z.tolist(),
            "delta_z": [0.0] * len(z),
            "param_deltas": {},
            "edit_type": "noop",
        }

    except Exception as e:
        print(f"Warning: Failed to generate noop sample: {e}")
        return None


def generate_dataset(
    vae: torch.nn.Module,
    num_samples: int,
    device: str,
    seed: int = 42,
    single_ratio: float = 0.7,
    compound_ratio: float = 0.2,
    noop_ratio: float = 0.1,
) -> list[dict]:
    """
    Generate full dataset of edit samples.

    Args:
        vae: Trained VAE model
        num_samples: Total number of samples to generate
        device: Device for VAE inference
        seed: Random seed
        single_ratio: Fraction of single-parameter edits
        compound_ratio: Fraction of compound edits
        noop_ratio: Fraction of no-op samples

    Returns:
        List of sample dictionaries
    """
    rng = np.random.default_rng(seed)
    samples = []

    # Calculate counts
    num_single = int(num_samples * single_ratio)
    num_compound = int(num_samples * compound_ratio)
    num_noop = num_samples - num_single - num_compound

    print(f"Generating {num_single} single edits, {num_compound} compound edits, {num_noop} no-ops")

    # Generate single-parameter edits
    print("Generating single-parameter edits...")
    pbar = tqdm(total=num_single, desc="Single edits")
    attempts = 0
    max_attempts = num_single * 3

    while len([s for s in samples if s.get("edit_type") == "single"]) < num_single:
        sample = generate_single_edit_sample(vae, rng, device)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Could only generate {len(samples)} single samples after {attempts} attempts")
            break
    pbar.close()

    # Generate compound edits
    print("Generating compound edits...")
    pbar = tqdm(total=num_compound, desc="Compound edits")
    attempts = 0
    max_attempts = num_compound * 3

    while len([s for s in samples if s.get("edit_type") == "compound"]) < num_compound:
        sample = generate_compound_edit_sample(vae, rng, device)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Could only generate {len([s for s in samples if s.get('edit_type') == 'compound'])} compound samples")
            break
    pbar.close()

    # Generate no-op samples
    print("Generating no-op samples...")
    pbar = tqdm(total=num_noop, desc="No-op edits")
    attempts = 0
    max_attempts = num_noop * 3

    while len([s for s in samples if s.get("edit_type") == "noop"]) < num_noop:
        sample = generate_noop_sample(vae, rng, device)
        if sample is not None:
            samples.append(sample)
            pbar.update(1)
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Could only generate {len([s for s in samples if s.get('edit_type') == 'noop'])} noop samples")
            break
    pbar.close()

    # Shuffle
    rng.shuffle(samples)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate latent edit training data")
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_16d/best_model.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Total number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/edit_data",
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

    # Load VAE
    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae, checkpoint = load_checkpoint(args.vae_checkpoint, device=args.device)
    vae.eval()
    print(f"  Latent dim: {vae.config.latent_dim}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    samples = generate_dataset(
        vae=vae,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
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
    metadata = {
        "vae_checkpoint": args.vae_checkpoint,
        "latent_dim": vae.config.latent_dim,
        "num_samples": len(samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
