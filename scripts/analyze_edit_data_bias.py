#!/usr/bin/env python3
"""
Analyze edit training data for systematic biases.

Checks:
1. Are increase/decrease deltas symmetric in magnitude?
2. Is there parameter-specific bias in delta directions?
3. Are there correlations between parameters in deltas?

Usage:
    python scripts/analyze_edit_data_bias.py --data-dir data/edit_data_aux
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_instruction(instruction: str) -> dict:
    """Extract parameter, direction, and magnitude from instruction."""
    instruction = instruction.lower()

    # Detect direction
    if any(word in instruction for word in ["longer", "increase", "larger", "bigger", "thicker", "wider"]):
        direction = "increase"
    elif any(word in instruction for word in ["shorter", "decrease", "smaller", "thinner", "narrower"]):
        direction = "decrease"
    else:
        direction = "unknown"

    # Detect parameter
    if "leg1" in instruction or "leg 1" in instruction:
        param = "leg1_length"
    elif "leg2" in instruction or "leg 2" in instruction:
        param = "leg2_length"
    elif "width" in instruction:
        param = "width"
    elif "thickness" in instruction or "thick" in instruction:
        param = "thickness"
    elif "hole1" in instruction or "hole 1" in instruction:
        param = "hole1_diameter"
    elif "hole2" in instruction or "hole 2" in instruction:
        param = "hole2_diameter"
    else:
        param = "unknown"

    return {"param": param, "direction": direction}


def analyze_data(data_path: Path):
    """Analyze training data for biases."""

    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {data_path}")

    # Group by parameter and direction
    deltas_by_param_dir = defaultdict(list)
    all_deltas = []

    for sample in data:
        instruction = sample["instruction"]
        delta = np.array(sample["delta_z"])

        parsed = parse_instruction(instruction)
        key = (parsed["param"], parsed["direction"])
        deltas_by_param_dir[key].append(delta)
        all_deltas.append(delta)

    all_deltas = np.array(all_deltas)

    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL DELTA STATISTICS")
    print("="*60)
    print(f"Mean delta (per dim): {all_deltas.mean(axis=0)}")
    print(f"Std delta (per dim): {all_deltas.std(axis=0)}")
    print(f"Mean delta norm: {np.linalg.norm(all_deltas, axis=1).mean():.4f}")

    # Per parameter-direction statistics
    print("\n" + "="*60)
    print("PER PARAMETER-DIRECTION ANALYSIS")
    print("="*60)

    params = ["leg1_length", "leg2_length", "width", "thickness", "hole1_diameter", "hole2_diameter"]

    for param in params:
        inc_key = (param, "increase")
        dec_key = (param, "decrease")

        inc_deltas = deltas_by_param_dir.get(inc_key, [])
        dec_deltas = deltas_by_param_dir.get(dec_key, [])

        if not inc_deltas or not dec_deltas:
            print(f"\n{param}: Missing data (inc={len(inc_deltas)}, dec={len(dec_deltas)})")
            continue

        inc_deltas = np.array(inc_deltas)
        dec_deltas = np.array(dec_deltas)

        # Mean deltas
        inc_mean = inc_deltas.mean(axis=0)
        dec_mean = dec_deltas.mean(axis=0)

        # Norms
        inc_norm = np.linalg.norm(inc_deltas, axis=1).mean()
        dec_norm = np.linalg.norm(dec_deltas, axis=1).mean()

        # Cosine similarity between mean increase and mean decrease
        cos_sim = np.dot(inc_mean, dec_mean) / (np.linalg.norm(inc_mean) * np.linalg.norm(dec_mean) + 1e-8)

        # Check if they're opposite (should be close to -1)
        print(f"\n{param}:")
        print(f"  Samples: increase={len(inc_deltas)}, decrease={len(dec_deltas)}")
        print(f"  Mean norm: increase={inc_norm:.4f}, decrease={dec_norm:.4f}, ratio={inc_norm/dec_norm:.2f}")
        print(f"  Cosine(inc_mean, dec_mean): {cos_sim:.3f} (should be ~-1 for opposite directions)")

        # Dominant dimensions
        inc_dominant = np.argmax(np.abs(inc_mean))
        dec_dominant = np.argmax(np.abs(dec_mean))
        print(f"  Dominant dim: increase=dim{inc_dominant} ({inc_mean[inc_dominant]:.3f}), decrease=dim{dec_dominant} ({dec_mean[dec_dominant]:.3f})")

    # Check for parameter confusion
    print("\n" + "="*60)
    print("PARAMETER DIRECTION SIMILARITY MATRIX")
    print("="*60)
    print("(Cosine similarity between mean delta vectors)")
    print("High positive = similar direction, High negative = opposite direction")
    print()

    # Compute mean delta for each param-direction combo
    mean_deltas = {}
    for param in params:
        for direction in ["increase", "decrease"]:
            key = (param, direction)
            if key in deltas_by_param_dir and deltas_by_param_dir[key]:
                mean_deltas[f"{param[:5]}_{direction[:3]}"] = np.array(deltas_by_param_dir[key]).mean(axis=0)

    # Print similarity matrix for key comparisons
    print("Key comparisons (same param, opposite direction - should be negative):")
    for param in params:
        inc_key = f"{param[:5]}_inc"
        dec_key = f"{param[:5]}_dec"
        if inc_key in mean_deltas and dec_key in mean_deltas:
            cos = np.dot(mean_deltas[inc_key], mean_deltas[dec_key]) / (
                np.linalg.norm(mean_deltas[inc_key]) * np.linalg.norm(mean_deltas[dec_key]) + 1e-8
            )
            status = "✓" if cos < -0.5 else "⚠️" if cos < 0 else "❌"
            print(f"  {param}: {cos:.3f} {status}")

    print("\nKey comparisons (different params - should be near 0 or negative):")
    print("  leg1_inc vs leg2_inc:", end=" ")
    if "leg1__inc" in mean_deltas and "leg2__inc" in mean_deltas:
        cos = np.dot(mean_deltas["leg1__inc"], mean_deltas["leg2__inc"]) / (
            np.linalg.norm(mean_deltas["leg1__inc"]) * np.linalg.norm(mean_deltas["leg2__inc"]) + 1e-8
        )
        print(f"{cos:.3f}")
    else:
        print("N/A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/edit_data_aux")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.json"

    if not train_path.exists():
        print(f"Error: {train_path} not found")
        return

    analyze_data(train_path)


if __name__ == "__main__":
    main()
