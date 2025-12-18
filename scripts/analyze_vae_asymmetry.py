#!/usr/bin/env python3
"""
Analyze VAE latent space for directional asymmetry.

Tests whether the VAE encodes "increase" and "decrease" directions symmetrically.
If the latent space is asymmetric, this could explain why the LLM struggles
with increase operations.

Hypothesis: Increase deltas have higher variance than decrease deltas in latent space.

Usage (requires torch_geometric - run on GPU machine):
    python scripts/analyze_vae_asymmetry.py --n-samples 100
    python scripts/analyze_vae_asymmetry.py --n-samples 200 --output outputs/vae_asymmetry.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_cad.data.l_bracket import LBracket
from graph_cad.data.graph_extraction import extract_graph_from_solid
from graph_cad.training.vae_trainer import load_checkpoint


# Parameter ranges
PARAM_RANGES = {
    'leg1_length': (50, 200),
    'leg2_length': (50, 200),
    'width': (20, 60),
    'thickness': (3, 12),
    'hole1_diameter': (4, 12),
    'hole2_diameter': (4, 12),
}

# Test magnitudes for each parameter
PARAM_MAGNITUDES = {
    'leg1_length': [10, 20, 30, 50],
    'leg2_length': [10, 20, 30, 50],
    'width': [5, 10, 15],
    'thickness': [1, 2, 3],
    'hole1_diameter': [1, 2, 3],
    'hole2_diameter': [1, 2, 3],
}


def load_vae(checkpoint_path: str, device: str):
    """Load VAE from checkpoint."""
    vae, _ = load_checkpoint(checkpoint_path, device=device)
    vae.eval()
    return vae


def graph_to_tensors(graph, device: str):
    """Convert BRepGraph to PyTorch tensors."""
    x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)
    return x, edge_index, edge_attr


def encode_bracket(vae, bracket: LBracket, device: str) -> np.ndarray:
    """Encode a bracket to latent space."""
    # Build the solid and extract graph
    solid = bracket.to_solid()
    graph = extract_graph_from_solid(solid)
    x, edge_index, edge_attr = graph_to_tensors(graph, device)

    with torch.no_grad():
        mu, _ = vae.encode(x, edge_index, edge_attr)

    return mu.cpu().numpy().flatten()


def generate_test_brackets(
    param: str,
    magnitude: float,
    n_samples: int,
    rng: np.random.Generator,
) -> list:
    """Generate triplets of (base, decreased, increased) brackets."""

    pmin, pmax = PARAM_RANGES[param]
    triplets = []
    attempts = 0
    max_attempts = n_samples * 50  # Try many times to get enough samples

    while len(triplets) < n_samples and attempts < max_attempts:
        attempts += 1

        try:
            # Generate a random valid bracket
            base_bracket = LBracket.random(rng)

            # Check if the target param has room for both increase and decrease
            base_value = getattr(base_bracket, param)

            if base_value - magnitude < pmin + 1:
                continue  # Not enough room to decrease
            if base_value + magnitude > pmax - 1:
                continue  # Not enough room to increase

            # Try to build to verify it's valid
            base_bracket.to_solid()

            # Create decreased version using with_modified (handles constraints)
            dec_bracket = base_bracket.with_modified(param, -magnitude, clamp=False)
            dec_bracket.to_solid()

            # Create increased version
            inc_bracket = base_bracket.with_modified(param, magnitude, clamp=False)
            inc_bracket.to_solid()

            # Verify the changes actually happened (not clamped)
            dec_value = getattr(dec_bracket, param)
            inc_value = getattr(inc_bracket, param)

            if abs((base_value - dec_value) - magnitude) > 0.1:
                continue  # Decrease was clamped
            if abs((inc_value - base_value) - magnitude) > 0.1:
                continue  # Increase was clamped

            triplets.append({
                'base': base_bracket,
                'decreased': dec_bracket,
                'increased': inc_bracket,
                'param': param,
                'magnitude': magnitude,
                'base_value': base_value,
            })
        except Exception as e:
            continue

    return triplets


def analyze_deltas(
    vae,
    triplets: list,
    device: str,
) -> dict:
    """Analyze latent deltas for increase vs decrease."""

    increase_deltas = []
    decrease_deltas = []

    for triplet in triplets:
        z_base = encode_bracket(vae, triplet['base'], device)
        z_dec = encode_bracket(vae, triplet['decreased'], device)
        z_inc = encode_bracket(vae, triplet['increased'], device)

        # Delta for decrease: z_dec - z_base (what we'd predict for "make smaller")
        delta_dec = z_dec - z_base

        # Delta for increase: z_inc - z_base (what we'd predict for "make larger")
        delta_inc = z_inc - z_base

        decrease_deltas.append(delta_dec)
        increase_deltas.append(delta_inc)

    decrease_deltas = np.array(decrease_deltas)
    increase_deltas = np.array(increase_deltas)

    # Compute statistics
    results = {
        'n_samples': len(triplets),

        # Magnitude statistics
        'dec_magnitude_mean': float(np.mean(np.linalg.norm(decrease_deltas, axis=1))),
        'dec_magnitude_std': float(np.std(np.linalg.norm(decrease_deltas, axis=1))),
        'inc_magnitude_mean': float(np.mean(np.linalg.norm(increase_deltas, axis=1))),
        'inc_magnitude_std': float(np.std(np.linalg.norm(increase_deltas, axis=1))),

        # Per-dimension variance
        'dec_per_dim_var': float(np.mean(np.var(decrease_deltas, axis=0))),
        'inc_per_dim_var': float(np.mean(np.var(increase_deltas, axis=0))),

        # Consistency: are deltas pointing in consistent directions?
        # Compute mean delta and measure alignment
        'dec_mean_delta': np.mean(decrease_deltas, axis=0).tolist(),
        'inc_mean_delta': np.mean(increase_deltas, axis=0).tolist(),

        # Alignment: cosine similarity of each delta to the mean delta
        'dec_alignment_mean': float(np.mean([
            np.dot(d, np.mean(decrease_deltas, axis=0)) /
            (np.linalg.norm(d) * np.linalg.norm(np.mean(decrease_deltas, axis=0)) + 1e-8)
            for d in decrease_deltas
        ])),
        'inc_alignment_mean': float(np.mean([
            np.dot(d, np.mean(increase_deltas, axis=0)) /
            (np.linalg.norm(d) * np.linalg.norm(np.mean(increase_deltas, axis=0)) + 1e-8)
            for d in increase_deltas
        ])),

        # Symmetry check: are increase and decrease deltas opposite?
        'inc_dec_cosine': float(np.mean([
            np.dot(increase_deltas[i], decrease_deltas[i]) /
            (np.linalg.norm(increase_deltas[i]) * np.linalg.norm(decrease_deltas[i]) + 1e-8)
            for i in range(len(triplets))
        ])),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze VAE latent space asymmetry")
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_aux/best_model.pt",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per parameter/magnitude combination",
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
        default=None,
        help="Output JSON file (optional)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load VAE
    print(f"Loading VAE from {args.vae_checkpoint}")
    vae = load_vae(args.vae_checkpoint, device)

    rng = np.random.default_rng(args.seed)

    all_results = {}

    print("\n" + "=" * 70)
    print("VAE LATENT SPACE ASYMMETRY ANALYSIS")
    print("=" * 70)

    # Analyze each parameter
    for param in PARAM_RANGES.keys():
        print(f"\n--- {param} ---")

        param_results = {}

        for magnitude in PARAM_MAGNITUDES[param]:
            triplets = generate_test_brackets(param, magnitude, args.n_samples, rng)

            if len(triplets) < 10:
                print(f"  {magnitude}mm: insufficient samples ({len(triplets)} generated)")
                continue

            print(f"  {magnitude}mm: generated {len(triplets)} triplets")

            results = analyze_deltas(vae, triplets, device)
            param_results[magnitude] = results

            # Print summary
            inc_var = results['inc_per_dim_var']
            dec_var = results['dec_per_dim_var']
            var_ratio = inc_var / dec_var if dec_var > 0 else float('inf')

            inc_align = results['inc_alignment_mean']
            dec_align = results['dec_alignment_mean']

            symmetry = results['inc_dec_cosine']

            print(f"  {magnitude:2d}mm: var_ratio={var_ratio:.2f} (inc/dec), "
                  f"alignment inc={inc_align:.2f} dec={dec_align:.2f}, "
                  f"symmetry={symmetry:.2f}")

        all_results[param] = param_results

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_inc_vars = []
    all_dec_vars = []
    all_inc_aligns = []
    all_dec_aligns = []
    all_symmetries = []

    for param, param_results in all_results.items():
        for mag, results in param_results.items():
            all_inc_vars.append(results['inc_per_dim_var'])
            all_dec_vars.append(results['dec_per_dim_var'])
            all_inc_aligns.append(results['inc_alignment_mean'])
            all_dec_aligns.append(results['dec_alignment_mean'])
            all_symmetries.append(results['inc_dec_cosine'])

    print(f"\nVariance (per latent dimension):")
    print(f"  Increase: mean={np.mean(all_inc_vars):.4f}, std={np.std(all_inc_vars):.4f}")
    print(f"  Decrease: mean={np.mean(all_dec_vars):.4f}, std={np.std(all_dec_vars):.4f}")
    print(f"  Ratio (inc/dec): {np.mean(all_inc_vars)/np.mean(all_dec_vars):.2f}")

    print(f"\nAlignment (consistency of delta direction):")
    print(f"  Increase: mean={np.mean(all_inc_aligns):.3f}")
    print(f"  Decrease: mean={np.mean(all_dec_aligns):.3f}")

    print(f"\nSymmetry (inc vs dec should be -1.0 if perfectly opposite):")
    print(f"  Mean cosine: {np.mean(all_symmetries):.3f}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    var_ratio = np.mean(all_inc_vars) / np.mean(all_dec_vars)
    if var_ratio > 1.2:
        print(f"⚠️  Increase deltas have {var_ratio:.1f}x higher variance than decrease.")
        print("   This could explain why the LLM struggles with increase operations.")
    elif var_ratio < 0.8:
        print(f"⚠️  Decrease deltas have {1/var_ratio:.1f}x higher variance than increase.")
        print("   This is unexpected given the LLM's decrease bias.")
    else:
        print("✓  Variance is roughly symmetric between increase and decrease.")

    align_diff = np.mean(all_dec_aligns) - np.mean(all_inc_aligns)
    if align_diff > 0.1:
        print(f"⚠️  Decrease deltas are more consistent (alignment diff: {align_diff:.2f}).")
        print("   Decrease directions are more predictable in latent space.")
    elif align_diff < -0.1:
        print(f"⚠️  Increase deltas are more consistent (alignment diff: {-align_diff:.2f}).")
    else:
        print("✓  Alignment is roughly symmetric.")

    mean_symmetry = np.mean(all_symmetries)
    if mean_symmetry > -0.8:
        print(f"⚠️  Inc/dec deltas are not opposite (cosine: {mean_symmetry:.2f}, should be -1.0).")
        print("   The VAE doesn't encode increase/decrease as mirror operations.")
    else:
        print(f"✓  Inc/dec deltas are reasonably opposite (cosine: {mean_symmetry:.2f}).")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
