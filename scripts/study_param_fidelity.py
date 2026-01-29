#!/usr/bin/env python3
"""
Parameter Fidelity Study for Phase 4.

Tests how accurately the VAE param_head can reconstruct parameters
from the latent space, without any instruction-based edits.

For each geometry type:
1. Generate N random geometries
2. Encode through VAE to latent z
3. Predict parameters from z via param_head
4. Compare predicted vs original parameters

Saves results to outputs/param_fidelity_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
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
from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig


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
    rng = np.random.default_rng(seed)
    cls = GEOMETRY_CLASSES[geometry_type]
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
            geometry.outer_dia,
            geometry.inner_dia,
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
            geometry.hole_dia,
            geometry.hole_x,
            geometry.hole_y,
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")


def load_vae(checkpoint_path: str, device: str) -> HeteroVAE:
    """Load trained HeteroVAE model."""
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


def test_single_geometry(
    vae: HeteroVAE,
    geometry_type: int,
    seed: int,
    device: str,
) -> dict:
    """Test parameter reconstruction for a single geometry."""
    # Create geometry
    geometry = create_random_geometry(geometry_type, seed)

    # Get original parameters
    original_params = geometry_to_params(geometry, geometry_type)
    original_params_norm = normalize_params(original_params, geometry_type)

    # Extract B-Rep graph
    try:
        solid = geometry.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)
    except Exception as e:
        return {
            'error': str(e),
            'seed': seed,
        }

    # Convert to HeteroData
    num_params = len(original_params)
    params_padded = torch.zeros(6)
    params_padded[:num_params] = original_params
    params_norm_padded = torch.zeros(6)
    params_norm_padded[:num_params] = original_params_norm
    params_mask = torch.zeros(6)
    params_mask[:num_params] = 1.0

    data = brep_graph_to_hetero_data(
        graph, geometry_type, params_padded, params_norm_padded, params_mask
    )
    data = data.to(device)

    # Encode to latent
    with torch.no_grad():
        mu, logvar = vae.encode(data)
        z = mu  # Use mean for inference

        # Predict parameters from param_head
        param_pred = vae.param_head(mu)[0]  # (6,)

        # Classify geometry type
        geo_logits = vae.geometry_type_head(mu)
        pred_type = geo_logits.argmax(dim=-1).item()

    # Denormalize predicted parameters
    pred_params_norm = param_pred[:num_params].cpu()
    pred_params = denormalize_params(pred_params_norm, geometry_type)

    # Compute errors
    abs_errors = (original_params - pred_params).abs()
    rel_errors = abs_errors / (original_params.abs() + 1e-6)
    norm_errors = (original_params_norm - pred_params_norm).abs()

    param_names = PARAM_NAMES[geometry_type]

    return {
        'seed': seed,
        'geometry_type': GEOMETRY_TYPE_NAMES[geometry_type],
        'predicted_type': GEOMETRY_TYPE_NAMES[pred_type],
        'type_correct': pred_type == geometry_type,
        'params': {
            name: {
                'original': float(original_params[i]),
                'predicted': float(pred_params[i]),
                'abs_error': float(abs_errors[i]),
                'rel_error': float(rel_errors[i]),
                'norm_error': float(norm_errors[i]),
            }
            for i, name in enumerate(param_names)
        },
        'summary': {
            'mae': float(abs_errors.mean()),
            'max_error': float(abs_errors.max()),
            'mean_rel_error': float(rel_errors.mean()),
            'mean_norm_error': float(norm_errors.mean()),
        }
    }


def run_study(
    vae: HeteroVAE,
    samples_per_type: int,
    device: str,
    base_seed: int = 42,
) -> dict:
    """Run parameter fidelity study across all geometry types."""
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'samples_per_type': samples_per_type,
            'base_seed': base_seed,
        },
        'by_geometry_type': {},
        'summary': {},
    }

    all_maes = []
    all_norm_errors = []
    type_correct_count = 0
    total_count = 0

    for geo_type in range(6):
        geo_name = GEOMETRY_TYPE_NAMES[geo_type]
        print(f"\nTesting {geo_name.upper()}...")

        type_results = []
        type_maes = []
        type_norm_errors = []
        type_correct = 0

        for i in range(samples_per_type):
            seed = base_seed + geo_type * 1000 + i
            result = test_single_geometry(vae, geo_type, seed, device)

            if 'error' not in result:
                type_results.append(result)
                type_maes.append(result['summary']['mae'])
                type_norm_errors.append(result['summary']['mean_norm_error'])
                if result['type_correct']:
                    type_correct += 1
                all_maes.append(result['summary']['mae'])
                all_norm_errors.append(result['summary']['mean_norm_error'])
                total_count += 1
                if result['type_correct']:
                    type_correct_count += 1
            else:
                print(f"  Sample {i}: Error - {result['error']}")

        # Compute per-type statistics
        type_summary = {
            'samples': len(type_results),
            'type_accuracy': type_correct / len(type_results) if type_results else 0,
            'mae_mean': float(np.mean(type_maes)) if type_maes else None,
            'mae_std': float(np.std(type_maes)) if type_maes else None,
            'mae_min': float(np.min(type_maes)) if type_maes else None,
            'mae_max': float(np.max(type_maes)) if type_maes else None,
            'norm_error_mean': float(np.mean(type_norm_errors)) if type_norm_errors else None,
        }

        results['by_geometry_type'][geo_name] = {
            'summary': type_summary,
            'samples': type_results,
        }

        print(f"  Type accuracy: {type_summary['type_accuracy']*100:.1f}%")
        print(f"  MAE: {type_summary['mae_mean']:.2f} +/- {type_summary['mae_std']:.2f} mm")
        print(f"  Norm error: {type_summary['norm_error_mean']:.4f}")

    # Overall summary
    results['summary'] = {
        'total_samples': total_count,
        'overall_type_accuracy': type_correct_count / total_count if total_count else 0,
        'overall_mae_mean': float(np.mean(all_maes)) if all_maes else None,
        'overall_mae_std': float(np.std(all_maes)) if all_maes else None,
        'overall_norm_error_mean': float(np.mean(all_norm_errors)) if all_norm_errors else None,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Parameter Fidelity Study")

    parser.add_argument("--vae-checkpoint", type=str,
                        default="outputs/hetero_vae/best_model.pt",
                        help="Path to HeteroVAE checkpoint")
    parser.add_argument("--samples-per-type", type=int, default=10,
                        help="Number of samples per geometry type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for results")
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

    # Run study
    print(f"\nRunning parameter fidelity study...")
    print(f"  Samples per type: {args.samples_per_type}")
    print(f"  Total samples: {args.samples_per_type * 6}")

    results = run_study(
        vae,
        samples_per_type=args.samples_per_type,
        device=device,
        base_seed=args.seed,
    )

    # Save results
    output_path = Path(args.output_dir) / "param_fidelity_study.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("PARAMETER FIDELITY STUDY COMPLETE")
    print('='*60)
    print(f"\nOverall Results:")
    print(f"  Type Accuracy: {results['summary']['overall_type_accuracy']*100:.1f}%")
    print(f"  Mean Absolute Error: {results['summary']['overall_mae_mean']:.2f} +/- {results['summary']['overall_mae_std']:.2f} mm")
    print(f"  Mean Normalized Error: {results['summary']['overall_norm_error_mean']:.4f}")

    print(f"\nPer-Type MAE (mm):")
    for geo_name, geo_data in results['by_geometry_type'].items():
        summary = geo_data['summary']
        print(f"  {geo_name:12s}: {summary['mae_mean']:6.2f} +/- {summary['mae_std']:.2f}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
