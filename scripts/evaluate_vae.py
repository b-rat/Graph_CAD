#!/usr/bin/env python3
"""
Evaluate and visualize trained Graph VAE.

This script provides various analysis and visualization tools for
understanding the trained VAE's reconstruction quality and latent space.

Usage:
    python scripts/evaluate_vae.py --checkpoint outputs/vae/best_model.pt
    python scripts/evaluate_vae.py --checkpoint outputs/vae/best_model.pt --visualize-latent
    python scripts/evaluate_vae.py --checkpoint outputs/vae/best_model.pt --interpolate --num-pairs 5
    python scripts/evaluate_vae.py --checkpoint outputs/vae/best_model.pt --device mps
"""

from __future__ import annotations

# Set MPS memory limit before importing torch (prevents OOM on Apple Silicon)
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from graph_cad.data import create_data_loaders
from graph_cad.training.vae_trainer import (
    compute_latent_metrics,
    evaluate,
    load_checkpoint,
    prepare_batch_targets,
)


def analyze_reconstruction(
    model,
    loader,
    device: str,
    num_samples: int = 5,
) -> dict:
    """
    Analyze reconstruction quality on sample graphs.

    Returns per-feature reconstruction errors and statistics.
    """
    model.eval()

    all_node_errors = []
    all_edge_errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            node_target, edge_target = prepare_batch_targets(
                batch,
                num_nodes=model.config.num_nodes,
                node_features=model.config.node_features,
                num_edges=model.config.num_edges,
                edge_features=model.config.edge_features,
            )

            # Per-sample, per-feature errors
            node_errors = (outputs["node_recon"] - node_target).abs()
            edge_errors = (outputs["edge_recon"] - edge_target).abs()

            all_node_errors.append(node_errors.cpu())
            all_edge_errors.append(edge_errors.cpu())

    # Concatenate
    all_node_errors = torch.cat(all_node_errors, dim=0)
    all_edge_errors = torch.cat(all_edge_errors, dim=0)

    # Compute statistics
    feature_names = [
        "face_type", "area", "dir_x", "dir_y", "dir_z",
        "centroid_x", "centroid_y", "centroid_z"
    ]

    node_stats = {}
    for i, name in enumerate(feature_names):
        errors = all_node_errors[..., i]
        node_stats[name] = {
            "mean": errors.mean().item(),
            "std": errors.std().item(),
            "max": errors.max().item(),
            "median": errors.median().item(),
        }

    edge_stats = {
        "edge_length": {
            "mean": all_edge_errors[..., 0].mean().item(),
            "std": all_edge_errors[..., 0].std().item(),
            "max": all_edge_errors[..., 0].max().item(),
        },
        "dihedral_angle": {
            "mean": all_edge_errors[..., 1].mean().item(),
            "std": all_edge_errors[..., 1].std().item(),
            "max": all_edge_errors[..., 1].max().item(),
        },
    }

    return {
        "node_features": node_stats,
        "edge_features": edge_stats,
        "overall_node_mae": all_node_errors.mean().item(),
        "overall_edge_mae": all_edge_errors.mean().item(),
    }


def analyze_latent_space(
    model,
    loader,
    device: str,
) -> dict:
    """
    Analyze latent space structure and properties.
    """
    model.eval()

    all_z = []
    all_mu = []
    all_logvar = []
    all_params = []  # Ground truth parameters if available

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, logvar = model.encode(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            z = model.reparameterize(mu, logvar)

            all_z.append(z.cpu())
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

            # Collect ground truth params if available
            if hasattr(batch, 'y') and batch.y is not None:
                all_params.append(batch.y.view(-1, 8).cpu())

    # Concatenate
    all_z = torch.cat(all_z, dim=0)
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    # Basic latent stats
    latent_dim = all_z.shape[1]
    per_dim_mean = all_z.mean(dim=0)
    per_dim_std = all_z.std(dim=0)
    per_dim_var = all_z.var(dim=0)

    # Active dimensions (variance > threshold)
    active_threshold = 0.1
    active_dims = (per_dim_std > active_threshold).sum().item()

    # Posterior collapse check (dimensions with very low variance)
    collapsed_threshold = 0.01
    collapsed_dims = (per_dim_std < collapsed_threshold).sum().item()

    # Average KL per dimension
    kl_per_dim = -0.5 * (1 + all_logvar.mean(dim=0) - all_mu.mean(dim=0).pow(2) - all_logvar.mean(dim=0).exp())

    analysis = {
        "num_samples": all_z.shape[0],
        "latent_dim": latent_dim,
        "active_dims": active_dims,
        "collapsed_dims": collapsed_dims,
        "mean_z_norm": all_z.norm(dim=1).mean().item(),
        "std_z_norm": all_z.norm(dim=1).std().item(),
        "per_dim_mean_range": [per_dim_mean.min().item(), per_dim_mean.max().item()],
        "per_dim_std_range": [per_dim_std.min().item(), per_dim_std.max().item()],
        "total_kl": kl_per_dim.sum().item(),
        "mean_kl_per_dim": kl_per_dim.mean().item(),
    }

    # Correlation with parameters if available
    if all_params:
        all_params = torch.cat(all_params, dim=0)

        # Compute correlation between each latent dim and each param
        correlations = np.corrcoef(
            all_z.numpy().T,
            all_params.numpy().T
        )[:latent_dim, latent_dim:]  # Shape: (latent_dim, 8)

        # Find most correlated latent dim for each parameter
        param_names = [
            "leg1_length", "leg2_length", "width", "thickness",
            "hole1_dist", "hole1_diam", "hole2_dist", "hole2_diam"
        ]
        best_correlations = {}
        for i, name in enumerate(param_names):
            best_dim = np.argmax(np.abs(correlations[:, i]))
            best_corr = correlations[best_dim, i]
            best_correlations[name] = {
                "best_latent_dim": int(best_dim),
                "correlation": float(best_corr),
            }

        analysis["param_correlations"] = best_correlations

    return analysis


def test_interpolation(
    model,
    loader,
    device: str,
    num_pairs: int = 3,
    num_steps: int = 10,
) -> dict:
    """
    Test latent space interpolation quality.

    Returns statistics on how smoothly features change during interpolation.
    """
    model.eval()

    # Get sample latent vectors
    samples = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            mu, _ = model.encode(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            samples.append(mu)
            if len(samples) * mu.shape[0] >= num_pairs * 2:
                break

    all_z = torch.cat(samples, dim=0)

    results = []
    for i in range(num_pairs):
        z1 = all_z[i * 2]
        z2 = all_z[i * 2 + 1]

        # Interpolate
        with torch.no_grad():
            node_interp, edge_interp = model.interpolate(z1, z2, num_steps)

        # Check smoothness: compute step-to-step differences
        node_diffs = (node_interp[1:] - node_interp[:-1]).abs().mean(dim=(1, 2))
        edge_diffs = (edge_interp[1:] - edge_interp[:-1]).abs().mean(dim=(1, 2))

        # Smoothness = low variance in step sizes
        node_smoothness = node_diffs.std().item() / (node_diffs.mean().item() + 1e-8)
        edge_smoothness = edge_diffs.std().item() / (edge_diffs.mean().item() + 1e-8)

        results.append({
            "pair": i,
            "node_smoothness": node_smoothness,
            "edge_smoothness": edge_smoothness,
            "mean_node_step": node_diffs.mean().item(),
            "mean_edge_step": edge_diffs.mean().item(),
        })

    return {
        "num_pairs": num_pairs,
        "num_steps": num_steps,
        "pairs": results,
        "avg_node_smoothness": np.mean([r["node_smoothness"] for r in results]),
        "avg_edge_smoothness": np.mean([r["edge_smoothness"] for r in results]),
    }


def test_sampling(
    model,
    device: str,
    num_samples: int = 100,
) -> dict:
    """
    Test quality of samples from the prior.

    Checks if sampled features fall within reasonable ranges.
    """
    model.eval()

    with torch.no_grad():
        node_samples, edge_samples = model.sample(num_samples, device)

    # Move to CPU for analysis
    node_samples = node_samples.cpu()
    edge_samples = edge_samples.cpu()

    # Check feature ranges
    # Face type should be close to 0 or 1
    face_types = node_samples[..., 0]
    valid_face_type = ((face_types >= -0.5) & (face_types <= 1.5)).float().mean().item()

    # Areas should be positive
    areas = node_samples[..., 1]
    valid_area = (areas > -0.1).float().mean().item()

    # Direction vectors should have reasonable magnitude
    directions = node_samples[..., 2:5]
    dir_norms = directions.norm(dim=-1)
    valid_direction = ((dir_norms > 0.5) & (dir_norms < 2.0)).float().mean().item()

    # Edge lengths should be positive
    edge_lengths = edge_samples[..., 0]
    valid_edge_length = (edge_lengths > -0.1).float().mean().item()

    # Dihedral angles should be in [0, pi]
    dihedral = edge_samples[..., 1]
    valid_dihedral = ((dihedral >= -0.5) & (dihedral <= 3.5)).float().mean().item()

    return {
        "num_samples": num_samples,
        "valid_face_type_ratio": valid_face_type,
        "valid_area_ratio": valid_area,
        "valid_direction_ratio": valid_direction,
        "valid_edge_length_ratio": valid_edge_length,
        "valid_dihedral_ratio": valid_dihedral,
        "overall_validity": np.mean([
            valid_face_type, valid_area, valid_direction,
            valid_edge_length, valid_dihedral
        ]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize trained Graph VAE"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of test samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for results JSON",
    )

    # Analysis options
    parser.add_argument(
        "--analyze-reconstruction",
        action="store_true",
        help="Analyze reconstruction quality per feature",
    )
    parser.add_argument(
        "--analyze-latent",
        action="store_true",
        help="Analyze latent space structure",
    )
    parser.add_argument(
        "--test-interpolation",
        action="store_true",
        help="Test interpolation smoothness",
    )
    parser.add_argument(
        "--test-sampling",
        action="store_true",
        help="Test sampling from prior",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=5,
        help="Number of pairs for interpolation test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses",
    )

    # Device argument
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection (auto-detect if not specified)
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, checkpoint = load_checkpoint(str(args.checkpoint), device)
    model.eval()

    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"Config: latent_dim={model.config.latent_dim}")

    # Create test loader
    print(f"\nCreating test dataset (size={args.test_size})...")
    _, _, test_loader = create_data_loaders(
        train_size=100,  # Minimal, not used
        val_size=100,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    results = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "config": {
            "latent_dim": model.config.latent_dim,
            "hidden_dim": model.config.hidden_dim,
        },
    }

    # Basic evaluation
    print("\nComputing basic metrics...")
    basic_metrics = evaluate(model, test_loader, beta=1.0, device=device)
    latent_metrics = compute_latent_metrics(model, test_loader, device)

    print(f"  Recon Loss: {basic_metrics['recon_loss']:.4f}")
    print(f"  Node MSE:   {basic_metrics['node_mse']:.6f}")
    print(f"  Edge MSE:   {basic_metrics['edge_mse']:.6f}")
    print(f"  Active dims: {latent_metrics['active_dims']}/{model.config.latent_dim}")

    results["basic_metrics"] = basic_metrics
    results["latent_metrics"] = latent_metrics

    # Optional analyses
    if args.all or args.analyze_reconstruction:
        print("\nAnalyzing reconstruction quality...")
        recon_analysis = analyze_reconstruction(model, test_loader, device)
        print(f"  Overall Node MAE: {recon_analysis['overall_node_mae']:.6f}")
        print(f"  Overall Edge MAE: {recon_analysis['overall_edge_mae']:.6f}")
        print("  Per-feature Node MAE:")
        for name, stats in recon_analysis["node_features"].items():
            print(f"    {name:12s}: {stats['mean']:.6f} (std={stats['std']:.6f})")
        results["reconstruction_analysis"] = recon_analysis

    if args.all or args.analyze_latent:
        print("\nAnalyzing latent space...")
        latent_analysis = analyze_latent_space(model, test_loader, device)
        print(f"  Active dims: {latent_analysis['active_dims']}")
        print(f"  Collapsed dims: {latent_analysis['collapsed_dims']}")
        print(f"  Mean ||z||: {latent_analysis['mean_z_norm']:.3f}")
        print(f"  Total KL: {latent_analysis['total_kl']:.2f}")
        if "param_correlations" in latent_analysis:
            print("  Parameter correlations:")
            for name, info in latent_analysis["param_correlations"].items():
                print(f"    {name:12s}: dim {info['best_latent_dim']:2d}, r={info['correlation']:.3f}")
        results["latent_analysis"] = latent_analysis

    if args.all or args.test_interpolation:
        print(f"\nTesting interpolation ({args.num_pairs} pairs)...")
        interp_results = test_interpolation(
            model, test_loader, device, num_pairs=args.num_pairs
        )
        print(f"  Avg node smoothness: {interp_results['avg_node_smoothness']:.4f}")
        print(f"  Avg edge smoothness: {interp_results['avg_edge_smoothness']:.4f}")
        results["interpolation"] = interp_results

    if args.all or args.test_sampling:
        print("\nTesting prior sampling...")
        sampling_results = test_sampling(model, device, num_samples=100)
        print(f"  Overall validity: {sampling_results['overall_validity']:.1%}")
        print(f"  Valid face types: {sampling_results['valid_face_type_ratio']:.1%}")
        print(f"  Valid directions: {sampling_results['valid_direction_ratio']:.1%}")
        results["sampling"] = sampling_results

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = args.checkpoint.parent / "evaluation_results.json"

    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
