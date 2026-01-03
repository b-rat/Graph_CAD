#!/usr/bin/env python3
"""
Evaluate and analyze trained Variable Topology Graph VAE.

Checks for:
- Latent dimension collapse (low variance dimensions)
- Parameter correlations with latent dimensions
- Topology clustering (face count, fillet, holes)
- Face type encoding quality

Usage:
    python scripts/evaluate_variable_vae.py --checkpoint outputs/vae_variable/best_model.pt
    python scripts/evaluate_variable_vae.py --checkpoint outputs/vae_variable/best_model.pt --all
"""

from __future__ import annotations

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from graph_cad.data.dataset import VariableLBracketDataset, VariableLBracketRanges
from graph_cad.models.graph_vae import VariableGraphVAE, VariableGraphVAEConfig


def load_checkpoint(checkpoint_path: str, device: str) -> tuple[VariableGraphVAE, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = VariableGraphVAEConfig(**checkpoint["config"])
    model = VariableGraphVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def analyze_latent_dimensions(
    model: VariableGraphVAE,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    Analyze latent space for dimension collapse.

    Returns per-dimension statistics and identifies collapsed dimensions.
    """
    all_z = []
    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            face_types = batch.face_types.to(device)
            node_mask = batch.node_mask.to(device)
            edge_mask = batch.edge_mask.to(device)

            # Encode (note: signature is x, face_types, edge_index, edge_attr, batch, node_mask)
            mu, logvar = model.encode(
                x, face_types, edge_index, edge_attr, batch_idx, node_mask
            )
            z = model.reparameterize(mu, logvar)

            all_z.append(z.cpu())
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    all_z = torch.cat(all_z, dim=0)
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    latent_dim = all_z.shape[1]

    # Per-dimension statistics
    per_dim_mean = all_z.mean(dim=0).numpy()
    per_dim_std = all_z.std(dim=0).numpy()
    per_dim_var = all_z.var(dim=0).numpy()

    # Identify collapsed dimensions
    collapse_threshold = 0.01
    active_threshold = 0.1

    collapsed_dims = np.where(per_dim_std < collapse_threshold)[0].tolist()
    active_dims = np.where(per_dim_std > active_threshold)[0].tolist()
    weak_dims = np.where((per_dim_std >= collapse_threshold) & (per_dim_std <= active_threshold))[0].tolist()

    # Posterior statistics
    mean_posterior_std = torch.exp(0.5 * all_logvar).mean(dim=0).numpy()

    # KL per dimension
    kl_per_dim = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp()).mean(dim=0).numpy()

    return {
        "num_samples": int(all_z.shape[0]),
        "latent_dim": latent_dim,
        "collapsed_dims": collapsed_dims,
        "active_dims": active_dims,
        "weak_dims": weak_dims,
        "num_collapsed": len(collapsed_dims),
        "num_active": len(active_dims),
        "num_weak": len(weak_dims),
        "per_dim_mean": per_dim_mean.tolist(),
        "per_dim_std": per_dim_std.tolist(),
        "per_dim_var": per_dim_var.tolist(),
        "mean_posterior_std": mean_posterior_std.tolist(),
        "kl_per_dim": kl_per_dim.tolist(),
        "total_kl": float(kl_per_dim.sum()),
        "mean_z_norm": float(all_z.norm(dim=1).mean()),
        "variance_preserved": float((per_dim_var / per_dim_var.max()).mean()),
    }


def analyze_parameter_correlations(
    model: VariableGraphVAE,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    Analyze correlation between latent dimensions and L-bracket parameters.
    """
    all_z = []
    all_params = []

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            face_types = batch.face_types.to(device)
            node_mask = batch.node_mask.to(device)
            edge_mask = batch.edge_mask.to(device)

            mu, logvar = model.encode(
                x, face_types, edge_index, edge_attr, batch_idx, node_mask
            )
            z = model.reparameterize(mu, logvar)

            all_z.append(z.cpu())
            # y is flattened by PyG, reshape to (batch_size, 4)
            batch_y = batch.y.cpu().view(-1, 4)
            all_params.append(batch_y)

    all_z = torch.cat(all_z, dim=0).numpy()
    all_params = torch.cat(all_params, dim=0).numpy()

    latent_dim = all_z.shape[1]
    num_params = all_params.shape[1]  # Should be 4

    # Parameter names (4 core params for variable topology)
    param_names = ["leg1_length", "leg2_length", "width", "thickness"][:num_params]

    # Compute full correlation matrix
    combined = np.concatenate([all_z, all_params], axis=1)
    corr_matrix = np.corrcoef(combined.T)
    z_param_corr = corr_matrix[:latent_dim, latent_dim:]  # (latent_dim, num_params)

    # Find best latent dim for each parameter
    best_correlations = {}
    for i, name in enumerate(param_names):
        best_dim = int(np.argmax(np.abs(z_param_corr[:, i])))
        best_corr = float(z_param_corr[best_dim, i])
        max_abs_corr = float(np.max(np.abs(z_param_corr[:, i])))
        best_correlations[name] = {
            "best_latent_dim": best_dim,
            "correlation": best_corr,
            "max_abs_correlation": max_abs_corr,
        }

    # Check for parameter entanglement (high correlation between different params' best dims)
    param_param_corr = corr_matrix[latent_dim:, latent_dim:]

    # Find entangled parameters (high correlation in latent space)
    entanglement = {}
    for i, name1 in enumerate(param_names):
        for j, name2 in enumerate(param_names):
            if i < j:
                # Correlation of their latent representations
                dim1 = best_correlations[name1]["best_latent_dim"]
                dim2 = best_correlations[name2]["best_latent_dim"]
                if dim1 == dim2:
                    entanglement[f"{name1}_{name2}"] = {
                        "shared_dim": dim1,
                        "param_correlation": float(param_param_corr[i, j]),
                    }

    return {
        "param_names": param_names,
        "best_correlations": best_correlations,
        "z_param_corr_matrix": z_param_corr.tolist(),
        "param_entanglement": entanglement,
    }


def analyze_topology_clustering(
    model: VariableGraphVAE,
    loader: DataLoader,
    device: str,
) -> dict:
    """
    Analyze if different topologies cluster in latent space.

    Groups samples by:
    - Number of faces (nodes)
    - Has fillet (face_type == 2)
    - Number of holes (face_type == 1)
    """
    topology_data = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_idx = batch.batch.to(device)
            face_types = batch.face_types.to(device)
            node_mask = batch.node_mask.to(device)
            edge_mask = batch.edge_mask.to(device)

            mu, _ = model.encode(
                x, face_types, edge_index, edge_attr, batch_idx, node_mask
            )

            # Extract topology info per sample
            batch_size = mu.shape[0]
            for i in range(batch_size):
                sample_mask = batch_idx == i
                sample_face_types = face_types[sample_mask]
                sample_node_mask = node_mask[sample_mask]

                # Only count real nodes
                real_face_types = sample_face_types[sample_node_mask.bool()]

                num_faces = int(real_face_types.shape[0])
                num_holes = int((real_face_types == 1).sum())
                has_fillet = int((real_face_types == 2).any())

                topology_key = f"faces={num_faces}_holes={num_holes}_fillet={has_fillet}"
                topology_data[topology_key].append(mu[i].cpu().numpy())

    # Compute centroid and spread for each topology
    topology_stats = {}
    all_centroids = []

    for key, vectors in topology_data.items():
        vectors = np.array(vectors)
        centroid = vectors.mean(axis=0)
        spread = vectors.std(axis=0).mean()
        all_centroids.append((key, centroid))

        topology_stats[key] = {
            "count": len(vectors),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "spread": float(spread),
        }

    # Compute inter-topology distances
    inter_distances = {}
    for i, (key1, c1) in enumerate(all_centroids):
        for j, (key2, c2) in enumerate(all_centroids):
            if i < j:
                dist = float(np.linalg.norm(c1 - c2))
                inter_distances[f"{key1}_vs_{key2}"] = dist

    # Check if similar topologies are closer
    avg_inter_dist = np.mean(list(inter_distances.values())) if inter_distances else 0

    return {
        "num_topology_groups": len(topology_data),
        "topology_stats": topology_stats,
        "inter_distances": inter_distances,
        "avg_inter_distance": float(avg_inter_dist),
    }


def print_dimension_analysis(analysis: dict):
    """Pretty print dimension analysis."""
    print("\n" + "="*60)
    print("LATENT DIMENSION ANALYSIS")
    print("="*60)

    print(f"\nSamples analyzed: {analysis['num_samples']}")
    print(f"Latent dimensions: {analysis['latent_dim']}")
    print(f"\nDimension health:")
    print(f"  Active (std > 0.1):    {analysis['num_active']}/{analysis['latent_dim']}")
    print(f"  Weak (0.01 < std < 0.1): {analysis['num_weak']}/{analysis['latent_dim']}")
    print(f"  Collapsed (std < 0.01):  {analysis['num_collapsed']}/{analysis['latent_dim']}")

    if analysis['collapsed_dims']:
        print(f"\n  ⚠️  Collapsed dimensions: {analysis['collapsed_dims']}")
    else:
        print(f"\n  ✓ No collapsed dimensions")

    print(f"\nVariance preserved: {analysis['variance_preserved']:.1%}")
    print(f"Total KL divergence: {analysis['total_kl']:.2f}")
    print(f"Mean ||z||: {analysis['mean_z_norm']:.3f}")

    # Show top/bottom variance dimensions
    stds = np.array(analysis['per_dim_std'])
    sorted_dims = np.argsort(stds)

    print(f"\nTop 5 most active dimensions:")
    for dim in sorted_dims[-5:][::-1]:
        print(f"  dim {dim:2d}: std={stds[dim]:.4f}")

    print(f"\nBottom 5 least active dimensions:")
    for dim in sorted_dims[:5]:
        print(f"  dim {dim:2d}: std={stds[dim]:.4f}")


def print_correlation_analysis(analysis: dict):
    """Pretty print correlation analysis."""
    print("\n" + "="*60)
    print("PARAMETER CORRELATION ANALYSIS")
    print("="*60)

    print("\nBest latent dimension for each parameter:")
    for name, info in analysis['best_correlations'].items():
        corr = info['correlation']
        abs_corr = info['max_abs_correlation']
        dim = info['best_latent_dim']
        strength = "✓ strong" if abs_corr > 0.5 else "⚠️ weak" if abs_corr > 0.2 else "❌ very weak"
        print(f"  {name:15s}: dim {dim:2d}, r={corr:+.3f} ({strength})")

    if analysis['param_entanglement']:
        print("\n⚠️  Entangled parameters (share same latent dim):")
        for pair, info in analysis['param_entanglement'].items():
            print(f"  {pair}: shared dim {info['shared_dim']}")


def print_topology_analysis(analysis: dict):
    """Pretty print topology clustering analysis."""
    print("\n" + "="*60)
    print("TOPOLOGY CLUSTERING ANALYSIS")
    print("="*60)

    print(f"\nTopology groups found: {analysis['num_topology_groups']}")
    print(f"Average inter-group distance: {analysis['avg_inter_distance']:.3f}")

    print("\nTopology group statistics:")
    for key, stats in sorted(analysis['topology_stats'].items()):
        print(f"  {key}: n={stats['count']}, spread={stats['spread']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Variable Topology Graph VAE"
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use",
    )

    # Analysis options
    parser.add_argument("--dimensions", action="store_true", help="Analyze latent dimensions")
    parser.add_argument("--correlations", action="store_true", help="Analyze parameter correlations")
    parser.add_argument("--topology", action="store_true", help="Analyze topology clustering")
    parser.add_argument("--all", action="store_true", help="Run all analyses")

    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.dimensions or args.correlations or args.topology or args.all):
        args.all = True

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model, checkpoint = load_checkpoint(str(args.checkpoint), device)
    print(f"Model loaded (latent_dim={model.config.latent_dim})")

    # Create test dataset
    print(f"\nCreating test dataset (size={args.test_size})...")
    dataset = VariableLBracketDataset(
        num_samples=args.test_size,
        max_nodes=model.config.max_nodes,
        max_edges=model.config.max_edges,
        seed=args.seed + 999,  # Different seed from training
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = {
        "checkpoint": str(args.checkpoint),
        "config": {
            "latent_dim": model.config.latent_dim,
            "max_nodes": model.config.max_nodes,
            "max_edges": model.config.max_edges,
        },
    }

    # Run analyses
    if args.all or args.dimensions:
        print("\nAnalyzing latent dimensions...")
        dim_analysis = analyze_latent_dimensions(model, loader, device)
        print_dimension_analysis(dim_analysis)
        results["dimension_analysis"] = dim_analysis

    if args.all or args.correlations:
        print("\nAnalyzing parameter correlations...")
        corr_analysis = analyze_parameter_correlations(model, loader, device)
        print_correlation_analysis(corr_analysis)
        results["correlation_analysis"] = corr_analysis

    if args.all or args.topology:
        print("\nAnalyzing topology clustering...")
        topo_analysis = analyze_topology_clustering(model, loader, device)
        print_topology_analysis(topo_analysis)
        results["topology_analysis"] = topo_analysis

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = args.checkpoint.parent / "latent_analysis.json"

    print(f"\n\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
