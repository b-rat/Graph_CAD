#!/usr/bin/env python3
"""
Diagnostic script to test VAE reconstruction quality.

Tests whether the VAE can reconstruct:
1. Samples from the training data (should work if VAE trained correctly)
2. New random samples (tests generalization)

This helps diagnose whether inference issues are due to:
- Pipeline bugs (training samples also fail)
- Generalization failure (only new samples fail)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from graph_cad.data.l_bracket import VariableLBracket
from graph_cad.data.graph_extraction import extract_graph_from_solid_variable
from graph_cad.training.vae_trainer import load_checkpoint
from graph_cad.utils.geometric_solver import solve_params_from_features


def load_training_sample(data_dir: Path, sample_idx: int = 0):
    """Load a sample from the training data."""
    # Load the npz files
    node_features = np.load(data_dir / "node_features.npy")
    face_types = np.load(data_dir / "face_types.npy")
    node_masks = np.load(data_dir / "node_masks.npy")
    edge_features = np.load(data_dir / "edge_features.npy")
    edge_indices = np.load(data_dir / "edge_indices.npy")
    edge_masks = np.load(data_dir / "edge_masks.npy")
    params = np.load(data_dir / "params.npy")

    # Get a single sample
    return {
        "node_features": node_features[sample_idx],
        "face_types": face_types[sample_idx],
        "node_mask": node_masks[sample_idx],
        "edge_features": edge_features[sample_idx],
        "edge_index": edge_indices[sample_idx],
        "edge_mask": edge_masks[sample_idx],
        "params": params[sample_idx],
    }


def encode_decode_training_sample(vae, sample, device):
    """Encode and decode a training sample (already padded)."""
    # Convert to tensors
    x = torch.tensor(sample["node_features"], dtype=torch.float32, device=device).unsqueeze(0)
    face_types = torch.tensor(sample["face_types"], dtype=torch.long, device=device).unsqueeze(0)
    node_mask = torch.tensor(sample["node_mask"], dtype=torch.float32, device=device).unsqueeze(0)
    edge_attr = torch.tensor(sample["edge_features"], dtype=torch.float32, device=device).unsqueeze(0)
    edge_index = torch.tensor(sample["edge_index"], dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        # Encode - need to reshape for encoder
        # The encoder expects (num_nodes, features) not (batch, num_nodes, features)
        # But we need batch dimension for proper handling
        x_flat = x.squeeze(0)
        face_types_flat = face_types.squeeze(0)
        node_mask_flat = node_mask.squeeze(0)
        edge_attr_flat = edge_attr.squeeze(0)

        # Find actual edges (non-padded)
        num_edges = int(sample["edge_mask"].sum())
        edge_index_flat = edge_index.squeeze(0)[:, :num_edges]
        edge_attr_flat = edge_attr_flat[:num_edges]

        mu, logvar = vae.encode(
            x_flat, face_types_flat, edge_index_flat, edge_attr_flat,
            batch=None, node_mask=node_mask_flat
        )
        z = mu  # Use mean for deterministic reconstruction

        # Decode
        decoded = vae.decode(z)

    return {
        "z": z.cpu().numpy(),
        "decoded_node_features": decoded["node_features"].cpu().numpy()[0],
        "decoded_face_types": decoded["face_type_logits"].argmax(dim=-1).cpu().numpy()[0],
        "decoded_node_mask": (torch.sigmoid(decoded["node_mask_logits"]) > 0.5).cpu().numpy()[0],
    }


def encode_decode_new_sample(vae, bracket, device):
    """Encode and decode a new random sample."""
    solid = bracket.to_solid()
    graph = extract_graph_from_solid_variable(solid)

    # Get raw features
    gt_features = graph.node_features
    gt_face_types = graph.face_types
    num_nodes = len(gt_features)
    num_edges = graph.edge_index.shape[1]

    # Pad to match training format
    max_nodes = vae.config.max_nodes
    max_edges = vae.config.max_edges

    x_padded = np.zeros((max_nodes, gt_features.shape[1]), dtype=np.float32)
    x_padded[:num_nodes] = gt_features

    face_types_padded = np.zeros(max_nodes, dtype=np.int64)
    face_types_padded[:num_nodes] = gt_face_types

    node_mask = np.zeros(max_nodes, dtype=np.float32)
    node_mask[:num_nodes] = 1.0

    edge_attr_padded = np.zeros((max_edges, graph.edge_features.shape[1]), dtype=np.float32)
    edge_attr_padded[:num_edges] = graph.edge_features

    edge_index_padded = np.zeros((2, max_edges), dtype=np.int64)
    edge_index_padded[:, :num_edges] = graph.edge_index

    # Convert to tensors
    x = torch.tensor(x_padded, dtype=torch.float32, device=device)
    face_types_t = torch.tensor(face_types_padded, dtype=torch.long, device=device)
    node_mask_t = torch.tensor(node_mask, dtype=torch.float32, device=device)
    edge_attr = torch.tensor(edge_attr_padded, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index_padded, dtype=torch.long, device=device)

    with torch.no_grad():
        mu, logvar = vae.encode(
            x, face_types_t, edge_index[:, :num_edges], edge_attr[:num_edges],
            batch=None, node_mask=node_mask_t
        )
        z = mu

        decoded = vae.decode(z)

    return {
        "z": z.cpu().numpy(),
        "gt_features": gt_features,
        "gt_face_types": gt_face_types,
        "gt_params": bracket.to_dict(),
        "num_nodes": num_nodes,
        "decoded_node_features": decoded["node_features"].cpu().numpy()[0],
        "decoded_face_types": decoded["face_type_logits"].argmax(dim=-1).cpu().numpy()[0],
        "decoded_node_mask": (torch.sigmoid(decoded["node_mask_logits"]) > 0.5).cpu().numpy()[0],
    }


def compute_reconstruction_metrics(gt_features, decoded_features, node_mask):
    """Compute reconstruction metrics."""
    num_valid = int(node_mask.sum())

    # Compare only valid nodes
    gt = gt_features[:num_valid]
    dec = decoded_features[:num_valid]

    mse = np.mean((gt - dec) ** 2)
    mae = np.mean(np.abs(gt - dec))

    # Per-feature MSE
    feature_names = ["area", "dir_x", "dir_y", "dir_z", "cx", "cy", "cz",
                     "curv1", "curv2", "bbox_diag", "bbox_cx", "bbox_cy", "bbox_cz"]
    feature_mse = {}
    for i, name in enumerate(feature_names):
        feature_mse[name] = np.mean((gt[:, i] - dec[:, i]) ** 2)

    return {
        "mse": mse,
        "mae": mae,
        "num_valid": num_valid,
        "feature_mse": feature_mse,
    }


def test_geometric_solver(features, face_types, node_mask, gt_params=None):
    """Test geometric solver on features."""
    try:
        solved = solve_params_from_features(
            node_features=features,
            face_types=face_types,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, 2), dtype=np.float32),
            node_mask=node_mask.astype(np.float32) if node_mask is not None else None,
        )
        result = {
            "leg1_length": solved.leg1_length,
            "leg2_length": solved.leg2_length,
            "width": solved.width,
            "thickness": solved.thickness,
        }
        if gt_params:
            result["errors"] = {
                "leg1": solved.leg1_length - gt_params["leg1_length"],
                "leg2": solved.leg2_length - gt_params["leg2_length"],
                "width": solved.width - gt_params["width"],
                "thickness": solved.thickness - gt_params["thickness"],
            }
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Diagnose VAE reconstruction")
    parser.add_argument("--vae-checkpoint", type=str, required=True)
    parser.add_argument("--training-data", type=str, default="data/edit_data_variable_13d")
    parser.add_argument("--num-training-samples", type=int, default=5)
    parser.add_argument("--num-new-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # Load VAE
    print(f"\nLoading VAE from {args.vae_checkpoint}...")
    vae, checkpoint = load_checkpoint(args.vae_checkpoint, device=device)
    vae.eval()
    print(f"  VAE config: max_nodes={vae.config.max_nodes}, latent_dim={vae.config.latent_dim}")

    # Check if training data exists
    training_data_path = Path(args.training_data)
    has_training_data = (training_data_path / "node_features.npy").exists()

    print("\n" + "=" * 70)
    print("TEST 1: Reconstruction on TRAINING DATA samples")
    print("=" * 70)

    if has_training_data:
        for i in range(args.num_training_samples):
            print(f"\n--- Training Sample {i} ---")
            sample = load_training_sample(training_data_path, i)

            # Ground truth params
            param_names = ["leg1_length", "leg2_length", "width", "thickness"]
            gt_params = {name: sample["params"][j] for j, name in enumerate(param_names)}
            print(f"GT params: leg1={gt_params['leg1_length']:.1f}, leg2={gt_params['leg2_length']:.1f}, "
                  f"width={gt_params['width']:.1f}, thickness={gt_params['thickness']:.1f}")

            # Encode and decode
            result = encode_decode_training_sample(vae, sample, device)

            # Compute metrics
            num_valid = int(sample["node_mask"].sum())
            metrics = compute_reconstruction_metrics(
                sample["node_features"][:num_valid],
                result["decoded_node_features"][:num_valid],
                sample["node_mask"][:num_valid]
            )
            print(f"Reconstruction MSE: {metrics['mse']:.6f}")
            print(f"Reconstruction MAE: {metrics['mae']:.6f}")

            # Test geometric solver on ground truth
            gt_solved = test_geometric_solver(
                sample["node_features"],
                sample["face_types"],
                sample["node_mask"],
                gt_params
            )
            print(f"Solver on GT: leg1={gt_solved.get('leg1_length', 'ERR'):.1f}, "
                  f"leg2={gt_solved.get('leg2_length', 'ERR'):.1f}")

            # Test geometric solver on decoded
            dec_solved = test_geometric_solver(
                result["decoded_node_features"],
                result["decoded_face_types"],
                result["decoded_node_mask"],
                gt_params
            )
            print(f"Solver on Decoded: leg1={dec_solved.get('leg1_length', 'ERR'):.1f}, "
                  f"leg2={dec_solved.get('leg2_length', 'ERR'):.1f}")
            if "errors" in dec_solved:
                print(f"  Errors: leg1={dec_solved['errors']['leg1']:+.1f}, "
                      f"leg2={dec_solved['errors']['leg2']:+.1f}")
    else:
        print(f"Training data not found at {training_data_path}")
        print("Skipping training data test.")

    print("\n" + "=" * 70)
    print("TEST 2: Reconstruction on NEW RANDOM samples")
    print("=" * 70)

    rng = np.random.default_rng(args.seed)

    for i in range(args.num_new_samples):
        print(f"\n--- New Random Sample {i} ---")

        # Generate random bracket
        bracket = VariableLBracket.random(rng)
        gt_params = bracket.to_dict()
        print(f"GT params: leg1={gt_params['leg1_length']:.1f}, leg2={gt_params['leg2_length']:.1f}, "
              f"width={gt_params['width']:.1f}, thickness={gt_params['thickness']:.1f}, "
              f"holes={bracket.num_holes_leg1}+{bracket.num_holes_leg2}, fillet={bracket.has_fillet}")

        # Encode and decode
        result = encode_decode_new_sample(vae, bracket, device)

        # Compute metrics
        num_valid = result["num_nodes"]
        # Need to handle case where decoded has fewer valid nodes
        dec_num_valid = int(result["decoded_node_mask"].sum())
        compare_nodes = min(num_valid, dec_num_valid)

        if compare_nodes > 0:
            mse = np.mean((result["gt_features"][:compare_nodes] -
                          result["decoded_node_features"][:compare_nodes]) ** 2)
            print(f"Reconstruction MSE: {mse:.6f} (comparing {compare_nodes} nodes)")
        else:
            print("No valid nodes to compare!")

        # Z-vector stats
        z = result["z"]
        print(f"Latent z: norm={np.linalg.norm(z):.4f}, min={z.min():.3f}, max={z.max():.3f}")

        # Test geometric solver on ground truth
        gt_solved = test_geometric_solver(
            result["gt_features"],
            result["gt_face_types"],
            None,  # No mask for GT
            gt_params
        )
        print(f"Solver on GT: leg1={gt_solved.get('leg1_length', 'ERR'):.1f}, "
              f"leg2={gt_solved.get('leg2_length', 'ERR'):.1f}, "
              f"width={gt_solved.get('width', 'ERR'):.1f}, "
              f"thickness={gt_solved.get('thickness', 'ERR'):.1f}")

        # Test geometric solver on decoded
        dec_solved = test_geometric_solver(
            result["decoded_node_features"],
            result["decoded_face_types"],
            result["decoded_node_mask"],
            gt_params
        )
        print(f"Solver on Decoded: leg1={dec_solved.get('leg1_length', 'ERR'):.1f}, "
              f"leg2={dec_solved.get('leg2_length', 'ERR'):.1f}, "
              f"width={dec_solved.get('width', 'ERR'):.1f}, "
              f"thickness={dec_solved.get('thickness', 'ERR'):.1f}")
        if "errors" in dec_solved:
            print(f"  Errors: leg1={dec_solved['errors']['leg1']:+.1f}, "
                  f"leg2={dec_solved['errors']['leg2']:+.1f}, "
                  f"width={dec_solved['errors']['width']:+.1f}, "
                  f"thickness={dec_solved['errors']['thickness']:+.1f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
If training samples reconstruct well but new samples don't:
  → VAE is overfitting / not generalizing

If both training and new samples fail:
  → Inference pipeline bug (encoding/decoding mismatch with training)

If geometric solver fails on decoded but works on GT:
  → VAE reconstruction doesn't preserve geometric properties
""")


if __name__ == "__main__":
    main()
