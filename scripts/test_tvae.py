#!/usr/bin/env python3
"""
Test Transformer VAE latent space parameter correlations.

Checks if the learned latent space encodes L-bracket parameters
(leg1, leg2, width, thickness) by computing correlations between
latent dimensions and ground truth parameters.

Usage:
    python scripts/test_tvae.py
    python scripts/test_tvae.py path/to/model.pt
    python scripts/test_tvae.py --model path/to/model.pt --test-size 1000
"""

import argparse
import torch
import numpy as np
from scipy.stats import pearsonr
from graph_cad.data.dataset import create_variable_data_loaders
from graph_cad.models.graph_vae import VariableGraphVAEConfig, VariableGraphVAEEncoder
from graph_cad.models.transformer_decoder import TransformerDecoderConfig, TransformerGraphVAE


def main():
    parser = argparse.ArgumentParser(
        description="Test Transformer VAE latent space parameter correlations"
    )
    parser.add_argument(
        "model", nargs="?", default="outputs/vae_transformer/best_model.pt",
        help="Path to model checkpoint (default: outputs/vae_transformer/best_model.pt)"
    )
    parser.add_argument(
        "--model", "-m", dest="model_flag", default=None,
        help="Path to model checkpoint (alternative to positional arg)"
    )
    parser.add_argument(
        "--test-size", type=int, default=500,
        help="Number of test samples (default: 500)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    args = parser.parse_args()

    # Use --model flag if provided, otherwise use positional arg
    model_path = args.model_flag if args.model_flag else args.model

    # Load model
    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu")
    encoder = VariableGraphVAEEncoder(VariableGraphVAEConfig(**ckpt["encoder_config"]))

    # Load param_head info if available (new models with aux loss)
    use_param_head = ckpt.get("use_param_head", False)
    num_params = ckpt.get("num_params", 4)
    model = TransformerGraphVAE(
        encoder, TransformerDecoderConfig(**ckpt["decoder_config"]),
        use_param_head=use_param_head, num_params=num_params
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    if use_param_head:
        print(f"Model has param_head for {num_params} parameters")

    # Get test data
    print(f"Generating {args.test_size} test samples...")
    _, _, test_loader = create_variable_data_loaders(
        train_size=100, val_size=100, test_size=args.test_size,
        batch_size=args.batch_size
    )

    # Collect latents and parameters
    all_z, all_params = [], []
    with torch.no_grad():
        for batch in test_loader:
            mu, _ = model.encode(
                batch.x, batch.face_types, batch.edge_index, batch.edge_attr,
                batch.batch, batch.node_mask
            )
            all_z.append(mu)
            # batch.y may be flattened by PyG DataLoader, reshape to (batch_size, 4)
            y = batch.y
            if y.dim() == 1:
                y = y.view(-1, 4)
            all_params.append(y)

    z = torch.cat(all_z).numpy()
    params = torch.cat(all_params).numpy()

    print(f"Collected {len(z)} samples")
    print(f"Latent shape: {z.shape}, Params shape: {params.shape}")

    # Correlations
    param_names = ["leg1", "leg2", "width", "thickness"]
    print("\n" + "=" * 50)
    print("Parameter correlations with latent dimensions:")
    print("=" * 50)

    max_correlations = []
    for i, name in enumerate(param_names):
        correlations = [abs(pearsonr(z[:, j], params[:, i])[0]) for j in range(z.shape[1])]
        max_corr = max(correlations)
        max_dim = correlations.index(max_corr)
        max_correlations.append(max_corr)
        print(f"{name:12s}: r = {max_corr:.3f} (best dim: {max_dim})")

    print("=" * 50)

    # Overall assessment
    avg_corr = np.mean(max_correlations)
    print(f"\nAverage max correlation: {avg_corr:.3f}")

    if avg_corr > 0.7:
        print("SUCCESS: Latent space encodes parameters well!")
    elif avg_corr > 0.5:
        print("PARTIAL: Some parameter encoding, room for improvement")
    else:
        print("POOR: Latent space not encoding parameters effectively")


if __name__ == "__main__":
    main()
