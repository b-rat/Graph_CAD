#!/usr/bin/env python3
"""
Inference script for the Latent Editor.

Takes a CAD model (STEP file or L-bracket parameters) and a natural language
instruction, runs inference through the full pipeline:
  1. Extract graph from CAD model
  2. Encode graph through VAE to get latent vector
  3. Apply instruction through Latent Editor to get edited latent
  4. Decode edited latent through VAE to get reconstructed graph
  5. Predict parameters from decoded features using FeatureRegressor
  6. Generate edited STEP file from predicted parameters

Usage:
    # Edit an existing STEP file
    python scripts/infer_latent_editor.py \
        --input bracket.step \
        --instruction "make leg1 20mm longer" \
        --output edited_bracket.step

    # Generate a random L-bracket and edit it
    python scripts/infer_latent_editor.py \
        --random-bracket \
        --instruction "make it wider" \
        --output edited_bracket.step

    # Specify L-bracket parameters directly
    python scripts/infer_latent_editor.py \
        --params "leg1_length=100,leg2_length=80,width=30,thickness=5" \
        --instruction "increase hole diameters by 2mm" \
        --output edited_bracket.step

    # VAE-only mode (skip LLM, for testing)
    python scripts/infer_latent_editor.py \
        --random-bracket \
        --instruction "test" \
        --vae-only --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_params(params_str: str) -> dict[str, float]:
    """Parse parameter string like 'leg1_length=100,width=30'."""
    params = {}
    for item in params_str.split(","):
        key, value = item.strip().split("=")
        params[key.strip()] = float(value.strip())
    return params


def load_vae(checkpoint_path: str, device: str):
    """Load trained VAE model."""
    from graph_cad.training.vae_trainer import load_checkpoint

    print(f"Loading VAE from {checkpoint_path}...")
    vae, checkpoint = load_checkpoint(checkpoint_path, device=device)
    vae.eval()
    print(f"  VAE latent dim: {vae.config.latent_dim}")
    return vae


def load_feature_regressor(checkpoint_path: str, device: str):
    """Load trained FeatureRegressor model."""
    from graph_cad.models.feature_regressor import load_feature_regressor as _load

    print(f"Loading FeatureRegressor from {checkpoint_path}...")
    regressor, checkpoint = _load(checkpoint_path, device=device)
    regressor.eval()
    print(f"  Config: {regressor.config.hidden_dims}")
    return regressor


def load_latent_regressor(checkpoint_path: str, device: str):
    """Load trained LatentRegressor model (predicts params directly from z)."""
    from dataclasses import dataclass
    import torch.nn as nn

    @dataclass
    class LatentRegressorConfig:
        latent_dim: int = 32
        hidden_dims: tuple[int, ...] = (256, 128, 64)
        dropout: float = 0.1
        use_batch_norm: bool = True
        num_params: int = 4

    class LatentRegressor(nn.Module):
        def __init__(self, config: LatentRegressorConfig):
            super().__init__()
            self.config = config
            layers = []
            in_dim = config.latent_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout))
                in_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.output_head = nn.Linear(in_dim, config.num_params)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = self.backbone(z)
            return self.output_head(h)

    print(f"Loading LatentRegressor from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = LatentRegressorConfig(**checkpoint["config"])
    regressor = LatentRegressor(config)
    regressor.load_state_dict(checkpoint["model_state_dict"])
    regressor.to(device)
    regressor.eval()
    print(f"  Config: {config.latent_dim}D → {config.hidden_dims} → {config.num_params}D")
    return regressor, config


def load_editor(checkpoint_path: str, config, device: str, force_cpu: bool = False):
    """Load trained Latent Editor model."""
    from graph_cad.models.latent_editor import (
        LatentEditor,
        LatentEditorConfig,
        load_llm_with_lora,
    )
    from graph_cad.training.edit_trainer import load_editor_checkpoint

    print(f"Loading Latent Editor from {checkpoint_path}...")

    # Load checkpoint first to get the saved config
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config")

    # Use saved config if available, else create new one
    if saved_config is not None:
        editor_config = saved_config
        # Override quantization based on device
        if device == "cuda" and not force_cpu:
            editor_config.use_4bit = True
            editor_config.use_8bit = False
        else:
            # CPU or MPS: disable quantization (not supported)
            editor_config.use_4bit = False
            editor_config.use_8bit = False
    else:
        editor_config = LatentEditorConfig(
            latent_dim=config.latent_dim if hasattr(config, "latent_dim") else 16,
            use_4bit=True if device == "cuda" and not force_cpu else False,
            use_8bit=False,
        )

    print(f"  Config: latent_dim={editor_config.latent_dim}, 4bit={editor_config.use_4bit}")

    # Create editor without LLM first
    editor = LatentEditor(editor_config)

    # Load LLM with LoRA
    print(f"  Loading {editor_config.model_name} with LoRA...")
    print(f"  This may take a few minutes and require significant memory...")

    # Determine device map
    if device == "cuda":
        device_map = "auto"
    elif device == "mps":
        # MPS doesn't support device_map="auto" well, load to CPU first
        device_map = {"": "cpu"}
        print(f"  Note: Loading to CPU first, then moving to MPS")
    else:
        device_map = {"": "cpu"}

    llm, tokenizer = load_llm_with_lora(editor_config, device_map=device_map)
    editor.set_llm(llm, tokenizer)

    # Load projector and LoRA weights from checkpoint
    from graph_cad.training.edit_trainer import load_editor_checkpoint
    load_editor_checkpoint(editor, checkpoint_path, device="cpu")
    print(f"  Loaded weights from epoch {checkpoint['epoch']}")

    # Move projectors to target device
    # Note: LLM stays where device_map put it
    if device == "mps":
        # For MPS, keep projectors on CPU to match LLM
        editor.latent_projector.to("cpu")
        editor.output_projector.to("cpu")
    else:
        editor.latent_projector.to(device)
        editor.output_projector.to(device)

    editor.eval()

    return editor, checkpoint


def graph_to_tensor(graph, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert BRepGraph to PyTorch tensors."""
    x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)
    return x, edge_index, edge_attr


def encode_graph(vae, graph, device: str) -> torch.Tensor:
    """Encode a BRepGraph to latent vector using VAE."""
    x, edge_index, edge_attr = graph_to_tensor(graph, device)

    with torch.no_grad():
        mu, logvar = vae.encode(x, edge_index, edge_attr, batch=None)
        # Use mean (deterministic) in inference mode
        z = mu

    return z


def decode_latent(vae, z: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Decode latent vector to graph features."""
    with torch.no_grad():
        node_recon, edge_recon = vae.decode(z)

    return node_recon.cpu().numpy()[0], edge_recon.cpu().numpy()[0]


def print_graph_comparison(
    original_graph,
    original_features: tuple[np.ndarray, np.ndarray],
    edited_features: tuple[np.ndarray, np.ndarray],
):
    """Print comparison of original and edited graph features."""
    orig_nodes, orig_edges = original_features
    edit_nodes, edit_edges = edited_features

    print("\n" + "=" * 60)
    print("GRAPH FEATURE COMPARISON")
    print("=" * 60)

    # Node feature comparison
    print("\nNode Features (8 per face: type, area, dir_x/y/z, cent_x/y/z):")
    print("-" * 60)
    node_diff = edit_nodes - orig_nodes
    for i in range(orig_nodes.shape[0]):
        diff_norm = np.linalg.norm(node_diff[i])
        if diff_norm > 0.01:  # Only show significant changes
            print(f"  Face {i}: Δ = {diff_norm:.4f}")
            print(f"    Original: {orig_nodes[i][:4]}")  # First 4 features
            print(f"    Edited:   {edit_nodes[i][:4]}")

    # Edge feature comparison
    print("\nEdge Features (2 per edge: length, dihedral_angle):")
    print("-" * 60)
    edge_diff = edit_edges - orig_edges
    significant_edge_changes = []
    for i in range(orig_edges.shape[0]):
        diff_norm = np.linalg.norm(edge_diff[i])
        if diff_norm > 0.01:
            significant_edge_changes.append((i, diff_norm, orig_edges[i], edit_edges[i]))

    if significant_edge_changes:
        for i, diff_norm, orig, edit in significant_edge_changes[:5]:  # Show top 5
            print(f"  Edge {i}: Δ = {diff_norm:.4f}")
            print(f"    Original: length={orig[0]:.4f}, angle={orig[1]:.4f}")
            print(f"    Edited:   length={edit[0]:.4f}, angle={edit[1]:.4f}")
    else:
        print("  No significant edge changes detected.")

    # Overall statistics
    print("\nOverall Statistics:")
    print("-" * 60)
    print(f"  Node MSE: {np.mean(node_diff**2):.6f}")
    print(f"  Edge MSE: {np.mean(edge_diff**2):.6f}")
    print(f"  Max node change: {np.max(np.abs(node_diff)):.4f}")
    print(f"  Max edge change: {np.max(np.abs(edge_diff)):.4f}")


def print_latent_comparison(z_src: torch.Tensor, z_edited: torch.Tensor, delta_z: torch.Tensor):
    """Print comparison of source and edited latent vectors."""
    z_src_np = z_src.cpu().numpy().flatten()
    z_edited_np = z_edited.cpu().numpy().flatten()
    delta_np = delta_z.cpu().numpy().flatten()

    print("\n" + "=" * 60)
    print("LATENT SPACE COMPARISON")
    print("=" * 60)

    print(f"\nLatent dimension: {len(z_src_np)}")
    print(f"Source ‖z‖: {np.linalg.norm(z_src_np):.4f}")
    print(f"Edited ‖z‖: {np.linalg.norm(z_edited_np):.4f}")
    print(f"Delta  ‖Δz‖: {np.linalg.norm(delta_np):.4f}")

    # Show top changes by dimension
    print("\nTop 5 latent dimension changes:")
    sorted_idx = np.argsort(np.abs(delta_np))[::-1]
    for i in sorted_idx[:5]:
        print(f"  dim {i:2d}: {z_src_np[i]:+.4f} → {z_edited_np[i]:+.4f} (Δ = {delta_np[i]:+.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the Latent Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to input STEP file",
    )
    input_group.add_argument(
        "--random-bracket",
        action="store_true",
        help="Generate a random L-bracket for testing",
    )
    input_group.add_argument(
        "--params",
        type=str,
        help="L-bracket parameters (e.g., 'leg1_length=100,width=30')",
    )

    # Instruction
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Natural language edit instruction",
    )

    # Model checkpoints
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_16d_lowbeta/best_model.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--editor-checkpoint",
        type=str,
        default="outputs/latent_editor/best_model.pt",
        help="Path to trained Latent Editor checkpoint",
    )
    parser.add_argument(
        "--regressor-checkpoint",
        type=str,
        default="outputs/feature_regressor/best_model.pt",
        help="Path to trained FeatureRegressor checkpoint (for parameter prediction)",
    )
    parser.add_argument(
        "--latent-regressor-checkpoint",
        type=str,
        default=None,
        help="Path to trained LatentRegressor checkpoint (predicts params directly from z, bypasses decoder)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save edited STEP file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--save-graphs",
        action="store_true",
        help="Save original and edited graph features to JSON",
    )

    # Display
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed comparison output",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution (no quantization)",
    )

    # Testing mode
    parser.add_argument(
        "--vae-only",
        action="store_true",
        help="Only run VAE encode/decode (skip LLM, for testing)",
    )

    # Seed for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random bracket generation",
    )

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load or create CAD model
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading CAD Model")
    print("=" * 60)

    from graph_cad.data import LBracket, extract_graph, extract_graph_from_solid

    if args.input:
        print(f"Loading STEP file: {args.input}")
        graph = extract_graph(args.input)
        bracket = None  # Don't have parameters for STEP file input
        print(f"  Extracted graph: {graph.num_faces} faces, {graph.num_edges} edges")

    elif args.random_bracket:
        print("Generating random L-bracket...")
        rng = np.random.default_rng(args.seed)
        bracket = LBracket.random(rng)
        print(f"  Parameters: {bracket.to_dict()}")
        solid = bracket.to_solid()
        graph = extract_graph_from_solid(solid)
        print(f"  Graph: {graph.num_faces} faces, {graph.num_edges} edges")

        # Save original STEP
        original_step = output_dir / "original.step"
        bracket.to_step(str(original_step))
        print(f"  Saved original to: {original_step}")

    else:  # --params
        print(f"Creating L-bracket from parameters: {args.params}")
        params = parse_params(args.params)
        bracket = LBracket(**params)
        print(f"  Full parameters: {bracket.to_dict()}")
        solid = bracket.to_solid()
        graph = extract_graph_from_solid(solid)
        print(f"  Graph: {graph.num_faces} faces, {graph.num_edges} edges")

        # Save original STEP
        original_step = output_dir / "original.step"
        bracket.to_step(str(original_step))
        print(f"  Saved original to: {original_step}")

    # =========================================================================
    # Step 2: Load models
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Loading Models")
    print("=" * 60)

    vae = load_vae(args.vae_checkpoint, device)

    editor = None
    if not args.vae_only:
        editor, editor_checkpoint = load_editor(
            args.editor_checkpoint, vae.config, device, force_cpu=args.force_cpu
        )
    else:
        print("  Skipping LLM loading (--vae-only mode)")

    # =========================================================================
    # Step 3: Encode original graph
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Encoding Original Graph")
    print("=" * 60)

    z_src = encode_graph(vae, graph, device)
    print(f"  Encoded to latent: z ∈ ℝ^{z_src.shape[-1]}")
    print(f"  ‖z‖ = {z_src.norm().item():.4f}")

    # Decode to verify reconstruction
    orig_node_recon, orig_edge_recon = decode_latent(vae, z_src)
    print(f"  Reconstruction check: node_shape={orig_node_recon.shape}, edge_shape={orig_edge_recon.shape}")

    # =========================================================================
    # Step 4: Apply edit instruction
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Applying Edit Instruction")
    print("=" * 60)
    print(f"  Instruction: \"{args.instruction}\"")

    if args.vae_only:
        print("  [VAE-only mode] Using zero delta (no edit)")
        delta_z = torch.zeros_like(z_src)
        z_edited = z_src.clone()
    else:
        # Move z_src to the device where editor expects it
        # For MPS/CPU mode, editor runs on CPU
        if device == "mps" or args.force_cpu:
            z_src_editor = z_src.cpu()
        else:
            z_src_editor = z_src

        with torch.no_grad():
            outputs = editor(z_src_editor, [args.instruction])
            delta_z = outputs["delta_z"]
            z_edited = outputs["z_edited"]

        # Move back to VAE device for decoding
        delta_z = delta_z.to(device)
        z_edited = z_edited.to(device)

    print(f"  Predicted delta: ‖Δz‖ = {delta_z.norm().item():.4f}")
    print(f"  Edited latent: ‖z'‖ = {z_edited.norm().item():.4f}")

    # =========================================================================
    # Step 5: Decode edited latent
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Decoding Edited Latent")
    print("=" * 60)

    edit_node_recon, edit_edge_recon = decode_latent(vae, z_edited)
    print(f"  Decoded graph: node_shape={edit_node_recon.shape}, edge_shape={edit_edge_recon.shape}")

    # =========================================================================
    # Step 6: Predict parameters from decoded features
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Predicting Parameters")
    print("=" * 60)

    orig_params_pred = None
    edit_params_pred = None
    predicted_original_bracket = None
    edited_bracket = None
    param_names = None
    use_latent_regressor = False

    # Check for latent regressor first (preferred - bypasses decoder)
    latent_regressor_path = Path(args.latent_regressor_checkpoint) if args.latent_regressor_checkpoint else None
    regressor_path = Path(args.regressor_checkpoint)

    if latent_regressor_path and latent_regressor_path.exists():
        # Use latent regressor (z → params directly)
        use_latent_regressor = True
        latent_regressor, latent_reg_config = load_latent_regressor(
            args.latent_regressor_checkpoint, device
        )
        param_names = ["leg1_length", "leg2_length", "width", "thickness"]

        # Denormalize function for variable topology params
        from graph_cad.data.dataset import VariableLBracketRanges
        ranges = VariableLBracketRanges()

        def denormalize_latent_params(params_norm):
            mins = torch.tensor([
                ranges.leg1_length[0], ranges.leg2_length[0],
                ranges.width[0], ranges.thickness[0],
            ], device=params_norm.device)
            maxs = torch.tensor([
                ranges.leg1_length[1], ranges.leg2_length[1],
                ranges.width[1], ranges.thickness[1],
            ], device=params_norm.device)
            return params_norm * (maxs - mins) + mins

        # Predict parameters directly from latent vectors
        with torch.no_grad():
            orig_params_norm = latent_regressor(z_src)
            orig_params_pred = denormalize_latent_params(orig_params_norm).cpu().numpy().flatten()

            edit_params_norm = latent_regressor(z_edited)
            edit_params_pred = denormalize_latent_params(edit_params_norm).cpu().numpy().flatten()

        print("  Using LatentRegressor (z → params, bypasses decoder)")

    elif regressor_path.exists():
        # Use feature regressor (decode → features → params)
        from graph_cad.models.parameter_regressor import PARAMETER_NAMES, denormalize_parameters
        param_names = PARAMETER_NAMES

        regressor = load_feature_regressor(args.regressor_checkpoint, device)

        # Predict parameters from decoded features
        with torch.no_grad():
            # Original
            orig_node_t = torch.tensor(orig_node_recon, dtype=torch.float32, device=device).unsqueeze(0)
            orig_edge_t = torch.tensor(orig_edge_recon, dtype=torch.float32, device=device).unsqueeze(0)
            orig_params_norm = regressor(orig_node_t, orig_edge_t)
            orig_params_pred = denormalize_parameters(orig_params_norm).cpu().numpy().flatten()

            # Edited
            edit_node_t = torch.tensor(edit_node_recon, dtype=torch.float32, device=device).unsqueeze(0)
            edit_edge_t = torch.tensor(edit_edge_recon, dtype=torch.float32, device=device).unsqueeze(0)
            edit_params_norm = regressor(edit_node_t, edit_edge_t)
            edit_params_pred = denormalize_parameters(edit_params_norm).cpu().numpy().flatten()

        print("  Using FeatureRegressor (decode → features → params)")

    else:
        print(f"  No regressor checkpoint found.")
        print("  Options:")
        print(f"    --latent-regressor-checkpoint outputs/latent_regressor/best_model.pt")
        print(f"    --regressor-checkpoint outputs/feature_regressor/best_model.pt")

    # Display results if we have predictions
    if orig_params_pred is not None:
        print("\n  Predicted Parameters (mm):")
        print("  " + "-" * 55)
        print(f"  {'Parameter':<20} {'Original':>12} {'Edited':>12} {'Change':>10}")
        print("  " + "-" * 55)
        for i, name in enumerate(param_names):
            orig_val = orig_params_pred[i]
            edit_val = edit_params_pred[i]
            change = edit_val - orig_val
            print(f"  {name:<20} {orig_val:>12.2f} {edit_val:>12.2f} {change:>+10.2f}")

        # Compare with ground truth if we have a bracket
        if bracket is not None:
            print("\n  Ground Truth Comparison (Original):")
            print("  " + "-" * 40)
            gt_params = bracket.to_dict()
            for i, name in enumerate(param_names):
                if name in gt_params:
                    gt_val = gt_params[name]
                    pred_val = orig_params_pred[i]
                    error = pred_val - gt_val
                    print(f"  {name:<20} GT={gt_val:>8.2f}  Pred={pred_val:>8.2f}  Err={error:>+6.2f}")

        # Create L-bracket geometry (only for feature regressor with 8 params)
        if not use_latent_regressor and len(param_names) == 8:
            predicted_original_bracket = None
            try:
                predicted_original_bracket = LBracket(
                    leg1_length=float(orig_params_pred[0]),
                    leg2_length=float(orig_params_pred[1]),
                    width=float(orig_params_pred[2]),
                    thickness=float(orig_params_pred[3]),
                    hole1_distance=float(orig_params_pred[4]),
                    hole1_diameter=float(orig_params_pred[5]),
                    hole2_distance=float(orig_params_pred[6]),
                    hole2_diameter=float(orig_params_pred[7]),
                )
                print("\n  Successfully created predicted original L-bracket geometry!")
            except ValueError as e:
                print(f"\n  Warning: Could not create predicted original L-bracket: {e}")

            try:
                edited_bracket = LBracket(
                    leg1_length=float(edit_params_pred[0]),
                    leg2_length=float(edit_params_pred[1]),
                    width=float(edit_params_pred[2]),
                    thickness=float(edit_params_pred[3]),
                    hole1_distance=float(edit_params_pred[4]),
                    hole1_diameter=float(edit_params_pred[5]),
                    hole2_distance=float(edit_params_pred[6]),
                    hole2_diameter=float(edit_params_pred[7]),
                )
                print("  Successfully created edited L-bracket geometry!")
            except ValueError as e:
                print(f"\n  Warning: Could not create valid L-bracket from predicted params: {e}")
                edited_bracket = None
        elif use_latent_regressor:
            # Latent regressor only predicts 4 core params - can't generate STEP without hole params
            print("\n  Note: LatentRegressor predicts 4 core params only (no holes).")
            print("  STEP file generation requires hole parameters - skipping geometry creation.")

    # =========================================================================
    # Step 7: Compare and display results
    # =========================================================================
    if args.verbose:
        print_latent_comparison(z_src, z_edited, delta_z)
        print_graph_comparison(
            graph,
            (orig_node_recon, orig_edge_recon),
            (edit_node_recon, edit_edge_recon),
        )

    # =========================================================================
    # Step 8: Save results
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 8: Saving Results")
    print("=" * 60)

    # Save latent vectors
    latent_data = {
        "instruction": args.instruction,
        "z_src": z_src.cpu().numpy().tolist(),
        "z_edited": z_edited.cpu().numpy().tolist(),
        "delta_z": delta_z.cpu().numpy().tolist(),
        "delta_norm": delta_z.norm().item(),
    }
    if orig_params_pred is not None and param_names is not None:
        latent_data["original_params"] = {
            name: float(orig_params_pred[i]) for i, name in enumerate(param_names)
        }
        latent_data["edited_params"] = {
            name: float(edit_params_pred[i]) for i, name in enumerate(param_names)
        }
        latent_data["regressor_type"] = "latent" if use_latent_regressor else "feature"
    latent_path = output_dir / "latent_vectors.json"
    with open(latent_path, "w") as f:
        json.dump(latent_data, f, indent=2)
    print(f"  Saved latent vectors to: {latent_path}")

    if args.save_graphs:
        graph_data = {
            "original": {
                "node_features": orig_node_recon.tolist(),
                "edge_features": orig_edge_recon.tolist(),
            },
            "edited": {
                "node_features": edit_node_recon.tolist(),
                "edge_features": edit_edge_recon.tolist(),
            },
        }
        graph_path = output_dir / "graph_features.json"
        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)
        print(f"  Saved graph features to: {graph_path}")

    # Save predicted original STEP file (for apples-to-apples comparison with edited)
    if predicted_original_bracket is not None:
        pred_orig_step_path = output_dir / "predicted_original.step"
        predicted_original_bracket.to_step(str(pred_orig_step_path))
        print(f"  Saved predicted original STEP to: {pred_orig_step_path}")

    # Save edited STEP file
    if edited_bracket is not None:
        edited_step_path = output_dir / "edited.step"
        if args.output:
            edited_step_path = Path(args.output)
        edited_bracket.to_step(str(edited_step_path))
        print(f"  Saved edited STEP to: {edited_step_path}")
    elif args.output:
        print(f"  Could not save STEP file (parameter prediction failed)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Input: {args.input or ('random L-bracket' if args.random_bracket else args.params)}")
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  Delta magnitude: {delta_z.norm().item():.4f}")

    # Quick diff summary
    node_mse = np.mean((edit_node_recon - orig_node_recon) ** 2)
    edge_mse = np.mean((edit_edge_recon - orig_edge_recon) ** 2)
    print(f"  Graph change: node_MSE={node_mse:.6f}, edge_MSE={edge_mse:.6f}")

    if orig_params_pred is not None and edit_params_pred is not None and param_names is not None:
        param_changes = edit_params_pred - orig_params_pred

        # Show all parameter changes in logical order
        print("\n  Parameter Changes:")
        print("  " + "-" * 50)
        print(f"  {'Parameter':<20} {'Original':>10} {'Edited':>10} {'Δ':>10}")
        print("  " + "-" * 50)
        for i, name in enumerate(param_names):
            print(f"  {name:<20} {orig_params_pred[i]:>10.2f} {edit_params_pred[i]:>10.2f} {param_changes[i]:>+10.2f}")


if __name__ == "__main__":
    main()
