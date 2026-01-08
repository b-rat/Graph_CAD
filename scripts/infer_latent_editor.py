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
    """Load trained FeatureRegressor model (fixed or variable topology)."""
    import torch.nn as nn
    from dataclasses import dataclass

    print(f"Loading FeatureRegressor from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]

    # Detect variable vs fixed topology based on config keys
    if "max_nodes" in config_dict:
        # Variable topology feature regressor
        @dataclass
        class VariableFeatureRegressorConfig:
            max_nodes: int = 20
            max_edges: int = 50
            node_features: int = 13  # area, dir_xyz, centroid_xyz, curv, bbox_diagonal, bbox_center_xyz
            edge_features: int = 2
            hidden_dims: tuple[int, ...] = (512, 256, 128, 64)
            dropout: float = 0.1
            use_batch_norm: bool = True
            num_params: int = 4

            @property
            def input_dim(self) -> int:
                return self.max_nodes * self.node_features + self.max_edges * self.edge_features

        class VariableFeatureRegressor(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                layers = []
                in_dim = config.input_dim
                for hidden_dim in config.hidden_dims:
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    if config.use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(config.dropout))
                    in_dim = hidden_dim
                self.backbone = nn.Sequential(*layers)
                self.output_head = nn.Linear(in_dim, config.num_params)

            def forward(self, node_features, edge_features):
                batch_size = node_features.shape[0]
                node_flat = node_features.view(batch_size, -1)
                edge_flat = edge_features.view(batch_size, -1)
                x = torch.cat([node_flat, edge_flat], dim=-1)
                h = self.backbone(x)
                return self.output_head(h)

        config = VariableFeatureRegressorConfig(**config_dict)
        regressor = VariableFeatureRegressor(config)
        regressor.load_state_dict(checkpoint["model_state_dict"])
        regressor.to(device)
        regressor.eval()
        print(f"  Type: Variable topology (4 params)")
        print(f"  Input: {config.max_nodes}×{config.node_features} + {config.max_edges}×{config.edge_features} = {config.input_dim}D")
        return regressor, "variable"
    else:
        # Fixed topology feature regressor
        from graph_cad.models.feature_regressor import load_feature_regressor as _load
        regressor, _ = _load(checkpoint_path, device=device)
        regressor.eval()
        print(f"  Type: Fixed topology (8 params)")
        print(f"  Config: {regressor.config.hidden_dims}")
        return regressor, "fixed"


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


def load_full_latent_regressor(checkpoint_path: str, device: str):
    """Load trained FullLatentRegressor model (predicts ALL params from z)."""
    from dataclasses import dataclass
    import torch.nn as nn

    @dataclass
    class FullLatentRegressorConfig:
        latent_dim: int = 32
        hidden_dim: int = 256
        num_layers: int = 3
        dropout: float = 0.1
        max_holes_per_leg: int = 2

    class FullLatentRegressor(nn.Module):
        def __init__(self, config: FullLatentRegressorConfig):
            super().__init__()
            self.config = config

            # Shared backbone
            layers = []
            in_dim = config.latent_dim
            for _ in range(config.num_layers):
                layers.extend([
                    nn.Linear(in_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ])
                in_dim = config.hidden_dim
            self.backbone = nn.Sequential(*layers)

            # Core params head
            self.core_head = nn.Linear(config.hidden_dim, 4)

            # Fillet head
            self.fillet_head = nn.Linear(config.hidden_dim, 1)
            self.fillet_exists_head = nn.Linear(config.hidden_dim, 1)

            # Hole heads
            self.hole1_heads = nn.ModuleList([
                nn.Linear(config.hidden_dim, 2) for _ in range(config.max_holes_per_leg)
            ])
            self.hole1_exists_head = nn.Linear(config.hidden_dim, config.max_holes_per_leg)

            self.hole2_heads = nn.ModuleList([
                nn.Linear(config.hidden_dim, 2) for _ in range(config.max_holes_per_leg)
            ])
            self.hole2_exists_head = nn.Linear(config.hidden_dim, config.max_holes_per_leg)

        def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
            h = self.backbone(z)
            core_params = self.core_head(h)
            fillet_radius = self.fillet_head(h)
            fillet_exists = torch.sigmoid(self.fillet_exists_head(h))
            hole1_params = torch.stack([head(h) for head in self.hole1_heads], dim=1)
            hole1_exists = torch.sigmoid(self.hole1_exists_head(h))
            hole2_params = torch.stack([head(h) for head in self.hole2_heads], dim=1)
            hole2_exists = torch.sigmoid(self.hole2_exists_head(h))

            return {
                "core_params": core_params,
                "fillet_radius": fillet_radius,
                "fillet_exists": fillet_exists,
                "hole1_params": hole1_params,
                "hole1_exists": hole1_exists,
                "hole2_params": hole2_params,
                "hole2_exists": hole2_exists,
            }

    print(f"Loading FullLatentRegressor from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = FullLatentRegressorConfig(**checkpoint["config"])
    regressor = FullLatentRegressor(config)
    regressor.load_state_dict(checkpoint["model_state_dict"])
    regressor.to(device)
    regressor.eval()
    print(f"  Config: {config.latent_dim}D → {config.num_layers}×{config.hidden_dim} → multi-head")
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


def decode_latent(vae, z: torch.Tensor) -> dict:
    """Decode latent vector to graph features.

    Returns dict with:
        - node_features: (num_nodes, node_dim) numpy array
        - edge_features: (num_edges, edge_dim) numpy array
        - face_types: (num_nodes,) numpy array (for VariableGraphVAE)
        - node_mask: (num_nodes,) numpy array (for VariableGraphVAE)
    """
    with torch.no_grad():
        output = vae.decode(z)

    # Handle both dict output (VariableGraphVAE) and tuple output (GraphVAE)
    if isinstance(output, dict):
        # VariableGraphVAE returns dict with logits
        node_features = output["node_features"].cpu().numpy()[0]
        edge_features = output["edge_features"].cpu().numpy()[0]

        # Convert logits to predictions
        node_mask = (torch.sigmoid(output["node_mask_logits"]) > 0.5).cpu().numpy()[0]
        face_types = output["face_type_logits"].argmax(dim=-1).cpu().numpy()[0]

        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "face_types": face_types,
            "node_mask": node_mask,
        }
    else:
        # GraphVAE returns tuple (node_recon, edge_recon)
        node_recon, edge_recon = output
        return {
            "node_features": node_recon.cpu().numpy()[0],
            "edge_features": edge_recon.cpu().numpy()[0],
            "face_types": None,
            "node_mask": None,
        }


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
        help="Path to trained LatentRegressor checkpoint (predicts 4 core params from z)",
    )
    parser.add_argument(
        "--full-latent-regressor-checkpoint",
        type=str,
        default=None,
        help="Path to trained FullLatentRegressor checkpoint (predicts ALL params from z, can generate STEP)",
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
    orig_decoded = decode_latent(vae, z_src)
    orig_node_recon = orig_decoded["node_features"]
    orig_edge_recon = orig_decoded["edge_features"]
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

    edit_decoded = decode_latent(vae, z_edited)
    edit_node_recon = edit_decoded["node_features"]
    edit_edge_recon = edit_decoded["edge_features"]
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
    use_geometric_solver = False

    # Try geometric solver first (preferred for VariableGraphVAE with 13D features)
    if edit_decoded.get("face_types") is not None and edit_decoded.get("node_mask") is not None:
        from graph_cad.utils.geometric_solver import solve_params_from_features
        from graph_cad.data.l_bracket import VariableLBracket

        use_geometric_solver = True
        print("  Using Geometric Solver (deterministic parameter extraction)")

        try:
            # Solve original params
            orig_solved = solve_params_from_features(
                node_features=orig_decoded["node_features"],
                face_types=orig_decoded["face_types"],
                edge_index=np.zeros((2, 0), dtype=np.int64),  # Not needed for solving
                edge_features=np.zeros((0, 2), dtype=np.float32),
                node_mask=orig_decoded["node_mask"].astype(np.float32),
            )
            param_names = ["leg1_length", "leg2_length", "width", "thickness"]
            orig_params_pred = np.array([
                orig_solved.leg1_length,
                orig_solved.leg2_length,
                orig_solved.width,
                orig_solved.thickness,
            ])

            # Create original bracket from solved params
            try:
                predicted_original_bracket = VariableLBracket(**orig_solved.to_dict())
                n_holes = predicted_original_bracket.num_holes_leg1 + predicted_original_bracket.num_holes_leg2
                has_fillet = "with fillet" if predicted_original_bracket.has_fillet else "no fillet"
                print(f"  Original: {n_holes} holes, {has_fillet}")
            except ValueError as e:
                print(f"  Warning: Could not create original bracket: {e}")
                predicted_original_bracket = None

            # Solve edited params
            edit_solved = solve_params_from_features(
                node_features=edit_decoded["node_features"],
                face_types=edit_decoded["face_types"],
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_features=np.zeros((0, 2), dtype=np.float32),
                node_mask=edit_decoded["node_mask"].astype(np.float32),
            )
            edit_params_pred = np.array([
                edit_solved.leg1_length,
                edit_solved.leg2_length,
                edit_solved.width,
                edit_solved.thickness,
            ])

            # Create edited bracket from solved params
            try:
                edited_bracket = VariableLBracket(**edit_solved.to_dict())
                n_holes = edited_bracket.num_holes_leg1 + edited_bracket.num_holes_leg2
                has_fillet = "with fillet" if edited_bracket.has_fillet else "no fillet"
                print(f"  Edited: {n_holes} holes, {has_fillet}")
            except ValueError as e:
                print(f"  Warning: Could not create edited bracket: {e}")
                edited_bracket = None

        except Exception as e:
            print(f"  Geometric solver failed: {e}")
            print("  Falling back to regressor...")
            use_geometric_solver = False

    # Check for regressors in order of priority (only used if geometric solver fails)
    full_latent_regressor_path = Path(args.full_latent_regressor_checkpoint) if args.full_latent_regressor_checkpoint else None
    latent_regressor_path = Path(args.latent_regressor_checkpoint) if args.latent_regressor_checkpoint else None
    regressor_path = Path(args.regressor_checkpoint)

    # Import ranges for denormalization
    from graph_cad.data.dataset import VariableLBracketRanges
    ranges = VariableLBracketRanges()

    # For full regressor: store full predictions for STEP generation
    full_regressor_outputs_orig = None
    full_regressor_outputs_edit = None
    use_full_regressor = False

    # Only use regressors if geometric solver wasn't used or failed
    if not use_geometric_solver and full_latent_regressor_path and full_latent_regressor_path.exists():
        # Use full latent regressor (z → ALL params, can generate STEP)
        use_full_regressor = True
        use_latent_regressor = True
        full_regressor, full_reg_config = load_full_latent_regressor(
            args.full_latent_regressor_checkpoint, device
        )
        param_names = ["leg1_length", "leg2_length", "width", "thickness"]

        # Denormalize core params
        def denormalize_core(params_norm):
            mins = torch.tensor([
                ranges.leg1_length[0], ranges.leg2_length[0],
                ranges.width[0], ranges.thickness[0],
            ], device=params_norm.device)
            maxs = torch.tensor([
                ranges.leg1_length[1], ranges.leg2_length[1],
                ranges.width[1], ranges.thickness[1],
            ], device=params_norm.device)
            return params_norm * (maxs - mins) + mins

        # Predict ALL parameters from latent vectors
        with torch.no_grad():
            full_regressor_outputs_orig = full_regressor(z_src)
            orig_core = denormalize_core(full_regressor_outputs_orig["core_params"])
            orig_params_pred = orig_core.cpu().numpy().flatten()

            full_regressor_outputs_edit = full_regressor(z_edited)
            edit_core = denormalize_core(full_regressor_outputs_edit["core_params"])
            edit_params_pred = edit_core.cpu().numpy().flatten()

        print("  Using FullLatentRegressor (z → ALL params, can generate STEP)")

    elif not use_geometric_solver and latent_regressor_path and latent_regressor_path.exists():
        # Use simple latent regressor (z → 4 core params only)
        use_latent_regressor = True
        latent_regressor, latent_reg_config = load_latent_regressor(
            args.latent_regressor_checkpoint, device
        )
        param_names = ["leg1_length", "leg2_length", "width", "thickness"]

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

        print("  Using LatentRegressor (z → 4 core params, cannot generate STEP)")

    elif not use_geometric_solver and regressor_path.exists():
        # Use feature regressor (decode → features → params)
        regressor, regressor_type = load_feature_regressor(args.regressor_checkpoint, device)

        if regressor_type == "variable":
            # Variable topology: 4 params
            param_names = ["leg1_length", "leg2_length", "width", "thickness"]
            use_latent_regressor = True  # Same behavior as latent regressor (4 params, no STEP)

            # Denormalize function for variable topology params
            from graph_cad.data.dataset import VariableLBracketRanges
            ranges = VariableLBracketRanges()

            def denormalize_fn(params_norm):
                mins = torch.tensor([
                    ranges.leg1_length[0], ranges.leg2_length[0],
                    ranges.width[0], ranges.thickness[0],
                ], device=params_norm.device)
                maxs = torch.tensor([
                    ranges.leg1_length[1], ranges.leg2_length[1],
                    ranges.width[1], ranges.thickness[1],
                ], device=params_norm.device)
                return params_norm * (maxs - mins) + mins
        else:
            # Fixed topology: 8 params
            from graph_cad.models.parameter_regressor import PARAMETER_NAMES, denormalize_parameters
            param_names = PARAMETER_NAMES
            denormalize_fn = denormalize_parameters

        # Predict parameters from decoded features
        with torch.no_grad():
            # Original
            orig_node_t = torch.tensor(orig_node_recon, dtype=torch.float32, device=device).unsqueeze(0)
            orig_edge_t = torch.tensor(orig_edge_recon, dtype=torch.float32, device=device).unsqueeze(0)
            orig_params_norm = regressor(orig_node_t, orig_edge_t)
            orig_params_pred = denormalize_fn(orig_params_norm).cpu().numpy().flatten()

            # Edited
            edit_node_t = torch.tensor(edit_node_recon, dtype=torch.float32, device=device).unsqueeze(0)
            edit_edge_t = torch.tensor(edit_edge_recon, dtype=torch.float32, device=device).unsqueeze(0)
            edit_params_norm = regressor(edit_node_t, edit_edge_t)
            edit_params_pred = denormalize_fn(edit_params_norm).cpu().numpy().flatten()

        print(f"  Using FeatureRegressor ({regressor_type} topology, decode → features → {len(param_names)} params)")

    elif not use_geometric_solver:
        print(f"  No parameter extraction method available.")
        print("  For VariableGraphVAE with 13D features, use --vae-checkpoint outputs/vae_variable_13d/best_model.pt")
        print("  Fallback options:")
        print(f"    --full-latent-regressor-checkpoint outputs/full_latent_regressor/best_model.pt  (generates STEP)")
        print(f"    --latent-regressor-checkpoint outputs/latent_regressor/best_model.pt  (4 core params only)")
        print(f"    --regressor-checkpoint outputs/feature_regressor/best_model.pt  (via decoder)")

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
        elif use_full_regressor:
            # Full latent regressor - can generate STEP with VariableLBracket
            from graph_cad.data.l_bracket import VariableLBracket

            def outputs_to_variable_bracket(outputs, core_params_mm):
                """Convert full regressor outputs to VariableLBracket."""
                # Core params (already denormalized)
                leg1 = float(core_params_mm[0])
                leg2 = float(core_params_mm[1])
                width = float(core_params_mm[2])
                thickness = float(core_params_mm[3])

                # Fillet
                fillet_exists = outputs["fillet_exists"].cpu().item() > 0.5
                if fillet_exists:
                    fillet_radius = float(outputs["fillet_radius"].cpu().item() * ranges.fillet_radius[1])
                else:
                    fillet_radius = 0.0

                # Holes on leg1
                hole1_diams = []
                hole1_dists = []
                hole1_exists = outputs["hole1_exists"].cpu().numpy().flatten()
                hole1_params = outputs["hole1_params"].cpu().numpy()[0]  # (2, 2)
                for i in range(2):
                    if hole1_exists[i] > 0.5:
                        diam = hole1_params[i, 0] * (ranges.hole_diameter[1] - ranges.hole_diameter[0]) + ranges.hole_diameter[0]
                        dist = hole1_params[i, 1] * leg1  # Relative to leg length
                        hole1_diams.append(float(diam))
                        hole1_dists.append(float(dist))

                # Holes on leg2
                hole2_diams = []
                hole2_dists = []
                hole2_exists = outputs["hole2_exists"].cpu().numpy().flatten()
                hole2_params = outputs["hole2_params"].cpu().numpy()[0]  # (2, 2)
                for i in range(2):
                    if hole2_exists[i] > 0.5:
                        diam = hole2_params[i, 0] * (ranges.hole_diameter[1] - ranges.hole_diameter[0]) + ranges.hole_diameter[0]
                        dist = hole2_params[i, 1] * leg2
                        hole2_diams.append(float(diam))
                        hole2_dists.append(float(dist))

                return VariableLBracket(
                    leg1_length=leg1,
                    leg2_length=leg2,
                    width=width,
                    thickness=thickness,
                    fillet_radius=fillet_radius,
                    hole1_diameters=tuple(hole1_diams),
                    hole1_distances=tuple(hole1_dists),
                    hole2_diameters=tuple(hole2_diams),
                    hole2_distances=tuple(hole2_dists),
                )

            # Create predicted original bracket
            try:
                predicted_original_bracket = outputs_to_variable_bracket(
                    full_regressor_outputs_orig, orig_params_pred
                )
                n_holes = predicted_original_bracket.num_holes_leg1 + predicted_original_bracket.num_holes_leg2
                has_fillet = "with fillet" if predicted_original_bracket.has_fillet else "no fillet"
                print(f"\n  Created predicted original VariableLBracket ({n_holes} holes, {has_fillet})")
            except ValueError as e:
                print(f"\n  Warning: Could not create predicted original bracket: {e}")
                predicted_original_bracket = None

            # Create edited bracket
            try:
                edited_bracket = outputs_to_variable_bracket(
                    full_regressor_outputs_edit, edit_params_pred
                )
                n_holes = edited_bracket.num_holes_leg1 + edited_bracket.num_holes_leg2
                has_fillet = "with fillet" if edited_bracket.has_fillet else "no fillet"
                print(f"  Created edited VariableLBracket ({n_holes} holes, {has_fillet})")
            except ValueError as e:
                print(f"\n  Warning: Could not create edited bracket: {e}")
                edited_bracket = None

        elif use_latent_regressor:
            # Simple latent regressor or variable feature regressor - only 4 params
            print("\n  Note: Regressor predicts 4 core params only (leg1, leg2, width, thickness).")
            print("  Use --full-latent-regressor-checkpoint for STEP generation.")

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
        if use_full_regressor:
            latent_data["regressor_type"] = "full_latent"
        elif use_latent_regressor:
            latent_data["regressor_type"] = "latent"
        else:
            latent_data["regressor_type"] = "feature"
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
