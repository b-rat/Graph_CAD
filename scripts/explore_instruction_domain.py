#!/usr/bin/env python3
"""
Systematic exploration of the instruction domain for the Latent Editor.

This script loads models once and runs inference across a grid of:
  - Parameters: leg1_length, leg2_length (width/thickness not encoded in latent)
  - Directions: increase, decrease
  - Magnitudes: 10, 20, 30, 50mm
  - Starting brackets: sampled across the parameter space

IMPORTANT: Uses explicit +/- signs in instructions (required by current model).

Outputs a comprehensive JSON file with all results for analysis.

Usage:
    python scripts/explore_instruction_domain.py \
        --num-brackets 10 \
        --output outputs/exploration/results.json

    # Quick test run
    python scripts/explore_instruction_domain.py \
        --num-brackets 3 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Instruction Domain Definition
# =============================================================================

# CRITICAL: Instructions must use +/- signs for the model to work correctly
# The model learned to rely on +/- tokens rather than "longer/shorter" semantics

INSTRUCTION_DOMAIN = {
    "leg1_length": {
        "display_name": "leg1",
        "magnitudes": [10, 20, 30, 50],
        "increase_templates": [
            "make {name} +{amount}mm longer",
            "change {name} length by +{amount}mm",
            "{name} +{amount}mm",
        ],
        "decrease_templates": [
            "make {name} -{amount}mm shorter",
            "change {name} length by -{amount}mm",
            "{name} -{amount}mm",
        ],
    },
    "leg2_length": {
        "display_name": "leg2",
        "magnitudes": [10, 20, 30, 50],
        "increase_templates": [
            "make {name} +{amount}mm longer",
            "change {name} length by +{amount}mm",
            "{name} +{amount}mm",
        ],
        "decrease_templates": [
            "make {name} -{amount}mm shorter",
            "change {name} length by -{amount}mm",
            "{name} -{amount}mm",
        ],
    },
}

# Quick mode uses subset
QUICK_DOMAIN = {
    "leg1_length": {
        "display_name": "leg1",
        "magnitudes": [20],
        "increase_templates": ["make {name} +{amount}mm longer"],
        "decrease_templates": ["make {name} -{amount}mm shorter"],
    },
    "leg2_length": {
        "display_name": "leg2",
        "magnitudes": [20],
        "increase_templates": ["make {name} +{amount}mm longer"],
        "decrease_templates": ["make {name} -{amount}mm shorter"],
    },
}

# Parameter names for latent regressor (4 core params)
PARAM_NAMES = ["leg1_length", "leg2_length", "width", "thickness"]


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class TrialResult:
    """Result from a single inference trial."""
    # Input
    bracket_id: int
    bracket_seed: int
    bracket_params: dict[str, float]

    # Instruction
    parameter: str
    direction: str
    magnitude: float
    instruction: str

    # Output
    delta_norm: float
    original_params_pred: dict[str, float]
    edited_params_pred: dict[str, float]
    param_changes: dict[str, float]

    # Metrics
    target_param_change: float  # Actual change in target parameter
    target_param_pct: float  # Percentage of requested magnitude achieved
    correct_direction: bool  # Did it move in the right direction?

    # Timing
    inference_time_ms: float


# =============================================================================
# Model Loading (done once)
# =============================================================================

class ModelBundle:
    """Container for all loaded models."""

    def __init__(
        self,
        vae_checkpoint: str,
        editor_checkpoint: str,
        latent_regressor_checkpoint: str,
        device: str,
    ):
        self.device = device
        self.vae = None
        self.vae_type = None
        self.editor = None
        self.latent_regressor = None

        self._load_models(vae_checkpoint, editor_checkpoint, latent_regressor_checkpoint)

    def _load_models(
        self,
        vae_checkpoint: str,
        editor_checkpoint: str,
        latent_regressor_checkpoint: str,
    ):
        """Load all models once."""
        from graph_cad.models.graph_vae import VariableGraphVAEConfig, VariableGraphVAEEncoder
        from graph_cad.models.transformer_decoder import TransformerDecoderConfig, TransformerGraphVAE
        from graph_cad.models.latent_editor import LatentEditor, load_llm_with_lora
        from graph_cad.training.edit_trainer import load_editor_checkpoint

        print("=" * 60)
        print("LOADING MODELS (this happens once)")
        print("=" * 60)

        # Load VAE (Transformer VAE)
        print(f"\n[1/3] Loading VAE from {vae_checkpoint}...")
        checkpoint = torch.load(vae_checkpoint, map_location=self.device, weights_only=False)

        if "decoder_config" in checkpoint:
            # TransformerGraphVAE
            encoder_config = VariableGraphVAEConfig(**checkpoint["encoder_config"])
            encoder = VariableGraphVAEEncoder(encoder_config)
            decoder_config = TransformerDecoderConfig(**checkpoint["decoder_config"])
            use_param_head = checkpoint.get("use_param_head", False)
            num_params = checkpoint.get("num_params", 4)

            self.vae = TransformerGraphVAE(
                encoder, decoder_config,
                use_param_head=use_param_head,
                num_params=num_params,
            )
            self.vae.load_state_dict(checkpoint["model_state_dict"])
            self.vae.to(self.device)
            self.vae.eval()
            self.vae_type = "transformer"
            print(f"  VAE type: TransformerGraphVAE")
            print(f"  Latent dim: {decoder_config.latent_dim}")
        else:
            raise ValueError("Only Transformer VAE is supported. Use outputs/vae_transformer_aux2_w100/best_model.pt")

        # Load Latent Regressor (z -> params directly)
        print(f"\n[2/3] Loading Latent Regressor from {latent_regressor_checkpoint}...")
        self._load_latent_regressor(latent_regressor_checkpoint)
        print(f"  Config: {self.latent_regressor.config.latent_dim}D → {self.latent_regressor.config.hidden_dims} → 4D")

        # Load Latent Editor (most expensive)
        print(f"\n[3/3] Loading Latent Editor from {editor_checkpoint}...")
        print("  This may take a few minutes...")

        checkpoint = torch.load(editor_checkpoint, map_location="cpu", weights_only=False)
        editor_config = checkpoint.get("config")

        if self.device == "cuda":
            editor_config.use_4bit = True
            editor_config.use_8bit = False
            device_map = "auto"
        else:
            editor_config.use_4bit = False
            editor_config.use_8bit = False
            device_map = {"": "cpu"}

        self.editor = LatentEditor(editor_config)
        llm, tokenizer = load_llm_with_lora(editor_config, device_map=device_map)
        self.editor.set_llm(llm, tokenizer)
        load_editor_checkpoint(self.editor, editor_checkpoint, device="cpu")

        if self.device == "cuda":
            self.editor.latent_projector.to(self.device)
            self.editor.output_projector.to(self.device)

        self.editor.eval()
        print(f"  Loaded from epoch {checkpoint['epoch']}")

        print("\n" + "=" * 60)
        print("ALL MODELS LOADED")
        print("=" * 60 + "\n")

    def _load_latent_regressor(self, checkpoint_path: str):
        """Load the latent regressor model."""
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

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = LatentRegressorConfig(**checkpoint["config"])
        self.latent_regressor = LatentRegressor(config)
        self.latent_regressor.load_state_dict(checkpoint["model_state_dict"])
        self.latent_regressor.to(self.device)
        self.latent_regressor.eval()


# =============================================================================
# Inference Functions
# =============================================================================

def run_single_inference(
    models: ModelBundle,
    bracket_params: dict[str, float],
    instruction: str,
) -> dict:
    """Run inference for a single bracket + instruction pair."""
    from graph_cad.data.l_bracket import VariableLBracket
    from graph_cad.data.graph_extraction import extract_graph_from_solid_variable
    from graph_cad.data.dataset import VariableLBracketRanges

    device = models.device
    MAX_NODES = 20
    MAX_EDGES = 50

    # Create bracket and extract graph
    bracket = VariableLBracket(**bracket_params)
    solid = bracket.to_solid()
    graph = extract_graph_from_solid_variable(solid)

    num_nodes = graph.num_faces
    num_edges = graph.num_edges

    # Pad to fixed size for VAE
    node_features = np.zeros((MAX_NODES, 13), dtype=np.float32)
    node_features[:num_nodes] = graph.node_features

    face_types = np.zeros(MAX_NODES, dtype=np.int64)
    face_types[:num_nodes] = graph.face_types

    node_mask = np.zeros(MAX_NODES, dtype=np.float32)
    node_mask[:num_nodes] = 1.0

    edge_features = np.zeros((MAX_EDGES, 2), dtype=np.float32)
    edge_features[:num_edges] = graph.edge_features

    edge_index = np.zeros((2, MAX_EDGES), dtype=np.int64)
    edge_index[:, :num_edges] = graph.edge_index

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    ft = torch.tensor(face_types, dtype=torch.long, device=device)
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_features, dtype=torch.float32, device=device)
    nm = torch.tensor(node_mask, dtype=torch.float32, device=device)
    batch = torch.zeros(MAX_NODES, dtype=torch.long, device=device)

    # Encode
    with torch.no_grad():
        mu, _ = models.vae.encode(x, ft, ei, ea, batch, nm)
        z_src = mu

    # Apply instruction
    start_time = time.perf_counter()
    with torch.no_grad():
        z_src_editor = z_src.cpu() if device != "cuda" else z_src
        outputs = models.editor(z_src_editor, [instruction])
        delta_z = outputs["delta_z"].to(device)
        z_edited = outputs["z_edited"].to(device)
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # Predict parameters from latent vectors (not decoded features)
    ranges = VariableLBracketRanges()

    def denormalize(params_norm):
        mins = torch.tensor([
            ranges.leg1_length[0], ranges.leg2_length[0],
            ranges.width[0], ranges.thickness[0],
        ], device=params_norm.device)
        maxs = torch.tensor([
            ranges.leg1_length[1], ranges.leg2_length[1],
            ranges.width[1], ranges.thickness[1],
        ], device=params_norm.device)
        return params_norm * (maxs - mins) + mins

    with torch.no_grad():
        orig_params_norm = models.latent_regressor(z_src)
        edit_params_norm = models.latent_regressor(z_edited)
        orig_params = denormalize(orig_params_norm).cpu().numpy().flatten()
        edit_params = denormalize(edit_params_norm).cpu().numpy().flatten()

    return {
        "delta_norm": float(delta_z.norm().item()),
        "original_params": {name: float(orig_params[i]) for i, name in enumerate(PARAM_NAMES)},
        "edited_params": {name: float(edit_params[i]) for i, name in enumerate(PARAM_NAMES)},
        "inference_time_ms": inference_time_ms,
    }


def generate_instruction(
    domain_config: dict,
    parameter: str,
    direction: str,
    magnitude: float,
    template_idx: int = 0,
) -> str:
    """Generate an instruction from the domain configuration."""
    config = domain_config[parameter]
    display_name = config["display_name"]

    if direction == "increase":
        templates = config["increase_templates"]
    else:
        templates = config["decrease_templates"]

    template = templates[template_idx % len(templates)]
    return template.format(name=display_name, amount=int(magnitude))


def sample_brackets(
    num_brackets: int,
    seed: int,
    stratified: bool = True,
) -> list[dict]:
    """Sample brackets across the parameter space."""
    from graph_cad.data.l_bracket import VariableLBracket, VariableLBracketRanges

    rng = np.random.default_rng(seed)
    brackets = []

    if stratified and num_brackets >= 3:
        # Sample from different regions: small, medium, large
        regions = ["small", "medium", "large"]
        per_region = num_brackets // 3
        remainder = num_brackets % 3

        for region_idx, region in enumerate(regions):
            n = per_region + (1 if region_idx < remainder else 0)
            for i in range(n):
                bracket = _sample_bracket_in_region(rng, region)
                brackets.append({
                    "params": _bracket_to_core_params(bracket),
                    "seed": int(rng.integers(0, 2**31)),
                    "region": region,
                })
    else:
        # Pure random sampling
        for i in range(num_brackets):
            bracket = VariableLBracket.random(rng)
            brackets.append({
                "params": _bracket_to_core_params(bracket),
                "seed": int(rng.integers(0, 2**31)),
                "region": "random",
            })

    return brackets


def _bracket_to_core_params(bracket) -> dict:
    """Extract core params from a VariableLBracket for recreation."""
    return {
        "leg1_length": bracket.leg1_length,
        "leg2_length": bracket.leg2_length,
        "width": bracket.width,
        "thickness": bracket.thickness,
        "fillet_radius": bracket.fillet_radius,
        "hole1_diameters": list(bracket.hole1_diameters),
        "hole1_distances": list(bracket.hole1_distances),
        "hole2_diameters": list(bracket.hole2_diameters),
        "hole2_distances": list(bracket.hole2_distances),
    }


def _sample_bracket_in_region(rng: np.random.Generator, region: str):
    """Sample a bracket biased toward a particular region of parameter space."""
    from graph_cad.data.l_bracket import VariableLBracket, VariableLBracketRanges

    ranges = VariableLBracketRanges()

    # Define region biases (0-1 range within each parameter's bounds)
    if region == "small":
        bias_low, bias_high = 0.0, 0.4
    elif region == "large":
        bias_low, bias_high = 0.6, 1.0
    else:  # medium
        bias_low, bias_high = 0.3, 0.7

    def sample_param(param_range: tuple[float, float]) -> float:
        low, high = param_range
        t = rng.uniform(bias_low, bias_high)
        return low + t * (high - low)

    # Try up to 10 times to generate a valid bracket
    for attempt in range(10):
        try:
            bracket = VariableLBracket(
                leg1_length=sample_param(ranges.leg1_length),
                leg2_length=sample_param(ranges.leg2_length),
                width=sample_param(ranges.width),
                thickness=sample_param(ranges.thickness),
                fillet_radius=0.0,  # Keep simple
                hole1_diameters=(),  # No holes for simpler testing
                hole1_distances=(),
                hole2_diameters=(),
                hole2_distances=(),
            )
            return bracket
        except ValueError:
            continue

    # Fallback to random
    return VariableLBracket.random(rng)


# =============================================================================
# Main Exploration Loop
# =============================================================================

def run_exploration(
    models: ModelBundle,
    brackets: list[dict],
    domain: dict,
    progress: bool = True,
) -> list[TrialResult]:
    """Run the full exploration across all brackets and instructions."""
    results = []

    # Build trial list
    trials = []
    for bracket_idx, bracket_info in enumerate(brackets):
        for param_name, config in domain.items():
            for magnitude in config["magnitudes"]:
                for direction in ["increase", "decrease"]:
                    instruction = generate_instruction(
                        domain, param_name, direction, magnitude
                    )
                    trials.append({
                        "bracket_idx": bracket_idx,
                        "bracket_info": bracket_info,
                        "param_name": param_name,
                        "direction": direction,
                        "magnitude": magnitude,
                        "instruction": instruction,
                    })

    print(f"\nRunning {len(trials)} trials across {len(brackets)} brackets...")

    iterator = tqdm(trials, disable=not progress, desc="Exploring")

    for trial in iterator:
        bracket_info = trial["bracket_info"]

        try:
            inference_result = run_single_inference(
                models,
                bracket_info["params"],
                trial["instruction"],
            )

            # Compute metrics
            param_name = trial["param_name"]
            direction = trial["direction"]
            magnitude = trial["magnitude"]

            orig_val = inference_result["original_params"][param_name]
            edit_val = inference_result["edited_params"][param_name]
            actual_change = edit_val - orig_val

            # For direction correctness
            expected_sign = 1 if direction == "increase" else -1
            correct_direction = (actual_change * expected_sign) > 0

            # Percentage of target achieved (signed)
            if magnitude > 0:
                target_pct = (actual_change * expected_sign / magnitude) * 100
            else:
                target_pct = 0.0

            # Compute all param changes
            param_changes = {
                name: inference_result["edited_params"][name] - inference_result["original_params"][name]
                for name in PARAM_NAMES
            }

            result = TrialResult(
                bracket_id=trial["bracket_idx"],
                bracket_seed=bracket_info["seed"],
                bracket_params=bracket_info["params"],
                parameter=param_name,
                direction=direction,
                magnitude=magnitude,
                instruction=trial["instruction"],
                delta_norm=inference_result["delta_norm"],
                original_params_pred=inference_result["original_params"],
                edited_params_pred=inference_result["edited_params"],
                param_changes=param_changes,
                target_param_change=actual_change,
                target_param_pct=target_pct,
                correct_direction=correct_direction,
                inference_time_ms=inference_result["inference_time_ms"],
            )
            results.append(result)

        except Exception as e:
            print(f"\nError on trial {trial['instruction']}: {e}")
            continue

    return results


def compute_summary(results: list[TrialResult]) -> dict:
    """Compute summary statistics from results."""
    if not results:
        return {}

    # Group by parameter
    by_param = {}
    for r in results:
        if r.parameter not in by_param:
            by_param[r.parameter] = []
        by_param[r.parameter].append(r)

    summary = {
        "overall": {
            "total_trials": len(results),
            "correct_direction_pct": 100 * sum(r.correct_direction for r in results) / len(results),
            "mean_target_pct": float(np.mean([r.target_param_pct for r in results])),
            "mean_inference_time_ms": float(np.mean([r.inference_time_ms for r in results])),
        },
        "by_parameter": {},
    }

    for param, param_results in by_param.items():
        inc_results = [r for r in param_results if r.direction == "increase"]
        dec_results = [r for r in param_results if r.direction == "decrease"]

        summary["by_parameter"][param] = {
            "total_trials": len(param_results),
            "correct_direction_pct": 100 * sum(r.correct_direction for r in param_results) / len(param_results),
            "mean_target_pct": float(np.mean([r.target_param_pct for r in param_results])),
            "increase": {
                "trials": len(inc_results),
                "correct_direction_pct": 100 * sum(r.correct_direction for r in inc_results) / len(inc_results) if inc_results else 0,
                "mean_target_pct": float(np.mean([r.target_param_pct for r in inc_results])) if inc_results else 0,
                "mean_actual_change": float(np.mean([r.target_param_change for r in inc_results])) if inc_results else 0,
            },
            "decrease": {
                "trials": len(dec_results),
                "correct_direction_pct": 100 * sum(r.correct_direction for r in dec_results) / len(dec_results) if dec_results else 0,
                "mean_target_pct": float(np.mean([r.target_param_pct for r in dec_results])) if dec_results else 0,
                "mean_actual_change": float(np.mean([r.target_param_change for r in dec_results])) if dec_results else 0,
            },
        }

    # By magnitude
    by_magnitude = {}
    for r in results:
        key = f"{r.parameter}_{r.magnitude}"
        if key not in by_magnitude:
            by_magnitude[key] = []
        by_magnitude[key].append(r)

    summary["by_magnitude"] = {
        key: {
            "trials": len(mag_results),
            "correct_direction_pct": 100 * sum(r.correct_direction for r in mag_results) / len(mag_results),
            "mean_target_pct": float(np.mean([r.target_param_pct for r in mag_results])),
        }
        for key, mag_results in by_magnitude.items()
    }

    return summary


def print_summary(summary: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("EXPLORATION SUMMARY")
    print("=" * 70)

    overall = summary["overall"]
    print(f"\nOverall ({overall['total_trials']} trials):")
    print(f"  Correct direction: {overall['correct_direction_pct']:.1f}%")
    print(f"  Mean target achieved: {overall['mean_target_pct']:.1f}%")
    print(f"  Mean inference time: {overall['mean_inference_time_ms']:.1f}ms")

    print("\nBy Parameter:")
    print("-" * 70)
    print(f"{'Parameter':<18} {'Trials':>8} {'Correct%':>10} {'Target%':>10} {'Inc%':>8} {'Dec%':>8}")
    print("-" * 70)

    for param, stats in summary["by_parameter"].items():
        print(f"{param:<18} {stats['total_trials']:>8} {stats['correct_direction_pct']:>10.1f} "
              f"{stats['mean_target_pct']:>10.1f} {stats['increase']['correct_direction_pct']:>8.1f} "
              f"{stats['decrease']['correct_direction_pct']:>8.1f}")

    print("-" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Systematically explore the instruction domain (leg lengths only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Exploration parameters
    parser.add_argument(
        "--num-brackets",
        type=int,
        default=10,
        help="Number of random brackets to test (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bracket generation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test only leg1/leg2 with single magnitude (20mm)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Stratified sampling across parameter space regions",
    )

    # Model checkpoints (defaults to current working configuration)
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_transformer_aux2_w100/best_model.pt",
    )
    parser.add_argument(
        "--editor-checkpoint",
        type=str,
        default="outputs/latent_editor_tvae/best_model.pt",
    )
    parser.add_argument(
        "--latent-regressor-checkpoint",
        type=str,
        default="outputs/latent_regressor_tvae/best_model.pt",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exploration/results.json",
        help="Output JSON file path",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Select domain
    domain = QUICK_DOMAIN if args.quick else INSTRUCTION_DOMAIN

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load models once
    start_time = time.time()
    models = ModelBundle(
        args.vae_checkpoint,
        args.editor_checkpoint,
        args.latent_regressor_checkpoint,
        device,
    )
    load_time = time.time() - start_time
    print(f"Model loading took {load_time:.1f}s")

    # Sample brackets
    print(f"\nSampling {args.num_brackets} brackets (stratified={args.stratified})...")
    brackets = sample_brackets(args.num_brackets, args.seed, args.stratified)

    # Count expected trials
    num_trials = 0
    for param, config in domain.items():
        num_trials += len(config["magnitudes"]) * 2  # increase + decrease
    num_trials *= len(brackets)
    print(f"Expected trials: {num_trials}")

    # Run exploration
    explore_start = time.time()
    results = run_exploration(models, brackets, domain)
    explore_time = time.time() - explore_start

    # Compute summary
    summary = compute_summary(results)
    print_summary(summary)

    # Build output
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_brackets": args.num_brackets,
            "num_trials": len(results),
            "total_time_seconds": explore_time,
            "model_load_time_seconds": load_time,
            "seed": args.seed,
            "quick_mode": args.quick,
        },
        "checkpoints": {
            "vae": args.vae_checkpoint,
            "editor": args.editor_checkpoint,
            "latent_regressor": args.latent_regressor_checkpoint,
        },
        "domain": {
            param: {
                "magnitudes": config["magnitudes"],
            }
            for param, config in domain.items()
        },
        "brackets": brackets,
        "summary": summary,
        "trials": [asdict(r) for r in results],
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
