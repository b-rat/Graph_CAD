#!/usr/bin/env python3
"""
Systematic exploration of the instruction domain for the Latent Editor.

This script loads models once and runs inference across a grid of:
  - Parameters: leg1, leg2, width, thickness, hole1_diameter, hole2_diameter
  - Directions: increase, decrease
  - Magnitudes: parameter-appropriate values (e.g., 10/20/30mm for legs)
  - Starting brackets: sampled across the parameter space

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
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Instruction Domain Definition
# =============================================================================

@dataclass
class InstructionTemplate:
    """Template for generating instructions."""
    parameter: str  # Internal parameter name
    display_name: str  # Human-readable name for instructions
    direction: str  # "increase" or "decrease"
    magnitude: float  # Amount in mm
    templates: list[str]  # Instruction templates with {name} and {amount} placeholders


# Define the instruction domain
INSTRUCTION_DOMAIN = {
    "leg1_length": {
        "display_name": "leg1",
        "magnitudes": [10, 20, 30, 50],
        "increase_templates": [
            "make {name} {amount}mm longer",
            "increase {name} by {amount}mm",
            "extend {name} by {amount}mm",
        ],
        "decrease_templates": [
            "make {name} {amount}mm shorter",
            "decrease {name} by {amount}mm",
            "shorten {name} by {amount}mm",
        ],
    },
    "leg2_length": {
        "display_name": "leg2",
        "magnitudes": [10, 20, 30, 50],
        "increase_templates": [
            "make {name} {amount}mm longer",
            "increase {name} by {amount}mm",
            "extend {name} by {amount}mm",
        ],
        "decrease_templates": [
            "make {name} {amount}mm shorter",
            "decrease {name} by {amount}mm",
            "shorten {name} by {amount}mm",
        ],
    },
    "width": {
        "display_name": "width",
        "magnitudes": [5, 10, 15],
        "increase_templates": [
            "make it {amount}mm wider",
            "increase width by {amount}mm",
        ],
        "decrease_templates": [
            "make it {amount}mm narrower",
            "decrease width by {amount}mm",
        ],
    },
    "thickness": {
        "display_name": "thickness",
        "magnitudes": [1, 2, 3],
        "increase_templates": [
            "make it {amount}mm thicker",
            "increase thickness by {amount}mm",
        ],
        "decrease_templates": [
            "make it {amount}mm thinner",
            "decrease thickness by {amount}mm",
        ],
    },
    "hole1_diameter": {
        "display_name": "hole1",
        "magnitudes": [1, 2, 3],
        "increase_templates": [
            "make {name} {amount}mm larger",
            "increase {name} diameter by {amount}mm",
        ],
        "decrease_templates": [
            "make {name} {amount}mm smaller",
            "decrease {name} diameter by {amount}mm",
        ],
    },
    "hole2_diameter": {
        "display_name": "hole2",
        "magnitudes": [1, 2, 3],
        "increase_templates": [
            "make {name} {amount}mm larger",
            "increase {name} diameter by {amount}mm",
        ],
        "decrease_templates": [
            "make {name} {amount}mm smaller",
            "decrease {name} diameter by {amount}mm",
        ],
    },
}

# Quick mode uses subset
QUICK_DOMAIN = {
    "leg1_length": {
        "display_name": "leg1",
        "magnitudes": [20],
        "increase_templates": ["make {name} {amount}mm longer"],
        "decrease_templates": ["make {name} {amount}mm shorter"],
    },
    "leg2_length": {
        "display_name": "leg2",
        "magnitudes": [20],
        "increase_templates": ["make {name} {amount}mm longer"],
        "decrease_templates": ["make {name} {amount}mm shorter"],
    },
}


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
    node_mse: float
    edge_mse: float

    # Timing
    inference_time_ms: float


@dataclass
class ExplorationResults:
    """Complete exploration results."""
    # Metadata
    timestamp: str
    num_brackets: int
    num_trials: int
    total_time_seconds: float

    # Model info
    vae_checkpoint: str
    editor_checkpoint: str
    regressor_checkpoint: str

    # Domain info
    parameters_tested: list[str]
    magnitudes_tested: dict[str, list[float]]

    # Results
    trials: list[dict]

    # Summary statistics
    summary: dict


# =============================================================================
# Model Loading (done once)
# =============================================================================

class ModelBundle:
    """Container for all loaded models."""

    def __init__(
        self,
        vae_checkpoint: str,
        editor_checkpoint: str,
        regressor_checkpoint: str,
        device: str,
    ):
        self.device = device
        self.vae = None
        self.editor = None
        self.regressor = None

        self._load_models(vae_checkpoint, editor_checkpoint, regressor_checkpoint)

    def _load_models(
        self,
        vae_checkpoint: str,
        editor_checkpoint: str,
        regressor_checkpoint: str,
    ):
        """Load all models once."""
        from graph_cad.training.vae_trainer import load_checkpoint as load_vae_checkpoint
        from graph_cad.models.feature_regressor import load_feature_regressor
        from graph_cad.models.latent_editor import (
            LatentEditor,
            load_llm_with_lora,
        )
        from graph_cad.training.edit_trainer import load_editor_checkpoint

        print("=" * 60)
        print("LOADING MODELS (this happens once)")
        print("=" * 60)

        # Load VAE
        print(f"\n[1/3] Loading VAE from {vae_checkpoint}...")
        self.vae, _ = load_vae_checkpoint(vae_checkpoint, device=self.device)
        self.vae.eval()
        print(f"  Latent dim: {self.vae.config.latent_dim}")

        # Load FeatureRegressor
        print(f"\n[2/3] Loading FeatureRegressor from {regressor_checkpoint}...")
        self.regressor, _ = load_feature_regressor(regressor_checkpoint, device=self.device)
        self.regressor.eval()
        print(f"  Architecture: {self.regressor.config.hidden_dims}")

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


# =============================================================================
# Inference Functions
# =============================================================================

def run_single_inference(
    models: ModelBundle,
    bracket_params: dict[str, float],
    instruction: str,
) -> dict:
    """Run inference for a single bracket + instruction pair."""
    from graph_cad.data import LBracket, extract_graph_from_solid
    from graph_cad.models.parameter_regressor import PARAMETER_NAMES, denormalize_parameters

    device = models.device

    # Create bracket and extract graph
    bracket = LBracket(**bracket_params)
    solid = bracket.to_solid()
    graph = extract_graph_from_solid(solid)

    # Convert to tensors
    x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)

    # Encode
    with torch.no_grad():
        mu, logvar = models.vae.encode(x, edge_index, edge_attr, batch=None)
        z_src = mu

    # Apply instruction
    start_time = time.perf_counter()
    with torch.no_grad():
        z_src_editor = z_src.cpu() if device != "cuda" else z_src
        outputs = models.editor(z_src_editor, [instruction])
        delta_z = outputs["delta_z"].to(device)
        z_edited = outputs["z_edited"].to(device)
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # Decode
    with torch.no_grad():
        orig_node_recon, orig_edge_recon = models.vae.decode(z_src)
        edit_node_recon, edit_edge_recon = models.vae.decode(z_edited)

    # Predict parameters
    with torch.no_grad():
        orig_params_norm = models.regressor(orig_node_recon, orig_edge_recon)
        edit_params_norm = models.regressor(edit_node_recon, edit_edge_recon)
        orig_params = denormalize_parameters(orig_params_norm).cpu().numpy().flatten()
        edit_params = denormalize_parameters(edit_params_norm).cpu().numpy().flatten()

    # Compute metrics
    orig_node_np = orig_node_recon.cpu().numpy()[0]
    edit_node_np = edit_node_recon.cpu().numpy()[0]
    orig_edge_np = orig_edge_recon.cpu().numpy()[0]
    edit_edge_np = edit_edge_recon.cpu().numpy()[0]

    node_mse = float(np.mean((edit_node_np - orig_node_np) ** 2))
    edge_mse = float(np.mean((edit_edge_np - orig_edge_np) ** 2))

    return {
        "delta_norm": float(delta_z.norm().item()),
        "original_params": {name: float(orig_params[i]) for i, name in enumerate(PARAMETER_NAMES)},
        "edited_params": {name: float(edit_params[i]) for i, name in enumerate(PARAMETER_NAMES)},
        "node_mse": node_mse,
        "edge_mse": edge_mse,
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
    from graph_cad.data import LBracket

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
                    "params": bracket.to_dict(),
                    "seed": int(rng.integers(0, 2**31)),
                    "region": region,
                })
    else:
        # Pure random sampling
        for i in range(num_brackets):
            bracket = LBracket.random(rng)
            brackets.append({
                "params": bracket.to_dict(),
                "seed": int(rng.integers(0, 2**31)),
                "region": "random",
            })

    return brackets


def _sample_bracket_in_region(rng: np.random.Generator, region: str) -> "LBracket":
    """Sample a bracket biased toward a particular region of parameter space."""
    from graph_cad.data import LBracket, LBracketRanges

    ranges = LBracketRanges()

    # Define region biases (0-1 range within each parameter's bounds)
    if region == "small":
        bias_low, bias_high = 0.0, 0.4
    elif region == "large":
        bias_low, bias_high = 0.6, 1.0
    else:  # medium
        bias_low, bias_high = 0.3, 0.7

    def sample_param(param_range: tuple[float, float]) -> float:
        low, high = param_range
        # Sample within biased region
        t = rng.uniform(bias_low, bias_high)
        return low + t * (high - low)

    leg1 = sample_param(ranges.leg1_length)
    leg2 = sample_param(ranges.leg2_length)
    width = sample_param(ranges.width)
    thickness = sample_param(ranges.thickness)
    hole1_d = sample_param(ranges.hole1_diameter)
    hole2_d = sample_param(ranges.hole2_diameter)

    # Compute valid hole distances
    # Hole must be at least radius from leg end and from inner corner
    min_hole1_dist = hole1_d / 2 + 2  # 2mm margin from end
    max_hole1_dist = leg1 - thickness - hole1_d / 2 - 2  # 2mm margin from corner
    hole1_dist = rng.uniform(
        max(min_hole1_dist, 10),
        max(max_hole1_dist, min_hole1_dist + 10)
    )

    min_hole2_dist = hole2_d / 2 + 2
    max_hole2_dist = leg2 - thickness - hole2_d / 2 - 2
    hole2_dist = rng.uniform(
        max(min_hole2_dist, 10),
        max(max_hole2_dist, min_hole2_dist + 10)
    )

    return LBracket(
        leg1_length=leg1,
        leg2_length=leg2,
        width=width,
        thickness=thickness,
        hole1_distance=hole1_dist,
        hole1_diameter=hole1_d,
        hole2_distance=hole2_dist,
        hole2_diameter=hole2_d,
    )


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
    from graph_cad.models.parameter_regressor import PARAMETER_NAMES

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
                for name in PARAMETER_NAMES
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
                node_mse=inference_result["node_mse"],
                edge_mse=inference_result["edge_mse"],
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
        description="Systematically explore the instruction domain",
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
        help="Quick mode: test only leg1/leg2 with single magnitude",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Stratified sampling across parameter space regions",
    )

    # Model checkpoints
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default="outputs/vae_16d_lowbeta/best_model.pt",
    )
    parser.add_argument(
        "--editor-checkpoint",
        type=str,
        default="outputs/latent_editor/best_model.pt",
    )
    parser.add_argument(
        "--regressor-checkpoint",
        type=str,
        default="outputs/feature_regressor/best_model.pt",
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
        args.regressor_checkpoint,
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
            "regressor": args.regressor_checkpoint,
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
