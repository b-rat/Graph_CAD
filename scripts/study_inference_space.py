#!/usr/bin/env python3
"""
Inference Space Study for Phase 4.

Explores the full inference pipeline by capturing three stages:
1. Ground truth starting parameters
2. VAE-predicted starting parameters (before LLM editing)
3. LLM-predicted ending parameters (after instruction)

This helps diagnose where errors originate:
- VAE encoding error (ground truth → VAE-predicted start)
- LLM edit accuracy (VAE-predicted start → LLM-predicted end)
- Total pipeline error (ground truth → LLM-predicted end)

Usage:
    # Run on GPU (RunPod)
    python scripts/study_inference_space.py \
        --vae-checkpoint outputs/hetero_vae_v2/best_model.pt \
        --llm-checkpoint outputs/llm_instruct_v2/best_model.pt \
        --samples-per-type 20 \
        --output-dir outputs/inference_space_study

    # Quick test (fewer samples)
    python scripts/study_inference_space.py --samples-per-type 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    Tube, Channel, Block, Cylinder, BlockHole, SimpleBracket,
)


GEOMETRY_CLASSES = {
    0: SimpleBracket,
    1: Tube,
    2: Channel,
    3: Block,
    4: Cylinder,
    5: BlockHole,
}

# Standard instructions for each geometry type
# Format: (instruction, target_param, expected_delta_sign)
# expected_delta_sign: +1 for increase, -1 for decrease
INSTRUCTIONS_BY_TYPE = {
    0: [  # Bracket
        ("make leg1 +20mm longer", "leg1_length", +1),
        ("make leg1 -15mm shorter", "leg1_length", -1),
        ("make leg2 +25mm longer", "leg2_length", +1),
        ("make width +10mm wider", "width", +1),
        ("make thickness +3mm thicker", "thickness", +1),
    ],
    1: [  # Tube
        ("make it +20mm longer", "length", +1),
        ("make it -15mm shorter", "length", -1),
        ("make outer_dia +10mm bigger", "outer_dia", +1),
        ("make inner_dia +5mm larger", "inner_dia", +1),
    ],
    2: [  # Channel
        ("make width +15mm wider", "width", +1),
        ("make height +25mm taller", "height", +1),
        ("make length +30mm longer", "length", +1),
        ("make thickness +2mm thicker", "thickness", +1),
    ],
    3: [  # Block
        ("make length +20mm longer", "length", +1),
        ("make it -30mm shorter in length", "length", -1),
        ("make width +15mm wider", "width", +1),
        ("make height +25mm taller", "height", +1),
    ],
    4: [  # Cylinder
        ("make it +20mm longer", "length", +1),
        ("make it -15mm shorter", "length", -1),
        ("make diameter +10mm bigger", "diameter", +1),
        ("make diameter -8mm smaller", "diameter", -1),
    ],
    5: [  # BlockHole
        ("make length +20mm longer", "length", +1),
        ("make width +15mm wider", "width", +1),
        ("make height +25mm taller", "height", +1),
        ("make hole_dia +5mm bigger", "hole_dia", +1),
    ],
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


def load_vae(checkpoint_path: str, device: str):
    """Load trained HeteroVAE model."""
    from graph_cad.models.hetero_vae import HeteroVAE, HeteroVAEConfig

    print(f"Loading HeteroVAE from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = HeteroVAEConfig(**checkpoint["config"])
    model = HeteroVAE(config, use_param_head=True, num_params=6)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    return model, config


def load_llm(checkpoint_path: str, vae_config, device: str):
    """Load trained ExtendedLatentEditor."""
    from graph_cad.models.extended_latent_editor import (
        ExtendedLatentEditor,
        ExtendedLatentEditorConfig,
    )
    from graph_cad.models.latent_editor import load_llm_with_lora, LatentEditorConfig

    print(f"Loading ExtendedLatentEditor from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create config
    config = ExtendedLatentEditorConfig(
        latent_dim=vae_config.latent_dim,
        training_mode="instruct",
    )

    # Create model without LLM first
    model = ExtendedLatentEditor(config, llm=None, tokenizer=None)

    # Load state dict (excluding LLM weights)
    state_dict = checkpoint["model_state_dict"]
    compatible_state = {k: v for k, v in state_dict.items() if not k.startswith('llm.')}
    model.load_state_dict(compatible_state, strict=False)

    # Load Mistral with LoRA
    print("  Loading Mistral 7B with LoRA...")
    lora_config = LatentEditorConfig()
    mistral, tokenizer = load_llm_with_lora(lora_config, device_map="auto")

    # Load trained LoRA adapter weights
    lora_path = checkpoint_path.replace('.pt', '_lora')
    if Path(lora_path).exists():
        print(f"  Loading trained LoRA adapter from {lora_path}...")
        mistral.load_adapter(lora_path, adapter_name="default")
    else:
        print(f"  Warning: LoRA adapter not found at {lora_path}, using untrained weights")

    model.set_llm(mistral, tokenizer)

    # Move non-LLM components to device
    model.latent_projector = model.latent_projector.to(device)
    model.output_projector = model.output_projector.to(device)
    model.class_head = model.class_head.to(device)
    model.param_heads = model.param_heads.to(device)
    model.pretrain_encoder = model.pretrain_encoder.to(device)

    model.eval()
    print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    return model


def test_single_sample(
    vae,
    llm,
    geometry_type: int,
    seed: int,
    instruction: str,
    target_param: str,
    expected_sign: int,
    device: str,
) -> dict:
    """
    Test a single geometry with a specific instruction.

    Returns a dict with:
    - ground_truth_start: Original parameters
    - vae_predicted_start: VAE-reconstructed parameters (before LLM)
    - llm_predicted_end: LLM-predicted parameters (after instruction)
    - errors at each stage
    """
    # Create geometry
    geometry = create_random_geometry(geometry_type, seed)

    # Get ground truth parameters
    gt_params = geometry_to_params(geometry, geometry_type)
    gt_params_norm = normalize_params(gt_params, geometry_type)
    num_params = len(gt_params)
    param_names = PARAM_NAMES[geometry_type]

    # Extract B-Rep graph
    try:
        solid = geometry.to_solid()
        graph = extract_brep_hetero_graph_from_solid(solid)
    except Exception as e:
        return {'error': str(e), 'seed': seed}

    # Convert to HeteroData
    params_padded = torch.zeros(6)
    params_padded[:num_params] = gt_params
    params_norm_padded = torch.zeros(6)
    params_norm_padded[:num_params] = gt_params_norm
    params_mask = torch.zeros(6)
    params_mask[:num_params] = 1.0

    data = brep_graph_to_hetero_data(
        graph, geometry_type, params_padded, params_norm_padded, params_mask
    )
    data = data.to(device)

    # Stage 1: Encode through VAE
    with torch.no_grad():
        mu, logvar = vae.encode(data)
        z = mu  # Use mean for inference

        # Get VAE-predicted parameters (before any editing)
        vae_param_pred = vae.param_head(mu)[0]  # (6,)

        # Classify geometry type
        geo_logits = vae.geometry_type_head(mu)
        vae_pred_type = geo_logits.argmax(dim=-1).item()

    # Denormalize VAE-predicted params
    vae_params_norm = vae_param_pred[:num_params].cpu()
    vae_params = denormalize_params(vae_params_norm, geometry_type)

    # Stage 2: Apply instruction through LLM
    with torch.no_grad():
        outputs = llm.forward_instruct(
            z, [instruction],
            geometry_types=torch.tensor([geometry_type], device=device)
        )

        # Get predicted parameter delta
        param_delta = outputs['param_pred'][0]  # (6,)
        z_edited = outputs['z_edited'][0]  # (32,)

        # LLM also classifies - get its prediction
        llm_class_logits = outputs['class_logits'][0]
        llm_pred_type = llm_class_logits.argmax(dim=-1).item()

    # Compute LLM-predicted end params
    # Note: LLM outputs delta from the VAE-predicted baseline
    llm_params_norm = vae_params_norm + param_delta[:num_params].cpu()
    llm_params_norm = llm_params_norm.clamp(0, 1)
    llm_params = denormalize_params(llm_params_norm, geometry_type)

    # Also compute what the current inference script does (delta from ground truth)
    # This is for comparison
    llm_params_from_gt_norm = gt_params_norm + param_delta[:num_params].cpu()
    llm_params_from_gt_norm = llm_params_from_gt_norm.clamp(0, 1)
    llm_params_from_gt = denormalize_params(llm_params_from_gt_norm, geometry_type)

    # Find target parameter index
    target_idx = param_names.index(target_param)

    # Compute errors and metrics
    result = {
        'seed': seed,
        'geometry_type': GEOMETRY_TYPE_NAMES[geometry_type],
        'instruction': instruction,
        'target_param': target_param,
        'expected_sign': expected_sign,
        'vae_pred_type': GEOMETRY_TYPE_NAMES[vae_pred_type],
        'llm_pred_type': GEOMETRY_TYPE_NAMES[llm_pred_type],
        'type_correct': vae_pred_type == geometry_type,

        # Per-parameter breakdown
        'params': {},

        # Summary metrics
        'summary': {},
    }

    # Build per-parameter data
    for i, name in enumerate(param_names):
        gt_val = float(gt_params[i])
        vae_val = float(vae_params[i])
        llm_val = float(llm_params[i])  # LLM edit from VAE baseline
        llm_from_gt_val = float(llm_params_from_gt[i])  # LLM edit from GT baseline

        # Errors
        vae_error = vae_val - gt_val  # VAE reconstruction error (signed)
        llm_delta_from_vae = llm_val - vae_val  # LLM's edit delta
        total_error = llm_val - gt_val  # End-to-end error (signed)

        # For the current inference approach (delta from GT)
        llm_delta_from_gt = llm_from_gt_val - gt_val

        result['params'][name] = {
            'ground_truth': gt_val,
            'vae_predicted': vae_val,
            'llm_predicted': llm_val,
            'llm_predicted_from_gt': llm_from_gt_val,
            'vae_error': vae_error,
            'llm_delta_from_vae': llm_delta_from_vae,
            'llm_delta_from_gt': llm_delta_from_gt,
            'total_error': total_error,
            'is_target': name == target_param,
        }

    # Summary metrics
    target_data = result['params'][target_param]

    # Check direction correctness
    llm_delta = target_data['llm_delta_from_vae']
    direction_correct = (llm_delta > 0) == (expected_sign > 0) if llm_delta != 0 else False

    # Non-target parameter stability (should be near zero delta)
    non_target_deltas = [
        abs(result['params'][name]['llm_delta_from_vae'])
        for name in param_names if name != target_param
    ]

    result['summary'] = {
        # VAE reconstruction quality
        'vae_mae': float(np.mean([abs(result['params'][n]['vae_error']) for n in param_names])),
        'vae_target_error': target_data['vae_error'],

        # LLM edit quality
        'direction_correct': direction_correct,
        'target_delta_achieved': llm_delta,
        'target_delta_from_gt': target_data['llm_delta_from_gt'],
        'non_target_mean_delta': float(np.mean(non_target_deltas)) if non_target_deltas else 0,
        'non_target_max_delta': float(np.max(non_target_deltas)) if non_target_deltas else 0,

        # End-to-end
        'total_mae': float(np.mean([abs(result['params'][n]['total_error']) for n in param_names])),
    }

    return result


def run_study(
    vae,
    llm,
    samples_per_instruction: int,
    device: str,
    base_seed: int = 42,
) -> dict:
    """Run inference space study across all geometry types and instructions."""
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'samples_per_instruction': samples_per_instruction,
            'base_seed': base_seed,
        },
        'by_geometry_type': {},
        'summary': {},
    }

    # Aggregate metrics
    all_vae_maes = []
    all_direction_correct = []
    all_target_deltas = []
    all_non_target_deltas = []

    for geo_type in range(6):
        geo_name = GEOMETRY_TYPE_NAMES[geo_type]
        instructions = INSTRUCTIONS_BY_TYPE[geo_type]

        print(f"\n{'='*60}")
        print(f"Testing {geo_name.upper()}")
        print(f"{'='*60}")

        geo_results = {
            'instructions': {},
            'summary': {},
        }

        geo_vae_maes = []
        geo_direction_correct = []
        geo_target_deltas = []
        geo_non_target_deltas = []

        for instr_idx, (instruction, target_param, expected_sign) in enumerate(instructions):
            print(f"\n  Instruction: \"{instruction}\"")
            print(f"  Target: {target_param}, Expected: {'increase' if expected_sign > 0 else 'decrease'}")

            instr_results = []
            instr_direction_correct = 0

            for sample_idx in range(samples_per_instruction):
                seed = base_seed + geo_type * 10000 + instr_idx * 1000 + sample_idx

                result = test_single_sample(
                    vae, llm, geo_type, seed,
                    instruction, target_param, expected_sign, device
                )

                if 'error' not in result:
                    instr_results.append(result)

                    # Aggregate
                    geo_vae_maes.append(result['summary']['vae_mae'])
                    all_vae_maes.append(result['summary']['vae_mae'])

                    if result['summary']['direction_correct']:
                        instr_direction_correct += 1
                        geo_direction_correct.append(1)
                        all_direction_correct.append(1)
                    else:
                        geo_direction_correct.append(0)
                        all_direction_correct.append(0)

                    geo_target_deltas.append(abs(result['summary']['target_delta_achieved']))
                    all_target_deltas.append(abs(result['summary']['target_delta_achieved']))

                    geo_non_target_deltas.append(result['summary']['non_target_mean_delta'])
                    all_non_target_deltas.append(result['summary']['non_target_mean_delta'])
                else:
                    print(f"    Sample {sample_idx}: Error - {result['error']}")

            # Instruction-level summary
            n_samples = len(instr_results)
            instr_summary = {
                'samples': n_samples,
                'direction_accuracy': instr_direction_correct / n_samples if n_samples else 0,
                'mean_target_delta': float(np.mean([r['summary']['target_delta_achieved'] for r in instr_results])) if instr_results else 0,
                'mean_vae_mae': float(np.mean([r['summary']['vae_mae'] for r in instr_results])) if instr_results else 0,
            }

            print(f"    Direction accuracy: {instr_summary['direction_accuracy']*100:.0f}%")
            print(f"    Mean target delta: {instr_summary['mean_target_delta']:+.2f} mm")
            print(f"    Mean VAE MAE: {instr_summary['mean_vae_mae']:.2f} mm")

            geo_results['instructions'][instruction] = {
                'target_param': target_param,
                'expected_sign': expected_sign,
                'summary': instr_summary,
                'samples': instr_results,
            }

        # Geometry-level summary
        geo_results['summary'] = {
            'total_samples': len(geo_vae_maes),
            'direction_accuracy': float(np.mean(geo_direction_correct)) if geo_direction_correct else 0,
            'mean_vae_mae': float(np.mean(geo_vae_maes)) if geo_vae_maes else 0,
            'mean_target_delta': float(np.mean(geo_target_deltas)) if geo_target_deltas else 0,
            'mean_non_target_delta': float(np.mean(geo_non_target_deltas)) if geo_non_target_deltas else 0,
        }

        print(f"\n  {geo_name} Summary:")
        print(f"    Direction accuracy: {geo_results['summary']['direction_accuracy']*100:.1f}%")
        print(f"    Mean VAE MAE: {geo_results['summary']['mean_vae_mae']:.2f} mm")
        print(f"    Mean target delta: {geo_results['summary']['mean_target_delta']:.2f} mm")
        print(f"    Mean non-target delta: {geo_results['summary']['mean_non_target_delta']:.2f} mm")

        results['by_geometry_type'][geo_name] = geo_results

    # Overall summary
    results['summary'] = {
        'total_samples': len(all_vae_maes),
        'overall_direction_accuracy': float(np.mean(all_direction_correct)) if all_direction_correct else 0,
        'overall_vae_mae': float(np.mean(all_vae_maes)) if all_vae_maes else 0,
        'overall_vae_mae_std': float(np.std(all_vae_maes)) if all_vae_maes else 0,
        'overall_target_delta_mean': float(np.mean(all_target_deltas)) if all_target_deltas else 0,
        'overall_non_target_delta_mean': float(np.mean(all_non_target_deltas)) if all_non_target_deltas else 0,
    }

    return results


def print_summary_table(results: dict):
    """Print a nicely formatted summary table."""
    print("\n" + "="*80)
    print("INFERENCE SPACE STUDY - SUMMARY")
    print("="*80)

    # Per-geometry summary
    print("\nPer-Geometry Results:")
    print("-"*80)
    print(f"{'Geometry':<12} {'Dir Acc':<10} {'VAE MAE':<12} {'Target Δ':<12} {'Non-Target Δ':<12}")
    print("-"*80)

    for geo_name, geo_data in results['by_geometry_type'].items():
        s = geo_data['summary']
        print(f"{geo_name:<12} {s['direction_accuracy']*100:>6.1f}%    {s['mean_vae_mae']:>8.2f} mm  {s['mean_target_delta']:>8.2f} mm  {s['mean_non_target_delta']:>8.2f} mm")

    print("-"*80)

    # Overall
    s = results['summary']
    print(f"\nOverall Results:")
    print(f"  Total samples: {s['total_samples']}")
    print(f"  Direction accuracy: {s['overall_direction_accuracy']*100:.1f}%")
    print(f"  VAE reconstruction MAE: {s['overall_vae_mae']:.2f} ± {s['overall_vae_mae_std']:.2f} mm")
    print(f"  Mean target parameter delta: {s['overall_target_delta_mean']:.2f} mm")
    print(f"  Mean non-target parameter delta: {s['overall_non_target_delta_mean']:.2f} mm")

    # Error attribution
    print(f"\nError Attribution:")
    print(f"  VAE introduces ~{s['overall_vae_mae']:.1f}mm error before LLM editing")
    print(f"  Non-target params drift ~{s['overall_non_target_delta_mean']:.1f}mm during editing")


def main():
    parser = argparse.ArgumentParser(description="Inference Space Study")

    parser.add_argument("--vae-checkpoint", type=str,
                        default="outputs/hetero_vae_v2/best_model.pt",
                        help="Path to HeteroVAE checkpoint")
    parser.add_argument("--llm-checkpoint", type=str,
                        default="outputs/llm_instruct_v2/best_model.pt",
                        help="Path to LLM checkpoint")
    parser.add_argument("--samples-per-instruction", type=int, default=10,
                        help="Number of samples per instruction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/inference_space_study",
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

    # Load models
    vae, vae_config = load_vae(args.vae_checkpoint, device)
    llm = load_llm(args.llm_checkpoint, vae_config, device)

    # Calculate total samples
    total_instructions = sum(len(INSTRUCTIONS_BY_TYPE[i]) for i in range(6))
    total_samples = total_instructions * args.samples_per_instruction

    print(f"\nRunning inference space study...")
    print(f"  Instructions per geometry: {[len(INSTRUCTIONS_BY_TYPE[i]) for i in range(6)]}")
    print(f"  Total instructions: {total_instructions}")
    print(f"  Samples per instruction: {args.samples_per_instruction}")
    print(f"  Total samples: {total_samples}")

    # Run study
    results = run_study(
        vae, llm,
        samples_per_instruction=args.samples_per_instruction,
        device=device,
        base_seed=args.seed,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "inference_space_study.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print_summary_table(results)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
