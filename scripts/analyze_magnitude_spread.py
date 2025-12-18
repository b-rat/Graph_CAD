#!/usr/bin/env python3
"""
Analyze magnitude spread in edit predictions.

For each trial, compare:
- Instruction magnitude (e.g., "20mm longer" → 20)
- Actual target parameter change
- Sum of absolute changes across all parameters

This tests the hypothesis that the model produces the correct total magnitude
but spreads it across entangled parameters rather than concentrating on the target.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def analyze_trial(trial: dict) -> dict:
    """Analyze a single trial for magnitude spread."""

    param_changes = trial["param_changes"]
    target_param = trial["parameter"]
    instruction_magnitude = trial["magnitude"]
    direction = trial["direction"]

    # Calculate sum of absolute changes
    sum_abs_changes = sum(abs(v) for v in param_changes.values())

    # Get target parameter change
    target_change = param_changes.get(target_param, 0)

    # Calculate spillover (change to non-target params)
    spillover = sum_abs_changes - abs(target_change)

    # Expected sign
    expected_sign = 1 if direction == "increase" else -1
    actual_sign = np.sign(target_change) if target_change != 0 else 0
    correct_direction = (expected_sign == actual_sign)

    return {
        "parameter": target_param,
        "direction": direction,
        "instruction_magnitude": instruction_magnitude,
        "target_change": target_change,
        "sum_abs_changes": sum_abs_changes,
        "spillover": spillover,
        "correct_direction": correct_direction,
        "target_pct_of_instruction": (abs(target_change) / instruction_magnitude * 100) if instruction_magnitude > 0 else 0,
        "sum_pct_of_instruction": (sum_abs_changes / instruction_magnitude * 100) if instruction_magnitude > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze magnitude spread in predictions")
    parser.add_argument("input_file", type=Path, help="Path to full_study JSON file")
    parser.add_argument("--by-param", action="store_true", help="Show breakdown by parameter")
    parser.add_argument("--by-magnitude", action="store_true", help="Show breakdown by instruction magnitude")
    parser.add_argument("--by-direction", action="store_true", help="Show breakdown by direction")
    args = parser.parse_args()

    # Load data
    with open(args.input_file) as f:
        data = json.load(f)

    trials = data["trials"]
    print(f"Analyzing {len(trials)} trials from {args.input_file.name}\n")

    # Analyze all trials
    results = [analyze_trial(t) for t in trials]

    # Overall statistics
    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    instruction_mags = [r["instruction_magnitude"] for r in results]
    target_changes = [abs(r["target_change"]) for r in results]
    sum_changes = [r["sum_abs_changes"] for r in results]
    spillovers = [r["spillover"] for r in results]
    target_pcts = [r["target_pct_of_instruction"] for r in results]
    sum_pcts = [r["sum_pct_of_instruction"] for r in results]

    print(f"\nInstruction magnitude:     mean={np.mean(instruction_mags):6.2f}mm  (range: {min(instruction_mags)}-{max(instruction_mags)})")
    print(f"Target param |change|:     mean={np.mean(target_changes):6.2f}mm  (std: {np.std(target_changes):.2f})")
    print(f"Sum of all |changes|:      mean={np.mean(sum_changes):6.2f}mm  (std: {np.std(sum_changes):.2f})")
    print(f"Spillover to other params: mean={np.mean(spillovers):6.2f}mm  (std: {np.std(spillovers):.2f})")

    print(f"\nTarget as % of instruction:  mean={np.mean(target_pcts):5.1f}%  (std: {np.std(target_pcts):.1f}%)")
    print(f"Sum as % of instruction:     mean={np.mean(sum_pcts):5.1f}%  (std: {np.std(sum_pcts):.1f}%)")

    # Direction accuracy
    correct = sum(1 for r in results if r["correct_direction"])
    print(f"\nDirection accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

    # Hypothesis test: Is sum ≈ instruction magnitude?
    print("\n" + "-" * 70)
    print("HYPOTHESIS: Sum of |changes| ≈ instruction magnitude")
    print("-" * 70)

    # Bin by how close sum is to instruction
    close_count = sum(1 for r in results if 0.5 <= r["sum_pct_of_instruction"] <= 1.5)
    under_count = sum(1 for r in results if r["sum_pct_of_instruction"] < 0.5)
    over_count = sum(1 for r in results if r["sum_pct_of_instruction"] > 1.5)

    # Actually use percentage ranges that make sense
    bins = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 200), (200, float('inf'))]
    print("\nSum as % of instruction magnitude:")
    for low, high in bins:
        count = sum(1 for r in results if low <= r["sum_pct_of_instruction"] < high)
        pct = 100 * count / len(results)
        bar = "#" * int(pct / 2)
        if high == float('inf'):
            print(f"  {low:3d}%+      : {count:4d} ({pct:5.1f}%) {bar}")
        else:
            print(f"  {low:3d}-{high:<3d}%  : {count:4d} ({pct:5.1f}%) {bar}")

    # By parameter
    if args.by_param:
        print("\n" + "=" * 70)
        print("BY PARAMETER")
        print("=" * 70)

        by_param = defaultdict(list)
        for r in results:
            by_param[r["parameter"]].append(r)

        print(f"\n{'Parameter':<16} {'N':>5} {'Target':>8} {'Sum':>8} {'Spill':>8} {'Tgt%':>7} {'Sum%':>7} {'DirAcc':>7}")
        print("-" * 78)

        for param in sorted(by_param.keys()):
            pr = by_param[param]
            n = len(pr)
            tgt = np.mean([r["target_change"] for r in pr])
            sm = np.mean([r["sum_abs_changes"] for r in pr])
            sp = np.mean([r["spillover"] for r in pr])
            tpct = np.mean([r["target_pct_of_instruction"] for r in pr])
            spct = np.mean([r["sum_pct_of_instruction"] for r in pr])
            dacc = 100 * sum(1 for r in pr if r["correct_direction"]) / n
            print(f"{param:<16} {n:>5} {tgt:>+8.2f} {sm:>8.2f} {sp:>8.2f} {tpct:>6.1f}% {spct:>6.1f}% {dacc:>6.1f}%")

    # By instruction magnitude
    if args.by_magnitude:
        print("\n" + "=" * 70)
        print("BY INSTRUCTION MAGNITUDE")
        print("=" * 70)

        by_mag = defaultdict(list)
        for r in results:
            by_mag[r["instruction_magnitude"]].append(r)

        print(f"\n{'Magnitude':>10} {'N':>5} {'Target':>8} {'Sum':>8} {'Spill':>8} {'Tgt%':>7} {'Sum%':>7}")
        print("-" * 65)

        for mag in sorted(by_mag.keys()):
            mr = by_mag[mag]
            n = len(mr)
            tgt = np.mean([abs(r["target_change"]) for r in mr])
            sm = np.mean([r["sum_abs_changes"] for r in mr])
            sp = np.mean([r["spillover"] for r in mr])
            tpct = np.mean([r["target_pct_of_instruction"] for r in mr])
            spct = np.mean([r["sum_pct_of_instruction"] for r in mr])
            print(f"{mag:>10}mm {n:>5} {tgt:>8.2f} {sm:>8.2f} {sp:>8.2f} {tpct:>6.1f}% {spct:>6.1f}%")

    # By direction
    if args.by_direction:
        print("\n" + "=" * 70)
        print("BY DIRECTION")
        print("=" * 70)

        by_dir = defaultdict(list)
        for r in results:
            by_dir[r["direction"]].append(r)

        print(f"\n{'Direction':>10} {'N':>5} {'Target':>8} {'Sum':>8} {'Spill':>8} {'Tgt%':>7} {'Sum%':>7} {'DirAcc':>7}")
        print("-" * 72)

        for direction in ["increase", "decrease"]:
            dr = by_dir[direction]
            n = len(dr)
            tgt = np.mean([abs(r["target_change"]) for r in dr])
            sm = np.mean([r["sum_abs_changes"] for r in dr])
            sp = np.mean([r["spillover"] for r in dr])
            tpct = np.mean([r["target_pct_of_instruction"] for r in dr])
            spct = np.mean([r["sum_pct_of_instruction"] for r in dr])
            dacc = 100 * sum(1 for r in dr if r["correct_direction"]) / n
            print(f"{direction:>10} {n:>5} {tgt:>8.2f} {sm:>8.2f} {sp:>8.2f} {tpct:>6.1f}% {spct:>6.1f}% {dacc:>6.1f}%")


if __name__ == "__main__":
    main()
