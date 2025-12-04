#!/usr/bin/env python3
"""
Generate synthetic L-bracket dataset for training.

Usage:
    python scripts/generate_l_brackets.py --output data/raw --count 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from graph_cad.data import LBracket, LBracketRanges

if TYPE_CHECKING:
    from numpy.random import Generator


def generate_dataset(
    output_dir: Path,
    count: int,
    rng: Generator,
    ranges: LBracketRanges,
) -> list[dict]:
    """
    Generate L-bracket dataset.

    Args:
        output_dir: Directory for STEP files.
        count: Number of brackets to generate.
        rng: NumPy random generator.
        ranges: Parameter ranges for generation.

    Returns:
        List of metadata dictionaries for each bracket.
    """
    step_dir = output_dir / "step"
    step_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for i in range(count):
        bracket = LBracket.random(rng, ranges)

        filename = f"bracket_{i:05d}.step"
        filepath = step_dir / filename

        bracket.to_step(filepath)

        record = {"id": i, "filename": filename, **bracket.to_dict()}
        metadata.append(record)

        if (i + 1) % 100 == 0 or i == 0:
            print(f"Generated {i + 1}/{count} brackets")

    return metadata


def save_metadata(metadata: list[dict], output_path: Path) -> None:
    """Save metadata to CSV file."""
    if not metadata:
        return

    fieldnames = list(metadata[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic L-bracket dataset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of brackets to generate (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Parameter range overrides
    parser.add_argument("--leg1-length", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--leg2-length", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--width", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--thickness", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--hole1-diameter", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--hole2-diameter", type=float, nargs=2, metavar=("MIN", "MAX"))

    args = parser.parse_args()

    # Build ranges with any overrides
    ranges = LBracketRanges()
    if args.leg1_length:
        ranges.leg1_length = tuple(args.leg1_length)
    if args.leg2_length:
        ranges.leg2_length = tuple(args.leg2_length)
    if args.width:
        ranges.width = tuple(args.width)
    if args.thickness:
        ranges.thickness = tuple(args.thickness)
    if args.hole1_diameter:
        ranges.hole1_diameter = tuple(args.hole1_diameter)
    if args.hole2_diameter:
        ranges.hole2_diameter = tuple(args.hole2_diameter)

    print(f"Generating {args.count} L-brackets...")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print(f"Parameter ranges: {ranges}")
    print()

    rng = np.random.default_rng(args.seed)

    try:
        metadata = generate_dataset(args.output, args.count, rng, ranges)
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)

    metadata_path = args.output / "metadata.csv"
    save_metadata(metadata, metadata_path)
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"STEP files saved to: {args.output / 'step'}")
    print("Done!")


if __name__ == "__main__":
    main()
