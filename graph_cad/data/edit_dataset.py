"""
Dataset for latent editing task.

Contains instruction templates and dataset class for training
the LLM-based latent editor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from numpy.random import Generator


# Instruction templates for each parameter type
# Each template is a format string that takes specific keyword arguments
INSTRUCTION_TEMPLATES = {
    "leg1_length": [
        "make leg1 {delta:+.0f}mm {direction}",
        "change leg1 length by {delta:+.0f}mm",
        "{direction} the horizontal leg by {abs_delta:.0f}mm",
        "set leg1 to {new_value:.0f}mm",
        "adjust the first leg length to {new_value:.0f}mm",
        "leg1_length = {new_value:.0f}",
    ],
    "leg2_length": [
        "make leg2 {delta:+.0f}mm {direction}",
        "change leg2 length by {delta:+.0f}mm",
        "{direction} the vertical leg by {abs_delta:.0f}mm",
        "set leg2 to {new_value:.0f}mm",
        "adjust the second leg length to {new_value:.0f}mm",
        "leg2_length = {new_value:.0f}",
    ],
    "width": [
        "make it {delta:+.0f}mm {direction}",
        "change width by {delta:+.0f}mm",
        "{direction} the bracket width by {abs_delta:.0f}mm",
        "set width to {new_value:.0f}mm",
        "adjust the Y-extent to {new_value:.0f}mm",
        "width = {new_value:.0f}",
    ],
    "thickness": [
        "make it {delta:+.1f}mm {direction}",
        "change thickness by {delta:+.1f}mm",
        "{direction} the material thickness by {abs_delta:.1f}mm",
        "set thickness to {new_value:.1f}mm",
        "thickness = {new_value:.1f}",
    ],
    "hole1_diameter": [
        "make hole1 {delta:+.1f}mm {direction}",
        "change hole1 diameter by {delta:+.1f}mm",
        "{direction} the first hole by {abs_delta:.1f}mm",
        "set hole1 diameter to {new_value:.1f}mm",
        "hole1_diameter = {new_value:.1f}",
    ],
    "hole2_diameter": [
        "make hole2 {delta:+.1f}mm {direction}",
        "change hole2 diameter by {delta:+.1f}mm",
        "{direction} the second hole by {abs_delta:.1f}mm",
        "set hole2 diameter to {new_value:.1f}mm",
        "hole2_diameter = {new_value:.1f}",
    ],
    "hole1_distance": [
        "move hole1 {delta:+.0f}mm {dist_direction}",
        "shift hole1 position by {delta:+.0f}mm",
        "move hole1 {abs_delta:.0f}mm {dist_direction}",
        "set hole1 distance to {new_value:.0f}mm",
        "hole1_distance = {new_value:.0f}",
    ],
    "hole2_distance": [
        "move hole2 {delta:+.0f}mm {dist_direction}",
        "shift hole2 position by {delta:+.0f}mm",
        "move hole2 {abs_delta:.0f}mm {dist_direction}",
        "set hole2 distance to {new_value:.0f}mm",
        "hole2_distance = {new_value:.0f}",
    ],
}

# Compound edit templates (multiple parameters at once)
COMPOUND_TEMPLATES = [
    ("make it bigger", {"leg1_length": 20, "leg2_length": 20, "width": 5}),
    ("make it smaller", {"leg1_length": -20, "leg2_length": -20, "width": -5}),
    ("make it longer", {"leg1_length": 30, "leg2_length": 30}),
    ("make it shorter", {"leg1_length": -30, "leg2_length": -30}),
    ("make the legs longer", {"leg1_length": 25, "leg2_length": 25}),
    ("increase the size", {"leg1_length": 15, "leg2_length": 15, "width": 5}),
    ("make it thicker", {"thickness": 2}),
    ("make it thinner", {"thickness": -2}),
    ("make the holes bigger", {"hole1_diameter": 2, "hole2_diameter": 2}),
    ("make the holes smaller", {"hole1_diameter": -2, "hole2_diameter": -2}),
    ("widen the bracket", {"width": 10}),
    ("narrow the bracket", {"width": -10}),
    ("keep it the same", {}),  # No-op for identity learning
    ("no changes", {}),
    ("leave it unchanged", {}),
]


def generate_instruction(
    param: str,
    delta: float,
    old_value: float,
    rng: Generator,
) -> str:
    """
    Generate a natural language instruction for a parameter edit.

    Args:
        param: Parameter name being edited.
        delta: Change amount (positive or negative).
        old_value: Original value before edit.
        rng: Random number generator.

    Returns:
        Natural language instruction string.
    """
    templates = INSTRUCTION_TEMPLATES.get(param, [])
    if not templates:
        return f"change {param} by {delta:+.1f}"

    template = rng.choice(templates)

    # Prepare format arguments
    new_value = old_value + delta
    abs_delta = abs(delta)

    # Direction words for size changes
    if delta > 0:
        direction = "longer" if "length" in param else "wider" if "width" in param else "thicker" if "thickness" in param else "bigger"
    else:
        direction = "shorter" if "length" in param else "narrower" if "width" in param else "thinner" if "thickness" in param else "smaller"

    # Direction words for distance (hole position)
    dist_direction = "toward the corner" if delta < 0 else "away from the corner"

    try:
        return template.format(
            delta=delta,
            abs_delta=abs_delta,
            new_value=new_value,
            direction=direction,
            dist_direction=dist_direction,
        )
    except KeyError:
        # Fallback if template has unknown keys
        return f"change {param} by {delta:+.1f}mm"


@dataclass
class EditSample:
    """A single edit training sample."""

    instruction: str
    z_src: torch.Tensor  # Source latent (16D)
    z_tgt: torch.Tensor  # Target latent (16D)
    delta_z: torch.Tensor  # z_tgt - z_src
    param_deltas: dict[str, float]  # Parameter changes


class LatentEditDataset(Dataset):
    """
    Dataset for latent editing training.

    Loads pre-generated edit pairs from disk or holds them in memory.
    Each sample contains: instruction, source latent, target latent, delta.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        samples: list[dict] | None = None,
    ):
        """
        Initialize dataset from file or list.

        Args:
            data_path: Path to JSON file with edit samples.
            samples: List of sample dictionaries (alternative to file).
        """
        self.samples = []

        if data_path is not None:
            self._load_from_file(data_path)
        elif samples is not None:
            self.samples = samples
        else:
            raise ValueError("Must provide either data_path or samples")

    def _load_from_file(self, path: str | Path) -> None:
        """Load samples from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        for item in data:
            sample = {
                "instruction": item["instruction"],
                "z_src": torch.tensor(item["z_src"], dtype=torch.float32),
                "z_tgt": torch.tensor(item["z_tgt"], dtype=torch.float32),
                "delta_z": torch.tensor(item["delta_z"], dtype=torch.float32),
                "param_deltas": item.get("param_deltas", {}),
            }
            # Load direction label if present, otherwise derive from param_deltas
            if "direction" in item:
                sample["direction"] = torch.tensor(item["direction"], dtype=torch.float32)
            elif item.get("param_deltas"):
                # Derive direction from first (usually only) param delta
                delta_val = list(item["param_deltas"].values())[0]
                sample["direction"] = torch.tensor(1.0 if delta_val > 0 else 0.0, dtype=torch.float32)
            else:
                # Default to increase for noop/unknown
                sample["direction"] = torch.tensor(0.5, dtype=torch.float32)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample.

        Returns:
            Dictionary with keys: instruction, z_src, z_tgt, delta_z, direction
        """
        sample = self.samples[idx]

        # Handle both dict and tensor formats
        if isinstance(sample.get("z_src"), torch.Tensor):
            return sample

        result = {
            "instruction": sample["instruction"],
            "z_src": torch.tensor(sample["z_src"], dtype=torch.float32),
            "z_tgt": torch.tensor(sample["z_tgt"], dtype=torch.float32),
            "delta_z": torch.tensor(sample["delta_z"], dtype=torch.float32),
            "param_deltas": sample.get("param_deltas", {}),
        }
        # Add direction if present
        if "direction" in sample:
            result["direction"] = torch.tensor(sample["direction"], dtype=torch.float32)
        return result

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for sample in self.samples:
            item = {
                "instruction": sample["instruction"],
                "z_src": sample["z_src"].tolist() if isinstance(sample["z_src"], torch.Tensor) else sample["z_src"],
                "z_tgt": sample["z_tgt"].tolist() if isinstance(sample["z_tgt"], torch.Tensor) else sample["z_tgt"],
                "delta_z": sample["delta_z"].tolist() if isinstance(sample["delta_z"], torch.Tensor) else sample["delta_z"],
                "param_deltas": sample.get("param_deltas", {}),
            }
            data.append(item)

        with open(path, "w") as f:
            json.dump(data, f)


def collate_edit_batch(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with stacked tensors and list of instructions.
    """
    result = {
        "instructions": [sample["instruction"] for sample in batch],
        "z_src": torch.stack([sample["z_src"] for sample in batch]),
        "z_tgt": torch.stack([sample["z_tgt"] for sample in batch]),
        "delta_z": torch.stack([sample["delta_z"] for sample in batch]),
    }
    # Add direction if present in samples
    if "direction" in batch[0]:
        result["direction"] = torch.stack([sample["direction"] for sample in batch])
    return result


class PairedLatentEditDataset(Dataset):
    """
    Dataset for contrastive latent editing training.

    Each sample contains paired increase/decrease edits for the same
    source bracket and parameter. This enables contrastive learning
    where the model must produce opposite deltas for opposite instructions.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        samples: list[dict] | None = None,
    ):
        """
        Initialize dataset from file or list.

        Args:
            data_path: Path to JSON file with paired edit samples.
            samples: List of sample dictionaries (alternative to file).
        """
        self.samples = []

        if data_path is not None:
            self._load_from_file(data_path)
        elif samples is not None:
            self.samples = samples
        else:
            raise ValueError("Must provide either data_path or samples")

    def _load_from_file(self, path: str | Path) -> None:
        """Load samples from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        for item in data:
            self.samples.append({
                "z_src": torch.tensor(item["z_src"], dtype=torch.float32),
                "param": item["param"],
                # Increase direction
                "instruction_inc": item["instruction_inc"],
                "z_tgt_inc": torch.tensor(item["z_tgt_inc"], dtype=torch.float32),
                "delta_z_inc": torch.tensor(item["delta_z_inc"], dtype=torch.float32),
                "delta_inc": item["delta_inc"],
                # Decrease direction
                "instruction_dec": item["instruction_dec"],
                "z_tgt_dec": torch.tensor(item["z_tgt_dec"], dtype=torch.float32),
                "delta_z_dec": torch.tensor(item["delta_z_dec"], dtype=torch.float32),
                "delta_dec": item["delta_dec"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a paired sample.

        Returns:
            Dictionary with keys for both increase and decrease edits.
        """
        sample = self.samples[idx]

        # Handle both dict and tensor formats
        if isinstance(sample.get("z_src"), torch.Tensor):
            return sample

        return {
            "z_src": torch.tensor(sample["z_src"], dtype=torch.float32),
            "param": sample["param"],
            # Increase
            "instruction_inc": sample["instruction_inc"],
            "z_tgt_inc": torch.tensor(sample["z_tgt_inc"], dtype=torch.float32),
            "delta_z_inc": torch.tensor(sample["delta_z_inc"], dtype=torch.float32),
            "delta_inc": sample["delta_inc"],
            # Decrease
            "instruction_dec": sample["instruction_dec"],
            "z_tgt_dec": torch.tensor(sample["z_tgt_dec"], dtype=torch.float32),
            "delta_z_dec": torch.tensor(sample["delta_z_dec"], dtype=torch.float32),
            "delta_dec": sample["delta_dec"],
        }


def collate_paired_edit_batch(batch: list[dict]) -> dict:
    """
    Collate function for paired DataLoader.

    Args:
        batch: List of paired sample dictionaries.

    Returns:
        Batched dictionary with stacked tensors and lists of instructions.
    """
    return {
        "z_src": torch.stack([sample["z_src"] for sample in batch]),
        "params": [sample["param"] for sample in batch],
        # Increase direction
        "instructions_inc": [sample["instruction_inc"] for sample in batch],
        "z_tgt_inc": torch.stack([sample["z_tgt_inc"] for sample in batch]),
        "delta_z_inc": torch.stack([sample["delta_z_inc"] for sample in batch]),
        # Decrease direction
        "instructions_dec": [sample["instruction_dec"] for sample in batch],
        "z_tgt_dec": torch.stack([sample["z_tgt_dec"] for sample in batch]),
        "delta_z_dec": torch.stack([sample["delta_z_dec"] for sample in batch]),
    }
