"""
Parameter normalization for multi-geometry dataset.

Each geometry type has different parameter ranges. This module provides
per-type min-max normalization to [0, 1] range for uniform treatment
in the LLM regression heads.

Normalization scheme: normalized = (value - min) / (max - min)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from graph_cad.data.brep_types import (
    GEOMETRY_BRACKET,
    GEOMETRY_TUBE,
    GEOMETRY_CHANNEL,
    GEOMETRY_BLOCK,
    GEOMETRY_CYLINDER,
    GEOMETRY_BLOCKHOLE,
    MAX_PARAMS,
)


# =============================================================================
# Parameter Ranges (min, max) in mm
# =============================================================================

# Bracket: leg1, leg2, width, thickness
BRACKET_RANGES = torch.tensor([
    [50.0, 200.0],   # leg1_length
    [50.0, 200.0],   # leg2_length
    [20.0, 60.0],    # width
    [3.0, 12.0],     # thickness
], dtype=torch.float32)

# Tube: length, outer_dia, inner_dia
TUBE_RANGES = torch.tensor([
    [30.0, 200.0],   # length
    [20.0, 100.0],   # outer_dia
    [6.0, 90.0],     # inner_dia (derived from ratio * outer)
], dtype=torch.float32)

# Channel: width, height, length, thickness
CHANNEL_RANGES = torch.tensor([
    [30.0, 100.0],   # width
    [30.0, 100.0],   # height
    [50.0, 200.0],   # length
    [2.0, 10.0],     # thickness
], dtype=torch.float32)

# Block: length, width, height
BLOCK_RANGES = torch.tensor([
    [20.0, 150.0],   # length
    [20.0, 150.0],   # width
    [10.0, 100.0],   # height
], dtype=torch.float32)

# Cylinder: length, diameter
CYLINDER_RANGES = torch.tensor([
    [20.0, 200.0],   # length
    [10.0, 100.0],   # diameter
], dtype=torch.float32)

# BlockHole: length, width, height, hole_dia, hole_x, hole_y
BLOCKHOLE_RANGES = torch.tensor([
    [30.0, 150.0],   # length
    [30.0, 150.0],   # width
    [15.0, 80.0],    # height
    [5.0, 30.0],     # hole_dia
    [-60.0, 60.0],   # hole_x (relative to center)
    [-60.0, 60.0],   # hole_y (relative to center)
], dtype=torch.float32)


# Map geometry type ID to ranges tensor
PARAM_RANGES = {
    GEOMETRY_BRACKET: BRACKET_RANGES,
    GEOMETRY_TUBE: TUBE_RANGES,
    GEOMETRY_CHANNEL: CHANNEL_RANGES,
    GEOMETRY_BLOCK: BLOCK_RANGES,
    GEOMETRY_CYLINDER: CYLINDER_RANGES,
    GEOMETRY_BLOCKHOLE: BLOCKHOLE_RANGES,
}


# Parameter names per geometry type
PARAM_NAMES = {
    GEOMETRY_BRACKET: ["leg1_length", "leg2_length", "width", "thickness"],
    GEOMETRY_TUBE: ["length", "outer_dia", "inner_dia"],
    GEOMETRY_CHANNEL: ["width", "height", "length", "thickness"],
    GEOMETRY_BLOCK: ["length", "width", "height"],
    GEOMETRY_CYLINDER: ["length", "diameter"],
    GEOMETRY_BLOCKHOLE: ["length", "width", "height", "hole_dia", "hole_x", "hole_y"],
}


def get_param_count(geometry_type: int) -> int:
    """Get number of parameters for a geometry type."""
    return len(PARAM_NAMES[geometry_type])


def normalize_params(
    params: Tensor,
    geometry_type: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Normalize parameters to [0, 1] range for a specific geometry type.

    Args:
        params: Raw parameters in mm, shape (num_params,) or (batch, num_params)
        geometry_type: Geometry type ID (0-5)
        device: Optional device to move ranges tensor to

    Returns:
        Normalized parameters in [0, 1] range
    """
    ranges = PARAM_RANGES[geometry_type]
    if device is not None:
        ranges = ranges.to(device)
    elif params.device != ranges.device:
        ranges = ranges.to(params.device)

    num_params = params.shape[-1]
    mins = ranges[:num_params, 0]
    maxs = ranges[:num_params, 1]

    return (params - mins) / (maxs - mins + 1e-8)


def denormalize_params(
    params_norm: Tensor,
    geometry_type: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Denormalize parameters from [0, 1] range to mm.

    Args:
        params_norm: Normalized parameters in [0, 1], shape (num_params,) or (batch, num_params)
        geometry_type: Geometry type ID (0-5)
        device: Optional device to move ranges tensor to

    Returns:
        Parameters in mm
    """
    ranges = PARAM_RANGES[geometry_type]
    if device is not None:
        ranges = ranges.to(device)
    elif params_norm.device != ranges.device:
        ranges = ranges.to(params_norm.device)

    num_params = params_norm.shape[-1]
    mins = ranges[:num_params, 0]
    maxs = ranges[:num_params, 1]

    return params_norm * (maxs - mins) + mins


def normalize_params_to_latent(
    params: Tensor,
    geometry_type: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Normalize parameters to latent-compatible range [-2, 2].

    For direct latent supervision, we want parameters in a range similar
    to the VAE latent space. This maps [0, 1] normalized params to [-2, 2].

    Args:
        params: Raw parameters in mm
        geometry_type: Geometry type ID
        device: Optional device

    Returns:
        Parameters scaled to [-2, 2] range
    """
    params_norm = normalize_params(params, geometry_type, device)
    return params_norm * 4.0 - 2.0


def denormalize_params_from_latent(
    params_latent: Tensor,
    geometry_type: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Denormalize parameters from [-2, 2] latent range to mm.

    Args:
        params_latent: Parameters in [-2, 2] range
        geometry_type: Geometry type ID
        device: Optional device

    Returns:
        Parameters in mm
    """
    params_norm = (params_latent + 2.0) / 4.0
    return denormalize_params(params_norm, geometry_type, device)


def pad_params(
    params: Tensor,
    geometry_type: int,
    max_params: int = MAX_PARAMS,
) -> tuple[Tensor, Tensor]:
    """
    Pad parameters to fixed size with mask.

    Args:
        params: Parameters for this geometry type, shape (num_params,) or (batch, num_params)
        geometry_type: Geometry type ID
        max_params: Maximum number of parameters (default 6 for BlockHole)

    Returns:
        Tuple of:
            - padded_params: Shape (..., max_params), padded with zeros
            - param_mask: Shape (..., max_params), 1 for real params, 0 for padding
    """
    is_batched = params.dim() == 2
    if not is_batched:
        params = params.unsqueeze(0)

    batch_size = params.shape[0]
    num_params = params.shape[1]
    device = params.device
    dtype = params.dtype

    padded = torch.zeros(batch_size, max_params, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_params, device=device, dtype=dtype)

    padded[:, :num_params] = params
    mask[:, :num_params] = 1.0

    if not is_batched:
        padded = padded.squeeze(0)
        mask = mask.squeeze(0)

    return padded, mask


def unpad_params(
    padded_params: Tensor,
    geometry_type: int,
) -> Tensor:
    """
    Remove padding from parameters.

    Args:
        padded_params: Padded parameters, shape (..., max_params)
        geometry_type: Geometry type ID

    Returns:
        Parameters with padding removed, shape (..., num_params)
    """
    num_params = get_param_count(geometry_type)
    return padded_params[..., :num_params]


class MultiGeometryNormalizer:
    """
    Batch normalizer that handles mixed geometry types.

    This class efficiently normalizes parameters when batches contain
    multiple geometry types.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        """
        Initialize normalizer.

        Args:
            device: Device to store range tensors on
        """
        self.device = device

        # Pre-compute padded ranges for efficient batched operations
        self.all_mins = torch.zeros(6, MAX_PARAMS, device=device)
        self.all_ranges = torch.ones(6, MAX_PARAMS, device=device)

        for geo_type, ranges in PARAM_RANGES.items():
            n = ranges.shape[0]
            self.all_mins[geo_type, :n] = ranges[:, 0].to(device)
            self.all_ranges[geo_type, :n] = (ranges[:, 1] - ranges[:, 0]).to(device)

    def normalize_batch(
        self,
        params: Tensor,
        geometry_types: Tensor,
    ) -> Tensor:
        """
        Normalize a batch of parameters with mixed geometry types.

        Args:
            params: Padded parameters, shape (batch, max_params)
            geometry_types: Geometry type IDs, shape (batch,)

        Returns:
            Normalized parameters in [0, 1] range
        """
        mins = self.all_mins[geometry_types]  # (batch, max_params)
        ranges = self.all_ranges[geometry_types]  # (batch, max_params)

        return (params - mins) / (ranges + 1e-8)

    def denormalize_batch(
        self,
        params_norm: Tensor,
        geometry_types: Tensor,
    ) -> Tensor:
        """
        Denormalize a batch of parameters.

        Args:
            params_norm: Normalized parameters, shape (batch, max_params)
            geometry_types: Geometry type IDs, shape (batch,)

        Returns:
            Parameters in mm
        """
        mins = self.all_mins[geometry_types]
        ranges = self.all_ranges[geometry_types]

        return params_norm * ranges + mins

    def to(self, device: torch.device | str) -> MultiGeometryNormalizer:
        """Move normalizer to device."""
        self.device = device
        self.all_mins = self.all_mins.to(device)
        self.all_ranges = self.all_ranges.to(device)
        return self
