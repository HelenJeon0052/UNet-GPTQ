"""
The observation mask describes acquisition support and is always binary. The
separate ``dc_weight`` blends predictions and observations on that support;
``dc_weight=1`` recovers hard projection. MRI arithmetic stays in complex64,
independently of the denoiser's precision.
"""
from __future__ import annotations

import torch

import math
from numbers import Real

from unet_gptq.datas.fastmri_dataset import fft2c, ifft2c, fft2_unshifted, ifft_unshifted

def validate_dc_weight(dc_weight: float) -> float:
    """Validate a fixed observation weight without any device synchronization."""

    if isinstance(dc_weight, bool) or not isinstance(dc_weight, Real):
        raise ValueError("dc_weight must be a finite real number in [0, 1]")
    weight = float(dc_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("dc_weight must be a finite real number in [0, 1]")
    return weight

def prepare_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Return a binary bool mask broadcastable to a pair tensor [B,2,H,W]."""
    if like.ndim != 4 or like.shape[1] != 2 or any(d == 0 for d in like.shape):
        raise ValueError("Expected a nonempty pair tensor [B,2,H,W]")
    if torch.is_complex(mask):
        raise ValueError("mask must contain only finite binary values 0 or 1")
    if mask.ndim == 2:
        shaped = mask[None, None]
    elif mask.ndim == 3 and mask.shape[-2:] == like.shape[-2:]:
        if mask.shape[0] in (1, like.shape[0]):
            shaped = mask[:, None]
        elif mask.shape[0] == 2:
            shaped = mask[None]
        else:
            shaped = mask
    else:
        shaped = mask
    if shaped.ndim != 4 or any(
        actual not in (1, expected)
        for actual, expected in zip(shaped.shape, like.shape)
    ):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(like.shape)}"
        )
    if mask.dtype != torch.bool and not bool(
        (torch.isfinite(mask) & ((mask == 0) | (mask == 1))).all()
    ):
        raise ValueError("mask must contain only finite binary values 0 or 1")

    return shaped.to(device=like.device, dtype=torch.bool)

def _prepare_pair(image: torch.Tensor, observed_kspace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 4 or image.shape[1] != 2 or not torch.is_complex(image):
        raise ValueError("Expected complex image [B,2,H,W]")
    if observed_kspace.shape != image.shape or not torch.is_complex(observed_kspace):
        raise ValueError("Expected complex observed k-space [B,2,H,W]")
    if observed_kspace.device != image.device:
        raise ValueError("image and observed k-space must be on the same device")
    return image.to(torch.complex64), observed_kspace.to(torch.complex64)


def project_observations(
    image: torch.Tensor, observed_kspace: torch.Tensor, mask: torch.Tensor, *, dc_weight: float = 0.5) -> torch.Tensor:

    """
    Blend observed coefficients, preserving every unobserved coefficient.
    """

    
    weight = validate_dc_weight(dc_weight)
    image, observed_kspace = _prepare_pair(image, observed_kspace)
    mask = prepare_mask(mask, image)
    if weight == 0.0:
        return image
    predicted = fft2_unshifted(image)
    corrected = observed_kspace if weight == 1.0 else (
        (1.0 - weight) * predicted + weight * observed_kspace
    )
    return ifft_unshifted(torch.where(mask, corrected, predicted))

def observed_residual(
    image: torch.Tensor, observed_kspace: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:

    """
    Relative L2 error at sampled k-space points
    as using the same FFT and dtype as DC.

    The residual need not be zero after soft DC. For noisy coarse sampling,
    supply that step's noisy observations when assessing its DC operation.
    """

    image, observed_kspace = _prepare_pair(image, observed_kspace)
    mask_p = prepare_mask(mask, image)

    diff = torch.where(mask_p, fft2_unshifted(image) - observed_kspace, 0.0)
    ref = torch.where(mask_p, observed_kspace, 0.0)

    return torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(1e-12)

