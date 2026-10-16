"""
Explicit Normalization, mask convention and packing for DiffusionRecon duo 
an aim: to fit raw fastMRI coil combination is a seperate preprocessing option not taking ckpt / domain compatibility

"""
from __future__ import annotations
import numpy as np
import torch

from typing import Any, Literal

from unet_gptq.datas.fastmri_dataset import fft2_unshifted, image_to_4ch


def prep_diffusionrecon_pair(
    image: torch.Tensor, centered_mask: torch.Tensor, *, normalization: Literal["component_max", "none"] = "component_max", eps: float = 1e-8
) -> dict[str, Any]:
    """
    component_max divides each slice by max(|real|,|imag|), matching the upstream
    sampling loader. This uses the full reference and must be disclosed as oracle
    normalization in synthetic experiments. Real observations require a separate
    observation- or training-derived scale policy.

    Normalization precedes FFT/masking. Save scale to restore image/metric units.
    For an unbatched sample, unsqueeze returned packed tensors and mask for duo.
    centered_mask: a newly generated physical frequency mask for the
    loader's conventional image.

    Not to reinterpret the upstream mask asset: seperate checks are required to
    verify the preprocessed source and convention
    """
    if not torch.is_complex(image):
        raise TypeError("image must be native complex")
    if image.ndim < 3 or image.shape[-3] != 2 or any(d == 0 for d in image.shape):
        raise ValueError("image must have shape [...,2,H,W] with nonempty dimensions")
    if not torch.isfinite(image).all():
        raise ValueError("image contains NaN or Inf")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if normalization == "component_max":
        scale = torch.maximum(image.real.abs(), image.imag.abs()).amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(eps)
    
    elif normalization == "none":
        scale = torch.ones_like(image.real[..., :1, :1]) #  last two dimensions is kept, but preserves the broadcasting shape
    else:
        raise ValueError(f"unknown normalization={normalization!r}")
    

    if torch.is_complex(centered_mask):
        raise TypeError("mask must be real or bool")
    if centered_mask.ndim < 2 or centered_mask.shape[-2:] != image.shape[-2:]:
        raise ValueError("mask spatial dimensions must match image")
    if not torch.isfinite(centered_mask).all() or not ((centered_mask == 0) | (centered_mask == 1)).all():
        raise ValueError("mask must contain only binary 0/1 values")

    try:
        mask = torch.broadcast_to(centered_mask.to(device=image.device, dtype=torch.bool), image.shape)
    except RuntimeError as e:
        raise ValueError(f"mask must broadcast to image") from e
    
    mask = torch.fft.ifftshift(mask, dim=(-2, -1))
    normalized = image / scale
    ks_eq = fft2_unshifted(normalized)

    if not torch.isfinite(ks_eq).all():
        raise ValueError(f"ks_eq contain NaN or Inf {ks_eq}")
    k_obs = mask * ks_eq #  binary guided by a mask : 1 where data is kept, 0 where it’s skipped
    model_image = image_to_4ch(normalized)
    model_ks = image_to_4ch(k_obs)

    return {
        "image": normalized, "scale": scale, "ks_eq": ks_eq, "k_obs": k_obs, "sampling_mask": mask, "model_image": model_image, "model_kspace": model_ks, "normalization": normalization, "source_commit": SOURCE_COMMIT
    }