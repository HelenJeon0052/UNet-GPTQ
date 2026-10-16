"""
DiffuseRecon k-space guidance for a single-coil
two-slice image pair
"""
from __future__ import annotations

import torch

from unet_gptq.datas.fastmri_dataset import fft2c, ifft2c

def project_observations(
    image: torch.Tensor, observed_kspace: torch.Tensor, mask: torck.Tensor
) -> torch.Tensor:

    """
    image, observed_kspace: complex [B,2,H,W]; mask: bool [H,W].
    """

    if image.ndim != 4 or image.shape[1] != 2 or not torch.is_complex(image):
        raise ValueError(f"Expected complex image [B,2,H,W]")
    if observed_kspace.shape != image.shape or not torch.is_complex(observed_kspace):
        raise ValueError(f"Expected complex observed k-space [B,2,H,W]")
    if mask.dtype != torch.bool or mask.shape != image.shape[-2:]:
        raise ValueError(f"Expected boolean mask [H,W]")
    
    ks = torch.where(mask, observed_kspace, fft2c(image))

    return ifft2c(ks)

def observed_residual(
    image: torch.Tensor, observed_kspace: torch.Tensor, mask: torck.Tensor
) -> torch.Tensor:

    """
    Relative L2 error at sampled k-space points.
    """

    diff = (fft2c(image) - observed_kspace)[..., mask] # restricts the residual to only those sampled positions
    ref = observed_kspace[..., mask]

    return torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref).clamp_min(1e-12)

