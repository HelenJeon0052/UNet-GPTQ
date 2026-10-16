"""
Single-coil centered Fourier operator; all MRI images are complex tensors.
DiffuseRecon uses four *real* model channels [Re(z0), Im(z0), Re(z1), Im(z1)].
"""



from __future__ import annotations
import torch

from unet_gptq.datas.fastmri_dataset import fft2_unshifted, ifft_unshifted

def img_to_channels(img: torch.Tensor) -> torch.Tensor:
    """
    [B,2,H,W] complex -> [B,4,H,W] real.
    """
    if img.ndim != 4 or img.shape[1] != 2 or not torch.is_complex(img):
        raise ValueError(f"Expected complex image [B,2,H,W]")
    
    return torch.stack((img.real, img.imag), dim = 2).flatten(1, 2)

def channels_to_img(channels: torch.Tensor) -> torch.Tensor:
    """
    [B,4,H,W] real -> [B,2,H,W] complex.
    """
    if channels.ndim != 4 or channels.shape[1] != 4 or not torch.is_floating_point(channels):
        raise ValueError(f"Expected real value channels [B,2,H,W]")
    
    return torch.complex(channels[:, 0::2], channels[:, 1::2])

def _pair_mask(mask:torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device = like.device, dtype = like.real.dtype)
    if mask.shape == like.shape or mask.shape == (like.shape[0], 1, *like.shape[-2:]):
        return mask
    if mask.shape == like.shape[-2:]:
        return mask.view(1, 1, *mask.shape)
    if mask.ndim == 3 and mask.shape[-2:] == like.shape[-2:]:
        if mask.shape[0] == like.shape[1] and mask.shape[0] != like.shape[0]:
            return mask.unsqueeze(0)
        return amsk.unsqueeze(1)
    raise ValueError(
        f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(like.shape)}"
    )

def _mixture_duo(channels: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if channels.ndim != 4 or channels.shape[1] != 4 or not torch.is_floating_point(channels):
        raise ValueError(f"Expected real value channels [B,2,H,W]")
    
    measured = fft2_unshifted(channels_to_img(channels))
    weight = _pair_mask(mask, measured)
    mixed = weight * observed + (1.0 - weight) * measured
    to_r = img_to_channels(ifft_unshifted(mixed))
    
    return to_r

def cartesian_mask(height: int, width: int, acceleration: int, center_fraction: float, seed: int) -> torch.Tensor:
    """
    Retrospective 1D column mask in centered k-space coordinates
    Exactly ceil (width / acceleration) columns are sampled.
    A fixed central region is retained.
    """


    import math

    if height <= 0 or width <= 0 or acceleration < 1:
        raise ValueError(f"Invalid image size or acceleration")
    if not (0 < center_fraction <= 1):
        raise ValueError(f"center fraction must be in (0, 1])")
    
    n_total = min(width, math.ceil(width / acceleration))
    n_center = max(1, round(width * center_fraction))

    if n_center > n_total:
        raise ValueError(f"Center fraction exceeds sampling budget")
    
    columns = torch.zeros(width, dtype = torch.bool)
    start = (width - n_center) // 2
    columns[start : start + n_center] = True
    candidates = torch.arange(width)[~columns]
    gen = torch.Generator().manual_seed(seed)
    selected = candidates[torch.randperm(len(candidates), generator = gen)[:n_total - n_center]]

    columns[selected] = True

    return columns.unsqueeze(0).expand(height, -1).clone()

    

