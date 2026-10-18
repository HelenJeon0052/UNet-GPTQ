"""
Single-coil centered Fourier operator; all MRI images are complex tensors.
DiffuseRecon uses four *real* model channels [Re(z0), Im(z0), Re(z1), Im(z1)].
"""



from __future__ import annotations
import torch

from unet_gptq.domain.data_consistency import prepare_mask, project_observations

def img_to_channels(img: torch.Tensor) -> torch.Tensor:
    """
    [B,2,H,W] complex -> [B,4,H,W] real.
    """
    if img.ndim != 4 or img.shape[1] != 2 or not torch.is_complex(img):
        raise ValueError(f"Expected complex image [B,2,H,W]")
    
    return torch.stack((img.real, img.imag), dim = 2).flatten(1, 2)

def channels_to_img(channels: torch.Tensor) -> torch.Tensor:
    """
    [B,4,H,W] real -> [B,2,H,W] complex64 for MRI/DC arithmetic
    """
    if channels.ndim != 4 or channels.shape[1] != 4 or not torch.is_floating_point(channels):
        raise ValueError("Expected real value channels [B,4,H,W]")
    
    channels = channels.to(torch.float32)
    
    return torch.complex(channels[:, 0::2], channels[:, 1::2])

def _mixture_duo(channels: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor, *, dc_weight: float = 0.5) -> torch.Tensor:
    """Four-real-channel interface to the shared soft/hard DC operation."""
    
    img = project_observations(channels_to_img(channels), observed, mask, dc_weight=dc_weight)
    if not torch.is_tensor(img):
        raise ValueError(f"img from first obs must be tensor, got {type(img)}")
    return img_to_channels(img)

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

    

