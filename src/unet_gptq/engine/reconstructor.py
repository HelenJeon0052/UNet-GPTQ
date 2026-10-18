"""
Single-pair DiffuseRecon KSG + coarse-to-fine sampling.
both slices in this pair are averaged over N stochastic samples.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from unet_gptq.diffusion.samplers import reconstruct_pair
from unet_gptq.domain.forward_operator import cartesian_mask
from unet_gptq.datas.fastmri_dataset import ifft2c, fft2_unshifted, ifft_unshifted



"""
reconstruction, target, observed_kspace: complex [1,2,H,W]
mask: bool [H,W]
"""
@dataclass
class ReconstructionResult:
    reconstruction: torch.Tensor
    target: torch.Tensor
    observed_kspace: torch.Tensor
    mask: torch.Tensor

def reconstruct_from_fully_sampled(
    model,
    coarse_diffusion,
    refine_diffusion,
    full_kspace: torch.Tensor,
    *,
    acceleration: int,
    center_frac: float,
    mask_seed: int = 0,
    sample_seed: int = 0,
    num_samples: int = 10,
    refine_steps: int = 20,
    dc_weight: float = 0.5,
    kspace_layout: str = "centered",
) -> ReconstructionResult:

    """Reconstruct a pair using the common unshifted, complex64 DC contract.
    The default input is centered/orthonormal k-space (as in fastMRI loaders).
    """

    if full_kspace.ndim != 3 or full_kspace.shape[0] !=2 or not torch.is_complex(full_kspace):
        raise ValueError(f"Expected two complex single-coil slices [2,H,W]")

    full = full_kspace.unsqueeze(0).to(torch.complex64)
    if kspace_layout == "centered":
        # Convert via image space: moving mask indices alone does not reconcile
        # the input spatial shift and FFT normalization of these conventions.
        full = fft2_unshifted(ifft2c(full))
    elif kspace_layout != "unshifted":
        raise ValueError("kspace_layout must be 'centered' or 'unshifted'")
    centered_mask = cartesian_mask(*full.shape[-2:], acceleration, center_frac, mask_seed)
    mask = torch.fft.ifftshift(centered_mask, dim=(-2, -1)).to(full.device)
    observed = torch.where(mask, full, torch.zeros_like(full))

    """
    normalized each slice from zero-filled observations
    from the withheld k-space coefficients or the full target image
    """

    zero_filled = ifft_unshifted(observed)

    scale = zero_filled.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    normalized_observed = observed / scale
    devices = [full.device.index or 0] if full.is_cuda else[]

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(sample_seed)
        if full.is_cuda:
            torch.cuda.manual_seed_all(sample_seed)
        _, normalized_recon = reconstruct_pair(
                    model, coarse_diffusion, refine_diffusion, normalized_observed, mask, num_samples = num_samples, refine_steps = refine_steps, dc_weight=dc_weight)

    return ReconstructionResult(normalized_recon * scale, ifft_unshifted(full), observed, mask)
