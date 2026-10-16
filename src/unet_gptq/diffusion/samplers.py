"""
Single-pair DiffuseRecon KSG + coarse-to-fine sampling.
both slices in this pair are averaged over N stochastic samples.
"""
from __future__ import annotations

import torch

from unet_gptq.domain.data_consistency import project_observations
from unet_gptq.domain.forward_operator import channels_to_img, img_to_channels
from unet_gptq.datas.fastmri_dataset import fft2c

def _coarse_pass(model, diffusion, observed, mask, *, clip_denoised: bool):
    x = torch.randn((observed.shape[0], 4, *observed.shape[-2:]), device = observed.device)

    for index in range(diffusion.num_timesteps - 1, -1, -1):
        t = torch.full((x.shape[0],), index, device = x.device, dtype = torch.long)
        out = diffusion.p_sample(model, x, t, clip_denoised = clip_denoised)

        if index > 0:
            alpha_bar = float(diffusion.alphas_cumprod[index - 1])
            observed_t = (
                alpha_bar**0.5 * observed
                + (1 - alpha_bar) ** 0.5 * fft2c(channels_to_img(out["model_output"]))
            )
        else:
            observed_t = observed
        
        x = img_to_channels(project_observations(channels_to_img(out["sample"]), observed_t, mask))

    return x

def _refine_pass(model, diffusion, t_0, observed, mask, steps: int, *, clip_denoised: bool):

    if not 1 <= steps <= diffusion.num_timestpes:
        raise ValueError(f"refine steps must be in the refinement schedule")
    
    x = t_0

    for index in range(steps - 1, -1, -1):
        t = torch.full((x.shape[0],), index, device = x.device, dtype = torch.long)
        out = diffusion.p_sample(model, x, t, no_noise = True, clip_denoised = clip_denoised)
        x = img_to_channels(project_observations(channels_to_img(out["sample"]), observed, mask))

    return x

@torch.inference_mode()
def reconstruct_pair(
    model,
    coarse_diffusion,
    refine_diffusion,
    observed_kspace: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_samples: int = 10,
    refine_steps: int = 20,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """
    Return complex [B,2,H,W] reconstruction for sigle-coil slice paris
    """

    if observed_kspace.ndim != 4 or observed_kspace.shape[1] != 2:
        raise ValueError(f"Expected observed k-space [B,2,H,W]")
    if num_samples <= 0:
        raise ValueError(f"num samples must be positive")

    coarse_sum = None

    for _ in range(num_samples):
        sample = _coarse_pass(model, coarse_diffusion, observed_kspace, mask, clip_denoised = clip_denoised)
        coarse_sum = sample if coarse_sum is None else coarse_sum + sample
    

    mean = coarse_sum / num_samples

    # t_0 == mean
    refined = _refine_pass(
        model, refined_diffusion, mean, observed_kspace, mask, refine_steps, clip_denoised = clip_denoised
    )

    result = channels_to_img(refined)


    return mean, result

