"""
Single-pair DiffuseRecon KSG + coarse-to-fine sampling.
Coarse-to-fine sampling with shared unshifted, complex64 MRI operations.
"""
from __future__ import annotations

import torch

from contextlib import nullcontext
from typing import TYPE_CHECKING

from unet_gptq.domain.data_consistency import prepare_mask, project_observations, validate_dc_weight
from unet_gptq.domain.forward_operator import channels_to_img, img_to_channels
from unet_gptq.datas.fastmri_dataset import fft2_unshifted

if TYPE_CHECKING:
    from .diagnostics import SamplingDiagnostics

def _sample_step(model, diffusion, x, t, diagnostics, *, stage, sample_index,
                 step_index, **kwargs):
    scope = (
        diagnostics.observe(model, stage=stage, sample_index=sample_index,
                            step_index=step_index)
        if diagnostics is not None else nullcontext()
    )
    with scope:
        return diffusion.p_sample(model, x, t, **kwargs)

def _coarse_pass(model, diffusion, observed, mask, *, clip_denoised, dc_weight, diagnostics, sample_index):
    
    x = torch.randn((observed.shape[0], 4, *observed.shape[-2:]), device = observed.device, dtype=torch.float32)

    for step_index, index in enumerate(range(diffusion.num_timesteps - 1, -1, -1)):
        t = torch.full((x.shape[0],), index, device = x.device, dtype = torch.long)
        out = _sample_step(
            model, diffusion, x, t, diagnostics, stage="coarse",
            sample_index=sample_index, step_index=step_index,
            clip_denoised=clip_denoised,
        )

        if index > 0:
            alpha_bar = float(diffusion.alphas_cumprod[index - 1])
            observed_t = (
                alpha_bar**0.5 * observed
                + (1 - alpha_bar)**0.5 * fft2_unshifted(channels_to_img(out["model_output"]))
            )
        else:
            observed_t = observed
        corrected = project_observations(
            channels_to_img(out["sample"]), observed_t, mask, dc_weight=dc_weight
        )
        if diagnostics is not None:
            diagnostics.after_dc(
                corrected, observed_t, observed, mask, stage="coarse",
                sample_index=sample_index, step_index=step_index,
                timestep=index, dc_weight=dc_weight,
            )
        x = img_to_channels(corrected)

    return x

def _refine_pass(model, diffusion, t_0, observed, mask, steps, *, clip_denoised, dc_weight, diagnostics):

    """if not 1 <= steps <= diffusion.num_timestpes:
        raise ValueError(f"refine steps must be in the refinement schedule")"""
    
    x = t_0

    for step_index, index in enumerate(range(steps - 1, -1, -1)):
        t = torch.full((x.shape[0],), index, device = x.device, dtype = torch.long)
        out = _sample_step(model, diffusion, x, t, diagnostics, stage="refined", sample_index=0, step_index=step_index, no_noise=True, clip_denoised=clip_denoised)
        
        corrected = project_observations(channels_to_img(out["sample"]), observed, mask, dc_weight=dc_weight)
        if diagnostics is not None:
            diagnostics.after_dc(
                corrected, observed, observed, mask, stage="refined",
                sample_index=0, step_index=step_index,
                timestep=index, dc_weight=dc_weight,
            )
        
        x = img_to_channels(corrected)

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
    dc_weight: float = 0.5,
    diagnostics: SamplingDiagnostics | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Return complex [B,2,H,W] reconstruction for sigle-coil slice pairs

    Observations/masks use unshifted FFT coordinates with norm='backward'.
    dc_weight is shared by both stages: 0.5 is the initial soft-DC baseline,
    1 is hard projection, and 0 disables correction. Detailed diagnostics are
    optional and must be collected outside performance measurements.
    """
    if (observed_kspace.ndim != 4 or observed_kspace.shape[1] != 2 or not torch.is_complex(observed_kspace)):
        raise ValueError("Expected complex observed k-space [B,2,H,W]")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if (not isinstance(refine_steps, int) or isinstance(refine_steps, bool) or not 0 <= refine_steps <= refine_diffusion.num_timesteps):
        raise ValueError("refine_steps must be in the refinement schedule")
    if coarse_diffusion.num_timesteps < 1:
        raise ValueError("The coarse schedule must contain at least one step")
    dc_weight = validate_dc_weight(dc_weight)
    observed = observed_kspace.to(torch.complex64)
    
    mask = prepare_mask(mask, observed)
    coarse_sum = None

    for sample_index in range(num_samples):
        sample = _coarse_pass(model, coarse_diffusion, observed, mask, clip_denoised = clip_denoised, dc_weight=dc_weight, diagnostics=diagnostics, sample_index=sample_index)
        coarse_sum = sample if coarse_sum is None else coarse_sum + sample
    

    mean = coarse_sum / num_samples

    # t_0 == mean
    refined = _refine_pass(
        model, refine_diffusion, mean, observed_kspace, mask, refine_steps, clip_denoised = clip_denoised, dc_weight=dc_weight, diagnostics=diagnostics
    )

    result = channels_to_img(refined)


    return channels_to_img(mean), result

