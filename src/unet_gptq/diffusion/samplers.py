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
from unet_gptq.diffusion.aggregation import overlapping_pair_components

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


@torch.inference_mode()
def reconstruct_adjacent_aggregate(
    model, coarse_diffusion, refine_diffusion, four_observed_slices, mask, *,
    num_samples=1, refine_steps=20, clip_denoised=True, dc_weight=1.0,
):
    """FP comparison: aggregate three overlapping pairs then refine the center.

    ``four_observed_slices`` is complex [B,4,H,W] in order i-1,i,i+1,i+2.
    The central pair runs first so resetting the same RNG state before the
    single-pair run produces the same central coarse trajectory. This follows
    the upstream overlap idea but runs samples sequentially, not in the
    upstream batch layout; it is not claimed to be bitwise upstream parity.
    """
    if (not torch.is_tensor(four_observed_slices) or four_observed_slices.ndim != 4
            or four_observed_slices.shape[1] != 4 or not torch.is_complex(four_observed_slices)):
        raise ValueError("Expected four complex observed slices [B,4,H,W]")
    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if type(refine_steps) is not int or not 0 <= refine_steps <= refine_diffusion.num_timesteps:
        raise ValueError("refine_steps must be within the refinement schedule")
    if coarse_diffusion.num_timesteps < 1:
        raise ValueError("The coarse schedule must contain at least one step")
    dc_weight = validate_dc_weight(dc_weight)
    observed = four_observed_slices.to(torch.complex64)
    mask = prepare_mask(mask, observed[:, 1:3])
    pair_means = [None, None, None]
    for pair_index in (1, 0, 2):
        pair = observed[:, pair_index:pair_index + 2]
        total = None
        for sample_index in range(num_samples):
            sample = _coarse_pass(model, coarse_diffusion, pair, mask,
                                  clip_denoised=clip_denoised, dc_weight=dc_weight,
                                  diagnostics=None, sample_index=sample_index)
            total = sample if total is None else total + sample
        pair_means[pair_index] = total / num_samples
    aggregate = torch.cat(overlapping_pair_components(pair_means), dim=1)
    refined = _refine_pass(model, refine_diffusion, aggregate, observed[:, 1:3], mask,
                           refine_steps, clip_denoised=clip_denoised,
                           dc_weight=dc_weight, diagnostics=None)
    return channels_to_img(aggregate), channels_to_img(refined), channels_to_img(pair_means[1])

