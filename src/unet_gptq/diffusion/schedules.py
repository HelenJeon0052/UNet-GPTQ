"""
Create the re-spaced DDPM schedules for coarse and fine passes
from DiffuseRecon

"""
from __future__ import annotations


def make_schedules(
    *, train_steps: int = 2000, coarse_steps: int = 200, refine_schedule_steps: int = 500
):
    try:
        from improved_diffusion.script_util_duo import create_gaussian_diffusion
    except ImportError as e:
        raise ImportError(f"Install the official DiffuseRecon fork first")
    
    if not (0 < coarse_steps <= train_steps and 0 < refine_schedule_steps <= train_steps):
        raise ValueError(f"Wrong re-spaced schedule length")
    
    common = dict(
        steps = train_steps,
        learn_sigma = False,
        sigma_small = False,
        noise_schedule = "cosine",
        use_kl = False,
        predict_xstart = False,
        rescale_timesteps = True,
        rescale_learned_sigmas = True,
    )

    return (
        create_gaussian_diffusion(**common, timestep_respacing=str(coarse_steps)),
        create_gaussian_diffusion(**common, timestep_respacing=str(refine_schedule_steps)),
    )