"""
Single-pair DiffuseRecon KSG + coarse-to-fine sampling.
both slices in this pair are averaged over N stochastic samples.
"""
from __future__ import annotations

import torch
from pathlib import Path

def load_denoiser(
    ckpt: str | Path,
    *,
    device: torch.device,
    image_size: int = 320,
    num_channels: int = 128,
    num_res_blocks: int = 3,
    dropout: float = 0.3,
) -> torch.nn.Module:

    try:
        from improved_diffusion.script_util_duo import create_model, model_and_diffusion_defaults
    
    except ImportError as e:
        raise ImportError(
            "Install the official DiffuseRecon fork: git clone "
            "https://github.com/cpeng93/DiffuseRecon external/DiffuseRecon "
            "&& pip install -e external/DiffuseRecon"
        ) from e

    cfg = model_and_diffusion_defaults()
    cfg.update(
        image_size = image_size,
        num_channels = num_channels,
        num_res_blocks = num_res_blocks,
        dropout = dropout,
        learn_sigma = False,
        class_cond = False,
    )

    model = create_model(
        **{key: cfg[key] for key in(
            "image_size", "num_channels", "num_res_blocks", "learn_sigma",
            "class_cond", "use_checkpoint", "attention_resolutions", "num_heads",
            "num_heads_upsample", "use_scale_shift_norm", "dropout",
        )}
    )

    try:
        state = torch.load(ckpt, map_location = "cpu", weights_only = True)
    
    except TypeError as e:
        state = torch.load(ckpt, map_location = "cpu")

    
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint must be a state_dict of model param")

    model.load_state_dict(state, strict = True)
    return model.to(device).eval()
