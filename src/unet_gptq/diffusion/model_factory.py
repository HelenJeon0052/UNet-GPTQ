"""Shared factory configuration for calibration and paired reconstruction."""
from __future__ import annotations

import json


def resolve_model_config(factory_module, *, model_options: str, image_size: int,
                         timestep_respacing: str, use_fp16: bool) -> dict:
    config = dict(factory_module.model_and_diffusion_defaults())
    options = json.loads(model_options)
    if not isinstance(options, dict):
        raise ValueError("--model_options must be a JSON object.")
    unknown = options.keys() - config.keys()
    if unknown:
        raise ValueError(f"Unknown model options for this factory: {sorted(unknown)}")
    for reserved in ("image_size", "timestep_respacing", "use_fp16"):
        if reserved in options:
            raise ValueError(f"Set {reserved} via the CLI; do not duplicate it in --model_options.")
    config.update(options)
    config["image_size"] = image_size
    config["timestep_respacing"] = timestep_respacing
    if "use_fp16" in config:
        config["use_fp16"] = use_fp16
    return config
