"""Shared, explicitly resolved experiment configuration; no ML imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _keys(value, required, optional, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    missing, unknown = required - value.keys(), value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"{where}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _path(value, base, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def read_shared_config(path):
    """Validate experiment controls and make config-relative paths absolute.

    Input files need not exist yet; each runner validates the inputs it uses.
    Explicit CLI options override mapped defaults, and actual arguments are
    saved in the run manifest so overrides cannot be mistaken for the config.
    """
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    cfg = json.loads(raw)
    _keys(cfg, {"checkpoint", "image_size", "model_options", "mask", "sampling", "execution", "data"}, set(), "config")
    _keys(cfg["mask"], {"path", "layout"}, set(), "mask")
    _keys(cfg["sampling"], {"coarse_steps", "refine_steps", "num_samples", "seed", "model_init_seed", "dc_weight", "clip_denoised"}, set(), "sampling")
    _keys(cfg["execution"], {"device", "precision"}, set(), "execution")
    _keys(cfg["data"], set(), {"h5", "data_root", "file1", "file2", "slice_index", "patient_id", "exam_id", "before", "after"}, "data")
    if not isinstance(cfg["model_options"], dict):
        raise ValueError("model_options must be an object")
    if {"image_size", "timestep_respacing", "use_fp16"} & cfg["model_options"].keys():
        raise ValueError("Set image_size, schedule and precision in their dedicated fields")
    if type(cfg["image_size"]) is not int or cfg["image_size"] < 1:
        raise ValueError("image_size must be a positive integer")
    sampling = cfg["sampling"]
    if type(sampling["clip_denoised"]) is not bool:
        raise ValueError("sampling.clip_denoised must be a boolean")
    for key in ("coarse_steps", "num_samples"):
        if type(sampling[key]) is not int or sampling[key] < 1:
            raise ValueError(f"sampling.{key} must be a positive integer")
    for key in ("refine_steps", "seed", "model_init_seed"):
        if type(sampling[key]) is not int or sampling[key] < 0:
            raise ValueError(f"sampling.{key} must be a nonnegative integer")
    if sampling["seed"] >= 2**32 or sampling["model_init_seed"] >= 2**32:
        raise ValueError("seeds must be below 2**32")
    dc = sampling["dc_weight"]
    if isinstance(dc, bool) or not isinstance(dc, (int, float)) or not math.isfinite(dc) or not 0 <= dc <= 1:
        raise ValueError("sampling.dc_weight must be finite and in [0,1]")
    if cfg["mask"]["layout"] not in ("centered", "unshifted"):
        raise ValueError("mask.layout must be centered or unshifted")
    if cfg["execution"]["precision"] not in ("fp16", "fp32"):
        raise ValueError("execution.precision must be fp16 or fp32")
    if not isinstance(cfg["execution"]["device"], str) or not cfg["execution"]["device"]:
        raise ValueError("execution.device must be nonempty")
    if "slice_index" in cfg["data"] and (type(cfg["data"]["slice_index"]) is not int or cfg["data"]["slice_index"] < 0):
        raise ValueError("data.slice_index must be a nonnegative integer")
    for name in ("patient_id", "exam_id", "file1", "file2", "before", "after"):
        if name in cfg["data"] and (not isinstance(cfg["data"][name], str) or not cfg["data"][name]):
            raise ValueError(f"data.{name} must be a nonempty string")
    cfg["checkpoint"] = _path(cfg["checkpoint"], path.parent, "checkpoint")
    cfg["mask"]["path"] = _path(cfg["mask"]["path"], path.parent, "mask.path")
    for name in ("h5", "data_root"):
        if name in cfg["data"]:
            cfg["data"][name] = _path(cfg["data"][name], path.parent, f"data.{name}")
    cfg["config_provenance"] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return cfg


def shared_defaults(cfg, entrypoint):
    """Map one configuration to the two existing command-line contracts."""
    data, sample, execution = cfg["data"], cfg["sampling"], cfg["execution"]
    common = {"device": execution["device"]}
    if entrypoint == "pilot":
        return {**common, "ckpt": cfg["checkpoint"], "h5": data.get("h5"),
                "slice_index": data.get("slice_index", 7), "crop_size": cfg["image_size"],
                "mask_path": cfg["mask"]["path"], "mask_layout": cfg["mask"]["layout"],
                "seed": sample["seed"], "model_init_seed": sample["model_init_seed"],
                "coarse_steps": sample["coarse_steps"], "refine_steps": sample["refine_steps"],
                "num_samples": sample["num_samples"], "dc_weight": sample["dc_weight"],
                "clip_denoised": sample["clip_denoised"],
                "precision": execution["precision"], "model_options": json.dumps(cfg["model_options"])}
    if entrypoint == "comparison":
        return {**common, "model_path": cfg["checkpoint"], "data_root": data.get("data_root"),
                "file1": data.get("file1", ""), "file2": data.get("file2", ""),
                "slice_index": data.get("slice_index", 0), "image_size": cfg["image_size"],
                "mask_path": cfg["mask"]["path"], "mask_layout": cfg["mask"]["layout"],
                "seeds": [sample["seed"]], "model_init_seed": sample["model_init_seed"],
                "timestep_respacing": str(sample["coarse_steps"]), "refine_steps": sample["refine_steps"],
                "num_samples": sample["num_samples"], "dc_weight": sample["dc_weight"],
                "clip_denoised": sample["clip_denoised"],
                "use_fp16": execution["precision"] == "fp16", "model_options": json.dumps(cfg["model_options"]),
                "eval_patient_id": data.get("patient_id", ""), "eval_exam_id": data.get("exam_id", "")}
    raise ValueError(f"Unknown entrypoint {entrypoint}")


def parse_with_shared_config(parser, argv, entrypoint):
    """Apply defaults from --config before parsing explicit CLI overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    cfg = read_shared_config(known.config) if known.config else None
    if cfg is not None:
        parser.set_defaults(**shared_defaults(cfg, entrypoint))
    return parser.parse_args(argv), cfg
