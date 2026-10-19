"""Generate a DiffuseRecon Conv2d W4 artifact for run_exp.py.

Use trusted legacy pickle {'img': complex [H,W]} slice pairs listed in JSONL.
The JSONL split must be calibration/train; never select evaluation cases here.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from unet_gptq.diffusion.model_factory import resolve_model_config
from unet_gptq.quantization.calibration import (
    CalibrationSample, CalibrationStage, calibrate_conv2d_model, select_timesteps,
)
from unet_gptq.quantization.gptq import GPTQConfig
from unet_gptq.train.run_exp import normalize_image, sha256_file, source_info


def load_calibration_samples(manifest_path: Path, data_root: Path | None, *, split: str,
                             max_samples: int, image_size: int) -> tuple[list[CalibrationSample], dict]:
    """Use the first N rows of the requested split, in manifest order.

    No masks or reconstruction outputs are read. Per-slice normalization and
    channel order exactly match run_exp.load_case / img_to_channels.
    """
    if split not in ("calibration", "train") or max_samples < 1 or image_size < 1:
        raise ValueError("Use calibration/train split, positive max_samples and image_size")
    manifest_path = manifest_path.expanduser().resolve()
    root = data_root.expanduser().resolve() if data_root else manifest_path.parent
    records = []
    for line_no, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Manifest line {line_no} must be an object")
        if record.get("split") == split:
            records.append((line_no, record))
    if not records:
        raise ValueError(f"No records for split={split!r}")
    samples, identities = [], set()
    for line_no, record in records[:max_samples]:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Manifest line {line_no}: sample_id must be a nonempty string")
        images, files, scales = [], [], []
        for key in ("file1", "file2"):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Manifest line {line_no}: {key} is required")
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else root / path).resolve()
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if not isinstance(payload, dict) or "img" not in payload:
                raise ValueError(f"{path}: expected pickle dictionary with img")
            image, scale = normalize_image(payload["img"])
            if image.shape != (image_size, image_size):
                raise ValueError(f"{path}: expected {(image_size, image_size)}, got {image.shape}")
            images.append(image)
            scales.append(scale)
            files.append({"path": str(path), "sha256": sha256_file(path)})
        identity = tuple(sorted(f["sha256"] for f in files))
        if files[0]["path"] == files[1]["path"] or identity in identities:
            raise ValueError("Duplicate slice pair in calibration data")
        identities.add(identity)
        first, second = images
        packed = np.stack([first.real, first.imag, second.real, second.imag])[None]
        samples.append(CalibrationSample(sample_id, torch.from_numpy(packed.copy()), {
            "split": split, "manifest_line": line_no, "image_files": files,
            "normalization_scales": scales, "contrast": record.get("contrast"),
        }))
    return samples, {
        "data_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "data_root": str(root), "split": split, "available_pairs": len(records),
        "selected_pairs": len(samples), "sample_selection": "first_N_matching_rows_in_manifest_order",
        "normalization": "per-slice max(abs(real),abs(imag)); same as run_exp; full-reference/oracle scale",
        "channel_order": ["Re(slice1)", "Im(slice1)", "Re(slice2)", "Im(slice2)"],
        "leakage_check": "declared split only; patient/volume disjointness must be established by dataset manifests",
    }


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--calibration_manifest", type=Path, required=True)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--split", choices=("calibration", "train"), default="calibration")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--output_dir", required=True, help="NEW directory; never overwritten")
    parser.add_argument("--factory_module", default="improved_diffusion.script_util_duo")
    parser.add_argument("--model_options", default="{}")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--timestep_respacing", default="")
    parser.add_argument("--refine_steps", type=int, default=20)
    parser.add_argument("--stage", choices=("coarse", "refined", "both"), default="both")
    parser.add_argument("--num_calibration_timesteps", type=int, default=16)
    parser.add_argument("--coarse_timesteps", type=int, nargs="+")
    parser.add_argument("--refine_timesteps", type=int, nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_fp16", action="store_true", help="Same convert_to_fp16 path as run_exp; source weights stay FP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_init_seed", type=int, default=0)
    parser.add_argument("--include", nargs="*", default=[], help="Case-sensitive fnmatch module names")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hessian_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max_hessian_mib", type=int, default=512)
    parser.add_argument("--patch_chunk_size", type=int, default=1024)
    parser.add_argument("--max_patches_per_batch", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_argparser().parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError(args.output_dir)
    if args.num_calibration_timesteps < 1 or args.max_hessian_mib < 1 or args.refine_steps < 0:
        raise ValueError("Positive timestep count/memory limit and nonnegative refine_steps required")
    if any(not 0 <= v < 2**32 for v in (args.seed, args.model_init_seed)):
        raise ValueError("Seeds must be in [0,2**32)")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use one CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    if args.coarse_timesteps is not None and args.stage == "refined":
        raise ValueError("--coarse_timesteps requires coarse calibration")
    if args.refine_timesteps is not None and (args.stage == "coarse" or not args.refine_steps):
        raise ValueError("--refine_timesteps requires active refinement")
    random.seed(args.model_init_seed)
    np.random.seed(args.model_init_seed)
    torch.manual_seed(args.model_init_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    module = importlib.import_module(args.factory_module)
    config = resolve_model_config(module, model_options=args.model_options, image_size=args.image_size,
                                  timestep_respacing=args.timestep_respacing, use_fp16=args.use_fp16)
    if config.get("class_cond"):
        raise ValueError("Class-conditional checkpoints require an explicit label calibration policy")
    model, coarse, refined = module.create_model_and_two_diffusion(**config)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    if args.refine_steps > refined.num_timesteps:
        raise ValueError("refine_steps exceeds the factory's refinement schedule")
    stages = []
    if args.stage in ("coarse", "both"):
        stages.append(CalibrationStage("coarse", coarse, select_timesteps(
            coarse.num_timesteps, args.num_calibration_timesteps, args.coarse_timesteps)))
    if args.stage in ("refined", "both") and args.refine_steps:
        stages.append(CalibrationStage("refined", refined, select_timesteps(
            args.refine_steps, args.num_calibration_timesteps, args.refine_timesteps)))
    if not stages:
        raise ValueError("No active calibration stages")
    samples, provenance = load_calibration_samples(
        args.calibration_manifest, args.data_root, split=args.split,
        max_samples=args.max_samples, image_size=args.image_size,
    )
    provenance.update(factory_module=args.factory_module, factory_source=source_info(module),
                      model_config=config, model_init_seed=args.model_init_seed,
                      refine_steps=args.refine_steps, timestep_selection=(
                          "explicit local indices where supplied; otherwise endpoint-inclusive uniform indices"),
                      entrypoint=source_info(importlib.import_module(__name__)),
                      environment={"torch": torch.__version__, "numpy": np.__version__, "device": str(device)})
    paths = calibrate_conv2d_model(
        model, args.model_path, samples, stages, args.output_dir, provenance=provenance,
        config=GPTQConfig(group_size=args.group_size, block_size=args.block_size,
                          damping=args.damping, symmetric=args.symmetric, act_order=args.act_order),
        device=args.device, use_fp16=args.use_fp16, seed=args.seed, include=args.include, exclude=args.exclude,
        hessian_dtype=getattr(torch, args.hessian_dtype), max_hessian_bytes=args.max_hessian_mib * 1024**2,
        patch_chunk_size=args.patch_chunk_size, max_patches_per_batch=args.max_patches_per_batch,
    )
    print(json.dumps({"status": "completed", "reconstruction": "not_run",
                      "--w4_model_path": str(paths["checkpoint"]),
                      "--w4_manifest": str(paths["manifest"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
