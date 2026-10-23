"""Reusable FP-pilot runner; keep all three supplied .py files together.



Run from your U-Net-GPTQ project with its dependencies installed:
    PYTHONPATH=src python /path/to/inspect_utils.py \
        --ckpt /models/model.pt --h5 /data/file.h5 --slice-index 7 \
        --out ~/unet_gptq_runs --plot
"""        
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any, Iterator

import numpy as np
import torch

if __package__:
    from .inspect_fastmri import inspect_pair
    from .simple_matrics import analyze_run, validate_arrays
else:
    from inspect_fastmri import inspect_pair
    from simple_matrics import analyze_run, validate_arrays


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    seed = _positive_int(seed, "seed", minimum=0)
    if seed >= 2 ** 32:
        raise ValueError("seed must be below 2**32")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def resolve_device(device: str, precision: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("supported devices are cpu and cuda[:index]")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu --precision fp32 for a CPU run")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA index {resolved.index} is unavailable")
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")
    if precision == "fp16" and resolved.type != "cuda":
        raise ValueError("fp16 requires CUDA; select fp32 on CPU")
    return resolved


def load_checkpoint_model(ckpt_path: str | Path, *, device: torch.device,
                          image_size: int = 320, coarse_steps: int = 100,
                          precision: str = "fp16", model_config: str | Path | None = None
                          ) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Strictly load the configured architecture; no silent partial weights."""
    from improved_diffusion.script_util_duo import model_and_diffusion_defaults, create_model_and_two_diffusion

    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    device = resolve_device(str(device), precision)
    image_size, coarse_steps = _positive_int(image_size, "image_size"), _positive_int(coarse_steps, "coarse_steps")
    config = model_and_diffusion_defaults()
    config.update(image_size=image_size, num_channels=128, num_res_blocks=3,
                  learn_sigma=False, dropout=0.3, diffusion_steps=4000,
                  noise_schedule="cosine", timestep_respacing=str(coarse_steps))
    if model_config is not None:
        overrides = json.loads(Path(model_config).expanduser().read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("model config must be a JSON object")
        unknown = set(overrides) - config.keys()
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        config.update(overrides)
    if config["image_size"] != image_size:
        raise ValueError("model image_size must equal crop_size")
    if "use_fp16" in config:
        config["use_fp16"] = precision == "fp16"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a nonempty state dict")
    model, coarse, refinement = create_model_and_two_diffusion(**config)
    model.load_state_dict(state, strict=True)
    del state
    if coarse.num_timesteps != coarse_steps:
        raise ValueError(f"expected {coarse_steps} coarse steps, got {coarse.num_timesteps}")
    model.eval().requires_grad_(False)
    if precision == "fp16":
        if not callable(getattr(model, "convert_to_fp16", None)):
            raise TypeError("this model does not provide DiffuseRecon's convert_to_fp16()")
        model.convert_to_fp16()  # upstream selectively preserves FP32 output layers
        if "dtype" in vars(model) and isinstance(model.dtype, torch.dtype):
            model.dtype = torch.float16
    model.to(device)
    return model, coarse, refinement, config


def normalize_pair(image: torch.Tensor, *, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Complex [2,H,W] -> [1,2,H,W], scaled per slice by max(|Re|,|Im|).

    This reproduces the pilot's oracle normalization from full reference data.
    Keep the returned scale when comparing FP/W4 or restoring image units.
    """
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 2 or not torch.is_complex(image):
        raise ValueError("image must be a complex tensor [2,H,W]")
    if image.numel() == 0 or not torch.isfinite(image).all():
        raise ValueError("image is empty or contains NaN/Inf")
    image = image.detach().to(device=device, dtype=torch.complex64).unsqueeze(0)
    scale = torch.maximum(image.real.abs(), image.imag.abs()).amax(dim=(-2, -1), keepdim=True)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("each slice must have a finite nonzero normalization scale")
    return image / scale, scale


def make_sampling_mask(height: int, width: int, *, acceleration: float = 4,
                       center_fraction: float = 0.08, seed: int = 0,
                       device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centered mask for export, unshifted mask for reconstruction)."""
    from unet_gptq.domain.forward_operator import cartesian_mask

    height, width = _positive_int(height, "height"), _positive_int(width, "width")
    seed = _positive_int(seed, "seed", minimum=0)
    if not np.isfinite(acceleration) or acceleration < 1:
        raise ValueError("acceleration must be finite and >= 1")
    if not np.isfinite(center_fraction) or not 0 < center_fraction <= 1 / acceleration:
        raise ValueError("center_fraction must be positive and <= 1/acceleration")
    centered = torch.as_tensor(cartesian_mask(height, width, acceleration=acceleration,
                                             center_fraction=center_fraction, seed=seed)).detach()
    if centered.shape != (height, width) or torch.is_complex(centered):
        raise ValueError("cartesian_mask must return a real [H,W] centered mask")
    if not torch.isfinite(centered).all() or not ((centered == 0) | (centered == 1)).all() or not centered.any():
        raise ValueError("cartesian_mask returned a nonbinary/empty sampling mask")
    centered = centered.to(device="cpu", dtype=torch.float32)
    unshifted = torch.fft.ifftshift(centered, dim=(-2, -1)).to(device)
    return centered, unshifted


def prepare_observation(reference: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic combined-image measurement: y=M*FFT(reference), zf=IFFT(y)."""
    if reference.ndim != 4 or reference.shape[1] != 2 or not torch.is_complex(reference) or not reference.numel():
        raise ValueError("reference must be nonempty complex [batch,2,H,W]")
    if not torch.isfinite(reference).all():
        raise ValueError("reference contains NaN/Inf")
    if mask.ndim != 2 or mask.shape != reference.shape[-2:] or mask.device != reference.device:
        raise ValueError("mask must be [H,W] on the reference device")
    if torch.is_complex(mask) or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask must be finite real binary")
    observed = torch.fft.fft2(reference, dim=(-2, -1), norm="backward") * mask
    zero_filled = torch.fft.ifft2(observed, dim=(-2, -1), norm="backward")
    return observed, zero_filled


@contextmanager
def reconstruction_progress(model: Any, enabled: bool = True) -> Iterator[None]:
    """Count actual forward calls; always remove the hook, including on error."""
    if not enabled:
        yield
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield
        return
    with tqdm(desc="FP reconstruction", unit="forward") as progress:
        def update_progress(_module: Any, _inputs: Any, _output: Any) -> None:
            # A non-None forward-hook return value would replace model output.
            progress.update(1)

        handle = model.register_forward_hook(update_progress)
        try:
            yield
        finally:
            handle.remove()


def export_legacy_inputs(image: torch.Tensor, centered_mask: torch.Tensor,
                         out: str | Path, slice_index: int) -> None:
    """Export unnormalized loader images with .pkl extensions, not fake .pt."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for slot in range(2):
        with (out / f"slice{slice_index + slot}.pkl").open("wb") as handle:
            pickle.dump({"img": image[slot].detach().cpu().numpy()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (out / "mask_centered.pkl").open("wb") as handle:
        pickle.dump(centered_mask.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_reconstruction(output: str | Path, *, reference: torch.Tensor, observed: torch.Tensor,
                         zero_filled: torch.Tensor, coarse: torch.Tensor, reconstruction: torch.Tensor,
                         mask: torch.Tensor, scale: torch.Tensor, slice_index: int) -> Path:
    tensors = dict(reference=reference, observed=observed, zero_filled=zero_filled,
                   coarse=coarse, reconstruction=reconstruction, mask=mask, scale=scale)
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all reconstruction outputs must be tensors")
    if not torch.isfinite(scale).all() or (scale <= 0).any() or tuple(scale.shape) != (*reference.shape[:2], 1, 1):
        raise ValueError("scale must be finite positive [batch,2,1,1]")
    arrays = {name: value.detach().cpu().numpy() for name, value in tensors.items()}
    slice_index = _positive_int(slice_index, "slice_index", minimum=0)
    arrays["slice_indices"] = np.asarray([slice_index, slice_index + 1], dtype=np.int64)
    validate_arrays(arrays)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Pass a file handle so NumPy cannot silently add a second .npz suffix.
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return output


def run_pilot(ckpt: str | Path, h5: str | Path, *, out: str | Path = Path.home() / "unet_gptq_runs",
              slice_index: int = 7, crop_size: int = 320, acs_width: int = 24,
              acceleration: float = 4, center_fraction: float = 0.08, seed: int = 0,
              coarse_steps: int = 100, refine_steps: int = 20, dc_weight: float = 1.0,
              device: str = "cuda:0", precision: str = "fp16", model_config: str | Path | None = None,
              deterministic: bool = True, progress: bool = True, plot: bool = False,
              export_pickles: bool = False) -> Path:
    """Check one real loader pair, run the existing sampler, save and score it."""
    slice_index = _positive_int(slice_index, "slice_index", minimum=0)
    crop_size, acs_width = _positive_int(crop_size, "crop_size"), _positive_int(acs_width, "acs_width")
    coarse_steps, refine_steps = _positive_int(coarse_steps, "coarse_steps"), _positive_int(refine_steps, "refine_steps", minimum=0)
    seed = _positive_int(seed, "seed", minimum=0)
    if seed >= 2 ** 32:
        raise ValueError("seed must be below 2**32")
    if not np.isfinite(dc_weight) or not 0 <= dc_weight <= 1:
        raise ValueError("dc_weight must be between 0 and 1")
    ckpt, h5 = Path(ckpt).expanduser().resolve(), Path(h5).expanduser().resolve()
    for path in (ckpt, h5):
        if not path.is_file():
            raise FileNotFoundError(path)
    resolved_device = resolve_device(device, precision)
    sample = inspect_pair(h5, slice_index=slice_index, crop_size=crop_size, acs_width=acs_width)
    centered_mask, mask = make_sampling_mask(crop_size, crop_size, acceleration=acceleration,
                                             center_fraction=center_fraction, seed=seed, device=resolved_device)
    reference, scale = normalize_pair(sample["image"], device=resolved_device)
    observed, zero_filled = prepare_observation(reference, mask)
    run_dir = Path(out).expanduser().resolve() / f"fp_{h5.stem}_s{slice_index}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    arguments = dict(ckpt=str(ckpt), h5=str(h5), slice_index=slice_index, crop_size=crop_size,
                     acs_width=acs_width, acceleration=acceleration, center_fraction=center_fraction,
                     seed=seed, coarse_steps=coarse_steps, refine_steps=refine_steps, dc_weight=dc_weight,
                     device=str(resolved_device), precision=precision, deterministic=deterministic,
                     model_config=str(Path(model_config).expanduser().resolve()) if model_config else None)
    manifest = {
        "schema_version": 1, "status": "INITIALIZING", "arguments": arguments,
        "acquisition": str(sample.get("contrast", "")), "target_source": sample.get("target_source"),
        "slice_indices": [slice_index, slice_index + 1],
        "measurement_model": "synthetic_singlecoil_from_combined_reference",
        "normalization": "oracle_per_slice_component_max", "fft": "unshifted/backward",
        "mask_convention": "unshifted in NPZ; centered only in legacy export",
        "effective_acceleration": float(mask.numel() / mask.sum().item()),
        "h5_size_bytes": h5.stat().st_size, "h5_mtime_ns": h5.stat().st_mtime_ns,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "checkpoint_flair_suitability": "NOT_VALIDATED",
    }

    def save_manifest() -> None:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_manifest()
    try:
        model, coarse_diffusion, refinement, config = load_checkpoint_model(
            ckpt, device=resolved_device, image_size=crop_size, coarse_steps=coarse_steps,
            precision=precision, model_config=model_config)
        if hasattr(refinement, "num_timesteps") and refine_steps > refinement.num_timesteps:
            raise ValueError(f"refine_steps={refine_steps} exceeds refinement schedule {refinement.num_timesteps}")
        manifest.update(status="RUNNING", model_config=config, checkpoint_sha256=sha256_file(ckpt))
        save_manifest()
        if export_pickles:
            export_legacy_inputs(sample["image"], centered_mask, run_dir, slice_index)
        from unet_gptq.diffusion.samplers import reconstruct_pair

        # Model construction may consume randomness: reset directly before sampling.
        seed_everything(seed, deterministic=deterministic)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        started = time.perf_counter()
        with torch.no_grad(), reconstruction_progress(model, enabled=progress):
            coarse, reconstruction = reconstruct_pair(
                model, coarse_diffusion, refinement, observed, mask, num_samples=1,
                refine_steps=refine_steps, dc_weight=dc_weight)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        manifest["inference_seconds"] = time.perf_counter() - started
        npz_path = save_reconstruction(
            run_dir / "reconstruction.npz", reference=reference, observed=observed,
            zero_filled=zero_filled, coarse=coarse, reconstruction=reconstruction,
            mask=mask, scale=scale, slice_index=slice_index)
        result = analyze_run(npz_path, plot=plot)
        manifest.update(status="COMPLETED", input_checks=result["input_checks"],
                        outputs=["reconstruction.npz", "metrics.csv", "metrics_summary.json"] + (["comparison.png"] if plot else []))
        save_manifest()
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        save_manifest()
        raise
    print("Saved run:", run_dir)
    for row in result["metrics"]:
        print(f"slice {row['slice_index']} | {row['stage']}: "
              f"NMSE={row['magnitude_nmse']:.6g}, DC={row['dc_relative_l2']:.3e}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path.home() / "unet_gptq_runs", help="parent directory for unique runs")
    parser.add_argument("--slice-index", type=int, default=7)
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--acs-width", type=int, default=24)
    parser.add_argument("--acceleration", type=float, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coarse-steps", type=int, default=100)
    parser.add_argument("--refine-steps", type=int, default=20)
    parser.add_argument("--dc-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--model-config", type=Path, help="optional JSON architecture/schedule overrides")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export-pickles", action="store_true")
    run_pilot(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())