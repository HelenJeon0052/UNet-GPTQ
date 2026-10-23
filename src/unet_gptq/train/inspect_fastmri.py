from __future__ import annotations
import json
import sys

sys.path.insert(0, "src")

import argparse
from collections import Counter
import csv
import json
from numbers import Integral
from pathlib import Path
import tempfile
from typing import Any, Sequence

import h5py
import numpy as np
from unet_gptq.datas.fastmri_dataset import FastMRISlicePairs

AXES = (-2, -1)


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def ifft2c_numpy(kspace: np.ndarray) -> np.ndarray:
    """Independent centered/orthonormal inverse FFT for raw fastMRI k-space."""
    kspace = np.asarray(kspace)
    if kspace.ndim < 2 or not np.iscomplexobj(kspace) or not kspace.size or not np.isfinite(kspace).all():
        raise ValueError("kspace must be nonempty, finite, complex [..., H, W]")
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=AXES), axes=AXES, norm="ortho"), axes=AXES)


def center_crop_numpy(array: np.ndarray, crop_size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(crop_size, Integral):
        height = width = _integer(crop_size, "crop_size")
    elif isinstance(crop_size, tuple) and len(crop_size) == 2:
        height, width = (_integer(size, "crop dimension") for size in crop_size)
    else:
        raise ValueError("crop_size must be an integer or (height, width)")
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError("center crop requires at least two dimensions")
    h, w = array.shape[-2:]
    if height > h or width > w:
        raise ValueError(f"crop {(height, width)} exceeds image {(h, w)}")
    top, left = (h - height) // 2, (w - width) // 2
    return array[..., top:top + height, left:left + width]


def inspect_acquisition(h5_path: str | Path, *, verbose: bool = True) -> dict[str, Any]:
    path = Path(h5_path).expanduser().resolve()
    with h5py.File(path, "r") as volume:
        if "kspace" not in volume:
            raise KeyError(f"{path}: missing kspace")
        kspace = volume["kspace"]
        if kspace.ndim not in (3, 4) or any(size == 0 for size in kspace.shape):
            raise ValueError(f"{path}: expected nonempty [S,H,W] or [S,C,H,W], got {kspace.shape}")
        if not np.issubdtype(kspace.dtype, np.complexfloating):
            raise TypeError(f"{path}: expected complex kspace, got {kspace.dtype}")
        result = {
            "path": str(path), "acquisition": _text(volume.attrs.get("acquisition", "UNKNOWN")),
            "slices": int(kspace.shape[0]), "coils": int(kspace.shape[1]) if kspace.ndim == 4 else 1,
            "height": int(kspace.shape[-2]), "width": int(kspace.shape[-1]),
            "has_rss": "reconstruction_rss" in volume, "has_esc": "reconstruction_esc" in volume,
            "datasets": list(volume.keys()),
        }
    if result["slices"] >= 2:
        first = (result["slices"] - 2) // 2
        result["center_pair"] = [first, first + 1]
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def numpy_pair_reference(h5_path: str | Path, slice_index: int = 10,
                         crop_size: int = 320, acs_width: int = 24) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only two slices, reproduce square ACS/SOS combination, then crop.

    ks_eq is synthetic single-coil k-space of the combined/cropped image;
    it is not the acquired multicoil k-space.
    """
    slice_index = _integer(slice_index, "slice_index", minimum=0)
    crop_size, acs_width = _integer(crop_size, "crop_size"), _integer(acs_width, "acs_width")
    metadata = inspect_acquisition(h5_path, verbose=False)
    if slice_index + 1 >= metadata["slices"]:
        raise IndexError(f"need adjacent slices {slice_index}, {slice_index + 1}; volume has {metadata['slices']}")
    with h5py.File(metadata["path"], "r") as volume:
        raw = np.asarray(volume["kspace"][slice_index:slice_index + 2])
        target_source = ("reconstruction_rss" if "reconstruction_rss" in volume else
                         "reconstruction_esc" if volume["kspace"].ndim == 3 and "reconstruction_esc" in volume else "rss_from_coils")
        stored = np.asarray(volume[target_source][slice_index:slice_index + 2]) if target_source in volume else None
    if not np.isfinite(raw).all():
        raise ValueError("raw k-space contains NaN or Inf")
    with np.errstate(over="ignore", invalid="ignore"):
        raw = raw.astype(np.complex64)
    if not np.isfinite(raw).all():
        raise ValueError("raw k-space overflows complex64")
    if raw.ndim == 3:
        raw = raw[:, None]
    raw = raw.astype(np.complex128)  # independent high-precision FFT reference
    coils = ifft2c_numpy(raw)
    if raw.shape[1] == 1:
        sensitivity = np.ones_like(coils)
    else:
        h, w = raw.shape[-2:]
        if acs_width > min(h, w):
            raise ValueError(f"acs_width={acs_width} exceeds {(h, w)}")
        top, left = h // 2 - acs_width // 2, w // 2 - acs_width // 2
        low = np.zeros_like(raw)
        region = (..., slice(top, top + acs_width), slice(left, left + acs_width))
        low[region] = raw[region]
        low_coils = ifft2c_numpy(low)
        low_rss = np.sqrt(np.sum(np.abs(low_coils) ** 2, axis=1, keepdims=True))
        sensitivity = low_coils / np.maximum(low_rss, 1e-8)
    power = np.sum(np.abs(sensitivity) ** 2, axis=1)
    combined = np.sum(sensitivity.conj() * coils, axis=1) / np.maximum(power, 1e-8)
    image = center_crop_numpy(combined, crop_size)
    rss = center_crop_numpy(np.sqrt(np.sum(np.abs(coils) ** 2, axis=1)), crop_size)
    if stored is not None:
        if stored.ndim != 3 or stored.shape[0] != 2 or np.iscomplexobj(stored) or not np.isfinite(stored).all() or (stored < 0).any():
            raise ValueError("stored target must be finite nonnegative real [2,H,W]")
        with np.errstate(over="ignore", invalid="ignore"):
            stored = stored.astype(np.float32)
        if not np.isfinite(stored).all():
            raise ValueError("stored target overflows float32")
    expected = {
        "sens": center_crop_numpy(sensitivity, crop_size), "image": image,
        "rss_from_coils": rss, "ks_eq": np.fft.fft2(image, axes=AXES, norm="backward"),
        "target_rss": center_crop_numpy(stored, crop_size) if stored is not None else rss,
        "coil_combine_valid": center_crop_numpy(power > 1e-8, crop_size),
    }
    metadata.update(slice_indices=[slice_index, slice_index + 1], target_source=target_source,
                    measurement_model="synthetic_singlecoil_from_combined_reference")
    return expected, metadata


def validate_sample(sample: dict[str, Any], expected: dict[str, np.ndarray], *,
                    rtol: float = 3e-5, atol: float = 1e-12) -> dict[str, dict[str, float]]:
    """Validate loader shapes, dtypes, finiteness, and independent NumPy errors."""
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")
    report = {}
    for name, reference in expected.items():
        if name not in sample:
            raise KeyError(f"loader omitted {name}")
        value = sample[name]
        actual = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        dtype = np.bool_ if name == "coil_combine_valid" else np.complex64 if name in ("sens", "image", "ks_eq") else np.float32
        if actual.dtype != np.dtype(dtype) or actual.shape != reference.shape:
            raise ValueError(f"{name}: got {actual.shape}/{actual.dtype}, expected {reference.shape}/{np.dtype(dtype)}")
        if not actual.size or not np.isfinite(actual).all():
            raise ValueError(f"{name}: empty array or NaN/Inf")
        if name == "coil_combine_valid":
            if not np.array_equal(actual, reference):
                raise ValueError("coil_combine_valid differs from NumPy reference")
            continue
        error = actual.astype(reference.dtype) - reference
        maximum = float(np.max(np.abs(error)))
        denominator = float(np.linalg.norm(reference.ravel()))
        relative = float(np.linalg.norm(error.ravel()) / denominator) if denominator else (0.0 if maximum == 0 else float("inf"))
        report[name] = {"relative_l2": relative, "max_abs": maximum}
        if relative > rtol and maximum > atol:
            raise ValueError(f"{name}: NumPy check failed: relative_L2={relative:.3e}, max_abs={maximum:.3e}")
    return report


def inspect_pair(h5_path: str | Path, slice_index: int = 10, crop_size: int = 320,
                 acs_width: int = 24, *, verbose: bool = True,
                 rtol: float = 3e-5, atol: float = 1e-12) -> dict[str, Any]:
    """Return the original loader sample after checking it, without normalization."""
    expected, metadata = numpy_pair_reference(h5_path, slice_index, crop_size, acs_width)
    

    with tempfile.TemporaryDirectory(prefix="fastmri_pair_") as directory:
        manifest = Path(directory) / "pair.jsonl"
        manifest.write_text(json.dumps({"path": metadata["path"], "slice": int(slice_index)}) + "\n", encoding="utf-8")
        sample = FastMRISlicePairs(manifest, crop_size=int(crop_size), acs_width=int(acs_width), sens_method="acs_sos")[0]
    if sample.get("target_source") != metadata["target_source"]:
        raise ValueError("loader target_source disagrees with HDF5 contents")
    report = validate_sample(sample, expected, rtol=rtol, atol=atol)
    if verbose:
        print("Acquisition:", metadata["acquisition"], "| slices:", metadata["slice_indices"])
        print("Target source:", metadata["target_source"], "| normalization: none")
        for name, values in report.items():
            print(f"{name}: relative_L2={values['relative_l2']:.3e}, max_abs={values['max_abs']:.3e}")
        print("PASS: loader matches independent NumPy reference")
    return sample


def sample_info(sample: dict[str, Any], *, verbose: bool = True) -> dict[str, Any]:
    result = {name: sample.get(name) for name in ("filename", "contrast", "slice_index", "n_coils", "target_source", "measurement_model")}
    result["image_shape"] = list(sample["image"].shape)
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def filter_files(DATA_ROOT: str | Path, output_csv: str | Path = "fastmri_inventory.csv", *,
                 acquisition: str | None = None, verbose: bool = True) -> dict[str, Any]:
    """Inventory HDF5 metadata; report unreadable files without hiding failures."""
    root = Path(DATA_ROOT).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files under {root}")
    rows, errors = [], []
    for path in files:
        try:
            info = inspect_acquisition(path, verbose=False)
            if acquisition is not None and acquisition.casefold() not in info["acquisition"].casefold():
                continue
            rows.append({key: value for key, value in info.items() if key not in ("datasets", "center_pair")})
        except (OSError, KeyError, ValueError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    output = Path(output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "acquisition", "slices", "coils", "height", "width", "has_rss", "has_esc"]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = dict(sorted(Counter(row["acquisition"] for row in rows).items()))
    if verbose:
        print("Acquisition counts:", counts)
        print(f"Saved {len(rows)} volumes to {output}; unreadable files: {len(errors)}")
        for error in errors[:5]:
            print(error["path"], error["error"])
    return {"rows": rows, "errors": errors, "counts": counts, "output_csv": str(output)}


def inspect_files(DATA_ROOT: str | Path, names: Sequence[str] | None = None, *,
                  slice_index: int = 7, crop_size: int = 320, acs_width: int = 24,
                  limit: int = 3) -> list[dict[str, Any]]:
    root = Path(DATA_ROOT).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    limit = _integer(limit, "limit")
    paths = [root / name for name in names] if names is not None else sorted(root.rglob("*.h5"))[:limit]
    if not paths:
        raise FileNotFoundError(f"no selected .h5 files under {root}")
    samples = []
    for path in paths:
        sample = inspect_pair(path, slice_index, crop_size, acs_width)
        sample_info(sample)
        samples.append(sample)
    return samples


# Keep old entry points usable; no hard-coded file2/file7/file9 placeholders.
test_acquisition = inspect_acquisition
zoler = inspect_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info").add_argument("path", type=Path)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("path", type=Path)
    inventory.add_argument("--out", type=Path, default=Path("fastmri_inventory.csv"))
    inventory.add_argument("--acquisition", help="case-insensitive substring, e.g. FLAIR")
    pair = commands.add_parser("pair")
    pair.add_argument("path", type=Path)
    pair.add_argument("--slice-index", type=int, default=7)
    pair.add_argument("--crop-size", type=int, default=320)
    pair.add_argument("--acs-width", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command == "info":
        inspect_acquisition(args.path)
    elif args.command == "inventory":
        result = filter_files(args.path, args.out, acquisition=args.acquisition)
        return 1 if result["errors"] else 0
    else:
        sample_info(inspect_pair(args.path, args.slice_index, args.crop_size, args.acs_width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
