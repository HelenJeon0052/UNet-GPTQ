"""Magnitude/complex metrics for the saved FP pilot.
    python simple_matrics.py /path/to/run --plot
"""

from __future__ import annotations
import argparse 
import pickle

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import json
import hashlib
import zipfile
import pickle
from typing import Any
from datetime import datetime, timezone

AXES = (-2, -1)
STAGES = ("zero_filled", "coarse", "reconstruction")


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or array.size == 0:
        raise ValueError(f"{name} must be a nonempty numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _matched(value: Any, reference: Any) -> tuple[np.ndarray, np.ndarray]:
    value, reference = _array(value, "value"), _array(reference, "reference")
    if value.shape != reference.shape:
        raise ValueError(f"shape mismatch: {value.shape} != {reference.shape}")
    dtype = np.complex128 if np.iscomplexobj(value) or np.iscomplexobj(reference) else np.float64
    return value.astype(dtype, copy=False), reference.astype(dtype, copy=False)


def _ratio(numerator: float, denominator: float) -> float:
    # A zero reference has no relative scale; do not hide it behind epsilon.
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)

def relative_l2(value: Any, reference: Any) -> float:
    """||value-reference||_2 / ||reference||_2; shapes must match exactly."""
    value, reference = _matched(value, reference)
    return _ratio(float(np.linalg.norm((value - reference).ravel())),
                  float(np.linalg.norm(reference.ravel())))


def nmse(value: Any, reference: Any, *, magnitude: bool = True) -> float:
    """Squared relative error. Magnitude NMSE ignores complex phase."""
    value, reference = _matched(value, reference)
    if magnitude:
        value, reference = np.abs(value), np.abs(reference)
    return _ratio(float(np.sum(np.abs(value - reference) ** 2, dtype=np.float64)),
                  float(np.sum(np.abs(reference) ** 2, dtype=np.float64)))


def psnr(value: Any, reference: Any) -> float:
    """Magnitude PSNR in dB, using this reference slice's maximum as peak."""
    value, reference = _matched(value, reference)
    value, reference = np.abs(value), np.abs(reference)
    mse = float(np.mean((value - reference) ** 2))
    peak = float(reference.max())
    if mse == 0:
        return float("inf")
    if peak == 0:
        return float("-inf")
    return float(20 * np.log10(peak) - 10 * np.log10(mse))

def resolve_npz(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        preferred = path / "reconstruction.npz"
        candidates = [preferred] if preferred.is_file() else sorted(path.glob("*.npz"))
        if len(candidates) != 1:
            raise ValueError(f"{path}: expected one run NPZ, found {len(candidates)}")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = ("reference", "observed", "mask", *STAGES)
    missing = set(required) - arrays.keys()
    if missing:
        raise ValueError(f"missing NPZ arrays: {sorted(missing)}")
    reference = _array(arrays["reference"], "reference")
    if reference.ndim != 4 or reference.shape[1] != 2 or not np.iscomplexobj(reference):
        raise ValueError("reference must be complex [batch, 2, H, W]")
    for name in ("observed", *STAGES):
        value = _array(arrays[name], name)
        if value.shape != reference.shape or not np.iscomplexobj(value):
            raise ValueError(f"{name} must be complex with shape {reference.shape}")
    mask = np.asarray(arrays["mask"])
    if mask.ndim not in (2, 3, 4) or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError("mask must have matching spatial dimensions")
    if mask.dtype.kind not in "bifu" or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must contain only finite binary values")
    try:
        np.broadcast_to(mask, reference.shape)
    except ValueError as exc:
        raise ValueError("mask cannot broadcast to reference") from exc
    if not mask.any():
        raise ValueError("mask must contain at least one observed location")
    if "slice_indices" in arrays:
        indices = arrays["slice_indices"]
        if indices.shape not in ((2,), reference.shape[:2]) or indices.dtype.kind not in "iu" or (indices < 0).any():
            raise ValueError("slice_indices must be nonnegative integers [2] or [batch, 2]")

def load_run(path: str | Path) -> dict[str, np.ndarray]:
    """Load numeric arrays without permitting pickled objects."""
    with np.load(resolve_npz(path), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    validate_arrays(arrays)
    return arrays


def validate_saved_inputs(arrays: dict[str, np.ndarray], *, tolerance: float = 3e-5) -> dict[str, float]:
    """Check unshifted/backward FFT, observation masking, and zero filling."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    validate_arrays(arrays)
    checks = {
        "observed_relative_l2": relative_l2(
            arrays["observed"], arrays["mask"] * np.fft.fft2(arrays["reference"], axes=AXES, norm="backward")),
        "zero_filled_relative_l2": relative_l2(
            arrays["zero_filled"], np.fft.ifft2(arrays["observed"], axes=AXES, norm="backward")),
    }
    failed = {name: error for name, error in checks.items() if error > tolerance}
    if failed:
        raise ValueError(f"saved input consistency check failed (tolerance={tolerance}): {failed}")
    return checks

def compute_metrics(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """One row per batch pair, slice, and stage; no averaging across slices."""
    validate_arrays(arrays)
    indices = arrays.get("slice_indices")
    rows = []
    for pair in range(arrays["reference"].shape[0]):
        for slot in range(2):
            target = arrays["reference"][pair, slot]
            observed = arrays["observed"][pair, slot]
            mask = np.broadcast_to(arrays["mask"], arrays["reference"].shape)[pair, slot]
            slice_index = "" if indices is None else int(indices[slot] if indices.ndim == 1 else indices[pair, slot])
            for stage in STAGES:
                image = arrays[stage][pair, slot]
                rows.append({
                    "stage": stage, "pair_index": pair, "slice_in_pair": slot,
                    "slice_index": slice_index,
                    "magnitude_nmse": nmse(image, target),
                    "complex_nmse": nmse(image, target, magnitude=False),
                    "magnitude_rmse": float(np.sqrt(np.mean((np.abs(image.astype(np.complex128)) - np.abs(target.astype(np.complex128))) ** 2))),
                    "magnitude_psnr_db": psnr(image, target),
                    "dc_relative_l2": relative_l2(mask * np.fft.fft2(image, norm="backward"), observed),
                })
    return rows

def plot_run(arrays: dict[str, np.ndarray], output: str | Path, *, pair_index: int = 0) -> Path:
    """Save both slices with shared reference-based magnitude and error limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validate_arrays(arrays)
    if isinstance(pair_index, bool) or not isinstance(pair_index, int) or not 0 <= pair_index < arrays["reference"].shape[0]:
        raise ValueError("pair_index outside the saved batch")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target = np.abs(arrays["reference"][pair_index])
    vmax = max(float(target.max()), np.finfo(float).eps)
    error_max = max(float(np.abs(np.abs(arrays[name][pair_index]) - target).max()) for name in STAGES)
    error_max = max(error_max, np.finfo(float).eps)
    figure, axes = plt.subplots(4, 4, figsize=(13, 12), constrained_layout=True)
    indices = arrays.get("slice_indices")
    try:
        for slot in range(2):
            label = str(slot) if indices is None else str(indices[slot] if indices.ndim == 1 else indices[pair_index, slot])
            for column, name in enumerate(("reference", *STAGES)):
                image = np.abs(arrays[name][pair_index, slot])
                axes[2 * slot, column].imshow(image, cmap="gray", vmin=0, vmax=vmax)
                axes[2 * slot, column].set_title(f"slice {label}: {name}")
                axes[2 * slot + 1, column].imshow(np.abs(image - target[slot]), cmap="magma", vmin=0, vmax=error_max)
                axes[2 * slot + 1, column].set_title("absolute magnitude error")
        for axis in axes.flat:
            axis.axis("off")
        figure.savefig(output, dpi=150)
    finally:
        plt.close(figure)
    return output


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_safe_json(data), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")

def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def resolve_runs(path: str | Path) -> list[Path]:
    """Explicit run root, consolidated NPZ, or one runner outputs.npz.

    Expand all FP seeds in an explicit run_exp root. Never pick the newest run,
    infer a condition from a name, or accidentally treat inputs.npz as a result.
    """
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".npz" or path.name == "inputs.npz":
            raise ValueError(f"Choose a reconstruction NPZ or run directory: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    consolidated = path / "reconstruction.npz"
    outputs = sorted(path.glob("seed_*/fp/outputs.npz"))
    if consolidated.is_file() and outputs:
        raise ValueError(f"Both output schemas found; pass the exact NPZ: {path}")
    if consolidated.is_file():
        return [consolidated]
    if (path / "inputs.npz").is_file() and outputs:
        return outputs
    if (path / "outputs.npz").is_file():
        return [path / "outputs.npz"]
    candidates = [p for p in sorted(path.glob("*.npz")) if p.name != "inputs.npz"]
    if len(candidates) == 1:
        return candidates
    raise ValueError(f"No unique result in {path}; pass a specific NPZ or a run root")

def load_saved(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    files = resolve_runs(path)
    if len(files) != 1:
        raise ValueError("Multiple FP seeds found: pass an individual outputs.npz to analyze_run()")
    path = files[0]
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    source_files = [path]
    root, run_info, schema, seed, condition = path.parent, {}, "consolidated", None, "fp"
    if "reference" not in arrays and {"coarse", "refined"} <= arrays.keys():
        candidates = [parent for parent in list(path.parents)[:4]
                      if (parent / "inputs.npz").is_file()]
        if not candidates:
            raise FileNotFoundError(f"inputs.npz needed alongside the run containing {path}")
        root = candidates[0]
        source_files.append(root / "inputs.npz")
        with np.load(root / "inputs.npz", allow_pickle=False) as data:
            arrays.update(reference=data["reference"], observed=data["k_obs"], mask=data["mask"])
            if "scales" in data.files:
                arrays["scale"] = data["scales"]
        arrays["reconstruction"] = arrays.pop("refined")
        arrays["zero_filled"] = np.fft.ifft2(arrays["observed"], axes=AXES, norm="backward")
        run_info = _json(path.parent / "run.json")
        schema, condition = "run_exp", path.parent.name
        if path.parent.parent.name.startswith("seed_"):
            seed = int(path.parent.parent.name.removeprefix("seed_"))
    manifest = _json(root / "manifest.json")
    args = manifest.get("arguments", {})
    if "slice_indices" not in arrays:
        indices = manifest.get("slice_indices")
        if indices is None and args.get("slice_index") is not None:
            indices = [args["slice_index"], args["slice_index"] + 1]
        if indices is not None:
            arrays["slice_indices"] = np.asarray(indices, dtype=np.int64)
    validate_arrays(arrays)
    if seed is None:
        seed = args.get("seed")
    precision = args.get("precision")
    if precision is None and "use_fp16" in args:
        precision = "fp16" if args["use_fp16"] else "fp32"
    config = manifest.get("model_config", manifest.get("environment", {}).get("factory_config"))
    meta = {
        "npz_path": str(path), "run_root": str(root), "schema": schema,
        "condition": condition, "seed": seed, "dc_weight": args.get("dc_weight"),
        "precision": precision, "coarse_steps": args.get("coarse_steps", args.get("timestep_respacing")),
        "refine_steps": args.get("refine_steps"),
        "num_samples": args.get("num_samples", 1 if schema == "consolidated" and manifest else None),
        "checkpoint_sha256": manifest.get("checkpoint_sha256", manifest.get("fp_checkpoint_sha256")),
        "model_config": config,
        "sampled_fraction": float(np.mean(arrays["mask"])),
        "inference_seconds": manifest.get("inference_seconds", run_info.get("sampling_seconds")),
        "timing_policy": run_info.get("timing_policy", "inspect_utils: sampling includes progress hook; no warmup recorded" if schema == "consolidated" and manifest else "unknown"),
        "peak_allocated_bytes": run_info.get("memory", {}).get("peak_allocated_bytes"),
        "source_files": [{"path": str(item), "sha256": _sha256(item)} for item in source_files],
        "manifest": manifest, "run_info": run_info,
    }
    return arrays, meta

METRICS = ("magnitude_nmse", "complex_nmse", "magnitude_rmse", "magnitude_psnr_db", "dc_relative_l2")


def export_pilot_inputs(path: str | Path, output: str | Path, *,
                        phase_check: bool = False, tolerance: float = 3e-5) -> Path:
    """Prepare legacy pickle inputs from saved arrays, without re-running MRI inference.

    Both native and phase-check exports have an UNSHIFTED mask.pkl. The .pt
    suffixes preserve the existing runner CLI; contents are pickle {'img': ...},
    not torch.save archives. Source NPZ and manifest are never changed.
    """
    arrays, meta = load_saved(path)
    checks = validate_saved_inputs(arrays, tolerance=tolerance)
    reference = arrays["reference"].astype(np.complex64)
    if reference.shape[0] != 1:
        raise ValueError("Export requires exactly one saved pair [1,2,H,W]")
    indices = arrays.get("slice_indices")
    if indices is None:
        raise ValueError("Slice IDs missing: keep the source manifest.json beside the NPZ")
    indices = np.asarray(indices).reshape(-1)
    if len(indices) != 2 or indices[1] != indices[0] + 1:
        raise ValueError("Export requires two adjacent slice IDs")
    masks = np.broadcast_to(arrays["mask"], reference.shape)
    if not np.array_equal(masks[0, 0], masks[0, 1]):
        raise ValueError("A single mask.pkl cannot encode different masks for the two slices")
    mask = np.array(masks[0, 0], dtype=np.float32, copy=True)
    h, w = reference.shape[-2:]
    if h != w:
        raise ValueError("The current runner expects square images")
    scales = np.asarray(arrays.get("scale", np.ones((1, 2, 1, 1))), dtype=np.float32)
    if scales.shape == (2,):
        scales = scales.reshape(1, 2, 1, 1)
    if scales.shape != (1, 2, 1, 1) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Saved scale must be positive finite [1,2,1,1] or [2]")
    image = reference * scales
    expected_reference = reference
    expected_observed = arrays["observed"]
    if phase_check:
        if h % 2 or w % 2:
            raise ValueError("Checkerboard/half-grid shift equivalence requires even H and W")
        phase = (1 - 2 * (np.indices((h, w)).sum(axis=0) % 2)).astype(np.float32)
        image = image * phase
        expected_reference = reference * phase
        mask = np.fft.fftshift(mask, axes=AXES).copy()
        expected_observed = np.fft.fftshift(expected_observed, axes=AXES)
    # Verify the normalization the existing runner will apply to these pickles.
    component_max = np.maximum(np.abs(image.real), np.abs(image.imag)).max(axis=AXES, keepdims=True)
    if not np.isfinite(image).all() or not np.isfinite(component_max).all() or (component_max <= 0).any():
        raise ValueError("Exported images cannot be represented as finite normalized complex64")
    roundtrip = image / component_max
    export_checks = {
        "reference_relative_l2": relative_l2(roundtrip, expected_reference),
        "observed_relative_l2": relative_l2(mask * np.fft.fft2(roundtrip, axes=AXES), expected_observed),
        "magnitude_reference_relative_l2": relative_l2(np.abs(roundtrip), np.abs(reference)),
        "magnitude_zero_filled_relative_l2": relative_l2(
            np.abs(np.fft.ifft2(mask * np.fft.fft2(roundtrip, axes=AXES), axes=AXES)),
            np.abs(arrays["zero_filled"])),
    }
    if any(not np.isfinite(value) or value > tolerance for value in export_checks.values()):
        raise ValueError(f"Export would change the saved measurement/normalization: {export_checks}")
    output = Path(output).expanduser().resolve()
    # The old failing helper may have left an empty directory; it is safe to reuse.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Use a new or empty input directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    image_files = []
    for slot, slice_id in enumerate(indices):
        destination = output / f"slice{int(slice_id)}.pt"
        with destination.open("xb") as handle:
            pickle.dump({"img": np.ascontiguousarray(image[0, slot], dtype=np.complex64)},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        image_files.append(destination.name)
    with (output / "mask.pkl").open("xb") as handle:
        pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(output / "prepared_inputs.json", {
        "source_files": meta["source_files"], "source_dc_weight": meta["dc_weight"],
        "phase_check": phase_check, "mask_layout": "unshifted", "fft_norm": "backward",
        "slice_indices": [int(i) for i in indices], "image_files": image_files,
        "image_encoding": "pickle dictionary with complex64 img; .pt legacy suffix",
        "scale_restored": "scale" in arrays, "input_checks": checks, "export_checks": export_checks,
        "mask_file": "mask.pkl", "note": "No reconstruction was run; choose DC weight explicitly for the next experiment.",
    })
    print(f"Prepared inputs: {output}")
    print(f"  {image_files[0]}, {image_files[1]}, mask.pkl, prepared_inputs.json")
    print(f"  phase_check={phase_check}; use --mask_layout unshifted with run_exp")
    return output


def mask_shifted_and_diagnostic(path: str | Path, name: str) -> Path:
    """Compatibility wrapper for the old helper; no manifest['input'] dependency.

    A relative name is resolved beside the source run, as in the old helper.
    New callers can use export_pilot_inputs(..., phase_check=True) directly.
    """
    source = Path(path).expanduser().resolve()
    source_root = source if source.is_dir() else source.parent
    destination = Path(name).expanduser()
    if not destination.is_absolute():
        destination = source_root.parent / destination
    return export_pilot_inputs(path, destination, phase_check=True)

def stage_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for stage in STAGES:
        group = [row for row in rows if row["stage"] == stage]
        summary = {"stage": stage, "n_slice_rows": len(group)}
        for name in METRICS:
            values = np.asarray([row[name] for row in group], dtype=float)
            # Do not silently drop +/-inf PSNR (exact reconstruction/zero target).
            with np.errstate(invalid="ignore"):
                summary[name + "_mean"] = float(np.mean(values))
            summary[name + "_nonfinite_count"] = int((~np.isfinite(values)).sum())
        result.append(summary)
    return result

def analyze_run(path: str | Path, *, output_dir: str | Path | None = None,
                plot: bool = False, tolerance: float = 3e-5) -> dict[str, Any]:
    """Compatibility API used by inspect_utils.run_pilot(). No inference or input writes."""
    arrays, meta = load_saved(path)
    checks = validate_saved_inputs(arrays, tolerance=tolerance)
    rows = compute_metrics(arrays)
    out = Path(meta["npz_path"]).parent if output_dir is None else Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "metrics.csv", rows)
    summaries = stage_summary(rows)
    _write_csv(out / "stage_summary.csv", summaries)
    summary = {**meta, "input_checks": checks, "num_metric_rows": len(rows),
               "fft": "unshifted/backward", "psnr_peak": "per-reference-slice magnitude maximum",
               "aggregation": "arithmetic mean across saved batch/slice rows; not independent subjects",
               "zero_reference_policy": "relative errors are 0 for identical zeros, otherwise inf",
               "stages": summaries}
    _write_json(out / "metrics_summary.json", summary)
    if plot:
        for batch in range(arrays["reference"].shape[0]):
            plot_run(arrays, out / ("comparison.png" if batch == 0 else f"comparison_batch{batch}.png"), pair_index=batch)
    return {**summary, "metrics": rows, "output_dir": str(out)}


def compare_inputs(base: dict, candidate: dict, tolerance: float) -> dict:
    """Detect equivalent saved measurements, including the even-grid phase probe."""
    if base["reference"].shape != candidate["reference"].shape:
        return {"relation": "different_shape"}
    shape = base["reference"].shape
    bm = np.broadcast_to(base["mask"], shape)
    cm = np.broadcast_to(candidate["mask"], shape)
    result = {
        "magnitude_reference_relative_l2": relative_l2(np.abs(candidate["reference"]), np.abs(base["reference"])),
        "magnitude_zero_filled_relative_l2": relative_l2(np.abs(candidate["zero_filled"]), np.abs(base["zero_filled"])),
        "reference_relative_l2": relative_l2(candidate["reference"], base["reference"]),
        "observed_relative_l2": relative_l2(candidate["observed"], base["observed"]),
        "same_mask": bool(np.array_equal(bm, cm)),
        "relation": "different_or_unverified",
    }
    if result["same_mask"] and max(result["reference_relative_l2"], result["observed_relative_l2"]) <= tolerance:
        result["relation"] = "same_inputs"
    elif all(n % 2 == 0 for n in shape[-2:]):
        phase = 1 - 2 * (np.indices(shape[-2:]).sum(axis=0) % 2)
        result["checkerboard_reference_relative_l2"] = relative_l2(candidate["reference"], base["reference"] * phase)
        result["shifted_observed_relative_l2"] = relative_l2(candidate["observed"], np.fft.fftshift(base["observed"], axes=AXES))
        result["shifted_mask_equal"] = bool(np.array_equal(cm, np.fft.fftshift(bm, axes=AXES)))
        if (result["shifted_mask_equal"] and max(result["checkerboard_reference_relative_l2"],
                result["shifted_observed_relative_l2"], result["magnitude_reference_relative_l2"],
                result["magnitude_zero_filled_relative_l2"]) <= tolerance):
            result["relation"] = "checkerboard_equivalent"
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "미기록"
    return f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def write_report(results: list[dict], comparisons: list[dict], out: Path) -> None:
    lines = ["# FP reconstruction 파일별 비교", "",
             "저장된 NPZ에서 다시 계산한 결과입니다. 각 실행·seed를 따로 표시하고, stage 요약은 해당 파일의 slice 행을 같은 가중치로 평균했습니다.", "",
             "| 실행 | α | seed | 정밀도 | ZF NMSE | coarse NMSE | 최종 NMSE | 최종/ZF | 최종 PSNR(dB) | 최종 DC relative L2 |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    
    for result in results:
        stages = {row["stage"]: row for row in result["stages"]}
        zf, coarse, final = (stages[stage]["magnitude_nmse_mean"] for stage in STAGES)
        change = ("비교가 불가능합니다(유한하지 않은 지표)" if not np.isfinite([final, zf]).all()
                  else "낮습니다" if final < zf else "높습니다" if final > zf else "같습니다")
        reduction = 100 * (1 - final / coarse) if coarse > 0 else None
        runtime = "미기록" if result['inference_seconds'] is None else f"{_fmt(result['inference_seconds'])}초"
        lines.append(f"- {result['label']}: 최종 magnitude NMSE는 자체 zero-filled보다 {change}. "
                     f"coarse 대비 최종 NMSE 감소율은 {_fmt(reduction)}%입니다. "
                     f"저장된 sampling 시간: {runtime}.")
    lines += ["", "## 비교 조건 확인", ""]
    for item in comparisons:
        lines.append(f"- {item['label']} ↔ {results[0]['label']}: `{item['relation']}`. "
                     f"설정 차이: {', '.join(item['configuration_differences']) or '확인된 차이 없음'}. "
                     f"미확인 설정: {', '.join(item['configuration_unknown']) or '없음'}.")
    lines += ["", "## 지표 정의와 해석 범위", "",
              "- Magnitude NMSE = Σ(|복원|−|reference|)² / Σ|reference|². 낮을수록 좋습니다.",
              "- Complex NMSE = Σ|복원−reference|² / Σ|reference|². 각 실행에 저장된 같은 convention의 reference를 사용합니다.",
              "- PSNR은 slice별 reference magnitude 최댓값을 peak로 계산한 dB입니다. 요약은 slice별 PSNR의 산술평균입니다.",
              "- DC relative L2 = ||M FFT(복원)−observed||₂ / ||observed||₂. 낮은 값만으로 비관측 영역의 복원 품질을 보장하지 않습니다.",
              "- FFT는 unshifted/backward로 검증했습니다. 파일에 저장된 정규화 스케일을 임의로 다시 맞추지 않았습니다.",
              "- 이 파이프라인의 reference는 결합한 complex image이며 fastMRI reconstruction_rss를 평가 target으로 사용한 것이 아닙니다. 측정은 결합 영상에서 합성한 single-coil 측정입니다.",
              "- 인접 slice와 여러 stage를 독립 환자 수나 독립 반복 수로 세지 않습니다. 이 표에는 신뢰구간·유의성 검정을 적용하지 않았습니다.",
              "- checkerboard_equivalent는 입력·관측·mask의 변환 관계를 확인했다는 뜻입니다. checkpoint 학습 convention의 적합성을 입증하지는 않습니다.",
              "- 서로 다른 입력이나 checkpoint/seed/steps/정밀도 설정은 DC 가중치만의 효과로 해석할 수 없습니다.",
              "- runtime은 저장된 실행 단위 값입니다. warmup 없는 단일 측정이며, inspect_utils의 progress hook 포함 여부 등 timing policy는 provenance.json에 보존했습니다.",
              "- coarse/final을 GT에 맞춰 재정규화하지 않았습니다. SSIM은 이 도구에서 계산하지 않았습니다.",
              "", "## 근거 파일", "",
              "- per_slice_metrics.csv: 실행·slice·stage별 지표", "- comparison_summary.csv: 실행·stage별 평균과 기록된 설정",
              "- provenance.json: 원본 경로/SHA256, manifest, 입력 일치 검사, 실행 간 비교 조건",
              "- figures/: --plot 사용 시 비교 그림. 같은 실행의 모든 stage에서 동일한 magnitude/error 색 범위를 사용합니다.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(results)), 4.7), constrained_layout=True)
    try:
        for index, (stage, color) in enumerate(zip(STAGES, ("#818b9c", "#e29c45", "#337bb7"))):
            values = [next(row["magnitude_nmse_mean"] for row in result["stages"] if row["stage"] == stage) for result in results]
            bars = ax.bar(x + (index - 1) * .25, values, .25, label=stage, color=color)
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8)
        ax.set_xticks(x, [f"Run {i+1}" for i in x])
        ax.set_ylabel("Mean magnitude NMSE (lower is better)")
        ax.set_title("Saved FP reconstruction metrics")
        finite_values = [row["magnitude_nmse_mean"] for result in results for row in result["stages"]
                         if np.isfinite(row["magnitude_nmse_mean"])]
        ax.set_ylim(0, max([1e-6, *finite_values]) * 1.35)
        ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0, -.22, "Run IDs follow comparison_summary.csv; adjacent slices are not independent subjects.",
                transform=ax.transAxes, fontsize=8)
        fig.savefig(out / "nmse_comparison.png", dpi=180)
        fig.savefig(out / "nmse_comparison.svg")
    finally:
        plt.close(fig)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="one or more result NPZ files or run directories")
    parser.add_argument("--out", type=Path, help="NEW analysis directory; default: ./fp_metrics_TIMESTAMP")
    parser.add_argument("--prepare-inputs", type=Path, help="export slice*.pt + unshifted mask.pkl to a NEW or empty directory; no reconstruction")
    parser.add_argument("--phase-check", action="store_true", help="with --prepare-inputs: multiply images by checkerboard AND shift the sampling mask")
    parser.add_argument("--labels", nargs="+", help="optional labels, one for each input path")
    parser.add_argument("--dc-weight", type=float, help="user-reported alpha when manifests are absent; conflicts with saved alpha are rejected")
    parser.add_argument("--plot", action="store_true", help="save PNG/SVG figures without opening a GUI")
    parser.add_argument("--tolerance", type=float, default=3e-5)
    args = parser.parse_args(argv)
    if args.labels and len(args.labels) != len(args.paths):
        parser.error("--labels must have one label per input path")
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be finite and nonnegative")
    if args.dc_weight is not None and (not np.isfinite(args.dc_weight) or not 0 <= args.dc_weight <= 1):
        parser.error("--dc-weight must be finite and between 0 and 1")
    if args.phase_check and args.prepare_inputs is None:
        parser.error("--phase-check requires --prepare-inputs")
    if args.prepare_inputs is not None:
        if len(args.paths) != 1 or args.out is not None or args.labels or args.dc_weight is not None or args.plot:
            parser.error("Input export takes one saved run and --prepare-inputs [--phase-check]; omit analysis options")
        export_pilot_inputs(args.paths[0], args.prepare_inputs, phase_check=args.phase_check,
                            tolerance=args.tolerance)
        return 0
    selected = []
    for index, path in enumerate(args.paths):
        files = resolve_runs(path)
        for file in files:
            label = args.labels[index] if args.labels else Path(path).expanduser().stem
            if len(files) > 1:
                label += ":" + file.parent.parent.name
            selected.append((file, label))
    if len({path for path, _ in selected}) != len(selected):
        parser.error("the same output file was selected more than once")
    # Validate every run before creating deliverables. Reading never invokes a sampler.
    loaded = [load_saved(path) for path, _ in selected]
    for _, meta in loaded:
        meta["dc_weight_source"] = "manifest" if meta["dc_weight"] is not None else "unknown"
        if args.dc_weight is not None:
            if meta["dc_weight"] is not None and not np.isclose(meta["dc_weight"], args.dc_weight, rtol=0, atol=1e-12):
                parser.error(f"--dc-weight conflicts with the manifest for {meta['npz_path']}")
            if meta["dc_weight"] is None:
                meta["dc_weight"] = args.dc_weight
                meta["dc_weight_source"] = "user_reported_cli"
    checks = [validate_saved_inputs(arrays, tolerance=args.tolerance) for arrays, _ in loaded]
    out = (args.out or Path.cwd() / f"fp_metrics_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}Z").expanduser().resolve()
    if out.exists():
        parser.error(f"analysis directory already exists; choose a new --out: {out}")
    out.mkdir(parents=True)
    results, rows, summary_rows, comparisons = [], [], [], []
    for index, (((path, label), (arrays, meta)), check) in enumerate(zip(zip(selected, loaded), checks), 1):
        metrics = compute_metrics(arrays)
        result = {**meta, "label": label, "run_id": index, "input_checks": check,
                  "stages": stage_summary(metrics)}
        results.append(result)
        identity = {"run_id": index, "label": label, "seed": meta["seed"],
                    "condition": meta["condition"], "dc_weight": meta["dc_weight"], "precision": meta["precision"]}
        rows.extend({**identity, **row} for row in metrics)
        summary_rows.extend({**identity, **row} for row in result["stages"])
        comparison = compare_inputs(loaded[0][0], arrays, args.tolerance)
        difference, unknown = [], []
        for key in ("checkpoint_sha256", "model_config", "seed", "coarse_steps", "refine_steps", "precision", "num_samples", "dc_weight"):
            left, right = loaded[0][1][key], meta[key]
            if left is None or right is None:
                unknown.append(key)
            elif str(left) != str(right):
                difference.append(key)
        comparisons.append({"run_id": index, "label": label, **comparison,
                            "configuration_differences": difference, "configuration_unknown": unknown})
        print(f"Run {index}: {label} | alpha={meta['dc_weight']} | seed={meta['seed']} | {path}")
        for stage in result["stages"]:
            print(f"  {stage['stage']:14s} NMSE={stage['magnitude_nmse_mean']:.6g} "
                  f"PSNR={stage['magnitude_psnr_db_mean']:.4g} dB DC={stage['dc_relative_l2_mean']:.3e}")
        if comparison["relation"] not in ("same_inputs", "checkerboard_equivalent"):
            print("  NOTE: input equivalence to Run 1 was NOT confirmed.")
    _write_csv(out / "per_slice_metrics.csv", rows)
    _write_csv(out / "comparison_summary.csv", summary_rows)
    _write_json(out / "provenance.json", {"runs": results, "input_comparisons": comparisons,
                "analyzer_sha256": _sha256(Path(__file__)), "numpy_version": np.__version__})
    write_report(results, comparisons, out)
    if args.plot:
        try:
            import matplotlib  # optional: numerical results remain available without it
        except ImportError:
            print("matplotlib unavailable: CSV/report saved; install matplotlib for figures.")
        else:
            figures = out / "figures"
            figures.mkdir()
            plot_summary(results, figures)
            for index, (arrays, _) in enumerate(loaded, 1):
                for batch in range(arrays["reference"].shape[0]):
                    plot_run(arrays, figures / f"run_{index:02d}_batch_{batch}.png", pair_index=batch)
    with zipfile.ZipFile(out / "analysis_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(out.rglob("*")):
            if file.is_file() and file.name != "analysis_bundle.zip":
                archive.write(file, file.relative_to(out))
    print(f"Saved: {out}") 