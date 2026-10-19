"""
Paired FP / GPTQ-w4 reconstruction runner. See README.md for checking contracts.
"""


from __future__ import annotations
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import inspect
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
import pickle
import torch
import numpy as np


from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from unet_gptq.domain.data_consistency import validate_dc_weight
from unet_gptq.diffusion.diagnostics import SamplingDiagnostics
from unet_gptq.diffusion.model_factory import resolve_model_config

from unet_gptq.domain.forward_operator import channels_to_img, img_to_channels
from unet_gptq.diffusion.samplers import reconstruct_pair

STAGES = ("coarse", "refined")
QUALITY_KEYS = ("magnitude_nmse", "magnitude_rmse", "psnr_db", "dc_relative_l2")

def sha256_file(path:str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalize_image(image: Any) -> tuple[np.ndarray, float]:
    """Preserve the original max(real/imag component) scaling, per slice."""
    value = np.asarray(image)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
        raise ValueError("Each input img must be a numeric [H,W] array.")
    value = value.astype(np.complex64)
    if not np.isfinite(value).all():
        raise ValueError("Input img contains NaN or infinity.")
    scale = float(max(np.abs(value.real).max(), np.abs(value.imag).max()))
    if scale == 0:
        raise ValueError("All-zero img has an undefined normalization scale.")
    return (value / scale).astype(np.complex64), scale


def validate_mask(mask: Any, shape: tuple[int, int], layout: str) -> np.ndarray:
    value = np.asarray(mask)
    if value.shape == (1, 1, *shape):
        value = value[0, 0]
    if value.shape != shape:
        raise ValueError(f"Mask must have shape {shape} or {(1, 1, *shape)}, got {value.shape}")
    if np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError(f"Mask must be finite and real, got {value.dtype}")
    if not np.isin(value, [0, 1]).all() or not value.any():
        raise ValueError(f"A sampling mask contract requires binary {0, 1}")
    if layout == "centered":
        value = np.fft.ifftshift(value, axes=(-2, -1))
    elif layout != "unshifted":
        raise ValueError(f"unknown mask layout: {layout}")
    return np.array(value, dtype=np.float32, copy=True)


    

@dataclass(frozen=True)
class Case:
    reference: np.ndarray # [1,2,H,W] complex64, normalized separately per slice
    observed: np.ndarray # M * FFT(reference)
    mask: np.ndarray # [1,2,H,W] real binary, unshifted
    scales: tuple[float, float]


def load_case(args: argparse.Namespace) -> tuple[Case, dict]:
    root = Path(args.data_root)
    paths = [root / (args.file1 or f"{args.image}_{args.slice_index}.pt"),
             root / (args.file2 or f"{args.image}_{args.slice_index}_2.pt")]
    images, scales = [], []
    for path in paths:
        # These are the user's legacy pickle files, not torch.save archives.
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or "img" not in payload:
            raise ValueError(f"{path} must contain a pickle dictionary with key 'img'.")
        image, scale = normalize_image(payload["img"])
        if image.shape != (args.image_size, args.image_size):
            raise ValueError(f"{path}: expected {(args.image_size,) * 2}, got {image.shape}.")
        images.append(image)
        scales.append(scale)
    with Path(args.mask_path).open("rb") as stream:
        mask = validate_mask(pickle.load(stream), images[0].shape, args.mask_layout)
    reference = np.stack(images)[None]
    mask = np.broadcast_to(mask, reference.shape).copy()
    observed = (np.fft.fft2(reference, axes=(-2, -1), norm="backward") * mask).astype(np.complex64)
    case = Case(reference, observed, mask, tuple(scales))
    metadata = {
        "image_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths],
        "mask_file": {"path": str(Path(args.mask_path).resolve()), "sha256": sha256_file(args.mask_path)},
        "normalization": "per-slice max(abs(real), abs(imag))", "scales": scales,
        "fft": {"layout": "unshifted", "norm": "backward", "axes": [-2, -1]},
        "input_mask_layout": args.mask_layout, "effective_mask_sha256": array_hash(mask),
        "observed_sha256": array_hash(observed), "reference_sha256": array_hash(reference),
        "sampled_fraction": float(mask.mean()),
        "reference_kind": "normalized complex input pair; magnitude is not multicoil RSS",
    }
    return case, metadata

def validate_quant_manifest(manifest: dict, fp_hash: str, w4_hash: str) -> None:
    expected = {
        "schema_version": 1, "method": "gptq", "weight_bits": 4, "activation_quantization": False, "fp_checkpoint_sha256": fp_hash, "w4_checkpoint_sha256": w4_hash
    }

    for key, value in expected.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise ValueError(f"w4 manifest {key!r} must be {value!r}")
    
    if fp_hash == w4_hash:
        raise ValueError("FP and W4 ckpt bytes are identical")
    if manifest.get("format") not in ("dequantized_state_dict", "custom"):
        raise ValueError("W4 format must be dequanatized_state_dict or custom")
    names = manifest.get("quantized_parameters")
    
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("List the actual quantized param names in the W4 manifest")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate quantized param names")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict) or not calibration.get("description"):
        raise ValueError("W4 manifest requires calibration.description - data split and timestep selection")
    verification = manifest.get("verification")
    if (verification is not None or calibration.get("artifact_verification_required")) and (
        not isinstance(verification, dict) or verification.get("status") != "passed"
    ):
        raise ValueError("W4 artifact verification has not passed")

def relative_l2(error: np.ndarray, reference: np.ndarray) -> float | None:
    denominator = float(np.sum(np.abs(reference.astype(np.complex128)) ** 2))
    numerator = float(np.sum(np.abs(error.astype(np.complex128)) ** 2))
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return math.sqrt(numerator / denominator)


def validate_outputs(outputs: dict, case: Case) -> None:
    for stage in STAGES:
        value = outputs.get(stage)
        if not isinstance(value, np.ndarray) or not np.iscomplexobj(value):
            raise ValueError(f"{stage} must be a complex NumPy array [B,2,H,W].")
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[1:] != case.reference.shape[1:]:
            raise ValueError(f"Invalid {stage} shape: {value.shape}; expected [B,2,H,W].")
        if not np.isfinite(value).all():
            raise ValueError(f"{stage} contains NaN or infinity.")



def quality_rows(outputs: dict, case: Case, seed: int, condition: str) -> list[dict]:
    rows = []
    for stage in STAGES:
        for sample_id, sample in enumerate(outputs[stage]):
            for pair_index, estimate in enumerate(sample):
                target = case.reference[0, pair_index].astype(np.complex128)
                estimate = estimate.astype(np.complex128)
                magnitude = np.abs(target)
                difference = np.abs(estimate) - magnitude
                mse = float(np.mean(difference**2))
                peak = float(magnitude.max())
                psnr = 10 * math.log10(peak**2 / mse) if mse > 0 else None
                measured = case.observed[0, pair_index]
                residual = case.mask[0, pair_index] * np.fft.fft2(estimate) - measured
                rows.append({
                    "seed": seed, "condition": condition, "stage": stage, 
                    "sample_index": sample_id, "pair_index": pair_index,
                    "magnitude_nmse": float(np.sum(difference**2)) / np.sum(magnitude**2),
                    "magnitude_rmse": math.sqrt(mse), "psnr_db": psnr, "psnr_is_posinf": mse == 0, "data_range": peak, "dc_relative_l2": relative_l2(residual, measured)
                })
    
    return rows

def compare_outputs(fp: dict, other: dict, *, atol: float = 0, rtol: float = 0) -> dict:
    result = {}
    for stage in STAGES:
        a, b = fp[stage], other[stage]
        if a.shape != b.shape:
            raise ValueError(f"{stage} shape differs between runs: {a.shape} vs {b.shape}.")
        difference = b.astype(np.complex128) - a.astype(np.complex128)
        result[stage] = {"allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
                         "max_abs_error": float(np.abs(difference).max()),
                         "complex_rmse": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
                         "relative_l2": relative_l2(difference, a)}
    return result


def paired_rows(fp_rows: list[dict], w4_rows: list[dict], eligible: bool) -> list[dict]:
    key = lambda row: (row["stage"], row["sample_index"], row["pair_index"])
    lookup = {key(row): row for row in fp_rows}
    if set(lookup) != {key(row) for row in w4_rows}:
        raise ValueError("FP/W4 sample indices or stages do not match.")
    result = []
    for w4 in w4_rows:
        fp = lookup[key(w4)]
        row = {name: w4[name] for name in ("seed", "stage", "sample_index", "pair_index")}
        row["eligible"] = eligible
        for name in QUALITY_KEYS:
            row[name + "_fp"] = fp[name]
            row[name + "_w4"] = w4[name]
            row["delta_" + name] = None if fp[name] is None or w4[name] is None else w4[name] - fp[name]
        result.append(row)
    return result


def summarize_pairs(rows: list[dict]) -> dict:
    """Average within each seed first; samples/slices are not independent seeds."""
    summary = {}
    for stage in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage and row["eligible"]]
        metrics = {}
        for metric in QUALITY_KEYS:
            name = "delta_" + metric
            groups: dict[int, list] = {}
            for row in stage_rows:
                groups.setdefault(row["seed"], []).append(row[name])
            # A partially undefined seed (e.g. infinite PSNR) is not silently averaged.
            means = {str(seed): float(np.mean(values)) for seed, values in groups.items()
                     if all(value is not None for value in values)}
            values = list(means.values())
            metrics[name] = {"by_seed": means, "n_seeds": len(values),
                             "n_undefined_seeds": len(groups) - len(values),
                             "mean": float(np.mean(values)) if values else None,
                             "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        summary[stage] = metrics
    return {"delta_direction": "W4 - FP", "aggregation": "mean over samples and pair images within seed, then over seeds",
            "inference_scope": "seed variability for one input pair; not patient-level confidence intervals",
            "stages": summary}

            
def run_experiment(case: Case, backend: Callable, args: argparse.Namespace, out: Path) -> dict:
    """
    Orchestration is backend-independant, failure and pairing logic are allowed for tests.
    """

    np.savez_compressed(out / "inputs.npz", reference = case.reference, k_obs=case.observed, mask=case.mask, scales=np.asarray(case.scales))
    metrics, pairs, runs, failures = [], [], [], []
    audits = []

    
    for index, seed in enumerate(args.seeds):
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir()
        order = list(args.conditions)

        if index % 2:
            order.reverse()
        
        available, rows_by_condition, metadata = {}, {}, {}
        
        for condition in order + (["fp_repeat"] if args.check_fp_repeat else []):
            if condition == "fp_repeat" and "fp" not in available:
                continue
            run_dir = seed_dir / condition
            run_dir.mkdir()
            print(f"[seed = {seed}] {condition}", flush = True)

            try:
                outputs, info = backend("fp" if condition == "fp_repeat" else condition, seed)
                validate_outputs(outputs, case)
                np.savez_compressed(run_dir / "outputs.npz", **{stage: outputs[stage] for stage in STAGES})
                info = dict(info, seed=seed, condition=condition, status="ok")
                diagnostic_rows = info.pop("_diagnostic_rows", None)
                if diagnostic_rows is not None:
                    for filename, rows in diagnostic_rows.items():
                        write_csv(run_dir / filename, [dict(row, seed=seed, condition=condition) for row in rows])
                write_json(run_dir / "run.json", info)
                runs.append(info)
                available[condition], metadata[condition] = outputs, info
                if condition != "fp_repeat":
                    rows_by_condition[condition] = quality_rows(outputs, case, seed, condition)
                    metrics.extend(rows_by_condition[condition])
            except Exception as e:
                failure = {"seed": seed, "condition": condition, "status": "failed", "error_type": type(e).__name__, "error":str(e)}
                failures.append(failure)
                write_json(run_dir / "run.json", failure)
                (run_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"    failed: {type(e).__name__}:{e}", file=sys.stderr, flush=True)
        audit = {"seed": seed, "order": order, "fp_repeat_passed": None, "global_rng_start_states_match": None, "global_rng_end_states_match": None, "per_step_noise_verified": False, "eligible": False}
        if "fp" in available and "fp_repeat" in available:
            try:
                comparison = compare_outputs(available["fp"], available["fp_repeat"], atol=args.repeat_atol, rtol=args.repeat_rtol)
                hashes = [metadata[name].get("rng_end_sha256") for name in ("fp", "fp_repeat")]
                starts = [metadata[name].get("rng_start_sha256") for name in ("fp", "fp_repeat")]
                audit["fp_repeat"] = comparison
                audit["fp_repeat_passed"] = (all(item["allclose"] for item in comparison.values())
                                                and bool(hashes[0]) and hashes[0] == hashes[1]
                                                and bool(starts[0]) and starts[0] == starts[1]
                )
            
            except ValueError as e:
                audit["fp_repeat_passed"] = False
                audit["fp_repeat_error"] = str(e)

        if "fp" in available and "w4" in available:
            try:
                hashes = [metadata[name].get("rng_end_sha256") for name in ("fp", "w4")]
                starts = [metadata[name].get("rng_start_sha256") for name in ("fp", "w4")]

                audit["global_rng_start_states_match"] = bool(starts[0]) and starts[0] == starts[1]
                audit["global_rng_end_states_match"] = bool(hashes[0]) and hashes[0] == hashes[1]
                audit["fp_w4_output_difference"] = compare_outputs(available["fp"],  available["w4"])


                audit["eligible"] = (audit["fp_repeat_passed"] is True
                                     and audit["global_rng_start_states_match"] is True
                                     and audit["global_rng_end_states_match"] is True
                )

                pairs.extend(paired_rows(rows_by_condition["fp"], rows_by_condition["w4"], audit["eligible"]))
            except ValueError as e:
                failures.append({"seed": seed, "condition": "pairing", "error": str(e)})


        audits.append(audit)
        write_json(seed_dir / "audit.json", audit)
        # ckpt progress
        write_csv(out / "metrics.csv", metrics)
        write_csv(out / "paired_deltas.csv", pairs)
        write_json(out / "progress.json", {"completed_seeds": len(audits), "audits": audits, "failures": failures})

    comparison_req = "w4" in args.conditions
    gates_failed = (comparison_req and any(not a["eligible"] for a in audits)) or (args.check_fp_repeat and any(a["fp_repeat_passed"] is not True for a in audits))

    summary = summarize_pairs(pairs)
    summary.update(status="failed" if failures or gates_failed else "completed", failures = failures, audits = audits, runs = runs,
                   note="Eligibility checks repetition and global RNG use, not explicit per-step noise"            
    )

    write_json(out / "summary.json", summary)
    return summary

def source_info(module: Any) -> dict:
    path = Path(inspect.getfile(module)).resolve()
    result = {"file": str(path), "sha256": sha256_file(path), "commit": None, "dirty": None}
    try:
        result["commit"] = subprocess.check_output(["git", "-C", str(path.parent), "rev-parse", "HEAD"],
                                                  stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        result["dirty"] = bool(subprocess.check_output(["git", "-C", str(path.parent), "status", "--porcelain"],
                                                      stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def import_callable(spec: str) -> Callable:
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Use module:function, got {spec!r}.")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable.")
    return function

class TorchBackend:

    def __init__(self, args: argparse.Namespace, case: Case, manifest: dict | None):
        # precede first torch / CUDA import in this process
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        
        self.torch, self.args, self.case, self.manifest = torch, args, case, manifest
        self.device = torch.device(args.device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("The runner must support a single CPU or CUDA device")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but is not available; use --device cpu")
            torch.cuda.set_device(self.device)
        
        torch.use_deterministic_algorithms(not args.allow_nondeterministic)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = not args.allow_nondeterministic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.factory_module = importlib.import_module(args.factory_module)
        self.factory = self.factory_module.create_model_and_two_diffusion
        self.sampler = import_callable(args.sampler)
        self.loader = import_callable(args.w4_loader) if args.w4_loader else None
        self.config = resolve_model_config(
            self.factory_module, model_options=args.model_options, image_size=args.image_size,
            timestep_respacing=args.timestep_respacing, use_fp16=args.use_fp16,
        )
        if manifest and manifest["format"] == "custom" and self.loader is None:
            raise ValueError("format=custom requires --w4_loader module:function.")
        if self.loader and (manifest is None or manifest["format"] != "custom"):
            raise ValueError(f"--w4_loader is only used with manifest format as custom")
        
    def seed(self, value: int) -> None:
        random.seed(value)
        np.random.seed(value)
        self.torch.manual_seed(value)

    def rng_state(self) -> dict:
        torch = self.torch
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    
    def restore_rng(self, state: dict) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        self.torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"]:
            self.torch.cuda.set_rng_state_all(state["torch_cuda"])
    
    def rng_hash(self) -> str:
        state = self.rng_state()
        digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
        for value in [state["torch_cpu"], *state["torch_cuda"]]:
            digest.update(value.cpu().numpy().tobytes())
        return digest.hexdigest()
    
    def load_state(self, path: str) -> dict:
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not state or any(not self.torch.is_tensor(x) for x in state.values()):
            raise ValueError(f"{path}: expected a tensor state_dict or {{'state_dict': ...}}.")
        return state
    
    def load_dequantized(self, model: Any, fp: dict) -> dict:
        torch = self.torch
        state = self.load_state(self.args.w4_model_path)
        names = set(self.manifest["quantized_parameters"])
        parameters = dict(model.named_parameters())
        if fp.keys() != state.keys() or not names <= parameters.keys():
            raise ValueError("FP /W4 state keys or declared quantized parameter names do not match the model")
        changed = []
        for name, before in fp.items():
            after = state[name]
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(f"W4 state shape/dtype changed for {name}; retain FP storage dtype for this format.")
            if not torch.isfinite(after).all():
                raise ValueError(f"W4 state contains nonfinite values in {name}.")
            equal = torch.equal(before, after)
            if name not in names and not equal:
                raise ValueError(f"Undeclared parameter/buffer changed: {name}.")
            if name in names:
                if not after.is_floating_point() or after.ndim < 2 or not name.endswith("weight"):
                    raise ValueError(f"{name} must be a floating-point Linear/Conv weight tensor.")
                if not equal:
                    changed.append(name)
        
        if not changed:
            raise ValueError(f"W4 has no changed quantized weights; refusing a silent FP-vs-FP comparison")
        
        model.load_state_dict(state, strict=True)

        return {"format": "dequantized_state_dict", "changed_quantized_parameters": changed,
                "declared_quantized_numel": sum(parameters[name].numel() for name in names),
                "total_parameter_numel": sum(value.numel() for value in parameters.values()),
                "packed_int4_execution": False,
                "verification": "state differences checked; GPTQ method/bitwidth are artifact-producer declarations"}
        
    def as_complex_numpy(self, value: Any) -> np.ndarray:
        torch = self.torch
        if not torch.is_tensor(value) or value.ndim != 4:
            raise ValueError("Sampler must return two tensors: complex [B,2,H,W] or real [B,4,H,W].")
        value = value.detach().cpu()
        if value.is_complex() and value.shape[1] == 2:
            return value.to(torch.complex64).numpy().copy()
        if not value.is_complex() and value.shape[1] == 4:
            array = value.float().numpy()
            return (array[:, 0::2] + 1j * array[:, 1::2]).astype(np.complex64)
        raise ValueError(f"Unsupported sampler output dtype/shape: {value.dtype}, {value.shape}.")

    def _build_model(self, condition: str):
        """Load a fresh condition; never quantize or reuse the FP reference in place."""
        torch, args = self.torch, self.args
        self.seed(args.model_init_seed)
        model, diffusion, diffusion2 = self.factory(**copy.deepcopy(self.config))
        fp = self.load_state(args.model_path)
        model.load_state_dict(fp, strict=True)
        quant_info = {"format": "original_fp", "packed_int4_execution": False}
        if condition == "w4" and self.manifest["format"] == "dequantized_state_dict":
            quant_info = self.load_dequantized(model, fp)
        if args.use_fp16:
            if not hasattr(model, "convert_to_fp16"):
                raise TypeError("Model does not include convert_to_fp16()")
            model.convert_to_fp16()
        model.to(self.device)
        if condition == "w4" and self.manifest["format"] == "custom":
            loaded = self.loader(
                model=model, checkpoint_path=Path(args.w4_model_path),
                manifest=copy.deepcopy(self.manifest), device=self.device,
            )
            if not isinstance(loaded, tuple) or len(loaded) != 2:
                raise TypeError("Custom loader must return (nn.Module, JSON-serializable metadata)")
            model, quant_info = loaded
            if not isinstance(model, torch.nn.Module) or not isinstance(quant_info, dict):
                raise TypeError("Invalid custom W4 loader result")
        model.eval()
        model.requires_grad_(False)
        return model, diffusion, diffusion2, quant_info

    def reconstruction_policy(self) -> dict:
        return {"fft_layout": "unshifted", "fft_norm": "backward",
                "mri_dtype": "complex64", "dc_weight": self.args.dc_weight,
                "dc_method": ("hard" if self.args.dc_weight == 1 else
                              "disabled" if self.args.dc_weight == 0 else "soft"),
                "mask": "binary; separate from dc_weight"}
    
    def __call__(self, condition: str, seed: int) -> tuple[dict, dict]:
        torch, args = self.torch, self.args
        model = diffusion = diffusion2 = reference_model = obs = mask = result = None
        diagnostic = replay = None
        
        try:
            load_start = time.perf_counter()
            model, diffusion, diffusion2, quant_info = self._build_model(condition)
            
            obs = torch.from_numpy(self.case.observed.copy()).to(self.device, dtype=torch.complex64)
            mask = torch.from_numpy(self.case.mask.copy()).to(self.device)
            dtypes: dict[str, int] = {}

            for parameter in model.parameters():
                dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
            
            parameter = None

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            
            load_seconds = time.perf_counter() - load_start

            sampler_options = dict(num_samples=args.num_samples, refine_steps=args.refine_steps,
                                   clip_denoised=args.clip_denoised, dc_weight=args.dc_weight)
            self.seed(seed)
            sampling_state = self.rng_state()
            
            rng_start = self.rng_hash()
            memory = {"baseline_allocated_bytes": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                memory["baseline_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
            start = time.perf_counter()
            
            with torch.enable_grad():
                result = self.sampler(model, diffusion, diffusion2, obs, mask, **sampler_options)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            seconds = time.perf_counter() - start
            rng_end = self.rng_hash()
            sampling_end_state = self.rng_state()
            if self.device.type == "cuda":
                memory["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
                memory["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ValueError("reconstruct_pair must return (coarse, refined).")
            outputs = {stage: self.as_complex_numpy(value) for stage, value in zip(STAGES, result)}
            info = {"sampling_seconds": seconds, "load_seconds": load_seconds,
                    "timing_policy": "no warmup; sampling only; synchronized; model loading and file IO excluded",
                    "rng_start_sha256": rng_start, "rng_end_sha256": rng_end,
                    "parameter_numel_by_dtype": dtypes, "use_fp16": args.use_fp16,
                    "activation_quantization": False, "quantization": quant_info,
                    "reconstruction_policy": self.reconstruction_policy(),
                    "memory": memory, "per_step_noise_verified": False,
                    "diagnostics": {"status": "disabled"}}
            # Validate metadata before considering a custom run successful.
            if args.diagnostics:
                # Reload to reset model/sampler state. Keep the reference model and
                # all hook/CPU logging costs outside the measured run and its peak memory.
                model = diffusion = diffusion2 = result = None
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    model, diffusion, diffusion2, _ = self._build_model(condition)
                    if condition == "w4":
                        reference_model, _, _, _ = self._build_model("fp")
                    diagnostic = SamplingDiagnostics(reference_model=reference_model, trajectory=condition)
                    self.restore_rng(sampling_state)
                    with torch.enable_grad():
                        replay = self.sampler(model, diffusion, diffusion2, obs, mask,
                                              diagnostics=diagnostic, **sampler_options)
                    replay_outputs = {stage: self.as_complex_numpy(value)
                                      for stage, value in zip(STAGES, replay)}
                    agreement = compare_outputs(outputs, replay_outputs,
                                                atol=args.repeat_atol, rtol=args.repeat_rtol)
                    if (not all(row["allclose"] for row in agreement.values())
                            or self.rng_hash() != rng_end):
                        raise ValueError("Diagnostic replay changed outputs or RNG use; results are not comparable.")
                    if not diagnostic.nfe or not diagnostic.dc_rows:
                        raise ValueError("Sampler did not emit denoiser/DC diagnostics.")
                    info["diagnostics"] = {
                        "status": "completed", "replay_matches_timed_run": True,
                        "nfe": diagnostic.nfe, "nfe_source": "diagnostic_replay",
                        "comparison": "common_input_on_w4_trajectory" if condition == "w4" else "not_compared",
                        "output_scope": "raw_model_output_before_DC",
                        "fp_reference_forwards": diagnostic.nfe if condition == "w4" else 0,
                    }
                    info["_diagnostic_rows"] = {
                        "denoiser_differences.csv": diagnostic.denoiser_rows,
                        "dc_residuals.csv": diagnostic.dc_rows,
                    }
                finally:
                    self.restore_rng(sampling_end_state)
            json.dumps(info, allow_nan=False)
            return outputs, info
        finally:
            model = diffusion = diffusion2 = reference_model = obs = mask = result = None
            diagnostic = replay = None
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def metadata(self) -> dict:
        torch = self.torch
        sources = {"factory": source_info(self.factory_module),
                   "sampler": source_info(importlib.import_module(self.sampler.__module__))}
        if self.loader:
            sources["w4_loader"] = source_info(importlib.import_module(self.loader.__module__))
        return {"python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "device": str(self.device),
                "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else platform.processor(),
                "deterministic_algorithms": not self.args.allow_nondeterministic,
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
                "factory_config": self.config, "sources": sources,
                "reconstruction_policy": self.reconstruction_policy()}

def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Original FP torch state_dict.")
    parser.add_argument("--w4_model_path", default="", help="Existing GPTQ-W4 artifact; no quantizer is run here.")
    parser.add_argument("--w4_manifest", default="", help="manifest.json from unet_gptq.train.calibrate_gptq.")
    parser.add_argument("--w4_loader", default="", help="module:function for manifest format=custom only.")
    parser.add_argument("--conditions", nargs="+", choices=("fp", "w4"), default=["fp", "w4"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--slice_index", type=int, default=0)
    parser.add_argument("--file1", default="", help="Override the first legacy filename; relative to data_root.")
    parser.add_argument("--file2", default="", help="Override the second legacy filename; relative to data_root.")
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--mask_layout", choices=("unshifted", "centered"), default="unshifted")
    parser.add_argument("--save_path", required=True, help="A NEW directory; existing output directories are rejected.")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--num_samples", type=int, default=1, help="Passed unchanged to the sampler; pilot default=1.")
    parser.add_argument("--refine_steps", type=int, default=20)
    parser.add_argument("--clip_denoised", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dc_weight", type=float, default=0.5, help="Shared soft-DC weight in [0,1]; 1 gives hard DC. Default 0.5 is a pilot value.")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True, help="Replay outside timing to log raw denoiser differences and post-DC residuals.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model_init_seed", type=int, default=0)
    parser.add_argument("--check_fp_repeat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat_atol", type=float, default=0.0)
    parser.add_argument("--repeat_rtol", type=float, default=0.0)
    parser.add_argument("--allow_nondeterministic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_fp16", action="store_true", help="Use the model's convert_to_fp16 for BOTH conditions.")
    parser.add_argument("--factory_module", default="improved_diffusion.script_util_duo")
    parser.add_argument("--sampler", default="unet_gptq.diffusion.samplers:reconstruct_pair")
    parser.add_argument("--model_options", default="{}", help="JSON overrides of the pretrained model/factory defaults.")
    parser.add_argument("--timestep_respacing", default="")
    return parser


def validate_args(args: argparse.Namespace) -> None:

    validate_dc_weight(args.dc_weight)
    if "fp" not in args.conditions or len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Conditions must include fp, without duplicates.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2 ** 32 for seed in [*args.seeds, args.model_init_seed]):
        raise ValueError("Use unique sampling seeds and integer seeds in [0,2**32).")
    if args.image_size < 1 or args.num_samples < 1 or args.refine_steps < 0 or args.slice_index < 0:
        raise ValueError("Invalid image_size, num_samples, refine_steps, or slice_index.")
    if any(not math.isfinite(value) or value < 0 for value in (args.repeat_atol, args.repeat_rtol)):
        raise ValueError("Repeat tolerances must be finite and nonnegative.")
    if not args.image and not (args.file1 and args.file2):
        raise ValueError("Specify --image, or both --file1 and --file2.")
    if "w4" in args.conditions and not (args.w4_model_path and args.w4_manifest):
        raise ValueError("W4 requires both --w4_model_path and --w4_manifest. Use --conditions fp for FP-only.")
    for name in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        if int(os.environ.get(name, "1")) > 1:
            raise ValueError("This runner is single-process; do not launch it with torchrun/mpirun.")


def main(argv: list[str] | None = None) -> int:
    args = create_argparser().parse_args(argv)
    validate_args(args)
    out = Path(args.save_path).resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Choose a new run directory.")
    case, input_info = load_case(args)
    fp_hash = sha256_file(args.model_path)
    quant_manifest = None
    if "w4" in args.conditions:
        quant_manifest = json.loads(Path(args.w4_manifest).read_text(encoding="utf-8"))
        validate_quant_manifest(quant_manifest, fp_hash, sha256_file(args.w4_model_path))
    backend = TorchBackend(args, case, quant_manifest)
    provenance = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "arguments": vars(args), "fp_checkpoint_sha256": fp_hash,
                  "input": input_info, "w4_artifact": quant_manifest,
                  "environment": backend.metadata(), "runner": source_info(sys.modules[__name__]),
                  "noise_contract": "same seed restored after each fresh model load; global Python/NumPy/Torch RNG only",
                  "per_step_noise_verified": False}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "manifest.json", provenance)
    try:
        summary = run_experiment(case, backend, args, out)
    except Exception as exc:
        write_json(out / "fatal_error.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    print(f"{summary['status']}: {out}", flush=True)
    return 0 if summary["status"] == "completed" else 2

if __name__ == "__main__":
    raise SystemExit(main())
