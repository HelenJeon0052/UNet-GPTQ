"""Bounded-memory, FP-teacher Conv2d GPTQ calibration and artifact verification.

This uses forward-noised reference images, not reverse-sampling/DC trajectories.
Every layer sees the same deterministic states of the unchanged FP teacher.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .conv_gptq import Conv2dGPTQ
from .gptq import GPTQConfig, _file_sha256, save_gptq_artifact, tensor_sha256


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    x_start: torch.Tensor  # CPU float32 [1,4,H,W], normalized adjacent slices
    provenance: dict


@dataclass(frozen=True)
class CalibrationStage:
    name: str
    diffusion: object
    timesteps: tuple[int, ...]  # local indices in this diffusion's schedule


def select_timesteps(available: int, count: int, explicit: Sequence[int] | None = None) -> tuple[int, ...]:
    """Select deterministic, endpoint-inclusive uniform local indices."""
    if available < 1 or count < 1:
        raise ValueError("available and count must be positive")
    if explicit is None:
        n = min(count, available)
        values = (available // 2,) if n == 1 else tuple(
            i * (available - 1) // (n - 1) for i in range(n)
        )
    else:
        values = tuple(explicit)
    if (not values or any(type(t) is not int or not 0 <= t < available for t in values)
            or len(values) != len(set(values))):
        raise ValueError(f"Timesteps must be unique integer indices in [0,{available})")
    return values


def model_timesteps(diffusion: object, local: torch.Tensor) -> torch.Tensor:
    """Match DiffuseRecon SpacedDiffusion._WrappedModel exactly, once.

    q_sample indexes the respaced alpha schedule; the U-Net receives original
    training indices, optionally scaled to the 1000-step embedding convention.
    """
    if hasattr(diffusion, "timestep_map"):
        mapping = torch.as_tensor(diffusion.timestep_map, device=local.device, dtype=local.dtype)
        result = mapping[local]
        if diffusion.rescale_timesteps:
            result = result.float() * (1000.0 / diffusion.original_num_steps)
        return result
    scale = getattr(diffusion, "_scale_timesteps", None)
    if not callable(scale):
        raise TypeError("Diffusion must expose timestep_map or _scale_timesteps")
    return scale(local)


def _state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected a tensor state_dict or {'state_dict': ...}")
    return state


def _parameter_name(module_name: str, parameter: str = "weight") -> str:
    return f"{module_name}.{parameter}" if module_name else parameter


def _select_layers(model, state, include, exclude, hessian_dtype, max_hessian_bytes):
    modules = dict(model.named_modules())
    for pattern in (*include, *exclude):
        if not any(_matches(name, (pattern,)) for name in modules):
            raise ValueError(f"Layer pattern matched no module: {pattern}")
    aliases: dict[int, list[str]] = {}
    for name, tensor in state.items():
        if tensor.numel():
            aliases.setdefault(tensor.untyped_storage().data_ptr(), []).append(name)
    selected, excluded = {}, []
    for name, layer in modules.items():
        direct = list(layer.named_parameters(recurse=False))
        if not direct:
            continue
        key = _parameter_name(name)
        reason = None
        if type(layer) is not nn.Conv2d:
            reason = "only_plain_nn_Conv2d_supported"
        elif include and not _matches(name, include):
            reason = "not_in_include_patterns"
        elif _matches(name, exclude):
            reason = "excluded_by_pattern"
        elif layer.groups != 1:
            reason = "grouped_or_depthwise_convolution"
        elif isinstance(layer.padding, str) or layer.padding_mode != "zeros":
            reason = "unsupported_padding"
        elif key not in state or not state[key].is_floating_point():
            reason = "missing_or_nonfloating_checkpoint_weight"
        elif len(aliases.get(state[key].untyped_storage().data_ptr(), [])) > 1:
            reason = "shared_checkpoint_storage"
        else:
            columns = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            if columns**2 * torch.empty((), dtype=hessian_dtype).element_size() > max_hessian_bytes:
                reason = "hessian_exceeds_memory_limit"
        if reason:
            excluded.append({"layer": name, "type": type(layer).__name__, "reason": reason,
                             "parameters": [_parameter_name(name, p) for p, _ in direct]})
        else:
            selected[name] = key
    return selected, excluded


def _matches(name, patterns):
    from fnmatch import fnmatchcase
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def _stage_metadata(stage):
    diffusion = stage.diffusion
    local = torch.tensor(stage.timesteps, dtype=torch.long)
    mapping = getattr(diffusion, "timestep_map", list(range(diffusion.num_timesteps)))
    alpha = torch.as_tensor(diffusion.alphas_cumprod, dtype=torch.float64)
    if (alpha.shape != (diffusion.num_timesteps,) or not torch.isfinite(alpha).all()
            or not ((alpha > 0) & (alpha <= 1)).all() or len(mapping) != diffusion.num_timesteps):
        raise ValueError(f"Invalid alpha schedule or timestep mapping for {stage.name}")
    return {"name": stage.name, "num_timesteps": diffusion.num_timesteps,
            "local_timesteps": list(stage.timesteps),
            "original_timesteps": [int(mapping[t]) for t in stage.timesteps],
            "model_timesteps": model_timesteps(diffusion, local).tolist(),
            "alphas_cumprod_sha256": tensor_sha256(alpha),
            "selected_alphas_cumprod": alpha[local].tolist(),
            "original_num_steps": getattr(diffusion, "original_num_steps", diffusion.num_timesteps),
            "rescale_timesteps": bool(getattr(diffusion, "rescale_timesteps", False))}


def _states(samples, stages, seed, device):
    for sample in samples:
        x0 = sample.x_start.to(device=device)
        for stage in stages:
            for timestep in stage.timesteps:
                payload = json.dumps([seed, sample.sample_id, stage.name, timestep]).encode()
                noise_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)
                generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                noise = torch.randn(sample.x_start.shape, generator=generator, dtype=torch.float32).to(device)
                local = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)
                xt = stage.diffusion.q_sample(x0, local, noise=noise)
                if xt.shape != x0.shape or not torch.isfinite(xt).all():
                    raise ValueError("q_sample returned invalid calibration states")
                yield xt, model_timesteps(stage.diffusion, local), {
                    "sample_id": sample.sample_id, "stage": stage.name, "local_timestep": timestep,
                    "noise_seed": noise_seed,
                }


def _prepare_model(model, device, use_fp16):
    teacher = copy.deepcopy(model)
    if use_fp16:
        if not hasattr(teacher, "convert_to_fp16"):
            raise TypeError("A16 requires model.convert_to_fp16(), as in run_exp.py")
        teacher.convert_to_fp16()
    return teacher.to(device).eval().requires_grad_(False)


@torch.no_grad()
def calibrate_conv2d_model(
    model: nn.Module, fp_checkpoint: str | Path, samples: Sequence[CalibrationSample],
    stages: Sequence[CalibrationStage], output_dir: str | Path, *, provenance: dict,
    config: GPTQConfig = GPTQConfig(), device: str = "cpu", use_fp16: bool = False,
    seed: int = 0, include: Sequence[str] = (), exclude: Sequence[str] = (),
    hessian_dtype: torch.dtype = torch.float32, max_hessian_bytes: int = 512 * 1024**2,
    patch_chunk_size: int = 1024, max_patches_per_batch: int = 4096,
) -> dict[str, Path]:
    """Create a NEW W4 artifact; neither model nor source checkpoint is mutated.

    One Hessian is live at a time. Replaying the FP teacher for every layer
    trades execution time for bounded curvature memory. Factorization needs
    several additional dense matrices beyond max_hessian_bytes.
    """
    fp_checkpoint, output_dir = Path(fp_checkpoint).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if output_dir == fp_checkpoint or fp_checkpoint.is_relative_to(output_dir):
        raise ValueError("Output must not contain or replace the FP checkpoint")
    if config.bits != 4 or not samples or not stages:
        raise ValueError("W4 calibration requires nonempty samples and stages")
    if hessian_dtype not in (torch.float32, torch.float64) or max_hessian_bytes < 1:
        raise ValueError("Use float32/float64 Hessians and a positive memory limit")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0,2**32)")
    if len({s.sample_id for s in samples}) != len(samples) or any(not s.sample_id for s in samples):
        raise ValueError("Calibration sample IDs must be nonempty and unique")
    for sample in samples:
        value = sample.x_start
        if (value.device.type != "cpu" or value.dtype != torch.float32 or value.ndim != 4
                or value.shape[:2] != (1, 4) or min(value.shape) < 1 or not torch.isfinite(value).all()):
            raise ValueError("Calibration x_start must be finite CPU float32 [1,4,H,W]")
    if len({s.name for s in stages}) != len(stages):
        raise ValueError("Calibration stage names must be unique")
    for stage in stages:
        select_timesteps(stage.diffusion.num_timesteps, 1, stage.timesteps)
    stage_info = [_stage_metadata(s) for s in stages]
    original = _state(fp_checkpoint)
    before_hash = _file_sha256(fp_checkpoint)
    model_state = model.state_dict()
    if model_state.keys() != original.keys() or any(
        model_state[n].shape != v.shape or not torch.equal(model_state[n].detach().cpu(), v.to(model_state[n].dtype))
        for n, v in original.items()
    ):
        raise ValueError("Model must be loaded from the exact FP checkpoint before calibration")
    source_hashes = {n: tensor_sha256(v) for n, v in model_state.items()}
    selected, excluded = _select_layers(model, original, include, exclude, hessian_dtype, max_hessian_bytes)
    if not selected:
        raise ValueError("No supported Conv2d layers selected; inspect patterns and Hessian memory limit")
    device = torch.device(device)
    teacher = _prepare_model(model, device, use_fp16)
    modules = dict(teacher.named_modules())
    runtime_fp_hashes = {key: tensor_sha256(teacher.state_dict()[key]) for key in selected.values()}
    # Verification uses the same held-fixed calibration state; this is not a
    # reconstruction quality test or held-out generalization measurement.
    probe_x, probe_t, probe_info = next(_states(samples, stages, seed, device))
    probe_fp = teacher(probe_x, probe_t).detach().cpu()
    if not torch.isfinite(probe_fp).all():
        raise ValueError("FP teacher produced nonfinite outputs")
    results, summaries = {}, {}
    for name, key in selected.items():
        collector = Conv2dGPTQ(
            modules[name], config, source_weight=original[key], hessian_dtype=hessian_dtype,
            max_hessian_bytes=max_hessian_bytes, patch_chunk_size=patch_chunk_size,
            max_patches_per_batch=max_patches_per_batch,
        )
        try:
            with collector.capture():
                for xt, model_t, _ in _states(samples, stages, seed, device):
                    output = teacher(xt, model_t)
                    if not torch.isfinite(output).all():
                        raise ValueError(f"Nonfinite FP teacher output while collecting {name}")
            if collector.calibration_batches == 0:
                excluded.append({"layer": name, "parameters": [key], "type": "Conv2d",
                                 "reason": "not_executed_during_calibration"})
                continue
            result = collector.quantize()
            # Results stay on CPU so only one layer's GPU workspace is live.
            results[key] = result
            summaries[key] = collector.calibration_summary()
            summaries[key]["runtime_weight_dtype"] = str(modules[name].weight.dtype)
        finally:
            collector.free()
        print(f"calibrated {name}: {summaries[key]['selected_patches']} patches", flush=True)
    del teacher, modules, collector
    calibration = dict(provenance)
    calibration.update(
        description="FP teacher; q_sample of normalized calibration slice pairs at fixed diffusion timesteps",
        data_samples=[dict(sample_id=s.sample_id, **s.provenance) for s in samples],
        state_policy="q_sample(x0,t,fixed_noise); forward marginal; no reverse trajectory, mask or DC",
        layer_policy="independent FP teacher; one Conv2d Hessian at a time; no quantized feedback",
        stages=stage_info, seed=seed,
        noise_policy="CPU float32 standard normal; SHA256([seed,sample_id,stage,local_timestep]) seed per state",
        input_dtype="torch.float32", use_fp16=use_fp16, hessian_dtype=str(hessian_dtype),
        max_hessian_bytes=max_hessian_bytes, include_patterns=list(include), exclude_patterns=list(exclude),
        layer_collection=summaries, excluded_layers=excluded,
        excluded_parameters=[key for key in original if key not in results],
        artifact_verification_required=True,
        implementation_files_sha256={"calibration.py": _file_sha256(Path(__file__))},
    )
    paths = save_gptq_artifact(fp_checkpoint, results, output_dir, calibration=calibration)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    verification = {"status": "failed", "reconstruction": "not_run", "probe": probe_info}
    manifest["verification"] = verification
    try:
        if _file_sha256(fp_checkpoint) != before_hash:
            raise ValueError("FP checkpoint changed during calibration")
        if any(tensor_sha256(v) != source_hashes[n] for n, v in model.state_dict().items()):
            raise ValueError("Source model changed during calibration")
        w4_state = _state(paths["checkpoint"])
        changed = []
        if w4_state.keys() != original.keys():
            raise ValueError("Reloaded W4 state_dict keys differ")
        for name, value in original.items():
            after = w4_state[name]
            if value.shape != after.shape or value.dtype != after.dtype or not torch.isfinite(after).all():
                raise ValueError(f"Invalid reloaded tensor {name}")
            if not torch.equal(value, after):
                if name not in results:
                    raise ValueError(f"Unselected tensor changed: {name}")
                changed.append(name)
        if not changed:
            raise ValueError("FP and W4 weights are identical")
        reloaded = copy.deepcopy(model)
        reloaded.load_state_dict(w4_state, strict=True)
        if use_fp16:
            reloaded.convert_to_fp16()
        reloaded.to(device).eval().requires_grad_(False)
        runtime_changed = [key for key in results if tensor_sha256(reloaded.state_dict()[key]) != runtime_fp_hashes[key]]
        if not runtime_changed:
            raise ValueError("FP and W4 runtime weights are identical after activation-path conversion")
        probe_w4 = reloaded(probe_x, probe_t).detach().cpu()
        if probe_w4.shape != probe_fp.shape or not torch.isfinite(probe_w4).all():
            raise ValueError("Reloaded W4 output is invalid")
        delta = probe_w4.double() - probe_fp.double()
        verification.update(status="passed", fp_checkpoint_unchanged=True, source_model_unchanged=True,
                            strict_reload=True, changed_parameters=changed,
                            runtime_changed_parameters=runtime_changed, output_shape=list(probe_w4.shape),
                            probe_output_max_abs_difference=float(delta.abs().max()),
                            probe_output_mse=float(delta.square().mean()),
                            probe_outputs_differ=not torch.equal(probe_fp, probe_w4))
    except Exception as exc:
        verification["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        paths["manifest"].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
