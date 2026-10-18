"""Untimed, bounded-memory sampling diagnostics for paired FP/GPTQ runs.
The reference denoiser sees the exact input and mapped timestep received by the
trajectory model. Only scalar summaries survive a forward call; no trajectory
tensors are retained. Raw model outputs may include learned-variance channels.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import random
from typing import Any, Iterator

import numpy as np
import torch

from unet_gptq.domain.data_consistency import observed_residual


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])

def _clone_inputs(value: Any) -> Any:
    """Snapshot inputs before a custom model can modify them in place."""
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _input_metadata(args: tuple, kwargs: dict) -> dict:
    image = args[0] if args else kwargs.get("x")
    timestep = args[1] if len(args) > 1 else kwargs.get("timesteps", kwargs.get("t"))
    if not torch.is_tensor(image):
        raise TypeError("Denoiser diagnostics require a tensor input x.")
    if timestep is None:
        raise ValueError("Denoiser diagnostics require an actual model timestep.")
    if torch.is_tensor(timestep):
        timestep = timestep.detach().cpu().reshape(-1).tolist()
    elif isinstance(timestep, (int, float)):
        timestep = [timestep]
    else:
        raise TypeError("Denoiser timestep must be a tensor or number.")
    return {
        "input_shape": list(image.shape),
        "input_dtype": str(image.dtype),
        "timestep": timestep,
    }

def _validate_output(output: Any) -> None:
    if not torch.is_tensor(output) or not (output.is_floating_point() or output.is_complex()):
        raise TypeError("Denoiser diagnostics require a floating-point tensor output.")
    if output.numel() == 0 or not torch.isfinite(output).all():
        raise ValueError("Denoiser diagnostics require finite, nonempty output.")


class SamplingDiagnostics:
    """Observe an untimed trajectory, optionally comparing W4 against fresh FP.

    Call ``observe`` around each ``diffusion.p_sample``. Forward hooks capture
    timesteps *after* an external diffusion wrapper maps or rescales them. A
    reference call reuses the trajectory call's pre-forward RNG state and then
    restores its post-forward state, preserving subsequent sampling randomness.
    """

    def __init__(self, reference_model=None, trajectory: str = "fp"):
        if trajectory not in ("fp", "w4"):
            raise ValueError("trajectory must be 'fp' or 'w4'.")
        if reference_model is not None and trajectory != "w4":
            raise ValueError("An FP reference comparison requires a W4 trajectory.")
        self.reference_model = reference_model
        self.trajectory = trajectory
        self.denoiser_rows: list[dict] = []
        self.dc_rows: list[dict] = []
        self._active = False

    @property
    def nfe(self) -> int:
        """Completed trajectory-model forward calls, excluding FP reference calls."""
        return len(self.denoiser_rows)

    @contextmanager
    def observe(self, model, *, stage: str, sample_index: int, step_index: int) -> Iterator[None]:
        if self._active:
            raise RuntimeError("SamplingDiagnostics.observe scopes cannot overlap.")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Denoiser diagnostics require an nn.Module.")
        if self.reference_model is model:
            raise ValueError("Use a separately loaded FP reference model.")
        initial_nfe = self.nfe
        pending: list[dict] = []

        def before_forward(module, args, kwargs):
            entry = {"metadata": _input_metadata(args, kwargs)}
            if self.reference_model is not None:
                entry.update(args=_clone_inputs(args), kwargs=_clone_inputs(kwargs), rng=_rng_state())
            pending.append(entry)

        def after_forward(module, args, kwargs, output):
            entry = pending.pop()
            _validate_output(output)
            row = {
                "stage": stage, "sample_index": sample_index, "step_index": step_index,
                "call_index": self.nfe, "trajectory": self.trajectory,
                **entry["metadata"],
                "output_shape": list(output.shape), "output_dtype": str(output.dtype),
                "output_scope": "raw_model_output",
                "comparison": "not_compared", "reference_output_dtype": None,
                "mse": None, "relative_l2": None, "max_abs_error": None,
            }
            if self.reference_model is not None:
                caller_state = _rng_state()
                try:
                    _restore_rng(entry["rng"])
                    with torch.no_grad():
                        reference = self.reference_model(*entry["args"], **entry["kwargs"])
                finally:
                    _restore_rng(caller_state)
                _validate_output(reference)
                if reference.shape != output.shape:
                    raise ValueError("FP/W4 denoiser output shapes differ on common input.")
                reference_dtype = str(reference.dtype)
                dtype = torch.complex128 if reference.is_complex() or output.is_complex() else torch.float64
                reference = reference.detach().to(device=output.device, dtype=dtype)
                error = output.detach().to(dtype=dtype) - reference
                numerator = float(error.abs().square().sum().item())
                denominator = float(reference.abs().square().sum().item())
                row.update(
                    comparison="common_input_on_w4_trajectory",
                    reference_output_dtype=reference_dtype,
                    mse=numerator / error.numel(),
                    relative_l2=(math.sqrt(numerator / denominator) if denominator else
                                 (0.0 if numerator == 0 else None)),
                    max_abs_error=float(error.abs().max().item()),
                )
            self.denoiser_rows.append(row)
            
        handles = []
        self._active = True
        try:
            handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True))
            handles.append(model.register_forward_hook(after_forward, with_kwargs=True))
            yield
            if self.nfe == initial_nfe:
                raise RuntimeError("No denoiser forward was observed during p_sample.")
        finally:
            for handle in handles:
                handle.remove()
            pending.clear()
            self._active = False

    def after_dc(self, image, target, clean_observed, mask, *, stage: str,
                 sample_index: int, step_index: int, timestep: int, dc_weight: float) -> None:
        """Distinguish noisy coarse targets from the fixed clean observation."""
        self.dc_rows.append({
            "stage": stage, "sample_index": sample_index, "step_index": step_index,
            "timestep": int(timestep), "trajectory": self.trajectory,
            "dc_weight": float(dc_weight),
            "target_relative_l2": float(observed_residual(image, target, mask).item()),
            "clean_relative_l2": float(observed_residual(image, clean_observed, mask).item()),
            "fft_layout": "unshifted", "fft_norm": "backward", "dc_dtype": "torch.complex64",
        })
