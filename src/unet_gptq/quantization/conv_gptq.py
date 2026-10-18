"""Activation calibration for an ordinary, ungrouped :class:`torch.nn.Conv2d`.



The returned weights simulate weight-only quantization using floating-point
convolution. This module does not provide a packed INT4 inference kernel and
never updates the source layer. Calibrate on the same denoiser inputs and
timesteps used by reconstruction, rather than on clean images alone.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from .gptq import GPTQConfig, GPTQResult, HessianAccumulator, quantize_weight

class Conv2dGPTQ:
    """
    Collect input-patch curvature and quantize a Conv2d without changing it.
    """

    def __init__(
        self,
        layer: nn.Conv2d,
        config: GPTQConfig | None = None,
        *,
        hessian_dtype: torch.dtype = torch.float32,
        max_hessian_bytes: int = 512 * 1024**2,
        patch_chunk_size: int = 1024,
        max_patches_per_batch: int | None = 4096,
    ) -> None:
        if not isinstance(layer, nn.Conv2d):
            raise TypeError("Conv2dGPTQ requires an nn.Conv2d layer")
        if layer.groups != 1:
            raise ValueError("Conv2dGPTQ supports groups=1 only")
        if isinstance(layer.padding, str):
            raise ValueError("Conv2dGPTQ requires numeric padding, not string padding")
        if layer.padding_mode != "zeros":
            raise ValueError("Conv2dGPTQ supports padding_mode='zeros' only")
        if not layer.weight.is_floating_point():
            raise TypeError("Conv2dGPTQ requires real floating-point weights")
        if isinstance(patch_chunk_size, bool) or not isinstance(patch_chunk_size, int) or patch_chunk_size < 1:
            raise ValueError("patch_chunk_size must be a positive integer")
        if max_patches_per_batch is not None and (
            isinstance(max_patches_per_batch, bool)
            or not isinstance(max_patches_per_batch, int)
            or max_patches_per_batch < 1
        ):
            raise ValueError("max_patches_per_batch must be a positive integer or None")
        self.layer = layer
        # Snapshot before an activation-path conversion such as model.half().
        # Quantization must keep using the original FP checkpoint as its basis.
        self.source_weight = layer.weight.detach().clone()
        self._structure = self._layer_structure()
        self.config = GPTQConfig() if config is None else config
        if not isinstance(self.config, GPTQConfig):
            raise TypeError("config must be GPTQConfig or None")
        self.patch_chunk_size = patch_chunk_size
        self.max_patches_per_batch = max_patches_per_batch
        self.columns = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        self.accumulator = HessianAccumulator(
            self.columns,
            device=layer.weight.device,
            dtype=hessian_dtype,
            max_hessian_bytes=max_hessian_bytes,
            row_chunk_size=patch_chunk_size,
        )
        self.calibration_batches = 0
        self.candidate_patches = 0
        self.selected_patches = 0
        self._freed = False
        self._capture_handle = None

    def _require_open(self) -> None:
        if self._freed:
            raise RuntimeError("Conv2dGPTQ calibration storage has been freed")
        if self._layer_structure() != self._structure:
            raise RuntimeError("Conv2dGPTQ source layer structure changed after collector creation")

    def _layer_structure(self) -> tuple:
        return (
            tuple(self.layer.weight.shape), self.layer.in_channels,
            self.layer.out_channels, self.layer.kernel_size, self.layer.stride,
            self.layer.dilation, self.layer.padding, self.layer.padding_mode,
            self.layer.groups,
        )

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor, output: torch.Tensor | None = None) -> None:
        """Accumulate selected input patches; ``output`` is accepted but unused.

        Inputs must have shape ``[B, C, H, W]``. Activations are detached and
        their source tensor is never modified or retained after this call.
        """
        self._require_open()
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Conv2dGPTQ calibration inputs must be a tensor")
        if inputs.ndim != 4 or inputs.shape[1] != self.layer.in_channels:
            raise ValueError("Conv2dGPTQ inputs must have shape [B, in_channels, H, W]")
        if not inputs.is_floating_point():
            raise TypeError("Conv2dGPTQ calibration inputs must be real floating point")
        if inputs.shape[0] == 0 or inputs.shape[2] == 0 or inputs.shape[3] == 0:
            raise ValueError("Conv2dGPTQ calibration inputs must be nonempty")
        # Validate before accumulating any chunk, so an invalid late patch
        # cannot leave a partially collected batch with inconsistent counts.
        if not torch.isfinite(inputs).all():
            raise ValueError("Conv2dGPTQ calibration inputs must be finite")
        kh, kw = self.layer.kernel_size
        dh, dw = self.layer.dilation
        sh, sw = self.layer.stride
        ph, pw = self.layer.padding
        effective_h, effective_w = dh * (kh - 1) + 1, dw * (kw - 1) + 1
        padded = inputs.detach()
        if ph or pw:
            padded = F.pad(padded, (pw, pw, ph, ph))
        if padded.shape[2] < effective_h or padded.shape[3] < effective_w:
            raise ValueError("Conv2dGPTQ input is smaller than the effective convolution kernel")
        # These operations create views; only advanced indexing below copies
        # patches. Avoid reshape of the whole view, which would materialize it.
        patches = padded.unfold(2, effective_h, sh).unfold(3, effective_w, sw)
        patches = patches[..., ::dh, ::dw].permute(0, 2, 3, 1, 4, 5)
        batch, out_h, out_w = patches.shape[:3]
        spatial = out_h * out_w
        total = batch * spatial
        selected = total if self.max_patches_per_batch is None else min(total, self.max_patches_per_batch)
        for start in range(0, selected, self.patch_chunk_size):
            position = torch.arange(start, min(start + self.patch_chunk_size, selected), device=inputs.device)
            if selected != total:
                if selected == 1:
                    position = torch.full_like(position, total // 2)
                else:
                    position = torch.div(position * (total - 1), selected - 1, rounding_mode="floor")
            batch_index = torch.div(position, spatial, rounding_mode="floor")
            spatial_index = position.remainder(spatial)
            row_index = torch.div(spatial_index, out_w, rounding_mode="floor")
            col_index = spatial_index.remainder(out_w)
            rows = patches[batch_index, row_index, col_index].reshape(-1, self.columns)
            self.accumulator.add_rows(rows)
        self.calibration_batches += 1
        self.candidate_patches += total
        self.selected_patches += selected

    def calibration_summary(self) -> dict[str, object]:
        """Return JSON-compatible collection policy and counts, without tensors."""
        return {
            "calibration_batches": self.calibration_batches,
            "candidate_patches": self.candidate_patches,
            "selected_patches": self.selected_patches,
            "patch_selection": "all" if self.max_patches_per_batch is None else "uniform_deterministic",
            "max_patches_per_batch": self.max_patches_per_batch,
            "patch_chunk_size": self.patch_chunk_size,
            "columns": self.columns,
        }

    @torch.no_grad()
    def quantize(self) -> GPTQResult:
        """Return quantization artifacts in Conv2d kernel shape; keep FP intact.

        Scales and group indices use the flattened input-column axis. To run a
        simulation, copy ``result.dequantized_weight`` into a *separate* model
        while retaining the original bias. FP and GPTQ should use the same DC,
        sampling noise, timesteps, and precision for their activation path.
        """
        self._require_open()
        result = quantize_weight(self.source_weight.flatten(1), self.accumulator.hessian, self.config)
        return replace(
            result,
            dequantized_weight=result.dequantized_weight.reshape_as(self.source_weight),
            codes=result.codes.reshape_as(self.source_weight),
            calibration_rows=self.accumulator.num_rows,
        )

    @contextmanager
    def capture(self) -> Iterator[Conv2dGPTQ]:
        """Collect during layer calls and remove the pre-hook even on failure.

        Hooks collect input only and do not change model outputs or gradients.
        Use a single active collector on a layer; nested ``capture`` calls on
        this collector are rejected to prevent double counting.
        """
        self._require_open()
        if self._capture_handle is not None:
            raise RuntimeError("Conv2dGPTQ.capture is already active")

        def collect(_module: nn.Module, args: tuple, kwargs: dict) -> None:
            if args:
                value = args[0]
            elif "input" in kwargs:
                value = kwargs["input"]
            elif "x" in kwargs:
                value = kwargs["x"]
            else:
                raise ValueError("Conv2dGPTQ capture could not locate the convolution input")
            self.add_batch(value)

        handle = self.layer.register_forward_pre_hook(collect, with_kwargs=True)
        self._capture_handle = handle
        try:
            yield self
        finally:
            handle.remove()
            self._capture_handle = None

    def free(self) -> None:
        """Remove an active hook and release curvature storage permanently."""
        if self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None
        if not self._freed:
            self.accumulator.release()
            self._freed = True    



