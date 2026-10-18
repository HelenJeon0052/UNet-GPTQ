"""
Hessian-aware, weight-only GPTQ for a Linear layer or flattened weight matrix.


This implements sequential error compensation with an upper Cholesky factor of
H^{-1}, including blockwise propagation to future columns. Quantization grids
are static per output row/input-column group; act-order preserves the ORIGINAL
group membership. Codes are stored as int16 for inspection, not packed INT4.

Create collectors BEFORE casting a calibration model to FP16: the constructor
snapshots source weights while hooks collect the activations actually used at
runtime. Quantization returns a result without changing the module. Calibrate
one/few layers at a time: a dense Hessian is O(columns**2), and factorization
needs several additional matrices. The allocation guard covers the accumulator
only. No reconstruction/calibration dataset or diffusion schedule is inferred.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import torch

@dataclass(frozen=True)
class GPTQConfig:
    bits: int = 4
    group_size: int = 128
    block_size: int = 128
    damping: float = 0.01
    symmetric: bool = True
    act_order: bool = False

    def __post_init__(self):
        for name in ("bits", "group_size", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} > 0, but got {value}")
            if not 2 <= self.bits <= 8:
                raise ValueError("bits is between 2 and 8")
            if self.group_size != -1 and self.group_size < 1:
                raise ValueError("group size is -1 or positive")
            if self.block_size < 1:
                raise ValueError("block size must be positive")
            if (isinstance(self.damping, bool) or not isinstance(self.damping, Real) or not math.isfinite(self.damping) or self.damping <= 0):
                raise ValueError("damping must be finite and positive")
            if not isinstance(self.symmetric, bool) or not isinstance(self.act_order, bool):
                raise ValueError("symmetric and act_order must be bool")
            
            for name in ("bits", "group_size", "block_size"):
                object.__setattr__(self, name, int(getattr(self, name)))

            object.__setattr__(self, "damping", float(self.damping))
    
def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())

    return digest.hexdigest()

class HessianAccumulator:

    def __init__(self, columns: int, *, device = None, dtype = torch.float32, max_hessian_bytes: int = 512 * 1024**2, row_chunk_size: int = 4096):
        for name, value in (("columns", columns), ("max_hessian_bytes", max_hessian_bytes), ("row_chunk_size", row_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Hessian dtype is float32 or float64")
        required = columns**2 * torch.empty((), dtype=dtype).element_size()
        if required > max_hessian_bytes:
            raise MemoryError(f"Hessian alone needs {required} bytes, limit is {max_hessian_bytes}"
                                "Factorization requires additional workspace"                    
            )
        self.columns, self.row_chunk_size = columns, row_chunk_size
        self._gram = torch.zeros(columns, columns, device = device, dtype = dtype)
        self.num_rows = 0


    @torch.no_grad()
    def add_rows(self, rows: torch.Tensor) -> None:
        if self._gram is None:
            raise RuntimeError("Hessian accumulator has been released")
        if (rows.ndim != 2 or rows.shape[1] != self.columns or rows.shape[0] == 0 or not rows.is_floating_point()):
            raise ValueError(f"Expected non empty floating rows [N, {self.columns}]")
        if not torch.isfinite(rows).all():
            raise ValueError("calibration inputs is finite")
        for chunk in rows.detach().split(self.row_chunk_size):
            chunk = chunk.to(device = self._gram.device, dtype=self._gram.dtype)
            self._gram.addmm_(chunk.T, chunk)
        self.num_rows += rows.shape[0]

    @property
    def hessian(self) -> torch.Tensor:
        if self._gram is None:
            raise RuntimeError("Hessian accumulator has been released")
        if self.num_rows == 0:
            raise ValueError(f"No calibration rows have been collected, got {self.num_rows}")
        
        return self._gram * (2.0 / self.num_rows)
    
    def release(self) -> None:
        self._gram = None

@dataclass
class GPTQResult:
    dequantized_weight: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    group_index: torch.Tensor
    permutation: torch.Tensor
    dead_columns: torch.Tensor
    config: GPTQConfig
    relative_damping: float
    damping_value: float
    source_weight_sha256: str
    calibration_output_mse: float
    calibration_rows: int = 0

    def metadata(self) -> dict:
        return {
            "config": asdict(self.config), "shape": list(self.dequantized_weight.shape),
            "storage_dtype": str(self.dequantized_weight.dtype),
            "compute_dtype": str(self.scales.dtype), "grid": "static_original_column_groups",
            "integer_range": ([-(2 ** (self.config.bits - 1)), 2 ** (self.config.bits - 1) - 1]
                              if self.config.symmetric else [0, 2**self.config.bits - 1]),
            "relative_damping": self.relative_damping, "damping_value": self.damping_value,
            "dead_columns": int(self.dead_columns.sum().item()),
            "calibration_rows": self.calibration_rows,
            "calibration_output_mse": self.calibration_output_mse,
            "source_weight_sha256": self.source_weight_sha256,
            "packed_int4_execution": False,
        }


def _fit_grids(weight, config):
    rows, columns = weight.shape
    size = columns if config.group_size == -1 else config.group_size
    count = (columns + size - 1) // size
    scales = weight.new_empty(rows, count)
    zeros = weight.new_empty(rows, count)
    if config.symmetric:
        qmin, qmax = -(2 ** (config.bits - 1)), 2 ** (config.bits - 1) - 1
    else:
        qmin, qmax = 0, 2**config.bits - 1
    for group, start in enumerate(range(0, columns, size)):
        values = weight[:, start:start + size]
        if config.symmetric:
            extent = values.abs().amax(dim=1)
            scales[:, group] = torch.where(extent > 0, extent / qmax, 1.0)
            zeros[:, group] = 0
        else:
            lower = torch.minimum(values.amin(dim=1), values.new_zeros(rows))
            upper = torch.maximum(values.amax(dim=1), values.new_zeros(rows))
            span = upper - lower
            scales[:, group] = torch.where(span > 0, span / qmax, 1.0)
            zeros[:, group] = torch.round(-lower / scales[:, group]).clamp(qmin, qmax)
    groups = torch.arange(columns, device=weight.device) // size
    return scales, zeros, groups, qmin, qmax


@torch.no_grad()
def quantize_weight(weight: torch.Tensor, hessian: torch.Tensor,
                    config: GPTQConfig = GPTQConfig()) -> GPTQResult:
    """Quantize a matrix without mutating weight or H; return inspectable grids.

    H is a (possibly singular) empirical 2*mean(x*x.T) matrix. Zero-variance
    columns are zeroed and recorded. Entirely zero calibration, nonfinite
    inputs, or an unsuccessful damped Cholesky factorization are errors.
    """
    if not isinstance(config, GPTQConfig):
        raise TypeError("config must be GPTQConfig")
    if (weight.ndim != 2 or min(weight.shape) == 0 or not weight.is_floating_point()
            or not torch.isfinite(weight).all()):
        raise ValueError("weight must be a finite nonempty floating matrix")
    columns = weight.shape[1]
    if (hessian.shape != (columns, columns) or hessian.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(hessian).all()):
        raise ValueError("Hessian must be a finite float32/float64 [columns,columns] matrix")
    if not torch.allclose(hessian, hessian.T, rtol=1e-5, atol=1e-7):
        raise ValueError("Hessian must be symmetric")
    diagonal = hessian.diag()
    if (diagonal < 0).any():
        raise ValueError("Hessian has negative variance")
    dead = diagonal == 0
    if dead.all():
        raise ValueError("Calibration has no nonzero input features")
    # A valid Gram matrix cannot correlate a zero-variance column with others.
    if dead.any() and torch.count_nonzero(hessian[dead]):
        raise ValueError("Zero-variance Hessian columns must have zero covariance")
    work = weight.detach().to(device=hessian.device, dtype=hessian.dtype).clone()
    work[:, dead] = 0
    scales, zeros, groups, qmin, qmax = _fit_grids(work, config)
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("Weight range cannot be represented in the chosen compute dtype")
    damp = float(diagonal[~dead].mean().item()) * config.damping
    H = hessian.detach().clone()
    H[dead, dead] = 1.0
    H.diagonal().add_(damp)
    permutation = (torch.argsort(diagonal, descending=True, stable=True) if config.act_order
                   else torch.arange(columns, device=H.device))
    work = work[:, permutation]
    H = H[permutation][:, permutation]
    permuted_groups = groups[permutation]
    try:
        lower = torch.linalg.cholesky(H)
        inverse = torch.cholesky_inverse(lower)
        upper = torch.linalg.cholesky(inverse, upper=True)
    except RuntimeError as exc:
        raise ValueError("Damped Hessian factorization failed; inspect calibration or increase damping") from exc
    del H, lower, inverse
    quantized = torch.empty_like(work)
    codes = torch.empty_like(work, dtype=torch.int16)
    for start in range(0, columns, config.block_size):
        end = min(start + config.block_size, columns)
        block = work[:, start:end].clone()
        errors = torch.zeros_like(block)
        for offset, column in enumerate(range(start, end)):
            group = permuted_groups[column]
            scale, zero = scales[:, group], zeros[:, group]
            value = block[:, offset]
            code = torch.round(value / scale + zero).clamp(qmin, qmax)
            quant = (code - zero) * scale
            quantized[:, column] = quant
            codes[:, column] = code.to(torch.int16)
            error = (value - quant) / upper[column, column]
            block[:, offset:] -= error[:, None] * upper[column, column:end][None, :]
            errors[:, offset] = error
        if end < columns:
            work[:, end:] -= errors @ upper[start:end, end:]
    restore = torch.argsort(permutation)
    quantized = quantized[:, restore].to(device=weight.device, dtype=weight.dtype)
    if not torch.isfinite(quantized).all():
        raise ValueError("Quantized weights are nonfinite in the source storage dtype")
    delta = quantized.to(hessian) - weight.to(hessian)
    loss = float(((delta @ hessian) * delta).sum().item()) / (2 * weight.shape[0])
    if not math.isfinite(loss):
        raise ValueError("Calibration error overflowed; rescale inputs or use a float64 Hessian")
    return GPTQResult(
        dequantized_weight=quantized, codes=codes[:, restore].to(weight.device),
        scales=scales.to(weight.device), zero_points=zeros.to(weight.device),
        group_index=groups.to(weight.device), permutation=permutation.to(weight.device),
        dead_columns=dead.to(weight.device), config=config, relative_damping=config.damping,
        damping_value=damp, source_weight_sha256=tensor_sha256(weight),
        calibration_output_mse=max(0.0, loss),
    )


class GPTQ:
    """Collect Linear activations, then quantize an immutable source snapshot."""

    def __init__(self, layer: torch.nn.Linear, config: GPTQConfig | None = None, *,
                 hessian_dtype=torch.float32, max_hessian_bytes=512 * 1024**2):
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError("GPTQ accepts nn.Linear; use Conv2dGPTQ for nn.Conv2d")
        self.layer = layer
        self.config = GPTQConfig() if config is None else config
        if not isinstance(self.config, GPTQConfig):
            raise TypeError("config must be GPTQConfig or None")
        self.source_weight = layer.weight.detach().clone()
        self.accumulator = HessianAccumulator(
            layer.in_features, device=layer.weight.device, dtype=hessian_dtype,
            max_hessian_bytes=max_hessian_bytes,
        )
        self.calibration_batches = 0
        self._capturing = False

    def add_batch(self, inputs: torch.Tensor, output=None) -> None:
        if inputs.ndim < 2 or inputs.shape[-1] != self.source_weight.shape[1]:
            raise ValueError("Linear calibration inputs must have shape [...,in_features]")
        self.accumulator.add_rows(inputs.reshape(-1, inputs.shape[-1]))
        self.calibration_batches += 1

    @contextmanager
    def capture(self):
        """Attach an input-only hook; always remove it on exit, including errors."""
        if self._capturing:
            raise RuntimeError("Capture contexts cannot overlap")
        self._capturing = True
        def hook(module, args, kwargs):
            inputs = args[0] if args else kwargs.get("input")
            if not torch.is_tensor(inputs):
                raise TypeError("Expected a tensor Linear input")
            self.add_batch(inputs)
        handle = None
        try:
            handle = self.layer.register_forward_pre_hook(hook, with_kwargs=True)
            yield self
        finally:
            if handle is not None:
                handle.remove()
            self._capturing = False

    def quantize(self) -> GPTQResult:
        result = quantize_weight(self.source_weight, self.accumulator.hessian, self.config)
        result.calibration_rows = self.accumulator.num_rows
        return result

    def free(self) -> None:
        self.accumulator.release()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_gptq_artifact(fp_checkpoint: str | Path, results: Mapping[str, GPTQResult],
                       output_dir: str | Path, *, calibration: dict) -> dict[str, Path]:
    """Write W4 checkpoint, manifest and grid sidecar into a NEW directory.

    Keys in results must be full state_dict parameter names, e.g. 'conv.weight'.
    Source snapshots must exactly match the original checkpoint (shape/dtype/
    bytes). All unselected parameters and buffers are preserved. Supply a real
    calibration description, split/sample IDs and diffusion-state policy; this
    helper cannot validate dataset leakage. Selected tensors sharing checkpoint
    storage with other keys are unsupported: another alias could restore FP
    values during load_state_dict. No output is overwritten.
    """
    if not results:
        raise ValueError("Select at least one weight tensor")
    if not isinstance(calibration, dict) or not calibration.get("description"):
        raise ValueError("calibration.description is required")
    # Reject non-JSON provenance before any output is created.
    json.dumps(calibration, allow_nan=False)
    fp_checkpoint, output_dir = Path(fp_checkpoint), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    fp_hash = _file_sha256(fp_checkpoint)
    state = torch.load(fp_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError("FP checkpoint must contain a tensor state_dict")
    storage_names: dict[int, list[str]] = {}
    for name, value in state.items():
        if value.numel():
            storage_names.setdefault(value.untyped_storage().data_ptr(), []).append(name)
    updated = {name: value.clone() for name, value in state.items()}
    layers, grids = {}, {}
    changed = []
    for name, result in results.items():
        if name not in state or not name.endswith("weight") or state[name].ndim < 2:
            raise ValueError(f"Not a checkpoint weight parameter: {name}")
        aliases = storage_names.get(state[name].untyped_storage().data_ptr(), [])
        if len(aliases) > 1:
            raise ValueError(f"Shared checkpoint storage is unsupported for {name}: {aliases}")
        original, quantized = state[name], result.dequantized_weight.detach().cpu()
        if result.config.bits != 4:
            raise ValueError("The current paired runner requires W4 artifacts")
        if quantized.shape != original.shape or quantized.dtype != original.dtype:
            raise ValueError(f"Source shape/dtype mismatch for {name}; collect before casting the model")
        if tensor_sha256(original.reshape(original.shape[0], -1)) != result.source_weight_sha256:
            raise ValueError(f"Calibration source does not match the FP checkpoint: {name}")
        if not torch.isfinite(quantized).all():
            raise ValueError(f"Nonfinite quantized weight: {name}")
        if not torch.equal(original, quantized):
            changed.append(name)
        updated[name] = quantized.clone()
        layers[name] = result.metadata()
        grids[name] = {key: getattr(result, key).detach().cpu() for key in (
            "codes", "scales", "zero_points", "group_index", "permutation", "dead_columns"
        )}
    if not changed:
        raise ValueError("No weight changed; refusing an FP-vs-FP artifact")
    json.dumps(layers, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "weights_w4.pt"
    parameters = output_dir / "quantization.pt"
    manifest_path = output_dir / "manifest.json"
    torch.save(updated, checkpoint)
    torch.save(grids, parameters)
    manifest = {
        "schema_version": 1, "method": "gptq", "weight_bits": 4,
        "activation_quantization": False, "format": "dequantized_state_dict",
        "fp_checkpoint_sha256": fp_hash, "w4_checkpoint_sha256": _file_sha256(checkpoint),
        "quantized_parameters": list(results), "changed_parameters": changed,
        "calibration": calibration, "layers": layers,
        "quantization_parameters_file": parameters.name,
        "quantization_parameters_sha256": _file_sha256(parameters),
        "packed_int4_execution": False,
        "implementation": "unet_gptq.quantization.gptq; static groups; blockwise Hessian error feedback",
        "implementation_files_sha256": {
            filename: _file_sha256(Path(__file__).with_name(filename))
            for filename in ("gptq.py", "conv_gptq.py")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {"checkpoint": checkpoint, "manifest": manifest_path, "quantization": parameters}






