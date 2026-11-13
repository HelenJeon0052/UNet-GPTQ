"""Image quality metrics with explicit magnitude and complex conventions."""
from __future__ import annotations

import numpy as np
import torch

from unet_gptq.domain.data_consistency import observed_residual
from unet_gptq.metrics.quality import slice_quality


def image_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor,
    observed_kspace: torch.Tensor, mask: torch.Tensor,
) -> dict:
    """Average per-slice metrics for complex [B,2,H,W] images.

    ``nmse`` and ``psnr`` are compatibility aliases for magnitude metrics;
    phase-sensitive NMSE is named separately. If any slice has an undefined or
    infinite value, its aggregate is None rather than silently dropping it.
    Metric calculation is outside the timed reconstruction scope.
    """
    if reconstruction.shape != target.shape or reconstruction.shape != observed_kspace.shape:
        raise ValueError("Reconstruction, target and observations must take identical shapes")
    if reconstruction.ndim != 4 or reconstruction.shape[1] != 2 or not reconstruction.numel():
        raise ValueError("Expected nonempty tensors with shape [B,2,H,W]")
    if not all(torch.is_complex(x) for x in (reconstruction, target, observed_kspace)):
        raise TypeError("Reconstruction, target and observations must be complex")
    actual = reconstruction.detach().cpu().numpy().reshape(-1, *reconstruction.shape[-2:])
    reference = target.detach().cpu().numpy().reshape(-1, *target.shape[-2:])
    rows = [slice_quality(value, ref) for value, ref in zip(actual, reference)]

    def mean_if_defined(key):
        values = [row[key] for row in rows]
        return float(np.mean(values)) if values and all(value is not None for value in values) else None

    magnitude_nmse = mean_if_defined("magnitude_nmse")
    psnr = mean_if_defined("psnr_db")
    return {
        "magnitude_nmse": magnitude_nmse,
        "complex_nmse": mean_if_defined("complex_nmse"),
        "psnr_db": psnr,
        "nmse": magnitude_nmse,
        "psnr": psnr,
        "undefined_or_infinite_psnr_slices": sum(row["psnr_db"] is None for row in rows),
        "sampled_kspace_relative": float(observed_residual(reconstruction, observed_kspace, mask).item()),
        "aggregation": "unweighted mean of per-slice metrics",
    }

