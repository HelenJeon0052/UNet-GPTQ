"""Image-quality metrics for one complex-valued reference slice at a time."""
from __future__ import annotations

import math

import numpy as np


METRIC_SCHEMA_VERSION = 2
IMAGE_METRICS = ("magnitude_nmse", "complex_nmse", "magnitude_rmse", "magnitude_psnr_db")


def slice_quality(estimate, reference) -> dict:
    """Compare matching 2-D slices without independently scaling the images.

    A zero-energy reference makes both NMSEs and PSNR undefined (NaN). With a
    nonzero reference and exact magnitude reconstruction, magnitude PSNR is
    positive infinity. Status fields disambiguate these non-finite results.
    """
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    if estimate.ndim != 2 or estimate.shape != reference.shape or not estimate.size:
        raise ValueError("estimate and reference must be nonempty, matching 2-D slices")
    if not all(np.issubdtype(x.dtype, np.number) for x in (estimate, reference)):
        raise ValueError("estimate and reference must be numeric")
    estimate, reference = estimate.astype(np.complex128), reference.astype(np.complex128)
    if not (np.isfinite(estimate).all() and np.isfinite(reference).all()):
        raise ValueError("estimate and reference must contain only finite values")

    reference_magnitude = np.abs(reference)
    magnitude_difference = np.abs(estimate) - reference_magnitude
    with np.errstate(over="ignore", invalid="ignore"):
        magnitude_error = float(np.sum(magnitude_difference ** 2, dtype=np.float64))
        complex_error = float(np.sum(np.abs(estimate - reference) ** 2, dtype=np.float64))
        reference_energy = float(np.sum(reference_magnitude ** 2, dtype=np.float64))
    if not all(math.isfinite(x) for x in (magnitude_error, complex_error, reference_energy)):
        raise ValueError("metric calculation overflowed float64")
    mse = magnitude_error / reference.size
    peak = float(reference_magnitude.max())
    zero_reference = reference_energy == 0
    if zero_reference:
        magnitude_nmse = complex_nmse = magnitude_psnr = float("nan")
        nmse_status = psnr_status = "undefined_zero_reference"
    else:
        magnitude_nmse = magnitude_error / reference_energy
        complex_nmse = complex_error / reference_energy
        if not (math.isfinite(magnitude_nmse) and math.isfinite(complex_nmse)):
            raise ValueError("NMSE calculation overflowed float64")
        nmse_status = "finite"
        if mse == 0:
            magnitude_psnr, psnr_status = float("inf"), "positive_infinity"
        else:
            magnitude_psnr = 20 * math.log10(peak) - 10 * math.log10(mse)
            psnr_status = "finite"
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "magnitude_nmse": magnitude_nmse,
        "complex_nmse": complex_nmse,
        "nmse_status": nmse_status,
        "magnitude_rmse": math.sqrt(mse),
        "magnitude_psnr_db": magnitude_psnr,
        "magnitude_psnr_status": psnr_status,
        "reference_peak": peak,
        # Legacy runner name; this is the peak, not max(reference)-min(reference).
        "data_range": peak,
    }


def summarize_slice_quality(rows) -> dict:
    """Arithmetic mean of per-slice values, without dropping nonfinite rows."""
    rows = list(rows)
    if not rows:
        raise ValueError("at least one slice is required")
    result = {}
    for name in IMAGE_METRICS:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            result[name + "_mean"] = float(np.mean(values))
        result[name + "_nonfinite_count"] = int((~np.isfinite(values)).sum())
    return result
