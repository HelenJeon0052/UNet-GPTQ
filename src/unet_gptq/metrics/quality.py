"""Reference-based per-slice metrics shared by experiment entry points."""
from __future__ import annotations

import math
import numpy as np


def slice_quality(estimate, target) -> dict:
    """Compare one 2-D slice without independently rescaling either image.

    Magnitude NMSE ignores phase; complex NMSE includes phase. PSNR uses the
    maximum magnitude of this reference slice. Undefined or infinite values
    are represented by None plus explicit flags for JSON/CSV.
    """
    estimate, target = np.asarray(estimate), np.asarray(target)
    if estimate.ndim != 2 or estimate.shape != target.shape or not estimate.size:
        raise ValueError("estimate and target must be nonempty matching 2-D slices")
    if not (np.issubdtype(estimate.dtype, np.number) and np.issubdtype(target.dtype, np.number)):
        raise ValueError("estimate and target must be numeric")
    estimate = estimate.astype(np.complex128)
    target = target.astype(np.complex128)
    if not (np.isfinite(estimate).all() and np.isfinite(target).all()):
        raise ValueError("estimate and target must be finite")
    magnitude = np.abs(target)
    difference = np.abs(estimate) - magnitude
    magnitude_error = float(np.sum(difference ** 2))
    complex_error = float(np.sum(np.abs(estimate - target) ** 2))
    reference_energy = float(np.sum(magnitude ** 2))
    mse = magnitude_error / target.size
    peak = float(magnitude.max())
    psnr = 10.0 * math.log10(peak * peak / mse) if mse > 0 and peak > 0 else None
    return {
        "magnitude_nmse": magnitude_error / reference_energy if reference_energy else None,
        "complex_nmse": complex_error / reference_energy if reference_energy else None,
        "magnitude_rmse": math.sqrt(mse),
        "psnr_db": psnr,
        "psnr_is_posinf": bool(mse == 0 and peak > 0),
        "psnr_is_neginf": bool(mse > 0 and peak == 0),
        "data_range": peak,
    }
