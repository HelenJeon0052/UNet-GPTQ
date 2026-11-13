"""Adjacent-pair overlap aggregation, independent of the tensor backend."""
from __future__ import annotations


def overlapping_pair_components(coarse_pairs):
    """Return reconstructed central slices for pairs (i-1,i),(i,i+1),(i+1,i+2).

    Inputs contain real/imaginary channels [B,4,H,W]. Averaging is performed
    in complex-equivalent real/imaginary channels, before refinement.
    """
    if len(coarse_pairs) != 3:
        raise ValueError("Three adjacent coarse pairs are required")
    shape = coarse_pairs[0].shape
    if len(shape) != 4 or shape[1] != 4 or any(value.shape != shape for value in coarse_pairs):
        raise ValueError("Expected three identically shaped [B,4,H,W] coarse pairs")
    return ((coarse_pairs[0][:, 2:4] + coarse_pairs[1][:, 0:2]) / 2,
            (coarse_pairs[1][:, 2:4] + coarse_pairs[2][:, 0:2]) / 2)
