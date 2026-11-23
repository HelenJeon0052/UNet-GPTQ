from __future__ import annotations
import torch


from unet_gptq.domain.data_consistency import observed_residual
from unet_gptq.metrics.quality import (
    IMAGE_METRICS,
    METRIC_SCHEMA_VERSION,
    slice_quality,
    summarize_slice_quality,
)



def image_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor, observed_kspace: torch.Tensor, mask: torch.Tensor
) -> dict:


    if reconstruction.shape != target.shape or reconstruction.shape != observed_kspace.shape:
        raise ValueError(f"Reconstruction, target and observations must take identical shapes")
    
    if reconstruction.ndim != 4 or reconstruction.shape[1] != 2 or any(d == 0 for d  in reconstruction.shape):
        raise ValueError(f"expected non-empty complex pairs with shape [B, 2, H, W]")
        
    dc = observed_residual(reconstruction, observed_kspace, mask).item()
    
    estimates = reconstruction.detach().cpu().numpy()
    references = target.detach().cpu().numpy()

    rows = []
    for batch in range(references.shape[0]):
        for slot in range(references.shape[1]):
            rows.append(
                {
                    "batch_index": batch,
                    "slice_in_pair": slot,
                    **slice_quality(
                        estimates[batch, slot],
                        references[batch, slot],
                    )
                }
            )
    
    summary = summarize_slice_quality(rows)

    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "n_slices": len(rows),
        **{
            name: summary[name + "_mean"]
            for name in IMAGE_METRICS
        },
        **{
            name + "_nonfinite_count": summary[name + "_nonfinite_count"]
            for name in IMAGE_METRICS
        },
        "per_slice": rows,
        "sampled_kspace_relative": dc,
    }

