from __future__ import annotations
import torch


from unet_gptq.domain.data_consistency import observed_residual

def image_metrics(
    reconstruction: torch.Tensor, target: torch.Tensor, observed_kspace: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:


    if reconstruction.shape != target.shape or reconstruction.shape != observation_kspace.shape:
        raise ValueError(f"Reconstruction, target and observations must take identical shapes")
    
    err = (reconstruction - target).abs().float()
    reference = target.abs().float()
    nmse = (err.square().sum() / reference.square().sum().clamp_min(1e-12)).item()
    peak = reference.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    mse = (err / peak).square().mean(dim=(-2, -1))
    psnr = (-10 * torch.log10(mse.clamp_min(1e-12))).mean().item()

    return {
        "nmse": nmse, "psnr": psnr, "sampled_kspace_relative": observed_residual(reconstruction, observed_kspace, mask).item()
    }

