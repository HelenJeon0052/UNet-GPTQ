"""
The equivalent single-coil measurement used by mixture_duo is
    k_obs = M * fft2_unshifted(x)
where x is the coil-combined complex image on the DiffuseRecon grid.
"""
from __future__ import annotations
import json
import numpy as np
import torch
import hashlib

from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Literal
from numbers import Integral



SensMethod = Literal["acs_sos", "espirit"]
CropSize = int | tuple[int, int]
_SPATIAL = (-2, -1)

def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)

def _crop_shape(crop_size: CropSize) -> tuple[int, int]:
    if isinstance(crop_size, Integral):
        size = _positive_int(crop_size, "crop_size")
        return size, size
    
    if not isinstance(crop_size, tuple) or len(crop_size) != 2:
        raise ValueError(f"crop size must be an integer or a (h, w) tuple")
    
    return (_positive_int(crop_size[0], "crop_h"),
            _positive_int(crop_size[1], "crop_w")
    )

def _require_complex(data: torch.Tensor, name: str, min_ndim: int = 2) -> None:
    if not torch.is_complex(data):
        raise TypeError(f"{name} requires complex data, got {data.dtype}")
    if data.ndim < min_ndim or any(d == 0 for d in data.shape):
        raise ValueError(f"{name} requires non-empty data over {min_ndim}, got {tuple(data.shape)}")

def fft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Centered, orthonormal 2-D FFT
    Use only for crop/coil images
    """
    
    _require_complex(data, "fft2c")

    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fft2(data, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(data, dim=(-2, -1))

def ifft2c(data: torch.Tensor) ->  torch.Tensor:
    """
    Centered, orthonormal 2-D iFFT
    Use only for crop/coil images
    """
    
    _require_complex(data, "ifft2c")

    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifft2(data, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(data, dim=(-2, -1))

def fft2_unshifted(data: torch.Tensor) -> torch.Tensor:
    """
    DiffuseRecon / mixture_duo k-space: torch.fft.fft2, norm=backward
    """
    
    _require_complex(data, "fft2c_unshifted")

    return torch.fft.fft2(data, dim=(-2, -1), norm="backward")

def ifft_unshifted(data: torch.Tensor) -> torch.Tensor:
    
    _require_complex(data, "ifft_unshifted")
    
    return torch.fft.ifft2(data, dim=(-2, -1), norm="backward")
    
def center_crop(data: torch.Tensor, crop_size: CropSize) -> torch.Tensor:

    if data.ndim < 2:
        raise ValueError(f"center crop requires at least two dimensions")

    ch, cw = _crop_shape(crop_size)
    h, w = data.shape[-2:]
    if ch > h or cw > w:
        raise ValueError(f"image {h}, {w} is smaller than cropped size {ch}, {cw}")
    
    top = (h - ch) // 2
    left = (w - cw) // 2

    return data[..., top: top + ch, left: left + cw]

def rss_magnitude(coil_image: torch.Tensor) -> torch.Tensor:
    _require_complex(coil_image, "rss_magnitude", min_ndim = 3)
    return torch.sqrt((coil_image.abs()**2).sum(dim=-3))

def mvue(coil_images: torch.Tensor, sens: torch.Tensor, *, eps: float = 1e-8) ->  torch.Tensor:

    _require_complex(coil_images, "mvue", min_ndim = 3)
    _require_complex(sens, "mvue sensitivity", min_ndim = 3)

    if coil_images.shape != sens.shape:
        raise ValueError(f"coil_images and sensitivity must contain same size of shape")
    if coil_images.device != sens.device or coil_images.dtype != sens.dtype:
        raise ValueError(f"coil_images and sensitivity must contain same dtype and device")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive")
    weight = (sens.abs()**2).sum(dim=-3).clamp_min(eps)
    return (sens.conj() * coil_images).sum(dim=-3) / weight


def _acs_sos_sensitivity(ks: torch.Tensor, acs_width: int) -> torch.Tensor:
    """
    ks == k-space
    low-resolution SOS sensitivity from fully sampled ACS (explicit, not ESPIRiT)
    centered k-space has Data consistency (H//2, W//2)
    """

    width = _positive_int(acs_width, "acs_width")
    coils, h, w = ks.shape
    if width > min(h, w):
        raise ValueError(f"asc_width = {width} is invalid for shape: {(h, w)}")
    
    cy, cx = h // 2, w // 2
    half = width // 2
    top = cy - half
    left = cx - half
    low = torch.zeros_like(ks)

    low[..., top : top + width, left : left + width] = ks[..., top : top + width, left : left + width]

    coil_low = ifft2c(low)
    denom = rss_magnitude(coil_low).clamp_min(1e-8)
    return coil_low / denom

def _espirit_sensitivity(ks: torch.Tensor, calib_width: int) ->  torch.Tensor:

    width = _positive_int(calib_width, "calib_width")

    if width > min(ks.shape[-2:]):
        raise ValueError(f"calib width={width} exceeds spatial dimensions")
    
    if width < 6:
        raise ValueError(f"ESPRiT calib width must be larger than kernel width 6")
    
    try:
        import sigpy as sp
        from sigpy.mri.app import EspiritCalib
    except ImportError as e:
        raise ImportError(f"Espirit method requires sigpy") from e
    
    k_np = np.asarray(ks.detach().cpu().numpy())
    maps = EspiritCalib(k_np, calib_width=calib_width, show_pbar=True).run()
    sens = torch.from_numpy(np.asarray(maps)).to(device=ks.device, dtype=ks.dtype)

    if sens.shape != ks.shape or not torch.isfinite(sens).all():
        raise ValueError(f"ESPRiT returned invalid sensitivity maps")
        
    return sens

def estimate_sensitivity(
    ks: torch.Tensor,
    *,
    method: SensMethod = "acs_sos",
    acs_width: int = (3*8),
    calib_width: int = (3*8),
) -> torch.Tensor:

    """
    k-space is centered coil k-space with shape [C, H, W]
    """
    _require_complex(ks, "estimate_senstivity", min_ndim=3)
    if ks.ndim != 3:
        raise ValueError("estimate_sensitivity expects [C,H,W]")
    if not torch.isfinite(ks).all():
        raise ValueError("k-space contains NaN or Inf")

    if method == "acs_sos":
        return _acs_sos_sensitivity(ks, acs_width)
    if method == "espirit":
        return _espirit_sensitivity(ks, calib_width)
    
    raise ValueError(f"unknown sens_method={method!r}")

def _as_complex64(array: np.ndarray) -> torch.Tensor:
    if not np.iscomplexobj(array):
        raise TypeError(f"expected k-space as complex")
    if not np.isfinite(array).all():
        raise ValueError("k-space contains NaN or Inf")

    result = torch.from_numpy(np.ascontiguousarray(array, dtype=np.complex64))
    if not torch.isfinite(result).all():
        raise ValueError("k-space overflows complex64")

    return result

def export_sens(ks, coil_images, width, sens_method):
    
    sens = torch.stack(
        [
            estimate_sensitivity(
                ks[i], method=sens_method, acs_width=width,  calib_width=width,
            )
            for i in range(2)
        ]
    )

    image = torch.stack(
        [mvue(coil_images[i], sens[i]) for i in range(2)]
    )

    return sens, image


class FastMRISlicePairs(Dataset):
    """
    Lazy HDF5 adjacent-slice loader backed by a deterministic JSONL manifest.

    Each manifest line:
        {"path": "/data/file.h5", "slice": 12, "split": "pilot", "contrast": "FLAIR"}
    `slice` is the first index of the pair (slice, slice+1).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        crop_size: CropSize = 320,
        acs_width: int = (3*8),
        calib_width: int = (3*8),
        sens_method: SensMethod = "acs_sos",
        data_root: str | Path | None = None
        ) -> None:
            
            if sens_method not in ("acs_sos", "espirit"):
                raise ValueError(f"unknown sensitivity method")

            if split is not None:
                if not isinstance(split, str):
                    raise ValueError(f"split must be a string type, got {type(split).__name__}")
            
            self.crop_size = _crop_shape(crop_size)
            self.acs_width = _positive_int(acs_width, "acs_width")
            self.calib_width = _positive_int(calib_width, "calib_width")
            self.sens_method = sens_method
            
            self.split = split
            self.manifest_path = Path(manifest_path).expanduser().resolve()
            self.data_root = (Path(data_root).expanduser().resolve() if data_root is not None else self.manifest_path.parent)
            
            payload = self.manifest_path.read_bytes()
            self.manifest_sha256 = hashlib.sha256(payload).hexdigest()

            records: list[dict[str, Any]] = []

            for line_n, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"manifest line {line_n}: invalid JSON") from e
                if not isinstance(record, dict):
                    raise ValueError(f"manifest line {line_n}: expected JSON object")
                
                path, slice_index = record.get("path"), record.get("slice")
                if not isinstance(path, str) or not path:
                    raise ValueError(f"manifest line {line_n}: path must be a string or exists")
                if isinstance(slice_index, bool) or not isinstance(slice_index, int) or slice_index < 0:
                    raise ValueError(f"manifest line {line_n}: slice_index must be an int or exists")
                for key in ("split", "contrast", "acquisition"):
                    if record.get(key) is not None and not isinstance(record[key], str):
                        raise ValueError(f"manifest line {line_n}: key must be a string")
                if split is None or record.get("split") == split:
                    records.append(record)

            if not records:
                raise ValueError(f"manifest contains no samples for split == {split!r}")
            
            self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "manifest_path": str(self.manifest_path),
            "data_root": str(self.data_root), "split": self.split,
            "num_pairs": len(self), "crop_size": list(self.crop_size),
            "acs_width": self.acs_width, "calib_width": self.calib_width,
            "sens_method": self.sens_method, "dtype": "complex64",
            "acs_geometry": "square_dc_aligned",
            "acs_normalization_eps": 1e-8, "coil_combine_eps": 1e-8,
            "source_fft": "centered/ortho", "equivalent_fft": "unshifted/backward",
            "measurement_model": "synthetic_singlecoil_from_combined_reference",
            "normalization": "none", "rss_target_axis": "two_adjacent_slices",
        }
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        import h5py

        record = self.records[index]
        path = Path(record["path"]).expanduser()
        path = (path if path.is_absolute() else self.data_root / path).resolve()
        slice_index = int(record["slice"])
        target_array, target_source, acquisition = None, "rss_from_coils", ""
        with h5py.File(path, "r") as volume:
            if "kspace" not in volume:
                raise KeyError(f"{path}: missing kspace dataset")
            source = volume["kspace"]
            if source.ndim not in (3, 4):
                raise ValueError(f"{path}: expected [S,H,W] or [S,C,H,W]")
            if slice_index + 1 >= source.shape[0]:
                raise ValueError(f"{path}: need adjacent slices at {slice_index}; volume has {source.shape[0]}")
            raw = np.asarray(source[slice_index:slice_index + 2])
            attr = volume.attrs.get("acquisition", "")
            acquisition = attr.decode("utf-8") if isinstance(attr, bytes) else str(attr)
            target_key = ("reconstruction_rss" if "reconstruction_rss" in volume else
                          "reconstruction_esc" if source.ndim == 3 and "reconstruction_esc" in volume else None)
            if target_key is not None:
                target_array = np.asarray(volume[target_key][slice_index:slice_index + 2])
                target_source = target_key

        ks = _as_complex64(raw)
        if ks.ndim == 3:
            ks = ks[:, None]
        if any(d == 0 for d in ks.shape):
            raise ValueError(f"{path}: empty k-space dimension")
        n_coils = ks.shape[1]
        coil_images = ifft2c(ks)
        if n_coils == 1:
            sens = torch.ones_like(coil_images)
            image = coil_images[:, 0]
        else:
            sens = torch.stack([
                estimate_sensitivity(k, method=self.sens_method,
                                     acs_width=self.acs_width, calib_width=self.calib_width)
                for k in ks
            ])
            image = mvue(coil_images, sens)
        image = center_crop(image, self.crop_size)
        sens = center_crop(sens, self.crop_size)
        rss_crop = center_crop(rss_magnitude(coil_images), self.crop_size)
        if target_array is None:
            target = rss_crop.clone()
        else:
            if (target_array.ndim != 3 or target_array.shape[0] != 2
                    or np.iscomplexobj(target_array)):
                raise ValueError(f"{path}: expected real target [2,H,W], got {target_array.shape}")
            target = torch.from_numpy(np.ascontiguousarray(target_array, dtype=np.float32))
            if not torch.isfinite(target).all() or (target < 0).any():
                raise ValueError(f"{path}: magnitude target must be finite and nonnegative")
            target = center_crop(target, self.crop_size)
        if not torch.isfinite(image).all() or not torch.isfinite(rss_crop).all():
            raise ValueError(f"{path}: coil combination produced NaN or Inf")
        ks_eq = fft2_unshifted(image)
        if not torch.isfinite(ks_eq).all():
            raise ValueError(f"{path}: equivalent k-space overflows complex64")
        return {
            "image": image, "ks_eq": ks_eq, "target_rss": target,
            "rss_from_coils": rss_crop, "sens": sens,
            "coil_combine_valid": sens.abs().square().sum(dim=-3) > 1e-8,
            "n_coils": n_coils, "filename": path.name, "path": str(path),
            "slice_index": slice_index, "split": record.get("split") or "",
            "contrast": record.get("contrast") or record.get("acquisition") or acquisition,
            "target_source": target_source,
            "measurement_model": "synthetic_singlecoil_from_combined_reference",
            "manifest_sha256": self.manifest_sha256,
        }

def image_to_4ch(image: torch.Tensor) -> torch.Tensor:
    """
    complex [2,H,W] to Real [4,H,W]
    (Re1,Im1,Re2,Im2)
    """
    _require_complex(image, "image_to_ch", min_ndim=3)
    if image.shape[-3] != 2:
        raise ValueError(f"image shape expected two adjacent slices, but got {image.shape[-3]}")
    return torch.stack(
        [image[..., 0, :, :].real, image[..., 0, :, :].imag, image[..., 1, :, :].real, image[..., 1, :, :].imag], dim = -3
    )

def fourch_to_image(channels: torch.Tensor) ->  torch.Tensor:
    """
    Real to complex
    """

    if channels.ndim < 3 or channels.shape[-3] != 4:
        raise ValueError(f"an input must be [..., C, H, W]")
    if torch.is_complex(channels) or channels.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"a type of input must be real numbers")
    return torch.stack(
        [torch.complex(channels[..., 0, :, :], channels[..., 1, :, :]), torch.complex(channels[..., 2, :, :], channels[..., 3, :, :])], dim=-3
    )

def collate_slice_pairs(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = ("image", "ks_eq", "target_rss", "rss_from_coils", "coil_combine_valid")
    output = {key: torch.stack([s[key] for s in samples]) for key in fixed}
    for key in samples[0]:
        if key not in fixed:
            output[key] = [s[key] for s in samples]
    return output