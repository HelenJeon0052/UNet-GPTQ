"""Regression tests for Fourier conventions and the adjacent-slice data contract.



All numerical references use NumPy directly so an incorrect pair of forward and
inverse Torch functions cannot pass merely by cancelling its own mistakes.
Run from this directory with: python -m pytest -q tests/test_data_pipeline.py
"""
from __future__ import annotations
import json
import numpy as np

import torch
import h5py

import sys
import types
from pathlib import Path
import pytest

from unet_gptq.datas import fastmri_dataset as fm
from unet_gptq.datas.fastmri_plus import boxes_to_roi_mask
from unet_gptq.domain.forward_operator import cartesian_mask



# ================== utils ==================

def _np_fft2c(value: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(value, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def _np_ifft2c(value: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(value, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def _complex_fixture(shape: tuple[int, ...], seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)


def _np_crop(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = value.shape[-2:]
    crop_h, crop_w = shape
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return value[..., top : top + crop_h, left : left + crop_w]


def _write_volume(
    path: Path,
    kspace: np.ndarray,
    *,
    target: np.ndarray | None = None,
    target_key: str = "reconstruction_rss",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as volume:
        volume.create_dataset("kspace", data=kspace)
        if target is not None:
            volume.create_dataset(target_key, data=target)
        volume.attrs["acquisition"] = "AXFLAIR"


def _write_manifest(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def _record(path: Path, **overrides) -> dict:
    record = {"path": str(path), "slice": 0, "split": "train", "contrast": "FLAIR"}
    record.update(overrides)
    return record

# ================== tests ==================

def test_fft_round_trip() -> None:
    x = torch.randn(3, 24, 20) + 1j * torch.randn(3, 24, 20)
    r = fm.fft2c(x)
    torch.testing.assert_close(fm.ifft2c(r), x, rtol=1e-5, atol=1e-5)

def test_mask() -> None:
    """
    a test for checking that mask is deterministic or has an exact count
    """
    a = cartesian_mask(100, 100, acceleration = 6, center_fraction = 0.08, seed = 10)
    b = cartesian_mask(100, 100, acceleration = 6, center_fraction = 0.08, seed = 10)
    assert torch.equal(a, b)
    assert a.shape[-2:] == (100, 100)
    assert bool(((a == 0) | (a == 1)).all())

    rows = a.reshape(-1, 100)
    torch.testing.assert_close(
        rows, rows[:1].expand_as(rows), rtol=0, atol=0
    )
    expected_col = round(100 / 6)
    assert int(rows[0].sum()) == expected_col
    assert int(a.sum()) == rows.shape[0] * expected_col
    assert bool(rows[0, 46:54].all())    

def test_fastmri_plus() -> None:
    """
    a test for checking that a fastmri is inverted
    """
    box = {"x": 10, "y": 20, "width": 5, "height": 10, "label": "lesion"}
    mask, boxes = boxes_to_roi_mask([box], (100, 100), (100, 100))
    assert boxes[0] == {"x_0": 10, "y_0": 70, "x_1": 15, "y_1": 80, "label": "lesion"}
    assert int(mask.sum()) == 50

def test_dataset_contract(tmp_path) -> None:
    kspace = _complex_fixture((2, 4, 40, 32))
    coils = _np_ifft2c(kspace)
    rss = np.sqrt(np.sum(np.abs(coils) ** 2, axis=1)).astype(np.float32)
    h5_path = tmp_path / "file_brain_AXFLAIR_example.h5"
    _write_volume(h5_path, kspace, target=rss)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(h5_path)])
    dataset = fm.FastMRISlicePairs(manifest, split="train", crop_size=32, acs_width=8)
    sample, repeated = dataset[0], dataset[0]

    low_kspace = np.zeros_like(kspace)
    low_kspace[..., 16:24, 12:20] = kspace[..., 16:24, 12:20]
    low_coils = _np_ifft2c(low_kspace)
    denominator = np.maximum(np.sqrt(np.sum(np.abs(low_coils) ** 2, axis=1)), 1e-8)
    expected_sens = low_coils / denominator[:, None]
    weights = np.maximum(np.sum(np.abs(expected_sens) ** 2, axis=1), 1e-8)
    expected_image = np.sum(np.conjugate(expected_sens) * coils, axis=1) / weights
    expected_image = _np_crop(expected_image, (32, 32)).astype(np.complex64)

    shapes = {
        "image": (2, 32, 32),
        "ks_eq": (2, 32, 32),
        "target_rss": (2, 32, 32),
        "rss_from_coils": (2, 32, 32),
        "sens": (2, 4, 32, 32),
    }
    for key, shape in shapes.items():
        assert sample[key].shape == shape
        assert sample[key].device.type == "cpu"
        assert sample[key].dtype == (
            torch.complex64 if key in {"image", "ks_eq", "sens"} else torch.float32
        )
        torch.testing.assert_close(sample[key], repeated[key], rtol=0, atol=0)
    torch.testing.assert_close(sample["image"], torch.from_numpy(expected_image), rtol=3e-5, atol=1e-5)
    torch.testing.assert_close(
        sample["sens"],
        torch.from_numpy(_np_crop(expected_sens, (32, 32))).to(torch.complex64),
        rtol=3e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        sample["ks_eq"],
        torch.from_numpy(np.fft.fft2(expected_image, axes=(-2, -1), norm="backward")).to(torch.complex64),
        rtol=3e-5,
        atol=1e-4,
    )
    for key in ("target_rss", "rss_from_coils"):
        torch.testing.assert_close(sample[key], torch.from_numpy(_np_crop(rss, (32, 32))), rtol=2e-5, atol=1e-5)
    assert sample["n_coils"] == 4
    assert sample["slice_index"] == 0
    for key in ("filename", "path", "split", "contrast", "target_source"):
        assert isinstance(sample[key], str)
        assert sample[key]
    assert sample["filename"] == h5_path.name
    assert Path(sample["path"]) == h5_path.resolve()

@pytest.mark.parametrize("with_esc", [True, False])
def test_singlecoil(tmp_path, with_esc) -> None:
    """
    goal: to test single coil dataset using esc or magnitude fallbcak
    """

    kspace = _complex_fixture((2, 20, 18))
    image = _np_ifft2c(kspace).astype(np.complex64)
    # to detect ignoring target of HDF5, a different value of coil magnitude is used
    esc = np.full((2, 20, 18), 7.0, dtype=np.float32)
    h5_path = tmp_path / "single.h5"
    _write_volume(h5_path, kspace, target=esc if with_esc else None, target_key = "reconstruction_esc")
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(h5_path)])
    sample = fm.FastMRISlicePairs(manifest, crop_size=(16, 14))[0]
    torch.testing.assert_close(sample["image"], torch.from_numpy(_np_crop(image, (16, 14))), rtol=2e-5, atol=1e-5)
    expected_target = esc if with_esc else np.abs(image).astype(np.float32)
    torch.testing.assert_close(sample["target_rss"], torch.from_numpy(_np_crop(expected_target, (16, 14))), rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(sample["sens"], torch.ones((2, 1, 16, 14), dtype=torch.complex64))
    assert sample["n_coils"] == 1

@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("fft2c", _np_fft2c),
        ("ifft2c", _np_ifft2c),
        ("fft2_unshifted", lambda x: np.fft.fft2(x, axes=(-2, -1), norm="backward")),
        ("ifft_unshifted", lambda x: np.fft.ifft2(x, axes=(-2, -1), norm="backward")),
    ],
)

def test_fft_matches_independent_numpy_reference(name, reference) -> None:
    # Odd spatial sizes expose incorrect shift order. Leading dimensions must
    # remain arbitrary; the pair dimension is not a real/imaginary channel axis.
    for shape in ((7, 9), (2, 3, 7, 9)):
        array = _complex_fixture(shape)
        actual = getattr(fm, name)(torch.from_numpy(array))
        expected = torch.from_numpy(reference(array)).to(torch.complex64)
        assert actual.shape == array.shape
        assert actual.is_complex()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_fft_rejects_real_data_and_missing_spatial_dimensions() -> None:
    for name in ("fft2c", "ifft2c", "fft2_unshifted", "ifft_unshifted"):
        function = getattr(fm, name)
        with pytest.raises((TypeError, ValueError)):
            function(torch.ones(4, 5))
        with pytest.raises((TypeError, ValueError)):
            function(torch.ones(5, dtype=torch.complex64))


def test_center_crop_rectangular_contract_and_validation() -> None:
    value = torch.arange(2 * 9 * 11).reshape(2, 9, 11)
    torch.testing.assert_close(fm.center_crop(value, (5, 7)), value[..., 2:7, 2:9])
    torch.testing.assert_close(fm.center_crop(value, 5), value[..., 2:7, 3:8])
    for invalid in (0, -1, True, 3.5, (0, 4), (3, True), (3, 4.5), (3,), (10, 7)):
        with pytest.raises((TypeError, ValueError)):
            fm.center_crop(value, invalid)


def test_rss_supports_batches_and_zero_coil_images() -> None:
    coils = torch.zeros(2, 3, 4, 5, dtype=torch.complex64)
    coils[:, 0] = 3.0
    coils[:, 1] = 4.0j
    actual = fm.rss_magnitude(coils)
    assert actual.shape == (2, 4, 5)
    torch.testing.assert_close(actual, torch.full((2, 4, 5), 5.0))
    torch.testing.assert_close(fm.rss_magnitude(torch.zeros_like(coils)), torch.zeros_like(actual))


def test_mvue_recovers_known_complex_image_and_handles_zero_support() -> None:
    image = torch.from_numpy(_complex_fixture((2, 7, 9)))
    sens = torch.tensor([1.0 + 0.5j, 0.2 - 0.7j, -0.3 + 0.4j], dtype=torch.complex64)
    sens = sens.reshape(1, 3, 1, 1).expand(2, 3, 7, 9).clone()
    coil_images = sens * image[:, None]
    actual = fm.mvue(coil_images, sens)
    torch.testing.assert_close(actual, image, rtol=1e-5, atol=1e-5)
    zero = fm.mvue(torch.zeros_like(coil_images), torch.zeros_like(sens), eps=1e-8)
    assert torch.isfinite(zero).all()
    torch.testing.assert_close(zero, torch.zeros_like(image))


def test_multicoil_missing_target_falls_back_to_coil_rss(tmp_path) -> None:
    kspace = _complex_fixture((2, 3, 20, 18))
    h5_path = tmp_path / "multi.h5"
    _write_volume(h5_path, kspace)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(h5_path)])
    sample = fm.FastMRISlicePairs(manifest, crop_size=16, acs_width=6)[0]
    torch.testing.assert_close(sample["target_rss"], sample["rss_from_coils"])
    assert isinstance(sample["target_source"], str)


@pytest.mark.parametrize("invalid_slice", [None, True, 0.5, "1"])
def test_dataset_rejects_missing_or_noninteger_slice(tmp_path, invalid_slice) -> None:
    h5_path = tmp_path / "valid.h5"
    _write_volume(h5_path, _complex_fixture((2, 20, 18)))
    record = _record(h5_path)
    if invalid_slice is None:
        del record["slice"]
    else:
        record["slice"] = invalid_slice
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [record])
    with pytest.raises(ValueError, match="manifest line 1: slice_index"):
        fm.FastMRISlicePairs(manifest, crop_size=16)[0]


@pytest.mark.parametrize("slice_index", [-1, 1, 2])
def test_dataset_rejects_pairs_outside_volume(tmp_path, slice_index) -> None:
    h5_path = tmp_path / "two_slices.h5"
    _write_volume(h5_path, _complex_fixture((2, 20, 18)))
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(h5_path, slice=slice_index)])
    with pytest.raises((IndexError, ValueError)):
        fm.FastMRISlicePairs(manifest, crop_size=16)[0]

def test_espirit_executes_calibration_and_forwards_calib_width(tmp_path, monkeypatch) -> None:
    calls = []

    class FakeEspiritCalib:
        def __init__(self, kspace, *, calib_width, **kwargs):
            self.kspace = np.asarray(kspace)
            self.call = {"calib_width": calib_width, "run": False}
            calls.append(self.call)

        def run(self):
            self.call["run"] = True
            return np.ones_like(self.kspace, dtype=np.complex64)

    sigpy = types.ModuleType("sigpy")
    mri = types.ModuleType("sigpy.mri")
    app = types.ModuleType("sigpy.mri.app")
    app.EspiritCalib = FakeEspiritCalib
    mri.app = app
    sigpy.mri = mri
    sigpy.to_device = lambda array, *args, **kwargs: np.asarray(array)
    for name, module in (("sigpy", sigpy), ("sigpy.mri", mri), ("sigpy.mri.app", app)):
        monkeypatch.setitem(sys.modules, name, module)

    kspace = _complex_fixture((2, 3, 20, 18))
    sens = fm.estimate_sensitivity(
        torch.from_numpy(kspace[0]), method="espirit", acs_width=6, calib_width=11
    )
    assert sens.shape == (3, 20, 18)
    assert sens.dtype == torch.complex64
    torch.testing.assert_close(sens, torch.ones_like(sens))
    h5_path = tmp_path / "espirit.h5"
    _write_volume(h5_path, kspace)
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(h5_path)])
    sample = fm.FastMRISlicePairs(
        manifest, crop_size=16, sens_method="espirit", acs_width=6, calib_width=11
    )[0]
    assert sample["sens"].shape == (2, 3, 16, 16)
    assert len(calls) == 3
    assert all(call == {"calib_width": 11, "run": True} for call in calls)


def test_collation_handles_variable_coil_counts_and_stacks_fixed_fields(tmp_path) -> None:
    paths = [tmp_path / "single.h5", tmp_path / "multi.h5"]
    _write_volume(paths[0], _complex_fixture((2, 20, 18)))
    _write_volume(paths[1], _complex_fixture((2, 3, 20, 18)))
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [_record(path) for path in paths])
    dataset = fm.FastMRISlicePairs(manifest, crop_size=16, acs_width=6)
    samples = [dataset[0], dataset[1]]
    batch = fm.collate_slice_pairs(samples)
    for key in ("image", "ks_eq", "target_rss", "rss_from_coils"):
        assert batch[key].shape == (2, 2, 16, 16)
        torch.testing.assert_close(batch[key], torch.stack([sample[key] for sample in samples]))
    assert isinstance(batch["sens"], list)
    assert [value.shape for value in batch["sens"]] == [(2, 1, 16, 16), (2, 3, 16, 16)]
    for index in range(2):
        torch.testing.assert_close(batch["sens"][index], samples[index]["sens"])

def test_mvue_honors_custom_eps():
    coils = torch.ones((1, 2, 3), dtype=torch.complex64)
    actual = fm.mvue(coils, torch.ones_like(coils), eps=4.0)
    expected = torch.full((2, 3), 0.25, dtype=torch.complex64)
    torch.testing.assert_close(actual, expected)

def test_fourch_conversion_preserves_channel_order():
    image = torch.tensor(
        [[[1 + 2j]], [[3 + 4j]]], dtype=torch.complex64
    )
    channels = torch.tensor([1., 2., 3., 4.]).reshape(4, 1, 1)
    torch.testing.assert_close(fm.image_to_4ch(image), channels)
    torch.testing.assert_close(fm.fourch_to_image(channels), image)
    torch.testing.assert_close(fm.fourch_to_image(channels[None]), image[None])

def test_dataset_provenance_preserves_config(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    record = {"path": "unused.h5", "slice": 0, "split": "train"}
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    for crop_size, expected in [(16, [16, 16]), ((16, 14), [16, 14])]:
        dataset = fm.FastMRISlicePairs(
            manifest, split="train", crop_size=crop_size
        )
        info = dataset.provenance()
        assert info["split"] == "train"
        assert info["crop_size"] == expected
        assert info["num_pairs"] == 1

def test_dataset_reports_invalid_json_line(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{broken json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest line 1: invalid JSON"):
        fm.FastMRISlicePairs(manifest)