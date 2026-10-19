"""CPU integration tests; no DiffuseRecon checkpoint or reconstruction implied."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from unet_gptq.quantization.calibration import (
    CalibrationSample, CalibrationStage, calibrate_conv2d_model, model_timesteps, select_timesteps,
)
from unet_gptq.quantization.gptq import GPTQConfig
from unet_gptq.train.calibrate_gptq import load_calibration_samples, main as calibrate_main
from unet_gptq.train.run_exp import (
    Case, TorchBackend, create_argparser, sha256_file, validate_quant_manifest,
)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Conv2d(4, 5, 3, padding=1)
        self.depthwise = nn.Conv2d(5, 5, 3, padding=1, groups=5)
        self.time = nn.Linear(1, 5)
        self.out_conv = nn.Conv2d(5, 4, 1)
        self.unused = nn.Conv2d(4, 4, 1)
        self.string_padding = nn.Conv2d(4, 4, 3, padding="same")
        self.register_buffer("constant", torch.tensor(0.01))

    def convert_to_fp16(self):
        # Match the mixed torso/output policy of DiffuseRecon rather than .half().
        self.in_conv.half()
        self.depthwise.half()

    def forward(self, x, timesteps):
        h = self.in_conv(x.to(self.in_conv.weight.dtype))
        emb = self.time(timesteps.float()[:, None] / 1000).to(h.dtype)[:, :, None, None]
        h = self.depthwise(torch.nn.functional.silu(h + emb))
        return self.out_conv(h.float()) + self.constant


class TinySpacedDiffusion:
    def __init__(self, mapping=(0, 3, 8, 13, 19)):
        self.timestep_map = list(mapping)
        self.num_timesteps = len(mapping)
        self.original_num_steps = 20
        self.rescale_timesteps = True
        self.alphas_cumprod = np.exp(-np.asarray(mapping, dtype=np.float64) / 8 - 0.01)

    def q_sample(self, x_start, t, noise=None):
        alpha = torch.as_tensor(self.alphas_cumprod, device=x_start.device, dtype=x_start.dtype)[t]
        alpha = alpha[:, None, None, None]
        return alpha.sqrt() * x_start + (1 - alpha).sqrt() * noise


@pytest.fixture(autouse=True)
def small_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def setup_model(tmp_path):
    torch.manual_seed(13)
    model = TinyDenoiser()
    path = tmp_path / "fp.pt"
    torch.save(model.state_dict(), path)
    generator = torch.Generator().manual_seed(7)
    samples = [CalibrationSample(f"cal-{i}", torch.randn(1, 4, 8, 8, generator=generator),
                                 {"split": "calibration"}) for i in range(2)]
    stages = [CalibrationStage("coarse", TinySpacedDiffusion(), (0, 2, 4)),
              CalibrationStage("refined", TinySpacedDiffusion((0, 1, 2, 3)), (0, 1))]
    return model, path, samples, stages


def produce(setup_model, out, **kwargs):
    model, path, samples, stages = setup_model
    return calibrate_conv2d_model(
        model, path, samples, stages, out, provenance={"test_scope": "tiny_cpu_model"},
        config=GPTQConfig(group_size=4, block_size=7, act_order=True),
        max_patches_per_batch=8, patch_chunk_size=3, **kwargs,
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("fp16", [False, True])
def test_calibrate_quantize_save_reload_and_output(setup_model, tmp_path, wrapped, fp16):
    model, fp_path, samples, stages = setup_model
    if wrapped:
        torch.save({"state_dict": model.state_dict(), "step": 100}, fp_path)
    source_bytes = fp_path.read_bytes()
    source_state = copy.deepcopy(model.state_dict())
    paths = produce(setup_model, tmp_path / "artifact", use_fp16=fp16)
    manifest = json.loads(paths["manifest"].read_text())
    validate_quant_manifest(manifest, sha256_file(fp_path), sha256_file(paths["checkpoint"]))
    assert fp_path.read_bytes() == source_bytes
    assert model.training  # Source object, including precision/training state, stays intact.
    assert all(torch.equal(v, source_state[k]) for k, v in model.state_dict().items())
    assert all(p.requires_grad for p in model.parameters())
    assert manifest["format"] == "dequantized_state_dict"
    assert manifest["packed_int4_execution"] is False
    assert manifest["verification"]["status"] == "passed"
    assert manifest["verification"]["reconstruction"] == "not_run"
    assert manifest["verification"]["runtime_changed_parameters"]
    assert manifest["verification"]["probe_output_max_abs_difference"] > 0
    assert manifest["verification"]["probe_outputs_differ"] is True
    assert set(manifest["quantized_parameters"]) == {"in_conv.weight", "out_conv.weight"}
    reasons = {row["layer"]: row["reason"] for row in manifest["calibration"]["excluded_layers"]}
    assert reasons["depthwise"] == "grouped_or_depthwise_convolution"
    assert reasons["time"] == "only_plain_nn_Conv2d_supported"
    assert reasons["unused"] == "not_executed_during_calibration"
    assert reasons["string_padding"] == "unsupported_padding"
    for summary in manifest["calibration"]["layer_collection"].values():
        assert summary["calibration_batches"] == 10  # 2 samples * (3 coarse + 2 refined).
        assert summary["selected_patches"] == 80
    inputs = manifest["calibration"]["layer_collection"]["in_conv.weight"]["input_dtypes_by_batch"]
    assert inputs == {"torch.float16" if fp16 else "torch.float32": 10}
    assert manifest["calibration"]["layer_collection"]["out_conv.weight"]["input_dtypes_by_batch"] == {"torch.float32": 10}
    coarse = manifest["calibration"]["stages"][0]
    assert coarse["local_timesteps"] == [0, 2, 4]
    assert coarse["original_timesteps"] == [0, 8, 19]
    assert coarse["model_timesteps"] == [0.0, 400.0, 950.0]
    saved = torch.load(paths["checkpoint"], weights_only=True)
    grids = torch.load(paths["quantization"], weights_only=True)
    assert sha256_file(paths["quantization"]) == manifest["quantization_parameters_sha256"]
    for name, value in saved.items():
        if name in grids:
            grid = grids[name]
            codes = grid["codes"].flatten(1)
            group = grid["group_index"]
            decoded = (codes - grid["zero_points"][:, group]) * grid["scales"][:, group]
            assert torch.equal(decoded.reshape_as(value), value)
            assert codes.min() >= -8 and codes.max() <= 7
            assert value.dtype == source_state[name].dtype  # Original FP storage, including A16 runs.
        else:
            assert torch.equal(value, source_state[name])
    reloaded = TinyDenoiser()
    reloaded.load_state_dict(saved, strict=True)
    x, t = samples[0].x_start, torch.tensor([400.0])
    assert not torch.equal(model(x, t), reloaded(x, t))


def fake_factory(monkeypatch):
    module = types.ModuleType("tiny_calibration_factory")
    module.__file__ = __file__
    module.model_and_diffusion_defaults = lambda: {
        "image_size": 8, "timestep_respacing": "", "class_cond": False,
    }
    module.create_model_and_two_diffusion = lambda **kwargs: (
        TinyDenoiser(), TinySpacedDiffusion(), TinySpacedDiffusion((0, 1, 2, 3)))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_generated_artifact_loads_through_existing_runner(setup_model, tmp_path, monkeypatch):
    model, fp_path, _, _ = setup_model
    module = fake_factory(monkeypatch)
    paths = produce(setup_model, tmp_path / "artifact")
    manifest = json.loads(paths["manifest"].read_text())
    validate_quant_manifest(manifest, sha256_file(fp_path), sha256_file(paths["checkpoint"]))
    args = create_argparser().parse_args([
        "--model_path", str(fp_path), "--w4_model_path", str(paths["checkpoint"]),
        "--w4_manifest", str(paths["manifest"]), "--data_root", str(tmp_path),
        "--mask_path", "not_used_in_model_loading", "--save_path", str(tmp_path / "results"),
        "--image_size", "8", "--device", "cpu", "--factory_module", module.__name__,
    ])
    image = np.ones((1, 2, 8, 8), dtype=np.complex64)
    case = Case(image, image, image.real.copy(), (1.0, 1.0))
    backend = TorchBackend(args, case, manifest)
    fp, _, _, _ = backend._build_model("fp")
    w4, _, _, info = backend._build_model("w4")
    assert fp is not w4
    assert torch.equal(fp.in_conv.weight, model.in_conv.weight)
    assert not torch.equal(fp.in_conv.weight, w4.in_conv.weight)
    assert info["changed_quantized_parameters"]
    corrupt = torch.load(paths["checkpoint"], weights_only=True)
    corrupt["in_conv.bias"] += 0.1
    corrupt_path = tmp_path / "corrupt.pt"
    torch.save(corrupt, corrupt_path)
    backend.args.w4_model_path = str(corrupt_path)
    with pytest.raises(ValueError, match="Undeclared parameter"):
        backend._build_model("w4")


def test_cli_from_legacy_pairs_to_runner_manifest(setup_model, tmp_path, monkeypatch):
    _, fp_path, _, _ = setup_model
    module = fake_factory(monkeypatch)
    for i in range(4):
        image = np.random.default_rng(i).normal(size=(8, 8)) + 1j * np.random.default_rng(i + 9).normal(size=(8, 8))
        with (tmp_path / f"slice{i}.pt").open("wb") as stream:
            pickle.dump({"img": image}, stream)
    rows = [{"sample_id": "ignored-test", "split": "test", "file1": "missing.pt", "file2": "missing2.pt"},
            {"sample_id": "train-0", "split": "calibration", "file1": "slice0.pt", "file2": "slice1.pt"},
            {"sample_id": "train-1", "split": "calibration", "file1": "slice2.pt", "file2": "slice3.pt"}]
    data_manifest = tmp_path / "calibration.jsonl"
    data_manifest.write_text("\n".join(json.dumps(row) for row in rows))
    out = tmp_path / "cli_artifact"
    assert calibrate_main([
        "--model_path", str(fp_path), "--calibration_manifest", str(data_manifest),
        "--output_dir", str(out), "--factory_module", module.__name__, "--image_size", "8",
        "--refine_steps", "2", "--num_calibration_timesteps", "2", "--max_samples", "1",
        "--max_patches_per_batch", "8", "--group_size", "4", "--exclude", "out_conv",
    ]) == 0
    result = json.loads((out / "manifest.json").read_text())
    validate_quant_manifest(result, sha256_file(fp_path), sha256_file(out / "weights_w4.pt"))
    data = result["calibration"]
    assert data["selected_pairs"] == 1 and data["available_pairs"] == 2
    assert data["data_samples"][0]["sample_id"] == "train-0"
    assert data["data_samples"][0]["manifest_line"] == 2
    assert data["stages"][0]["local_timesteps"] == [0, 4]
    assert data["stages"][1]["local_timesteps"] == [0, 1]
    assert data["factory_module"] == module.__name__
    assert result["quantized_parameters"] == ["in_conv.weight"]
    with pytest.raises(ValueError, match="calibration/train"):
        load_calibration_samples(data_manifest, None, split="test", max_samples=1, image_size=8)


def test_same_states_are_replayed_for_each_layer(setup_model, tmp_path):
    first = produce(setup_model, tmp_path / "all", seed=19)
    second = produce(setup_model, tmp_path / "subset", seed=19, include=("in_conv",))
    a, b = [torch.load(paths["checkpoint"], weights_only=True) for paths in (first, second)]
    assert torch.equal(a["in_conv.weight"], b["in_conv.weight"])
    assert torch.equal(b["out_conv.weight"], setup_model[0].out_conv.weight)


def test_memory_and_explicit_exclusion_are_recorded(setup_model, tmp_path):
    paths = produce(setup_model, tmp_path / "limited", max_hessian_bytes=128, exclude=("unused",))
    manifest = json.loads(paths["manifest"].read_text())
    reasons = {v["layer"]: v["reason"] for v in manifest["calibration"]["excluded_layers"]}
    assert reasons["in_conv"] == "hessian_exceeds_memory_limit"
    assert reasons["unused"] == "excluded_by_pattern"
    assert manifest["quantized_parameters"] == ["out_conv.weight"]


def test_output_and_source_protection(setup_model, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "result.txt"
    sentinel.write_text("keep")
    before = setup_model[1].read_bytes()
    with pytest.raises(FileExistsError):
        produce(setup_model, out)
    assert sentinel.read_text() == "keep" and setup_model[1].read_bytes() == before
    with torch.no_grad():
        setup_model[0].in_conv.weight.add_(0.1)
    with pytest.raises(ValueError, match="exact FP checkpoint"):
        produce(setup_model, tmp_path / "mismatch")
    assert not (tmp_path / "mismatch").exists()


@pytest.mark.parametrize("timesteps", [(-1,), (5,), (1, 1), (), (True,)])
def test_invalid_timestep_policy_is_rejected(timesteps):
    with pytest.raises(ValueError, match="Timesteps"):
        select_timesteps(5, 2, timesteps)


def test_uniform_selection_and_model_timestep_mapping():
    assert select_timesteps(10, 4) == (0, 3, 6, 9)
    assert select_timesteps(2, 16) == (0, 1)
    assert select_timesteps(5, 1) == (2,)
    diffusion = TinySpacedDiffusion()
    assert model_timesteps(diffusion, torch.tensor([0, 2, 4])).tolist() == [0.0, 400.0, 950.0]
    diffusion.rescale_timesteps = False
    assert model_timesteps(diffusion, torch.tensor([0, 2, 4])).tolist() == [0, 8, 19]


def test_unknown_layer_pattern_and_fp_vs_fp_rejected(setup_model, tmp_path):
    with pytest.raises(ValueError, match="matched no module"):
        produce(setup_model, tmp_path / "bad_pattern", include=("typo.*",))
    with torch.no_grad():
        setup_model[0].in_conv.weight.zero_()
    torch.save(setup_model[0].state_dict(), setup_model[1])
    with pytest.raises(ValueError, match="No weight changed"):
        produce(setup_model, tmp_path / "no_changes", include=("in_conv",))
    assert not (tmp_path / "no_changes").exists()


def test_failed_reload_is_not_reported_as_success(setup_model, tmp_path, monkeypatch):
    def fail_reload(*args, **kwargs):
        raise RuntimeError("injected reload failure")
    monkeypatch.setattr(TinyDenoiser, "load_state_dict", fail_reload)
    out = tmp_path / "failed_artifact"
    with pytest.raises(RuntimeError, match="injected reload failure"):
        produce(setup_model, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["verification"]["status"] == "failed"
    assert manifest["verification"]["reconstruction"] == "not_run"
    with pytest.raises(ValueError, match="verification has not passed"):
        validate_quant_manifest(manifest, sha256_file(setup_model[1]), sha256_file(out / "weights_w4.pt"))
    # An interrupted write cannot fall back to the legacy optional-block path.
    del manifest["verification"]
    with pytest.raises(ValueError, match="verification has not passed"):
        validate_quant_manifest(manifest, sha256_file(setup_model[1]), sha256_file(out / "weights_w4.pt"))
