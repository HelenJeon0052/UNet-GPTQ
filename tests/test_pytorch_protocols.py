"""Small CPU integration tests; no pretrained model or clinical data used."""
from __future__ import annotations

import unittest
from unittest import mock
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import numpy as np

try:
    import torch
    import h5py
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is not installed in this environment") from exc

from unet_gptq.diffusion.samplers import reconstruct_adjacent_aggregate, reconstruct_pair
from unet_gptq.metrics.metrics import image_metrics
from unet_gptq.train.run_exp import Case, quality_rows


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.05))

    def forward(self, x, t):
        return x * self.scale


class TinyDiffusion:
    num_timesteps = 3
    alphas_cumprod = np.array([0.9, 0.7, 0.4])

    def p_sample(self, model, x, t, no_noise=False, **kwargs):
        output = model(x, t)
        noise = torch.randn_like(x)
        return {"model_output": output, "sample": x * 0.8 + (0 if no_noise else noise * 0.03)}


class PyTorchProtocolsTests(unittest.TestCase):
    def test_actual_sampler_keeps_central_trajectory_and_counts_nfe(self):
        model, schedule = TinyDenoiser().eval(), TinyDiffusion()
        observed = torch.complex(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4))
        mask = torch.tensor([[1, 0, 1, 0]] * 4, dtype=torch.bool)
        count = [0]
        handle = model.register_forward_hook(lambda *_: count.__setitem__(0, count[0] + 1))
        try:
            torch.manual_seed(12)
            single = reconstruct_pair(model, schedule, schedule, observed[:, 1:3], mask,
                                      num_samples=2, refine_steps=2, dc_weight=1.0)
            self.assertEqual(count[0], 2 * 3 + 2)
            count[0] = 0
            torch.manual_seed(12)
            adjacent = reconstruct_adjacent_aggregate(model, schedule, schedule, observed, mask,
                                                       num_samples=2, refine_steps=2, dc_weight=1.0)
            self.assertEqual(count[0], 3 * 2 * 3 + 2)
            self.assertTrue(torch.equal(single[0], adjacent[2]))
            self.assertEqual(tuple(adjacent[1].shape), (1, 2, 4, 4))
        finally:
            handle.remove()

    def test_pilot_metrics_match_runner_per_slice_contract(self):
        target = torch.ones((1, 2, 4, 4), dtype=torch.complex64)
        estimate = target * 0.5j
        mask = torch.ones((4, 4), dtype=torch.bool)
        observed = torch.fft.fft2(target)
        result = image_metrics(estimate, target, observed, mask)
        self.assertAlmostEqual(result["magnitude_nmse"], 0.25)
        self.assertAlmostEqual(result["complex_nmse"], 1.25)
        self.assertEqual(result["nmse"], result["magnitude_nmse"])
        case = Case(target.numpy(), observed.numpy(), np.broadcast_to(mask.numpy(), target.shape), (1.0, 1.0))
        outputs = {"coarse": estimate.numpy(), "refined": estimate.numpy()}
        row = quality_rows(outputs, case, 0, "fp")[0]
        self.assertAlmostEqual(row["magnitude_nmse"], result["magnitude_nmse"])
        self.assertAlmostEqual(row["complex_nmse"], result["complex_nmse"])
        self.assertAlmostEqual(row["psnr_db"], result["psnr_db"])

    def test_shared_config_connects_fp_pilot_and_both_protocols(self):
        """Real PyTorch/HDF5 with a toy model; no clinical or pretrained claim."""
        from unet_gptq.train import inspect_utils, run_exp, compare_fp_protocols
        from unet_gptq.train.simple_matrics import compare_inputs, load_saved

        factory = types.ModuleType("improved_diffusion.script_util_duo")
        factory.__file__ = __file__
        factory.model_and_diffusion_defaults = lambda: {
            "image_size": 8, "timestep_respacing": "3", "use_fp16": False,
        }
        factory.create_model_and_two_diffusion = lambda **kwargs: (TinyDenoiser(), TinyDiffusion(), TinyDiffusion())
        parent = types.ModuleType("improved_diffusion")
        parent.__path__ = []
        parent.script_util_duo = factory
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {
            "improved_diffusion": parent, "improved_diffusion.script_util_duo": factory,
        }):
            root = Path(directory)
            checkpoint = root / "source.pt"
            torch.save(TinyDenoiser().state_dict(), checkpoint)
            h5 = root / "synthetic.h5"
            generator = np.random.default_rng(7)
            kspace = (generator.normal(size=(4, 8, 8)) + 1j * generator.normal(size=(4, 8, 8))).astype(np.complex64)
            with h5py.File(h5, "w") as volume:
                volume.create_dataset("kspace", data=kspace)
                volume.attrs["acquisition"] = "SYNTHETIC"
            mask = root / "mask.pkl"
            with mask.open("wb") as stream:
                pickle.dump(np.ones((8, 8), dtype=np.float32), stream)
            source_run = inspect_utils.run_pilot(
                checkpoint, h5, out=root / "exports", slice_index=1, crop_size=8, acs_width=4,
                coarse_steps=3, refine_steps=2, dc_weight=1.0, device="cpu", precision="fp32",
                model_options="{}", mask_path=mask, progress=False, export_pickles=True, export_neighbors=True,
            )
            config = {
                "checkpoint": str(checkpoint), "image_size": 8, "model_options": {},
                "mask": {"path": str(mask), "layout": "centered"},
                "sampling": {"coarse_steps": 3, "refine_steps": 2, "num_samples": 1,
                             "seed": 0, "model_init_seed": 0, "dc_weight": 1.0, "clip_denoised": True},
                "execution": {"device": "cpu", "precision": "fp32"},
                "data": {"h5": str(h5), "data_root": str(source_run), "slice_index": 1,
                         "file1": "slice1.pkl", "file2": "slice2.pkl", "before": "slice0.pkl", "after": "slice3.pkl"},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            pilots = root / "pilots"
            self.assertEqual(inspect_utils.main(["--config", str(config_path), "--out", str(pilots), "--no-progress"]), 0)
            pilot = next(pilots.iterdir())
            run = root / "paired"
            self.assertEqual(run_exp.main(["--config", str(config_path), "--conditions", "fp", "--save_path", str(run)]), 0)
            pilot_arrays, _ = load_saved(pilot)
            paired_arrays, _ = load_saved(run / "seed_0/fp/outputs.npz")
            self.assertEqual(compare_inputs(pilot_arrays, paired_arrays, 3e-5)["relation"], "same_inputs")
            np.testing.assert_allclose(pilot_arrays["reconstruction"], paired_arrays["reconstruction"], rtol=3e-5, atol=3e-5)
            info = json.loads((pilot / "manifest.json").read_text())
            self.assertEqual(info["nfe"]["actual_model_forwards"], 5)
            comparison = root / "protocols"
            self.assertEqual(compare_fp_protocols.main(["--config", str(config_path), "--save_path", str(comparison)]), 0)
            comparison_info = json.loads((comparison / "manifest.json").read_text())
            self.assertTrue(comparison_info["central_coarse_matches"])
            self.assertEqual(comparison_info["protocols"]["single_pair"]["nfe_actual"], 5)
            self.assertEqual(comparison_info["protocols"]["adjacent_aggregate"]["nfe_actual"], 11)


if __name__ == "__main__":
    unittest.main()
