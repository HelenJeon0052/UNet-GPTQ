"""Small CPU integration tests; no pretrained model or clinical data used."""
from __future__ import annotations

import unittest
import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("PyTorch is not installed in this environment") from exc

from unet_gptq.diffusion.samplers import reconstruct_adjacent_aggregate, reconstruct_pair
from unet_gptq.metrics.metrics import image_metrics
from unet_gptq.train.run_exp import Case, quality_rows


class TinyDenoiser(torch.nn.Module):
    def forward(self, x, t):
        return x * 0.05


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


if __name__ == "__main__":
    unittest.main()
