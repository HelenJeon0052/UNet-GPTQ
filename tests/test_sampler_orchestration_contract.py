"""Exercise actual sampler control flow with lightweight NumPy stand-ins.

This does not replace PyTorch or real-checkpoint integration tests.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from unet_gptq.diffusion.aggregation import overlapping_pair_components


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


class SamplerOrchestrationTests(unittest.TestCase):
    def test_central_coarse_is_paired_and_neighbor_pass_count_is_correct(self):
        source = (Path(__file__).parents[1] / "src/unet_gptq/diffusion/samplers.py").read_text()
        tree = ast.parse(source)
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name in ("reconstruct_pair", "reconstruct_adjacent_aggregate")]
        self.assertEqual(len(selected), 2)
        for node in selected:
            node.decorator_list = []
        calls = []

        def coarse(_model, _diffusion, observed, _mask, **kwargs):
            calls.append(("coarse", tuple(observed[0, :, 0, 0].real)))
            noise = np.random.random()
            return np.stack((observed[:, 0].real, observed[:, 0].imag,
                             observed[:, 1].real, observed[:, 1].imag), axis=1) + noise

        def refine(_model, _diffusion, start, observed, _mask, steps, **kwargs):
            calls.append(("refine", tuple(observed[0, :, 0, 0].real), steps))
            return start + 1

        fake_torch = SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor),
                                     is_complex=np.iscomplexobj, complex64=np.complex64,
                                     cat=lambda parts, dim: np.concatenate(parts, axis=dim))
        env = {"torch": fake_torch, "_coarse_pass": coarse, "_refine_pass": refine,
               "prepare_mask": lambda mask, _: mask, "validate_dc_weight": lambda value: value,
               "channels_to_img": lambda x: x[:, 0::2] + 1j * x[:, 1::2],
               "overlapping_pair_components": overlapping_pair_components}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "sampler_contract", "exec"), env)
        observed = np.array([10, 20, 30, 40], dtype=np.complex64).reshape(1, 4, 1, 1).view(Tensor)
        schedule = SimpleNamespace(num_timesteps=4)
        np.random.seed(5)
        single = env["reconstruct_pair"](None, schedule, schedule, observed[:, 1:3], None,
                                          num_samples=2, refine_steps=2, dc_weight=1)
        self.assertEqual(len([c for c in calls if c[0] == "coarse"]), 2)
        calls.clear()
        np.random.seed(5)
        adjacent = env["reconstruct_adjacent_aggregate"](None, schedule, schedule, observed, None,
                                                           num_samples=2, refine_steps=2, dc_weight=1)
        np.testing.assert_array_equal(single[0], adjacent[2])
        self.assertEqual([c[1] for c in calls if c[0] == "coarse"],
                         [(20, 30), (20, 30), (10, 20), (10, 20), (30, 40), (30, 40)])
        self.assertEqual(calls[-1], ("refine", (20, 30), 2))


if __name__ == "__main__":
    unittest.main()
