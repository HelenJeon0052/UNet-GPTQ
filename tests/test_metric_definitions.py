# code created by codex
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from unet_gptq.metrics.quality import slice_quality, summarize_slice_quality
from unet_gptq.train.simple_matrics import analyze_run, compute_metrics, stage_summary


def make_arrays():
    reference = np.array([[[[1, 1], [1, 1]], [[4, 4], [4, 4]]]], dtype=np.complex64)
    estimate = np.array([[[[0.5, 0.5], [0.5, 0.5]], [[3, 3], [3, 3]]]], dtype=np.complex64)
    mask = np.ones(reference.shape, dtype=np.float32)
    observed = np.fft.fft2(reference, axes=(-2, -1)).astype(np.complex64)
    return {"reference": reference, "observed": observed, "mask": mask,
            "zero_filled": reference.copy(), "coarse": estimate.copy(),
            "reconstruction": estimate.copy()}


class MetricDefinitionTests(unittest.TestCase):
    def test_phase_only_difference(self):
        ref = np.ones((2, 2), dtype=np.complex64)
        result = slice_quality(-ref, ref)
        self.assertEqual(result["magnitude_nmse"], 0.0)
        self.assertEqual(result["complex_nmse"], 4.0)
        self.assertTrue(math.isinf(result["magnitude_psnr_db"]))
        self.assertEqual(result["magnitude_psnr_status"], "positive_infinity")

    def test_perfect_and_zero_reference_are_distinct(self):
        ref = np.ones((2, 2), dtype=np.complex64)
        perfect = slice_quality(ref, ref)
        self.assertEqual(perfect["magnitude_nmse"], 0.0)
        self.assertEqual(perfect["complex_nmse"], 0.0)
        self.assertEqual(perfect["magnitude_psnr_db"], float("inf"))
        for estimate in (np.zeros_like(ref), ref):
            result = slice_quality(estimate, np.zeros_like(ref))
            for name in ("magnitude_nmse", "complex_nmse", "magnitude_psnr_db"):
                self.assertTrue(math.isnan(result[name]))
            self.assertEqual(result["nmse_status"], "undefined_zero_reference")
            self.assertEqual(result["magnitude_psnr_status"], "undefined_zero_reference")

    def test_per_slice_peak_and_mean_are_not_pooled_energy(self):
        arrays = make_arrays()
        rows = compute_metrics(arrays)
        coarse = [row for row in rows if row["stage"] == "coarse"]
        self.assertEqual([row["reference_peak"] for row in coarse], [1.0, 4.0])
        self.assertAlmostEqual(coarse[0]["magnitude_nmse"], 0.25)
        self.assertAlmostEqual(coarse[1]["magnitude_nmse"], 0.0625)
        self.assertAlmostEqual(coarse[0]["magnitude_psnr_db"], 6.020599913279624)
        self.assertAlmostEqual(coarse[1]["magnitude_psnr_db"], 12.041199826559248)
        summary = next(row for row in stage_summary(rows) if row["stage"] == "coarse")
        self.assertAlmostEqual(summary["magnitude_nmse_mean"], 0.15625)
        self.assertAlmostEqual(summary["magnitude_psnr_db_mean"], 9.030899869919436)
        self.assertNotAlmostEqual(summary["magnitude_nmse_mean"], 5 / 68)

    def test_undefined_values_are_not_silently_skipped(self):
        ref = np.ones((2, 2))
        finite = slice_quality(ref / 2, ref)
        perfect = slice_quality(ref, ref)
        undefined = slice_quality(ref, ref * 0)
        self.assertTrue(math.isinf(summarize_slice_quality([finite, perfect])["magnitude_psnr_db_mean"]))
        result = summarize_slice_quality([finite, undefined])
        self.assertTrue(math.isnan(result["magnitude_nmse_mean"]))
        self.assertEqual(result["magnitude_nmse_nonfinite_count"], 1)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            slice_quality(np.ones((2, 2)), np.ones((3, 2)))
        with self.assertRaises(ValueError):
            slice_quality(np.full((2, 2), np.nan), np.ones((2, 2)))

    def test_reanalysis_leaves_old_metric_files_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            np.savez(path / "reconstruction.npz", **make_arrays())
            for name in ("metrics.csv", "stage_summary.csv", "metrics_summary.json"):
                (path / name).write_text("old-result", encoding="utf-8")
            result = analyze_run(path)
            self.assertEqual(result["metric_schema_version"], 2)
            for name in ("metrics.csv", "stage_summary.csv", "metrics_summary.json"):
                self.assertEqual((path / name).read_text(encoding="utf-8"), "old-result")
            for name in ("metrics_v2.csv", "stage_summary_v2.csv", "metrics_summary_v2.json"):
                self.assertTrue((path / name).is_file())

    def test_adapters_share_metrics_and_report_undefined_deltas(self):
        # Exercise the adapters' NumPy logic even on a machine without torch.
        # This does not replace the real-PyTorch integration test below.
        class FakeTensor:
            def __init__(self, array):
                self.array = np.asarray(array)
                self.shape, self.ndim = self.array.shape, self.array.ndim

            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.array

        def module(name, **attributes):
            value = types.ModuleType(name)
            value.__dict__.update(attributes)
            return value

        stubs = {
            "torch": module("torch"),
            "unet_gptq.domain.data_consistency": module("unet_gptq.domain.data_consistency",
                validate_dc_weight=lambda x: x, observed_residual=lambda *args: types.SimpleNamespace(item=lambda: 0.0)),
            "unet_gptq.domain.forward_operator": module("unet_gptq.domain.forward_operator",
                channels_to_img=lambda x: x, img_to_channels=lambda x: x),
            "unet_gptq.diffusion.diagnostics": module("unet_gptq.diffusion.diagnostics", SamplingDiagnostics=object),
            "unet_gptq.diffusion.samplers": module("unet_gptq.diffusion.samplers", reconstruct_pair=lambda *a, **k: None),
        }
        root = Path(__file__).resolve().parents[1] / "src/unet_gptq"
        with patch.dict(sys.modules, stubs):
            def load(name, path):
                spec = importlib.util.spec_from_file_location(name, path)
                loaded = importlib.util.module_from_spec(spec)
                sys.modules[name] = loaded  # dataclass resolves its defining module
                self.addCleanup(sys.modules.pop, name, None)
                spec.loader.exec_module(loaded)
                return loaded

            runner = load("metric_test_runner", root / "train/run_exp.py")
            image = load("metric_test_image", root / "metrics/metrics.py")
            arrays = make_arrays()
            case = runner.Case(arrays["reference"], arrays["observed"], arrays["mask"], (1.0, 1.0))
            outputs = {"coarse": arrays["coarse"], "refined": arrays["reconstruction"]}
            runner_rows = runner.quality_rows(outputs, case, 0, "fp")
            pilot_rows = [row for row in compute_metrics(arrays) if row["stage"] == "coarse"]
            image_result = image.image_metrics(*(FakeTensor(arrays[key]) for key in
                ("coarse", "reference", "observed", "mask")))
            for name in ("magnitude_nmse", "complex_nmse", "magnitude_psnr_db"):
                for index in range(2):
                    self.assertAlmostEqual(pilot_rows[index][name], runner_rows[index][name])
                    self.assertAlmostEqual(pilot_rows[index][name], image_result["per_slice"][index][name])
                self.assertAlmostEqual(image_result[name], np.mean([row[name] for row in pilot_rows]))
            bad_ref = arrays["reference"].copy()
            bad_ref[0, 1] = 0
            bad_observed = np.fft.fft2(bad_ref, axes=(-2, -1)).astype(np.complex64)
            bad_case = runner.Case(bad_ref, bad_observed, arrays["mask"], (1.0, 1.0))
            fp = runner.quality_rows(outputs, bad_case, 0, "fp")
            w4 = runner.quality_rows(outputs, bad_case, 0, "w4")
            summary = runner.summarize_pairs(runner.paired_rows(fp, w4, eligible=True))
            metric = summary["stages"]["coarse"]["delta_magnitude_nmse"]
            self.assertIsNone(metric["mean"])
            self.assertEqual(metric["n_undefined_seeds"], 1)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("h5py"),
                     "PyTorch and h5py are needed for the three-path integration test")
class TorchIntegrationTests(unittest.TestCase):
    def test_three_paths_share_slice_metrics_and_mean(self):
        import torch
        from unet_gptq.metrics.metrics import image_metrics
        from unet_gptq.train.run_exp import Case, quality_rows

        arrays = make_arrays()
        case = Case(arrays["reference"], arrays["observed"], arrays["mask"], (1.0, 1.0))
        runner_rows = quality_rows({"coarse": arrays["coarse"], "refined": arrays["reconstruction"]}, case, 0, "fp")
        runner_rows = [row for row in runner_rows if row["stage"] == "coarse"]
        pilot_rows = [row for row in compute_metrics(arrays) if row["stage"] == "coarse"]
        image_result = image_metrics(torch.from_numpy(arrays["coarse"]), torch.from_numpy(arrays["reference"]),
                                     torch.from_numpy(arrays["observed"]), torch.from_numpy(arrays["mask"]))
        for name in ("magnitude_nmse", "complex_nmse", "magnitude_psnr_db"):
            for index in range(2):
                self.assertAlmostEqual(pilot_rows[index][name], runner_rows[index][name])
                self.assertAlmostEqual(pilot_rows[index][name], image_result["per_slice"][index][name])
            self.assertAlmostEqual(image_result[name], np.mean([row[name] for row in pilot_rows]))
        with self.assertRaises(ValueError):
            image_metrics(torch.from_numpy(arrays["coarse"]), torch.from_numpy(arrays["reference"]),
                          torch.from_numpy(arrays["observed"][:1, :1]), torch.from_numpy(arrays["mask"]))


if __name__ == "__main__":
    unittest.main()
