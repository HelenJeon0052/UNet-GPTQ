from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest

from unet_gptq.experiment_config import parse_with_shared_config, read_shared_config, shared_defaults
from unet_gptq.split_integrity import check_calibration_separation


class ExperimentContractTests(unittest.TestCase):
    def make_config(self, directory):
        value = {
            "checkpoint": "weights.pt", "image_size": 320, "model_options": {"num_channels": 128},
            "mask": {"path": "mask.pkl", "layout": "centered"},
            "sampling": {"coarse_steps": 100, "refine_steps": 20, "num_samples": 2,
                         "seed": 7, "model_init_seed": 0, "dc_weight": 1.0, "clip_denoised": True},
            "execution": {"device": "cpu", "precision": "fp32"},
            "data": {"h5": "brain.h5", "data_root": "data", "file1": "a.pkl", "file2": "b.pkl", "slice_index": 3},
        }
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(value))
        return path, value

    def test_shared_controls_map_identically(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_config(directory)
            cfg = read_shared_config(path)
            pilot, pair = shared_defaults(cfg, "pilot"), shared_defaults(cfg, "comparison")
            self.assertEqual(pilot["ckpt"], pair["model_path"])
            self.assertEqual(pilot["mask_path"], pair["mask_path"])
            self.assertEqual(pilot["dc_weight"], pair["dc_weight"])
            self.assertEqual(str(pilot["coarse_steps"]), pair["timestep_respacing"])
            self.assertEqual(pilot["seed"], pair["seeds"][0])
            self.assertEqual(pilot["model_options"], pair["model_options"])
            self.assertEqual(pilot["num_samples"], pair["num_samples"])
            self.assertEqual(pilot["clip_denoised"], pair["clip_denoised"])
            self.assertFalse(pair["use_fp16"])
            self.assertEqual(Path(pilot["mask_path"]).parent, Path(directory))

    def test_cli_override_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path, value = self.make_config(directory)
            parser = argparse.ArgumentParser()
            parser.add_argument("--config")
            parser.add_argument("--dc-weight", type=float)
            args, _ = parse_with_shared_config(parser, ["--config", str(path), "--dc-weight", "0.5"], "pilot")
            self.assertEqual(args.dc_weight, 0.5)
            value["sampling"]["dc_weight"] = 2
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                read_shared_config(path)

    def test_split_check_fails_known_overlap_but_does_not_invent_ids(self):
        calibration = {"data_samples": [{"patient_id": "p1", "exam_id": "e1", "image_files": [{"sha256": "a"}]}]}
        with self.assertRaisesRegex(ValueError, "patient_id"):
            check_calibration_separation(calibration, patient_id="p1")
        with self.assertRaisesRegex(ValueError, "file hash"):
            check_calibration_separation(calibration, evaluation_files=["a"])
        self.assertEqual(check_calibration_separation(calibration, patient_id="p2")["grouping"], "patient_id")
        self.assertEqual(check_calibration_separation(calibration, exam_id="e2")["grouping"], "exam_id")
        self.assertEqual(check_calibration_separation(calibration)["status"], "unverified")
        calibration["data_samples"].append({"exam_id": "e3"})
        self.assertEqual(check_calibration_separation(calibration, patient_id="p2")["status"], "unverified")

    def test_actual_cli_parsers_use_the_same_config(self):
        root = Path(__file__).parents[1] / "src/unet_gptq/train"
        def extract(filename, name, context):
            tree = ast.parse((root / filename).read_text())
            node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), filename, "exec"), context)
            return context[name]
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.make_config(directory)
            create = extract("run_exp.py", "create_argparser", {"argparse": argparse, "__doc__": "test"})
            compare, _ = parse_with_shared_config(create(), ["--config", str(path), "--save_path", str(Path(directory) / "out")], "comparison")
            captured = {}
            def fake_pilot(**kwargs):
                captured.update(kwargs)
            pilot_main = extract("inspect_utils.py", "main", {"argparse": argparse, "Path": Path,
                "parse_with_shared_config": parse_with_shared_config, "run_pilot": fake_pilot, "__doc__": "test"})
            self.assertEqual(pilot_main(["--config", str(path)]), 0)
            self.assertEqual(str(captured["ckpt"]), str(compare.model_path))
            self.assertEqual(str(captured["mask_path"]), str(compare.mask_path))
            self.assertEqual(captured["dc_weight"], compare.dc_weight)
            self.assertEqual(str(captured["coarse_steps"]), compare.timestep_respacing)
            self.assertEqual(captured["model_options"], compare.model_options)
            self.assertEqual(captured["clip_denoised"], compare.clip_denoised)


if __name__ == "__main__":
    unittest.main()
