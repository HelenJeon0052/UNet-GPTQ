"""Compare single-pair and adjacent-overlap FP protocols on the same case.

Requires four trusted, preprocessed adjacent image pickles, a saved mask, an FP
checkpoint and the DiffuseRecon factory. This is a controlled overlap ablation,
not a claim of exact parity with the upstream script's batching or evaluation.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch

from unet_gptq.diffusion.samplers import reconstruct_adjacent_aggregate, reconstruct_pair
from unet_gptq.experiment_config import parse_with_shared_config
from unet_gptq.train.run_exp import (
    TorchBackend, array_hash, create_argparser, load_case, normalize_image,
    quality_rows, sha256_file, validate_args, write_csv, write_json,
)


def load_neighbor(root, name, image_size):
    path = Path(name).expanduser()
    path = (path if path.is_absolute() else Path(root) / path).resolve()
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "img" not in payload:
        raise ValueError(f"{path} must contain a pickle dictionary with img")
    image, scale = normalize_image(payload["img"])
    if image.shape != (image_size, image_size):
        raise ValueError(f"{path}: expected {(image_size, image_size)}, got {image.shape}")
    return image, {"path": str(path), "sha256": sha256_file(path), "scale": scale}


def reconstruct_one(backend, observed, mask, seed, *, neighbor):
    args = backend.args
    model = coarse = refine = None
    handle = None
    try:
        model, coarse, refine, _ = backend._build_model("fp")
        if args.refine_steps > refine.num_timesteps:
            raise ValueError("refine_steps exceeds the refinement schedule")
        observation = torch.from_numpy(observed.copy()).to(backend.device, dtype=torch.complex64)
        mask_tensor = torch.from_numpy(mask.copy()).to(backend.device)
        count = {"nfe": 0}
        def count_forward(_module, _inputs, _output):
            count["nfe"] += 1
        handle = model.register_forward_hook(count_forward)
        backend.seed(seed)
        start_rng = backend.rng_hash()
        if backend.device.type == "cuda":
            torch.cuda.synchronize(backend.device)
        start = time.perf_counter()
        kwargs = dict(num_samples=args.num_samples, refine_steps=args.refine_steps,
                      clip_denoised=args.clip_denoised, dc_weight=args.dc_weight)
        if neighbor:
            coarse_output, refined, central = reconstruct_adjacent_aggregate(
                model, coarse, refine, observation, mask_tensor, **kwargs)
        else:
            coarse_output, refined = reconstruct_pair(model, coarse, refine, observation, mask_tensor, **kwargs)
            central = coarse_output
        if backend.device.type == "cuda":
            torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - start
        if count["nfe"] < 1:
            raise RuntimeError("No denoiser forward calls were observed")
        result = {"coarse": backend.as_complex_numpy(coarse_output),
                  "refined": backend.as_complex_numpy(refined),
                  "central_coarse": backend.as_complex_numpy(central)}
        return result, {"nfe_actual": count["nfe"],
                        "nfe_expected_if_one_forward_per_step": coarse.num_timesteps * args.num_samples * (3 if neighbor else 1) + args.refine_steps,
                        "coarse_steps_per_pair": coarse.num_timesteps, "refine_steps": args.refine_steps,
                        "samples_per_pair": args.num_samples, "seconds": elapsed,
                        "timing_scope": "sampler only; includes forward-count hook; no warmup", "rng_start_sha256": start_rng}
    finally:
        if handle is not None:
            handle.remove()
        model = coarse = refine = None
        gc.collect()
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    parser = create_argparser()
    parser.description = __doc__
    parser.add_argument("--before", help="slice i-1 pickle relative to data_root")
    parser.add_argument("--after", help="slice i+2 pickle relative to data_root")
    parser.set_defaults(conditions=["fp"], seeds=[0])
    args, shared = parse_with_shared_config(parser, argv, "comparison")
    if shared:
        args.before = args.before or shared["data"].get("before")
        args.after = args.after or shared["data"].get("after")
    validate_args(args)
    if args.conditions != ["fp"] or len(args.seeds) != 1 or not args.before or not args.after:
        raise ValueError("Compare FP only, with exactly one seed and both neighbor slice paths")
    out = Path(args.save_path).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    case, input_info = load_case(args)
    previous, previous_info = load_neighbor(args.data_root, args.before, args.image_size)
    following, following_info = load_neighbor(args.data_root, args.after, args.image_size)
    reference4 = np.stack([previous, *case.reference[0], following])[None]
    mask4 = np.broadcast_to(case.mask[0, 0], reference4.shape).copy()
    observed4 = (np.fft.fft2(reference4, axes=(-2, -1), norm="backward") * mask4).astype(np.complex64)
    observed4[:, 1:3] = case.observed  # Exactly the same central measurement.
    backend = TorchBackend(args, case, None)
    out.mkdir(parents=True, exist_ok=False)
    try:
        single, single_info = reconstruct_one(backend, case.observed, case.mask, args.seeds[0], neighbor=False)
        adjacent, adjacent_info = reconstruct_one(backend, observed4, case.mask[0, 0], args.seeds[0], neighbor=True)
        equal = bool(np.allclose(single["coarse"], adjacent["central_coarse"],
                                 atol=args.repeat_atol, rtol=args.repeat_rtol))
        rng_equal = single_info["rng_start_sha256"] == adjacent_info["rng_start_sha256"]
        rows = (quality_rows(single, case, args.seeds[0], "single_pair")
                + quality_rows(adjacent, case, args.seeds[0], "adjacent_aggregate"))
        np.savez_compressed(out / "inputs.npz", reference=case.reference, observed=case.observed,
                            mask=case.mask, four_reference=reference4, four_observed=observed4)
        np.savez_compressed(out / "outputs.npz", single_coarse=single["coarse"], single_refined=single["refined"],
                            adjacent_coarse=adjacent["coarse"], adjacent_refined=adjacent["refined"],
                            adjacent_central_coarse=adjacent["central_coarse"])
        write_csv(out / "metrics.csv", rows)
        result = {
            "status": "completed" if equal and rng_equal else "pairing_check_failed",
            "protocols": {"single_pair": single_info, "adjacent_aggregate": adjacent_info},
            "central_coarse_matches": equal, "same_initial_rng_state": rng_equal,
            "sampling_noise_note": "central pair runs first in both protocols; per-step noise is not independently recorded",
            "input": input_info, "neighbors": {"before": previous_info, "after": following_info},
            "neighbor_order": "i-1,i,i+1,i+2; adjacency is declared by the input manifest, not independently verified",
            "four_observed_sha256": array_hash(observed4),
            "measurement_model": "synthetic_singlecoil_from_combined_reference",
            "channel_order": ["Re(slice1)", "Im(slice1)", "Re(slice2)", "Im(slice2)"],
            "shared_config": shared["config_provenance"] if shared else None,
            "arguments": vars(args), "fp_checkpoint_sha256": sha256_file(args.model_path),
            "environment": backend.metadata(),
            "interpretation": "FP aggregation ablation; adjacent sampling uses more coarse evaluations and sequential batches",
        }
        write_json(out / "manifest.json", result)
        print(f"{result['status']}: {out}")
        return 0 if result["status"] == "completed" else 2
    except Exception as exc:
        write_json(out / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                          "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
