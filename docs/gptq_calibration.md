# DiffuseRecon Conv2d GPTQ calibration

The calibration entry point connects the existing `Conv2dGPTQ.capture()` and
`quantize()` methods to the real model's supported convolution layers, then
calls `save_gptq_artifact()`. It writes a separate artifact usable by the existing
paired reconstruction runner. The source model, source checkpoint, and existing
experiment directories are preserved. Output directories must be new.

## Data contract

Create a JSONL manifest of preprocessed adjacent slice pairs. Each file is a
trusted legacy pickle containing `{'img': complex_array_of_shape_H_W}`; the
`.pt` suffix does not mean it is a `torch.save` archive. This matches
`run_exp.py --file1/--file2`. Raw fastMRI HDF5 files need preprocessing before
this entry point.

```jsonl
{"sample_id":"cal-volume001-slice012","split":"calibration","file1":"volume001_12.pt","file2":"volume001_12_2.pt","contrast":"FLAIR"}
{"sample_id":"cal-volume002-slice020","split":"calibration","file1":"volume002_20.pt","file2":"volume002_20_2.pt","contrast":"FLAIR"}
```

Paths are relative to `--data_root`, or the manifest directory if omitted.
`--split` accepts `calibration` or `train`; it defaults to `calibration`.
The first `--max_samples` matching rows in manifest order are used (default 8).
IDs must be unique; duplicate file pairs are rejected. Other splits are ignored.
Patient/volume separation from evaluation must be established when creating
the manifests; checking a split label cannot prove that separation.

Each slice is normalized separately by `max(abs(real), abs(imag))`, using the
same `normalize_image()` as the paired runner. Channels are
`[Re(slice1), Im(slice1), Re(slice2), Im(slice2)]` in a float32 `[1,4,H,W]` tensor.
This uses the full-reference scale in synthetic experiments. The manifest saves
sample IDs, selected row numbers, file hashes, normalization scales, and the
calibration JSONL hash. Model/contrast compatibility still needs a verified FP
baseline for the selected checkpoint.

## Diffusion-state policy

The default is **forward-marginal calibration**:

\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

The factory's `q_sample()` constructs each state. This is a disclosed surrogate
for denoiser inputs; it does not replay reverse sampling or data-consistency
updates. No masks or reconstruction outputs are used during calibration.

`--stage both` collects coarse and refinement states. By default, up to 16
uniformly spaced local indices are selected per stage, including the endpoints
when at least two indices are selected. With a count of one, the middle index
is selected. Coarse indices cover the full respaced schedule. Refinement indices
cover only `0..refine_steps-1`, matching the active prefix of the runner's
refinement schedule. `--refine_steps 0` disables that stage.

`--coarse_timesteps 0 25 50 99` and `--refine_timesteps 0 5 10 19` override those
selections with explicit **local**, unique, in-range indices. State weights are
equal per selected sample/timestep call, subject to the same per-call patch cap.
Selecting timesteps or patch budgets based on evaluation results would require
a separate validation protocol.

The U-Net receives the original training index from `timestep_map`, optionally
scaled by `1000/original_num_steps`, exactly once, matching DiffuseRecon's
`SpacedDiffusion._WrappedModel`. The manifest records local indices, original
indices, effective model indices, selected alpha values, and a schedule hash.
It does not substitute a respaced index directly into the U-Net embedding.

Noise is generated as CPU float32 standard normal, with a SHA256-derived seed
for `(seed, sample_id, stage, local_timestep)`. Each layer replays exactly the
same states of an unchanged FP teacher. Previously quantized layers are never
fed back into calibration of later layers.

## Generate an artifact

Run from the UNet-GPTQ root. Make the directory containing upstream
`improved_diffusion/` importable in the same Python environment. Install the
project's dependencies and the upstream factory's dependencies first.

```bash
python -m pip install -e '.[test]'
export PYTHONPATH="$PWD/src:/absolute/path/to/DiffuseRecon${PYTHONPATH:+:$PYTHONPATH}"

# Example architecture/schedule from the upstream sampling recipe.
# Use the settings belonging to YOUR checkpoint; loading is strict.
GPTQ_MODEL_OPTIONS='{"num_channels":128,"num_res_blocks":3,"learn_sigma":false,"dropout":0.3,"diffusion_steps":4000,"noise_schedule":"cosine"}'

python -m unet_gptq.train.calibrate_gptq \
  --model_path /absolute/path/to/ema_0.9999_150000.pt \
  --calibration_manifest manifests/calibration_pairs.jsonl \
  --data_root /data/preprocessed/calibration \
  --split calibration --max_samples 8 \
  --image_size 320 --model_options "$GPTQ_MODEL_OPTIONS" \
  --timestep_respacing 100 --refine_steps 20 \
  --stage both --num_calibration_timesteps 16 \
  --device cuda:0 --use_fp16 \
  --group_size 128 --block_size 128 --damping 0.01 \
  --max_hessian_mib 512 --max_patches_per_batch 4096 \
  --seed 0 --output_dir artifacts/gptq_calibration_001
```

CPU operation is supported with `--device cpu`; omit `--use_fp16` for an FP32
activation baseline. Both CLI entry points use the same factory configuration
resolver. Architecture, respacing, `refine_steps`, and precision flags must match
the intended experiment. Class-conditional models are rejected until a label
calibration policy is supplied.

One Hessian is held at a time. This bounds curvature memory, but repeats the
full FP-teacher forward for every selected convolution. Cost grows with
`selected_layers * selected_pairs * selected_timesteps`. Start with one pair,
two timesteps, and `--include 'input_blocks.0.0'` to check a typical DiffuseRecon
input layer before expanding coverage. Unmatched include/exclude patterns fail
instead of silently selecting nothing. The final study must disclose its actual
layer/sample/timestep coverage.

`max_hessian_mib` limits the accumulator for one layer, **not total process or
GPU memory**. Dense Cholesky factorization needs several additional matrices.
Lower the limit or narrow the layer selection when memory is insufficient;
layers over the limit are excluded and recorded. Actual OOM or factorization
errors propagate, rather than being reported as a successful calibration.

## Supported layers and precision

Selection discovers named modules in the actual model. It supports ordinary
`nn.Conv2d` with `groups=1`, numeric padding, and zero padding mode. Stride and
dilation are handled by the existing input-patch collector.

Grouped/depthwise convolution, convolution subclasses, string/nonzero padding,
shared checkpoint storage, other module types, explicitly excluded modules,
and unexecuted layers remain FP. Linear time embeddings, Conv1d attention
projections, normalization, convolution biases, and buffers are preserved.
Every excluded parameter is listed in `calibration.excluded_parameters`, with
layer-level reasons in `calibration.excluded_layers`. A selected weight already
exactly on its grid can remain unchanged, but at least one selected and one
effective runtime weight must change.

Patch collection uses deterministic uniform spatial positions, capped at
`max_patches_per_batch` per layer invocation. Counts and input dtype histograms
are saved per layer. `--use_fp16` applies the factory model's
`convert_to_fp16()` path, just like the runner; DiffuseRecon keeps some paths,
including its output head, in FP32. Inspect the recorded dtypes when reporting
A16 coverage. Hessian accumulation/factorization uses float32 by default or
float64 with `--hessian_dtype float64`.

Each collector uses an explicit snapshot from the **original checkpoint** as
its weight source, including when collecting FP16 activations. The artifact
retains the source state_dict storage dtypes. It contains dequantized W4 grids
for floating-point convolution and int16 inspection codes in the sidecar.
`packed_int4_execution` is `false`. No packed-storage, throughput, or memory
reduction claim follows from this numerical simulation.

## Artifacts and verification

| File | Contents |
| --- | --- |
| `weights_w4.pt` | Full reloadable tensor state_dict; only selected weights replaced |
| `quantization.pt` | Per-layer codes, scales, zero points, groups, permutations, dead columns |
| `manifest.json` | Source/artifact hashes, configuration, implementation hashes, calibration provenance, exclusions, verification |

After writing, calibration reloads the checkpoint strictly into a fresh model.
It checks that the source file and source model are unchanged, all undeclared
tensors are identical, and FP/W4 weights differ both before and after runtime
precision conversion. It measures a fixed calibration-state output difference
and validates its shape/finiteness. A zero difference on that single probe is
recorded honestly; it is not a reconstruction-quality guarantee.

`verification.status` becomes `passed` only after those checks execute. A failed
reload produces `failed`, propagates the exception, and is rejected by the
paired runner. An interrupted artifact missing its required verification block
is also rejected. `verification.reconstruction` remains `not_run`: this command
does not execute MRI reconstruction. Existing hand-authored manifests without
the optional verification block remain supported by the runner.

## Connect to the paired runner

Use a separate evaluation pair and the same factory and precision settings:

```bash
python -m unet_gptq.train.run_exp \
  --model_path /absolute/path/to/ema_0.9999_150000.pt \
  --w4_model_path artifacts/gptq_calibration_001/weights_w4.pt \
  --w4_manifest artifacts/gptq_calibration_001/manifest.json \
  --conditions fp w4 \
  --data_root /data/preprocessed/evaluation \
  --file1 evaluation_12.pt --file2 evaluation_12_2.pt \
  --mask_path /absolute/path/to/mask.pt --mask_layout unshifted \
  --image_size 320 --model_options "$GPTQ_MODEL_OPTIONS" \
  --timestep_respacing 100 --refine_steps 20 \
  --device cuda:0 --use_fp16 --seeds 0 1 2 \
  --save_path results/fp_w4_001
```

The runner loads separate FP and W4 models, verifies the checkpoint hashes,
checks tensor differences, and refuses FP-vs-FP comparisons. Sampling, FFT,
mask/DC operations, diagnostics, and existing results are otherwise preserved.

## Validation scope

```bash
python -m pytest -q tests/test_gptq_calibration.py
python -m pytest -q
```

The CPU tests use a small timestep-conditioned Conv2d model and a synthetic
respaced diffusion. They exercise calibration hooks, coarse/refinement timestep
mapping, FP32 and mixed FP16 collection against the original FP weights, W4
grids, saving/reloading, output differences, source preservation, exclusions,
memory limits, deterministic state replay, CLI data selection, and the existing
runner's actual model-loading/manifest-validation paths. They also inject a
reload failure and check that it is recorded and rejected.

Actual pretrained DiffuseRecon checkpoint inference, MRI reconstruction,
FLAIR validity, CUDA execution, and packed INT4 kernels are **not executed by
these CPU tests**. They must be reported separately after the required model,
data, and execution environment are available.

Recorded validation on 2026-10-19 (Python 3.12, PyTorch 2.14.1+cpu):

| Check | Result |
| --- | --- |
| Complete repository CPU suite, `python -m pytest -q` | 45 passed, including 17 new calibration cases and 28 existing cases |
| Calibration and paired-runner `--help` | Passed; public package import works without optional local inspection/plotting files |
| Strict reload and FP/W4 runtime weight/output difference on the small model | Passed in FP32 and mixed FP16 modes |
| Injected reload failure / missing required verification | Recorded as failure and rejected by runner |
| Pretrained DiffuseRecon checkpoint / MRI reconstruction | Not run; checkpoint and calibration/evaluation data were not supplied in this workspace |
| CUDA execution | Not run; CUDA unavailable in the validation environment |
| Packed INT4 execution | Not implemented; dequantized weight simulation only |
