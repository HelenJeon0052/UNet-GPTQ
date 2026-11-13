# Paired reconstruction and FP protocol comparison

These commands require Python, PyTorch, NumPy, h5py, an installed compatible
DiffuseRecon `improved_diffusion` package, a compatible pretrained checkpoint,
and trusted local MRI data. CUDA and the checkpoint are not supplied here.
Plotting additionally requires matplotlib. Install dependencies in your own
environment and verify the checkpoint's anatomy, contrast, architecture and
normalization before interpreting a FLAIR result.

The measurement used here is **synthetic single-coil k-space generated from a
coil-combined complex image**. It is not an acquired multi-coil forward model.
Normalization uses each full reference slice's maximum real/imaginary component
(an oracle scale for this simulation). `fp16` uses DiffuseRecon's selective
conversion; inspect recorded dtypes because not every operation is necessarily
FP16. FFT and data consistency are complex64 in both conditions.

## 1. Prepare one set of images and a mask

If the required four adjacent legacy pickle images and a centered/unshifted
mask already exist, use them directly and skip this step. Otherwise, from a
fastMRI HDF5 volume and the selected FP checkpoint, run a pilot once to export
them. This performs a real FP reconstruction and requires a compatible model
and data. Set the paths for your own machine:

```bash
python -m unet_gptq.train.inspect_utils \
  --ckpt /path/to/model.pt --h5 /path/to/brain.h5 \
  --slice-index 16 --crop-size 320 \
  --coarse-steps 100 --refine-steps 20 --dc-weight 1.0 \
  --precision fp16 --device cuda:0 --seed 0 \
  --export-pickles --export-neighbors --out /path/to/pilot_runs
```

The printed new run directory contains `slice15.pkl`, `slice16.pkl`,
`slice17.pkl`, `slice18.pkl` and `mask_centered.pkl`, as well as the FP
reconstruction and manifest. `--export-neighbors` requires a valid previous and
next adjacent slice; it verifies that the overlapping preprocessed central
slices match before exporting. The input pickles must be trusted.

## 2. Use one configuration for both runners

Copy `configs/paired_example.json` to an ignored local file, e.g.
`configs/paired_local.json`. Replace **all** example paths with your real
checkpoint, HDF5 file, the exported-pair directory and the mask. Set the exact
checkpoint architecture in `model_options`. `data.file1`, `data.file2`,
`data.before`, and `data.after` are relative to `data.data_root` unless
absolute. The config's checkpoint, HDF5, data root and mask paths are resolved
relative to the JSON file when not absolute.

Both entry points load the same checkpoint, saved mask/layout, model options,
sampling schedule, DC weight, sample count, seeds, device and precision.
Explicit CLI arguments override the shared file; the resolved arguments and
configuration-file SHA256 are saved. Avoid overrides when comparing runs.
When no config is passed, both CLIs now default to DC weight 0.5, FP32, and
100 coarse steps, but the older pilot's hard-coded model defaults can still
differ from factory defaults. Use the shared config for a fair comparison.

```bash
# FP pilot on the HDF5 pair with the saved shared mask.
python -m unet_gptq.train.inspect_utils \
  --config configs/paired_local.json --out /path/to/shared_pilots

# Same FP checkpoint/protocol on the exported pickle pair.
python -m unet_gptq.train.run_exp \
  --config configs/paired_local.json --conditions fp \
  --save_path /path/to/results/fp_paired
```

The pilot records `manifest.json`, per-slice metrics, inputs/outputs and the
actual counted U-Net forwards. The paired runner writes its input and output
arrays, source/environment metadata and `run.json`; by default its *untimed*
diagnostic replay reports actual NFE. It records input channel order, mask
and input hashes, normalization scales, schedules, seeds and DC policy.

Verify that the separately produced inputs and saved settings match before
interpreting differences. The existing analysis tool compares the input arrays
and reports known configuration differences:

```bash
python -m unet_gptq.train.simple_matrics \
  /path/to/shared_pilots/PRINTED_PILOT_DIRECTORY \
  /path/to/results/fp_paired/seed_0/fp/outputs.npz \
  --out /path/to/results/pilot_input_audit
```

Inspect `provenance.json`: expect `relation: same_inputs` and empty
`configuration_differences` and `configuration_unknown`. The input audit uses
a tolerance and does not prove that every random draw or backend operation is
identical. Pilot and runner may fail to agree; do not assume they match.

## 3. Compare single-pair versus adjacent-pair aggregation in FP

With the four adjacent inputs and the same shared config:

```bash
python -m unet_gptq.train.compare_fp_protocols \
  --config configs/paired_local.json \
  --save_path /path/to/results/fp_protocol_comparison
```

For the central pair `(i,i+1)`, the single protocol averages its coarse samples
then refines. The adjacent protocol computes coarse predictions for `(i-1,i)`,
`(i,i+1)`, `(i+1,i+2)`, averages overlapping predictions of `i` and `i+1`, and
refines that central pair. The central pair is run first in both procedures
after resetting the same seed. The program records whether its coarse outputs
and initial global RNG states agree, and returns a nonzero status if either
check fails. It does **not** prove per-step noise equivalence. Input adjacency
is supplied by the user and recorded; filenames do not prove slice identity.

Use `dc_weight=1.0` if testing the hard-DC version. This is a controlled
aggregation ablation inspired by the upstream code: samples run sequentially,
whereas the upstream script uses batches and its own surrounding procedure.
It is not an exact reproduction of that script or its published results.
The adjacent protocol uses three times as many coarse sample trajectories;
compare actual recorded NFE and timing as well as quality.

## 4. Create and use a GPTQ W4 artifact

Follow [the calibration guide](gptq_calibration.md) with the same checkpoint,
model options, schedule and precision. The manifest records the original
checkpoint hash, selected/excluded layers, the selected parameter-element
fraction, sidecar-grid reload and source preservation checks. The calibration
uses a forward-noising surrogate; it does not replay the reconstruction's
reverse/DC trajectory.

For separation checks, add verified `patient_id` or `exam_id` strings to each
calibration JSONL record and the evaluation `data` section of the shared JSON.
The runner rejects matching file hashes and known overlapping IDs. It verifies
patient-level separation only when all calibration IDs and the evaluation ID
are supplied; otherwise it may verify examination-level separation, or reports
`unverified`. The identity values themselves must be trustworthy. Require a
verified patient- or exam-level result for a study run:

```bash
python -m unet_gptq.train.run_exp \
  --config configs/paired_local.json --conditions fp w4 \
  --w4_model_path /path/to/artifact/weights_w4.pt \
  --w4_manifest /path/to/artifact/manifest.json \
  --require_split_validation --save_path /path/to/results/fp_w4
```

If only examination IDs are available, patient-level disjointness remains
unverified. Full split integrity across *all* validation/test manifests is a
separate dataset-level task. The W4 checkpoint contains dequantized grid values
in floating-point tensors; this supports quality experiments but does not prove
packed INT4 storage or acceleration.

## Metric definitions

The paired runner and `image_metrics()` use the same per-slice functions:

- `magnitude_nmse`: squared difference between magnitudes, divided by
  reference magnitude energy; ignores phase.
- `complex_nmse`: squared magnitude of the complex difference, divided by the
  same reference energy; sensitive to phase.
- `psnr_db`: magnitude PSNR with each reference slice's maximum magnitude as
  its data range. No per-output rescaling is performed.

`image_metrics()` returns the unweighted mean of defined per-slice metrics;
its legacy `nmse` and `psnr` keys alias magnitude values. An undefined or
infinite value yields a null aggregate instead of silently excluding slices.
This changes the old `nmse` interpretation, which previously used complex
error. Always record the metric name and aggregation used when comparing older
results.
