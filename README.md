# U-Net-GPTQ

Controlled experiments on weight quantization for diffusion-based MRI reconstruction.

This project studies how GPTQ weight quantization affects reconstruction quality, with particular attention to lesion regions, low-contrast structures, and fine spatial detail. A sequential experiment runner compares predefined conditions under a shared reconstruction protocol.

**First milestone:** automate the paired **FP16 vs. GPTQ W4A16** comparison. Add activation-precision experiments only after the baseline is reproducible.

> **Implementation status:** this README specifies the proposed repository layout, experiment protocol, and command-line interfaces. The listed modules and commands are implementation targets; this document does not establish that they are implemented or that experiments have been completed.

**Implemented calibration entry point:** `python -m unet_gptq.train.calibrate_gptq`
connects `Conv2dGPTQ` to the DiffuseRecon factory and writes `weights_w4.pt`,
`quantization.pt`, and `manifest.json` for `unet_gptq.train.run_exp`.
See [GPTQ calibration and validation](docs/gptq_calibration.md) for the executable
commands, calibration split/timestep policy, exclusions, and CPU test scope.
The artifact simulates W4 weights with floating-point convolution; it does not
implement packed INT4 inference. Real-checkpoint reconstruction results are not
established by the CPU integration tests.

## Research Questions

1. **Weight quantization:** How much does GPTQ W4A16 change MRI reconstruction quality relative to the original FP16 model under identical reconstruction conditions?
2. **Activation precision:** With GPTQ weights fixed, does assigning A8 to selected denoiser calls preserve more useful structure than assigning A8 randomly under a matched allocation budget?

The initial integration target is [DiffuseRecon](https://github.com/cpeng93/DiffuseRecon). Its reconstruction procedure should be wrapped by an adapter, with changes documented and applied consistently across conditions. Before a Brain FLAIR study, verify the selected checkpoint's anatomy, contrast, input representation, normalization, and FP reconstruction quality; checkpoint availability alone does not establish suitability.

## Experiment Matrix

“Same checkpoint” means that every condition originates from the same pretrained source weights, denoted by `W`. GPTQ produces one fixed quantized weight artifact, `W_hat`, which is reused by A1 and all B conditions.

| ID | Condition | Denoiser weights | Targeted activations | Purpose |
| --- | --- | --- | --- | --- |
| A0 | Original reference | Source checkpoint `W`, with FP16 inference | A16 at every call | Establish the unquantized reference |
| A1 | Weight-only GPTQ | Fixed GPTQ `W_hat` at W4 | A16 at every call | Isolate weight-quantization effects |
| B0 | Uniform activation quantization | Exactly the same `W_hat` as A1 | A4 at every call | Measure additional activation-quantization damage |
| B1 | Selected A8 allocation | Exactly the same `W_hat` as A1 | A8 at selected calls; A4 elsewhere | Test timestep-dependent activation precision |
| B2 | Random A8 allocation | Exactly the same `W_hat` as A1 | A8 at randomly selected calls; A4 elsewhere | Control for the allocation budget |

**Precision notation:** A16 means FP16 activations without an added low-bit activation quantizer. A4 and A8 mean 4-bit and 8-bit activation quantization at explicitly listed quantization sites. W4 applies to the declared quantized layer set. Any excluded layers, floating-point operations, or higher-precision accumulations must be recorded.

B1 and B2 change **activation precision**, not weight precision. They do not re-run GPTQ or introduce W8 weights.

Interpret the comparisons separately:

- **A0 vs. A1:** effect of weight-only quantization.
- **A1 vs. B0:** additional effect of activation quantization.
- **B0 vs. B1:** benefit of the selected A8 allocation.
- **B1 vs. B2:** benefit of selecting where to spend the A8 budget.

## Shared Experimental Controls

Store the common protocol in one configuration and resolve it once per experiment group.

| Component | Fixed across paired conditions |
| --- | --- |
| Model | Source checkpoint checksum, architecture, input/output representation, inference precision policy |
| Data | Case and slice manifests, preprocessing, normalization, reference images, annotation transforms |
| Acquisition | Observed k-space, exact mask, acceleration, coil handling, sensitivity maps when applicable |
| Sampling | Sampler, timestep/noise schedule, coarse/refinement stages, actual denoiser-call schedule, NFE |
| Data consistency | Update rule, strength, iterations, placement in the sampling loop, numerical precision |
| Randomness | Case-specific initial noise and all stochastic sampling draws |
| Evaluation | Metrics, image scaling, crop, ROI definitions, frequency bands, aggregation |
| Execution | Hardware, software environment, batch size, timing procedure |

**NFE** is the actual number of denoiser evaluations. Save each call's index, diffusion timestep or noise level, stage, and input shape. Nominal sampling steps are not necessarily NFE: the [DiffuseRecon reference implementation](https://github.com/cpeng93/DiffuseRecon) distinguishes initial coarse sampling from refinement.

Keep MRI forward operations, FFTs, data consistency, and metric computation under an explicitly fixed precision policy. Denoiser activation quantization should not silently change their arithmetic.

Use independent random streams for calibration selection, masks, sampling, and B2 allocation. A stable key such as `(case_id, slice_id, sampling_seed, call_index, draw_type)` should determine sampling noise independently of condition order. Saving or replaying the noise draws is preferable when branch-dependent random-number consumption could break pairing.

## Experimental Workflow

### Stage A: Establish the FP16 / W4A16 Baseline

1. Freeze the source checkpoint, dataset manifests, reconstruction settings, and quantized layer list.
2. Collect GPTQ calibration inputs from the calibration split across the intended diffusion states.
3. Generate `W_hat` once and save it with quantization parameters and calibration provenance.
4. Execute A0 and A1 sequentially with paired measurements and random draws.
5. Save reconstructions, quality metrics, denoiser-call diagnostics, and measured resource use.

A0 must load the original checkpoint into a fresh model. A1 must load the saved quantized artifact into another fresh model. **Never reuse an object that was quantized in place as the A0 reference.**

Cache GPTQ artifacts by the source checkpoint, calibration manifest, quantization settings, and implementation revision. A cache mismatch should require a new artifact rather than silently reusing or overwriting an incompatible one.

Calibration provenance should include sample IDs, diffusion-state selection, calibration dtype, group size, damping, quantization symmetry, scales/zero points, and excluded layers. Freeze these settings for A1 and all B conditions.

### Stage B: Select and Freeze the Activation Policy

1. Use calibration data to fit activation quantizer statistics for the declared sites and bitwidths.
2. Use A0/A1 diagnostics to identify candidate sensitive calls or noise regions.
3. On validation data, check whether replacing A4 with A8 at those calls actually improves the chosen reconstruction objective.
4. Fix the selected call set, allocation budget, quantizer settings, and B2 random-allocation seeds.
5. Save the policies before evaluating the final test split.

A large A0–A1 error identifies a weight-quantization sensitivity signal. It does **not** demonstrate that higher activation precision will repair that error. The A4-to-A8 validation ablation tests that hypothesis directly.

For local diagnostics, evaluate compared denoisers on the same input state. For end-to-end evaluation, let each reconstruction follow its own trajectory while keeping external random draws paired.

All B conditions should use the same pre-calibrated A4/A8 parameter bank and quantization sites; only the allocation policy changes. Disable observer updates during final evaluation.

### Final Evaluation

Run A0, A1, B0, B1, and B2 with frozen artifacts and policies on the held-out test manifest.

B1 and B2 must promote the same number of eligible denoiser calls to A8. When call shapes or coarse/refinement costs differ, also match allocation counts within stage/shape groups or use a declared cost-weighted budget. Equal call counts alone do not guarantee equal runtime or memory.

Use several predeclared B2 allocation seeds to measure the variability of the random control. Keep these separate from sampling seeds and report all repetitions.

Split by patient where reliable linkage exists; otherwise split by examination and state that limitation. Calibration, validation, and final test sets must not overlap at the chosen grouping level. Keep slices from the same examination together.

## Proposed Repository Structure

Files marked `[B]` are deferred until the A0/A1 runner works. Package `__init__.py` files are omitted below for readability.

```text
U-Net-GPTQ/
├── README.md
├── pyproject.toml                    # Package metadata and dependencies
├── .gitignore
├── configs/
│   ├── base.yaml                     # Shared protocol; initial default: A0, A1
│   └── conditions.yaml               # A0/A1 specs; add B0/B1/B2 in Stage B
├── manifests/
│   ├── calibration.csv
│   ├── validation.csv
│   └── test.csv
├── artifacts/                        # Generated, immutable experiment inputs
│   ├── model_spec.json               # Architecture and checkpoint compatibility
│   ├── sampling_plan.json            # Call indices, timesteps, stages, shapes
│   ├── quantization/
│   │   ├── weights_w4.pt              # W_hat, scales, zero points, layer metadata
│   │   └── calibration.json          # Source/config/manifest hashes and settings
│   └── policies/                     # [B] Frozen before final test
│       ├── activation_stats.pt
│       ├── selected_a8.json
│       └── random_a8_seed_*.json
├── src/
│   └── unet_gptq/
│       ├── data/
│       │   ├── fastmri_dataset.py     # Loading, preprocessing, manifest selection
│       │   └── sampling_masks.py     # Deterministic masks or saved-mask loading
│       ├── models/
│       │   ├── diffuserecon_adapter.py
│       │   └── checkpoint.py         # Fresh model loading and artifact validation
│       ├── quantization/
│       │   ├── gptq.py               # GPTQ calibration, quantization, serialization
│       │   ├── conv_gptq.py          # Convolution input-patch handling
│       │   ├── activation.py         # [B] A4/A8 observers and quantizers
│       │   └── precision_policy.py   # [B] Call-index-to-activation-bit mapping
│       ├── diffusion/
│       │   ├── schedules.py          # Explicit call schedule and timestep mapping
│       │   └── samplers.py           # Fixed reconstruction sampling procedure
│       ├── mri/
│       │   ├── forward_operator.py
│       │   └── data_consistency.py
│       ├── engine/
│       │   ├── runner.py             # Sequential condition orchestration
│       │   ├── reconstructor.py      # One case, condition, and sampling seed
│       │   └── evaluator.py          # Metrics and examination-level aggregation
│       ├── metrics/
│       │   ├── image_quality.py
│       │   ├── frequency.py
│       │   └── lesion_roi.py
│       └── utils/
│           ├── config.py             # Config validation and resolution
│           ├── reproducibility.py    # Stable case seeds and noise replay
│           └── io.py                 # Manifests, hashes, logs, result writing
├── scripts/
│   ├── run_experiments.py            # Main entry point: one command, many conditions
│   ├── calibrate_gptq.py             # Optional explicit weight-artifact preparation
│   ├── reconstruct.py               # Single-condition debugging entry point
│   └── build_precision_policy.py     # [B] Calibration/validation policy preparation
├── tests/
│   ├── test_condition_isolation.py
│   ├── test_paired_randomness.py
│   ├── test_quantization_roundtrip.py
│   ├── test_split_integrity.py
│   └── test_precision_policy.py      # [B] Frozen policy and matched allocation
└── outputs/                          # Separate experiment groups and conditions
```

Keep raw MRI data and source checkpoints outside the repository, or in ignored local directories referenced by configuration. Commit code, example configurations, and shareable manifest schemas. Exclude generated weight files, data-bearing manifests, reconstructions, and large outputs from Git.

The `src/unet_gptq/` package keeps imports under one namespace. Scripts should only parse arguments and call package functions; quantization and reconstruction logic belong in the package.

| Entry point or module | Responsibility |
| --- | --- |
| `scripts/run_experiments.py` | Parse the requested condition list and start the experiment group |
| `engine/runner.py` | Validate inputs, prepare/reuse GPTQ, reset state, run conditions sequentially, collect results |
| `engine/reconstructor.py` | Execute one reconstruction using the supplied model, measurement, schedule, and randomness |
| `scripts/reconstruct.py` | Debug one condition on a small case subset |
| `scripts/build_precision_policy.py` | Build and freeze Stage B policies from calibration/validation only |
| `tests/` | Verify experimental invariants; this is not the experiment runner |

No separate analysis package or configuration framework is required for the first milestone. A single shared YAML file plus a small condition registry is sufficient. A short pilot may expose the same fields as CLI arguments, provided the runner saves the complete resolved configuration.

## Planned Command-Line Interface

The following commands describe the intended interface and assume that the corresponding modules have been implemented and the package installed.

```bash
# First milestone: prepare/reuse one GPTQ artifact, then run A0 and A1 sequentially.
python scripts/run_experiments.py \
  --config configs/base.yaml \
  --conditions A0 A1

# Optional: prepare the GPTQ artifact separately.
python scripts/calibrate_gptq.py --config configs/base.yaml

# Stage B: fit activation statistics and validate/freeze the selected and random policies.
python scripts/build_precision_policy.py --config configs/base.yaml

# Final evaluation: require frozen Stage B artifacts and run every condition sequentially.
python scripts/run_experiments.py \
  --config configs/base.yaml \
  --conditions A0 A1 B0 B1 B2
```

The main runner should reject B conditions when their required artifacts are missing or incompatible. It must not select time regions or fit quantizers from test data.

Required configuration fields include source checkpoint and model specification, dataset root, split manifests, mask definition or artifact, sampler and schedule, expected NFE, DC settings, precision policy, seeds, GPTQ settings, and output directory.

## Runner Responsibilities

For each experiment group, the runner should:

1. Validate configuration, split separation, source identity, and artifact compatibility.
2. Prepare or load one compatible GPTQ artifact when any quantized condition is requested.
3. Resolve paired measurements, masks, and random draws for the evaluation cases.
4. Execute conditions in the requested order, creating a fresh model and resetting sampler state, hooks, observers, and caches. Check the observed call schedule and NFE against the shared sampling plan.
5. Save each completed case with its condition, case ID, sampling seed, and allocation seed.
6. Record failures explicitly and return a nonzero exit status if any requested run fails.

If resumable execution is added, skip only completed cases whose configuration and artifact hashes match. Never silently overwrite prior experiment groups or present incomplete cases as successful results.

## Outputs and Evaluation

```text
outputs/<experiment_id>/
├── resolved_config.yaml
├── provenance.json                   # Source/quantized hashes, revisions, environment
├── evaluation_manifest.csv
├── sampling_plan.json
├── conditions/
│   ├── A0/
│   ├── A1/
│   ├── B0/                           # [B]
│   ├── B1/                           # [B]
│   └── B2/allocation_seed_*/          # [B] Separate random-control repetitions
└── summary.csv
```

Each condition directory contains `reconstructions/`, `metrics.csv`, `timestep_errors.csv`, `costs.json`, and `run.log`. Identify diagnostics as local common-input errors or free-running trajectory errors; do not merge the two.

| Evaluation area | Planned measurements |
| --- | --- |
| Global quality | PSNR, SSIM, NMSE against the reconstruction reference |
| Local structure | ROI PSNR/SSIM and image-gradient agreement in predefined regions |
| Frequency preservation | Band-wise error, HF-SQNR, and optionally FRC with fixed preprocessing |
| Measurement consistency | Pre/post-DC residuals and DC correction magnitude |
| Timestep sensitivity | Denoiser output error at common input states; end-to-end trajectory divergence |
| Resource use | Serialized artifact size, peak device memory, reconstruction latency, actual NFE |

Aggregate paired quality differences by examination. For larger studies, report uncertainty by resampling independent examination/patient groups rather than treating slices or repeated seeds as independent patients.

Use [fastMRI+ annotations](https://github.com/microsoft/fastmri-plus) only after verifying their alignment with the reconstruction crop and orientation. The provided pathology boxes support ROI evaluation; exact lesion-boundary metrics require suitable contour annotations.

Post-DC residuals alone cannot establish structural preservation. Under an exact hard projection, enforced measurement residuals may be near zero despite errors in unmeasured information. High-frequency energy alone can also increase because of noise; compare it with reference-based structural measurements.

**Numerical simulation and execution performance must be reported separately.** Fake quantization or dequantized W4 weights can support reconstruction-quality experiments, but they do not establish packed low-bit storage or faster inference. Record the actual storage format and execution backend. Measure latency with synchronization and warmup, and exclude diagnostic hooks and file I/O from the timed reconstruction scope.

## Implementation Milestones

- **M1:** Load the selected checkpoint and reproduce a small FP16 reconstruction pilot.
- **M2:** Serialize/reload one GPTQ artifact and complete paired A0/A1 runs.
- **M3:** Add activation quantization, validation-based policy selection, and B0/B1/B2.
- **M4:** Run the frozen protocol on held-out data and evaluate a supported low-bit backend if available.

Verify the invariants that protect the comparisons: A0 remains unchanged after quantized runs; every quantized condition loads the same weight artifact; paired noise is independent of execution order; splits are disjoint; and B1/B2 meet the declared allocation budget.

Lesion-aware GPTQ calibration, low-rank residual correction, weight mixed precision, DDPM U-Net alternatives, and NCSN++ transfer checks remain possible extensions. Each introduces a separate experimental factor and should receive its own controlled comparison after the initial protocol is stable.

## Planned Stack and References

Python, PyTorch, NumPy, h5py, PyYAML, image-quality metric utilities, and pytest are sufficient for the initial implementation. Record exact package versions once the DiffuseRecon adapter is validated; no broad sweep manager is required.

- [DiffuseRecon — source implementation](https://github.com/cpeng93/DiffuseRecon).
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) and [reference code](https://github.com/IST-DASLab/gptq). Adapting GPTQ to the selected U-Net requires explicit handling of supported layer types.
- [fastMRI — data utilities and benchmark code](https://github.com/facebookresearch/fastMRI).
- [fastMRI+ — pathology annotations and coordinate guidance](https://github.com/microsoft/fastmri-plus).

## License

This project is in an early experimental stage, with implementation and validation ongoing. A license for the original project code has not yet been selected.

Third-party code, datasets, and pretrained checkpoints remain subject to their respective licenses and terms of use.
