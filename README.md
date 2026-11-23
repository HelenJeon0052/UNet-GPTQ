# U-Net-GPTQ

This project studies how 4-bit GPTQ weight quantization changes image quality and data consistency (DC) in diffusion-based MRI reconstruction. The primary comparison uses the same source checkpoint for the FP16 reference and its weight-only W4A16 derivative.

**Current stage (23 November 2026).** The pilot applies a DiffuseRecon prior trained on knee MRI to brain FLAIR. We have measured the FP16 baseline and the effect of input phase on reconstruction. The repository also contains code for GPTQ calibration and paired FP/W4 inference. For the main brain study, the planned `brain_ddpm_scratch_v1` training stage will supply a brain FLAIR checkpoint to the existing GPTQ comparison protocol. The pilot results below measure FP16 only. [Experiment report, 23 November 2026 (Korean)](reports/UNet-GPTQ-model-experiment-report-261123.md)

## Measured FP16 pilot

The pilot used adjacent slices 7 and 8 from fastMRI AXFLAIR volume 6002469. We synthesized single-coil k-space observations from the coil-combined complex images and applied a 4× undersampling mask. Both runs used the same checkpoint, sampling seed 1, DC weight 0.5, 100 coarse steps and 20 refinement steps. With the phase check enabled, we applied a checkerboard phase to the images and shifted the mask and observed k-space to the corresponding coordinates.

| Mean over slices 7 and 8 | Phase check OFF | Phase check ON |
|---|---:|---:|
| Zero-filled magnitude NMSE ↓ | 0.03437 | 0.03437 |
| Coarse magnitude NMSE ↓ | 0.52007 | 0.08417 |
| Final magnitude NMSE ↓ | 0.29307 | 0.02374 |
| Final magnitude PSNR ↑ | 16.32 dB | 27.25 dB |
| Final relative L2 residual on measured k-space ↓ | 0.01615 | 0.00865 |

Changing the input representation reduced final magnitude NMSE by 91.9% on this slice pair. This measures the FP model's sensitivity to the input convention. The training domains, coil models and training methods used in Score-MRI and SSDiffRecon differ from this pilot, so their published scores cannot be compared directly with these values. [Pilot measurements](reports/UNet-GPTQ-model-experiment-report-261123.md) · [DiffuseRecon](https://arxiv.org/html/2203.04292v2#S3) · [Score-MRI](https://arxiv.org/abs/2110.05243) · [SSDiffRecon](https://arxiv.org/abs/2306.16654)

## Implemented code and verification

| Component | Current implementation or recorded check |
|---|---|
| Reconstruction | The pilot represents two adjacent complex slices as four real U-Net input channels. The sampler uses the saved observations and mask and applies the same DC weight during coarse sampling and refinement. [FP pilot](src/unet_gptq/train/inspect_utils.py) · [Sampler](src/unet_gptq/diffusion/samplers.py) |
| Weight quantization and paired inference | Conv2d GPTQ collects calibration statistics from input patches. The comparison runner loads the original FP weights and a saved W4 artifact into separate models and uses the same observations, mask and sampling seed for each condition. [Conv2d calibration](src/unet_gptq/quantization/conv_gptq.py) · [Artifact serialization](src/unet_gptq/quantization/gptq.py) · [Comparison runner](src/unet_gptq/train/run_exp.py) |
| Metrics | Five metrics recomputed from three FP runs matched their saved CSV values across 12 slice-stage rows. All 36 data pipeline and metric tests passed on 23 November 2026, including the PyTorch test comparing three evaluation paths. [Metric definitions](docs/metric_definitions.md) · [Verification results](reports/UNet-GPTQ-model-experiment-report-261123.md#v2-지표-정의와-검증-결과) |

### Current runner defaults

| CLI | Input | Model precision | Sampling seeds | Coarse / refine steps | DC weight |
|---|---|---|---|---|---|
| `inspect_utils` | HDF5 slice pair, generated mask | FP16 | `--seed 1` | 100 / 20 | `--dc-weight 0.5` |
| `run_exp` | Two saved slices and a mask | FP32 by default, `--use_fp16` for FP16 | `--seeds 0 1 2` | 100 / 20 | `--dc_weight 0.5` |

The Python function `run_pilot()` defaults to DC weight 1.0. Pass `dc_weight=0.5` when calling it directly to match the reported pilot. [Pilot arguments](src/unet_gptq/train/inspect_utils.py) · [Comparison runner arguments](src/unet_gptq/train/run_exp.py)

## Planned upstream training stage — `brain_ddpm_scratch_v1`

The next milestone is a full-precision checkpoint trained on brain FLAIR. Training and evaluating the prior in the same anatomical domain removes knee-to-brain transfer as a confounding factor when estimating the effect of quantization. This stage defines the source weights for the existing GPTQ protocol. Its deliverables below describe planned work.

1. **Train the brain prior.** Split fully sampled axial brain FLAIR exams by patient where linkage is available, otherwise by exam. Train a two-channel DDPM U-Net from scratch in FP32 using the real and imaginary components of one slice. Reserve development volume 6002469 and the GPTQ calibration cases outside the final test set.
2. **Validate the FP reconstruction.** Select the exponential moving average (EMA) checkpoint using validation noise-prediction error. Connect the two-channel model to a compatible sampler, multi-coil measurement operator and DC update. On held-out validation cases, check phase, intensity scaling, mask alignment and FP reconstruction quality.
3. **Hand off one checkpoint.** Record the validated checkpoint hash, preprocessing, measurement and sampling settings. Use that checkpoint both for FP16 inference and as the source of the GPTQ W4 weights. [Brain FLAIR training and validation plan](reports/UNet-GPTQ-model-experiment-report-261123.md#fp-체크포인트-확보와-복원기-검증)

## GPTQ comparison protocol after checkpoint selection

| Planned condition | Weights | Activation precision |
|---|---|---|
| A0 · FP reference | Selected brain FLAIR checkpoint | FP16 |
| A1 · Weight-only GPTQ | One W4 artifact generated from A0 | FP16 |
| B0 · Uniform A4 | Same W4 artifact as A1 | A4 at every targeted call |
| B1 · Selected A8 | Same W4 artifact as A1 | A8 at calls selected on validation data, A4 elsewhere |
| B2 · Random A8 control | Same W4 artifact as A1 | A8 at randomly selected calls under the B1 budget, A4 elsewhere |

**First compare A1 with A0.** Pair each held-out case across conditions using the same observations, mask, normalization, sampling schedule, denoiser evaluation count, random draws and DC settings. Report patient-level W4 minus FP differences in NMSE and PSNR alongside measured k-space residuals and errors at unmeasured frequencies. Evaluate lesion boundaries and high-frequency detail in aligned fastMRI+ regions. For patients with reference segmentations at two time points, also measure lesion-volume change error in mL.

**Then evaluate B0–B2.** Freeze the W4 weights and activation calibration before testing. Compare B1 with B0 to measure the effect of promoting selected calls to A8, and B1 with B2 to test whether call selection helps at the same budget. The current dequantized-weight path supports quality comparisons. Measuring INT4 latency and memory requires a packed INT4 execution backend. [Follow-up experiment design](reports/UNet-GPTQ-model-experiment-report-261123.md#종단-병변-평가와-후속-개선)

Third-party code, data and checkpoints are subject to their respective distribution terms. [DiffuseRecon implementation](https://github.com/cpeng93/DiffuseRecon) · [GPTQ](https://github.com/IST-DASLab/gptq) · [fastMRI](https://github.com/facebookresearch/fastMRI) · [fastMRI+](https://github.com/microsoft/fastmri-plus)
