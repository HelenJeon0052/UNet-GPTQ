# Image-quality metrics and result versions

The unit of evaluation is one 2-D complex image slice. Image estimates and
their references must have matching shapes and scaling; the metric code does
not fit or normalize them independently. Calculations use float64/complex128
after the reconstruction, so this does not change the model's precision.

For reference `x` and estimate `x_hat`, both sums below cover one slice:

- `magnitude_nmse = sum((abs(x_hat) - abs(x))**2) / sum(abs(x)**2)`.
- `complex_nmse = sum(abs(x_hat - x)**2) / sum(abs(x)**2)`.
- `magnitude_rmse = sqrt(mean((abs(x_hat) - abs(x))**2))`.
- `magnitude_psnr_db = 20*log10(peak) - 10*log10(magnitude_mse)`, where
  `peak = max(abs(x))` and `magnitude_mse` is the square of `magnitude_rmse`.

PSNR uses the reference slice's maximum magnitude as its peak. It is not the
maximum of the reconstruction, a fixed 1.0, or `max(x)-min(x)`. The row field
`reference_peak` records this value; the legacy `data_range` field is an alias
for the same peak, not a peak-to-peak measurement. PSNR is calculated for
each slice first; summaries use the arithmetic mean of those dB values.
Likewise, NMSE summaries average per-slice NMSEs rather than pooling slice
numerators and denominators. Per-slice rows are retained so both results can
be audited. Differences in reference intensity and slice weighting can make
other aggregation choices produce different numbers.

`simple_matrics.py` summarizes the rows saved in one run. `image_metrics()`
returns per-slice rows and their arithmetic mean for its input batch.
`run_exp.py` records one row per seed, condition, stage, output sample, and
slice. Paired W4 minus FP deltas are averaged over samples and slices within
each eligible seed, then averaged over eligible seeds. This seed-level average
is a separate second aggregation step, not a count of independent patients.
For historical compatibility, `run_exp.py`'s `pair_index` is a position
within the two-slice pair; `slice_in_pair` repeats it explicitly.

## Zero and nonfinite values

- If the reference slice is all zero, the NMSE denominator and PSNR peak are
  zero: both NMSEs and magnitude PSNR are undefined (`NaN`), and their status
  is `undefined_zero_reference`. Magnitude RMSE remains defined.
- For a nonzero reference and zero magnitude error, magnitude PSNR is `+inf`
  with status `positive_infinity`. This also occurs for a phase-only error,
  even though complex NMSE is positive.
- Slice means do not silently discard `NaN` or `+inf`. Summary rows include
  nonfinite counts. A paired delta requiring a nonfinite value is recorded as
  empty/`null`, and its summary reports undefined seed counts instead of
  computing an average from only the remaining seeds.
- Python results and CSV rows can contain `nan` or `inf`. The pilot JSON
  encoder uses the strings `NaN` and `+Infinity`, together with status fields,
  rather than emitting invalid JSON numeric tokens. The `run_exp.py` legacy
  `psnr_db` alias remains blank for nonfinite PSNR and retains
  `psnr_is_posinf`. Callers serializing the raw `image_metrics()` result must
  likewise encode nonfinite values explicitly.
- These image-quality rules do not change the separate DC residual's own
  zero-observation handling.

## Compatibility with older results

New rows, summaries and manifests identify `metric_schema_version: 2`. Pilot
`analyze_run()` writes `metrics_v2.csv`, `stage_summary_v2.csv`, and
`metrics_summary_v2.json`, leaving the corresponding old filenames untouched.
The standalone multi-run analyzer already writes into a new directory; its
rows and `provenance.json` also record the metric version. `run_exp.py`
requires a new output directory and keeps `psnr_db` as an explicitly
documented alias for magnitude PSNR. New `complex_nmse` and
`magnitude_psnr_db` columns are added alongside the existing columns.

Interpret existing outputs according to the code that created them:

| Earlier output | Meaning before metric version 2 |
| --- | --- |
| `run_exp.py`: `magnitude_nmse`, `psnr_db` | Per-slice magnitude NMSE and magnitude PSNR, respectively. |
| `simple_matrics.py`: `magnitude_nmse`, `complex_nmse`, `magnitude_psnr_db` | The names already distinguished magnitude and complex metrics; zero-reference handling has changed. |
| `image_metrics()`: `nmse`, `psnr` | The intended NMSE used complex error pooled across the input; PSNR used complex error and clamping. An `observation_kspace` name error could prevent this function from returning at all. |

The corrected `image_metrics()` no longer exposes the ambiguous `nmse` and
`psnr` keys. Its new scalar metrics are per-slice means, and its PSNR uses
magnitude error, so callers must update their keys and must not relabel an old
complex PSNR as the new magnitude PSNR. Historical files without a version
remain unversioned; this change does not rewrite or infer their definitions.
For ordinary finite nonzero references, previous `run_exp.py` and pilot
magnitude values retain their definitions, subject to normal floating-point
rounding. Recalculate from saved arrays to compare a particular run.

## Verification

From the repository root, with the project dependencies and matplotlib installed:

```bash
PYTHONPATH=src MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_metric_definitions.py' -v
```

The tests include phase-only changes, all-zero references, exact magnitude
reconstruction, PSNR peak choice, per-slice aggregation, and preservation of
old pilot files. The real three-path integration test requires PyTorch and
h5py; if they are missing it is reported as skipped.

[code created by codex]
