# GLT-MOLT Compact Spectral PCA Sensitivity, PCA-128

Completed: 2026-07-08 23:41:54 +03:00

Script:

- `scripts/run_glt_molt_spectral_nulls.py`
- launcher: `scripts/launch_glt_molt_compact_pca_sweep.ps1`

Configuration:

- 5 stable multilingual encoder models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- PCA dimension: `128`
- ridge alpha: `100.0`
- 300 spectral null samples per row
- 256 random Givens rotations for row and column directions

Completed models:

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers/LaBSE`
- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `distilbert-base-multilingual-cased`

Status:

- The run finished with `5/5` model-alpha checkpoints.
- `failures: []` in `run_status.json`.

## Main Result

The GLT-MOLT spectral-null closure signal survives a PCA-128 setting on the compact stable-model subset.

Mean matrix-closure residual:

| PCA dim | Method | Observed | Spectral matched null | Mean empirical p |
|---:|---|---:|---:|---:|
| `128` | affine | `0.8153` | `0.9999` | `0.00332` |
| `128` | linear | `0.8581` | `0.9999` | `0.00332` |

Mean target cosine:

| PCA dim | linear | affine |
|---:|---:|---:|
| `128` | `0.6601` | `0.5570` |

Pair-level closure residuals remain below the spectral matched null for all six ordered-pair differences in both linear and affine variants.

## Interpretation

This is a sensitivity control, not a standalone proof of a Lie algebra. Together with the compact PCA-64 result, it shows that the operator-closure compression signal is not tied to a single PCA dimension on the stable multilingual subset.

The result should still be framed conservatively:

> GLT-MOLT commutators are substantially more compressible into the primitive `N,Q,M,T` operator span than singular-spectrum matched Givens nulls at PCA dimensions `64` and `128`, but this remains a controlled closure-compression diagnostic rather than a formal Lie-algebra result.

PCA-256 remains an optional follow-up. Given the repeated memory pressure from larger jobs, the next run should be launched only as a separate compact job, not as part of a combined long sweep.

## Files

- Status: `run_status.json`
- Configuration: `run_config.json`
- Closure summary: `csv/spectral_closure_by_alpha_method.csv`
- Pair summary: `csv/spectral_closure_summary.csv`
- Full closure rows: `csv/spectral_closure_all_models.csv`
- Operator fit summary: `csv/operator_fit_by_alpha_method.csv`
- Full operator fit rows: `csv/operator_fit_all_models.csv`
