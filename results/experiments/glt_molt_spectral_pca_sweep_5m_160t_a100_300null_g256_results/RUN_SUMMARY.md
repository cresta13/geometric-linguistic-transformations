# GLT-MOLT Compact Spectral PCA Sensitivity, PCA-64

Completed PCA slice: 2026-07-08 01:29:18 +03:00

Script:

- `scripts/run_glt_molt_spectral_pca_sweep.py`
- launcher: `scripts/launch_glt_molt_compact_pca_sweep.ps1`

Configuration:

- 5 stable multilingual encoder models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- completed PCA dimension: `64`
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

- `pca_64` finished with `5/5` model-alpha checkpoints.
- `failures: []` in `pca_64/run_status.json`.
- `pca_128` and `pca_256` were not completed in this compact run and should not be cited from this folder.

## Main Result

The GLT-MOLT spectral-null closure signal survives a lower-dimensional PCA-64 setting on the compact stable-model subset.

Mean matrix-closure residual:

| PCA dim | Method | Observed | Spectral matched null | Mean empirical p |
|---:|---|---:|---:|---:|
| `64` | affine | `0.8285` | `0.9996` | `0.00332` |
| `64` | linear | `0.8701` | `0.9995` | `0.00335` |

Mean target cosine:

| PCA dim | linear | affine |
|---:|---:|---:|
| `64` | `0.6941` | `0.5899` |

## Interpretation

This is a useful sensitivity control, not a full PCA sweep. The result shows that the spectral-null closure effect is not unique to the original PCA-128 setting: on five stable models, PCA-64 still yields commutators that are substantially more compressible into the primitive `N,Q,M,T` operator span than singular-spectrum matched Givens nulls.

The result remains conservative. It does not establish a formal Lie algebra, and it does not complete the planned PCA-dimension ablation. The missing next step is to run separate, smaller `pca_128` and `pca_256` jobs instead of one large sweep.

## Files

- PCA-64 status: `pca_64/run_status.json`
- PCA-64 closure summary: `pca_64/csv/spectral_closure_by_alpha_method.csv`
- PCA-64 pair summary: `pca_64/csv/spectral_closure_summary.csv`
- PCA-64 full closure rows: `pca_64/csv/spectral_closure_all_models.csv`
- PCA-64 operator fit summary: `pca_64/csv/operator_fit_by_alpha_method.csv`
- PCA-64 full operator fit rows: `pca_64/csv/operator_fit_all_models.csv`
