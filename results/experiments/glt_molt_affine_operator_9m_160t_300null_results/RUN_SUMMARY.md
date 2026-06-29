# GLT-MOLT 9-Model Affine Operator Audit

Completed: 2026-06-29

This run extends the GLT-MOLT operator-map audit to nine multilingual embedding models, seven languages, 160 templates per language, PCA dimension 128, and 300 random-subspace null samples.

Models:

- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `sentence-transformers/LaBSE`
- `intfloat/multilingual-e5-large`
- `BAAI/bge-m3`
- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `distilbert-base-multilingual-cased`
- `intfloat/multilingual-e5-large-instruct`
- `Qwen/Qwen3-Embedding-0.6B`

Languages: English, Spanish, French, German, Russian, Chinese, Arabic.

## Main Result

Additive deltas remain the strongest target-reconstruction method, while learned linear and affine maps are weaker endpoint predictors but expose a small and systematic operator-closure signal.

Mean target cosine by operator:

| Method | `N` | `Q` | `M` | `T` |
| --- | ---: | ---: | ---: | ---: |
| additive | `0.865` | `0.882` | `0.943` | `0.922` |
| linear | `0.712` | `0.705` | `0.788` | `0.748` |
| affine | `0.665` | `0.675` | `0.731` | `0.710` |

The learned maps therefore should not be treated as better target predictors than additive deltas in this setting.

## Matrix Closure

For learned operator maps, matrix commutators are high-residual but consistently more compressible into the primitive operator span than random subspaces.

Mean closure residual ratios:

| Method | Pair range | Random-subspace null |
| --- | ---: | ---: |
| linear | `0.964-0.985` | `~0.9999` |
| affine | `0.958-0.987` | `~0.9999` |

At the current `N=300` null resolution, mean empirical p-values are small across pairs. This is evidence for weak but systematic closure-like compression, not exact algebraic closure.

## Jacobi-Like Operator Norms

Relative Jacobi-like operator norms are low across all triples:

| Method | `NQM` | `NQT` | `NMT` | `QMT` |
| --- | ---: | ---: | ---: | ---: |
| linear | `0.062` | `0.069` | `0.067` | `0.070` |
| affine | `0.062` | `0.056` | `0.066` | `0.073` |

This is the most Lie-adjacent result in the repository so far, but it remains a diagnostic over learned PCA-space maps. It should be followed by stronger nulls, held-out template families, and operator-learning variants before being promoted as a Lie-algebra claim.

## Files

- Run config: `run_config.json`
- Run status: `run_status.json`
- Operator fit summary: `csv/operator_fit_summary.csv`
- Composition prediction summary: `csv/composition_prediction_summary.csv`
- Matrix closure summary: `csv/matrix_closure_summary.csv`
- Jacobi-like operator summary: `csv/matrix_jacobi_summary.csv`
- Closure figures: `figures/matrix_closure_linear.png`, `figures/matrix_closure_affine.png`
