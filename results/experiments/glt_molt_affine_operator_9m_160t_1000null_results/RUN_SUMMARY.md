# GLT-MOLT Affine Operator Audit, 9 Models, 1000 Nulls

Completed: 2026-06-30 00:40:15 +03:00

Script:

- `scripts/run_glt_molt_affine_operator_audit.py`

Configuration:

- 9 multilingual embedding models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- PCA dimension 128
- ridge alpha `10.0`
- 1000 random-subspace nulls

Status:

- Finished with `9/9` model checkpoints.
- `failures: []`

Main result:

Additive centroid maps remain better endpoint predictors than learned matrix maps, while learned linear and affine operators continue to show weak but systematic commutator closure against random-subspace nulls.

One-step target reconstruction, averaged over operators:

| Method | Mean target cosine |
|---|---:|
| additive | `0.903` |
| linear | `0.738` |
| affine | `0.695` |

Ordered composition target prediction, averaged over pairs:

| Method | Mean AB cosine | Mean BA cosine | Mean commutator cosine |
|---|---:|---:|---:|
| additive | `0.821` | `0.835` | `0.002` |
| linear | `0.591` | `0.615` | `0.010` |
| affine | `0.478` | `0.520` | `-0.017` |

Matrix commutator closure:

| Method | Mean closure residual | Residual range | Random-subspace null | Mean empirical p |
|---|---:|---:|---:|---:|
| linear | `0.975` | `0.964-0.985` | `0.9999` | `0.0020` |
| affine | `0.971` | `0.958-0.987` | `0.9999` | `0.0037` |

Matrix Jacobi-like residual:

| Method | Mean relative Jacobi operator norm | Range |
|---|---:|---:|
| linear | `0.067` | `0.062-0.070` |
| affine | `0.064` | `0.056-0.073` |

Interpretation:

The 1000-null rerun confirms the 300-null pattern rather than changing the story. Additive deltas are still the better predictive maps. Matrix operators are weaker endpoint predictors but produce reproducible algebraic diagnostics: commutators are not close to exactly closed, yet their residuals are consistently below random rank-matched subspaces, and Jacobi-like residuals remain low.

This should be framed as weak closure-like compression in ridge-regularized PCA-space maps, not as evidence of a formal Lie algebra. The next GLT-MOLT diagnostic should sweep `ridge_alpha` to test whether the algebraic cleanliness is a ridge-smoothing artifact or persists across predictive/regularization tradeoffs.
