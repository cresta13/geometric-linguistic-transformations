# GLT-MOLT Ridge Sweep, 9 Models, 300 Nulls

Completed: 2026-07-01 04:24:52 +03:00

Script:

- `scripts/run_glt_molt_ridge_sweep.py`

Configuration:

- 9 multilingual embedding models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- PCA dimension 128
- ridge alphas: `0.1`, `1.0`, `10.0`, `100.0`
- 300 random-subspace nulls

Status:

- Finished with `36/36` model-alpha checkpoints.
- `failures: []`

## Main Result

The ridge sweep confirms that the GLT-MOLT algebraic signal depends on the predictive/regularization tradeoff. Stronger ridge regularization makes learned operator maps algebraically cleaner, but not uniformly better predictors.

Mean one-step target cosine:

| Ridge alpha | additive | linear | affine |
|---:|---:|---:|---:|
| `0.1` | `0.903` | `0.681` | `0.659` |
| `1.0` | `0.903` | `0.712` | `0.685` |
| `10.0` | `0.903` | `0.738` | `0.695` |
| `100.0` | `0.903` | `0.712` | `0.640` |

Mean matrix-closure residual:

| Ridge alpha | linear | affine | random-subspace null |
|---:|---:|---:|---:|
| `0.1` | `0.995` | `0.994` | `~0.9999` |
| `1.0` | `0.991` | `0.991` | `~0.9999` |
| `10.0` | `0.975` | `0.970` | `~0.9999` |
| `100.0` | `0.882` | `0.855` | `~0.9999` |

Mean relative Jacobi-like operator norm:

| Ridge alpha | linear | affine |
|---:|---:|---:|
| `0.1` | `0.093` | `0.097` |
| `1.0` | `0.080` | `0.083` |
| `10.0` | `0.067` | `0.064` |
| `100.0` | `0.072` | `0.042` |

## Interpretation

The earlier `alpha=10` MOLT result is not an isolated artifact of one regularization value: closure residuals are below random-subspace nulls across all tested alphas. However, the cleanest closure appears under stronger regularization, especially `alpha=100`, where matrix commutator norms are also much smaller.

This matters. The result supports a real operator-geometry diagnostic, but it also shows that algebraic cleanliness can be amplified by ridge smoothing. The current claim should therefore remain conservative:

> GLT-MOLT finds weak closure-like compression in ridge-regularized PCA-space operator maps, with a clear tradeoff between endpoint-prediction strength and algebraic smoothness.

The next operator-level test should separate this from generic shrinkage by adding norm-matched/shrinkage-matched operator nulls.

## Files

- Run config: `run_config.json`
- Run status: `run_status.json`
- Operator fit by alpha: `csv/operator_fit_by_alpha_method.csv`
- Matrix closure by alpha: `csv/matrix_closure_by_alpha_method.csv`
- Matrix Jacobi by alpha: `csv/matrix_jacobi_by_alpha_method.csv`
- Full operator fit rows: `csv/operator_fit_all_models.csv`
- Full matrix closure rows: `csv/matrix_closure_all_models.csv`
- Full matrix Jacobi rows: `csv/matrix_jacobi_all_models.csv`
