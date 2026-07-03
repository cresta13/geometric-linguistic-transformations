# GLT-MOLT Spectral-Null Operator Closure, 9 Models

Completed: 2026-07-03 10:55:17 +03:00

Script:

- `scripts/run_glt_molt_spectral_nulls.py`

Configuration:

- 9 multilingual embedding models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- PCA dimension 128
- ridge alpha: `100.0`
- 300 null samples per row
- spectral null generation: random Givens row/column rotations preserving each commutator's singular-value spectrum

Status:

- Finished with `9/9` model-alpha checkpoints.
- `failures: []`

## Main Result

The GLT-MOLT closure signal survives a singular-spectrum matched null. This is a stricter operator-level control than random subspaces, Frobenius norm matching, or signed-permutation matching: each null matrix preserves the observed commutator's singular values, then randomizes the left/right singular directions through Givens rotations.

Mean matrix-closure residual:

| Ridge alpha | Method | Observed | Spectral matched null | Mean empirical p |
|---:|---|---:|---:|---:|
| `100` | affine | `0.855` | `0.99985` | `0.00333` |
| `100` | linear | `0.882` | `0.99985` | `0.00347` |

Mean target cosine:

| Ridge alpha | linear | affine |
|---:|---:|---:|
| `100` | `0.712` | `0.640` |

## Interpretation

This run closes the main post-ridge-sweep concern more directly. The `alpha=100` operators are algebraically cleaner but worse endpoint predictors. A skeptical explanation was that the cleaner closure might be caused by generic shrinkage or by the commutator spectrum itself. The spectral null weakens that explanation: even when the null matrices retain the same singular-value spectra as the observed commutators, the observed commutators remain more compressible into the primitive `N,Q,M,T` operator span.

The result should remain conservative:

> GLT-MOLT finds weak but robust closure-like compression in ridge-regularized PCA-space linguistic operator maps. The signal survives random-subspace, norm-matched, signed-permutation matched, and singular-spectrum matched null controls, but it is not a formal Lie-algebra proof and does not make the learned maps better target predictors than additive deltas.

The model-level pattern is uneven. `xlm-roberta-base` and `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` show the cleanest affine closure, while `Qwen/Qwen3-Embedding-0.6B` is close to null-like in residual magnitude. This should be treated as evidence of model-dependence, not as a universal representation claim.

## Files

- Run config: `run_config.json`
- Run status: `run_status.json`
- Closure summary by alpha/method: `csv/spectral_closure_by_alpha_method.csv`
- Closure summary by pair: `csv/spectral_closure_summary.csv`
- Full closure rows: `csv/spectral_closure_all_models.csv`
- Operator fit summary: `csv/operator_fit_by_alpha_method.csv`
- Full operator fit rows: `csv/operator_fit_all_models.csv`
