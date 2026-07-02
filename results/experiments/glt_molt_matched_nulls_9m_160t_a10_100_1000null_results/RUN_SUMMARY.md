# GLT-MOLT Matched-Null Operator Closure, 9 Models, 1000 Nulls

Completed: 2026-07-02 12:47:48 +03:00

Script:

- `scripts/run_glt_molt_matched_nulls.py`

Configuration:

- 9 multilingual embedding models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- PCA dimension 128
- ridge alphas: `10.0`, `100.0`
- 1000 null samples per row
- null controls:
  - random subspace
  - Gaussian norm-matched operator null
  - signed-permutation matched operator null

Status:

- Finished with `18/18` model-alpha checkpoints.
- `failures: []`

## Main Result

The GLT-MOLT closure signal survives stronger matched-null controls. Learned linear and affine operator commutators remain more compressible into the primitive operator span than random-subspace, Gaussian norm-matched, and signed-permutation matched operator nulls.

Mean matrix-closure residual:

| Ridge alpha | Method | Observed | Random-subspace null | Gaussian norm-matched null | Signed-permutation matched null |
|---:|---|---:|---:|---:|---:|
| `10` | affine | `0.970` | `0.99988` | `0.99988` | `0.99978` |
| `10` | linear | `0.975` | `0.99988` | `0.99988` | `0.99977` |
| `100` | affine | `0.855` | `0.99988` | `0.99988` | `0.99989` |
| `100` | linear | `0.882` | `0.99988` | `0.99988` | `0.99989` |

Mean empirical p-values:

| Ridge alpha | Method | vs random subspace | vs Gaussian norm-matched | vs signed-permutation matched |
|---:|---|---:|---:|---:|
| `10` | affine | `0.00363` | `0.00358` | `0.00843` |
| `10` | linear | `0.00211` | `0.00206` | `0.00899` |
| `100` | affine | `0.000999` | `0.000999` | `0.000999` |
| `100` | linear | `0.000999` | `0.000999` | `0.000999` |

Mean target cosine:

| Ridge alpha | linear | affine |
|---:|---:|---:|
| `10` | `0.738` | `0.695` |
| `100` | `0.712` | `0.640` |

## Interpretation

This is an important GLT-MOLT control. The ridge sweep showed that operator closure becomes cleaner under stronger regularization, raising the possibility that the apparent algebraic structure is generic shrinkage. This matched-null audit weakens that objection: the observed commutators are still substantially more compressible than norm-matched and signed-permutation matched operator nulls.

The result should remain conservative:

> GLT-MOLT finds weak but robust closure-like compression in ridge-regularized PCA-space linguistic operator maps. The signal survives norm-matched and signed-permutation matched null controls, but it is still not a formal Lie-algebra proof and does not make the learned maps better target predictors than additive deltas.

The strongest closure appears at `alpha=100`, where target prediction is worse than at `alpha=10`. This separates algebraic compression from endpoint reconstruction and should be treated as a central finding rather than a nuisance.

## Files

- Run config: `run_config.json`
- Run status: `run_status.json`
- Closure summary by alpha/method: `csv/matched_closure_by_alpha_method.csv`
- Closure summary by pair: `csv/matched_closure_summary.csv`
- Full closure rows: `csv/matched_closure_all_models.csv`
- Operator fit summary: `csv/operator_fit_by_alpha_method.csv`
- Operator norm summary: `csv/operator_norms_summary.csv`
- Full operator fit rows: `csv/operator_fit_all_models.csv`
- Full operator norm rows: `csv/operator_norms_all_models.csv`
