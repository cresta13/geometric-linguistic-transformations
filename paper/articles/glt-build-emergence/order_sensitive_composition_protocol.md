# GLT-BUILD-02: Order-Sensitive Composition

This protocol introduces a genuinely noncommuting pair into the controlled
synthetic language. It keeps the GLT-BUILD-01 architecture, source-support
curriculum and held-out scene split, changing only the operation algebra.

## Operations

`R` swaps the grammatical subject and object. `M` toggles a mark on the current
subject. Both operations are involutions, but they do not commute. Starting from
an unmarked state with subject `Mira` and object `Nora`:

```text
M then R: Mira becomes marked, then moves to the object role.
R then M: Nora becomes the subject, then Nora becomes marked.
```

The rendered target states are therefore different. `T` toggles tense and `N`
toggles negation; these independent operations form the commuting control pair.

## Arms

| Arm | Training command prefixes |
| --- | --- |
| `single_only` | identity and `R`, `M`, `T`, `N` |
| `pair_exposed` | the same singles plus `RM` and `TN` |

Both arms see all 12 combinations of tense, negation and mark placement for
identity/single-operation examples. Pair examples use the base source form. The
role-swap-closed split contains 360 training and 88 held-out directed scenes.
The model is two layers, four heads, width 64, batch 64, 1600 updates and three
seeds, with checkpoints at 0/400/800/1600.

## Primary test

The primary contrast is the pair `RM` versus reverse order `MR`:

```text
R(M(x)) != M(R(x))
```

The model must produce both ordered outcomes correctly and distinctly under
sequential generation. `TN` versus `NT` is the commuting control:

```text
T(N(x)) = N(T(x))
```

Direct pair-prefix accuracy is reported separately from sequential execution.
Oracle-intermediate rows distinguish a failure of the first operation from a
failure to apply the second operation to the generated intermediate.

## Interpretation

A positive result would establish order-sensitive composition in a finite
discrete synthetic action. It would not establish a Lie algebra. A negative
result would show that the prior sequential result does not automatically extend
to noncommuting operations. Only after this discrete test is understood should
the project introduce continuous features and infinitesimal generators.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/run_glt_build_02_order_sensitive_composition.py --out-dir results/experiments/glt_build_02_order_sensitive_composition_20261009b_results --seeds 0,1,2 --steps 1600 --checkpoints 0,400,800,1600 --threads 2 --eval-batch-size 32
```

The runner validates exact-match labels, unique keys, expected row counts and
checkpoint hashes before writing `SUMMARY.md` and `decision.json`.
