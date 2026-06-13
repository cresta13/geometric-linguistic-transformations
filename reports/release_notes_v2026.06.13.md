# Release v2026.06.13

This release freezes the current research artifact for external review and DOI archiving.

## Scope

- Research code lives in `scripts/`.
- Experiment-specific result tables and figures live in `results/experiments/`.
- Aggregate reviewer-facing tables live in `results/`.
- Draft paper candidates live in `paper/articles/`.
- The current reviewer PDF is `reports/2026-06-13_reviewer_revised_report.pdf`.
- The current Python review environment is pinned in `requirements.txt`.

## Significant Results Preserved

### Track 1: Geometric Transformation Vectors

- Delta vectors retain transformation-class information beyond simple endpoint-only baselines in several holdout regimes.
- Syntax-holdout accuracy of `1.0` is treated conservatively as endpoint/surface leakage rather than deep syntactic generalization; the old all-holdout comparison plot is excluded from the archival draft.
- Multiseed endpoint/delta/concat ablations and McNemar tests are preserved in `results/`.
- Layerwise/pooling, syntax representation, and modern-model spot checks are preserved under `results/experiments/`.

Key entry points:

- Draft: `paper/articles/geometric-transformation-vectors/draft.md`
- Main aggregate table: `results/reviewer_ablation_table.csv`
- McNemar tests: `results/ablation_multiseed_mcnemar.csv`
- Full semantic holdout: `results/experiments/lie_llm_full_semantic_holdout_results/`
- Syntax representation ablation: `results/experiments/syntax_representation_ablation_results/`
- UPAT audit: `results/experiments/upat_audit_results/`

### Track 2: Lie-Style Linguistic Operators

- Pairwise composition audits show measurable noncommutativity in transformation sequences.
- Antisymmetry is explicitly framed as a tautological implementation check, not evidence.
- Third-order signed-permutation coherence is framed as a diagnostic rather than a formal Jacobi identity.
- Semantic-equivalence controls and signed-permutation multiple-testing summaries are preserved.

Key entry points:

- Draft: `paper/articles/lie-style-linguistic-operators/draft.md`
- Composition results: `results/experiments/lie_composition_results/`
- Signed-permutation results: `results/experiments/lie_algebraic_identities_results/`
- Semantic-equivalence control: `results/experiments/lie_semantic_equivalence_results/`

## Reproducibility

Run scripts from the repository root. Example:

```powershell
.\.venv\Scripts\python.exe scripts\build_research_report.py
```

Large intermediate vector caches such as `*.npy`, local virtual environments, IDE files, and review zip archives are intentionally excluded from git.

## Status

This release is an archival research snapshot for review and citation. It is not a peer-reviewed article release.
