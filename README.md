# Geometric Linguistic Transformations

Research code, result tables, figures, and draft papers for experiments on linguistic transformation geometry in transformer embedding spaces.

## Current Status

This is an active research repository, not a submission-ready paper. The current record is intentionally conservative:

- Delta vectors often add transformation information beyond endpoint-only baselines.
- Syntax-holdout `1.0` results are treated as endpoint/surface leakage, not as evidence of deep generalization.
- UPAT hard-holdout results are reported as a boundary condition where `y_only` can beat `delta`.
- Lie-style antisymmetry is not evidence; it is a tautological implementation check.
- Third-order signed permutation coherence is framed as a diagnostic, not as a formal Lie algebra or Jacobi identity.
- Procrustes/cross-model transfer is a promising future direction, but needs null baselines before being promoted.

## Repository Map

- `paper/`
  - `research_program.md`: current research roadmap.
  - `reviewer_revision_plan.md`: reviewer-driven backlog and status.
  - `reviewer_response_round3.md`: latest reviewer-response note.
  - `articles/`: separate draft paper candidates.
  - `figures/`: curated figures used in drafts and reports.
- `research/`
  - `diary.md`: dated research diary.
  - `PROTOCOL.md`: working protocol.
- `reports/`
  - dated PDF packets for external review.
- `results/`
  - aggregated Track 1 ablation, McNemar, and confusion-analysis tables.
- `lie_*_results/`, `upat_*_results/`, `track1_*_results/`
  - experiment-specific CSVs, figures, and metadata.
- `run_*.py`, `lie_*.py`, `upat_*.py`
  - experiment scripts.
- `scripts/`
  - report/figure builders and post-hoc analysis scripts.

## Important Artifacts

- Main report: `reports/2026-06-13_reviewer_revised_report.pdf`
- Research roadmap: `paper/research_program.md`
- Track 1 draft: `paper/articles/geometric-transformation-vectors/draft.md`
- Track 2 draft: `paper/articles/lie-style-linguistic-operators/draft.md`

## Reproducibility Notes

The scripts require Python packages such as `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, and `matplotlib`.

Large intermediate vector caches (`*.npy`), local virtual environments, IDE files, `.env`, and review zip archives are intentionally excluded from git.

## License

MIT. See `LICENSE`.
