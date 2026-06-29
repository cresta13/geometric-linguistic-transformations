# Geometric Linguistic Transformations

Research code, result tables, figures, and draft papers for experiments on linguistic transformation geometry in transformer embedding spaces.

This repository develops **GLT** (**Geometric Linguistic Transformations**), a research program for probing whether linguistic transformations appear as reusable geometric objects in transformer embedding spaces.

Current tracks:

- **GLT-DV**: delta-vector diagnostics with endpoint controls.
- **GLT-SPOT**: signed-permutation operator tests for ordered composition.
- **GLT-XFER**: cross-model transformation-transfer stress tests.
- **GLT-MOLT**: planned matrix/operator extension motivated by Linear Relational Decoding.
- **GLT-AFFECT**: graded affective-transformation geometry, starting with text-only emotional polarity scales and reserving sensory grounding claims for independent perceptual data.
- The first GLT-AFFECT MVP suggests that text-embedding affective polarity is curved rather than a simple love/hate antipodal axis; marker-only, lexical-specificity, and bootstrap contrast controls weaken but preserve an affect-leading same-direction signal, so the claim is treated as cautious lexical affect-geometry evidence rather than grounded affect.

## Current Status

This is an active research repository, not a submission-ready paper. The current record is intentionally conservative:

- Delta vectors often add transformation information beyond endpoint-only baselines.
- Syntax-holdout `1.0` results are treated as endpoint/surface leakage, not as evidence of deep generalization.
- UPAT hard-holdout results are reported as a boundary condition where `y_only` can beat `delta`.
- Lie-style antisymmetry is not evidence; it is a tautological implementation check.
- Third-order signed permutation coherence is framed as a diagnostic, not as a formal Lie algebra or Jacobi identity.
- The 2026-06-27 extended multilingual GLT-SPOT audit scales the signed-permutation test to 7 models, 7 languages, 96 templates per language, and 5000 signed-null repeats; all four triples remain below null in all 49 model-language cells, while endpoint-derived controls remain a live limitation.
- The 2026-06-27 third-order endpoint-control audit confirms that endpoint and delta features strongly recover triple identity on held-out languages, so GLT-SPOT remains evidence for controlled signed-permutation coherence rather than endpoint-independent algebra.
- The 2026-06-28 new-model GLT-SPOT checks replicate the below-null signed-permutation pattern on `intfloat/multilingual-e5-large-instruct` and `Qwen/Qwen3-Embedding-0.6B` across all seven tested languages.
- The 2026-06-28 new-model endpoint-subspace residualization check shows that this below-null pattern survives removal of linear endpoint-derived triple-label, endpoint-position, and cyclic-sign subspaces in both newer models.
- The 2026-06-29 9-model endpoint-subspace residualization audit extends that stress test to older and newer multilingual embedding models; all four triples remain below exact signed-null after removing endpoint-derived linear rowspaces in all `63/63` model-language cells.
- The 2026-06-24 endpoint-subspace residualization audit shows that the multilingual signed-permutation signal largely survives removal of endpoint-derived sign, triple-label, and endpoint-position probe subspaces.
- Procrustes/cross-model transfer now survives `N=1000` null controls and held-out anchor alignment-size controls. RISE/MDV-style prototype comparisons, non-leaky hybrid feature-transfer, and spherical delta steering tests are now added; current results separate target-cosine accuracy from transformation-neighborhood retrieval, so Track 3 remains a stress-test/comparison track pending calibration, confidence intervals, and anchor-domain robustness.

## Positioning

This repository should be cited as a software/research-artifact snapshot, not as a peer-reviewed publication.

Closest related work:

- Freenor and Alvarez 2026, RISE: geometric rotations for discourse-level semantic-syntactic transformations across languages and embedding models. This is the closest neighboring work and is stronger than this repository on cross-lingual/cross-model scope. This repository instead uses direct endpoint/delta representations, classifier/ablation diagnostics, null-controlled Procrustes transfer stress tests, and ordered-composition diagnostics; it also reports endpoint leakage and hard-holdout failures explicitly.
- Xia and Kalita 2025, Linear Relational Decoding of Morphology in Language Models: relation-specific Jacobian-derived matrix operators can faithfully approximate many morphological relations in GPT-J and Llama-7b, including multilingual morphology. This motivates a future matrix/operator version of the Lie-style track, where commutators can be computed over learned affine or multiplicative maps rather than only over endpoint deltas.
- Park, Choe, and Veitch 2023/2024: formal Linear Representation Hypothesis framing for linear directions and representation geometry in LLMs. This repository is empirical and diagnostic rather than a formal LRH theory paper.
- De Raedt et al. 2021: geometric cross-lingual linguistic transformations with pretrained autoencoders. This repository studies paired-sentence displacement vectors and composition diagnostics in transformer embedding spaces, not cross-lingual autoencoder transfer.

## Repository Map

- `paper/`
  - `research_program.md`: current research roadmap.
  - `related_work_positioning.md`: RISE/LRH positioning note and claim-boundary checklist.
  - `reviewer_revision_plan.md`: reviewer-driven backlog and status.
  - `reviewer_response_round3.md`: latest reviewer-response note.
  - `articles/`: separate draft paper candidates.
  - `figures/`: curated figures used in drafts and reports.
- `research/`
  - `diary.md`: dated research diary.
  - `critical_review_2026-06-14.md`: current skeptical state review and next-direction map.
  - `PROTOCOL.md`: working protocol.
- `reports/`
  - dated PDF packets for external review.
- `results/`
  - aggregate Track 1 ablation, McNemar, and confusion-analysis tables.
  - `results/experiments/`: experiment-specific CSVs, figures, and metadata.
- `scripts/`
  - all experiment scripts, report builders, figure builders, and post-hoc analysis scripts.

## Important Artifacts

- Main live report: `reports/2026-06-23_reviewer_revised_report.pdf`
- Zenodo snapshot report: `reports/2026-06-13_reviewer_revised_report.pdf`
- Latest release notes for DOI archiving: `reports/release_notes_v2026.06.24.md`
- Research roadmap: `paper/research_program.md`
- Track 1 draft: `paper/articles/geometric-transformation-vectors/draft.md`
- Track 2 draft: `paper/articles/lie-style-linguistic-operators/draft.md`

## Citable Snapshot

The repository includes `CITATION.cff` for GitHub's citation widget and
`.zenodo.json` for Zenodo/GitHub release archiving. The latest intended
archival snapshot is `v2026.06.24`.

Latest Zenodo version DOI: [10.5281/zenodo.20829303](https://doi.org/10.5281/zenodo.20829303)
Latest Zenodo record: https://zenodo.org/records/20829303
Previous Zenodo version DOI: [10.5281/zenodo.20680414](https://doi.org/10.5281/zenodo.20680414)

## Reproducibility Notes

The scripts require Python packages such as `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, and `matplotlib`.
The current review environment is pinned in `requirements.txt`.

The datasets in this snapshot are synthetic controlled sentence-pair templates generated by the scripts in `scripts/`; no external natural-language corpus is required for the archived experiments.

Run scripts from the repository root, for example:

```powershell
.\.venv\Scripts\python.exe scripts\build_research_report.py
```

The reproducibility path is: run or inspect scripts in `scripts/`, compare generated outputs against CSVs and figures in `results/` and `results/experiments/`, then rebuild the reviewer packet with `scripts/build_research_report.py`.

Large intermediate vector caches (`*.npy`), local virtual environments, IDE files, `.env`, and review zip archives are intentionally excluded from git.

## License

MIT. See `LICENSE`.
