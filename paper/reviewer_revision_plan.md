# Reviewer-Driven Revision Plan

This file tracks the concrete changes required after the external review.

## Track 1: Geometric Transformation Vectors

### Already addressed in the draft

- Reframed `syntax=1.0` as a red flag for surface-form leakage, not a headline achievement.
- Added delta/y_only/concat/x_only multiseed ablation table.
- Added McNemar evidence from existing results.
- Highlighted `concat < delta`, especially for decoder models.
- Added pooling and modern-model limitations.
- Ran syntax representation ablation: `y_only=1.0` confirms syntax endpoint leakage.
- Ran a BERT syntax layerwise/pooling sanity check: layer `0` already reaches `1.0`, so syntax is not a last-layer geometry result.
- Ran a DeBERTa-v3-small spot-check: Linear SVC supports `delta > concat > y_only`, while logistic regression remains mixed.
- Ran larger/modern spot-checks on `bert-large-uncased` and `microsoft/deberta-v3-base`; `delta` is best for both Linear SVC and logistic regression on both models.
- Added UPAT hard-holdout as an explicit boundary result instead of leaving it hidden in CSV files.
- Ran full-semantic pooling ablation for BERT, RoBERTa, and GPT-2.
- Added confusion-matrix analysis with negation-vs-non-negation recall.
- Reframed layer-0 syntax accuracy as a surface-cue failure, not as evidence of semantic geometry.
- Marked UPAT Procrustes alignment and 100-permutation shuffle p-values as exploratory only.

### Must still be run

1. Resolve UPAT by expansion or matched-capacity comparison against the main dataset.
2. Add UPAT Procrustes null baselines:
   - random-label or random-pairing alignment
   - report null distribution for F1 gain
3. Increase UPAT shuffle controls from 100 to at least 1000 permutations before reporting precise p-values.
4. `x_only/y_only/concat/delta` for every non-syntax holdout split that is still missing this breakdown.
5. Convert large/modern spot-checks into multiseed runs if Track 1 is promoted to submission.
6. Proper related-work table and bibliography in final citation format.

## Track 2: Signed Permutation Coherence

### Already addressed in the draft

- Renamed the conceptual result away from "Jacobi-like" to "third-order signed permutation coherence".
- Explicitly distinguishes the diagnostic from the formal Lie-algebra Jacobi identity.
- Added Welch/Mann-Whitney p-values and Cohen's d for semantic equivalence control.
- Added dataset audit showing zero duplicate endpoint rows.
- Reframed negation failures as a substantive result.
- Added GPT-2/DistilGPT-2 signed-permutation replication. `QMT` remains below null for both decoders, while negation-containing triples are model-dependent.
- Added GPT-2/DistilGPT-2 pairwise composition summaries.
- Added multiple-testing correction over the 20 model/triple signed-permutation tests.
- Added a working hypothesis for why `QMT` is cross-architecture coherent.
- Rewrote antisymmetry as a tautological implementation check rather than evidence.
- Added explicit caveat that current Lie-style templates contain stable lexical markers and remain synthetic.

### Must still be run

1. Rename code and CSV columns away from `jacobi_*`.
2. Add grammar-generated templates.
3. Add endpoint-only controls.
4. Add layerwise and pooling ablations for composition diagnostics.
5. Add focused negation analysis before expanding the operator set.
6. Convert the QMT working hypothesis into a grammar-generated preregistered test.
7. Add null baselines for pairwise commutator norms `||[A,B]||` using matched random/shuffled operations.

## Global blockers

- Related work positioning is incomplete.
- Related work positioning is started, but citation formatting and comparison table remain incomplete.
- Current model set now has draft-level larger/modern spot-checks, but not multiseed larger-model confirmation.
- Mean pooling needs empirical justification for the full-semantic and composition experiments.
