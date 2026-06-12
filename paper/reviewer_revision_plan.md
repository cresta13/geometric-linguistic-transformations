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

### Must still be run

1. `x_only/y_only/concat/delta` for every non-syntax holdout split that is still missing this breakdown.
2. Full-semantic pooling ablation:
   - BERT/RoBERTa `[CLS]` vs mean pooling
   - GPT-2/DistilGPT-2 last-token vs mean pooling
3. Convert large/modern spot-checks into multiseed runs if Track 1 is promoted to submission.
4. Error analysis over confusion matrices.
5. Proper related-work table and bibliography.

## Track 2: Signed Permutation Coherence

### Already addressed in the draft

- Renamed the conceptual result away from "Jacobi-like" to "third-order signed permutation coherence".
- Explicitly distinguishes the diagnostic from the formal Lie-algebra Jacobi identity.
- Added Welch/Mann-Whitney p-values and Cohen's d for semantic equivalence control.
- Added dataset audit showing zero duplicate endpoint rows.
- Reframed negation failures as a substantive result.
- Added GPT-2/DistilGPT-2 signed-permutation replication. `QMT` remains below null for both decoders, while negation-containing triples are model-dependent.

### Must still be run

1. Rename code and CSV columns away from `jacobi_*`.
2. Add grammar-generated templates.
3. Add endpoint-only controls.
4. Add layerwise and pooling ablations for composition diagnostics.
5. Add focused negation analysis before expanding the operator set.
6. Re-run composition summaries for decoder models, not only third-order signed permutation summaries.

## Global blockers

- Related work positioning is incomplete.
- Current model set now has draft-level larger/modern spot-checks, but not multiseed larger-model confirmation.
- Mean pooling needs empirical justification for the full-semantic and composition experiments.
