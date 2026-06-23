# Geometric Transformation Vectors in Transformer Embedding Spaces

## Working thesis

Sentence-pair displacement vectors can add transformation information beyond endpoint-only representations in controlled transformer embedding probes, especially under Linear SVC. They should not be treated as pure linguistic operators or as classifier-invariant evidence.

## Evidence

- Multiseed full-semantic ablations show `delta > y_only` across the original model set under Linear SVC.
- McNemar tests and 95% seed-level effect intervals support the Linear SVC delta advantage in the main full-semantic setting.
- Logistic regression is mixed, so the current claim is probe-dependent rather than classifier-invariant.
- Syntax `1.0` is now explicitly reinterpreted as target/surface leakage because `y_only` also reaches `1.0`.
- Pooling/layerwise checks, BERT-large, and DeBERTa-v3-base spot-checks support the controlled baseline claim.
- UPAT is reported as a hard-holdout boundary where `delta > y_only` is not guaranteed.

## Risks

- Target endpoints remain strong, so endpoint leakage is not eliminated.
- The main positive result is strongest for Linear SVC; logistic regression does not support `delta > y_only` for every model.
- The main multiseed table still uses mean pooling.
- UPAT boundary results block any universal operator claim.
- Cross-model/Rise-aware steering results belong in Track 3, not as the headline of this Track 1 paper.

## Next experiments

- Decide whether the paper should center Linear SVC as the primary probe or add more classifiers before submission.
- Run large/modern spot-checks as multiseed if this becomes the submission priority.
- Add representation ablation tables for remaining holdouts.
- Tighten bibliography and final related-work formatting.
