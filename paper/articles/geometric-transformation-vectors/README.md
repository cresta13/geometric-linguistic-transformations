# Geometric Transformation Vectors in Transformer Embedding Spaces

## Working thesis

Sentence-pair displacement vectors can add transformation information beyond endpoint-only representations in controlled transformer embedding probes, but they should not be treated as pure linguistic operators.

## Evidence

- Multiseed full-semantic ablations show `delta > y_only` and `delta > concat` across the original model set.
- McNemar tests support the delta advantage in the main full-semantic setting.
- Syntax `1.0` is now explicitly reinterpreted as target/surface leakage because `y_only` also reaches `1.0`.
- Pooling/layerwise checks, BERT-large, and DeBERTa-v3-base spot-checks support the controlled baseline claim.
- UPAT is reported as a hard-holdout boundary where `delta > y_only` is not guaranteed.

## Risks

- Target endpoints remain strong, so endpoint leakage is not eliminated.
- The main multiseed table still uses mean pooling.
- UPAT boundary results block any universal operator claim.
- Cross-model/Rise-aware steering results belong in Track 3, not as the headline of this Track 1 paper.

## Next experiments

- Add confidence intervals for `delta - y_only` and `delta - concat`.
- Run large/modern spot-checks as multiseed if this becomes the submission priority.
- Add representation ablation tables for remaining holdouts.
- Tighten bibliography and final related-work formatting.
