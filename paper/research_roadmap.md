# Research Roadmap

This file tracks completed research-driven changes and future work needed before any submission-grade paper. These items are not blockers for the Zenodo software/research-artifact snapshot.

Current public numbering follows publication priority, not chronology:

1. **Track 1 / GLT-STEER**: activation-space final-marker steering; current primary short-paper target.
2. **Track 2 / GLT-SPOT + GLT-MOLT**: Lie-adjacent signed-composition and learned-operator diagnostics.
3. **Track 3 / GLT-DV**: endpoint-controlled delta-vector diagnostics.
4. **Track 4 / GLT-XFER**: cross-model transformation-transfer stress tests.
5. **Track 5 / GLT-AFFECT**: graded affective geometry.
6. **Track 6 / GLT-DIM**: effective dimensionality of transformation subspaces.
7. **Track 7 / GLT-XLING**: cross-lingual transformation geometry.

Historical notes before 2026-08 used a different numbering scheme: old Track 1 = GLT-DV, old Track 2 = GLT-SPOT, old Track 3 = GLT-XFER, old Track 4 = GLT-STEER. Current active docs should use the numbering above.

## Track 1 / GLT-STEER: Activation-Space Final-Marker Steering

### Why this is the current primary paper

GLT-STEER is the clearest current short-paper candidate because it has a behavior-level intervention, explicit no-steering/wrong-vector controls, logit-level evidence, intervention-position controls, and clear negative boundaries.

The central claim is intentionally narrow:

> Final-position surface markers (`?`, `!`, `...`) are steerable in GPT-2-style residual streams via mean transformation-delta injection, while the same recipe does not yet support robust lexical or sentence-internal rewriting.

### Already addressed

- Focused GPT-2 question steering: target question-mark rate around `0.935`, controls at `0.0000`.
- Copy-prompt preservation audit: question-and-preserved rate up to `0.9750`, matched no-steering/wrong-vector controls at `0.0000`.
- Copy-prompt no-steering baseline: GPT-2 and DistilGPT-2 produce `0.0000` question marks across `960` no-steering rows.
- Hard out-of-template GPT-2 question audit: marker effect survives on structurally diverse sources.
- DistilGPT-2 replication: marker-form effect survives, but content preservation is weaker and layer/gain sensitive.
- Negation layer sweep: negative/boundary result under the current recipe.
- Exclamation and ellipsis controls: support the Final Marker Hypothesis beyond question marks.
- Final-marker logit audit: no-steering marker rates are `0.0000`, while target steering moves intended marker tokens into top ranks.
- Position-of-intervention audit: single prompt-position edits do not work; distributed or repeated interventions do.
- Marker-composition audit: combined final-marker vectors show competition/saturation, not clean algebraic order structure.
- Question/modality composition audit: negative for the current modality recipe.
- Wilson CI audit: headline Track 1 rows now include `N` and 95% confidence intervals.
- DistilGPT-2 layer/gain tuning disclosure is now recorded in the diary and Track 1 draft.
- Fixed-parameter confirmatory audit: question, exclamation, and ellipsis steering remain separated from controls on fresh hard-heldout sources without any layer/gain search inside the run.
- Runtime form-control applicability audit: steering beats prompt-only and matched vector controls for final-marker induction under the tested protocol, but deterministic `string_append_source` is perfect. This bounds GLT-STEER as a diagnostic/intervention result rather than a replacement for ordinary text postprocessing.

### Stopping rule for submission

Track 1 is short-paper-ready when:

1. The Final Marker Hypothesis is the central claim.
2. Activation/representation-steering related work is incorporated.
3. Headline tables include `N` and 95% CI.
4. DistilGPT-2 layer/gain tuning history is disclosed.
5. Marker composition is framed as competition/saturation, not as Lie-algebra evidence.

Current status: these items are complete for the current short-paper scope, and the fixed-parameter confirmatory audit plus runtime applicability audit have also been archived. The next work is writing, figure/table selection, and package cleanup. New controls go to future work unless requested by an external reviewer or a concrete venue requirement.

## Track 2 / GLT-SPOT + GLT-MOLT: Lie-Adjacent Diagnostics

### Already addressed

- Renamed the third-order endpoint diagnostic away from "Jacobi-like" to "third-order signed permutation coherence".
- Added semantic-equivalence controls, dataset duplicate-endpoint audit, decoder replication, and multiple-testing correction.
- Added multilingual signed-permutation audits across 7 languages and multilingual encoders.
- Added endpoint-subspace residualization: the signed-permutation signal survives removal of simple linear endpoint-derived probe subspaces.
- Added GLT-MOLT learned linear/affine operator audits.
- Added ridge sweep, matched operator nulls, spectral nulls, and compact PCA-64/PCA-128 sensitivity checks.
- Central MOLT split is now explicit: additive deltas predict endpoints better, while learned operators expose weak closure-like compression under nulls.

### Future submission work

1. Rename code/CSV columns away from `jacobi_*`.
2. Add endpoint-balanced multilingual grammar templates.
3. Add target-only and endpoint-only controls for the six third-order composition endpoints.
4. Explain the `NQM` versus `QMT` regime shift before expanding operators.
5. Add optional PCA-256 and layerwise sensitivity only if this track becomes a submission target.

## Track 3 / GLT-DV: Endpoint-Controlled Delta-Vector Diagnostics

### Already addressed

- Reframed `syntax=1.0` as target/surface leakage, not a headline result.
- Added delta/y_only/concat/x_only multiseed ablation table.
- Added McNemar evidence and seed-level 95% effect intervals.
- Narrowed the claim to the robust Linear SVC result because logistic regression is mixed.
- Added syntax representation ablation and layer-0 syntax sanity check.
- Added DeBERTa-v3-small, BERT-large, and DeBERTa-v3-base spot-checks.
- Added UPAT as a bounded hard-holdout result.
- Added full-semantic pooling ablation and confusion/negation analysis.

### Future submission work

1. Keep UPAT as a bounded hard-holdout unless expansion or matched-capacity comparison changes the conclusion.
2. Add remaining non-syntax holdout representation ablations if GLT-DV becomes a submission target.
3. Convert large/modern spot-checks into multiseed runs only if this track is promoted.
4. Tighten final bibliography and comparison table.

## Track 4 / GLT-XFER: Cross-Model Transfer Stress Tests

### Already addressed

- Scaled UPAT-large Procrustes nulls to `N=1000`.
- Added random-label, random-pairing, and random-orthogonal controls.
- Added held-out alignment-size curve with auxiliary anchor texts disjoint from classifier train/test endpoints.
- Added first-pass RISE/MDV-style prototype comparison.
- Added non-leaky hybrid RISE-Procrustes feature transfer.
- Added movement-level spherical delta steering comparison.

### Future submission work

1. Add confidence intervals and direction-family summaries for the held-out alignment curve.
2. Stress-test anchor-domain diversity.
3. Compare against a more faithful RISE implementation if feasible.
4. Keep GLT-XFER framed as stress testing, not as first-discovery cross-model geometry.

## Track 5 / GLT-AFFECT: Graded Affective Geometry

### Already addressed

- Added text-only affective polarity MVP over `hate -> dislike -> indifferent -> like -> love`.
- Added marker-pooling control.
- Added lexical-specificity control against size, attention, and random-label ladders.
- Added paired bootstrap contrasts showing a small but stable affect-specific excess over generic lexical replacement.

### Future submission work

1. Add length/frequency-matched neutral-word ladder.
2. Keep affect claims text-representation-only until non-textual grounding exists.
3. Treat psychophysical grounding as a separate future subtrack.

## Track 6 / GLT-DIM: Effective Dimensionality

Future measurements:

1. PCA spectrum per transformation class.
2. Participation ratio of delta singular values.
3. Accuracy versus retained PCA dimensions.
4. Sample-complexity curves per transformation class.

## Track 7 / GLT-XLING: Cross-Lingual Transformation Geometry

Future measurements:

1. Matched transformation pairs in English and non-English languages.
2. Within-language delta separability.
3. Alignment of transformation spaces across languages.
4. Classifier or centroid transfer across languages.

## Frozen Scope for This Cycle

The current cycle should converge on Track 1 / GLT-STEER. Tracks 2-7 remain valuable research records and future-paper candidates, but they should not receive new ad hoc controls until one of the following happens:

- a natural-corpus or natural-language validation experiment becomes available;
- an external reviewer asks for a specific additional control;
- a track is explicitly promoted to the next submission target.

This avoids a research-debt spiral where every new control creates a new uncontrolled side question before the strongest current paper is written.
