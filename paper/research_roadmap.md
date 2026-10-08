# Research Roadmap

This file tracks completed research-driven changes and future work needed before any submission-grade paper. These items are not blockers for the Zenodo software/research-artifact snapshot.

Current public numbering follows publication priority, not chronology:

1. **Track 1 / GLT-STEER**: activation-space final-marker steering; current primary short-paper target.
2. **Next line / GLT-BUILD**: controlled training-time emergence of transformation structure in tiny synthetic-language transformers.
3. **Track 2 / GLT-SPOT + GLT-MOLT**: Lie-adjacent signed-composition and learned-operator diagnostics.
4. **Track 3 / GLT-DV**: endpoint-controlled delta-vector diagnostics.
5. **Track 4 / GLT-XFER**: cross-model transformation-transfer stress tests.
6. **Track 5 / GLT-AFFECT**: graded affective geometry.
7. **Track 6 / GLT-DIM**: effective dimensionality of transformation subspaces.
8. **Track 7 / GLT-XLING**: cross-lingual transformation geometry.

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

## Next Main Line / GLT-BUILD: Training-Time Emergence

### Why this is the next research line

The earlier GLT tracks inspect frozen models. That is useful, but it also means the observed structures were formed by unknown data, unknown optimization history, and many interacting learned features.

GLT-BUILD turns the problem into a controlled laboratory setup:

> Train a tiny decoder-only Transformer from scratch on a fully specified synthetic language, then observe when transformation geometry appears during training.

### Core question

How do linguistic transformations form inside a neural network during training, and what is the minimal mathematical structure needed to describe them?

### Initial experiment

First planned experiment:

- `paper/articles/glt-build-emergence/experiment_spec.md`

Execution update, 2026-09-15: the first pilot finished; its raw operator-fit cosine
was near-perfect before training, and sampled unseen combinations failed. The
next bounded step is [the corrected composition audit](articles/glt-build-emergence/composition_audit_protocol.md),
with exhaustive scene coverage, saved weights/readouts, and actual sequential
generation. Keep the existing training setup and three seeds; interpret this audit
before introducing a new curriculum or model family.

This audit is now complete: known commands transfer, general composition fails,
and sequential SQ succeeds only in the order that preserves the familiar input
form before Q. Oracle intermediates expose a second-step support boundary, and
initialization/embedding controls prevent a learned-geometry claim from high fit
alone. The next design decision is a controlled training-support comparison;
no further run is automatically queued.

A subsequent user-authorized run implemented
[the paired input-support comparison](articles/glt-build-emergence/input_support_protocol.md):
two curricula, three paired seeds, one fixed budget and a role-swap-closed split.
Its completed result is `input_support_rescue_under_fixed_protocol`: expanded
source-state teaching restores sequential execution, while base-only does not.

A second bounded audit implemented
[the explicit pair-prefix holdout](articles/glt-build-emergence/pair_prefix_holdout_protocol.md).
It found 1.0 sequential execution in both arms but 0.0 direct generalization to
unseen pair prefixes in the single-operation-only arm. This separates text-mediated
primitive reuse from explicit command-prefix composition. GLT-BUILD remains a
controlled discrete-composition line, not evidence for a Lie algebra.

Initial target:

- synthetic language with known latent states;
- explicit operation tokens first, implicit emergence later;
- transformations such as tense, negation, voice, question mood, and role swap;
- held-out lexemes, templates, and compositions;
- multiple checkpoints and random seeds;
- additive, linear, affine, low-rank/contextual, and nonlinear representation models;
- composition diagnostics and causal interventions.

### Boundary

GLT-BUILD should not start by claiming a Lie algebra. The first promoted objects are discrete operations and their group/monoid/action or matrix-representation structure. Lie-algebra tests belong only after continuous or quasi-continuous features are introduced and shown to admit meaningful parameterization.

### First stopping rule

GLT-BUILD-01 is complete when it can answer:

1. whether the model learned held-out operation combinations;
2. when and where each operation became recoverable in hidden states;
3. which minimal representation model describes each operation;
4. whether composition laws hold better than matched nulls;
5. whether discovered objects causally affect behavior.

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

This avoids a research-debt spiral where every new control creates a new uncontrolled side question before the strongest current paper is written. GLT-BUILD is the exception only because it is a new controlled-emergence program with its own stopping rule, not another ad hoc control added to GLT-STEER.
