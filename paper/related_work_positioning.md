# Related Work Positioning: RISE and Our Tracks

This note records how the project should position itself after reading:

Freenor and Alvarez, **"Mapping Semantic & Syntactic Relationships with Geometric Rotation"**, ICLR 2026.

## What RISE Already Covers

RISE is the closest and strongest neighboring work. It treats discourse-level semantic-syntactic transformations as geometric operations in sentence-embedding space.

Key points from the paper:

- Method: Rotor-Invariant Shift Estimation (RISE), a spherical/Riemannian rotor method that maps neutral-to-transformed sentence pairs through tangent-space prototypes.
- Transformations: negation, conditionality, and politeness.
- Scope: seven languages and three multilingual embedding models (`text-embedding-3-large`, `bge-m3`, `mBERT`).
- Baselines: mean difference vectors and Procrustes alignment.
- Evaluation: cosine similarity between predicted transformed embeddings and target transformed embeddings on held-out sets, plus cross-language and cross-model transfer.
- Main implication for us: we cannot claim novelty for the broad statement that discourse-level semantic-syntactic transformations have cross-lingual/cross-model geometric structure.

## What Our Work Should Not Claim

We should not write any of the following as novelty claims:

- "First evidence that linguistic transformations are geometric operations in sentence embeddings."
- "First cross-model transformation geometry result."
- "Universal transformation subspaces across embedding models" without qualification.
- "Procrustes alignment establishes universality" without comparing to RISE-like spherical methods.

## Remaining Distinct Contributions

Our current repository still has several distinct contributions if framed narrowly.

### Track 1: Endpoint-Controlled Delta Diagnostics

RISE focuses on predicting transformed embeddings via rotor/MDV-style operations. Track 1 asks a different diagnostic question:

> Does `embedding(y) - embedding(x)` add class-discriminative transformation information beyond `x_only`, `y_only`, and `concat` baselines?

Distinct value:

- explicit endpoint leakage controls
- multiseed delta/y_only/concat/x_only ablations
- McNemar tests
- hard negative/boundary reporting via UPAT
- syntax `1.0` reinterpreted as endpoint/surface leakage rather than a success

This makes Track 1 a conservative control paper, not a stronger version of RISE.

### Track 2: Ordered Composition Diagnostics

RISE reports that its transformations behave commutatively in the tangent-space framing. Track 2 asks almost the opposite question:

> Do ordered linguistic transformations show noncommutative or signed-permutation composition structure in embedding endpoints?

Distinct value:

- pairwise order diagnostics (`AB` versus `BA`)
- semantic-equivalence controls
- third-order signed-permutation diagnostic
- negation as a composition instability rather than simply a robust one-step transformation

This is the cleanest conceptual separation from RISE. It should be framed as a diagnostic of local composition/order effects, not as a steering method.

### Track 3: Cross-Model Transfer Stress Tests

Track 3 is closest to RISE and therefore needs the most careful positioning.

Current distinct value:

- direct classifier transfer over transformation deltas
- `N=1000` random-label, random-pairing, and random-orthogonal Procrustes nulls
- held-out alignment-size curve using auxiliary anchor texts disjoint from classifier train/test texts
- explicit reporting of raw, full-anchor, and held-out-anchor transfer gaps

Now addressed as a first-pass comparison:

Track 3 now includes a first-pass RISE-inspired comparison on UPAT:

- MDV/prototype prediction baseline on UPAT
- spherical normalization / tangent-space variant if feasible
- comparison of classifier-transfer F1 versus target-embedding prediction similarity

Current interpretation:

- Prototype methods answer a target-embedding prediction question.
- Procrustes delta-classifier transfer answers a class-discriminative transfer question.
- Within-model prototype target cosine is high.
- Cross-model prototype prediction is harder and model-pair dependent.
- The simplified `rise_style` baseline improves nearest-target label retrieval over MDV in the current UPAT setup, but it is not a full reproduction of RISE.
- A non-leaky hybrid test that combines aligned `delta_only` features with all-class MDV/RISE-style prototype scores does not improve cross-model transformation-label F1. The current interpretation is that target-reconstructive geometry and class-discriminative geometry are not automatically complementary under this classifier-transfer metric.
- A movement-level spherical delta steering test gives a more nuanced split: linear centroid steering is best for cross-model target cosine, while RISE-style prediction is best for nearest-target label F1. Spherical/tangent steering does not improve target cosine in the current uncalibrated form, but it slightly improves label F1 over linear deltas.

Track 3 remains a useful stress-test package, but it still should not be promoted as a main novelty paper against RISE until confidence intervals, anchor-domain robustness, and a more faithful RISE implementation are added.

## Recommended Framing

Use this framing in future drafts:

> RISE shows that some discourse-level semantic-syntactic transformations can be modeled as spherical/geodesic operations across languages and embedding models. Our work is complementary: we stress-test simpler endpoint and delta representations under endpoint-only controls, hard holdouts, Procrustes nulls, and held-out alignment anchors, and we separately test whether ordered linguistic transformations exhibit noncommutative composition diagnostics.

## Immediate Backlog Changes

1. Add RISE to every related-work section as the closest prior work.
2. Stop using "universal transformation subspaces" as an unqualified Track 3 title.
3. Rename Track 3 working title toward "stress-testing cross-model transformation transfer".
4. Extend the first RISE/MDV comparison with confidence intervals and, if feasible, a more faithful RISE implementation before promoting Track 3.
5. Extend movement-level composition with train-only step-size calibration and confidence intervals before making any complementarity claim.
6. Emphasize Track 2 composition/order diagnostics as the most distinct paper-level contribution.
