# Critical Review: Current Research State

Date: 2026-06-14

This note is a deliberately skeptical review of the repository after the RISE-aware UPAT work, hybrid transfer test, and spherical delta steering experiment.

## Executive Verdict

The repository is now a credible research artifact and a useful exploratory program. It is not yet a submission-ready single paper.

The strongest scientific value is not one grand claim. It is the combination of controlled positive results and openly reported boundary conditions:

- delta vectors can add information beyond endpoint-only baselines
- syntax-perfect results are endpoint/surface leakage, not a triumph
- QMT-like composition shows local signed-permutation coherence
- negation breaks or destabilizes the composition story
- cross-model Procrustes transfer survives strong nulls
- RISE-style target reconstruction and delta-classifier transfer answer different questions
- simple hybridization does not automatically improve the main metrics

This is not nonsense. The work contains several real, falsifiable empirical facts. The main danger is over-narrating them as a unified theory before the controls are complete.

## What Is Actually Done

### Track 1: Endpoint-Controlled Delta Representations

Status: most mature draft.

Main result:

- In the main full-semantic multiseed setting, `delta` beats `y_only` and `concat` across the original tested model set.
- McNemar tests support the result.
- BERT-large and DeBERTa-v3-base spot-checks support the delta advantage.
- Pooling/layerwise checks reduce the risk that the mean-pooling choice alone creates the effect.

Critical caveats:

- `y_only` is still strong.
- syntax `1.0` is confirmed as target/surface leakage.
- UPAT hard-holdout does not show a reliable `delta > y_only` advantage.
- The main multiseed result still needs confidence intervals and broader holdout ablation tables.

Best paper framing:

> Delta vectors add reproducible relational information beyond endpoint-only baselines in controlled synthetic transformation probes, but they are not pure linguistic operators.

### Track 2: Signed-Permutation / Composition Diagnostics

Status: interesting and distinct, but needs stronger data generation.

Main result:

- Pairwise composition is order-sensitive.
- Semantic-equivalence controls show that the diagnostic is not just arbitrary wording noise.
- The third-order result is correctly reframed as signed-permutation coherence, not a Jacobi identity.
- `QMT` is the strongest local result across encoder and decoder checks.
- Grammar-generated pairwise composition controls now preserve below-null commutator coherence, but endpoint-only controls remain near perfect.

Critical caveats:

- Hand-written templates may contain stable lexical markers.
- Grammar-generated endpoints still contain explicit order/pair markers.
- Negation-containing triples are unstable.
- Endpoint-only controls for composition are still needed.
- Pairwise commutator norms need null baselines before being promoted.

Best paper framing:

> Some controlled ordered transformations exhibit local composition-sensitive geometry; the result is not a global Lie algebra.

### Track 3: RISE-Aware Cross-Model Transfer Stress Tests

Status: promising but most exposed to related-work criticism.

Main result:

- Cross-model Procrustes transfer survives `N=1000` random-label, random-pairing, and random-orthogonal nulls.
- Held-out alignment anchors recover most of the full-anchor transfer effect.
- RISE/MDV-style target prediction is strong within-model but harder cross-model.
- Non-leaky hybrid prototype-score features do not beat aligned `delta_only` for transformation-label F1.
- Spherical delta steering separates target cosine from retrieval label F1:
  - `linear_delta` is best for target cosine
  - `rise_only` is best for transformation-neighborhood retrieval
  - uncalibrated spherical delta movement slightly improves label F1 but lowers target cosine

Critical caveats:

- RISE is the closest and stronger neighboring paper for broad cross-model/cross-lingual transformation geometry.
- The current `rise_style` implementation is only a simplified baseline, not a faithful reproduction.
- Anchor-domain robustness is not yet tested.
- Confidence intervals over directions and examples are missing.
- Step-size calibration for spherical movement is not done.

Best paper framing:

> A stress-test comparison showing that cross-model class-discriminative transfer, target reconstruction, and spherical movement are separable geometric questions.

## What Is Not Ready

1. A unified "theory of linguistic operators" is not ready.
2. A claim of first discovery of geometric linguistic transformations is not valid because RISE already occupies that territory.
3. A strong natural-language generalization claim is not ready because the datasets are synthetic controlled templates.
4. A steering/generation claim is not ready because all current steering is embedding-space target prediction, not causal intervention in generation.

## Current Research Directions

### Direction A: Finish Track 1 as a Conservative Baseline Paper

This is the lowest-risk publication path.

Required:

- confidence intervals for `delta - y_only` and `delta - concat`
- remaining holdout representation-ablation tables
- clean final bibliography
- self-review against endpoint leakage objections

### Direction B: Make Track 2 a Composition Diagnostics Paper

This is the most distinct intellectual contribution.

Required:

- endpoint-balanced grammar generation for `N,Q,M,T`
- target-only controls for third-order composition endpoints
- stronger pairwise commutator nulls beyond the current random-pairing/norm-matched controls
- layer/pooling ablation

### Direction C: Develop Track 3 as a RISE-Aware Stress-Test Paper

This is exciting but must be very carefully positioned.

Required:

- bootstrap confidence intervals
- direction-family summaries
- anchor-domain robustness
- train-only step-size calibration for spherical delta steering
- more faithful RISE implementation or explicit statement that the current baseline is simplified

### Direction D: Move Toward Causal Editing

This is future work, not current evidence.

Required:

- residual-stream injection in a decoder model
- random-vector and wrong-class controls
- generation-level metrics
- careful separation from embedding retrieval experiments

## Repository Hygiene Review

Strengths:

- Scripts and results are now mostly discoverable through `scripts/README.md` and `results/README.md`.
- Main research state is documented in `paper/research_program.md`, `paper/reviewer_revision_plan.md`, and `research/diary.md`.
- The Zenodo snapshot is ethically framed as software/research artifact, not a peer-reviewed paper.
- The repo includes code, CSVs, figures, requirements, citation metadata, and reports.

Weaknesses:

- The repository is large and still contains many historical exploratory result folders.
- Some old scripts are not clearly labeled as historical versus current.
- The main report date lagged behind the live research date; future reports should use the actual current date.
- Track 3 does not yet have its own article draft, even though it now has enough material for a structured outline.

Immediate cleanup recommendations:

- Add a Track 3 article folder only after calibration/CI, or add a placeholder outline clearly marked "not draft-ready."
- Add a `CURRENT_RESULTS.md` or keep this critical review as the compact state summary.
- Keep old exploratory folders, but avoid citing them from current papers unless they are part of the active evidence chain.
- Treat `reports/2026-06-13_reviewer_revised_report.pdf` as historical and build new live packets under the actual date.

## Bottom Line

This is a real research program, not AI-flavored noise. The results are interesting because they are not uniformly positive. They expose a structured split:

- delta vectors help classification beyond endpoints in some controlled settings
- endpoint leakage is real and sometimes dominant
- composition structure exists locally but fails globally
- cross-model alignment is surprisingly robust under nulls
- RISE-like geometry and delta geometry optimize different notions of success

The next best move is not adding more broad claims. It is adding calibration and confidence intervals to Track 3, while keeping Track 1 and Track 2 clean enough to become separate papers.
