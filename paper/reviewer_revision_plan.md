# Research Roadmap

This file tracks completed reviewer-driven changes and future work needed before any submission-grade paper. These items are not blockers for the Zenodo software/research-artifact snapshot.

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

### Future submission work

1. Resolve UPAT by expansion or matched-capacity comparison against the main dataset.
2. Add UPAT Procrustes null baselines:
   - random-label or random-pairing alignment
   - report null distribution for F1 gain
3. Increase UPAT shuffle controls from 100 to at least 1000 permutations before reporting precise p-values.
4. `x_only/y_only/concat/delta` for every non-syntax holdout split that is still missing this breakdown.
5. Convert large/modern spot-checks into multiseed runs if Track 1 is promoted to submission.
6. Proper related-work table and bibliography in final citation format.

## Track 3: Cross-Model Transformation Transfer

### Why this may become the main paper

The UPAT-large Procrustes results suggest that transformation geometry may transfer across architectures after low-dimensional alignment. However, RISE (Freenor and Alvarez, ICLR 2026) is now the closest and strongest neighboring paper for cross-lingual/cross-model semantic-syntactic geometry. Track 3 can only become a main paper if it is framed as a stress-test and comparison study rather than as the first broad geometry result.

### Newly addressed

- Added a pilot Procrustes null audit with `N=30` repeats for every non-identity cross-model direction.
- Random-pairing null: target anchor correspondences are shuffled before Procrustes fitting.
- Random-label null: correct alignment is retained, but source training labels are shuffled.
- Observed aligned F1 remains above both null baselines in every tested non-identity direction. This supports the cross-model transfer hypothesis, but the run is still a pilot because `N=30` limits empirical p-values to `0.0323`.
- Scaled the Procrustes null audit to `N=1000` repeats for every non-identity cross-model direction.
- Added a random-orthogonal control: matched centering and dimensionality are preserved, but the learned Procrustes map is replaced by a random orthogonal matrix.
- Observed aligned F1 remains above random-label, random-pairing, and random-orthogonal null baselines in every direction. No null repeat reaches observed aligned F1, so all empirical p-values are at the `N=1000` resolution floor: `0.000999`.
- Added a held-out alignment-size curve using `1200` auxiliary anchor texts disjoint from the classifier train/test texts.
- Held-out anchor mean F1 increases from `0.452046` at `25` anchors to `0.661928` at `1000` anchors, compared with raw cross-model mean F1 `0.241524` and full-anchor Procrustes mean F1 `0.684651`.
- At `1000` held-out anchors, every direction remains above its raw cross-model baseline.
- Added a first-pass RISE-aware comparison on UPAT:
  - `mdv_raw`, `mdv_unit`, and simplified spherical `rise_style` prototype baselines
  - within-model target cosine is high (`rise_style=0.923008`)
  - cross-model target prediction is harder (`rise_style` mean cosine `0.578347`, nearest-target label F1 `0.445518`)
  - aligned delta-classifier transfer remains higher on its own metric (`0.684651` mean F1)
- Added a non-leaky Hybrid RISE-Procrustes transfer test:
  - every pair is scored against all class prototypes, so the test label is never used to choose the prototype
  - cross-model `delta_only` remains strongest (`0.684651` mean F1)
  - the best hybrid/prototype feature set is `mdv_raw_hybrid_delta_scores` (`0.430839` mean F1)
  - current conclusion: prototype target-reconstruction scores do not preserve transformation identity better than aligned deltas under the classifier-transfer metric

### Future submission work

1. Add confidence intervals and direction-family summaries for the held-out alignment curve:
   - bootstrap over directions
   - summarize encoder-to-encoder, encoder-to-sentence-encoder, and sentence-encoder-to-encoder families
   - report whether any direction-family remains weak
2. Reverse-direction transfer:
   - small to large model
   - large to small model
   - sentence encoder to masked LM and back
3. Stress-test anchor-domain diversity:
   - anchors from a different template family
   - paraphrased anchor pools
   - natural sentence anchors if available
4. Strengthen the first-pass RISE/MDV comparison:
   - add confidence intervals
   - compare against the published RISE implementation if feasible
   - state clearly that target-embedding prediction and class-discriminative transfer are different metrics
5. Add movement-level composition tests:
   - linear centroid steering versus spherical delta steering
   - delta-then-RISE versus RISE-then-delta
   - target cosine, retrieval top-1, and retrieval label F1
6. Report the completed null and held-out alignment audits in the draft:
   - null mean/std/max
   - empirical p-value resolution
   - effect sizes versus each null
   - held-out alignment-size curve
7. Optional architecture expansion:
   - Llama/Mistral-style embedding spaces if local resources allow
   - otherwise stronger sentence encoders as a lower-cost proxy

## Track 4: Transformation Vectors as Editors

### Goal

Move from descriptive geometry to causal intervention.

### Minimal experiment

1. Use GPT-2.
2. Compute centroid deltas for one transformation, probably negation or question formation.
3. Inject the centroid into the residual stream during generation.
4. Compare against random-vector, norm-matched, and wrong-class controls.
5. Score generated text with transformation classifiers and manual examples.

## Track 5: Cross-Lingual Transformation Geometry

### Goal

Test whether transformation geometry is language-invariant rather than English-template-specific.

### Minimal experiment

1. Use mBERT or XLM-R.
2. Build matched transformation pairs in English and at least one non-English language.
3. Compare within-language delta separability.
4. Align transformation spaces across languages.
5. Test classifier or centroid transfer across languages.

## Track 6: Effective Dimensionality

### Goal

Measure whether transformations live in low-dimensional subspaces.

### Minimal measurements

1. PCA spectrum per transformation class.
2. Participation ratio of delta singular values.
3. Accuracy versus retained PCA dimensions.
4. Capacity curves per transformation class.

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

### Future submission work

1. Rename code and CSV columns away from `jacobi_*`.
2. Add grammar-generated templates.
3. Add endpoint-only controls.
4. Add layerwise and pooling ablations for composition diagnostics.
5. Add focused negation analysis before expanding the operator set.
6. Convert the QMT working hypothesis into a grammar-generated preregistered test.
7. Add null baselines for pairwise commutator norms `||[A,B]||` using matched random/shuffled operations.

## Future submission priorities

- Related work positioning is now explicitly RISE-aware, but final citation formatting and a comparison table remain future submission work.
- Current model set now has draft-level larger/modern spot-checks, but not multiseed larger-model confirmation.
- Mean pooling needs empirical justification for the full-semantic and composition experiments.
- The central narrative should be selected before submission. Current candidates are:
  - Track 1: endpoint-controlled transformation vectors
  - Track 3: RISE-aware cross-model transfer stress tests
  - Track 4: transformation vectors as causal editors
  - Track 5: cross-lingual transformation geometry
