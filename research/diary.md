# Research Diary

Numbering note as of 2026-08-25:

Current public numbering is Track 1 / GLT-STEER, Track 2 / GLT-SPOT + GLT-MOLT, Track 3 / GLT-DV, Track 4 / GLT-XFER, Track 5 / GLT-AFFECT, Track 6 / GLT-DIM, and Track 7 / GLT-XLING. Older diary entries before this note use the chronology-local numbering from the time they were written and should be read historically.

## 2026-06-10: Linear transformation vectors in transformer embeddings

Starting point: the GitHub project `cresta13/geometric-linguistic-transformations` established an exploratory baseline: linguistic transformations can often be recovered from embedding displacement vectors, especially with `delta = embedding(target) - embedding(source)`.

Main result fixed from the repository state:

- Transformation classes are linearly separable across several transformer models.
- Syntax and simple semantic transformations are recoverable from deltas.
- Entity, variant, syntax, and full semantic holdouts show that some structure generalizes beyond exact prompts.
- The project does not yet prove a Lie algebra, group action, or operator calculus. It gives evidence for reusable displacement geometry.

Why this pushed us forward:

If transformations are recoverable as directions, the next natural question is whether transformation composition has structure. We therefore moved from classification of deltas to composition, commutators, semantic controls, and third-order signed permutation diagnostics.

## 2026-06-11: From separability to composition diagnostics

Today we tested whether linguistic transformations behave like noncommuting operations in embedding space.

### Composition and noncommutativity

We compared ordered compositions such as `AB` versus `BA`.

Important observation:

- Some operation pairs are clearly noncommutative in embedding space.
- The strongest effects appear around tense and modality/negation interactions.
- This is more informative than pure class separability because it tests order sensitivity.

### Semantic equivalence control

We separated pairs where two formulations should be semantically equivalent from pairs where order changes meaning.

Important observation:

- Equivalent pairs have lower noncommutativity than non-equivalent pairs.
- This suggests the commutator signal is not just template wording noise.

### Algebraic identities

We first checked antisymmetry:

- `mean_cos_comm_ab_neg_comm_ba` is essentially `1.0`.
- `mean_relative_antisym_error` is `0.0`.

Interpretation:

This is expected because `[B,A]` is explicitly computed as the negative of `[A,B]`. It is a sanity check, not evidence of Lie structure.

We then focused on the nontrivial diagnostic:

```text
J(A,B,C) = ABC + BCA + CAB - ACB - CBA - BAC
```

This was initially called a Jacobi-like test, but after review we now use the more neutral name: third-order signed permutation coherence. The diagnostic is an alternating cancellation test over embedded composition endpoints, not a formal Lie-algebra Jacobi identity.

Controls added today:

- permutation-null baseline with 1000 random sign assignments
- bootstrap 95% confidence intervals with 2000 resamples
- all four triples from `N,Q,M,T`: `NQM`, `NQT`, `NMT`, `QMT`
- invariant check: all six composition endpoints are unique per Jacobi row

Key result after fixing duplicate endpoint templates:

- `QMT` shows robust cancellation across all three models, with `jacobi_to_null_mean_ratio` around `0.62-0.68`.
- `NMT` and `NQT` are mixed or worse than null for several models.
- The result is local, not global: there is evidence for structured third-order cancellation in some operation triples, not a universal Lie algebra.

Why this moves us forward:

The next research direction should not claim a full algebra. It should characterize which linguistic operators form locally algebraic patches, and why `QMT` behaves differently from triples involving negation.

## 2026-06-12: Control-driven updates and claim narrowing

Today we converted the methodological concerns into executable checks.

### Syntax holdout ablation

Question:

Does `syntax=1.0` reflect transformation geometry, or can the target endpoint alone solve the split?

Result:

- Linear SVC reaches `1.000` with `y_only`, `concat`, and `delta` for all five tested models.
- `x_only` remains at chance (`0.167`).
- A BERT layerwise/pooling check shows the artifact already appears at layer `0` with mean pooling for both `y_only` and `delta`.

Interpretation:

The syntax holdout is now explicitly reclassified as endpoint/surface leakage. It should not be used as headline evidence for geometric transformation vectors. The stronger Track 1 result is the full-semantic multiseed ablation where `delta` improves over `y_only` across models.

### Modern-model spot-check

Question:

Does the Track 1 delta advantage survive outside the original five models?

Result:

- `bert-large-uncased` could not be downloaded because of local disk limits.
- `microsoft/deberta-v3-small` was run as a compact modern-architecture spot-check.
- Linear SVC: `delta=0.871`, `y_only=0.804`, `concat=0.823`, `x_only=0.167`.
- Logistic regression: `concat=0.828`, `delta=0.796`, `y_only=0.770`, `x_only=0.167`.

Interpretation:

The spot-check supports the main delta claim for Linear SVC but is mixed across classifiers. This is enough for a draft-level outside-model check, not enough for a final broad model-family claim.

### Decoder Track 2 replication

Question:

Does the `QMT` signed-permutation result survive in GPT-style decoder models?

Result:

- GPT-2 `QMT`: ratio to permutation null `0.539`, CI `[0.519, 0.561]`.
- DistilGPT-2 `QMT`: ratio `0.771`, CI `[0.745, 0.797]`.
- Negation-containing triples remain inconsistent:
  - GPT-2 fails on `NMT` and `NQM`.
  - DistilGPT-2 fails strongly on `NQT`.

Interpretation:

`QMT` is now stronger because it survives encoder and decoder spot-checks. Negation becomes a core scientific result rather than a minor limitation: it marks where the local algebraic diagnostic breaks, and it breaks differently across architectures.

### Document state fixed today

- Track 1 draft now includes syntax ablation, layerwise syntax sanity check, McNemar/multiseed table, and DeBERTa-v3-small spot-check.
- Track 2 draft now includes decoder replication and treats negation as a result.
- `research_program.md` now orders the work correctly: finish Track 1 integration, handle syntax leakage, analyze negation, then expand grammar/templates/operators.
- The dated PDF packet was regenerated for external verification.

## 2026-06-13: Cache cleanup and stronger Track 1 model spot-checks

Question:

Can the Track 1 delta advantage survive on stronger models after freeing enough local disk space?

Cache cleanup:

- Removed old Hugging Face model caches for the original small models and sentence-transformer experiments.
- Removed stale review zip archives from `dist`.
- Kept the committed CSV/PDF/figure evidence intact.

Models run:

- `bert-large-uncased`
- `microsoft/deberta-v3-base`

Result:

| Model | Classifier | x_only | y_only | concat | delta |
|---|---|---:|---:|---:|---:|
| BERT-large | Linear SVC | `0.167` | `0.838` | `0.851` | `0.903` |
| BERT-large | Logistic regression | `0.167` | `0.750` | `0.760` | `0.854` |
| DeBERTa-v3-base | Linear SVC | `0.167` | `0.747` | `0.776` | `0.812` |
| DeBERTa-v3-base | Logistic regression | `0.167` | `0.726` | `0.694` | `0.752` |

Interpretation:

This is the strongest Track 1 spot-check so far. Unlike the earlier DeBERTa-v3-small result, both larger/modern models show `delta` as the best representation under both tested classifiers. This does not replace a full multiseed run, but it removes the immediate objection that the effect only appears in the original small/older model set.

## 2026-06-13: Second external review response

The second review identified one fatal remaining issue for Track 1 and several serious issues for both tracks.

### UPAT hard-holdout boundary

The UPAT audit had been present in the package but not discussed in the draft. This was a real problem because it partly contradicts the central Track 1 narrative.

Result:

- BERT: `delta=0.725`, `y_only=0.825`
- RoBERTa: `delta=0.675`, `y_only=0.725`
- GPT-2: `delta=0.425`, `y_only=0.475`
- DistilRoBERTa and DistilGPT-2 favor `delta`, but by small and non-significant margins.
- No UPAT delta-vs-y McNemar test reaches `p < 0.05`.

Interpretation:

UPAT is now included as a hard negative/boundary condition. The Track 1 claim is narrowed: `delta` adds information in the main full-semantic setting and larger-model spot-checks, but the advantage is not universal under small hard-holdout regimes.

### Full-semantic pooling ablation

We ran a reduced full-semantic pooling ablation (`N_BASE=150`) for BERT, RoBERTa, and GPT-2.

Result:

- BERT mean pooling remains strongest among tested pooling choices.
- RoBERTa mean pooling remains competitive and usually strongest.
- GPT-2 last-token pooling preserves `delta > y_only`, but the effect is smaller than with mean pooling.

Interpretation:

Mean pooling is no longer an untested assumption for the main result. The decoder interpretation is now qualified as partly pooling-dependent.

### Confusion and negation analysis

The working expectation Track 1 confusion matrices to explain Track 2 negation failures. The analysis showed a subtler result:

- In full-semantic Track 1, negation is usually easy and often near perfect recall.
- The hardest class is typically `uncertainty`.
- Therefore Track 2 negation failure is not because single-step negation deltas are unclassifiable.

Interpretation:

Negation is easy as a surface-labeled one-step transformation but unstable inside ordered third-order composition. This is now the bridge between the two tracks.

### Track 2 corrections

We added decoder pairwise composition summaries for GPT-2 and DistilGPT-2, so decoders are not present only in the favorable third-order test.

We also added multiple-testing correction over `4 triples x 5 models = 20` signed-permutation tests. Several triples pass in some models, but `QMT` is the only tested triple passing below-null across all five models. The claim is now:

> QMT is the only cross-architecture stable below-null signed-permutation triple in the current test set.

Working hypothesis:

`Q`, `M`, and `T` are clause-level operators that preserve the event frame while changing illocution, epistemic status, and temporal anchoring. Negation changes truth-conditional polarity and introduces scope/surface interactions, so it is less stable under ordered composition.

## 2026-06-13: Third review response and claim hygiene

The next review correctly flagged several places where the draft could still mislead a reader.

### Antisymmetry

We rewrote antisymmetry as a tautological implementation check. Because `[A,B]` is implemented as `delta_AB - delta_BA`, perfect antisymmetry is guaranteed for arbitrary vectors. It is not evidence about transformer latent spaces.

### UPAT and endpoint confounding

The UPAT result is now treated as a central boundary condition rather than an appendix artifact. In UPAT, `y_only` can beat `delta`, especially for BERT, RoBERTa, and GPT-2. This means the Track 1 claim cannot be universal. The current claim is only that delta adds useful relational information in the main full-semantic setting and several spot-checks, while endpoint-only features remain a serious confounder.

### Layer-0 syntax result

The layer-0 result is now called a red flag. Perfect syntax performance from uncontextualized embedding-layer features means the syntax split is dominated by lexical/form cues.

### Synthetic Lie templates

The Track 2 draft now explicitly states that the current composition dataset is synthetic and hand-written. Stable lexical markers may induce or suppress signed-permutation cancellation. Grammar-generated templates remain required before submission.

### Exploratory Procrustes and shuffle controls

UPAT Procrustes alignment and 100-permutation shuffle controls are now marked exploratory. The Procrustes gains need random-label/random-pairing null baselines. The `p=0.0099` values are resolution-limited by 100 permutations, not precise p-values.

Current status:

The research record is more honest and more useful, but still not main-track submission-ready. The next actual experiments should prioritize Procrustes null baselines, 1000+ shuffles, commutator norm nulls, and grammar-generated Lie templates.

## 2026-06-13: Strategic roadmap update

The current project has too many interesting experiments to submit as one coherent story. The strongest emerging narrative is:

> Transformers may encode linguistic operators as geometric objects in a low-dimensional, partially universal transformation subspace, transferable across architectures and possibly languages.

This changes the research priority.

### Candidate central tracks

1. Cross-model transformation transfer.
   - Current Procrustes results are large enough to be exciting.
   - They are not yet trustworthy enough to headline because they need random-label and random-pairing null baselines.
2. Transformation vectors as editors.
   - If GPT-2 residual-stream injection can induce negation or question formation, the project moves from probing to causal intervention.
3. Cross-lingual transformation geometry.
   - If mBERT/XLM-R aligns English and non-English transformation deltas, this directly addresses the English-template artifact concern.
4. Effective dimensionality.
   - Participation ratio and PCA retention curves can tell us whether transformations are single directions, low-dimensional subspaces, or broad regions.

### Practical roadmap

Month 1:

- close methodology holes: Procrustes nulls, 1000+ shuffles, commutator norm nulls
- keep antisymmetry out of the evidence narrative

Month 2:

- run the GPT-2 steering-vector experiment

Month 3:

- run the cross-lingual mBERT/XLM-R transfer experiment

Month 4-5:

- write around the strongest surviving central story, likely ACL 2027 or ICLR 2027 rather than a rushed 2026 submission

## 2026-06-13: UPAT-large Procrustes null pilot

After archiving the Zenodo snapshot, we resumed the live research track with the highest-risk/highest-upside question: are the large UPAT cross-model Procrustes gains real, or can they be reproduced by bad alignment controls?

New script:

- `scripts/run_upat_procrustes_nulls.py`

New outputs:

- `results/experiments/upat_large_results/csv/procrustes_null_raw.csv`
- `results/experiments/upat_large_results/csv/procrustes_null_summary.csv`
- `results/experiments/upat_large_results/figures/10_procrustes_null_random_pairing.png`
- `results/experiments/upat_large_results/figures/10_procrustes_null_random_labels.png`

Design:

- Keep the original UPAT-large train/test split and model set.
- Recompute model embedding spaces and original cross-model transfer setup.
- For each non-identity cross-model direction, evaluate two nulls:
  - random-pairing null: shuffle target anchor correspondences before Procrustes fitting
  - random-label null: keep matched anchors but shuffle source training labels
- Use `N=30` repeats per null/direction as a pilot.

Result:

- Mean observed aligned F1 across non-identity directions: about `0.685`.
- Mean random-label null F1: about `0.170`.
- Mean random-pairing null F1: about `0.141`.
- In every tested non-identity direction, observed aligned F1 exceeded both null baselines.
- The empirical p-value is capped by the pilot size at `1/(30+1)=0.0323`, so this is encouraging but not final.

Interpretation:

This is the first real support for promoting Track 3. The large Procrustes gains do not appear to be explained by shuffled labels or mismatched anchor pairings in this pilot. The next step is to scale the nulls to at least `N=1000`, add random-orthogonal controls, and test alignment-size/reverse-direction behavior before making a paper-level universality claim.

## 2026-06-14: UPAT-large Procrustes nulls scaled to N=1000

We scaled the UPAT-large cross-model Procrustes null audit from the `N=30` pilot to `N=1000` repeats and added a random-orthogonal control.

Updated script:

- `scripts/run_upat_procrustes_nulls.py`

Updated outputs:

- `results/experiments/upat_large_results/csv/procrustes_null_raw.csv`
- `results/experiments/upat_large_results/csv/procrustes_null_summary.csv`
- `results/experiments/upat_large_results/figures/10_procrustes_null_random_pairing.png`
- `results/experiments/upat_large_results/figures/10_procrustes_null_random_labels.png`
- `results/experiments/upat_large_results/figures/10_procrustes_null_random_orthogonal.png`

Design:

- Evaluate all `30` non-identity cross-model directions among the six UPAT-large models.
- Use `1000` repeats per direction and null type.
- Test three null controls:
  - random-pairing: shuffle target anchor correspondences before Procrustes fitting
  - random-label: keep matched anchors but shuffle source training labels
  - random-orthogonal: keep matched centering/dimensionality but replace the learned Procrustes map with a random orthogonal transform

Result:

- Raw null rows: `90,000` (`30` directions x `3` nulls x `1000` repeats).
- Summary rows: `90`.
- Mean observed aligned F1 across non-identity directions: `0.684651`.
- Mean null F1:
  - random-label: `0.173017`
  - random-pairing: `0.148036`
  - random-orthogonal: `0.112103`
- Mean observed-minus-null F1 gap:
  - random-label: `0.511633`
  - random-pairing: `0.536615`
  - random-orthogonal: `0.572548`
- No null repeat reached or exceeded the observed aligned F1 in any tested direction.
- Therefore all empirical p-values are at the `N=1000` resolution floor: `1/(1000+1)=0.000999`.

Interpretation:

This substantially strengthens Track 3. The UPAT-large Procrustes transfer effect is not explained by shuffled labels, mismatched anchor correspondences, or arbitrary orthogonal rotations. The remaining question is no longer whether the original alignment result trivially survives basic nulls; it does. The next risk is methodological: Procrustes may still exploit many anchor examples, so the next required experiment is an alignment-size curve with held-out anchor/evaluation separation.

## 2026-06-14: Held-out alignment-size curve

We tested the next methodological risk for Track 3: does cross-model Procrustes transfer still work when the alignment map is fitted on independent held-out anchor sentences rather than on the classifier train/test endpoints?

New script:

- `scripts/run_upat_alignment_size_heldout.py`

New outputs:

- `results/experiments/upat_large_results/csv/heldout_alignment_anchor_texts.csv`
- `results/experiments/upat_large_results/csv/heldout_alignment_curve_raw.csv`
- `results/experiments/upat_large_results/csv/heldout_alignment_curve_summary.csv`
- `results/experiments/upat_large_results/csv/heldout_alignment_curve_by_direction.csv`
- `results/experiments/upat_large_results/figures/11_heldout_alignment_size_curve.png`
- `results/experiments/upat_large_results/figures/11_heldout_alignment_by_direction.png`

Design:

- Generate `1200` auxiliary synthetic anchor texts from UPAT templates with separate seeds.
- Filter out all texts used in the main classifier train/test dataset.
- Fit Procrustes maps on held-out anchor text pairs only.
- Evaluate all `30` non-identity cross-model directions.
- Use alignment sizes `25, 50, 100, 250, 500, 1000`.
- Use `10` repeats per direction and alignment size.

Result:

- Raw rows: `1800` (`30` directions x `6` sizes x `10` repeats).
- Mean raw cross-model F1: `0.241524`.
- Mean original full-anchor Procrustes F1: `0.684651`.
- Held-out anchor mean F1 by alignment size:
  - `25`: `0.452046`
  - `50`: `0.573243`
  - `100`: `0.629405`
  - `250`: `0.653685`
  - `500`: `0.659098`
  - `1000`: `0.661928`
- At `1000` held-out anchors, the mean gap to full-anchor Procrustes is `-0.022722`.
- At `1000` held-out anchors, the mean gain over raw cross-model transfer is `+0.420404`.
- No direction at `1000` held-out anchors falls below its raw cross-model baseline.

Interpretation:

This is an important positive result. The cross-model transfer effect is not merely an artifact of aligning on the same endpoint strings used by the classifier train/test task. Independent held-out anchor sentences recover most of the full-anchor alignment effect. The remaining Track 3 risk is now more specific: we need confidence intervals, anchor-domain diversity checks, and perhaps stronger architecture coverage, but the held-out-anchor critique itself is no longer the main blocker.

## 2026-06-14: RISE positioning correction

We reviewed Freenor and Alvarez, **"Mapping Semantic & Syntactic Relationships with Geometric Rotation"** (ICLR 2026), the RISE paper.

This is the closest and strongest neighboring work. It already studies discourse-level semantic-syntactic transformations as geometric operations in sentence embeddings, with a spherical/Riemannian rotor method, seven languages, three multilingual embedding models, cross-language transfer, cross-model transfer, MDV and Procrustes baselines, and random-prototype controls.

Immediate consequence:

We must not claim novelty for the broad statement that linguistic transformations have cross-model or cross-lingual geometric structure. RISE already occupies that space more strongly.

Updated positioning:

- Track 1 becomes an endpoint-controlled delta diagnostic paper:
  - `delta` versus `x_only`, `y_only`, and `concat`
  - multiseed ablations
  - McNemar tests
  - explicit endpoint leakage and hard-holdout failures
- Track 2 becomes the clearest distinct paper:
  - RISE emphasizes reusable one-step transformations and commutative tangent-space behavior
  - our Track 2 asks where ordered transformations are noncommutative or locally composition-sensitive
- Track 3 becomes a RISE-aware stress-test paper:
  - `N=1000` null-controlled Procrustes transfer
  - held-out anchor alignment-size controls
  - required next step: MDV/RISE-style prototype comparison on UPAT

New required experiment:

Implement an UPAT comparison between:

- direct delta classifier transfer
- mean difference vector / prototype prediction
- spherical or tangent-space prototype prediction if feasible
- Procrustes-aligned classifier transfer

The goal is not to beat RISE by assertion. The goal is to show exactly what our simpler controls diagnose, where they agree with RISE-style geometry, and where ordered composition diagnostics reveal different structure.

## 2026-06-14: UPAT RISE-aware prototype comparison

We implemented the first RISE-aware comparison on UPAT.

New script:

- `scripts/run_upat_rise_aware_comparison.py`

New outputs:

- `results/experiments/upat_large_results/csv/rise_aware_comparison_raw.csv`
- `results/experiments/upat_large_results/csv/rise_aware_comparison_summary.csv`
- `results/experiments/upat_large_results/figures/12_rise_aware_target_cosine.png`
- `results/experiments/upat_large_results/figures/12_rise_aware_retrieval_f1.png`

Design:

- Compare three class-conditioned prototype methods:
  - `mdv_raw`: mean raw difference vector per transformation class
  - `mdv_unit`: mean unit-sphere difference vector per transformation class
  - `rise_style`: spherical log/exp prototype with Householder canonicalization to a shared reference direction
- Evaluate within-model and cross-model/full-anchor settings.
- Report target-embedding prediction metrics:
  - mean target cosine
  - nearest-target top-1 retrieval
  - nearest-target class accuracy/F1
- Also report the delta-classifier transfer F1 from the aligned delta pipeline for comparison.

Result:

Within-model:

- Prototype methods predict target embeddings very well by cosine:
  - `mdv_raw`: `0.922426`
  - `mdv_unit`: `0.921752`
  - `rise_style`: `0.923008`
- `rise_style` is slightly best on nearest-target label F1: `0.477279`.

Cross-model/full-anchor:

- Target cosine:
  - `mdv_raw`: `0.576897`
  - `mdv_unit`: `0.613559`
  - `rise_style`: `0.578347`
- Nearest-target label F1:
  - `mdv_raw`: `0.411362`
  - `mdv_unit`: `0.407674`
  - `rise_style`: `0.445518`
- Delta-classifier transfer F1 in the same aligned spaces remains `0.684651` on average.

Interpretation:

The prototype methods and the delta-classifier transfer answer different questions. MDV/RISE-style prototypes directly test whether a class-conditioned operation can predict the target embedding. Delta-classifier transfer tests whether the transformation class remains discriminable across model spaces. The first RISE-aware result says:

- one-step prototype geometry is strong within each model
- cross-model target prediction is much harder and model-pair dependent
- RISE-style spherical canonicalization improves nearest-target class retrieval over MDV in this UPAT setup, but not enough to match delta-classifier transfer F1

This is a useful positioning result, not a claim that our simplified RISE-style implementation matches the full RISE method.

## 2026-06-14: Hybrid RISE-Procrustes transformation transfer

We tested whether target-reconstructive prototype geometry can improve cross-model transformation identity transfer.

New script:

- `scripts/run_upat_hybrid_rise_procrustes.py`

New outputs:

- `results/experiments/upat_large_results/csv/hybrid_rise_procrustes_raw.csv`
- `results/experiments/upat_large_results/csv/hybrid_rise_procrustes_summary.csv`
- `results/experiments/upat_large_results/figures/13_hybrid_rise_procrustes_f1.png`
- `results/experiments/upat_large_results/figures/13_hybrid_rise_procrustes_heatmap.png`

Design:

- Keep the same UPAT train/test split and cross-model full-anchor Procrustes alignment.
- Learn class prototypes from the source model's train split.
- For every train/test pair, score the pair against every possible class prototype.
- Do not use the pair's true label to choose a prototype.
- Train transformation-label classifiers on:
  - `delta_only`
  - all-class prototype score features
  - `delta + prototype score` hybrid features

Result:

Cross-model/full-anchor mean macro F1:

- `delta_only`: `0.684651`
- best hybrid/prototype feature set, `mdv_raw_hybrid_delta_scores`: `0.430839`
- `mdv_raw_prototype_scores`: `0.413582`
- `rise_style_prototype_scores`: `0.375563`
- `mdv_unit_prototype_scores`: `0.369480`

Interpretation:

This is a useful negative result. Target-reconstructive MDV/RISE-style prototype scores do not automatically preserve class-discriminative transformation identity better than aligned deltas. The result supports a cleaner separation:

- RISE/MDV-style geometry asks "where should the target embedding land?"
- Procrustes/delta classifiers ask "is the transformation identity still discriminable across model spaces?"

The next composition should therefore be tested at the movement level, not just by concatenating classifier features:

- linear centroid steering: `normalize(x + class_centroid_delta)`
- spherical delta steering: project the centroid delta into the tangent space at `x` and move with an exp map
- RISE-then-delta and delta-then-RISE orderings
- metrics: target cosine, retrieval top-1, and retrieval label F1

## 2026-06-14: Spherical delta steering

We then tested the movement-level composition directly.

New script:

- `scripts/run_upat_spherical_delta_steering.py`

New outputs:

- `results/experiments/upat_large_results/csv/spherical_delta_steering_raw.csv`
- `results/experiments/upat_large_results/csv/spherical_delta_steering_summary.csv`
- `results/experiments/upat_large_results/figures/14_spherical_delta_target_cosine.png`
- `results/experiments/upat_large_results/figures/14_spherical_delta_retrieval_top1.png`
- `results/experiments/upat_large_results/figures/14_spherical_delta_retrieval_label_f1.png`

Design:

- Learn class centroid deltas on the train split.
- Compare:
  - `linear_delta`: `unit(x + class_delta)`
  - `spherical_delta`: project class delta into the tangent space at `x`, then apply the sphere exp map
  - `rise_only`: simplified RISE-style tangent prototype prediction
  - residual orderings after `linear_delta`, `spherical_delta`, or `rise_only`
  - `hybrid_average`: average of `spherical_delta` and `rise_only`
- Evaluate target cosine, nearest-target retrieval top-1, and nearest-target label F1.

Result:

Within-model:

- Best target cosine: `rise_then_ours_residual`, `0.923427`
- `rise_only`: target cosine `0.923008`, retrieval label F1 `0.477279`
- `linear_delta`: target cosine `0.921752`, retrieval label F1 `0.463490`

Cross-model/full-anchor:

- Best target cosine: `linear_delta`, `0.613559`
- `spherical_delta`: target cosine `0.604795`, retrieval label F1 `0.412695`
- `linear_delta` retrieval label F1: `0.407674`
- Best retrieval label F1: `rise_only`, `0.445518`, with lower target cosine `0.578347`
- `hybrid_average`: target cosine `0.596853`, retrieval label F1 `0.433902`

Interpretation:

The movement-level result is not a simple hybrid win. Uncalibrated spherical delta steering does not improve cross-model target cosine over linear centroid steering, although it slightly improves retrieval label F1. RISE-style prediction remains best for transformation-neighborhood retrieval while being worse for exact target cosine.

This separates the metrics:

- Linear deltas are currently better at "hit the exact target embedding."
- RISE-style movement is currently better at "land in a target neighborhood with the right transformation label."
- A useful next test is train-only step-size calibration for spherical movement, because the uncalibrated tangent step may be geometrically correct in direction but not in length.

## 2026-06-14: Grammar-generated pairwise composition controls

We returned to the Lie-like direction and added the first grammar-generated Track 2 control.

New script:

- `scripts/run_lie_composition_grammar_controls.py`

New outputs:

- `results/experiments/lie_composition_grammar_results/csv/grammar_composition_dataset.csv`
- `results/experiments/lie_composition_grammar_results/csv/grammar_composition_summary.csv`
- `results/experiments/lie_composition_grammar_results/csv/grammar_endpoint_controls.csv`
- `results/experiments/lie_composition_grammar_results/csv/grammar_commutator_nulls.csv`
- `results/experiments/lie_composition_grammar_results/figures/01_relative_commutator_norm_heatmap.png`

Design:

- Generate `720` pairwise composition rows for `N,Q,M,T`.
- Vary subjects, actions, contexts, negation forms, modality markers, future forms, and question/order templates.
- Compute `AB` and `BA` endpoint deltas from the same source sentence.
- Report relative commutator norm:

```text
||delta_AB - delta_BA|| / mean(||delta_AB||, ||delta_BA||)
```

- Add endpoint-only controls for pair-label classification.
- Add three `N=1000` commutator nulls:
  - same-pair shuffled `AB`/`BA` pairing
  - any-pair shuffled endpoints
  - norm-matched random directions

Result:

- Relative commutator norms remain nonzero and structured under grammar variation.
- Across all tested model/pair rows, observed mean relative commutator norms are below the same-pair shuffled null means at the current empirical resolution floor (`p=0.000999`).
- Mean observed relative commutator norm across null rows: `0.5913`.
- Mean nulls:
  - same-pair random pairing: `0.8346`
  - any-pair random pairing: `1.0218`
  - norm-matched random directions: `1.4192`
- Endpoint and delta controls are too strong:
  - `ab_endpoint_only`: macro F1 `1.0000`
  - `ab_delta_only`: macro F1 `1.0000`
  - `commutator_delta`: macro F1 `1.0000`
  - `source_only`: macro F1 `0.0476`, chance accuracy `0.1667`

Interpretation:

This is a real step toward the Lie-like track, but not a clean win. Grammar variation does not destroy pairwise order structure, and observed commutator norms are far below shuffled/null baselines. However, endpoint-only and delta-only controls perfectly recover the pair label. That means the generated `AB`/`BA` endpoints still contain strong explicit order markers.

The next Track 2 step should be endpoint-balanced grammar generation: make pair labels harder to read from either endpoint alone while preserving ordered semantic composition. Only after that should we promote the pairwise commutator result as evidence for endpoint-independent algebraic structure.

## 2026-06-23: Multilingual max audit for signed-permutation structure

We ran the strongest Track 2 scale-up so far.

Experiment:

- Script: `scripts/run_lie_multilingual_max_audit.py`
- Output: `results/experiments/lie_multilingual_max_results/`
- Languages: English, Spanish, French, German, Russian, Chinese, Arabic
- Models:
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `sentence-transformers/LaBSE`
  - `intfloat/multilingual-e5-large`
  - `BAAI/bge-m3`
  - `bert-base-multilingual-cased`
- Dataset: `3360` rows, `8298` unique texts
- Nulls: `2000` signed-null samples per row
- PCA dimension: `96`

Main result:

| Triple | Mean ratio to signed-null | Std | Cells below null |
|---|---:|---:|---:|
| `NQM` | `0.580` | `0.072` | `35/35` |
| `QMT` | `0.620` | `0.085` | `35/35` |
| `NQT` | `0.701` | `0.073` | `35/35` |
| `NMT` | `0.772` | `0.049` | `35/35` |

This is the strongest evidence so far that the third-order signed-permutation diagnostic is not only an English-template artifact. The signal survives across seven languages and five multilingual encoders.

But the result also revises the story. The earlier `QMT` claim was true for the original English encoder/decoder table, where `QMT` was the only triple passing across all five tested models. It is not the global project claim anymore. In the multilingual audit, all four triples are below null and `NQM` is the strongest global triple.

Controls:

- Source-only held-out-language macro F1 is chance-like: `0.0476` for every model.
- Endpoint/delta/commutator controls remain high:
  - `ab_delta_only` mean macro F1 ranges from about `0.679` to `0.867`.
  - `commutator_delta` mean macro F1 ranges from about `0.497` to `0.758`.
- Cross-language centroid consistency is moderate and high-variance:
  - pair commutators: mean cosine `0.312`, std `0.332`
  - triple signed vectors: mean cosine `0.322`, std `0.339`

Interpretation:

This moves us closer to the Lie-algebra direction because the non-tautological third-order signal survives a much broader stress test. It is still not proof of a Lie algebra. Endpoint information remains strong, cross-language centroid consistency is not clean, and the triple ranking changes across regimes.

Next forced step:

Build endpoint-balanced multilingual templates and add target-only controls for the six third-order endpoints. The core question is now not "does QMT survive?", but "which signed-permutation cancellations survive when endpoints stop carrying easy class/order markers?"

## 2026-06-24: Endpoint-subspace residualization audit

We ran the next endpoint-artifact control for the Track 2 signed-permutation result.

New scripts:

- `scripts/run_lie_endpoint_residualization_audit.py`
- `scripts/run_lie_endpoint_subspace_residualization_audit.py`

New outputs:

- `results/experiments/lie_endpoint_residualization_results/`
- `results/experiments/lie_endpoint_subspace_residualization_results/`

Design:

- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 7 multilingual models:
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `sentence-transformers/LaBSE`
  - `intfloat/multilingual-e5-large`
  - `BAAI/bge-m3`
  - `bert-base-multilingual-cased`
  - `xlm-roberta-base`
  - `distilbert-base-multilingual-cased`
- 96 templates per language
- PCA dimension 96
- exact signed-null over the admissible endpoint sign assignments
- held-out-language endpoint probes

Endpoint probes:

| Probe | Mean macro F1 | Chance |
|---|---:|---:|
| cyclic versus anticyclic from endpoint delta | `0.522386` | `0.500000` |
| endpoint position from endpoint delta | `0.273599` | `0.166667` |
| triple label from single endpoint delta | `0.755258` | `0.250000` |

Main subspace-residualized result:

| Triple | Raw | Remove sign | Remove triple label | Remove endpoint position | Remove all |
|---|---:|---:|---:|---:|---:|
| `NMT` | `0.764073` | `0.763916` | `0.763325` | `0.764539` | `0.764100` |
| `NQM` | `0.543409` | `0.534386` | `0.544664` | `0.537042` | `0.538300` |
| `NQT` | `0.676606` | `0.671906` | `0.679131` | `0.675365` | `0.677458` |
| `QMT` | `0.589554` | `0.589631` | `0.589615` | `0.602188` | `0.603670` |

Interpretation:

The signed-permutation signal largely survives removal of endpoint sign, triple-label, and endpoint-position probe subspaces. This is stronger than merely saying that endpoint controls exist. Endpoints do encode task information, especially triple label from a single endpoint delta, but the third-order signed-permutation cancellation is not explained by those simple linear probe rowspaces.

This is still not a Lie algebra proof. It is a stronger diagnostic result:

> In the current multilingual synthetic setting, the third-order signed-permutation effect is robust to simple endpoint-derived linear residualization controls.

Next forced step:

Build endpoint-balanced multilingual templates and nonlinear/adversarial endpoint controls. The remaining criticism is no longer only "endpoint classifiers are strong"; it is "there may be nonlinear or lexical endpoint artifacts that the linear residualization audit does not remove."

## 2026-06-24: Linear Relational Decoding as a matrix-operator follow-up

We reviewed Xia and Kalita 2025, **"Linear Relational Decoding of Morphology in Language Models"** (`arXiv:2507.14640`).

Why it matters:

- The paper extends Linear Relational Embeddings to language-model morphology.
- It reports that relation-specific Jacobian-derived matrix operators can faithfully approximate many morphological relations.
- It separates additive and multiplicative mechanisms:
  - additive / bias-like movement
  - multiplicative matrix maps
  - affine `W_r x + b_r` maps
- Morphology appears especially compatible with multiplicative linear maps, including multilingual morphology.

Implication for this project:

Our current Track 2 signed-permutation diagnostic is still endpoint/delta based. That is a useful proxy, but it is not the strongest possible Lie-style formulation. The more literal version should learn operator-valued maps for `N,Q,M,T`:

```text
y ~= W_op x + b_op
```

and then compute:

```text
[W_A, W_B] = W_A W_B - W_B W_A
```

plus matrix Jacobi residuals. This would move the project closer to an actual algebraic claim than endpoint signed-permutation sums alone.

New follow-up:

Add an affine/operator-valued Track 2 experiment comparing additive delta-only, multiplicative matrix-only, and affine map variants. Use morphology as a possible sanity benchmark: if the pipeline cannot recover structure in a relation class where Linear Relational Decoding reports strong linearity, the method needs revision before broader linguistic-operator claims.

## 2026-06-25: GLT-AFFECT planning note

We added a new planned track:

**GLT-AFFECT: Graded Affective Transformation Geometry**

The core shift is from categorical operators to graded semantic axes. The first candidate axis is emotional polarity:

```text
hate (-2) -> dislike (-1) -> indifferent (0) -> like (+1) -> love (+2)
```

This gives a way to test linearity, curvature, saturation, and opposition directly. The important methodological correction is that antisymmetry is not evidence: `delta(A -> B) = -delta(B -> A)` follows from subtraction. It can stay only as a sanity check.

The second correction is about affect/negation. A tempting example says:

```text
affect then negation: "I love you" -> "I do not love you"
negation then affect: "I do not love you" -> "I hate you"
```

This is not a valid commutator unless both paths are defined as transformations toward comparable endpoints. "I do not love you" and "I hate you" are different affective states, not two equivalent endpoint orderings.

The first GLT-AFFECT experiment should therefore be narrow:

- text-only emotional polarity scale
- adjacent and non-adjacent deltas
- linearity and curvature tests
- explicit statement that we test language-representation geometry, not real affective grounding

Grounding is a separate future step. Pyrfume and psychophysical odor datasets make it possible to compare text-derived odor descriptor geometry with independent perceptual dissimilarity matrices, but that belongs to a later GLT-AFFECT-GROUNDING subtrack.

## 2026-06-25: GLT-AFFECT polarity MVP

We ran the first text-only GLT-AFFECT experiment.

Script:

- `scripts/run_glt_affect_polarity_mvp.py`

Outputs:

- `results/experiments/glt_affect_polarity_mvp_results/`

Design:

- 7 multilingual models
- 7 languages
- 160 templates per language
- 5 affective levels:

```text
hate (-2) -> dislike (-1) -> indifferent (0) -> like (+1) -> love (+2)
```

Main adjacent step norms:

| Step | Mean delta norm |
|---|---:|
| `hate -> dislike` | `3.896` |
| `dislike -> indifferent` | `5.046` |
| `indifferent -> like` | `5.377` |
| `like -> love` | `3.314` |

Opposition test:

| Comparison | Value |
|---|---:|
| row cosine, `neutral -> love` versus `neutral -> hate` | `0.614` |
| centroid cosine | `0.613` |
| norm ratio love/hate | `1.053` |

Curvature summary:

| Metric | Value |
|---|---:|
| adjacent-step CV | `0.224` |
| cosine `hate->dislike` vs `dislike->neutral` | `-0.343` |
| cosine `dislike->neutral` vs `neutral->like` | `-0.682` |
| cosine `neutral->like` vs `like->love` | `-0.279` |

Interpretation:

The affective polarity scale is not a straight line in embedding space. The middle steps are larger than the edge steps, and love/hate are not antipodal directions. A better first interpretation is involvement geometry: love and hate both move away from indifferent language toward emotionally loaded regions, while polarity is not represented as a simple one-dimensional sign.

The line-additivity sanity check produced near-zero residuals, as it should from vector subtraction. This confirms the arithmetic but is not a scientific result.

Next GLT-AFFECT steps:

- redesign templates to reduce lexical asymmetry across languages
- add more graded emotion axes, such as calm/anxious or trust/distrust
- add bootstrapped confidence intervals over templates
- later, compare text-derived descriptor geometry with non-textual psychophysical anchors through Pyrfume

## 2026-06-24: Structure-constants closure audit

We ran the first GLT-MOLT-style bridge experiment.

Script:

- `scripts/run_lie_structure_constants_audit.py`

Outputs:

- `results/experiments/lie_structure_constants_results/`

Design:

- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 7 multilingual models:
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
  - `sentence-transformers/LaBSE`
  - `intfloat/multilingual-e5-large`
  - `BAAI/bge-m3`
  - `bert-base-multilingual-cased`
  - `xlm-roberta-base`
  - `distilbert-base-multilingual-cased`
- 160 templates per language
- 31,776 unique texts
- PCA dimension 128
- 1000 random-subspace null samples

Main closure result:

| Pair | Mean closure residual | Random-subspace null |
|---|---:|---:|
| `MT - TM` | `0.767` | `0.986` |
| `NM - MN` | `0.772` | `0.987` |
| `NQ - QN` | `0.853` | `0.984` |
| `NT - TN` | `0.885` | `0.984` |
| `QT - TQ` | `0.878` | `0.984` |
| `QM - MQ` | `0.905` | `0.984` |

Important caveat:

Chinese has zero commutator norm for `NM - MN` and `MT - TM` across all models. These rows are template degeneracies, not evidence of perfect closure. After removing zero-commutator rows, the nonzero overall mean closure residual is still below the random-subspace null (`0.885` versus `0.984`), so the partial closure signal survives.

Jacobi-like closure result:

| Triple | Mean relative Jacobi closure norm |
|---|---:|
| `NMT` | `0.309` |
| `QMT` | `0.333` |
| `NQM` | `0.431` |
| `NQT` | `0.853` |

Interpretation:

This is not a Lie algebra proof, but it is the first result that resembles structure constants rather than only endpoint signed sums. Pairwise commutators are partially compressible into the span of primitive operator centroids better than random subspaces. The next step should replace centroid displacement operators with learned affine/multiplicative maps `W_op x + b_op`, then compute matrix commutators directly.

## 2026-06-24: GLT-MOLT affine operator pilot

We ran the first matrix/operator version of the Track 2 audit.

Script:

- `scripts/run_glt_molt_affine_operator_audit.py`

Outputs:

- `results/experiments/glt_molt_affine_operator_results/`

Design:

- 7 multilingual models
- 7 languages
- 160 templates per language
- PCA dimension 128
- ridge alpha `10.0`
- 200 random-subspace nulls

The audit compares three operator families:

- additive centroid maps: `y ~= x + delta_op`
- linear matrix maps: `y ~= W_op x`
- affine matrix maps: `y ~= W_op x + b_op`

One-step target reconstruction:

| Method | Mean target cosine |
|---|---:|
| additive | `0.895` |
| linear | `0.735` |
| affine | `0.693` |

Ordered composition target prediction:

| Method | Mean AB cosine | Mean BA cosine |
|---|---:|---:|
| additive | `0.809` | `0.823` |
| linear | `0.595` | `0.617` |
| affine | `0.476` | `0.522` |

Matrix closure:

| Method | Mean closure residual | Random-subspace null |
|---|---:|---:|
| affine | `0.965` | `0.9999` |
| linear | `0.970` | `0.9999` |

Matrix Jacobi-like residual:

| Method | Mean relative Jacobi operator norm |
|---|---:|
| affine | `0.063` |
| linear | `0.067` |

Interpretation:

This is a clean split. Additive deltas are better for hitting target endpoints, while learned matrix operators are worse at prediction but produce cleaner algebraic diagnostics. This does not mean the matrices are better linguistic models; ridge regularization may make them algebraically smooth. The next GLT-MOLT step should sweep `ridge_alpha` and report the tradeoff among target cosine, commutator norm, closure residual, and Jacobi residual.

## 2026-06-30: GLT-MOLT 9-model 1000-null confirmation

The 9-model GLT-MOLT affine/operator run completed with 1000 random-subspace nulls and no model failures.

Outputs:

- `results/experiments/glt_molt_affine_operator_9m_160t_1000null_results/`

Main confirmation:

- Additive maps remain better endpoint predictors than learned matrix maps (`0.903` mean target cosine versus `0.738` linear and `0.695` affine).
- Composition prediction follows the same split: additive maps reach `0.821/0.835` mean AB/BA cosine, while linear maps reach `0.591/0.615` and affine maps `0.478/0.520`.
- Matrix commutator closure remains weak but systematic against rank-matched random subspaces:
  - linear mean residual `0.975` versus random null `~0.9999`, mean empirical p `0.0020`
  - affine mean residual `0.971` versus random null `~0.9999`, mean empirical p `0.0037`
- Jacobi-like residuals remain low (`0.067` linear, `0.064` affine).

Interpretation:

The 1000-null rerun does not change the story; it stabilizes it. The operator maps are not better predictors, and the commutators are far from exact closure, but the closure residuals are consistently better than random subspaces. The caution is still ridge regularization: the next experiment should sweep `ridge_alpha` and report the target-prediction/closure/Jacobi tradeoff.

## 2026-07-01: GLT-MOLT ridge sweep

We ran the GLT-MOLT ridge-alpha sweep to test whether the operator-valued closure signal is stable across regularization or mostly a ridge-smoothing artifact.

Outputs:

- `scripts/run_glt_molt_ridge_sweep.py`
- `results/experiments/glt_molt_ridge_sweep_9m_160t_300null_results/`

Setup:

- 9 multilingual embedding models
- 7 languages: English, Spanish, French, German, Russian, Chinese, Arabic
- 160 templates per language
- PCA dimension `128`
- ridge alphas: `0.1`, `1.0`, `10.0`, `100.0`
- 300 random-subspace nulls

Main result:

- Additive target prediction is unchanged at mean target cosine `0.903`.
- Linear/affine target prediction is best around `alpha=10` (`0.738` linear, `0.695` affine) and worsens by `alpha=100`.
- Closure residuals improve with stronger ridge smoothing:
  - linear: `0.995 -> 0.991 -> 0.975 -> 0.882`
  - affine: `0.994 -> 0.991 -> 0.970 -> 0.855`
- Jacobi-like norms also generally improve, especially affine at `alpha=100` (`0.042`).

Interpretation:

The closure-like signal is not isolated to one alpha, but algebraic cleanliness is regularization-sensitive. This strengthens GLT-MOLT as a diagnostic while weakening any premature Lie-algebra claim. The next operator-level control must compare against norm-matched and shrinkage-matched operator nulls rather than only random subspaces.

## 2026-07-02: GLT-MOLT matched-null operator closure

We ran the direct follow-up to the ridge sweep: a GLT-MOLT matched-null audit that compares learned operator closure against random-subspace, Gaussian norm-matched, and signed-permutation matched operator nulls.

Outputs:

- `scripts/run_glt_molt_matched_nulls.py`
- `results/experiments/glt_molt_matched_nulls_9m_160t_a10_100_1000null_results/`

Setup:

- 9 multilingual embedding models
- 7 languages
- 160 templates per language
- PCA dimension `128`
- ridge alphas: `10.0`, `100.0`
- 1000 null samples per row

Main result:

| Ridge alpha | Method | Observed closure | Random-subspace null | Gaussian norm-matched null | Signed-permutation matched null |
|---:|---|---:|---:|---:|---:|
| `10` | affine | `0.970` | `0.99988` | `0.99988` | `0.99978` |
| `10` | linear | `0.975` | `0.99988` | `0.99988` | `0.99977` |
| `100` | affine | `0.855` | `0.99988` | `0.99988` | `0.99989` |
| `100` | linear | `0.882` | `0.99988` | `0.99988` | `0.99989` |

At `alpha=100`, all mean empirical p-values are at the `N=1000` resolution floor (`0.000999`) across all three null families. At `alpha=10`, the observed closure remains below every null family with mean p-values around `0.002-0.009`.

Interpretation:

This is a meaningful strengthening of GLT-MOLT. The closure-like compression is not explained by random subspaces, norm-matched Gaussian operator maps, or signed-permutation matched operator maps. The central caveat remains: `alpha=100` is cleaner algebraically but worse for target prediction, so the result is about operator-space compression rather than endpoint reconstruction.

Next step:

Test singular-spectrum/shrinkage-matched nulls and layer/PCA-dimension sensitivity. If the signal survives those controls, GLT-MOLT becomes the closest current track to a serious Lie-algebra-style claim.

## 2026-07-03: GLT-MOLT spectral-null operator closure

We ran the direct follow-up to the matched-null audit: a singular-spectrum matched null for the `alpha=100` GLT-MOLT setting.

Artifacts:

- `scripts/run_glt_molt_spectral_nulls.py`
- `results/experiments/glt_molt_spectral_nulls_9m_160t_a100_300null_g256_results/`

Configuration:

- 9 multilingual embedding models
- 7 languages
- 160 templates per language
- PCA dimension 128
- ridge alpha `100`
- 300 spectral null samples per row
- 256 random Givens rotations for row and column directions

Main result:

| Method | Observed closure | Spectral matched null | Mean empirical p |
|---|---:|---:|---:|
| affine | `0.855` | `0.99985` | `0.00333` |
| linear | `0.882` | `0.99985` | `0.00347` |

Interpretation:

The closure-like compression survives a stricter null that preserves each commutator's singular-value spectrum. This weakens the explanation that the `alpha=100` result is only generic ridge shrinkage or spectral shape. The result is still not a Lie-algebra proof: target prediction is worse at `alpha=100`, and model-level closure magnitudes are uneven, with `Qwen/Qwen3-Embedding-0.6B` close to null-like in residual magnitude.

Next step:

Run layer/PCA-dimension ablations for GLT-MOLT before expanding operator families. If the spectral-null signal survives those ablations, this track becomes the most serious route toward a Lie-style paper.

## 2026-07-08: Compact GLT-MOLT PCA-64 spectral sensitivity

We ran a compact follow-up to test whether the GLT-MOLT spectral-null closure signal survives in a lower-dimensional PCA space. Full 9-model PCA sweeps were too heavy for the machine, so this run uses five stable multilingual encoders and treats PCA-64 as a completed sensitivity slice rather than a complete PCA sweep.

Artifacts:

- `scripts/launch_glt_molt_compact_pca_sweep.ps1`
- `results/experiments/glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256_results/`

Configuration:

- 5 stable multilingual encoder models
- 7 languages
- 160 templates per language
- PCA dimension `64`
- ridge alpha `100`
- 300 spectral null samples per row
- 256 random Givens rotations

Main result:

| Method | Observed closure | Spectral matched null | Mean empirical p |
|---|---:|---:|---:|
| affine | `0.8285` | `0.9996` | `0.00332` |
| linear | `0.8701` | `0.9995` | `0.00335` |

Interpretation:

The PCA-64 compact run supports the claim that the spectral-null closure-compression signal is not unique to PCA-128. It remains a bounded result: PCA-128 and PCA-256 should be run as separate smaller jobs before claiming a completed PCA-dimension ablation.

## 2026-08-25: Retrospective catch-up for August GLT-STEER work

This entry intentionally records an August gap in the diary rather than pretending the notes were written at run time. The public research record had ended on 2026-07-08, while several GLT-STEER experiments were completed between 2026-08-01 and 2026-08-21. This catch-up records the major decisions and boundaries so that the Track 1 / GLT-STEER narrative is not reconstructed silently after the fact.

### 2026-08-01: DistilGPT-2 tuning, ellipsis, and first marker composition

Artifacts:

- `results/experiments/distilgpt2_question_hard_oot_best_layer2_gain10_20260801_results/`
- `results/experiments/gpt2_ellipsis_copy_prompt_steering_20260801_results/`
- `results/experiments/gpt2_ellipsis_hard_oot_layer2_20260801_results/`
- `results/experiments/gpt2_question_exclamation_marker_composition_layer2_3_20260801_results/`

Main results:

- DistilGPT-2 `layer=2, gain=1.0` was selected from the layer/gain sweep as the best known question-steering setting before interpreting the hard out-of-template audit.
- On hard out-of-template sources, DistilGPT-2 still produces question markers at high rates (`0.725-0.800`) under the target vector, but strict question-and-preserved rates are much lower (`0.025-0.225`). This is marker-form transfer, not semantic-preservation replication.
- Ellipsis steering supports the Final Marker Hypothesis beyond question marks: hard out-of-template target ellipsis rates reach `0.925-0.950`, with non-target ellipsis controls at `0.0000`.
- The first `question + exclamation` composition test shows clean single-vector effects but mixed/saturated combined effects. The result should be read as final-marker competition, not order-sensitive algebra.

Tuning disclosure:

The DistilGPT-2 `layer=2, gain=1.0` choice was not preregistered as a train/dev/test protocol. It came from a layer/gain sweep over copy-prompt settings, and the hard-OOT run was interpreted after that choice. The hard-OOT source sentences are structurally different, but prompt families and metrics were already known. This must remain a limitation in the Track 1 / GLT-STEER draft.

### 2026-08-08 to 2026-08-10: Final-marker logit audit and modality boundary

Artifacts:

- `results/experiments/gpt2_final_marker_logit_audit_layer2_3_20260810_v2_results/`
- `results/experiments/gpt2_question_modality_composition_layer2_3_20260808_v2_results/`

Main results:

- The final-marker logit audit is the strongest current mechanistic evidence for Track 1 / GLT-STEER. No-steering marker rates remain `0.0000`, while target steering moves the intended marker token into the top rank in most GPT-2 sequences.
- Aggregate GPT-2 target marker rates: `?=0.8542`, `!=0.9063`, `...=0.8750`.
- The question/modality composition audit is negative for the current modality recipe: question steering remains strong, but modality markers stay at `0.000`. This reinforces the boundary between final-position markers and sentence-internal or lexical transformations.

Interpretation:

The central claim should be narrowed to final-position surface-marker steering. This is closer to activation-steering / representation-engineering work than to a broad new algebraic claim, and the related-work section must say so explicitly.

### 2026-08-21: Position audit and DistilGPT-2 final-marker transfer

Artifacts:

- `results/experiments/gpt2_question_position_intervention_audit_layer2_3_20260821_results/`
- `results/experiments/distilgpt2_final_marker_logit_audit_l1_2_3_gain10_20260821_results/`

Main results:

- Single prompt-position edits are null: first, middle, and last prompt-token-only interventions all produce `0.0000` question rate.
- Distributed or repeated interventions work: `target_prompt_all_once` reaches question rate `0.8625`, and `target_last_each_step` reaches `0.9604`.
- DistilGPT-2 final-marker transfer is positive but strongly marker-dependent: no-steering remains `0.0000`, while target rates are `?=0.2986`, `!=0.7778`, `...=0.5139`.

Interpretation:

The position audit argues against a trivial single-token prompt perturbation story. DistilGPT-2 argues for model-dependent transfer rather than full replication.

### 2026-08-25: CI audit and stopping rule

Artifacts:

- `scripts/summarize_glt_steer_headline_ci.py`
- `results/experiments/glt_steer_headline_ci_20260825_results/`

Main results:

- Wilson 95% confidence intervals and sample sizes are now reported for the main Track 1 / GLT-STEER headline rows.
- GPT-2 final-marker target rates remain well separated from no-steering:
  - `?=0.8542`, `N=96`, CI `[0.7700, 0.9111]`
  - `!=0.9063`, `N=96`, CI `[0.8313, 0.9499]`
  - `...=0.8750`, `N=96`, CI `[0.7941, 0.9270]`
- DistilGPT-2 remains positive but weaker:
  - `?=0.2986`, `N=144`, CI `[0.2299, 0.3778]`
  - `!=0.7778`, `N=144`, CI `[0.7032, 0.8380]`
  - `...=0.5139`, `N=144`, CI `[0.4330, 0.5941]`

Stopping rule:

Track 1 / GLT-STEER should now converge as a short paper / extended abstract around the Final Marker Hypothesis. After adding activation-steering related work, explicit N/CI, and the DistilGPT-2 tuning disclosure, new experiments should be moved to future work unless an external reviewer or concrete venue requirement asks for them.

## 2026-08-25: GLT-STEER fixed-parameter confirmation

Artifacts:

- `scripts/run_glt_steer_confirmatory_fixed_params.py`
- `results/experiments/glt_steer_confirmatory_fixed_params_20260825_results/`

Purpose:

This run tests the current Track 1 / GLT-STEER settings without doing another layer/gain search. GPT-2 uses layers `2,3` at gain `0.75`; DistilGPT-2 uses layer `2` at gain `1.0`. The source set is a fresh hard-heldout set of `48` sentences, and the targets are the three final-marker classes: question, exclamation, and ellipsis.

Main result:

| model | target | target marker rate | max control marker | target marker+preserved | max control marker+preserved |
|---|---|---:|---:|---:|---:|
| `gpt2` | `question` | `0.6562` | `0.0000` | `0.2396` | `0.0000` |
| `gpt2` | `exclamation` | `0.6875` | `0.0000` | `0.3958` | `0.0000` |
| `gpt2` | `ellipsis` | `0.8438` | `0.0000` | `0.3854` | `0.0000` |
| `distilgpt2` | `question` | `0.6319` | `0.0069` | `0.0556` | `0.0000` |
| `distilgpt2` | `exclamation` | `0.8958` | `0.0139` | `0.3472` | `0.0000` |
| `distilgpt2` | `ellipsis` | `0.8333` | `0.0000` | `0.1597` | `0.0000` |

Interpretation:

The confirmatory run supports the bounded Final Marker Hypothesis under fixed settings. Target final-marker rates remain clearly separated from matched controls. The preservation rates are not strong enough to promote this as general semantic editing, especially for DistilGPT-2 question steering, so the Track 1 claim remains form steering with partial and model-dependent preservation.

## 2026-08-27: GLT-STEER runtime form-control applicability audit reviewed

Artifacts:

- `scripts/run_glt_steer_apply_runtime_form_control.py`
- `results/experiments/glt_steer_apply_runtime_form_control_20260825_results/`

Purpose:

This run tests whether final-marker steering is practically useful as runtime form control, not only mechanistically interesting. The audit compares target steering against no-steering, strong prompt instructions, deterministic string append, wrong-marker vectors, random-norm vectors, and negative-target vectors.

Configuration:

- GPT-2: layers `2,3`, gain `0.75`
- DistilGPT-2: layer `2`, gain `1.0`
- Targets: question, exclamation, ellipsis
- Held-out sources: `80`
- Raw generations: `8640`
- Failures: none

Main result:

| model | target | N | target marker | target marker+content | malformed/repetitive |
|---|---|---:|---:|---:|---:|
| `gpt2` | `question` | `320` | `0.6281` | `0.3469` | `0.3469` |
| `gpt2` | `exclamation` | `320` | `0.6031` | `0.3844` | `0.2344` |
| `gpt2` | `ellipsis` | `320` | `0.6937` | `0.3219` | `0.3000` |
| `distilgpt2` | `question` | `160` | `0.4938` | `0.1688` | `0.6125` |
| `distilgpt2` | `exclamation` | `160` | `0.8438` | `0.4250` | `0.1938` |
| `distilgpt2` | `ellipsis` | `160` | `0.7250` | `0.2562` | `0.5687` |

Matched vector controls (`none`, `wrong_marker`, `random_norm`, `negative_target`) have `0.0000` marker-plus-content rate in every aggregate model/target cell. The strong-prompt baseline also has `0.0000` target-marker rate under this protocol. However, deterministic `string_append_source` is perfect (`1.0000` marker, content, and joint rates).

Interpretation:

This is a useful boundary result. It strengthens the causal intervention story because target steering changes model behavior where prompt-only and matched vector controls do not. It weakens any broad practical-editing claim because deterministic postprocessing is strictly better when the requested operation is only a known final marker. The correct application framing is therefore: GLT-STEER is an activation-space diagnostic and form-bias intervention, not a production replacement for ordinary string editing.
