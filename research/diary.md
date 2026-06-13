# Research Diary

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

## 2026-06-12: Reviewer-driven controls and claim narrowing

Today we converted the reviewer concerns into executable checks.

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

The reviewer expected Track 1 confusion matrices to explain Track 2 negation failures. The analysis showed a subtler result:

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
