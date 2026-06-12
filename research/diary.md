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
