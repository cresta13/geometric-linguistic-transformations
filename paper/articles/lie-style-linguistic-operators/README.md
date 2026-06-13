# Third-Order Signed Permutation Coherence for Linguistic Transformations in Transformer Embedding Spaces

## Working thesis

Some linguistic transformations behave like locally noncommuting operators in transformer embedding space. Certain triples, especially `QMT`, show third-order signed permutation cancellation stronger than permutation-null baselines, but the structure is local rather than a global Lie algebra.

## Evidence

- Composition tests show order-sensitive transformation behavior.
- Semantic equivalence controls separate wording variation from semantic noncommutativity.
- Antisymmetry is not evidence; it is a tautological implementation check under the implemented commutator.
- Signed permutation diagnostics with bootstrap CI and permutation-null controls show robust `QMT` cancellation.

## Main caution

A literal nested-commutator Jacobi expansion over the same six endpoint vectors cancels by construction. The useful diagnostic is the non-tautological alternating third-order composition sum compared against null sign assignments. It should not be presented as a formal Jacobi identity.

## Next experiments

- Expand beyond `N,Q,M,T`.
- Replace hand-written templates with grammar-driven generation.
- Add null baselines for pairwise commutator norms and Procrustes-style alignment claims before promoting them.
- Test contextual embeddings layer by layer.
- Add cross-model alignment for composition vectors.
