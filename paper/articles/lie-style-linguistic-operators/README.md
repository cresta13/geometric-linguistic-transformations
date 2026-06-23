# Third-Order Signed Permutation Coherence for Linguistic Transformations in Transformer Embedding Spaces

## Working thesis

Some linguistic transformations behave like locally noncommuting operators in transformer embedding space. The original English encoder/decoder table singled out `QMT`; the 2026-06-23 multilingual max audit broadens the result, with all tested triples below signed-null across 7 languages and 5 multilingual encoders. The structure is still local and controlled, not a global Lie algebra.

## Evidence

- Composition tests show order-sensitive transformation behavior.
- Semantic equivalence controls separate wording variation from semantic noncommutativity.
- Antisymmetry is not evidence; it is a tautological implementation check under the implemented commutator.
- Signed permutation diagnostics with bootstrap CI and permutation-null controls show robust controlled cancellation.
- Grammar-generated pairwise composition controls now show relative commutator norms below shuffled/norm-matched nulls, but endpoint-only controls are still near perfect.
- The multilingual max audit gives the strongest current scale-up: `NQM=0.580`, `QMT=0.620`, `NQT=0.701`, `NMT=0.772` as global mean ratios to signed-null across 35 model-language cells per triple.

## Main caution

A literal nested-commutator Jacobi expansion over the same six endpoint vectors cancels by construction. The useful diagnostic is the non-tautological alternating third-order composition sum compared against null sign assignments. It should not be presented as a formal Jacobi identity.

## Next experiments

- Build endpoint-balanced grammar generation before expanding the operator set.
- Add target-only controls for third-order composition diagnostics.
- Add endpoint-balanced multilingual generation to explain why `NQM` becomes strongest in the multilingual audit while `QMT` was the most stable English/decoder result.
- Add null baselines for pairwise commutator norms before promoting pairwise composition claims.
- Test contextual embeddings layer by layer and across pooling choices.
- Treat cross-model alignment for composition vectors as a separate future direction, not a blocker for the local signed-permutation paper.
