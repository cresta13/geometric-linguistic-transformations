# Scripts Index

All runnable research scripts live in this directory.

## Track 1

- `lie_llm_full_semantic_holdout_experiment.py`: full-semantic holdout.
- `lie_llm_syntax_holdout_experiment.py`: syntax holdout.
- `lie_llm_diverse_dataset_experiment.py`: diverse transformation dataset.
- `lie_llm_entity_holdout_experiment.py`: entity holdout.
- `lie_llm_variant_holdout_experiment.py`: variant holdout.
- `lie_llm_y_only_ablation_multiseed.py`: multiseed endpoint/delta ablation.
- `build_track1_effect_intervals.py`: 95% seed-level effect intervals for Track 1 `delta-y_only` and `delta-concat`.
- `run_syntax_representation_ablation.py`: syntax x/y/concat/delta ablation.
- `run_layerwise_pooling_ablation.py`: syntax layerwise/pooling ablation.
- `run_full_semantic_pooling_ablation.py`: full-semantic pooling ablation.
- `run_track1_spotcheck.py`: modern/larger model spot-checks.
- `make_paper_figures.py`: curated Track 1 figures.

## Track 2

- `lie_composition_dataset.py`: composition dataset utilities.
- `run_lie_composition_audit.py`: pairwise composition/noncommutativity audit.
- `run_lie_algebraic_identities.py`: signed-permutation diagnostic.
- `run_lie_semantic_equivalence_control.py`: semantic-equivalence control.
- `run_lie_composition_grammar_controls.py`: grammar-generated pairwise composition control for `N,Q,M,T`, with endpoint-only classifiers and commutator norm nulls.
- `run_lie_multilingual_max_audit.py`: multilingual signed-permutation and pairwise composition audit with held-out-language endpoint controls. Use `LIE_MULTI_OUT_DIR` for scale-up reruns without overwriting the canonical 2026-06-23 result; set `LIE_TRUST_REMOTE_CODE=1` for newer Hugging Face embedding models that require custom model code. Pooled tensors are cast to `float32` before NumPy conversion so CPU `bfloat16` models such as Qwen3 embeddings can be audited.
- `run_lie_multilingual_triple_endpoint_controls.py`: held-out-language third-order endpoint-control classifiers for source-only, endpoint-only, endpoint-concat, delta-concat, and signed endpoint vectors.
- `run_lie_endpoint_subspace_residualization_audit.py`: removes endpoint-derived triple-label, endpoint-position, and cyclic-sign rowspaces before recomputing signed-permutation ratios. Use `LIE_SUBSPACE_OUT_DIR` for separate reruns.
- `run_glt_molt_affine_operator_audit.py`: GLT-MOLT additive, linear, and affine operator audit over `N,Q,M,T`.
- `run_glt_molt_ridge_sweep.py`: GLT-MOLT ridge-alpha sweep that reuses each model's embeddings across alpha values and reports target-prediction, matrix-closure, and Jacobi tradeoffs.
- `run_glt_molt_matched_nulls.py`: GLT-MOLT operator-closure control against random-subspace, Gaussian norm-matched, and signed-permutation matched operator nulls.
- `build_signed_permutation_multiple_testing.py`: table-level multiple-testing correction.

## UPAT / Cross-Model

- `upat_dataset.py`: UPAT dataset construction.
- `upat_audit_results.py`: original UPAT audit.
- `run_upat_large.py`: larger UPAT/cross-model audit.
- `run_upat_procrustes_nulls.py`: random-pairing, random-label, and random-orthogonal null baselines for UPAT cross-model Procrustes transfer. Use `--n-null 1000` for the current full audit.
- `run_upat_alignment_size_heldout.py`: held-out anchor alignment-size curve for UPAT cross-model Procrustes transfer. The current full run uses `--sizes 25,50,100,250,500,1000 --repeats 10 --anchor-target-count 1200`.
- `run_upat_rise_aware_comparison.py`: first-pass MDV and simplified RISE-style prototype comparison for UPAT target-embedding prediction versus delta-classifier transfer.
- `run_upat_hybrid_rise_procrustes.py`: non-leaky hybrid transfer test. It scores each UPAT pair against all MDV/RISE-style class prototypes and tests whether prototype-score features improve cross-model transformation-label F1 beyond aligned `delta_only`.
- `run_upat_spherical_delta_steering.py`: movement-level steering test comparing linear centroid deltas, spherical/tangent delta movement, RISE-style prediction, residual orderings, and hybrid averaging on target cosine and retrieval metrics.

## Reports

- `build_reviewer_figures.py`: reviewer-response figures.
- `build_research_report.py`: dated PDF report.
- `analyze_confusion_negation.py`: confusion/negation summary tables.

Run scripts from the repository root so relative output paths resolve into `results/` and `results/experiments/`.
