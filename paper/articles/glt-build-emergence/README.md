# GLT-BUILD: Building and Understanding Internal Latent Dynamics

This folder stores the GLT-BUILD research design and initial experimental record.

GLT-BUILD asks how linguistic transformation structure emerges while a model is trained, rather than only probing the final representation space of already-trained models.

Current status:

- Research design: drafted.
- First pilot: completed on 2026-09-11; learned-structure claims are unsupported by the pilot's raw cosine alone.
- Corrected composition audit: completed on 2026-09-15, three seeds, all 88 held-out scenes. Known commands transfer; general composition and mean-vector editing remain unsupported.
- Paper status: not submission-ready; this is a protocol/specification folder.

Primary artifact:

- `experiment_spec.md`
- [Initial pilot](../../../results/experiments/glt_build_01_explicit_operator_smoke_20260911_results/SUMMARY.md)
- [Corrected protocol](composition_audit_protocol.md)
- [Corrected results](../../../results/experiments/glt_build_01_composition_audit_20260915_results/SUMMARY.md)
- [Paired input-support protocol](input_support_protocol.md)
- [Paired input-support results](../../../results/experiments/glt_build_01_input_support_20260915_results/SUMMARY.md)
- [Explicit pair-prefix holdout protocol](pair_prefix_holdout_protocol.md)
- [Explicit pair-prefix holdout results](../../../results/experiments/glt_build_01_pair_prefix_holdout_20261008_results/SUMMARY.md)
- [GLT-BUILD-02 order-sensitive protocol](order_sensitive_composition_protocol.md)
- [GLT-BUILD-02 order-sensitive results](../../../results/experiments/glt_build_02_order_sensitive_composition_20261009b_results/SUMMARY.md)

The informative contrast is text-mediated SQ: swapping roles and then asking a
question succeeds on 264/264 evaluations, but reversing those steps succeeds on
0/264. Other tested sequential pairs fail even with correct intermediate text.
This suggests a training-input support boundary; it does not demonstrate learned
noncommutativity. Linear predictability also exists at initialization and in
token-only pooled representations, so fit alone is insufficient for emergence.

The completed input-support follow-up isolates source-state support with paired initialization,
draw schedules and padded update budgets. It groups both participant orders in
one split and tests all nonbase single operations alongside real composition.
The six-run schedule and primary decision rule are fixed in the linked protocol;
the final decision is `input_support_rescue_under_fixed_protocol`.

The completed pair-prefix audit then separates sequential text execution from
explicit command-prefix composition. Sequential execution reaches 100% in both
arms, but the single-operation-only arm reaches 0% on unseen direct pair prefixes.
This supports primitive reuse through generated text, not an explicit learned
composition algebra. All current operations commute in the generator, so no
noncommutative or Lie-algebra claim follows.

Relationship to existing GLT tracks:

- GLT-DV asks whether transformation deltas are recoverable in frozen embeddings.
- GLT-SPOT and GLT-MOLT ask whether composition and learned operators show structured diagnostics.
- GLT-STEER asks whether transformation vectors can causally affect a pretrained model.
- GLT-BUILD asks when and how such structures emerge during controlled training.

The initial goal is not to claim a Lie algebra. The first GLT-BUILD experiments should test simpler and better-specified structures first: discrete operations, involutions, commutation/noncommutation, group or monoid actions, and matrix/operator representations.

GLT-BUILD-02 extends this line with an explicit noncommuting R/M pair. R swaps
roles and M toggles a mark on the current subject; T and N provide
an independent commuting control. The final audit supports order-sensitive
sequential execution on held-out scene groups, but direct pair-prefix
generalization remains weak. The result is behavioral and discrete, not a Lie
algebra claim.
