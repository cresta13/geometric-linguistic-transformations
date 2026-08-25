# GLT-STEER: Transformation Vectors as Activation-Space Editors

Status: short draft, not peer reviewed.

## Abstract

This draft tests whether transformation vectors learned from sentence-pair hidden-state differences can be injected back into a generative transformer as activation-space editors. In GPT-2, a question-transformation vector injected into early residual-stream layers reliably induces question-form outputs under matched controls. The strongest copy-prompt setting reaches question-and-preserved rates up to `0.975` on simple held-out sources, while no-steering, random-norm, wrong-class, and negative-vector controls remain at `0.000` in matched headline rows. Hard out-of-template sources preserve the question-marker effect but reduce content preservation. The emerging explanation is the **Final Marker Hypothesis**: final-position surface markers such as `?`, `!`, and `...` are reliably steerable via mean delta vectors, while lexical or sentence-internal transformations such as negation and modality are not under the same recipe. DistilGPT-2 replicates marker-form steering only after layer/gain tuning and does not replicate GPT-2-level hard out-of-template preservation. A first question/exclamation composition test is best interpreted as final-marker competition/saturation rather than noncommutative order structure.

## 1. Question

The representation-track result asks whether sentence-pair deltas classify transformation type:

```text
delta = h(target sentence) - h(source sentence)
```

GLT-STEER asks a stronger intervention question:

```text
If a transformation delta is injected during generation, does the model produce the corresponding transformation?
```

This is a behavior-level test. It is not enough for a formal algebraic claim, but it is stronger than offline separability because the vector is used causally inside a running model.

## 2. Method

The core recipe is:

1. Build synthetic source/target sentence pairs for a transformation class.
2. Extract GPT-style hidden states at selected transformer blocks.
3. Compute a centroid transformation vector from training pairs.
4. During generation, add the vector to the last prompt token at a chosen residual block.
5. Compare target steering against no-steering, wrong-class, random-norm, and negative-vector controls.

Primary implementation files:

- `../../../scripts/run_gpt2_activation_steering_pilot.py`
- `../../../scripts/run_gpt2_question_steering_controls.py`
- `../../../scripts/run_gpt2_transformation_copy_prompt_steering.py`
- `../../../scripts/run_gpt2_exclamation_copy_prompt_steering.py`
- `../../../scripts/run_gpt2_final_marker_copy_prompt_steering.py`
- `../../../scripts/run_gpt2_final_marker_logit_audit.py`
- `../../../scripts/run_gpt2_question_position_intervention_audit.py`
- `../../../scripts/run_gpt2_marker_composition_steering.py`
- `../../../scripts/run_glt_steer_confirmatory_fixed_params.py`
- `../../../scripts/summarize_glt_steer_headline_ci.py`

The basic intervention hook adds the learned vector to the last token hidden state inside a transformer block. The current implementation is intentionally simple: no learned controller, no prompt-specific optimization, and no gradient update at generation time.

## 2.1 Final Marker Hypothesis

The current GLT-STEER results are best summarized by one bounded claim:

> Final-position surface markers (`?`, `!`, `...`) are reliably steerable in GPT-2 via mean hidden-state delta vectors; lexical or sentence-internal transformations such as negation and modality are not reliably steerable under the same recipe.

This is a mechanistic interpretation, not just a list of successes and failures. It explains why question, exclamation, and ellipsis succeed under copy-like prompts, while negation and modality remain weak or negative. It also bounds the claim: GLT-STEER currently supports output-form editing, not general semantic rewriting.

## 2.2 Relation to Activation Steering Work

GLT-STEER is closest to the activation-engineering and representation-steering literature, not to the cross-model geometry literature that motivates earlier GLT tracks. Activation Addition / ActAdd computes activation differences from contrastive prompts and injects them at inference time to steer model behavior. Contrastive Activation Addition (CAA) averages residual-stream differences over positive/negative behavioral examples. Representation Engineering (RepE) frames this broader family of methods as population-level monitoring and manipulation of internal representations.

This draft should therefore not claim that adding hidden-state vectors during generation is new by itself. The narrower contribution is diagnostic: GLT-STEER derives vectors from controlled linguistic transformation pairs, tests whether those vectors behave as output-form editors, and shows a bounded mechanism. Final-position markers (`?`, `!`, `...`) are steerable under no-steering, wrong-vector, negative-vector, random-vector, logit-level, and position-of-intervention controls; lexical or sentence-internal edits such as negation and modality fail or remain weak under the same recipe.

The main distinction from ActAdd/CAA-style steering is therefore the object being tested. ActAdd and CAA ask whether activation additions can steer broad behaviors or preferences. GLT-STEER asks whether linguistic transformation deltas behave as reusable intervention vectors and where that claim breaks.

Closest activation-steering references to cite in the final version:

- Turner et al. 2023/2024, *Steering Language Models With Activation Engineering* / Activation Addition, arXiv:2308.10248.
- Panickssery et al. 2024, *Steering Llama 2 via Contrastive Activation Addition*, arXiv:2312.06681 / ACL 2024.
- Zou et al. 2023, *Representation Engineering: A Top-Down Approach to AI Transparency*, arXiv:2310.01405.

## 2.3 Statistical Reporting and Tuning Disclosure

The headline Track 1 / GLT-STEER rates now have an explicit derived CI audit:

- `../../../results/experiments/glt_steer_headline_ci_20260825_results/`

The audit reports Wilson 95% confidence intervals and sample sizes for question, exclamation, ellipsis, final-marker logit, position-of-intervention, and marker-composition headline rows. Composition cells are small (`N=40` per prompt-style/control row), so composition remains descriptive. The final-marker logit and position audits have larger aggregate cells (`N=96-144` for logit marker rows; `N=480` for position rows), so they are the strongest current statistical support for the Final Marker Hypothesis.

DistilGPT-2 should be reported with tuning caution. The `layer=2, gain=1.0` setting was selected from the DistilGPT-2 layer/gain sweep before the hard out-of-template audit was interpreted, but this was not a preregistered train/dev/test protocol. The hard-OOT sources are structurally distinct from the simple copy-prompt setting, yet prompt families and evaluation metrics were already known from earlier GPT-2 experiments. Therefore DistilGPT-2 is evidence for marker-form transfer with weak preservation, not a clean held-out semantic-editing replication.

## 2.4 Confirmatory Fixed-Parameter Audit

A fixed-parameter confirmatory audit reruns question, exclamation, and ellipsis steering on fresh hard-heldout sources without any layer or gain search inside the run.

Result folder:

- `../../../results/experiments/glt_steer_confirmatory_fixed_params_20260825_results/`

Configuration:

- GPT-2: layers `2,3`, gain `0.75`
- DistilGPT-2: layer `2`, gain `1.0`
- Heldout sources: `48`
- Raw generations: `6480`
- Controls: `none`, `wrong_marker`, `random_norm`, `negative_target`

Headline aggregate:

| model | target | target marker rate | max control marker | target marker+preserved | max control marker+preserved |
|---|---|---:|---:|---:|---:|
| `gpt2` | `question` | `0.6562` (`N=288`, CI `[0.600, 0.709]`) | `0.0000` | `0.2396` (`N=288`, CI `[0.194, 0.292]`) | `0.0000` |
| `gpt2` | `exclamation` | `0.6875` (`N=288`, CI `[0.632, 0.738]`) | `0.0000` | `0.3958` (`N=288`, CI `[0.341, 0.453]`) | `0.0000` |
| `gpt2` | `ellipsis` | `0.8438` (`N=288`, CI `[0.797, 0.881]`) | `0.0000` | `0.3854` (`N=288`, CI `[0.331, 0.443]`) | `0.0000` |
| `distilgpt2` | `question` | `0.6319` (`N=144`, CI `[0.551, 0.706]`) | `0.0069` | `0.0556` (`N=144`, CI `[0.028, 0.106]`) | `0.0000` |
| `distilgpt2` | `exclamation` | `0.8958` (`N=144`, CI `[0.835, 0.936]`) | `0.0139` | `0.3472` (`N=144`, CI `[0.274, 0.428]`) | `0.0000` |
| `distilgpt2` | `ellipsis` | `0.8333` (`N=144`, CI `[0.764, 0.885]`) | `0.0000` | `0.1597` (`N=144`, CI `[0.109, 0.228]`) | `0.0000` |

Interpretation:

The fixed-parameter audit supports the no-new-tuning version of the Final Marker Hypothesis. Target steering remains well separated from matched controls across all three final markers and both tested models. Strict marker-plus-content preservation is positive but modest, especially for DistilGPT-2 question steering, so this strengthens the form-steering claim rather than a semantic-editing claim.

## 3. Main Question-Steering Result

The focused GPT-2 question run completed `6800` generations with no failures.

Key result:

| condition | question mark rate |
|---|---:|
| target question vector | `0.9350` |
| random-norm control | `0.0000` |
| wrong-class control | `0.0000` |
| negative-target control | `0.0000` |

Result folder:

- `../../../results/experiments/gpt2_question_activation_steering_focused_20260714_results/`

The prompt-only concern was tested explicitly. Copy-like prompts without steering produce `0.0000` question marks across `960` no-steering rows from GPT-2 and DistilGPT-2.

Result folder:

- `../../../results/experiments/question_copy_prompt_none_baseline_20260716_results/`

## 4. Content Preservation

Question marks alone are not enough: the model could simply emit question-like fragments while losing the source sentence. The copy-prompt preservation audit tests whether the output both contains a question marker and preserves source content.

Headline GPT-2 result:

| source set | prompt family | target question-and-preserved |
|---|---|---:|
| in-template | copy-like prompts | `0.9625-0.9750` |
| simple out-of-template | copy-like prompts | `0.8250-0.9000` |

Result folder:

- `../../../results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/`

Interpretation:

The result is stronger than punctuation insertion, but still prompt-conditioned. The cleanest behavior appears when the prompt already asks the model to repeat or copy the source sentence.

## 5. Hard Out-of-Template Generalization

A hard out-of-template audit used structurally diverse sentences with passive clauses, subordinate clauses, proper names, and numeric/time expressions.

GPT-2:

| prompt style | target question mark | best matched control | strict question-and-preserved |
|---|---:|---:|---:|
| `repeat_sentence` | `0.7125` | `0.0375` | lower than simple OOT |
| `copy_sentence` | `0.7500` | `0.0375` | lower than simple OOT |
| `same_sentence` | `0.7500` | `0.0375` | `0.5125` |

Result folder:

- `../../../results/experiments/gpt2_question_hard_oot_copy_prompt_steering_20260716_results/`

DistilGPT-2, tuned setting (`layer=2`, `gain=1.0`):

| prompt style | target question mark | best matched control question mark | strict question-and-preserved |
|---|---:|---:|---:|
| `repeat_sentence` | `0.725` | `0.000` | `0.150` |
| `copy_sentence` | `0.750` | `0.000` | `0.025` |
| `same_sentence` | `0.800` | `0.000` | `0.225` |

Result folder:

- `../../../results/experiments/distilgpt2_question_hard_oot_best_layer2_gain10_20260801_results/`

Figure:

![DistilGPT-2 hard-OOT boundary](../../figures/glt_steer_distilgpt2_hard_oot_boundary.png)

Interpretation:

The hard-OOT result separates form steering from semantic editing. DistilGPT-2 preserves the marker effect cleanly, but content preservation is much weaker than in GPT-2 and much weaker than in the simple out-of-template setting. The `copy_sentence` strict joint row is `0.025`, which is effectively near-null. Therefore DistilGPT-2 should be reported as marker-form replication only, not as hard-OOT semantic-preservation replication.

## 6. Why Negation Fails Under This Recipe

The first non-question extension to negation is weak/negative under copy-like prompts.

Result folders:

- `../../../results/experiments/gpt2_negation_copy_prompt_steering_20260716_results/`
- `../../../results/experiments/gpt2_negation_copy_prompt_layer_sweep_20260716_results/`

The best negation target-and-preserved row in the layer sweep reaches only `0.1729`, while controls remain nontrivial, with maxima up to `0.1229`.

A delta-coherence diagnostic explains part of this boundary:

| layer | question mean pairwise cosine | negation mean pairwise cosine |
|---:|---:|---:|
| `2` | `0.9693` | `0.5621` |
| `3` | `0.9365` | `0.5665` |

Result folder:

- `../../../results/experiments/gpt2_steering_delta_coherence_20260716_results/`

Interpretation:

Question deltas are much more internally coherent than negation deltas. In addition, question formation has a final surface marker (`?`) that an autoregressive model can append at the end of generation, while negation often requires inserting or restructuring sentence-internal tokens such as `not`, `never`, or auxiliary forms.

## 7. Final-Marker Controls

To test whether the question result is question-specific or part of a broader final-marker phenomenon, we ran final-marker controls.

Figure:

![Final-marker controls](../../figures/glt_steer_final_marker_controls.png)

Exclamation:

| source set | exclamation-and-preserved |
|---|---:|
| in-template | `1.0000` |
| hard out-of-template | `0.8000` |

Result folder:

- `../../../results/experiments/gpt2_exclamation_copy_prompt_steering_20260716_results/`

Ellipsis:

| prompt style | target ellipsis rate | ellipsis-and-preserved | best non-target ellipsis rate |
|---|---:|---:|---:|
| `repeat_sentence` | `0.925` | `0.525` | `0.000` |
| `copy_sentence` | `0.925` | `0.475` | `0.000` |
| `same_sentence` | `0.950` | `0.575` | `0.000` |

Result folder:

- `../../../results/experiments/gpt2_ellipsis_hard_oot_layer2_20260801_results/`

Interpretation:

The question result is not isolated. Final surface markers are particularly steerable. This strengthens the intervention result while narrowing its interpretation: current GLT-STEER works best for output-form edits, not arbitrary semantic rewrites.

## 8. Logit-Level Final-Marker Audit

The final-marker logit audit checks whether target steering changes the model's next-token distribution during generation.

Result folder:

- `../../../results/experiments/gpt2_final_marker_logit_audit_layer2_3_20260810_v2_results/`

Aggregate summary:

| target | control | marker rate (N, 95% CI) | mean best marker prob | median best rank | top-1-any rate |
|---|---|---:|---:|---:|---:|
| `?` | `none` | `0.0000` (`N=96`, `[0.0000, 0.0385]`) | `0.0015` | `28.0` | `0.0000` |
| `?` | `target` | `0.8542` (`N=96`, `[0.7700, 0.9111]`) | `0.7537` | `1.0` | `0.8542` |
| `!` | `none` | `0.0000` (`N=96`, `[0.0000, 0.0385]`) | `0.0010` | `20.0` | `0.0000` |
| `!` | `target` | `0.9063` (`N=96`, `[0.8313, 0.9499]`) | `0.7718` | `1.0` | `0.9063` |
| `...` | `none` | `0.0000` (`N=96`, `[0.0000, 0.0385]`) | `0.0016` | `11.75` | `0.0000` |
| `...` | `target` | `0.8750` (`N=96`, `[0.7941, 0.9270]`) | `0.7403` | `1.0` | `0.8750` |

Interpretation:

This result supports the Final Marker Hypothesis at the logit level. Copy-like prompts alone do not emit these markers under the no-steering condition. Target steering repeatedly moves the intended marker into the top logit rank during generation.

DistilGPT-2 transfer audit:

- `../../../results/experiments/distilgpt2_final_marker_logit_audit_l1_2_3_gain10_20260821_results/`

| target | control | marker rate (N, 95% CI) | max marker rate | mean best marker prob | median best rank |
|---|---|---:|---:|---:|---:|
| `?` | `none` | `0.0000` (`N=144`, `[0.0000, 0.0260]`) | `0.0000` | `0.0002` | `32.0` |
| `?` | `target` | `0.2986` (`N=144`, `[0.2299, 0.3778]`) | `0.7500` | `0.2876` | `2.0` |
| `!` | `none` | `0.0000` (`N=144`, `[0.0000, 0.0260]`) | `0.0000` | `0.0004` | `28.75` |
| `!` | `target` | `0.7778` (`N=144`, `[0.7032, 0.8380]`) | `1.0000` | `0.7344` | `1.0` |
| `...` | `none` | `0.0000` (`N=144`, `[0.0000, 0.0260]`) | `0.0000` | `0.0007` | `11.75` |
| `...` | `target` | `0.5139` (`N=144`, `[0.4330, 0.5941]`) | `1.0000` | `0.4271` | `1.5` |

This is a positive but model-dependent transfer result. DistilGPT-2 preserves the prompt-only control result (`none = 0.0000`) and shows target-marker steering, but it does not match GPT-2 uniformly. Exclamation transfers most cleanly, ellipsis is strongest at layer `2`, and question remains weaker.

## 9. Position-of-Intervention Audit

The position audit tests whether the question vector works only because the current hook edits the last token during every generation step.

Result folder:

- `../../../results/experiments/gpt2_question_position_intervention_audit_layer2_3_20260821_results/`

Aggregate summary:

| condition | position mode | question rate | question+preserved |
|---|---|---:|---:|
| `none` | `none` | `0.0000` | `0.0000` |
| `target_last_each_step` | `last_each_step` | `0.9604` | `0.7917` |
| `target_prompt_all_once` | `prompt_all` | `0.8625` | `0.7896` |
| `target_prompt_first` | `prompt_first` | `0.0000` | `0.0000` |
| `target_prompt_middle` | `prompt_middle` | `0.0000` | `0.0000` |
| `target_prompt_last_once` | `prompt_last` | `0.0000` | `0.0000` |
| `random_last_each_step` | `last_each_step` | `0.0000` | `0.0000` |
| `wrong_last_each_step` | `last_each_step` | `0.0000` | `0.0000` |
| `negative_last_each_step` | `last_each_step` | `0.0000` | `0.0000` |

Interpretation:

Single-position prompt edits do not produce question-form outputs. Editing all prompt tokens once is strong and nearly matches repeated last-token steering on the joint metric. This suggests that the current effect is not merely a special property of the last prompt token; it can also be induced by a distributed prompt-state edit.

CI note: the aggregate position rows use `N=480` outputs per condition. The Wilson 95% CI is `[0.9390, 0.9745]` for `target_last_each_step` question rate and `[0.8288, 0.8904]` for `target_prompt_all_once` question rate. All single-position prompt edits have question-rate CI upper bound `0.0079`.

## 10. Marker Composition

The first composition steering test compares:

```text
A = question vector
B = exclamation vector
A+B = vector sum at one layer
A then B = A at layer 2, B at layer 3
B then A = B at layer 2, A at layer 3
```

Result folder:

- `../../../results/experiments/gpt2_question_exclamation_marker_composition_layer2_3_20260801_results/`

Same-sentence prompt summary:

| control | question rate | exclamation rate | both-marker rate | preserved rate |
|---|---:|---:|---:|---:|
| `none` | `0.000` | `0.000` | `0.000` | `0.800` |
| `A only` | `0.900` | `0.000` | `0.000` | `0.750` |
| `B only` | `0.000` | `0.950` | `0.000` | `0.800` |
| `A+B` | `0.750` | `0.225` | `0.125` | `0.450` |
| `A then B` | `0.750` | `0.275` | `0.125` | `0.450` |
| `B then A` | `0.775` | `0.175` | `0.125` | `0.475` |
| `random sum` | `0.000` | `0.000` | `0.000` | `0.575` |

Order contrast:

| prompt style | exact `AB == BA` | marker profile `AB == BA` | `AB == A+B` | `BA == A+B` |
|---|---:|---:|---:|---:|
| `copy_sentence` | `0.200` | `0.650` | `0.150` | `0.200` |
| `repeat_sentence` | `0.300` | `0.725` | `0.150` | `0.225` |
| `same_sentence` | `0.200` | `0.650` | `0.100` | `0.150` |

Figure:

![Composition marker profile](../../figures/glt_steer_composition_marker_profile.png)

Interpretation:

The single vectors behave cleanly and the random-sum control produces no target markers. The summed and ordered interventions produce mixed marker profiles and lower content preservation. The exact generated text differs across orders, but marker-level profiles are often similar. Without a repeated-run or alternative null for exact-text variability, the exact-output difference should not be interpreted as evidence for noncommutative order structure.

The safe conclusion is narrower: final-marker vectors can compete and saturate under combined interventions. This is not evidence for a Lie algebra. A stronger algebraic intervention test would need transformations that can genuinely co-occur without competing for the same final punctuation slot and would need null controls for order sensitivity.

CI note: composition rows use `N=40` sources per prompt-style/control row. For the same-sentence prompt, `A only` has question-rate CI `[0.7695, 0.9604]`, `B only` has exclamation-rate CI `[0.8350, 0.9862]`, and `A+B` has both-marker-rate CI `[0.0546, 0.2611]`. Exact `AB == BA` rows are also `N=40`; their CIs are wide enough that the order table should remain descriptive.

## 11. Current Claim

The defensible current claim is:

> Transformation deltas learned from hidden-state differences can act as activation-space editors for final-position surface markers in GPT-style residual streams, with the cleanest evidence in GPT-2 and marker-dependent transfer to DistilGPT-2. The effect survives no-steering, wrong-vector, negative-vector, random-vector, logit-level, position-of-intervention, and fixed-parameter confirmatory controls. It does not yet extend to lexical or sentence-internal transformations under the same recipe, and it is not a general semantic editing method.

## 12. What Is Not Claimed

This draft does not claim:

- that GLT-STEER solves semantic rewriting;
- that negation can be steered cleanly by the current method;
- that final-marker steering proves general linguistic transformation editing;
- that the composition test proves noncommutative algebra or order-sensitive operator composition;
- that the result establishes a Lie algebra in transformer activations;
- that prompt wording is irrelevant.

## 13. Submission Scope

This draft should now converge as a short paper / extended abstract around the Final Marker Hypothesis rather than adding more controls before submission. The declared stopping-rule items are satisfied for the current scope:

1. The central claim is the bounded Final Marker Hypothesis.
2. Activation/representation-steering related work is incorporated.
3. Headline tables report `N` and Wilson 95% confidence intervals.
4. The DistilGPT-2 layer/gain tuning history is disclosed.
5. Marker composition is framed as competition/saturation, not algebraic order structure.
6. A fixed-parameter confirmatory audit has been added without new layer/gain search.

For submission purposes, no additional experiment is required unless an external reviewer or venue requirement asks for a specific control.

## 14. Future Work

Future work should be separated from the current short-paper scope:

1. Redesign non-final-marker steering before making stronger composition claims. A first `question + modality` run shows that the question vector remains strong, but the current modality vector produces no evidential markers under copy-like hard out-of-template prompts.
2. Move from single last-token injection to multi-token or token-position-aware intervention for sentence-internal transformations such as negation and modality.
3. For a future Lie-style intervention track, define transformations whose composition has a clear expected target string and compare `AB`, `BA`, and `A+B` against that target rather than only marker profiles.
4. Add a second seed or additional prompt family only as robustness evidence for a later version, not as a blocker for the current Track 1 submission.
