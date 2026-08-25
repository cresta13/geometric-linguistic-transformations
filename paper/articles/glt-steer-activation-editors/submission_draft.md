# GLT-STEER: Transformation Deltas as Activation-Space Editors for Final Markers

Anna Simakova

Submission draft, 2026-08-25. Not peer reviewed.

## Abstract

This short draft studies whether linguistic transformation vectors learned from sentence-pair hidden-state differences can be injected back into a generative transformer as activation-space editors. In GPT-2, vectors derived from controlled transformation pairs reliably steer final-position surface markers such as `?`, `!`, and `...` under matched controls. The effect is visible in generated text, next-token marker logits, position-of-intervention audits, and a fixed-parameter confirmatory rerun on fresh hard-heldout sources. The claim is intentionally bounded: current GLT-STEER supports final-marker form steering, not general semantic rewriting and not a Lie-algebra claim.

## 1. Question

For a source sentence `x` and transformed sentence `y`, GLT computes a hidden-state displacement:

```text
delta = h(y) - h(x)
```

Earlier GLT tracks test whether these deltas classify transformation type offline. GLT-STEER asks a stronger intervention question:

```text
If a transformation delta is injected during generation, does the model produce the corresponding transformation marker?
```

This is a causal behavior-level test. It is still a narrow test: adding a final marker is easier for an autoregressive model than rewriting sentence-internal syntax or semantics.

## 2. Claim

The current defensible claim is:

> Transformation deltas learned from hidden-state differences can act as activation-space editors for final-position surface markers in GPT-style residual streams, with the cleanest evidence in GPT-2 and marker-dependent transfer to DistilGPT-2.

The claim is bounded by negative results:

- negation does not steer cleanly under the same recipe;
- question plus modality composition does not produce reliable modality markers;
- marker composition looks like competition/saturation, not algebraic order structure;
- DistilGPT-2 preserves marker form more reliably than full content.

## 3. Method

The core recipe is:

1. Build synthetic source/target pairs for a transformation class.
2. Extract hidden states at selected GPT-style residual blocks.
3. Compute a centroid transformation vector from training pairs.
4. During generation, add the vector to the residual stream.
5. Compare target steering against no-steering, wrong-marker, random-norm, and negative-vector controls.

Primary scripts:

- `scripts/run_gpt2_activation_steering_pilot.py`
- `scripts/run_gpt2_question_steering_controls.py`
- `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- `scripts/run_gpt2_exclamation_copy_prompt_steering.py`
- `scripts/run_gpt2_final_marker_copy_prompt_steering.py`
- `scripts/run_gpt2_final_marker_logit_audit.py`
- `scripts/run_gpt2_question_position_intervention_audit.py`
- `scripts/run_gpt2_marker_composition_steering.py`
- `scripts/run_glt_steer_confirmatory_fixed_params.py`
- `scripts/summarize_glt_steer_headline_ci.py`

## 4. Main Evidence

### 4.1 Question steering

The focused GPT-2 question run completed `6800` generations with no failures.

| condition | question mark rate |
|---|---:|
| target question vector | `0.9350` |
| random-norm control | `0.0000` |
| wrong-class control | `0.0000` |
| negative-target control | `0.0000` |

Result folder:

- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/`

Copy-like prompts without steering produce `0.0000` question marks across `960` no-steering rows from GPT-2 and DistilGPT-2:

- `results/experiments/question_copy_prompt_none_baseline_20260716_results/`

### 4.2 Content preservation

Question marks alone are not enough. The strongest copy-prompt setting measures whether the generated output both contains a question marker and preserves the source content.

| source set | prompt family | target question-and-preserved |
|---|---|---:|
| in-template | copy-like prompts | `0.9625-0.9750` |
| simple out-of-template | copy-like prompts | `0.8250-0.9000` |

Result folder:

- `results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/`

Hard out-of-template sources reduce preservation. GPT-2 still shows question-form steering; DistilGPT-2 is weaker and should be reported as marker-form transfer only.

### 4.3 Final-marker controls

The question result is not isolated. Exclamation and ellipsis are also steerable.

| marker | key result |
|---|---:|
| `!` | exclamation-and-preserved `1.0000` in-template, `0.8000` hard-OOT |
| `...` | ellipsis rate `0.925-0.950`, ellipsis-and-preserved `0.475-0.575` |

Result folders:

- `results/experiments/gpt2_exclamation_copy_prompt_steering_20260716_results/`
- `results/experiments/gpt2_ellipsis_hard_oot_layer2_20260801_results/`

### 4.4 Logit-level audit

The logit audit checks whether target steering changes the model's next-token distribution, not only the post-hoc generated string.

| target | control | marker rate (N, 95% CI) | median best rank |
|---|---|---:|---:|
| `?` | none | `0.0000` (`N=96`, `[0.0000, 0.0385]`) | `28.0` |
| `?` | target | `0.8542` (`N=96`, `[0.7700, 0.9111]`) | `1.0` |
| `!` | none | `0.0000` (`N=96`, `[0.0000, 0.0385]`) | `20.0` |
| `!` | target | `0.9063` (`N=96`, `[0.8313, 0.9499]`) | `1.0` |
| `...` | none | `0.0000` (`N=96`, `[0.0000, 0.0385]`) | `11.75` |
| `...` | target | `0.8750` (`N=96`, `[0.7941, 0.9270]`) | `1.0` |

Result folder:

- `results/experiments/gpt2_final_marker_logit_audit_layer2_3_20260810_v2_results/`

### 4.5 Position-of-intervention audit

Single prompt-token edits do not produce question outputs. Distributed prompt-state editing or repeated generation-step editing does.

| condition | position mode | question rate | question+preserved |
|---|---|---:|---:|
| none | none | `0.0000` | `0.0000` |
| target_last_each_step | last_each_step | `0.9604` | `0.7917` |
| target_prompt_all_once | prompt_all | `0.8625` | `0.7896` |
| target_prompt_first | prompt_first | `0.0000` | `0.0000` |
| target_prompt_middle | prompt_middle | `0.0000` | `0.0000` |
| target_prompt_last_once | prompt_last | `0.0000` | `0.0000` |

Result folder:

- `results/experiments/gpt2_question_position_intervention_audit_layer2_3_20260821_results/`

### 4.6 Fixed-parameter confirmatory audit

The confirmatory audit reruns question, exclamation, and ellipsis steering on fresh hard-heldout sources without any layer/gain search inside the run.

| model | target | target marker rate | max control marker | target marker+preserved | max control marker+preserved |
|---|---|---:|---:|---:|---:|
| `gpt2` | question | `0.6562` (`N=288`, CI `[0.600, 0.709]`) | `0.0000` | `0.2396` (`N=288`, CI `[0.194, 0.292]`) | `0.0000` |
| `gpt2` | exclamation | `0.6875` (`N=288`, CI `[0.632, 0.738]`) | `0.0000` | `0.3958` (`N=288`, CI `[0.341, 0.453]`) | `0.0000` |
| `gpt2` | ellipsis | `0.8438` (`N=288`, CI `[0.797, 0.881]`) | `0.0000` | `0.3854` (`N=288`, CI `[0.331, 0.443]`) | `0.0000` |
| `distilgpt2` | question | `0.6319` (`N=144`, CI `[0.551, 0.706]`) | `0.0069` | `0.0556` (`N=144`, CI `[0.028, 0.106]`) | `0.0000` |
| `distilgpt2` | exclamation | `0.8958` (`N=144`, CI `[0.835, 0.936]`) | `0.0139` | `0.3472` (`N=144`, CI `[0.274, 0.428]`) | `0.0000` |
| `distilgpt2` | ellipsis | `0.8333` (`N=144`, CI `[0.764, 0.885]`) | `0.0000` | `0.1597` (`N=144`, CI `[0.109, 0.228]`) | `0.0000` |

Result folder:

- `results/experiments/glt_steer_confirmatory_fixed_params_20260825_results/`

## 5. Boundary Results

### 5.1 Negation

Negation is weak under the same copy-prompt recipe. A layer sweep over GPT-2 layers `0-11` does not find a clean intervention site. The best target-and-preserved row reaches only `0.1729`, while matched controls remain nontrivial, with maxima up to `0.1229`.

The delta-coherence diagnostic gives one explanation: question deltas are much more internally coherent than negation deltas at the steering layers.

| layer | question mean pairwise cosine | negation mean pairwise cosine |
|---:|---:|---:|
| `2` | `0.9693` | `0.5621` |
| `3` | `0.9365` | `0.5665` |

### 5.2 Composition

Question/exclamation marker composition should not be interpreted as Lie-algebra evidence. Single vectors are clean; combined vectors produce mixed marker profiles and lower preservation. The safe conclusion is marker competition/saturation.

### 5.3 DistilGPT-2

DistilGPT-2 transfer is positive but model-, layer-, and marker-dependent. The tuned `layer=2`, `gain=1.0` setting was selected from a layer/gain sweep before the hard-OOT audit was interpreted. This was not a preregistered train/dev/test protocol and should remain a limitation.

## 6. Related Work

GLT-STEER is closest to activation-engineering and representation-steering work. Activation Addition / ActAdd computes activation differences from contrastive prompts and injects them at inference time to steer behavior. Contrastive Activation Addition averages residual-stream differences over positive and negative behavioral examples. Representation Engineering frames a broader family of methods for monitoring and manipulating model internals.

The contribution here is not that activation addition exists. The narrower contribution is diagnostic: GLT-STEER derives vectors from controlled linguistic transformation pairs and asks which linguistic transformations behave as reusable intervention vectors. The current answer is bounded but clear: final-position markers are steerable; sentence-internal transformations are not solved by this recipe.

Other related anchors:

- RISE studies geometric rotations for semantic-syntactic transformations across languages and embedding models.
- Task arithmetic and function-vector work are relevant to reusable direction-like computation.
- Linear Relational Decoding motivates matrix/operator extensions for future GLT-MOLT work.

## 7. Limitations

This draft does not claim:

- that GLT-STEER solves semantic rewriting;
- that negation can be steered cleanly by the current method;
- that final-marker steering proves general linguistic transformation editing;
- that marker composition proves noncommutative algebra;
- that the result establishes a Lie algebra in transformer activations;
- that prompt wording is irrelevant.

## 8. Reproducibility

All reported results are archived in `results/experiments/`. No new model inference is required to inspect the tables in this draft. The datasets are synthetic controlled sentence-pair templates generated by repository scripts.

Main draft source:

- `paper/articles/glt-steer-activation-editors/submission_draft.md`

Generated PDF:

- `reports/2026-08-25_glt_steer_submission_draft.pdf`

Build command:

```powershell
.\.venv\Scripts\python.exe scripts\build_glt_steer_submission_pdf.py
```

## References

- Turner et al. 2023/2024. Steering Language Models With Activation Engineering / Activation Addition. arXiv:2308.10248.
- Panickssery et al. 2024. Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681 / ACL 2024.
- Zou et al. 2023. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
- Freenor and Alvarez 2026. RISE: geometric rotations for semantic-syntactic transformations.
- Ilharco et al. 2023. Editing Models with Task Arithmetic.
- Todd et al. 2024. Function Vectors in Large Language Models.
