# GLT-STEER: Transformation Vectors as Activation-Space Editors

Status: short draft / intervention paper candidate.

This article candidate collects the behavior-level GLT-STEER results:

- Central claim: final-position surface markers (`?`, `!`, `...`) are reliably steerable via mean hidden-state delta vectors in GPT-2; lexical/sentence-internal transformations are not reliable under the same recipe.
- GPT-2 question activation steering with base-rate and prompt controls.
- Copy-prompt content-preservation follow-ups.
- Hard out-of-template generalization.
- DistilGPT-2 layer/gain sensitivity and hard-OOT boundary.
- Non-question boundary results for negation.
- Final-marker controls for exclamation and ellipsis.
- First marker-composition steering diagnostic.

Main draft:

- `draft.md`

Primary scripts:

- `scripts/run_gpt2_activation_steering_pilot.py`
- `scripts/run_gpt2_question_steering_controls.py`
- `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- `scripts/run_gpt2_exclamation_copy_prompt_steering.py`
- `scripts/run_gpt2_final_marker_copy_prompt_steering.py`
- `scripts/run_gpt2_marker_composition_steering.py`

Primary figures:

- `paper/figures/glt_steer_final_marker_controls.png`
- `paper/figures/glt_steer_distilgpt2_hard_oot_boundary.png`
- `paper/figures/glt_steer_composition_marker_profile.png`
