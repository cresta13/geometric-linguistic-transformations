# GLT-STEER: Transformation Vectors as Activation-Space Editors

Status: short draft / intervention paper candidate.

This article candidate collects the behavior-level GLT-STEER results:

- Central claim: final-position surface markers (`?`, `!`, `...`) are reliably steerable via mean hidden-state delta vectors in GPT-2; lexical/sentence-internal transformations are not reliable under the same recipe.
- GPT-2 question activation steering with base-rate and prompt controls.
- Logit-level final-marker audit showing that target steering moves `?`, `!`, and `...` to rank `1` during generation while no-steering marker rates remain `0.0000`.
- Position-of-intervention audit showing that single prompt-token edits fail, while all-prompt-token editing nearly matches repeated last-token steering.
- DistilGPT-2 final-marker logit transfer audit showing positive but layer- and marker-dependent transfer.
- Derived headline CI audit with sample sizes for the main Track 1 / GLT-STEER tables.
- Fixed-parameter confirmatory audit for question, exclamation, and ellipsis steering on fresh hard-heldout sources.
- Copy-prompt content-preservation follow-ups.
- Hard out-of-template generalization.
- DistilGPT-2 layer/gain sensitivity and hard-OOT boundary.
- Non-question boundary results for negation.
- Final-marker controls for exclamation and ellipsis.
- First marker-composition steering diagnostic.

Drafts:

- `draft.md`
- `submission_draft.md`

Generated submission PDF:

- `../../../reports/2026-08-25_glt_steer_submission_draft.pdf`

Primary scripts:

- `scripts/run_gpt2_activation_steering_pilot.py`
- `scripts/run_gpt2_question_steering_controls.py`
- `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- `scripts/run_gpt2_exclamation_copy_prompt_steering.py`
- `scripts/run_gpt2_final_marker_copy_prompt_steering.py`
- `scripts/run_gpt2_marker_composition_steering.py`
- `scripts/run_glt_steer_confirmatory_fixed_params.py`
- `scripts/summarize_glt_steer_headline_ci.py`
- `scripts/build_glt_steer_submission_pdf.py`

Primary figures:

- `paper/figures/glt_steer_final_marker_controls.png`
- `paper/figures/glt_steer_distilgpt2_hard_oot_boundary.png`
- `paper/figures/glt_steer_composition_marker_profile.png`
