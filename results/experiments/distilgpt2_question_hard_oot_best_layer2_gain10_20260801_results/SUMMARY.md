# DistilGPT-2 Hard Out-of-Template Question Steering

Date: 2026-08-01

This run tests whether the tuned DistilGPT-2 question-steering setting from the layer/gain sweep generalizes to structurally diverse out-of-template sources.

Configuration:

- Script: `scripts/run_gpt2_transformation_copy_prompt_steering.py`
- Model: `distilgpt2`
- Transformation: `question`
- Layer: `2`
- Gain: `1.0`
- Sources: `40` hard out-of-template declarative sentences copied from the GPT-2 hard OOT audit
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Controls: `none`, `target`, `random_norm`, `wrong_class`, `negative_target`
- Generations: `600`
- Failures: none

Important scoring note:

The generic `target_marker_rate` for `question` is intentionally broad and includes lexical markers such as `was`, `were`, and `did`. For this run, the reviewer-facing result should use the strict `question_mark_rate` and `strict_question_and_preserved_rate` in `csv/distilgpt2_hard_oot_question_preservation_summary.csv`.

Strict hard-OOT summary:

| prompt style | target question mark | best matched control question mark | target strict question-and-preserved | best matched control strict question-and-preserved |
|---|---:|---:|---:|---:|
| `repeat_sentence` | `0.725` | `0.000` | `0.150` | `0.000` |
| `copy_sentence` | `0.750` | `0.000` | `0.025` | `0.000` |
| `same_sentence` | `0.800` | `0.000` | `0.225` | `0.000` |

Interpretation:

The tuned DistilGPT-2 setting generalizes the question-mark intervention to hard out-of-template sources: no matched control produces question marks, while the target vector reaches `0.725-0.800`. However, content preservation is much weaker than in the easier in-template and simple out-of-template settings. The strongest strict joint row is `0.225` for `same_sentence`.

This is a useful boundary result. It supports the claim that the question vector steers output form in DistilGPT-2, but it does not support strong hard-OOT semantic rewriting for DistilGPT-2 under the current copy-prompt recipe.

Files:

- `csv/transformation_copy_prompt_raw.csv`
- `csv/transformation_copy_prompt_summary.csv`
- `csv/distilgpt2_hard_oot_question_preservation_summary.csv`
- `run_status.json`
