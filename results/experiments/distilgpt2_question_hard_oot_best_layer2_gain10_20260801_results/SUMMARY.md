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

The generic `target_marker_rate` for `question` is intentionally broad and includes lexical markers such as `was`, `were`, and `did`. For this run, the interpretable result should use the strict `question_mark_rate` and `strict_question_and_preserved_rate` in `csv/distilgpt2_hard_oot_question_preservation_summary.csv`.

Strict hard-OOT summary:

| prompt style | target question mark | best matched control question mark | target strict question-and-preserved | best matched control strict question-and-preserved |
|---|---:|---:|---:|---:|
| `repeat_sentence` | `0.725` | `0.000` | `0.150` | `0.000` |
| `copy_sentence` | `0.750` | `0.000` | `0.025` | `0.000` |
| `same_sentence` | `0.800` | `0.000` | `0.225` | `0.000` |

Interpretation:

The tuned DistilGPT-2 setting generalizes the question-mark intervention to hard out-of-template sources: no matched control produces question marks, while the target vector reaches `0.725-0.800`. This is a real output-form result. It does not replicate GPT-2's hard out-of-template content preservation: strict joint rates are `0.025-0.225`, and the `copy_sentence` joint row is effectively near-null at `0.025`.

This is a useful boundary result. It supports the claim that the question vector steers output form in DistilGPT-2, but it should be reported separately from content preservation. For hard out-of-template sources, DistilGPT-2 is a marker-form replication only, not a semantic-preservation replication.

Files:

- `csv/transformation_copy_prompt_raw.csv`
- `csv/transformation_copy_prompt_summary.csv`
- `csv/distilgpt2_hard_oot_question_preservation_summary.csv`
- `run_status.json`
