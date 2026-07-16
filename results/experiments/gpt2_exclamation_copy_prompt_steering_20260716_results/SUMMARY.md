# GPT-2 exclamation copy-prompt steering

This run tests whether the successful question-steering result is partly explained by the fact that question formation has a simple final surface marker (`?`).

## Setup

- Script: `scripts/run_gpt2_exclamation_copy_prompt_steering.py`
- Model: `gpt2`
- Transformation: declarative statement to exclamation-mark form
- Example: `The analyst completed the task.` -> `The analyst completed the task!`
- Layers: `2,3`
- Gain: `0.75`
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Controls: `none`, `target_exclamation`, `wrong_question`, `random_norm`, `negative_target`
- Raw generations: `2400`

## Main result

The final-marker transformation steers very cleanly.

Best target rows:

| source set | exclamation mark rate | exclamation-and-preserved rate |
|---|---:|---:|
| `in_template` | `1.0000` | `1.0000` |
| `hard_out_of_template` | `1.0000` | `0.8000` |

Maximum rates by control:

| control | max `!` rate | max `?` rate | max `!` and preserved rate |
|---|---:|---:|---:|
| `target_exclamation` | `1.0000` | `0.0000` | `1.0000` |
| `wrong_question` | `0.0000` | `0.9750` | `0.0000` |
| `random_norm` | `0.0250` | `0.0000` | `0.0250` |
| `negative_target` | `0.0000` | `0.0250` | `0.0000` |
| `none` | `0.0000` | `0.0000` | `0.0000` |

## Interpretation

This strongly supports the surface-marker explanation for part of the question-steering result. GPT-2 can be steered very reliably toward a final punctuation marker when the target transformation is `.` -> `!`, and the wrong-question vector selectively produces `?` rather than `!`.

This does not make the question result trivial, because the copy-prompt audits still show source preservation and prompt-only controls remain at zero. It does mean that transformations with easy final markers are much easier steering targets than transformations such as negation, which require changing sentence-internal structure.

## Files

- `csv/exclamation_copy_prompt_summary.csv`: aggregate target/control rates.
- `csv/exclamation_copy_prompt_raw.csv`: all generated outputs.
- `csv/exclamation_copy_prompt_sources.csv`: in-template and hard out-of-template source sentences.
- `csv/exclamation_training_pairs.csv`: synthetic statement/exclamation and statement/question training pairs.
- `run_status.json`: run metadata.
