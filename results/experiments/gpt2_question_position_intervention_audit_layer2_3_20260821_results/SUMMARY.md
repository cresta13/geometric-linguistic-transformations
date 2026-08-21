# GPT-2 Question Position-of-Intervention Audit

Date: 2026-08-21

This run tests whether GPT-2 question steering depends specifically on the current `last token at every generation step` hook, or whether the same question vector can work when applied to different prompt-token positions.

Configuration:

- model: `gpt2`
- layers: `2,3`
- gain: `0.75`
- prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- sources: `40` in-template plus `40` hard out-of-template sources
- controls and position modes:
  - `none`
  - `target_prompt_first`
  - `target_prompt_middle`
  - `target_prompt_last_once`
  - `target_prompt_all_once`
  - `target_last_each_step`
  - `wrong_last_each_step`
  - `random_last_each_step`
  - `negative_last_each_step`
- rows: `4320`

Headline aggregate:

| condition | position mode | question rate | question+preserved | mean preservation |
|---|---|---:|---:|---:|
| `none` | `none` | `0.0000` | `0.0000` | `0.8779` |
| `target_last_each_step` | `last_each_step` | `0.9604` | `0.7917` | `0.8703` |
| `target_prompt_all_once` | `prompt_all` | `0.8625` | `0.7896` | `0.8487` |
| `target_prompt_first` | `prompt_first` | `0.0000` | `0.0000` | `0.8834` |
| `target_prompt_middle` | `prompt_middle` | `0.0000` | `0.0000` | `0.8860` |
| `target_prompt_last_once` | `prompt_last` | `0.0000` | `0.0000` | `0.9007` |
| `random_last_each_step` | `last_each_step` | `0.0000` | `0.0000` | `0.8403` |
| `wrong_last_each_step` | `last_each_step` | `0.0000` | `0.0000` | `0.8914` |
| `negative_last_each_step` | `last_each_step` | `0.0000` | `0.0000` | `0.7726` |

By source set:

| source set | condition | question rate | question+preserved |
|---|---|---:|---:|
| in-template | `target_last_each_step` | `0.9917` | `0.9708` |
| in-template | `target_prompt_all_once` | `0.9542` | `0.9333` |
| hard out-of-template | `target_last_each_step` | `0.9292` | `0.6125` |
| hard out-of-template | `target_prompt_all_once` | `0.7708` | `0.6458` |

Interpretation:

The effect is not reproduced by editing only the first, middle, or last prompt token once. Those three single-position prompt interventions produce `0.0000` question rate.

However, applying the vector once to all prompt tokens (`prompt_all`) is strong: it reaches aggregate question rate `0.8625`, close to the current `last_each_step` rate `0.9604`, and has nearly identical aggregate question-and-preserved rate (`0.7896` vs `0.7917`).

This supports a more precise mechanism: GLT-STEER is not simply a magic last-prompt-token effect. It can also work as a distributed prompt-state intervention, but the vector must affect the prompt representation broadly or be re-applied during generation. A single edited prompt position is insufficient.

Files:

- `csv/question_position_intervention_summary.csv`
- `csv/question_position_intervention_raw.csv`
- `csv/question_position_sources.csv`
- `run_status.json`
