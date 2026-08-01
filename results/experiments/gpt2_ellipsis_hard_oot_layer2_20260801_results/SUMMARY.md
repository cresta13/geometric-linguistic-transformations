# GPT-2 Ellipsis Final-Marker Steering

Date: 2026-08-01

This run tests a second surface-ending operation beyond question formation and exclamation: steering `statement -> statement...` on structurally diverse hard out-of-template sources.

Configuration:

- Script: `scripts/run_gpt2_final_marker_copy_prompt_steering.py`
- Model: `gpt2`
- Target marker: `ellipsis`
- Target suffix: `...`
- Contrast marker: `question_mark`
- Contrast suffix: `?`
- Layer: `2`
- Gain: `0.75`
- Sources: `40` hard out-of-template declarative sentences
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Controls: `none`, `target_marker`, `wrong_contrast`, `random_norm`, `negative_target`
- Generations: `600`
- Failures: none

Summary:

| prompt style | target ellipsis rate | target ellipsis-and-preserved | best non-target ellipsis rate | wrong-contrast question rate |
|---|---:|---:|---:|---:|
| `repeat_sentence` | `0.925` | `0.525` | `0.000` | `0.875` |
| `copy_sentence` | `0.925` | `0.475` | `0.000` | `0.900` |
| `same_sentence` | `0.950` | `0.575` | `0.000` | `0.950` |

Interpretation:

The target ellipsis vector reliably induces the intended final marker on hard out-of-template sources, while no-steering, random-norm, and negative-target controls do not produce ellipses. The wrong-contrast vector selectively produces question marks rather than ellipses.

This supports the boundary interpretation of GLT-STEER: final surface markers are especially steerable. The result does not prove semantic editing, but it shows that the phenomenon is not question-specific.

Files:

- `csv/final_marker_copy_prompt_raw.csv`
- `csv/final_marker_copy_prompt_summary.csv`
- `csv/final_marker_copy_prompt_sources.csv`
- `csv/final_marker_training_pairs.csv`
- `run_status.json`
