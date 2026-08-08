# GPT-2 Question/Modality Composition Steering

Date: 2026-08-08

This run tests a cleaner composition pair than the previous question/exclamation marker test. The intended pair is:

```text
A = question steering
B = modality/evidentiality steering
```

These operations should be able to co-occur in principle: an output could contain both a question marker and an evidential marker such as `apparently`, `reportedly`, or `according to`.

Configuration:

- Script: `scripts/run_gpt2_question_modality_composition_steering.py`
- Model: `gpt2`
- A operator: `question`
- B operator: `modality`
- Early layer: `2`
- Late layer: `3`
- Gain: `0.75`
- Sources: `40` hard out-of-template declarative sentences
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Generations: `840`
- Failures: none

Controls:

- `none`
- `question_only_late`
- `modality_only_late`
- `question_plus_modality_late`
- `question_then_modality_layers`
- `modality_then_question_layers`
- `random_sum_late`

Main result:

The question vector works, but the modality vector does not produce the current modality markers at all.

| prompt style | control | question rate | modality rate | both-marker rate | question-and-preserved |
|---|---|---:|---:|---:|---:|
| `copy_sentence` | `question_only_late` | `0.975` | `0.000` | `0.000` | `0.575` |
| `copy_sentence` | `modality_only_late` | `0.000` | `0.000` | `0.000` | `0.000` |
| `copy_sentence` | `question_plus_modality_late` | `0.900` | `0.000` | `0.000` | `0.500` |
| `copy_sentence` | `question_then_modality_layers` | `0.900` | `0.000` | `0.000` | `0.525` |
| `copy_sentence` | `modality_then_question_layers` | `0.950` | `0.000` | `0.000` | `0.575` |
| `repeat_sentence` | `question_only_late` | `0.925` | `0.000` | `0.000` | `0.675` |
| `repeat_sentence` | `modality_only_late` | `0.000` | `0.000` | `0.000` | `0.000` |
| `repeat_sentence` | `question_plus_modality_late` | `0.900` | `0.000` | `0.000` | `0.650` |
| `same_sentence` | `question_only_late` | `0.975` | `0.000` | `0.000` | `0.775` |
| `same_sentence` | `modality_only_late` | `0.000` | `0.000` | `0.000` | `0.000` |
| `same_sentence` | `question_plus_modality_late` | `0.950` | `0.000` | `0.000` | `0.750` |

Order contrast:

| prompt style | exact `QM == MQ` | marker profile `QM == MQ` | `QM == Q+M` | `MQ == Q+M` |
|---|---:|---:|---:|---:|
| `copy_sentence` | `0.425` | `0.850` | `0.425` | `0.800` |
| `repeat_sentence` | `0.350` | `0.900` | `0.375` | `0.800` |
| `same_sentence` | `0.625` | `0.950` | `0.450` | `0.725` |

Interpretation:

This is a negative composition result for the current modality steering recipe. The question vector remains robust, but the modality vector does not create observable evidential/modality markers under this prompt family, layer pair, and gain. As a result, `question + modality`, `question then modality`, and `modality then question` mostly reduce to question steering.

The exact generated text still differs by order, but because modality markers are absent, this run should not be treated as a successful semantic composition test. It is a useful boundary: moving from final surface markers to lexical/evidential transformations requires redesigned prompts, metrics, intervention sites, or a different steering objective.

Files:

- `csv/question_modality_composition_raw.csv`
- `csv/question_modality_composition_summary.csv`
- `csv/question_modality_order_contrast_raw.csv`
- `csv/question_modality_order_contrast_summary.csv`
- `csv/question_modality_sources.csv`
- `csv/question_modality_training_pairs.csv`
- `run_status.json`
