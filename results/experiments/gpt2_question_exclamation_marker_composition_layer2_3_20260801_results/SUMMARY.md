# GPT-2 Question/Exclamation Marker Composition Steering

Date: 2026-08-02

This run tests whether two final-marker steering vectors compose like a simple additive intervention, and whether layer order changes the output profile.

Configuration:

- Script: `scripts/run_gpt2_marker_composition_steering.py`
- Model: `gpt2`
- A operator: `question_mark`, suffix `?`
- B operator: `exclamation`, suffix `!`
- Early layer: `2`
- Late layer: `3`
- Gain: `0.75`
- Sources: `40` hard out-of-template declarative sentences
- Prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- Generations: `840`
- Failures: none

Controls:

- `none`: no intervention
- `a_only_late`: question vector at late layer
- `b_only_late`: exclamation vector at late layer
- `a_plus_b_late`: sum of question and exclamation vectors at late layer
- `a_then_b_layers`: question vector at early layer, exclamation vector at late layer
- `b_then_a_layers`: exclamation vector at early layer, question vector at late layer
- `random_sum_late`: norm-matched random control for the summed vector

Same-sentence prompt summary:

| control | question rate | exclamation rate | both-marker rate | preserved rate |
|---|---:|---:|---:|---:|
| `none` | `0.000` | `0.000` | `0.000` | `0.800` |
| `a_only_late` | `0.900` | `0.000` | `0.000` | `0.750` |
| `b_only_late` | `0.000` | `0.950` | `0.000` | `0.800` |
| `a_plus_b_late` | `0.750` | `0.225` | `0.125` | `0.450` |
| `a_then_b_layers` | `0.750` | `0.275` | `0.125` | `0.450` |
| `b_then_a_layers` | `0.775` | `0.175` | `0.125` | `0.475` |
| `random_sum_late` | `0.000` | `0.000` | `0.000` | `0.575` |

Order contrast:

| prompt style | exact `AB == BA` | marker profile `AB == BA` | `AB == A+B` | `BA == A+B` |
|---|---:|---:|---:|---:|
| `copy_sentence` | `0.200` | `0.650` | `0.150` | `0.200` |
| `repeat_sentence` | `0.300` | `0.725` | `0.150` | `0.225` |
| `same_sentence` | `0.200` | `0.650` | `0.100` | `0.150` |

Interpretation:

Single-vector controls behave cleanly: the question vector produces question marks, the exclamation vector produces exclamation marks, and none/random controls produce neither marker. The summed and ordered two-vector interventions produce mixed marker profiles and lower content preservation.

The exact output is order-sensitive: `a_then_b_layers` and `b_then_a_layers` generate identical text in only `20-30%` of rows. At the coarser marker-profile level, the two orders agree more often (`65-72.5%`), so this is not a strong noncommutative algebra claim. It is a useful causal-intervention diagnostic showing that marker vectors interact nonlinearly under residual-stream steering.

Files:

- `csv/marker_composition_steering_raw.csv`
- `csv/marker_composition_steering_summary.csv`
- `csv/marker_composition_order_contrast_raw.csv`
- `csv/marker_composition_order_contrast_summary.csv`
- `csv/marker_composition_sources.csv`
- `csv/marker_composition_training_pairs.csv`
- `run_status.json`
