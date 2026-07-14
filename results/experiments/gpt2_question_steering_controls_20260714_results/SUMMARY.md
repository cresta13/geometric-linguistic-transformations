# GPT-2 Question Steering Controls Summary

## Status

Completed successfully.

- Started: `2026-07-14 19:29:20 +0300`
- Finished: `2026-07-14 19:53:00 +0300`
- Model: `gpt2`
- Residual layers: `2`, `3`
- Gain: `0.75`
- Training rows for centroids: `400`
- Source prompts: `120`
  - in-template test sources: `80`
  - out-of-template freeform sources: `40`
- Raw generations: `1200`
- Failures: none

## Questions

This run closes two reviewer-facing controls for the first GLT-STEER result.

1. What is the base rate of question marks in GPT-2 without steering?
2. Does the question-steering effect hold on source sentences outside the synthetic template family used to train the centroid?

## Base-Rate Control

No-steering GPT-2 produced no question marks under the same prompt format.

| source set | base question mark rate | rows |
|---|---:|---:|
| in-template test | `0.0000` | `160` |
| out-of-template freeform | `0.0000` | `80` |

This addresses the concern that GPT-2 may naturally produce question marks after declarative prompts such as:

```text
Input: The oracle built the tower.
Output:
```

In this setup, it does not.

## Out-of-Template Generalization

The same question vector was learned from the synthetic UPAT-style training templates and then applied to hand-written freeform sentences such as:

```text
The cat sat on the mat.
The kettle whistled on the stove.
A student solved the puzzle.
```

Question-mark rates:

| source set | none | target | random_norm | wrong_class | negative_target |
|---|---:|---:|---:|---:|---:|
| in-template test | `0.0000` | `0.9625` | `0.0000` | `0.0000` | `0.0000` |
| out-of-template freeform | `0.0000` | `0.8375` | `0.0000` | `0.0000` | `0.0000` |

The effect weakens outside the template family but remains strong.

## Example

Layer `2`, gain `0.75`.

Source:

```text
The cat sat on the mat.
```

No steering:

```text
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat...
```

Target question steering:

```text
The cat sat on the mat. The cat sat on the mat? The cat sat on the mat?
```

Random-norm control:

```text
The cat sat on the mat. Output: The cat sat on the mat...
```

Wrong-class control:

```text
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat...
```

Negative-target control:

```text
The cat sat on the mat. Output: The cat sat on the mat...
```

## Interpretation

The base-rate result supports the original steering finding: question marks are not appearing merely because GPT-2 often asks questions under this prompt.

The out-of-template result is stronger: a question-transformation vector learned from controlled templates transfers to novel freeform declarative sentences.

Safe claim:

> A GPT-2 question-transformation activation vector can steer both in-template and out-of-template declarative prompts toward question-mark generation, while no-steering, random-norm, wrong-class, and negative-vector controls produce no question marks under the same compact control setup.

## Caveats

- This remains question-form steering, not full semantic rewriting.
- Some target-marker counts are inflated by loose lexical markers such as `was`; question-mark rate is the cleaner metric here.
- The out-of-template set is hand-written and modest in size (`40` source sentences).
- This control uses the strongest focused setting family: layers `2` and `3`, gain `0.75`.
- Other transformation classes still need their own base-rate and out-of-template controls.

## Files

- Script: `scripts/run_gpt2_question_steering_controls.py`
- Status: `run_status.json`
- Sources: `csv/question_steering_control_sources.csv`
- Base-rate table: `csv/question_mark_base_rate.csv`
- Raw generations: `csv/question_steering_controls_raw.csv`
- Summary table: `csv/question_steering_controls_summary.csv`

