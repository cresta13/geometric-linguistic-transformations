# GPT-2 Hard Out-of-Template Question Steering Summary

## Purpose

This audit tests whether the GLT-STEER question result survives a harder out-of-template setting.

The earlier out-of-template set used simple declarative sentences. This run uses structurally diverse sentences with:

- passive constructions;
- subordinate clauses;
- proper names;
- numbers, dates, durations, and measurements;
- multi-clause sentence structure.

## Run Configuration

- model: `gpt2`
- target class: `question`
- layers: `2, 3`
- gain: `0.75`
- prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- source set: hard out-of-template freeform only
- controls: `none`, `target`, `random_norm`, `wrong_class`, `negative_target`
- source rows: `40`
- generated rows: `1200`
- failures: none

## Main Finding

Question steering survives the harder out-of-template source set.

Target-vector rows:

| prompt style | question-mark rate | question-and-preserved | mean content preservation |
|---|---:|---:|---:|
| `copy_sentence` | `0.7250` | `0.3250` | `0.7152` |
| `repeat_sentence` | `0.7125` | `0.3375` | `0.7527` |
| `same_sentence` | `0.7500` | `0.5125` | `0.8427` |

Matched control maxima:

| prompt style | best control question-mark rate | best control question-and-preserved |
|---|---:|---:|
| `copy_sentence` | `0.0375` | `0.0000` |
| `repeat_sentence` | `0.0000` | `0.0000` |
| `same_sentence` | `0.0000` | `0.0000` |

## Important Metric Note

For question steering, `question_mark_rate` is the cleanest marker in this hard source set.

The generic `target_marker_rate` is noisy here because the broader question-marker list includes words such as `was` and `were`, which appear naturally in passive and past-tense hard out-of-template sentences. Therefore this summary emphasizes:

- `question_mark_rate`;
- `question_and_preserved_rate`.

The post-hoc CSV `csv/question_hard_oot_question_preservation_summary.csv` records this stricter question-mark preservation summary.

## Interpretation

This is a stronger generalization result than the earlier simple out-of-template control.

The question vector still creates question-form continuations on structurally diverse sentences, while no-steering and wrong-vector controls almost never produce question marks. Content preservation is weaker than in the simpler copy-prompt run, but the `same_sentence` prompt still reaches `0.5125` on the stricter joint question-and-preserved metric.

The careful conclusion is:

> GPT-2 question steering generalizes beyond simple declarative templates to structurally diverse out-of-template sentences, but content-preserving rewriting becomes harder as sentence structure becomes more complex.

## Caveats

- This remains GPT-2-only.
- The hard source set has `40` hand-written sentences.
- The preservation metric is lexical and conservative.
- This does not solve semantic editing; it shows a controlled behavior-level intervention with partial content preservation.
