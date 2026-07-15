# GPT-2 Question Prompt Robustness Summary

## Status

Completed successfully.

- Started: `2026-07-15 12:08:32 +0300`
- Finished: `2026-07-15 13:10:36 +0300`
- Model: `gpt2`
- Residual layers: `2`, `3`
- Gain: `0.75`
- Prompt styles: `input_output`, `source_response`, `sentence_continuation`, `plain_statement`
- Source prompts: `80`
  - in-template test sources: `40`
  - out-of-template freeform sources: `40`
- Raw generations: `3200`
- Failures: none

## Question

Does the GPT-2 question-steering effect depend on the exact prompt wrapper `Input: ... Output:`, or does it survive across different neutral prompt formats?

## Main Result

The question-steering effect survives all four tested prompt styles.

Question-mark rates:

| source set | prompt style | none | target | random_norm | wrong_class | negative_target |
|---|---|---:|---:|---:|---:|---:|
| in-template test | `input_output` | `0.0000` | `0.9625` | `0.0000` | `0.0000` | `0.0000` |
| in-template test | `source_response` | `0.0000` | `0.9250` | `0.0000` | `0.0000` | `0.0000` |
| in-template test | `sentence_continuation` | `0.0000` | `0.9875` | `0.0000` | `0.0000` | `0.0000` |
| in-template test | `plain_statement` | `0.0000` | `0.7750` | `0.0250` | `0.0000` | `0.0000` |
| out-of-template freeform | `input_output` | `0.0000` | `0.8375` | `0.0000` | `0.0000` | `0.0000` |
| out-of-template freeform | `source_response` | `0.0000` | `0.8750` | `0.0000` | `0.0000` | `0.0000` |
| out-of-template freeform | `sentence_continuation` | `0.0000` | `0.9250` | `0.0000` | `0.0000` | `0.0000` |
| out-of-template freeform | `plain_statement` | `0.0000` | `0.8125` | `0.0000` | `0.0000` | `0.0000` |

## Interpretation

This control strengthens GLT-STEER.

The question-vector effect is not just an artifact of one prompt format. It holds for:

- the original `Input: ... Output:` wrapper;
- a `Source: ... Response:` wrapper;
- a `Sentence: ... Continuation:` wrapper;
- a bare plain-statement prompt.

It also holds for both in-template and out-of-template source sentences.

Safe claim:

> The GPT-2 question activation-steering effect survives prompt-wrapper variation and out-of-template source sentences, with no-steering and wrong-vector controls remaining near zero on the clean question-mark metric.

## Caveats

- This is still question-form steering, not full semantic rewriting.
- `plain_statement` is a harder and noisier generation setting, but the target effect remains large.
- The marker metrics other than `question_mark_hit` are noisy; the clean metric is literal question-mark generation.
- This remains GPT-2-only and question-only until replicated on more transformations and models.

## Files

- Script: `scripts/run_gpt2_question_prompt_robustness.py`
- Status: `run_status.json`
- Sources: `csv/question_prompt_robustness_sources.csv`
- Raw generations: `csv/question_prompt_robustness_raw.csv`
- Summary table: `csv/question_prompt_robustness_summary.csv`

