# Copy-Prompt None-Baseline Audit

## Purpose

This audit answers a specific reviewer concern:

> If copy-like prompts already tell the model to repeat the source sentence, maybe the prompt alone causes question marks, and the steering vector is unnecessary.

To test this, we isolate the `none` / no-steering condition from the GPT-2 and DistilGPT-2 copy-prompt preservation runs.

## Inputs

- `results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/csv/question_copy_prompt_summary.csv`
- `results/experiments/distilgpt2_question_copy_prompt_preservation_20260716_results/csv/question_copy_prompt_summary.csv`

Copy-like prompt styles:

- `repeat_sentence`
- `copy_sentence`
- `same_sentence`

## Main Finding

The copy-like prompts alone do **not** produce question marks.

Across GPT-2 and DistilGPT-2, all no-steering copy-like rows have question-mark rate `0.0000`.

| model | source set | max none question-mark rate | mean none question-mark rate | rows |
|---|---:|---:|---:|---:|
| DistilGPT-2 | in-template | `0.0000` | `0.0000` | `240` |
| DistilGPT-2 | out-of-template | `0.0000` | `0.0000` | `240` |
| GPT-2 | in-template | `0.0000` | `0.0000` | `240` |
| GPT-2 | out-of-template | `0.0000` | `0.0000` | `240` |

Global summary:

```text
models: distilgpt2,gpt2
copy-like prompt styles: copy_sentence, repeat_sentence, same_sentence
total no-steering rows: 960
max none question-mark rate: 0.0000
mean none question-mark rate: 0.0000
```

## Interpretation

This closes the immediate prompt-only alternative for the copy-like prompt result.

Copy-like prompts encourage source preservation, but they do not themselves create question form under the tested setup. The question form appears when the question-transformation steering vector is injected.

The careful conclusion is:

> Copy-like prompts explain why source content can be preserved, but they do not explain the emergence of question marks. The question-form effect requires the steering vector in these runs.

## Caveats

- This audit is limited to GPT-2 and DistilGPT-2.
- It tests the prompt families used in the copy-prompt preservation runs, not every possible copy/repeat prompt.
- It does not prove semantic editing; it only rules out the simplest prompt-only explanation for the observed question marks.
