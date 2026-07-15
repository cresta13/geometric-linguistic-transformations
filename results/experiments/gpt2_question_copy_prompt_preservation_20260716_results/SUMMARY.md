# GPT-2 Question Steering Copy-Prompt Preservation Summary

## Purpose

This follow-up tests whether more explicit copy/repeat prompt formats improve the content-preservation weakness found in the first preservation audit.

The run uses:

- model: `gpt2`
- layers: `2, 3`
- gain: `0.75`
- prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`, `quoted_echo`
- source sets: in-template test prompts and out-of-template freeform declaratives
- controls: `none`, `target`, `random_norm`, `wrong_class`, `negative_target`
- rows: `3200`
- failures: none

## Main Finding

Copy-like prompts make the steering result much cleaner.

For the target question vector, the joint rate of "question mark present and source content preserved" is high for three prompt families:

| source set | prompt style | target qmark | mean content preservation | question-and-preserved |
|---|---:|---:|---:|---:|
| in-template | repeat_sentence | `0.9750` | `0.9958` | `0.9625` |
| in-template | same_sentence | `1.0000` | `0.9875` | `0.9750` |
| in-template | copy_sentence | `1.0000` | `0.9833` | `0.9750` |
| out-of-template | repeat_sentence | `0.9500` | `0.9667` | `0.9000` |
| out-of-template | same_sentence | `0.9625` | `0.9625` | `0.8625` |
| out-of-template | copy_sentence | `0.9375` | `0.9375` | `0.8250` |

The `quoted_echo` prompt is a failure mode for content preservation:

| source set | prompt style | target qmark | mean content preservation | question-and-preserved |
|---|---:|---:|---:|---:|
| in-template | quoted_echo | `0.8000` | `0.2250` | `0.0500` |
| out-of-template | quoted_echo | `0.7875` | `0.1458` | `0.0125` |

## Interpretation

This strengthens the steering result from pure punctuation steering toward partial form-preserving rewriting, but only under prompt formats that already encourage copying or repeating the source sentence.

The cleanest current GLT-STEER claim is:

> Under copy-like prompts, a GPT-2 question-transformation activation vector can often preserve the source content while steering the continuation toward question form. This effect is sharply separated from no-steering and wrong-vector controls on the joint question-and-preserved metric.

## Caveats

- This remains GPT-2-only and question-only.
- The preservation metric is lexical and conservative, not a full semantic equivalence judge.
- Copy prompts make the task easier by explicitly encouraging source retention.
- `quoted_echo` shows that prompt wording is a real boundary condition.
- This is still not evidence for a complete linguistic algebra or robust natural-language editing.
