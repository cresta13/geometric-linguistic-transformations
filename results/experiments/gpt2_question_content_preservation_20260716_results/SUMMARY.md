# GPT-2 Question Steering Content-Preservation Audit

## Purpose

This audit asks whether the GPT-2 question-steering result is only adding question marks, or whether it can add question form while preserving recognizable source content.

It reuses the prompt-robustness setting and adds content-preservation flags.

## Result Files

- `csv/question_content_preservation_raw.csv`
- `csv/question_content_preservation_summary.csv`
- `csv/question_content_preservation_contrast.csv`

## Main Finding

The question-steering vector still produces question marks, but content preservation is prompt-dependent.

Structured prompts preserve content much better than bare prompts:

| source set | prompt style | target qmark | target preserved-with-question |
|---|---:|---:|---:|
| in-template | input/output | `0.9625` | `0.7500` |
| in-template | sentence continuation | `0.9875` | `0.8500` |
| in-template | source/response | `0.9250` | `0.7375` |
| in-template | plain statement | `0.7750` | `0.4875` |
| out-of-template | input/output | `0.8375` | `0.5000` |
| out-of-template | sentence continuation | `0.9250` | `0.7000` |
| out-of-template | source/response | `0.8750` | `0.2125` |
| out-of-template | plain statement | `0.8125` | `0.0875` |

Controls did not produce preserved-with-question outputs in the contrast table.

## Interpretation

This result narrows the GLT-STEER claim.

The question vector is not only exploiting a high natural question-mark base rate, and it is not restricted to the original training templates. However, clean content-preserving rewriting is not automatic. Prompt format changes whether the model preserves the source sentence while adding question form.

The strongest cautious claim after this audit is:

> A GPT-2 question-transformation activation vector can steer output form toward questions, and under structured prompts it can often preserve recognizable source content. The effect is not yet robust general-purpose semantic editing.

## Caveats

- This is GPT-2-only and question-only.
- Content preservation is measured with lightweight lexical/source-token checks, not full semantic equivalence.
- Bare prompts and quoted prompts remain weak.
- This audit motivated the follow-up copy-prompt preservation run.
