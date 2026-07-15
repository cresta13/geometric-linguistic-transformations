# DistilGPT-2 Question Steering Copy-Prompt Preservation Summary

## Purpose

This run replicates the GPT-2 copy-prompt preservation audit on `distilgpt2`.

The goal is to test whether the strong GPT-2 question-steering result transfers to a second GPT-family model under the same copy-like prompt design.

## Run Configuration

- model: `distilgpt2`
- layers: `2, 3`
- gain: `0.75`
- prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`, `quoted_echo`
- source sets: in-template test prompts and out-of-template freeform declaratives
- controls: `none`, `target`, `random_norm`, `wrong_class`, `negative_target`
- rows: `3200`
- failures: none

## Main Finding

The DistilGPT-2 replication is positive but much weaker than GPT-2.

For copy-like prompts, the target question vector produces question marks and some content-preserved questions, while no-steering and wrong-vector controls remain at `0.0000` question-and-preserved rate.

| source set | prompt style | target qmark | mean content preservation | question-and-preserved |
|---|---:|---:|---:|---:|
| in-template | repeat_sentence | `0.4875` | `0.8625` | `0.3875` |
| in-template | same_sentence | `0.4750` | `0.9667` | `0.4625` |
| in-template | copy_sentence | `0.4625` | `0.6042` | `0.2750` |
| out-of-template | repeat_sentence | `0.4125` | `0.9042` | `0.3125` |
| out-of-template | same_sentence | `0.4375` | `0.9042` | `0.3875` |
| out-of-template | copy_sentence | `0.4875` | `0.3542` | `0.1125` |

The `quoted_echo` prompt remains a failure mode for preservation:

| source set | prompt style | target qmark | mean content preservation | question-and-preserved |
|---|---:|---:|---:|---:|
| in-template | quoted_echo | `0.4500` | `0.0042` | `0.0000` |
| out-of-template | quoted_echo | `0.2500` | `0.0042` | `0.0000` |

## Interpretation

This is a bounded replication.

The direction of the effect transfers: the target vector changes output form, and controls do not create preserved questions. However, the magnitude is far smaller than GPT-2. This suggests that GLT-STEER is not merely a one-off GPT-2 artifact, but the effect is strongly model-dependent.

The cleanest current statement is:

> GPT-2 shows strong copy-prompt question steering with preservation. DistilGPT-2 shows the same qualitative separation from controls, but much weaker question-and-preserved rates.

## Caveats

- This remains question-only.
- This is not robust semantic editing.
- The preservation metric is lexical and conservative.
- DistilGPT-2 substantially weakens the effect, so model dependence must be reported rather than hidden.
