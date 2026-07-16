# GPT-2 Negation Copy-Prompt Steering Summary

## Purpose

This run tests whether the GLT-STEER copy-prompt result extends beyond question formation to negation.

The motivating question:

> If the question vector can steer GPT-2 toward question form, can a negation vector similarly steer copied source sentences toward negated form?

## Run Configuration

- model: `gpt2`
- target class: `negation`
- layers: `2, 3`
- gain: `0.75`
- prompt styles: `repeat_sentence`, `copy_sentence`, `same_sentence`
- source sets: in-template test prompts and out-of-template freeform declaratives
- controls: `none`, `target`, `random_norm`, `wrong_class`, `negative_target`
- rows: `2400`
- failures: none

## Main Finding

The negation steering result is weak and does not reproduce the strong question-steering pattern.

Target-vector rows:

| source set | prompt style | target marker rate | mean content preservation | target-and-preserved |
|---|---:|---:|---:|---:|
| in-template | copy_sentence | `0.0000` | `1.0000` | `0.0000` |
| in-template | repeat_sentence | `0.0375` | `0.9917` | `0.0375` |
| in-template | same_sentence | `0.0000` | `1.0000` | `0.0000` |
| out-of-template | copy_sentence | `0.0250` | `0.9583` | `0.0000` |
| out-of-template | repeat_sentence | `0.1375` | `0.9833` | `0.1125` |
| out-of-template | same_sentence | `0.0250` | `0.9917` | `0.0000` |

Control maxima by matched source/prompt cell:

| source set | prompt style | best control marker rate | best control target-and-preserved |
|---|---:|---:|---:|
| in-template | copy_sentence | `0.0500` | `0.0000` |
| in-template | repeat_sentence | `0.0500` | `0.0500` |
| in-template | same_sentence | `0.0000` | `0.0000` |
| out-of-template | copy_sentence | `0.1125` | `0.0375` |
| out-of-template | repeat_sentence | `0.1500` | `0.1375` |
| out-of-template | same_sentence | `0.0500` | `0.0125` |

## Interpretation

This is a negative or boundary result.

Unlike question steering, negation steering does not show a clean target-vs-control separation under the current copy-prompt setup. The best target row (`out-of-template`, `repeat_sentence`) reaches target-and-preserved rate `0.1125`, but the matched control maximum is `0.1375`.

The careful conclusion is:

> The current GLT-STEER method does not automatically transfer from question formation to negation. Negation likely needs different prompts, metrics, steering sites, or vector construction.

## Caveats

- Negation markers are harder to score lexically than question marks.
- The marker list may catch unrelated negation-like continuations.
- Copy-like prompts may be poorly matched to negation, because copying the source conflicts with inserting a new negation.
- This result should be treated as a design boundary, not as evidence that negation steering is impossible.
