# Demo: A Transformation Vector That Changes GPT-2 Output

## What This Demo Shows

This is the clearest short demo for GLT Lab.

We take a simple idea:

```text
statement -> question
```

Then we ask:

> If this transformation has a direction inside the model, can we add that direction back into GPT-2 while it is writing?

In the focused GPT-2 question-steering run, the answer is yes for question form.

## The Result in Plain Language

Without steering, GPT-2 mostly repeats a statement.

With the question-transformation vector added inside the model, it starts producing question marks.

With random, wrong-class, and negative vectors, it does not.

That makes this a good public demo because the effect is visible without needing to understand PCA, classifiers, Procrustes alignment, or matrix commutators.

## Demo Numbers

Focused run:

- Result folder: `results/experiments/gpt2_question_activation_steering_focused_20260714_results/`
- Model: `gpt2`
- Rows: `6800`
- Failures: none
- Best setting: layer `2`, gain `0.75`

Best setting:

| condition | question mark rate |
|---|---:|
| target question vector | `0.9375` |
| random-norm control | `0.0000` |
| wrong-class control | `0.0000` |
| negative-target control | `0.0000` |

Across all tested layers at gain `0.75`, target steering produced question marks in `0.9350` of generations.

## Demo Example

Source:

```text
The robot opened the portal.
```

No steering:

```text
The robot opened the portal. The robot opened the portal. The portal opened...
```

Target question steering:

```text
The robot opened the portal. The robot opened the portal? The robot opened the portal?
```

## Why This Fits the Hackathon Story

GLT Lab is about making research understandable and inspectable.

This demo shows the whole idea in one minute:

1. start with a research question;
2. build a controlled experiment;
3. compare against controls;
4. report the result with caveats;
5. keep scripts, CSVs, and summaries connected.

## Caveat to Say Out Loud

This does not mean we solved semantic editing. The model often repeats text, and the result currently shows question-form steering more than full sentence rewriting.

That honesty is part of the point: GLT Lab teaches people how to inspect evidence, not just how to celebrate positive results.

