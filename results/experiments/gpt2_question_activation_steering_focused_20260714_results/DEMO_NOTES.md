# Demo Notes: Question Steering

## One-Line Demo

Add a question-transformation vector inside GPT-2 and the model starts producing question marks; random and wrong-class controls do not.

## Plain-Language Explanation

GLT usually studies sentence embeddings after the model has already read text. This experiment asks a stronger question:

> Can the transformation vector do anything inside the model while it is generating?

For question formation, the answer is yes in a focused GPT-2 setting. When the question vector is injected into intermediate residual layers, GPT-2 starts repeating the sentence as a question. The same behavior does not appear for random vectors, wrong transformation vectors, or negative vectors.

## What to Show in a Video

1. Open the result folder.
2. Show `run_status.json`: finished, `6800/6800`, no failures.
3. Show the headline table from `SUMMARY.md`.
4. Show one before/after example.
5. Explain the caveat: this is form steering, not perfect semantic rewriting.

## Best Clip

Source:

```text
The robot opened the portal.
```

No steering:

```text
The robot opened the portal. The robot opened the portal. The portal opened...
```

Target steering:

```text
The robot opened the portal. The robot opened the portal? The robot opened the portal?
```

Controls:

```text
No question marks under random-norm, wrong-class, or negative-target controls.
```

## Why This Is Interesting

This makes GLT easier to understand. Instead of only saying that a classifier can recognize transformation vectors, we can show that one of those vectors changes model behavior.

## What Not to Claim

- Do not claim this proves a full linguistic algebra.
- Do not claim GPT-2 understands questions because of this vector.
- Do not claim semantic editing is solved.
- Do not claim all transformations steer equally well; the broad pilot suggests they do not.

## Strong Safe Claim

> This focused experiment shows that a GLT question vector can causally steer GPT-2 output form toward questions under residual-stream injection, with strong separation from random, wrong-class, and negative controls.

