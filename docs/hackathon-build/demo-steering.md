# Demo: GLT-STEER Question Activation Steering

The clearest Build Week demo is the GPT-2 question-steering experiment.

The question:

> If a question transformation has a direction inside the model, can adding that direction into GPT-2's residual stream make the model write question-like continuations?

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

The command reads committed CSVs and prints the key result. It does not download models or rerun heavy experiments.

## Main Observation

Without steering, GPT-2 mostly repeats a declarative statement.

With the question-transformation vector injected, GPT-2 starts producing question-mark continuations.

Random, wrong-class, and negative-vector controls do not show the same effect.

## Main Numbers

Focused GPT-2 question steering:

```text
target question vector question-mark rate: 0.9350
random_norm / wrong_class / negative_target: 0.0000
```

Base-rate control:

```text
no-steering question-mark base rate: 0.0000
```

Out-of-template control:

```text
target question vector: 0.8375
all compact controls: 0.0000
```

Copy-prompt preservation:

```text
question-and-preserved rate:
0.9625-0.9750 in-template
0.8250-0.9000 out-of-template
```

## Safe Interpretation

This is evidence that a question-transformation activation vector can steer GPT-2 toward question form, and that copy-like prompts can preserve source content while adding that form.

It is not evidence that semantic editing is solved. It is not proof of a complete linguistic algebra. It is a bounded behavior-level intervention result.
