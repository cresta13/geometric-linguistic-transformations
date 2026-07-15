# OpenAI Build Week Submission Notes

This repository is the project submission package for **OpenAI Build Week**.

## Project

**GLT Lab: Geometric Linguistic Transformations**

Short description:

> GLT Lab is a reproducible research project that tests whether linguistic transformations appear as reusable geometric objects in transformer embedding spaces. The current demo shows a GPT-2 question-transformation vector steering generation toward question-like output form under controlled activation injection.

Recommended Devpost category:

```text
Education
```

Secondary fit:

```text
Developer Tools
```

Use **Education** unless the submission strategy changes, because the repository is framed as a runnable research-learning artifact with clear scripts, controls, results, and caveats.

## Repository URL

```text
https://github.com/cresta13/geometric-linguistic-transformations
```

## Quick Judge Demo

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

This command does not download models or rerun heavy experiments. It reads committed CSV results and prints:

- the focused GPT-2 question-steering result;
- no-steering question-mark base rates;
- out-of-template generalization controls;
- prompt-robustness controls when available.

## Main Result to Show in Video

The core visible demo:

```text
Source: The cat sat on the mat.

No steering:
The cat sat on the mat. The cat sat on the mat...

Target question steering:
The cat sat on the mat. The cat sat on the mat? The cat sat on the mat?

Controls:
Random, wrong-class, and negative vectors do not produce the same question-mark effect.
```

Best concise numbers:

```text
Focused GPT-2 question steering:
target question vector question-mark rate: 0.9350
random_norm / wrong_class / negative_target: 0.0000

Base-rate control:
no-steering question-mark base rate: 0.0000

Out-of-template control:
target question vector: 0.8375
all compact controls: 0.0000

Prompt robustness:
target question vector stays strong across four prompt styles:
0.7750 to 0.9875 in-template
0.8125 to 0.9250 out-of-template
```

## Important Files for Judges

- `README.md`: main overview and quick demo command.
- `scripts/show_glt_steer_demo.py`: lightweight demo reader.
- `scripts/run_gpt2_activation_steering_pilot.py`: original activation-steering experiment runner.
- `scripts/run_gpt2_question_steering_controls.py`: base-rate and out-of-template controls.
- `scripts/run_gpt2_question_prompt_robustness.py`: prompt-wrapper robustness control.
- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md`: main steering result.
- `results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md`: reviewer controls.
- `results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md`: prompt robustness.
- `paper/research_program.md`: full research roadmap and scientific framing.
- `docs/hackathon-build/demo-steering.md`: short demo explanation.

## Devpost Fields

### Submitter Type

```text
Individual
```

### Country of Residence

Fill manually with the correct country of residence.

### Category

```text
Education
```

### Code Repository

```text
https://github.com/cresta13/geometric-linguistic-transformations
```

### Project Test Link / Instructions

Use this:

```text
Judges can inspect the committed demo results without rerunning heavy model experiments:

1. Clone the repository.
2. Install dependencies from requirements.txt.
3. Run:
   .\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py

The command reads committed CSV results and prints the main GPT-2 question-steering result, no-steering base-rate control, out-of-template generalization control, and prompt-robustness control.

Main result summaries are in:
- results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md
- results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md
- results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md
```

### Developer Tool / Installation Notes

Use this if Devpost asks for plugin/dev-tool instructions:

```text
This is a research-code repository, not a hosted web service.

Supported platform for the archived demo path:
- Windows PowerShell
- Python environment with dependencies from requirements.txt

Quick demo:
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py

The quick demo does not require downloading GPT-2 or rerunning heavy experiments. Full experiment scripts are included for reproducibility.
```

### /feedback Session ID

Devpost requires the Codex `/feedback` session ID for the session where most core functionality was built.

Do this manually in the Codex/ChatGPT thread used for the main build:

```text
/feedback
```

Copy the returned session ID into the Devpost field:

```text
/feedback Session ID where the majority of your project was worked on
```

I cannot reliably run this slash command from PowerShell; it must be issued in the Codex/ChatGPT UI.

## Demo Video Script Under 3 Minutes

### 0:00-0:20 Problem

I wanted to test whether linguistic transformations are visible as geometric directions inside transformer models.

The question is simple: if a statement becomes a question, is there a reusable question-direction inside the model?

### 0:20-0:45 What I Built

GLT Lab is a reproducible research repository for testing this kind of question. It links scripts, result CSVs, summaries, and caveats so the evidence can be inspected instead of just claimed.

### 0:45-1:30 Working Demo

Run:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

Show the output:

- target question vector produces question marks;
- no steering does not;
- random, wrong-class, and negative controls do not;
- out-of-template sentences still work;
- prompt robustness holds across several prompt wrappers.

### 1:30-2:15 How Codex and GPT-5.6 Were Used

Codex helped build the research workflow end to end:

- writing experiment scripts;
- debugging long-running runs;
- checking memory and process failures;
- turning raw CSV outputs into summaries;
- adding reviewer-facing controls;
- keeping claims conservative and linked to evidence.

### 2:15-2:45 What This Does and Does Not Claim

This does not prove a complete linguistic algebra and it does not solve semantic editing.

It does show that a question-transformation activation vector can steer GPT-2 output form under controlled intervention, and that the effect survives base-rate, out-of-template, and prompt-wrapper controls.

### 2:45-3:00 Close

GLT Lab is a working research artifact: a concrete example of using Codex to move from a mathematical question to controlled, inspectable AI experiments.

## Current Submission Checklist

- [x] Public repository.
- [x] README with quick demo.
- [x] Requirements file.
- [x] Working demo command.
- [x] Committed result CSVs.
- [x] Main steering result summary.
- [x] Base-rate and out-of-template controls.
- [x] Prompt-robustness controls.
- [x] Devpost field draft.
- [ ] Demo video under 3 minutes.
- [ ] `/feedback` session ID copied from Codex/ChatGPT UI.
- [ ] Final Devpost submission form filled.

