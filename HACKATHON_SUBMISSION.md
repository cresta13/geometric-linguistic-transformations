# OpenAI Build Week Submission Notes

This repository is the project submission package for **OpenAI Build Week 2026**.

## Project

**Geometric Linguistic Transformations**

Short description:

> A reproducible investigation into whether linguistic transformations form geometric directions inside transformers, and whether AI can make serious independent science accessible again.

Recommended Devpost category:

```text
Developer Tools
```

Secondary fit:

```text
Education
```

Use **Developer Tools** because the working artifact is a reproducible research workflow: scripts, controls, result tables, summaries, reports, and a lightweight judge demo. The educational message is important, but the submitted object is a testable developer/research toolchain.

## Repository URL

```text
https://github.com/cresta13/geometric-linguistic-transformations
```

## What Counts as the Working Project

This is not a separate Streamlit app or a synthetic hackathon-only product.

The working project is the GLT research repository itself:

- controlled sentence-pair templates;
- transformer embedding extraction and analysis scripts;
- endpoint, delta, concat, source-only, target-only, shuffle, norm, spectral, and hard-holdout controls;
- signed-composition and operator diagnostics;
- GPT-2 activation-steering experiments;
- committed result CSVs and summaries;
- draft papers, research notes, figures, reports, and Zenodo snapshots;
- a lightweight demo command for judges.

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

The clearest Build Week demo is the GPT-2 question-steering result:

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

## What Existed Before Build Week

Before Build Week, GLT already had several active research tracks:

- GLT-DV: delta-vector diagnostics with endpoint controls.
- GLT-SPOT: signed-permutation tests for ordered transformation composition.
- GLT-MOLT: matrix/operator diagnostics and null controls.
- GLT-XFER: cross-model transfer and alignment tests.
- GLT-AFFECT: affective-scale geometry controls.
- `paper/`, `reports/`, `results/`, `research/`, and `scripts/` as the main reproducibility structure.

## What Changed During Build Week

Build Week added the behavior-level steering track and the submission package:

- GLT-STEER GPT-2 question activation steering.
- Base-rate question-mark controls.
- Out-of-template generalization controls.
- Prompt-wrapper robustness controls.
- Content-preservation audits.
- A lightweight judge demo command.
- Build Week provenance and submission documentation.

Relevant commits since the Build Week period include:

```text
1de97bb | Add Build Week project story
6484607 | Document GPT-2 steering result
e1b90a9 | Add GLT steering demo path
360c719 | Add GPT-2 question steering controls
8231a61 | Add GPT-2 question prompt robustness runner
6b88916 | Add GPT-2 prompt robustness result
f466df1 | Add GPT-2 steering preservation audits
```

## How Codex and GPT-5.6 Were Used

Codex was the primary build environment. The human researcher set the direction, chose the claims, interpreted the evidence, and decided which results were honest enough to keep. Codex did most of the new code and repository work:

- writing and revising experiment scripts;
- running and monitoring long experiments;
- debugging failures and memory issues;
- organizing result folders;
- summarizing CSVs into readable evidence;
- adding reviewer-facing controls;
- updating README, paper drafts, and submission notes;
- keeping claims linked to scripts and result files.

GPT-5.6 was used as a reasoning partner:

- to revisit Lie-algebra and representation-geometry ideas;
- to compare related work;
- to challenge weak interpretations;
- to turn vague questions into concrete experiments;
- to explain unfamiliar methods in plain language.

AI did not make the scientific decisions. It made the choices understandable enough for the human researcher to make them.

## Human Story

Hi, I am Anna.

I am thirty-two, I have worked in IT for about fourteen years, and this research began because I decided to apply for a master's program in machine learning.

The university offers a scholarship, but first you have to demonstrate that you are capable of doing something real.

Then I remembered that, in 2012, I defended a university thesis connected with Lie algebras.

Fourteen years had passed. I had forgotten a great deal. Some ideas that were once familiar had become distant again.

So I asked GPT one simple question:

```text
Can Lie algebra be applied to transformers?
```

That question became this repository.

GLT is my example, not the destination. The larger message is that AI can help more people begin careful independent inquiry into questions that matter to them. Maybe the question is about language models. Maybe it is about a chronic condition, education, accessibility, climate, care work, or something no institution has had enough time to study yet.

AI does not remove the need for evidence, experts, safety, or humility. But it can help people read, compare, prototype, test, document, and keep moving.

If even a fraction of people spent two to four hours a week on a question they truly cared about, scientific progress could become much more distributed, and that could eventually make life better for many more people.

## Important Files for Judges

- `README.md`: main overview and quick demo command.
- `BUILD_WEEK_2026.md`: Build Week provenance and Codex/GPT-5.6 usage.
- `scripts/show_glt_steer_demo.py`: lightweight demo reader.
- `scripts/run_gpt2_activation_steering_pilot.py`: original activation-steering experiment runner.
- `scripts/run_gpt2_question_steering_controls.py`: base-rate and out-of-template controls.
- `scripts/run_gpt2_question_prompt_robustness.py`: prompt-wrapper robustness control.
- `scripts/analyze_gpt2_question_steering_content_preservation.py`: content-preservation audit.
- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md`: main steering result.
- `results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md`: reviewer controls.
- `results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md`: prompt robustness.
- `paper/research_program.md`: full research roadmap and scientific framing.
- `docs/hackathon-build/project-story.md`: plain-language human story.
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
Developer Tools
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

Main Build Week context is in:
- BUILD_WEEK_2026.md
- HACKATHON_SUBMISSION.md
- docs/hackathon-build/project-story.md

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

### 0:00-0:35 Personal Start

Hi, I am Anna.

I am thirty-two, and I have worked in IT for about fourteen years.

This project began when I decided to apply for a master's program in machine learning and remembered that my old university thesis, back in 2012, was connected with Lie algebras.

After fourteen years away from academic mathematics, I asked GPT a simple question:

```text
Can Lie algebra help us understand transformers?
```

### 0:35-1:00 Research Question

I began testing whether linguistic changes such as negation, question formation, tense, and modality create repeatable directions inside transformer embedding spaces.

Not only whether a model recognizes a word like "not", but whether the relationship between two sentences contains geometric information of its own.

### 1:00-1:30 Working Repository

This is a working research pipeline, not only a paper.

The repository contains scripts, result tables, figures, statistical analyses, research notes, draft papers, reports, and reproducible demo commands.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

### 1:30-2:00 Steering Result

The clearest Build Week result is GPT-2 question steering.

Without steering, GPT-2 repeats the statement. With the question-transformation vector added inside the model, it starts producing question marks. Random, wrong-class, and negative controls do not show the same effect.

This does not prove a complete linguistic algebra. It is one controlled behavior-level result in a larger research program.

### 2:00-2:30 Codex and GPT-5.6

Codex helped build the research workflow end to end: experiment scripts, controls, debugging, result summaries, documentation, and reproducibility checks.

GPT-5.6 helped me return to mathematical ideas I had forgotten, read related work critically, challenge my interpretations, and turn vague questions into testable experiments.

AI did not make the scientific decisions for me. It made it possible for me to understand the choices well enough to make them.

### 2:30-3:00 Broader Message

GLT is my example, not the destination.

The real point is that more people can begin serious, careful inquiry now. You do not necessarily need fifteen uninterrupted years at a university or a professor's title to begin asking a scientific question.

You need curiosity, honesty, and the willingness to test whether your beautiful idea survives contact with reality.

## Current Submission Checklist

- [x] Public repository.
- [x] README with quick demo.
- [x] Requirements file.
- [x] Working demo command.
- [x] Committed result CSVs for the main steering demo.
- [x] Main steering result summary.
- [x] Base-rate and out-of-template controls.
- [x] Prompt-robustness controls.
- [x] Build Week provenance file.
- [x] Devpost field draft.
- [ ] Demo video under 3 minutes.
- [ ] `/feedback` session ID copied from Codex/ChatGPT UI.
- [ ] Final Devpost submission form filled.
