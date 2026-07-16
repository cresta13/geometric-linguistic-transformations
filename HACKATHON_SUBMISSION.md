# OpenAI Build Week Submission Notes

This file is the public judge-facing submission guide for **OpenAI Build Week 2026**.

It is intentionally practical: what the project is, how to test it, what changed during Build Week, and what to show in the demo video.

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

Why Developer Tools:

This is a runnable research workflow: scripts, controls, result tables, summaries, reproducibility metadata, and a lightweight judge demo. It is not a separate web app and not a polished final paper.

## Repository URL

```text
https://github.com/cresta13/geometric-linguistic-transformations
```

## Judge Quick Test

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

This command does not download models or rerun heavy experiments. It reads committed CSV results and prints:

- the focused GPT-2 question-steering result;
- no-steering question-mark base rates;
- out-of-template generalization controls;
- prompt-robustness controls.

## What Was Built With Codex and GPT-5.6

During Build Week, the main completed addition was **GLT-STEER**:

- GPT-2 question activation steering;
- no-steering base-rate controls;
- out-of-template freeform source controls;
- prompt-wrapper robustness checks;
- content-preservation audit;
- copy-prompt preservation follow-up;
- public result summaries and CSV artifacts;
- documentation that makes the result inspectable by judges.

Codex was the primary implementation environment for scripts, result processing, debugging, summaries, and repository organization. GPT-5.6 was used as a reasoning partner for mathematical background, related-work reading, and turning vague objections into concrete controls.

The human role remained to choose the research direction, decide which claims were honest enough to keep, interpret results, and stay responsible for remaining mistakes.

## Main Result to Show

The clearest visible demo is GPT-2 question steering.

```text
Source: The cat sat on the mat.

No steering:
The cat sat on the mat. The cat sat on the mat...

Target question steering:
The cat sat on the mat. The cat sat on the mat? The cat sat on the mat?

Controls:
Random, wrong-class, and negative vectors do not produce the same question-mark effect.
```

Concise numbers:

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

Copy-prompt preservation:
question-and-preserved rate reaches 0.9625-0.9750 in-template
and 0.8250-0.9000 out-of-template for copy-like prompts.

Prompt-only control:
copy-like prompts without steering produce 0.0000 question marks
across 960 no-steering rows from GPT-2 and DistilGPT-2.

DistilGPT-2 replication:
the qualitative target-vs-control separation remains, but the effect is
weaker than GPT-2, with best question-and-preserved rate 0.4625.

Hard out-of-template question audit:
target question-mark rate stays 0.7125-0.7500 on structurally diverse
sentences; matched controls peak at 0.0375.
```

Safe interpretation:

> A GPT-2 question-transformation activation vector can steer output toward question form, and under copy-like prompts it can often preserve recognizable source content. This is not robust general-purpose semantic editing and not proof of a complete linguistic algebra.

The no-steering copy-prompt audit is important: copy-like prompts explain source retention, but they do not by themselves create question marks in the tested setup.

## Important Files for Judges

- `README.md`: main overview and quick demo command.
- `BUILD_WEEK_2026.md`: Build Week provenance and what changed during the event.
- `scripts/show_glt_steer_demo.py`: lightweight demo reader.
- `scripts/run_gpt2_activation_steering_pilot.py`: original activation-steering experiment runner.
- `scripts/run_gpt2_question_steering_controls.py`: base-rate and out-of-template controls.
- `scripts/run_gpt2_question_prompt_robustness.py`: prompt-wrapper robustness control.
- `scripts/analyze_gpt2_question_steering_content_preservation.py`: content-preservation audit.
- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md`
- `results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md`
- `results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md`
- `results/experiments/gpt2_question_content_preservation_20260716_results/SUMMARY.md`
- `results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/SUMMARY.md`
- `results/experiments/distilgpt2_question_copy_prompt_preservation_20260716_results/SUMMARY.md`
- `results/experiments/question_copy_prompt_none_baseline_20260716_results/SUMMARY.md`
- `results/experiments/gpt2_question_hard_oot_copy_prompt_steering_20260716_results/SUMMARY.md`
- `paper/research_program.md`: full research roadmap and scientific framing.
- `docs/hackathon-build/project-story.md`: plain-language project story.
- `docs/hackathon-build/demo-steering.md`: short demo explanation.

## Devpost Fields

### Submitter Type

```text
Individual
```

### Category

```text
Developer Tools
```

### Code Repository

```text
https://github.com/cresta13/geometric-linguistic-transformations
```

### Project Test Link / Instructions

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
- docs/hackathon-build/demo-steering.md
```

### Developer Tool / Installation Notes

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

Run this manually in the Codex/ChatGPT UI session where most core functionality was built:

```text
/feedback
```

Copy the returned session ID into the Devpost field.

## Video Outline

Keep the public YouTube demo under 3 minutes.

1. Personal start: returning to a Lie-algebra question after many years in IT.
2. Research question: do linguistic transformations leave geometric traces inside transformer representations?
3. Working repository: scripts, results, summaries, reports, and reproducibility.
4. Demo command: `scripts/show_glt_steer_demo.py`.
5. GLT-STEER result: question steering, controls, and preservation follow-up.
6. Codex/GPT-5.6 usage: Codex implemented and organized experiments; GPT-5.6 helped reason, critique, and turn objections into tests.
7. Broader message: AI can help more people begin careful independent inquiry, while evidence, criticism, and humility still matter.

## Current Submission Checklist

- [x] Public repository.
- [x] README with quick demo.
- [x] Requirements file.
- [x] Working demo command.
- [x] Committed result CSVs for the main steering demo.
- [x] Main steering result summary.
- [x] Base-rate and out-of-template controls.
- [x] Prompt-robustness controls.
- [x] Content-preservation and copy-prompt follow-up.
- [x] Copy-prompt prompt-only / no-steering baseline.
- [x] DistilGPT-2 copy-prompt replication.
- [x] Hard out-of-template question generalization audit.
- [x] Build Week provenance file.
- [x] Devpost field draft.
- [ ] Demo video under 3 minutes.
- [ ] `/feedback` session ID copied from Codex/ChatGPT UI.
- [ ] Final Devpost submission form filled.
