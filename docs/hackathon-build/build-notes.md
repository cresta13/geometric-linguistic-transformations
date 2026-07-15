# Build Notes

## 2026-07-14

Started OpenAI Build Week planning for the existing GLT repository.

Decision: do not create a separate project. The hackathon submission should be a productized layer over the current research repository.

Working project name: GLT Lab.

Recommended category: Education.

Core framing:

> Everyone can contribute to science now by running, inspecting, and extending controlled AI-research experiments.

Why this framing:

- It preserves the existing scientific work instead of pretending it is a consumer app.
- It creates a stronger Devpost story than a raw research archive.
- It fits the judging criteria: working project, design, impact, and quality of idea.
- It gives a clear demo path using already completed GLT steering results.

Most promising demo result:

- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/`
- GPT-2 question steering: target vector produces question marks at high rate while random, wrong-class, and negative controls do not.

Immediate next steps:

1. Write PRD for GLT Lab.
2. Add a contributor-oriented research map.
3. Add a demo walkthrough for the focused steering experiment. Done: `docs/hackathon-build/demo-steering.md`.
4. Prepare Devpost project description and video script.
5. Decide whether to add a minimal dashboard or keep the submission documentation-first with runnable scripts and result views.

Added `docs/hackathon-build/project-story.md` to capture the human narrative in simple language:

- returning to science after years in IT;
- reviving an old Lie-algebra background through AI-assisted research;
- using Codex and GPT-5.6 as a bridge from curiosity to runnable experiments;
- framing GLT Lab as public scientific participation rather than a finished theory.

## 2026-07-14 Steering Demo

Added the focused GPT-2 question-steering run as the clearest Build Week demo case.

Why this result is useful for the hackathon:

- It is visible without specialist math: the target vector makes GPT-2 produce question marks.
- It has simple controls: random, wrong-class, and negative vectors do not produce the same question-mark behavior.
- It connects the personal story to a working experiment: curiosity becomes a script, a controlled run, a table, and a caveat.
- It is honest enough for a public demo: this is form steering, not solved semantic editing.

Primary demo file:

- `docs/hackathon-build/demo-steering.md`

Primary result file:

- `results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md`

## 2026-07-16 Submission Packet

Added `HACKATHON_SUBMISSION.md` as the single checklist for the OpenAI Build Week submission.

The judge-facing quick path is now:

1. open `README.md`;
2. run `scripts/show_glt_steer_demo.py`;
3. inspect the three steering summaries:
   - focused steering result;
   - base-rate and out-of-template controls;
   - prompt-wrapper robustness controls.

The only remaining manual submission items are the demo video and the Codex `/feedback` session ID.
