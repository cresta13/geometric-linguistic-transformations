# OpenAI Build Week 2026 Provenance

This file documents what existed before OpenAI Build Week 2026, what changed during the event, and how Codex and GPT-5.6 were used.

## Submission Identity

Project title:

```text
Geometric Linguistic Transformations
```

Primary category:

```text
Developer Tools
```

Secondary fit:

```text
Education
```

Elevator pitch:

```text
A reproducible investigation into whether linguistic transformations form geometric directions inside transformers, and whether AI can make serious independent science accessible again.
```

## Working Project

The submitted project is a reproducible research workflow, not a separate web app.

It includes:

1. Synthetic controlled linguistic transformation datasets.
2. Scripts for extracting and analyzing transformer representations.
3. Delta-vector, endpoint, concat, source-only, and target-only controls.
4. Semantic and hard holdout experiments.
5. Shuffle, norm-matched, spectral, endpoint, and random-label controls.
6. Composition and learned-operator diagnostics.
7. GPT-2 activation-steering experiments.
8. CSV result tables, summaries, drafts, reports, and reproducibility metadata.
9. A lightweight judge demo that reads committed results.

Judge command:

```powershell
.\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
```

## What Existed Before Build Week

Before Build Week, the repository already contained the main GLT research program:

- GLT-DV: delta-vector diagnostics with endpoint controls.
- GLT-SPOT: signed-permutation composition tests.
- GLT-MOLT: learned matrix/operator audits.
- GLT-XFER: cross-model transfer and alignment stress tests.
- GLT-AFFECT: early graded affective-geometry experiments.
- `paper/`, `reports/`, `results/`, `research/`, and `scripts/` as the core research structure.
- Zenodo archived snapshots and citation metadata.

## What Changed During Build Week

Build Week added and packaged **GLT-STEER**, the behavior-level intervention track:

- focused GPT-2 question activation steering;
- base-rate controls for question-mark generation;
- out-of-template freeform controls;
- prompt-wrapper robustness controls;
- content-preservation audit;
- copy-prompt preservation follow-up;
- copy-prompt no-steering baseline audit;
- DistilGPT-2 copy-prompt replication;
- a lightweight demo command for judges;
- public submission and provenance notes.

The steering result was strengthened step by step:

1. A target question vector produced question marks while random, wrong-class, and negative controls did not.
2. No-steering base-rate controls showed that GPT-2 did not naturally add question marks under the tested declarative prompts.
3. Out-of-template prompts showed the effect was not restricted to the original template sentences.
4. Prompt-robustness tests showed the effect survived several prompt wrappers.
5. Content-preservation tests showed an important boundary: copy-like prompts preserve source content much better than bare or quoted prompts.
6. A no-steering copy-prompt audit showed that copy-like prompts alone do not produce question marks: `0.0000` question-mark rate across `960` no-steering rows from GPT-2 and DistilGPT-2.
7. A DistilGPT-2 replication preserved the qualitative target-vs-control separation but reduced the effect size substantially.

## How Codex Was Used

Codex was the primary implementation environment for the Build Week work.

It helped:

- write and revise experiment scripts;
- organize result folders;
- run and monitor long experiments;
- inspect logs and memory/process failures;
- add controls when a result looked too easy;
- turn CSV outputs into summaries;
- update README and research-program files;
- keep claims linked to scripts and result files.

The human researcher set the direction, chose the claims, interpreted the evidence, rejected overclaiming, and remained responsible for mistakes.

## How GPT-5.6 Was Used

GPT-5.6 was used as a reasoning partner:

- to revisit mathematical ideas from Lie algebras and representation geometry;
- to read and compare related work;
- to challenge weak claims;
- to turn vague intuitions into falsifiable controls;
- to explain unfamiliar methods in plain language.

AI did not make the scientific decisions. It made the choices understandable enough for the human researcher to make them.

## Main Build Week Result

The cleanest current Build Week claim is:

> Under copy-like prompts, a GPT-2 question-transformation activation vector can often preserve the source content while steering the continuation toward question form. This effect is sharply separated from no-steering and wrong-vector controls on the joint question-and-preserved metric. Copy-like prompts alone do not produce question marks in the tested no-steering controls.

This is a bounded result:

- GPT-2-only so far;
- question-only so far;
- prompt-dependent;
- model-dependent, with DistilGPT-2 much weaker than GPT-2;
- not robust general-purpose semantic editing;
- not proof of a complete linguistic algebra.

## Judge Verification Path

Recommended quick path:

1. Read `README.md`.
2. Run:

   ```powershell
   .\.venv\Scripts\python.exe scripts\show_glt_steer_demo.py
   ```

3. Inspect:

   ```text
   HACKATHON_SUBMISSION.md
   BUILD_WEEK_2026.md
   docs/hackathon-build/project-story.md
   docs/hackathon-build/demo-steering.md
   results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md
   results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md
   results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md
   results/experiments/gpt2_question_content_preservation_20260716_results/SUMMARY.md
   results/experiments/gpt2_question_copy_prompt_preservation_20260716_results/SUMMARY.md
   results/experiments/distilgpt2_question_copy_prompt_preservation_20260716_results/SUMMARY.md
   results/experiments/question_copy_prompt_none_baseline_20260716_results/SUMMARY.md
   paper/research_program.md
   ```

## Remaining Manual Submission Items

- Record a public YouTube demo video under 3 minutes.
- Run `/feedback` in the Codex/ChatGPT UI and paste the returned session ID into Devpost.
- Fill and submit the Devpost form.
