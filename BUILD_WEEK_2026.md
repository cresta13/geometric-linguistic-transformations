# OpenAI Build Week 2026 Provenance

This file explains what this repository is submitting to OpenAI Build Week 2026 and how Codex and GPT-5.6 were used.

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

## What the Working Project Is

This is not a new web app made only for the hackathon.

The working project is a reproducible research workflow:

1. Synthetic controlled linguistic transformation datasets.
2. Scripts for extracting and analyzing transformer representations.
3. Delta-vector, endpoint, concat, and source-only controls.
4. Semantic holdouts and hard holdouts.
5. Shuffle, norm-matched, spectral, endpoint, and random-label controls.
6. Composition and learned-operator diagnostics.
7. GPT-2 activation-steering experiments.
8. CSV result tables, figures, summaries, draft papers, and reports.
9. A research diary and roadmap that preserve negative and bounded results.
10. A lightweight judge demo that reads committed results without rerunning large models.

The repository can be inspected as a research artifact and tested through:

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

The Build Week work turned the repository into a clearer, testable submission package and added a new behavior-level track:

- GLT-STEER: GPT-2 activation steering with question-transformation vectors.
- A focused GPT-2 question-steering run.
- Base-rate controls for question-mark generation.
- Out-of-template generalization controls.
- Prompt-wrapper robustness controls.
- Content-preservation audits.
- A lightweight demo command for judges.
- Submission notes, demo script, and Build Week documentation.

Relevant commits since the Build Week period began include:

```text
1de97bb | Add Build Week project story
6484607 | Document GPT-2 steering result
e1b90a9 | Add GLT steering demo path
360c719 | Add GPT-2 question steering controls
8231a61 | Add GPT-2 question prompt robustness runner
6b88916 | Add GPT-2 prompt robustness result
f466df1 | Add GPT-2 steering preservation audits
```

## How Codex Was Used

Codex was the primary build environment for the Build Week work.

The human role was to set the research direction, decide which claims mattered, reject overclaiming, interpret results, and choose the story. Codex did most of the new repository implementation work:

- writing experiment scripts;
- organizing result folders;
- running and monitoring long experiments;
- inspecting logs and memory issues;
- adding controls when a result looked too easy;
- turning CSV outputs into summaries;
- updating README and paper drafts;
- keeping results linked to scripts;
- preparing the judge-facing demo path.

The point is not that Codex replaced scientific judgment. The point is that it made a serious research workflow possible for a person returning to academic-style work after many years away from it.

## How GPT-5.6 Was Used

GPT-5.6 was used as a reasoning partner:

- to revisit mathematical ideas from Lie algebras and representation geometry;
- to read and compare related work;
- to challenge weak claims;
- to turn vague intuitions into falsifiable controls;
- to explain unfamiliar methods in plain language;
- to help keep the project honest when results failed or became bounded.

AI did not make the scientific decisions. It made the choices understandable enough for the human researcher to make them.

## Human Story

This project began with a personal return to science.

Anna had worked in IT for about fourteen years. While preparing to apply to a machine-learning master's program, she remembered that her 2012 university thesis had been connected with Lie algebras. After many years away from academic mathematics, she asked a simple question:

```text
Can Lie algebra help us understand transformers?
```

That question became GLT.

The project is still imperfect and may remain bounded. Some results are positive, some fail, and some show leakage or endpoint dependence. That is part of the value: the repository preserves the process of testing a beautiful idea against reality.

The broader message is not "join this exact research topic." The message is that many people have questions they have carried for years. With AI, more of them can read, prototype, test, document, and contribute evidence. If even a fraction of people spent a few hours each week on careful independent inquiry, scientific progress could become much more distributed.

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
   docs/hackathon-build/project-story.md
   results/experiments/gpt2_question_activation_steering_focused_20260714_results/SUMMARY.md
   results/experiments/gpt2_question_steering_controls_20260714_results/SUMMARY.md
   results/experiments/gpt2_question_prompt_robustness_20260715_results/SUMMARY.md
   paper/research_program.md
   ```

4. For full research context, inspect `paper/`, `reports/`, `results/`, `research/`, and `scripts/`.

## Remaining Manual Submission Items

- Record a public YouTube demo video under 3 minutes.
- Run `/feedback` in the Codex/ChatGPT UI and paste the returned session ID into Devpost.
- Fill the Devpost submission form.
