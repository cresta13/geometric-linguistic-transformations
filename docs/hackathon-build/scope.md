# GLT Lab Scope

## Project Name

GLT Lab

## One-Sentence Summary

GLT Lab turns an active transformer-geometry research repository into a runnable public lab where people can inspect, reproduce, and extend experiments about linguistic transformations in embedding spaces.

## Hackathon Track

Education

This is primarily an education submission because the core goal is to make research participation legible and practical for people outside a formal lab. The project also has developer-tool qualities, but the strongest story is public scientific participation: users learn the question, run evidence, inspect limits, and contribute a small reproducible result.

## Core Idea

The project starts from the existing GLT research program:

> Do linguistic transformations such as question formation, negation, modality, tense shift, affect polarity, or ordered composition appear as reusable geometric objects in transformer representation spaces?

The hackathon version should not present GLT as a finished theory. It should present GLT as a working public research lab:

1. A person opens the repository.
2. They understand the research question in plain language.
3. They run or inspect a small experiment.
4. They see results, figures, and caveats.
5. They can add a new transformation, language, model, or control.
6. They produce a reproducible contribution instead of just reading a paper.

## Why This Matters

Most people experience AI research as finished papers, leaderboards, or opaque announcements. GLT Lab reframes research as something a motivated person can participate in now:

- run a real experiment locally;
- inspect where a claim works and where it fails;
- compare models and controls;
- contribute a small extension with clear provenance;
- learn scientific caution by seeing negative results preserved, not hidden.

The project is valuable even if the GLT hypothesis remains bounded, because the educational artifact is the workflow: transparent, reproducible, critical AI research made understandable. GLT is the example, not the only destination. The broader message is that AI can help people begin careful independent inquiry into many hard questions that matter to them.

## Primary Audience

- Curious people who have carried a hard question for years and want to learn how to investigate it carefully.
- Builders who want to enter AI-assisted research but do not know where to start.
- Students learning representation analysis, mechanistic interpretability, or NLP evaluation.
- Independent researchers who need an example of a structured, transparent research trail.
- Reviewers or collaborators who want to inspect how claims are connected to scripts, CSV files, and figures.

## What We Are Building for Build Week

### 1. A Public Research Entry Point

Create a cleaner README path that separates:

- "I am new; what is this?"
- "I know ML; show me the methods and evidence."
- "I want to run or contribute an experiment."

The first screen should make the social idea explicit: everyone can contribute to science by running controlled, inspectable experiments.

### 2. A Guided Contribution Workflow

Add a small, concrete workflow for contributors:

- choose a research track;
- choose one extension type: new language, new transformation, new model, new control, or repeatability check;
- run the relevant script;
- save results in the expected folder;
- write a short result note using a template;
- keep caveats attached to claims.

This can be documentation-first for the hackathon, with one runnable demo path.

### 3. A Demo Experiment Path

Use one lightweight, visually understandable GLT experiment as the demo path.

Recommended demo:

- GPT-2 question activation steering focused run, because it has a visible behavioral outcome: steering makes the model produce question marks.
- It is easier to explain in a video than matrix closure, spectral nulls, or signed-permutation diagnostics.
- It still connects to the deeper GLT research question: does a transformation direction do causal work, not just classify existing embeddings?

The demo should show:

- source sentence;
- no steering output;
- target question steering output;
- random/wrong/negative control outputs;
- summary metric table.

### 4. A Research Map

Add a concise map of GLT tracks:

- GLT-DV: delta-vector diagnostics.
- GLT-SPOT: ordered composition tests.
- GLT-MOLT: matrix/operator diagnostics.
- GLT-XFER: cross-model transfer.
- GLT-AFFECT: affective transformation geometry.
- GLT-STEER: activation steering from transformation deltas.

Each track needs:

- plain-language question;
- technical question;
- current result;
- strongest caveat;
- how to contribute.

### 5. A Devpost-Ready Story

Prepare submission text around this claim:

> GLT Lab is not only a research repository; it is an example of AI-assisted independent science. It shows how a person can start from an old question, learn fast with AI, build experiments, preserve caveats, and create a reproducible research trail.

## What We Are Not Building This Week

- A polished hosted SaaS platform.
- A full web app with accounts, profiles, or a database.
- A claim that GLT proves a complete linguistic algebra.
- A general-purpose benchmark leaderboard.
- A replacement for peer review.
- A complete no-code experiment builder.

These are intentionally out of scope because the hackathon deadline rewards a working, coherent project more than a scattered platform.

## Possible Product Names

Primary:

- GLT Lab

Alternatives:

- GLT Open Lab
- GLT Research Kit
- GLT Commons
- Geometric Linguistic Transformations Lab

Recommendation: use **GLT Lab**. It is short, connected to the repository name, and flexible enough to include both research and public participation.

## Demo Video Angle

Suggested 3-minute structure:

1. "Research should not be something people only read after it is finished."
2. Show GLT Lab and the plain-language research question.
3. Run or open the focused GPT-2 question steering result.
4. Show that target steering produces question marks while controls do not.
5. Show where the exact script, CSV, and summary live.
6. Show the research trail: script, CSV, result, caveat.
7. Close with the broader message: GLT is my example, but AI can help many people begin their own careful research path.

## Time Budget

Build Week deadline: July 21, 2026 at 5:00 PM PT.

The scope assumes a compact build:

- no new large overnight experiment is required for submission;
- use already completed GLT results as sample evidence;
- spend effort on clarity, runnable demo, documentation, and submission polish;
- add only small scripts or views if they directly improve the demo.

## Success Criteria

The project is ready to submit when:

- README explains the project to both non-specialists and technical readers.
- A judge can run at least one demo command or inspect a prepared result folder.
- The Devpost description clearly explains how Codex and GPT-5.6 accelerated the workflow.
- The demo video shows a working result, not just slides.
- The repository links claims to scripts, CSVs, figures, and caveats.
- The submission does not overclaim GLT as a solved theory.
