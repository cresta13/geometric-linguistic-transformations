# Research Fixation Protocol

This project is now maintained as an iterative research notebook.

## Rule 1: Daily research packet

At the end of each research session we create a dated report packet:

- `reports/YYYY-MM-DD_research_report.pdf`
- supporting source data remains in the corresponding `*_results/csv/` folders
- supporting plots remain in the corresponding `*_results/figures/` folders
- the report includes the core claims, key plots, CSV-derived tables, and the code listing for the newest experiment

The PDF is intended for external verification by an independent reader or analysis tool.

## Rule 2: Research diary

All important conceptual moves are recorded in `research/diary.md`.

Each entry should answer:

- What did we believe before this step?
- What did we test?
- What changed after seeing the results?
- Why does the next step follow from this one?

## Rule 3: Article candidates

When a result starts looking publishable, we create or update an article folder under:

- `paper/articles/<slug>/`

Each candidate gets:

- `README.md` with title, thesis, evidence, risks, and next experiments
- optional `figures/`
- optional `draft.md`

## Rule 4: Keep raw traces but delete regenerable bulk

We keep:

- CSV summaries and raw CSVs
- metadata JSON
- figures
- scripts

We may delete:

- regenerable `.npy` embedding caches
- all-pairs intermediate matrices when summaries/figures already exist
- `__pycache__`

When deletion happens, it is recorded in the diary or final session note.

## Rule 5: Prefer falsifiable claims

A claim is only promoted if it survives at least one control:

- holdout split
- permutation/null baseline
- semantic equivalence control
- cross-model replication
- bootstrap confidence interval

Negative or mixed results are kept. They are often the useful part.
