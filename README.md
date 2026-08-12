# LLM Systems for Data Scientists

An interactive, project-based curriculum for people who already know Python,
pandas, sklearn, statistics and offline evaluation, and who have never trained
a transformer, shipped a retrieval system, sized a KV cache, or written an
evaluation that had to catch a regression in a model they do not control.

Self-study, written up and shared. Static site, no backend, no accounts, no
cost, and no live API calls anywhere in the material.

- **Plan:** [PLAN.md](PLAN.md) — the governing document. Scope, the four-tier
  assessment ladder, the executable core, the 22 CI gates, and what is
  deliberately not being built.
- **How to work on it:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Live site:** <https://llm-systems-for-data-scientists.paullimale.workers.dev> — soft launch, so it is reachable but excluded
  from search engines by `docs/robots.txt` and a `noindex` meta tag. Toggle
  both together with `python tools/launch.py --go` / `--unlaunch`.

## Status — P1 in progress, 2026-08-09

P0 complete (platform + exemplar lesson). Course I under way: **Modules 0 and
1 complete, Module 2 at 2.1 of 5**.

| | |
|---|---|
| Lessons | 11 — Module 0 (0.1–0.5), Module 1 (1.1–1.5), Module 2 (2.1) |
| In-browser exercises | 24 |
| Quiz questions | 90 across 11 banks |
| Autograded mini-projects | 2 |
| Experiments | 10, plus 5 generated figures and one hand-run fixture recorder |
| Capstones | 0 (Capstone I closes P1) |
| CI gates | 22, run as 14 commands |

Full scope is 4 courses, 16 modules, 87 lessons, 4 capstones — about 540
learner-hours. Every phase is independently shippable; see PLAN.md §8.

## Quick start

Deploy is deliberately manual — nothing publishes on a push:

```bash
python tools/verify.py && .venv/Scripts/mkdocs build && npx wrangler deploy
```

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,docs]"
python tools/verify.py
.venv/Scripts/mkdocs serve
```

Try the mini-project the way a learner would:

```bash
cd projects/tokenizer_mini
python -m grader --seed 1
```

The shipped starter scores 26.6/100. The rubric that says why is published in
full in that directory's README.

## How the executable core works

No exercise, experiment or grader ever touches the network — that is gate 13,
not a convention. Real behaviour from production tokenizers is captured once
by a recorder run by hand (`experiments/record_tiktoken.py`), stamped with a
date and a version, and replayed from `data/fixtures/`. Everything else is
pure Python that runs identically in CPython and in Pyodide.

Every number quoted in a lesson carries a marker naming the experiment and key
that produced it, and gate 18 re-runs the experiment on every commit to check
the prose still agrees with the code.

## Licence

MIT for the code. Prose and curriculum content are the author's own.
