---
status: Verified
last_verified: 2026-08-09
volatility: low
---

# Mini-project 0 · The reliability report

Turn a raw run log into the numbers that decide whether a system is shippable
— and refuse to produce the ones the log cannot support.

```bash
cd projects/reliability_report
python -m grader --seed 1
```

Logs are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric is published in full in
[`projects/reliability_report/README.md`](https://github.com/paulyonghaoli/llm-systems-for-data-scientists/blob/main/projects/reliability_report/README.md).

This is the module's graded artifact and it draws on all four lessons: cohort
keys from [0.1](01-what-changes.md), failure accounting from
[0.2](02-anatomy.md), Wilson intervals from [0.3](03-evaluation-breaks.md),
and percentiles and cost units from [0.4](04-cost-and-latency.md).

## What you implement

```python
def report(records: list[dict], price_ratio: float) -> dict
```

returning counts, a success rate with a Wilson interval, nearest-rank p50 and
p95, cost in input-token units, cost per *success*, and a sorted list of
warnings.

## The rule that is the point

**If the log spans more than one `model_version`, `success_rate` and `ci` must
both be `None`.**

A single success rate computed across two different models is not a
measurement of either one — but it looks exactly like a measurement. Same
type, same range, same place on the dashboard, and no way for a reader to tell
it apart from a real one. Returning `None` is the only honest output, and it
is the only one that prompts the right next question.

Most of the work in this project is arithmetic you have done before. This
criterion is the one that is actually about LLM systems.

## Rubric

| | Criterion | Points |
|---|---|---:|
| **A** | Counts and cohorts | 20 |
| **B** | Success rate and Wilson interval, or `None` where not permitted | 20 |
| **C** | Nearest-rank p50 and p95 | 20 |
| **D** | Cost units and cost per success | 20 |
| **E** | Correct warnings, and no summary across versions | 20 |

Pass mark **80**. The shipped starter — every field populated, every question
answered, no warnings — scores **11.2**.

Four fixed edge cases run alongside the seeded logs: an empty log, a log where
nothing succeeded, a log spanning two versions, and a log where every call was
refused. Each breaks a different assumption that a happy-path report makes
silently, and the last one is the subtlest: a 0% success rate that is a
*quality* finding being filed as a reliability failure.
