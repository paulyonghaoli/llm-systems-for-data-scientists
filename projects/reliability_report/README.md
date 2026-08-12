# Mini-project 0 · The reliability report

Turn a raw run log into the numbers that decide whether a system is shippable
— and refuse to produce the ones the log cannot support.

```bash
cd projects/reliability_report
python -m grader --seed 1
```

Logs are generated fresh from the seed on every run. The rubric below is the
whole rubric; nothing is hidden.

## What you implement

One function, in `student.py`:

```python
def report(records: list[dict], price_ratio: float) -> dict
```

Each record has `model_version`, `params`, `prompt_hash`, `in_tokens`,
`out_tokens`, `latency_ms`, and `outcome` (one of `ok`, `refused`, `timeout`,
`invalid`, `error`).

Return exactly these keys:

| Key | Meaning |
|---|---|
| `n` | number of records |
| `cohorts` | distinct `(model_version, params signature, prompt_hash)` triples |
| `success_rate` | fraction with `outcome == "ok"`, or `None` — see below |
| `ci` | Wilson 95% interval as `(lo, hi)`, or `None` |
| `refusal_rate` | fraction with `outcome == "refused"`, `None` on an empty log |
| `p50_ms`, `p95_ms` | nearest-rank percentiles over **every** record |
| `cost_units` | `sum(in_tokens + out_tokens * price_ratio)` over **every** record |
| `cost_per_success` | `cost_units / n_ok`, or `None` if nothing succeeded |
| `warnings` | sorted list, from the fixed vocabulary below |

### Rules

1. **Percentiles are nearest-rank**: `sorted(values)[ceil(q*n) - 1]`. Not the
   mean, not `mean + 2σ`.
2. **The interval is Wilson**, not the normal approximation.
3. **Cost includes failures.** A timed-out request was billed. `cost_per_success`
   divides total cost by successes only.
4. **Warnings**, exactly these strings, sorted:
   `empty_log` · `multiple_model_versions` · `multiple_param_sets` ·
   `small_sample` (0 < n < 250) · `refusals_present`
5. **Refuse to summarise across model versions.** If the log spans more than
   one `model_version`, `success_rate` and `ci` must both be `None`.

Rule 5 is the point of the project. A single success rate computed over two
different models is not a measurement of either one, but it looks exactly like
a measurement — same type, same range, same place on the dashboard. Returning
`None` is the only honest output, and the only one a reader can act on.

Rule 4's `small_sample` threshold of 250 is not arbitrary: it is the smallest
sample size on lesson 0.3's grid whose 95% interval is within ±5 points.

## Rubric — 100 points

Scored as the fraction of cases satisfying each criterion, over 12 seeded logs
plus 4 fixed edge cases.

| | Criterion | Points |
|---|---|---:|
| **A** | Counts and cohorts | 20 |
| **B** | Success rate and Wilson interval, or `None` where not permitted | 20 |
| **C** | Nearest-rank p50 and p95 | 20 |
| **D** | Cost units and cost per success | 20 |
| **E** | Correct warnings, and no summary across versions | 20 |

Pass mark **80**.

The four edge cases are: an empty log, a log in which nothing succeeded, a log
spanning two model versions, and a log in which every call was refused. Each
one breaks a different assumption that a report written against the happy path
makes silently — division by zero, division by zero somewhere else, an invalid
summary, and a 0% success rate that is a *quality* result being reported as a
reliability failure.

## The starter

`student.py` ships with the report everybody writes first: every field
populated, every question answered, no warnings, no refusals. It scores
**11.2/100**. The breakdown names each wrong assumption individually.

## Useful commands

```bash
python -m grader --seed 7 --scenarios 20
python -m grader --reference --seed 1
python -m grader --sweep 30
```
