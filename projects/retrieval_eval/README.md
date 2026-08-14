# Mini-project 3 — the retrieval evaluation harness

Build the evaluation harness Module 3 needed and did not have until lesson 3.8.

```bash
cd projects/retrieval_eval
python -m grader --seed 1
```

Benchmarks are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric below is the whole rubric; nothing is
hidden.

## Why this one

Module 3 measured eight lessons' worth of retrieval systems using recall@10,
and lesson 3.8 established that this had nearly produced a defensible and wrong
recommendation: on recall lexical retrieval leads dense by nine points, and on
every rank-aware metric the ordering reverses. The harness that would have
caught it is not complicated. It is the one that keeps per-query values,
reports the ceiling of any bounded metric, and picks its statistical test from
the metric's type rather than from habit.

The benchmarks here have that structure built in. **System B is system A with
correct documents moved up and nothing else changed** — which is exactly what a
reranker does. So the two systems have *identical* recall@k by construction,
and B is genuinely better at ranking. A harness that reports only recall
concludes they are the same system. A harness that forces a binary test onto
MRR throws away the magnitudes that carry the difference.

## What to implement

Two functions in `student.py`.

### `evaluate(runs, qrels, k=10, phenomena=None) -> dict`

| argument | shape |
|---|---|
| `runs` | `{query_id: [doc_id, ...]}`, ranked best first |
| `qrels` | `{query_id: [doc_id, ...]}`, the correct documents |
| `k` | cutoff, an integer |
| `phenomena` | `{query_id: label}` or `None` |

Returns a dict with these keys:

| key | contents |
|---|---|
| `per_query` | `{query_id: {metric: score}}` for all five metrics |
| `means` | `{metric: mean over scored queries}` |
| `precision_ceiling` | the highest precision@k these gold labels permit |
| `n_scored` | how many queries contributed to the means |
| `by_phenomenon` | `{label: {metric: mean}}`, empty when `phenomena` is `None` |

The five metrics are `recall`, `precision`, `mrr`, `map` and `ndcg`, defined as
in lesson 3.8 §C. Two details decide several marks:

**A query with no correct documents is excluded from the means** rather than
scored zero, because scoring it zero penalises a system for a question that
has no answer. Give it `float("nan")` in `per_query` and skip it when
averaging. `n_scored` reports how many were left.

**The ceiling is `min(len(gold), k) / k` averaged over the scored queries.**
With one correct document and `k = 10`, nine slots are necessarily wrong and
precision@10 cannot exceed 0.1 — so a precision figure without this number
beside it is uninterpretable.

### `compare(a, b, metric) -> dict`

`a` and `b` are two runs' `per_query` mappings. Returns:

| key | contents |
|---|---|
| `metric` | the metric compared |
| `test` | `"mcnemar"` for `recall`, `"bootstrap"` for the rest |
| `diff` | `mean_b - mean_a` |
| `significant` | boolean |
| `p_value` | for McNemar |
| `ci_low`, `ci_high` | for the bootstrap, a 95% interval on the paired difference |

**Recall is one bit per query**, so an exact McNemar test over the discordant
pairs applies and is exact. **Everything else is continuous**, so the paired
comparison is a bootstrap over per-query differences — forcing a continuous
metric through McNemar means thresholding it first, which discards the
magnitudes the metric exists to measure.

Significance is `p < 0.05` for McNemar, and for the bootstrap it is the 95%
interval excluding zero.

## Rubric — 100 points

| | criterion | points | what it checks |
|---|---|---:|---|
| A | metric arithmetic | 25 | all five metrics match the reference, per query, on every benchmark |
| B | per-query retained | 20 | `per_query` is returned and complete |
| C | precision ceiling | 15 | the bound is computed from the gold labels |
| D | right test per metric | 25 | McNemar for recall, bootstrap for continuous, verdict matching the reference |
| E | breakdown by phenomenon | 10 | means per query type |
| F | degenerate cases | 5 | the six inputs below |

Pass mark is 80.

Criteria B and D carry nearly half the marks and neither is about arithmetic,
because the arithmetic is the part everyone gets right. Keeping the per-query
values is what makes every later question answerable, and choosing the test
from the metric's type is what stops a real reranking improvement being
reported as noise — or, as the starter does, an identical pair of systems being
reported as different.

## Degenerate cases (criterion F)

Your harness is run against each of these and must not raise:

- no queries at all
- a query whose gold list is empty
- a query where nothing was retrieved
- fewer results than `k`
- more gold documents than `k`
- every returned result correct

## The starter

`student.py` ships a harness whose metric arithmetic is **correct** — it scores
full marks on criterion A. What it does is return only the means, which
forecloses everything else: no per-query values, so no paired test; no ceiling,
so precision is uninterpretable; no breakdown, so a per-type regression is
invisible. `compare` is then left with two numbers to subtract, and it declares
any difference significant.

It scores 25/100, and the failures are worth reading before you start: the
first one it reports is that it called two systems with *identical* recall
significantly different.

## Grading

```bash
python -m grader --seed 1          # grade student.py
python -m grader --seed 7          # a different benchmark
python -m grader --reference       # the reference scores 100
python -m grader --sweep 30        # the reference on 30 consecutive seeds
```

CI runs the last two, so the reference is verified to score full marks against
its own rubric on every seed rather than on the one it was written against.
