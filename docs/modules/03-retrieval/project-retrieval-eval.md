---
status: Verified
last_verified: 2026-08-13
volatility: low
---

# Mini-project 3 · The retrieval evaluation harness

Build the harness this module needed and did not have until its last lesson.

```bash
cd projects/retrieval_eval
python -m grader --seed 1
```

Benchmarks are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric is published in full in
[`projects/retrieval_eval/README.md`](https://github.com/paulyonghaoli/llm-systems-for-data-scientists/blob/main/projects/retrieval_eval/README.md).

## Why this one closes the module

[Lesson 3.8](08-retrieval-metrics.md) established that Module 3 came close to
shipping a defensible and wrong recommendation. Every lesson before it reported
recall@10, on which lexical retrieval leads dense by nine points, and every
rank-aware metric reverses that ordering. The harness that would have caught it
is not sophisticated: it keeps per-query values, reports the ceiling of any
bounded metric, and picks its statistical test from the metric's type.

The benchmarks here have that structure built in. **System B is system A with
correct documents moved up and nothing else changed**, which is exactly what a
reranker does — so recall@k is *identical* between them by construction and B
is genuinely better at ranking. Across 30 seeds the recall difference is
exactly 0.0000 and never significant, while the MRR difference is significant
every time.

That makes two mistakes gradeable rather than merely describable. A harness
reporting only recall calls the two systems the same. A harness forcing a
binary test onto MRR discards the magnitudes carrying the difference.

## What you implement

```python
def evaluate(runs, qrels, k=10, phenomena=None) -> dict
def compare(a, b, metric) -> dict
```

`evaluate` returns `per_query`, `means`, `precision_ceiling`, `n_scored` and
`by_phenomenon`. `compare` takes two `per_query` mappings and returns the test
used, the difference, a verdict, and either a p-value or a confidence interval.

Two rules decide several marks. **A query with no correct documents is excluded
from the means rather than scored zero**, because scoring it zero penalises a
system for a question that has no answer. And **the ceiling is
`min(len(gold), k) / k` averaged over the scored queries** — with one correct
document and `k = 10`, precision@10 cannot exceed 0.1 whatever the system does.

## The rubric

| | Criterion | Points |
|---|---|---:|
| **A** | Metric arithmetic — all five metrics match the reference, per query | 25 |
| **B** | Per-query retained — the values are returned, not just means | 20 |
| **C** | Precision ceiling — the bound computed from the gold labels | 15 |
| **D** | Right test per metric — McNemar for recall, bootstrap for continuous | 25 |
| **E** | Breakdown by phenomenon — means per query type | 10 |
| **F** | Degenerate cases — six inputs that break a happy-path harness | 5 |

Pass mark 80.

## Why B and D carry nearly half the marks

Neither is about arithmetic, because the arithmetic is the part everyone gets
right — the starter scores full marks on criterion A.

**B is about foreclosure.** A function returning only a mean has destroyed
information that cannot be recovered: paired tests, confidence intervals and
per-query-type breakdowns all need the per-query values, and every one of those
is something you will want within a week of having the harness. Averaging is
one line the caller can do afterwards; the reverse is impossible.

**D is about choosing a test from the measurement rather than from habit.**
Recall is one bit per query, so an exact McNemar test over the discordant pairs
applies. MRR and nDCG are continuous, so the paired comparison is a bootstrap
over per-query differences — forcing them through McNemar means thresholding
first, which throws away exactly the positional information those metrics exist
to capture. [Lesson 3.6](06-reranking.md) is the case in point: the same
reranking is overwhelmingly significant at rank one and invisible at rank ten.

## What the starter's breakdown tells you

`student.py` ships correct metric arithmetic and returns only the means. It
scores **25/100**, and the first failure it reports is the one worth sitting
with:

```
D — recall: test 'none', expected 'mcnemar'; recall: significant=True,
    reference says False
```

It has declared two systems with *identical* recall to be significantly
different, because with only two means available there is nothing to test and
it guessed. That is not a contrived error. It is what a comparison looks like
when the harness that produced the numbers threw away everything a comparison
needs.
