---
status: Verified
last_verified: 2026-08-11
volatility: low
---

# Mini-project 2 · The extraction harness

Turn raw model output into validated records, and be honest about everything
you could not turn into one.

```bash
cd projects/extraction_harness
python -m grader --seed 1
```

Batches are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric is published in full in
[`projects/extraction_harness/README.md`](https://github.com/paulyonghaoli/llm-systems-for-data-scientists/blob/main/projects/extraction_harness/README.md).

## What you implement

```python
def process(outputs, expensive_repair) -> list[dict]
```

Each input is `{"text": str, "finish_reason": str}`; each result is a `status`
of `ok` / `repaired` / `truncated` / `rejected`, a `record` or `None`, and a
`reason` whenever nothing was recovered. The schema is
[lesson 2.4](04-structured-output.md)'s.

`expensive_repair` stands in for a model-based repair call and **returns
`None`**. That is deliberate: the exercise is about what it costs to reach for
it rather than what it would have returned, so a harness that calls it on every
output recovers exactly as much and loses criterion D.

## The rubric

| | Criterion | Points |
|---|---|---:|
| **A** | Recovery — at least as many valid records as the reference | 25 |
| **B** | No false accepts — never returns a record that fails the schema | 25 |
| **C** | Truncation caught — every non-`stop` finish reason reported, including the ones that parse and validate | 20 |
| **D** | Cost discipline — no more expensive repairs than the reference | 15 |
| **E** | Nothing dropped — one result per input, every rejection carries a reason | 10 |
| **F** | Degenerate cases | 5 |

Pass mark **80**. The shipped starter — parse, and accept whatever parses —
scores **26.2**.

## Why B and C carry half the marks between them

**B is a safety property rather than a quality one.** A harness that fails to
recover a record has produced a gap somebody will notice. A harness that emits
a record violating the schema has produced a bad record wearing exactly the
same clothes as a good one, which nobody will notice until it has been joined,
aggregated and reported on.

**C is the whole distance between this project and 2.4's repair ladder.**
Around a tenth of truncated outputs parse *and* satisfy the schema, so every
check in the previous lesson pronounces them healthy. The finish reason is the
only signal that reaches them; it arrives free in a response you have already
paid for; and checking it before anything else is a two-line fix. The starter
scores zero on this criterion for six batches in a row, which is what makes it
worth twenty points.

## What the starter's breakdown tells you

Run it before changing anything. On the first batch it recovers 78 records
where the reference recovers 99, accepts 14 that fail the schema, and treats 8
truncated outputs as good — three distinct failures, each mapping to one thing
the lesson said and the code does not yet do.
