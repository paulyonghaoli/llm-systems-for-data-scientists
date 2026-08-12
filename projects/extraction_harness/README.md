# Mini-project 2 · The extraction harness

Turn raw model output into validated records, and be honest about everything
you could not turn into one.

```bash
cd projects/extraction_harness
python -m grader --seed 1
```

Batches are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric below is the whole rubric.

## What you implement

One function, in `student.py`:

```python
def process(outputs, expensive_repair) -> list[dict]
```

| Argument | Meaning |
|---|---|
| `outputs` | list of `{"text": str, "finish_reason": str}` |
| `expensive_repair` | `callable(text) -> str \| None`, standing in for a model-based repair call. **Every invocation is counted** and charged against criterion D |

Return exactly one result per input:

| Key | Meaning |
|---|---|
| `status` | `"ok"` · `"repaired"` · `"truncated"` · `"rejected"` |
| `record` | the validated dict, or `None` |
| `reason` | why it was not recovered — required unless the status is `ok` or `repaired` |

The schema is lesson 2.4's: `record_id` (str), `status` (str, one of
`cancelled` / `fulfilled` / `pending`) and `quantity` (int) are all required,
extra keys are allowed, and a `bool` is not an `int`.

`expensive_repair` returns `None`. That is deliberate: the exercise is about
what it costs to reach for it, not what it would have returned. A harness that
calls it on everything recovers exactly as much and loses criterion D.

## Rubric — 100 points

Scored over 6 seeded batches of 120 outputs plus 4 fixed edge cases.

| | Criterion | Points | Passes when |
|---|---|---:|---|
| **A** | Recovery | 25 | At least as many valid records recovered as the reference |
| **B** | No false accepts | 25 | **Never** returns a record that fails the schema |
| **C** | Truncation caught | 20 | Every output with a non-`stop` finish reason is reported as `truncated` — including the ones that parse and validate |
| **D** | Cost discipline | 15 | No more `expensive_repair` calls than the reference makes |
| **E** | Nothing dropped | 10 | One result per input, and every non-recovered result carries a reason |
| **F** | Degenerate cases | 5 | Empty batch · all-truncated batch · a truncated output that looks valid · a content-filter refusal |

Pass mark: **80**.

**B is a safety property, not a quality one.** A harness that emits a record
violating the schema has done something worse than failing, because that record
now flows downstream wearing exactly the same clothes as a good one. It is
weighted accordingly.

**C is the criterion that distinguishes this project from lesson 2.4's ladder.**
Roughly a tenth of truncated outputs parse *and* satisfy the schema, so no
amount of parsing or validation reaches them. The finish reason is the only
signal, it arrives free in a response you already paid for, and checking it
first is the whole of the fix.

## The starter

`student.py` ships with the harness most people write first: parse, and if it
parses, accept it. Run the grader against it before changing anything. It
scores **26.2/100**, and the breakdown names the three claims it cannot
support — around 14 schema-violating records accepted per batch, 8 truncated
outputs treated as good, and roughly 20% fewer records recovered than the
reference manages.

## Useful commands

```bash
python -m grader --seed 7 --batches 10     # more batches, different seed
python -m grader --reference --seed 1      # what full marks looks like
python -m grader --sweep 30                # what CI runs
```
