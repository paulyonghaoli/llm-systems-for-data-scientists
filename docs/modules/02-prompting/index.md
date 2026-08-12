---
status: Draft
last_verified: 2026-08-09
volatility: low
---

# Module 2 · Prompting and structured output

Prompting has a reputation as the unserious part of this field, and the
reputation is half deserved: advice about wording dates in months and most of
it was never measured. This module is about the other half — the parts that
are engineering, have correct answers, and can be checked.

So the module deliberately does **not** teach you phrasings. It teaches the
machinery around the phrasing: how a prompt is assembled from untrusted
parts, how examples are selected, what voting over samples actually buys, and
how to make output conform to a schema rather than hoping it does.

**No exercise in this curriculum scores your prompt text by feeding it to a
model we wrote.** That would grade your ability to reverse-engineer our fake.
Where prompt quality genuinely matters, the lesson says so and points at a
measurement you run against a real model on your own data.

## Lessons

1. [2.1 Instruction structure and the trust boundary](01-instruction-structure.md) — **available**
2. [2.2 Few-shot selection is a retrieval problem](02-fewshot-selection.md) — **available**
3. [2.3 Decomposition, chaining and self-consistency](03-decomposition.md) — **available**
4. 2.4 Structured output: schema, validation, repair — *planned*
5. 2.5 Constrained decoding: the logit mask — *planned*

## Graded artifact

*Planned* — a structured-extraction harness scored on schema conformance,
repair rate and cost.

## What this module assumes

Modules 0 and 1. In particular [1.3](../01-tokens/03-context-windows.md), because
prompt assembly and token budgeting are the same problem seen from two sides.
