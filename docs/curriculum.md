# Curriculum map

Four courses, 16 modules, 88 lessons, four capstones — about 540 learner-hours,
or 12 semester credit-hours. The governing document is `PLAN.md` in the
repository; this page is the reader-facing view of it.

**Reader order is I → II → III → IV. Build order is not the same thing**, and
because course and module numbers are fixed from the start, build order can
change without breaking a single link.

Nothing below is linked until it is actually built.

## Course I · Working with LLMs

*Build and evaluate a retrieval or agent feature that survives contact with a
real user.* ≈ 135 hrs.

| Module | Status |
|---|---|
| **0 · From data science to LLM systems** | [**complete**](modules/00-transition/index.md) |
| **1 · Tokens, sampling and the API contract** | [**complete**](modules/01-tokens/index.md) |
| **2 · Prompting and structured output** | [**2.1–2.3 available**](modules/02-prompting/index.md) · 2.4–2.5 planned |
| 3 · Embeddings, retrieval and RAG | planned |
| 4 · Agents and tool use | planned |
| **Capstone I · Grounded assistant** | planned |

## Course II · Training and adaptation

*Reason about, train and adapt a model, and be right about what it costs.*
≈ 135 hrs.

| Module | Status |
|---|---|
| 5 · Transformer internals | planned |
| 6 · Training dynamics and memory arithmetic | planned |
| 7 · Adaptation — SFT, LoRA, QLoRA | planned |
| 8 · Preference optimization and scaling laws | planned |
| **Capstone II · Train it, then adapt it** | planned |

## Course III · AI infrastructure

*Data pipelines, distributed training as cost models, serving, quantization.*
≈ 135 hrs.

| Module | Status |
|---|---|
| 9 · Data pipelines and curation | planned |
| 10 · Distributed training as a cost model | planned |
| 11 · Inference serving | planned |
| 12 · Quantization and compression | planned |
| **Capstone III · Serving simulator** | planned |

## Course IV · Evaluation and production

*Design evaluations that catch real regressions and do not cry wolf.* ≈ 135 hrs.

| Module | Status |
|---|---|
| 13 · Eval design | planned |
| 14 · Judges and regression gates | planned |
| 15 · Production | planned |
| **Capstone IV · The harness that catches the regression** | planned |

## Status ladder

Every lesson declares one of four states, plus a last-verified date:

| State | Means |
|---|---|
| **Draft** | Written, not checked |
| **Reviewed** | Re-read on a different day; claims checked against sources |
| **Verified** | Every code path runs; every quoted number produced by a script |
| **Reproducible** | The above, plus regenerated from scratch in CI on every commit |

A lesson also carries a volatility flag. Anything marked volatile is re-audited
on a schedule, and CI fails if it goes stale.
