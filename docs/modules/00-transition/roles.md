---
status: Draft
last_verified: 2026-08-09
volatility: high
---

# Appendix · Who does what

A short field guide to how this work is divided, and which parts your existing
skills already cover. Job titles in this field are unstable and mean different
things at different companies, so this describes **work**, not titles.

Treat it as a map for deciding what to learn next, not as career advice — and
note the volatility flag at the top of this page.

## The five kinds of work

**Applied / product.** Building features on top of models somebody else
trained: retrieval, agents, structured output, the evaluation that says
whether it works. Courses I and IV.
*What you already have:* most of it. Data handling, evaluation design,
statistical scepticism. What is missing is the systems half — Module 0.2 is
the shape of it.

**Evaluation.** Designing the measurements everything else is judged by:
harnesses, contamination detection, judge calibration, regression gates.
Course IV.
*What you already have:* nearly everything. This is applied statistics with an
unusual measurement instrument. It is also, per the note below, the least
crowded of these five.

**Data infrastructure.** Corpus construction, deduplication, decontamination,
quality filtering, provenance and lineage. Course III, Module 9.
*What you already have:* a lot, if you have done pipeline work. The
techniques — MinHash, LSH, near-duplicate detection — are ordinary data
engineering applied at unusual scale.

**Inference / serving.** Making models fast and cheap to run: batching, KV
cache, quantization, throughput against latency budgets. Course III, Modules
11–12.
*What you already have:* less. This is systems performance work, and the
transferable core is the performance model rather than the kernel code —
which is why this curriculum teaches it as cost models.

**Training research.** Pre-training, adaptation, preference optimisation.
Course II.
*What you already have:* the mathematics, probably. What is hard to acquire
outside a lab is experience at scale, and that is a genuine barrier rather
than a knowledge gap.

## An honest note on which door to knock on

Training research overwhelmingly hires people who have already trained at
scale, which is circular and hard to break into from outside. Course II is in
this curriculum because the material is load-bearing for everything after it —
you cannot reason about KV cache size without understanding attention — not
because it is a plausible hiring path from self-study.

Evaluation, data infrastructure and serving hire on demonstrated skill, are
less crowded, and are the areas where a person arriving from data science has
the largest genuine head start. If the goal is work rather than curiosity,
Courses I and IV are the ones to finish first. That is also the build order
this project follows, and for the same reason.

## What this appendix is not

It is not a salary survey, a company ranking, or a prediction about which
skills will matter in three years. Those change faster than anything else on
this site and none of them can be verified by running code, which is the
standard everything else here is held to.
