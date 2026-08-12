---
status: Draft
last_verified: 2026-08-09
volatility: low
---

# Module 1 · Tokens, sampling and the API contract

Everything you send and everything you get back crosses one boundary, and this
module is about what that boundary actually guarantees. Not much, it turns
out, and the parts that are guaranteed are denominated in units you have to
compute rather than assume.

The module is built around a single habit: **measure the thing you are billed
for, on your own data, rather than estimating it.**

## Lessons

1. [1.1 Tokens are not words: BPE from scratch](01-bpe.md) — **available**
2. [1.2 Sampling: temperature, top-k, top-p](02-sampling.md) — **available**
3. [1.3 Context windows and chat templates](03-context-windows.md) — **available**
4. [1.4 The API contract](04-api-contract.md) — **available**
5. [1.5 Costing a design before you build it](05-costing-a-design.md) — **available**

## Graded artifact

[Mini-project 1 · The context packer](project-tokenizer.md) — fit a document
set into a context window and be exactly right about how much fits. Scored
against a published rubric on seeded inputs.

## What this module assumes

Python, and the habit of measuring rather than estimating. No prior exposure
to language models of any kind.
