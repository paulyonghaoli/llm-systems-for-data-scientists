---
status: Draft
last_verified: 2026-08-09
volatility: low
---

# Module 0 · From data science to LLM systems

You already know how to fit a model, hold out a test set, argue about a
metric, and be suspicious of a result that looks too good. Almost all of that
transfers. This module is about the parts that do not, and it is deliberately
short: five lessons that change what you are suspicious *of*.

The through-line is that you no longer own the model. It is a rented,
versioned, stochastic, metered service, and everything you used to do to a
model you now have to do to the system around one.

## Lessons

1. [0.1 What changes when the model is someone else's](01-what-changes.md) — **available**
2. [0.2 Anatomy of an LLM application](02-anatomy.md) — **available**
3. [0.3 Why your evaluation habits break](03-evaluation-breaks.md) — **available**
4. [0.4 Cost and latency are features](04-cost-and-latency.md) — **available**
5. [0.5 Lab: the demo that worked](05-lab-demo-that-worked.md) — **available**

## Graded artifact

[Mini-project 0 · The reliability report](project-reliability.md) — turn a raw
run log into the four numbers that decide whether a system is shippable, and
refuse to compute the ones the log cannot support.

## Appendix

[Who does what](roles.md) — a field guide to the roles this work is split
into, and which of them your existing skills already cover.

## What this module assumes

Python, pandas-level data handling, and undergraduate statistics — confidence
intervals, hypothesis tests, and the habit of asking how many samples a claim
rests on. No prior exposure to language models.

It assumes **no API key and no GPU**, here or anywhere else in the curriculum.
