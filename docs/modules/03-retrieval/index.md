---
status: Draft
last_verified: 2026-08-12
volatility: low
---

# Module 3 · Embeddings, retrieval and RAG

This is the module where the curriculum stops being about one model call and
starts being about a system. Retrieval is the part of an LLM application that
most often decides whether the whole thing works, and it is also the part with
the most measurable ground truth — which makes it the best-value module in
Course I and the one with the least room for hand-waving.

Everything here runs against a single corpus, described in
[`data/corpus/README.md`](https://github.com/paulyonghaoli/llm-systems-for-data-scientists/blob/main/data/corpus/README.md):
2,419 documents from a fictional logistics company, 200 labelled queries, and
seven difficulties deliberately planted and then *measured* rather than
asserted. Real embeddings from a pinned, permissively licensed model are
recorded once and shipped quantised, so every exercise runs in the browser with
no API key and no network.

!!! warning "The corpus is honest about what it is"
    It is synthetic, and its prose is template-generated. It rewards
    retrievers that handle the seven difficulties planted in it and says
    nothing about difficulties nobody thought to plant. Every lesson that
    reports a number from it says so.

## What this module does not claim

Building it produced a result worth putting at the top rather than burying.
The plan for this curriculum originally asserted that dense retrieval
*resolves* the vocabulary mismatch that lexical search cannot handle. Measured
on this corpus, BM25 finds the answer for 3.3% of those queries and dense
retrieval for 30.0% — nine times better, and still missing seven out of ten.

That claim was withdrawn rather than the measurement being softened. It is
also the reason this module has eight lessons instead of three: hybrid
retrieval, reranking and careful chunking are not refinements on a working
system, they are what closes a gap this large.

## Lessons

1. [3.1 Embedding geometry: the space is not shaped how you think](01-embedding-geometry.md) — **available**
2. [3.2 Exact and approximate search](02-exact-and-approximate.md) — **available**
3. [3.3 BM25, and why the old baseline still wins](03-bm25.md) — **available**
4. [3.4 Hybrid retrieval, and what fusion really recovers](04-hybrid-fusion.md) — **available**
5. [3.5 Chunking, and the benefit this corpus cannot show](05-chunking.md) — **available**
6. [3.6 Reranking, and the metric that hides it](06-reranking.md) — **available**

Lessons 3.7–3.8 (groundedness and citation, retrieval metrics) are in progress.
