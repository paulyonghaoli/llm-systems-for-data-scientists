---
status: Verified
last_verified: 2026-08-12
volatility: low
pyodide: true
prereqs: ["3.1", "3.2"]
---

# 3.5 · Chunking, and the benefit this corpus cannot show

## A · Why this matters

Chunking is the least examined decision in retrieval. Every guide specifies it
— 512 tokens, 50 overlap, split on paragraphs — and almost none of them
measures it, which is unusual for a parameter that determines what your index
contains.

Measured on this corpus, chunking is a **cost at every size tested**, and the
cost grows as the chunks get smaller:

| | recall@10 |
|---|---:|
| documents left whole | 0.614 <!-- computed: chunking_tradeoff.recall_document --> |
| 510-token chunks | 0.585 <!-- computed: chunking_tradeoff.recall_fixed_510 --> |
| 256-token, 64 overlap | 0.568 <!-- computed: chunking_tradeoff.recall_fixed_256_ov64 --> |
| 128-token chunks | 0.358 <!-- computed: chunking_tradeoff.recall_fixed_128 --> |

The best chunking measured is still
2.9 <!-- computed: chunking_tradeoff.cost_of_best_chunking_pts --> points below
not chunking, and the aggressive one costs
25.6 <!-- computed: chunking_tradeoff.cost_of_128_pts --> points — larger than
any gain anything else in this module produced.

**This is not an argument against chunking, and the distinction is the point of
the lesson.** Chunking exists to solve one problem: a document longer than the
embedding model's context has its tail silently discarded. This corpus has
31 <!-- computed: chunking_tradeoff.docs_over_limit --> documents over the
510-token budget out of
2,419 <!-- computed: chunking_tradeoff.n_documents --> — the condition chunking
is for occurs in
1.3% <!-- computed: chunking_tradeoff.pct_docs_over_limit -->% of it. What you
are seeing is the price of chunking charged against a corpus that has almost
nothing to gain from paying it.

A result measured where the mechanism does not apply is not evidence about the
mechanism. That sentence is worth more than the table.

!!! info "Terms used in this lesson"
    **Chunk** — a contiguous span of a document, indexed as its own unit.

    **Overlap** — tokens repeated between consecutive chunks, so that a
    sentence spanning a boundary appears whole in at least one of them.

    **Pooling** — turning several chunk scores back into one document score.
    Max-pooling takes the best chunk; sum-pooling adds them.

    **Candidate inflation** — the growth in the number of scored units when a
    corpus is chunked. Here 2,419 documents become up to
    7,082 <!-- computed: chunking_tradeoff.units_fixed_128 --> chunks.

    **Truncation** — what the model does with tokens past its context limit:
    discards them, silently, with no error and no flag on the vector.

## B · Mental model

**Chunking does not give the right document more chances to match. It gives
every wrong document more chances to match.**

That asymmetry is the whole cost, and it follows from how chunk scores become
document scores. A document is ranked by its *best* chunk, and a maximum over
more samples is larger in expectation for any document — the correct one and
all 2,418 others alike. Splitting the corpus three ways does not treble the
gold document's chance of being retrieved; it trebles the number of lottery
tickets held by everything it competes against.

The benefit sits on the other side of the ledger and is conditional. When a
document exceeds the model's context, everything past the limit contributes
nothing to its vector: not down-weighted, *absent*. A policy whose relevant
clause sits at token 700 is, to the index, a document that does not contain
that clause. Chunking is the only thing that recovers it, and no amount of
better retrieval will substitute, because the information was discarded before
retrieval began.

**The arithmetic is worth doing, because it is worse than it sounds.** Lesson
3.1 measured a query against a random document at a cosine of 0.659 with a
standard deviation of 0.072, and against its own gold document at 0.731. Treat
an irrelevant document's chunk scores as draws from that background
distribution and ask what its *best* chunk scores, which is what max-pooling
ranks it by. The expected maximum of `n` draws rises steadily: one chunk gives
0.659, three give 0.708, five give 0.729, and ten give **0.755**.

Read that last figure against 0.731. An irrelevant document cut into ten chunks
expects a best-chunk score higher than the average score of a document that
genuinely answers the query. It has not become more relevant; it has bought
more tickets. The margin lesson 3.1 measured at 0.072 is simply not large
enough to survive that, which is why the measured cost of chunking tracks the
number of units so closely.

So the decision is a comparison between a cost that scales with how finely you
split and a benefit that only exists above the context limit. On a corpus of
short documents the cost is all there is. On a corpus of hundred-page
contracts the benefit dominates and the same table would look completely
different.

??? question "If chunking hurts here, why did the 510-token chunking barely hurt at all (0.585 against 0.614)?"
    Because at 510 tokens almost every document becomes exactly one chunk —
    2,450 <!-- computed: chunking_tradeoff.units_fixed_510 --> chunks from
    2,419 documents, or
    1.01 <!-- computed: chunking_tradeoff.units_per_doc_fixed_510 --> per
    document. It is nearly the unchunked index, and it scores nearly the
    unchunked score. The remaining gap comes from the 31 documents that did
    split, and from the transform being refitted on a slightly different
    population.

## C · Mechanism

Fixed-size chunking has three parameters and each one has a failure attached.

**Size.** Measured in *model* tokens, not characters or words, because the
constraint being respected is the model's position budget. Chunking by
characters and hoping is how documents end up truncated anyway.

**Overlap.** A chunk boundary falling mid-sentence splits the evidence, so the
sentence appears in neither chunk as a coherent unit. Overlap of `o` tokens
means consecutive chunks advance by `size − o`, so the number of chunks per
document scales as `len / (size − o)`. At size 256 with overlap 64 that is one
chunk per 192 tokens — the corpus grew from 2,419 units to
4,188 <!-- computed: chunking_tradeoff.units_fixed_256_ov64 -->, which is where
the cost comes from.

**Pooling.** This one is rarely stated and it matters more than overlap.
Max-pooling ranks a document by its best chunk; sum-pooling adds every chunk's
score. Measured here, sum-pooling is catastrophic:

| chunking | max-pool | sum-pool |
|---|---:|---:|
| 510 | 0.585 | 0.472 <!-- computed: chunking_tradeoff.recall_fixed_510_sumpool --> |
| 256/64 | 0.568 | 0.205 <!-- computed: chunking_tradeoff.recall_fixed_256_ov64_sumpool --> |
| 128 | 0.358 | 0.165 <!-- computed: chunking_tradeoff.recall_fixed_128_sumpool --> |

Sum-pooling rewards documents for being *long*, since a long document has more
chunks and every chunk adds something. It re-introduces exactly the length bias
that BM25's `b` parameter exists to remove, in a system that had no length bias
to begin with. Max-pooling has the opposite bias — one lucky chunk carries a
whole document — and it is the lesser of the two by a wide margin here.

The transform from lesson 3.1 is refitted per granularity in this experiment.
The mean of 7,082 chunks is not the mean of 2,419 documents, and applying one
population's mean to another is precisely the mismatch lesson 3.1 warns about.
This is easy to get wrong when chunking is added to a system that already had
a fitted transform.

??? question "Does that order-statistic argument not apply equally to the correct document, which also gains chunks?"
    It does, and the two effects do not cancel. There is one correct document
    and 2,418 wrong ones, so the same rise in expected maximum is applied once
    in your favour and 2,418 times against you. The correct document only has
    to be beaten by *one* of them to drop out of the top ten.

## D · From data science to LLM systems

The nearest thing you have done is feature aggregation over groups: many rows
per entity, one prediction per entity, and a choice of aggregation that
determines what the model can see. Max-pooling chunk scores is `groupby.max()`,
and it carries the same hazard — a single extreme row decides the entity, so
the estimate has the variance of an order statistic rather than of a mean.

The analogy breaks in two ways that matter.

**The aggregation is over candidates, not features, so it changes the
competition.** Adding rows per entity in a grouped model does not affect other
entities. Adding chunks per document adds candidates that compete with every
other document, which is why the cost here scales with the total number of
units rather than with the number per document.

**There is no fitting step to absorb the choice.** In a grouped model, a
downstream learner can partly compensate for a poor aggregation. Retrieval has
no such stage: the pooling rule *is* the ranking, and a bad choice is not
recoverable later.

The habit that transfers unchanged is the useful one. You would never pick an
aggregation without measuring it, and chunk size, overlap and pooling deserve
exactly the treatment you would give `groupby` aggregations in a feature
pipeline — which is to say, a sweep and a held-out comparison rather than a
number from a blog post.

## E · Minimal implementation

Fixed-size chunking with overlap, in model tokens:

```python
def chunk_spans(n_tokens, size, overlap=0):
    """(start, end) spans covering a document of n_tokens."""
    if size <= 0 or overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    spans, start = [], 0
    while start < n_tokens:
        spans.append((start, min(start + size, n_tokens)))
        if start + size >= n_tokens:      # this chunk reached the end
            break
        start += step
    return spans or [(0, 0)]
```

The `break` is the line that repays attention. Without it, a document whose
length is not a multiple of `step` produces a final chunk that starts inside
the previous one and ends at the same place — a duplicate suffix, indexed as
its own unit, competing with its parent. With overlap this happens on most
documents rather than occasionally, and it inflates the index while adding
nothing.

The `spans or [(0, 0)]` fallback covers the empty document. An empty list means
the document has no chunks, so it silently leaves the index entirely — which is
a much worse outcome than one empty chunk that never matches anything.

??? question "Why does the overlap parameter inflate the index faster than the size parameter shrinks it?"
    Because chunks advance by `size − overlap`, so the count scales as
    `len / (size − overlap)` rather than `len / size`. At size 256 the corpus
    grows to 4,025 units; adding an overlap of 64 — a quarter of the chunk —
    takes it to 4,188 by shortening the effective step from 256 to 192. Overlap
    is bought in units, and units are what the cost is denominated in.

## F · Production practice

**Chunk because your documents exceed the context budget, not because chunking
is best practice.** Measure the length distribution first, in model tokens.
If the 95th percentile fits, chunking is buying you very little and this
lesson's table is what you should expect.

**Prefer the largest chunk that fits.** Measured here, quality is monotone in
size, and the cheapest configuration that respects the limit is also the best
one. The instinct to chunk small "so the retrieved context is focused" is
optimising the wrong stage — the generator's context budget is a separate
constraint, and lesson 3.7 addresses it separately.

**Max-pool, and know what it costs you.** Sum-pooling is measurably worse here.
Max-pooling's own bias — one lucky chunk carrying a document — is the reason a
reranker earns its place in lesson 3.6.

**Refit the transform on the units you actually index.** Chunks and documents
are different populations with different means.

**Record the span with every chunk.** `doc_id`, `start`, `end`. Without it you
cannot show a citation, cannot deduplicate overlapping hits from the same
document, and cannot reconstruct what the model was given when something goes
wrong.

??? question "Your documents average 200 tokens and someone proposes chunking at 128 with overlap. What do you say?"
    That it can only cost. Nothing is being truncated at 200 tokens, so there
    is no tail to recover, and chunking would roughly double the unit count to
    buy nothing. The measured version of this proposal is the 128-token row,
    which cost 25.6 points. Ask what problem the chunking is meant to solve
    and check that the corpus has it.

## G · Experiment

`python experiments/chunking_tradeoff.py`.

| granularity | units | per doc | recall@10 | gold truncated | gold fits |
|---|---:|---:|---:|---:|---:|
| document | 2,419 | 1.00 | **0.614** | 0.525 | 0.640 |
| fixed_510 | 2,450 | 1.01 | 0.585 | 0.525 | 0.603 |
| fixed_256_ov64 | 4,188 | 1.73 | 0.568 | 0.500 | 0.588 |
| fixed_256 | 4,025 | 1.66 | 0.409 | 0.300 | 0.441 |
| fixed_128 | 7,082 | 2.93 | 0.358 | 0.175 | 0.412 |

**Recall falls monotonically as units multiply.** 2,419 units score 0.614 and
7,082 score 0.358. The ordering follows the unit count rather than the chunk
size as such, which is what the candidate-inflation account in §B predicts.

**A prediction of mine that the data refused.** I expected the last two columns
to separate: chunking should help the
40 <!-- computed: chunking_tradeoff.queries_gold_truncated --> queries whose
gold document is truncated, because for those the relevant text may be past
the limit. It did not. At 510 tokens both slices score
0.525 <!-- computed: chunking_tradeoff.recall_fixed_510_truncated -->, exactly
the unchunked figure, and every finer chunking is worse on the truncated slice
than on the rest. Whatever the discarded tails contain, it is not the answers
to these queries — which is a property of how this corpus was built rather than
a general fact.

**So the honest reading is narrow.** On a corpus of short documents, chunking
costs between three and twenty-six points and buys nothing measurable. That is
a real finding about this corpus and it is not a finding about chunking, which
is defined by a mechanism this corpus barely contains. The way to know what
chunking does to *your* corpus is to run this comparison on it — which is what
§K asks you to do.

??? question "If max-pooling has this bias, why not rank chunks directly and never pool at all?"
    For some applications you should — if the answer is a passage and you are
    showing passages, chunks are the right unit and there is nothing to pool.
    Pooling exists because the *labels* here are documents, and the metric has
    to match the unit the task is defined over. Choosing the retrieval unit is
    partly a decision about what your users are actually asking for.

## H · Failure modes and cost traps

**Chunking by characters rather than model tokens.** The budget being respected
is the model's, and the ratio of characters to tokens varies with the text. A
900-character chunk is comfortably inside a 512-token budget for English prose
and well outside it for dense code or non-Latin scripts.

**Sum-pooling chunk scores.** Measured here at up to 36 points worse than
max-pooling. It rewards documents for length, in a system that otherwise has no
length bias.

**The duplicate final chunk.** Without the `break` in §E, most documents with
overlap gain a redundant trailing chunk that competes with the document it came
from.

**Reusing a document-fitted transform on chunks.** Different population,
different mean; the correction is then wrong for every chunk.

**Chunking small to "focus" the retrieved context.** This conflates the
retrieval unit with the generation unit. Retrieve at whatever granularity
retrieves best and trim for the generator afterwards; they are different
budgets and lesson 3.7 treats the second one.

**Discarding spans.** Without `start` and `end` you cannot cite, deduplicate,
or reconstruct. This is cheap to record and impossible to recover later.

**Generalising from a corpus without the mechanism.** The trap this lesson
walked into deliberately: it would have been easy to publish "chunking reduces
recall by up to 26 points" as a finding. It is a finding about 2,419 short
documents, and the sentence that makes it useful is the one naming the
condition under which it would not hold.

## I · Graded practice

<quiz-bank src="ret-l5"></quiz-bank>

<code-exercise src="ret-l5-spans"></code-exercise>

<code-exercise src="ret-l5-pooling"></code-exercise>

## J · Annotated references

- **Pinecone and LlamaIndex chunking guides (2023–2024)** — read as primary
  sources for the conventional numbers, and note how few of them report a
  measurement on a labelled query set.
- **Lewis et al. (2020), "Retrieval-Augmented Generation"** — the original
  setup used 100-word passages, a choice inherited far more widely than the
  reasoning behind it.
- **Robertson and Zaragoza (2009)** — §3 on length normalisation, which is the
  same problem sum-pooling reintroduces from a different direction.
- **Kamradt, "Levels of Text Splitting" (2024)** — a useful taxonomy of
  strategies beyond fixed-size, including semantic splitting, which this lesson
  does not measure and which the extension asks you to.

## K · Extension

*Off-platform, an afternoon.* Run §G's comparison on your own corpus, and
report the document length distribution alongside it in model tokens. Two
things are worth knowing: what fraction of your documents exceed the embedding
budget, which tells you whether chunking can help at all; and whether a
semantic splitter beats fixed-size at the same unit count — the comparison to
control is *units*, not chunk size, since this lesson's result says most of the
cost is carried by how many candidates you create.
