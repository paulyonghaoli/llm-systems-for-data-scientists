---
status: Verified
last_verified: 2026-08-12
volatility: low
pyodide: true
prereqs: ["3.3", "3.4"]
---

# 3.6 · Reranking, and the metric that hides it

## A · Why this matters

Here is the same reranker, measured on the same 176 queries, over the same
lexical first stage, at two different cutoffs:

| | before | after | | |
|---|---:|---:|---|---|
| recall@1 | 0.051 <!-- computed: reranking.lexical_at1 --> | **0.250 <!-- computed: reranking.lexical_rerank_at1 -->** | won 38, lost 3 | p below 0.0001 <!-- computed: reranking.lexical_pfloor_at1 --> |
| recall@10 | 0.705 <!-- computed: reranking.lexical_at10 --> | 0.727 <!-- computed: reranking.lexical_rerank_at10 --> | won 5, lost 1 | p = 0.22 <!-- computed: reranking.lexical_pfloor_at10 --> |

The right answer arrives **first**
4.9 <!-- computed: reranking.at1_gain_x -->× as often, and whether it appears in
the top ten at all barely moves. Both rows are true, both are the same
experiment, and a team that had standardised on recall@10 would have measured
this reranker, found
2.2 <!-- computed: reranking.at10_gain_pts --> points at p = 0.22, and
concluded that cross-encoders were not worth the latency.

That is the lesson. A reranker reorders a candidate list; it cannot retrieve
anything the first stage missed. So its entire effect lives in the ordering,
and a metric that only asks about set membership at rank ten is looking in the
one place the effect is not. This is the most consequential measurement error
in the module, and it is invisible — every number is correctly computed.

!!! info "Terms used in this lesson"
    **Bi-encoder** — the retrieval model of lessons 3.1–3.5. Query and document
    are embedded *separately*, so document vectors can be computed once and
    indexed.

    **Cross-encoder** — query and document go into the model **together** and
    one relevance score comes out. Much more accurate, and impossible to index:
    there is no document vector, so every candidate costs a forward pass at
    query time.

    **First stage** — the cheap retriever that produces candidates. Its recall
    is the ceiling on everything downstream.

    **Candidate depth** — how many documents are passed to the reranker. The
    main cost dial: reranker cost is linear in it.

    **MRR@k** — mean reciprocal rank. Averages `1/position` of the first correct
    result, so unlike recall@k it is sensitive to *where* in the list the answer
    sits. Lesson 3.8 covers the family.

## B · Mental model

**A reranker is a second opinion that is too expensive to ask about everything,
so you ask it only about the shortlist — and it can only reorder the
shortlist.**

Two consequences follow, and both are measured below.

**Its ceiling is the first stage's recall at the candidate depth.** The
recorded candidates here contain the gold document for
94.9% <!-- computed: reranking.candidate_ceiling_pct -->% of queries, and no
reranking of them can exceed that. The reranked hybrid reaches
0.801 <!-- computed: reranking.hybrid_rerank_at10 -->, which is
14.8 <!-- computed: reranking.reranker_shortfall_pts --> points short — so the
reranker is not merely bounded by its candidates, it is also imperfect within
them.

**Its benefit is concentrated where the first stage is badly *ordered*, not
where it is badly *recalled*.** That predicts which runs it should help, and
the prediction holds. Gate 25 established that BM25's top-ranked document is a
non-gold distractor for 94% of the distractor queries by construction; lexical
recall@1 is
0.051 <!-- computed: reranking.lexical_at1 -->, which is close to useless
ordering on top of respectable recall. There is enormous room to reorder, and
the reranker takes it. The dense run starts at
0.205 <!-- computed: reranking.dense_at1 --> — already four times better
ordered — and reranking it produces nothing significant at any cutoff.

??? question "Why can a cross-encoder not simply replace the retriever?"
    Because there is nothing to index. A bi-encoder embeds each document once,
    ahead of time, and query time is a matrix multiply. A cross-encoder scores
    a *pair*, so answering one query over this corpus would need 2,419 forward
    passes rather than 78 — and over a million documents it is not a latency
    problem, it is an impossibility. The two-stage design exists because the
    accurate model cannot be precomputed.

## C · Mechanism

A bi-encoder produces `f(query)` and `f(document)` independently and compares
them with a dot product. Everything the comparison can use has to survive being
compressed into 384 numbers *before either text has seen the other*. Lesson 3.1
measured what that costs: the margin between a correct document and an
arbitrary one was 0.072.

A cross-encoder takes `g(query, document)` — both texts concatenated into one
input, with attention running across the join — and emits a single logit.
Because the model sees the query while reading the document, it can resolve the
things a vector cannot: which of two similar policies mentions *this* topic,
whether a number in the document answers the number the query asked for,
whether an apparent match is the superseded version.

The scores reflect that. On a probe pair from this corpus the relevant document
scored **+1.85** and an unrelated one **−9.91** — a separation nothing in the
bi-encoder's geometry comes close to, where every document sat within 0.3 of
every other. The logits are unbounded and not comparable across models, so they
are useful for ranking within one query and for nothing else.

**Cost is linear in candidates, and that is the whole design constraint.** This
fixture recorded
78.4 <!-- computed: reranking.pairs_per_query --> pairs per query, which is the
union of two retrievers' top
50 <!-- computed: reranking.depth_per_retriever -->. Each is a transformer
forward pass over up to 512 tokens. Serving that at a hundred queries a second
means ~7,800 forward passes a second, which is a GPU-sized problem, and it is
why candidate depth is the first thing tuned when a reranked system is too
slow — and why §H warns about what that tuning silently costs.

**The cost is worth putting in units you can budget with, because the ratio is
larger than it sounds.** A bi-encoder query against this corpus is a single
matrix multiply of 2,419 by 384, which lesson 3.2 measured at roughly a tenth
of a millisecond, whereas reranking the same query means
78.4 <!-- computed: reranking.pairs_per_query --> transformer forward passes
over sequences of up to 512 tokens. Measured on the machine that recorded this
fixture, the cross-encoder managed about 318 pairs per second, so a single
query's reranking takes on the order of a quarter of a second — roughly two
thousand times the retrieval it follows. That ratio is why the two-stage design
exists at all, and it is also why candidate depth is the parameter everyone
reaches for when a reranked system misses its latency budget, since halving the
depth halves the dominant cost and appears to change nothing.

## D · From data science to LLM systems

This is a cascade, and you have built one: a cheap model filters, an expensive
model decides. Fraud systems do it, ad ranking does it, and the reasoning is
identical — the expensive model's accuracy is only affordable on a shortlist,
so the cheap model's job is recall rather than precision.

The analogy holds well and breaks in two places.

**The stages optimise different metrics, and saying so is not optional.** The
first stage should be tuned for recall at the candidate depth, because anything
it drops is unrecoverable; the second stage should be tuned for precision at
the cutoff you actually serve. Tuning both on the same metric is the standard
mistake, and it usually means tuning the first stage for precision, which
throws away candidates the reranker would have rescued.

**There is no joint training, so the stages can disagree about what relevance
means.** In a learned cascade both stages see the same labels. Here the
first-stage embeddings were trained on one distribution and the cross-encoder
on MS MARCO, and neither has seen your corpus. That is why the measurement
matters more than the architecture: the reranker helped the lexical run
enormously and the dense run not at all, and no amount of reasoning about
cascades would have told you which.

The habit that transfers unchanged: **measure the second stage at the operating
point**. You would not evaluate a fraud model at a threshold you never use.

??? question "The reranker did nothing for the dense run. Would you conclude that dense retrieval does not need reranking?"
    Only for this first stage on this corpus, because the finding is about how
    well ordered the input already was rather than about dense retrieval as a
    class. A different embedding model, or the same one on a corpus whose
    distractors it handles worse, could easily produce a badly ordered top ten
    that a cross-encoder would repair. The general claim the measurement
    supports is narrower and more useful: a reranker pays in proportion to how
    much reordering its input needs, so measure the input's ordering before
    buying the second stage.

## E · Minimal implementation

Reranking is a sort. The interesting parts are the defaults.

```python
def rerank(candidates, scores, limit=10, missing=-99.0):
    """Reorder candidates by cross-encoder score, best first."""
    # `missing` matters: a candidate with no recorded score must sink rather
    # than float. Using 0.0 would place unscored documents above every
    # genuinely-scored negative, and these logits are frequently negative --
    # an irrelevant document in this corpus scores around -10.
    return sorted(candidates,
                  key=lambda d: (-scores.get(d, missing), d))[:limit]
```

The tie-break on `d` is the same reproducibility guard as lessons 3.3 and 3.4.
The `missing` default is the one that bites: rerankers are applied to candidate
sets assembled from several retrievers, and it is easy to end up with a
document nobody scored. A default of `0.0` looks neutral and is not, because
the scale is not centred on zero — most documents score well below it.

Note what is *not* here. There is no re-retrieval, no score combination with
the first stage, and no normalisation. The first stage's scores are discarded
entirely once the reranker has spoken, which is the usual design and is worth
questioning when the first stage is strong: interpolating the two is a real
technique and this lesson does not measure it.

## F · Production practice

**Measure at the cutoff you serve.** If the application shows three results,
recall@10 is not a proxy for anything a user experiences. The table in §A is
the whole argument.

**Set candidate depth by measuring the first stage's recall at that depth.**
The reranker cannot exceed it. Here the candidates contain the answer 94.9% of
the time, so depth is not the binding constraint — the reranker's own accuracy
is, at 14.8 points below its ceiling.

**Do not tune the first stage for precision.** Its job is to not lose the
answer. Precision is the second stage's problem, and a first stage tuned to put
the right thing first is often one that has dropped candidates to do it.

**Budget for it honestly.** ~78 forward passes per query at up to 512 tokens is
several orders of magnitude more compute than the vector lookup it follows.
That is the price of the rank-1 improvement, and it should be compared against
alternatives — including simply showing ten results instead of three.

**Re-measure when either stage changes.** The reranker helped one first stage
enormously and another not at all. That relationship is empirical and will not
survive a change to the retriever.

??? question "Why tune the first stage for recall rather than precision, when precision is what users see?"
    Because the two stages are responsible for different failure modes, and
    only one of them is recoverable. A document the first stage never returns
    is invisible to everything downstream, whereas a poor ordering is exactly
    what the second stage is there to repair. Tuning the first stage for
    precision usually means tightening it until it drops candidates, and those
    candidates are disproportionately the ones a reranker would have promoted.

## G · Experiment

`python experiments/reranking.py`, over the recorded fixture.

| first stage | @1 | @3 | @5 | @10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| lexical | 0.051 | 0.466 | 0.517 | 0.705 | 0.287 |
| lexical + rerank | **0.250** | **0.670** | **0.716** | 0.727 | **0.459** |
| dense | 0.205 | 0.483 | 0.551 | 0.614 | 0.356 |
| dense + rerank | 0.227 | 0.511 | 0.580 | 0.642 | 0.381 |
| hybrid | 0.057 | 0.551 | 0.602 | 0.767 | 0.314 |
| hybrid + rerank | **0.233** | **0.665** | **0.750** | 0.801 | **0.454** |

**The effect is enormous and it is entirely in the ordering.** On the lexical
run, recall@5 goes from 0.517 to
0.716 <!-- computed: reranking.lexical_rerank_at5 --> winning
35 <!-- computed: reranking.lexical_won_at5 --> queries and losing
0 <!-- computed: reranking.lexical_lost_at5 -->, at
p below 0.0001 <!-- computed: reranking.lexical_pfloor_at5 -->. At rank ten the same
comparison is 5 against 1 at p = 0.22.

**It does nothing measurable for the dense run.** Every cutoff sits between
p = 0.5 and p = 0.65. The mechanism from §B predicts this: the dense run is
already reasonably ordered at the top, and a reranker's value is proportional
to how badly ordered its input is. It is a useful negative result — "add a
reranker" is not advice that applies uniformly.

**And it does not reach its own ceiling.** The candidates contain the gold
document 94.9% of the time; the best reranked configuration reaches 0.801. The
cross-encoder is more accurate than the retriever and it is not an oracle,
which is worth remembering before treating its scores as ground truth.

??? question "Could you use the cross-encoder's scores to decide when to answer at all, abstaining below some threshold?"
    Not with a fixed threshold, since these logits are unbounded and their
    scale shifts between queries, so a cutoff calibrated on one set of queries
    will not transfer to another. Abstention needs a calibrated probability
    rather than a raw logit, which means fitting a mapping on labelled data and
    re-fitting it whenever the model changes. Capstone I scores abstention
    directly, and it does so against the corpus's unanswerable queries rather
    than against a score threshold.

## H · Failure modes and cost traps

**Evaluating a reranker at recall@10.** Measured here: the same change is
p = 5.8e-11 at rank five and p = 0.22 at rank ten. Every number is computed
correctly and the conclusion is inverted.

**Reranking a first stage tuned for precision.** The reranker's ceiling is what
the first stage handed it. A first stage optimised to put the right document
first is often one that has dropped candidates to achieve it, and those are
exactly what the reranker was going to rescue.

**Defaulting missing scores to zero.** These logits are mostly negative, so an
unscored document at 0.0 outranks nearly everything legitimately scored. This
appears when candidate sets are assembled from more sources than were scored.

**Treating cross-encoder scores as calibrated.** They are unbounded logits,
comparable within one query and not across queries or models. A threshold like
`> 0` is arbitrary and will not transfer.

**Cutting candidate depth to hit a latency target.** It is the obvious dial and
the cost is invisible in any metric that does not look at the tail of the
candidate list — the same failure as lesson 3.2's `nprobe`, one stage later.

**Assuming it will help.** It did not help the dense run at any cutoff, at any
significance. The only way to know is to measure it against your first stage,
at your cutoff.

??? question "Reranking reaches 0.801 while its candidates contain the answer 94.9% of the time. Where would you spend effort next?"
    On the reranker rather than on the first stage, because the 14.8-point gap
    between them is entirely within the second stage's control. Deepening
    candidates raises a ceiling that is not currently binding, so it would cost
    latency and buy nothing. A stronger cross-encoder, or one fine-tuned on
    in-domain pairs, is where the remaining headroom actually sits.

## I · Graded practice

<quiz-bank src="ret-l6"></quiz-bank>

<code-exercise src="ret-l6-rerank"></code-exercise>

<code-exercise src="ret-l6-cutoff"></code-exercise>

## J · Annotated references

- **Nogueira and Cho (2019), "Passage Re-ranking with BERT"** — the paper that
  established the two-stage design. Short, and the ablations are the useful
  part.
- **Nogueira et al. (2020), "Document Ranking with a Pretrained
  Sequence-to-Sequence Model"** — monoT5, and a good account of why a
  cross-encoder can use information a bi-encoder structurally cannot.
- **Khattab and Zaharia (2020), "ColBERT"** — the middle ground: late
  interaction keeps most of the accuracy while remaining indexable. The right
  next thing to read if §C's cost argument bothers you.
- **Thakur et al. (2021), "BEIR"** — reranking results across eighteen
  datasets, and a good calibration for how much the gain varies by domain.

## K · Extension

*Off-platform, an afternoon and a GPU or patience.* Take your own first stage
and reranker, and produce §G's table for your corpus: every cutoff you might
serve, both stages, with paired tests. Then answer the question this lesson
cannot: does interpolating the first-stage score with the reranker score beat
using the reranker alone? The fixture here discards the retrieval score
entirely, which is conventional and, on a strong first stage, possibly wrong.
