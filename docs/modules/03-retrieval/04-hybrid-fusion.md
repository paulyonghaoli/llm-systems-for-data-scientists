---
status: Verified
last_verified: 2026-08-12
volatility: low
pyodide: true
prereqs: ["3.3"]
---

# 3.4 · Hybrid retrieval, and what fusion really recovers

## A · Why this matters

Lesson 3.3 ended on an inviting number. Lexical retrieval and dense retrieval
fail on different queries, and *either one finding the answer* reaches
0.858 <!-- computed: hybrid_fusion.union_ceiling --> against
0.705 <!-- computed: hybrid_fusion.lexical_recall --> for the better of the two
alone. There are
15.3 <!-- computed: hybrid_fusion.headroom_pts --> points lying on the table,
and combining two retrievers you already have is the obvious way to pick them
up.

This lesson is what happened when I tried. Three results, and the order matters
because each one revises the last.

**Reciprocal rank fusion with the parameters everyone ships scores
0.642 <!-- computed: hybrid_fusion.rrf_equal_k60 -->.** That is
6.3 <!-- computed: hybrid_fusion.default_loses_to_lexical_by_pts --> points
*worse* than not fusing at all. Adding a second retriever, using the constant
from the paper that introduced the method, made the system worse than the
component it started from.

**Tuned, it looks excellent.** Weighting the retrievers and choosing the rank
constant reaches
0.778 <!-- computed: hybrid_fusion.best_insample -->, winning
13 <!-- computed: hybrid_fusion.insample_won --> queries against lexical alone
and losing 0 <!-- computed: hybrid_fusion.insample_lost -->, at
p = 0.0002 <!-- computed: hybrid_fusion.p_insample_vs_lexical -->. On any
reasonable reading that is a decisive result.

**It does not survive honest evaluation.** That configuration was chosen from
25 <!-- computed: hybrid_fusion.grid_size --> candidates scored on the very
queries it is reported on. Tune on half the queries, evaluate on the other
half, and fusion gets
0.750 <!-- computed: hybrid_fusion.heldout_recall --> — still above lexical
alone, but by a margin of
14 <!-- computed: hybrid_fusion.heldout_won --> queries won to
6 <!-- computed: hybrid_fusion.heldout_lost --> lost, which a paired test puts
at p = 0.1153 <!-- computed: hybrid_fusion.p_heldout_vs_lexical -->. Not
significant. The honest summary is that tuned hybrid retrieval was **not
demonstrated** to beat the better single retriever on this benchmark, while the
default configuration was clearly demonstrated to be worse than it.

!!! info "Terms used in this lesson"
    **Fusion** — combining several ranked lists into one. The inputs are runs;
    the output is a run.

    **Reciprocal rank fusion (RRF)** — score each document as the sum over
    retrievers of `1 / (k + rank)`, using only its *position* in each list and
    never its score.

    **Rank constant `k`** — the damping term in that formula. Large `k`
    flattens the difference between rank 1 and rank 20; small `k` makes the
    very top of each list dominate.

    **Score normalisation** — the alternative to rank-based fusion: map each
    retriever's scores onto a common scale, then add them.

    **Headroom** — the gap between the better single retriever and the union
    of both, which is what perfect fusion would achieve.

    **Optimism** — the amount by which a result measured on the data used to
    choose it overstates the result on new data. Measured here at
    2.8 <!-- computed: hybrid_fusion.optimism_pts --> points.

## B · Mental model

**Fusion is a vote between retrievers, and RRF is a vote in which every
retriever gets one ballot regardless of how much it knows.**

That framing predicts both halves of the result. When two retrievers are
comparably good, pooling their opinions cancels their independent mistakes and
the vote is better than either. When one is meaningfully better than the other
— here 0.705 against
0.614 <!-- computed: hybrid_fusion.dense_recall --> — an equal vote drags the
stronger one toward the weaker, because the weaker retriever is *confident*
about its wrong answers. Its rank-1 mistake gets exactly the weight of the
strong retriever's rank-1 correct answer.

The second thing the framing predicts is why RRF ignores scores in the first
place. BM25 scores are unbounded sums over query terms; cosine similarities sit
in a narrow band near 0.65, as lesson 3.1 measured. These live on
incomparable scales, and there is no principled conversion between them —
normalising each to its own observed range makes them comparable *within a
query* and still leaves you asserting that the top of one list means the same
as the top of the other. Rank is the one thing both retrievers genuinely agree
on the meaning of, which is the case for RRF and also the reason it throws away
the information that would tell it which retriever to trust.

??? question "If both retrievers rank the gold document 11th, can any fusion of them find it at k = 10?"
    No. Fusion is a re-ordering of what it was given, so its ceiling is the
    union of the two candidate lists — which is why this experiment fuses the
    top 100 <!-- computed: hybrid_fusion.depth --> from each rather than the
    top 10. Fusing shallow lists caps the achievable gain before the fusion
    rule gets a say, and it is a common way to make a fusion experiment
    conclude that fusion does not work.

## C · Mechanism

RRF assigns each document a score summed over the runs it appears in:

$$
\text{RRF}(d) = \sum_{r \in \text{runs}} \frac{w_r}{k + \text{rank}_r(d)}
$$

with `w_r = 1` for every run in the original formulation. A document absent
from a run contributes nothing from it rather than a penalty, so appearing
respectably in both lists beats appearing at the top of one and nowhere in the
other — once `k` is large.

**What `k` does.** At `k = 60`, rank 1 contributes `1/61 = 0.0164` and rank 20
contributes `1/80 = 0.0125`; the whole top twenty is worth within thirty per
cent of the same amount, so position barely matters and what matters is
*appearing in both lists*. At `k = 1`, rank 1 contributes `0.5` and rank 20
contributes `0.048` — a factor of ten — so the fused ranking is dominated by
whatever each retriever put first. The parameter is not a smoothing detail; it
chooses between two different algorithms, one that rewards agreement and one
that rewards confidence.

That is why the tuned value here is
1 <!-- computed: hybrid_fusion.best_k --> rather than 60. When one retriever is
substantially better, you want the fusion to defer to strong individual
opinions rather than to demand consensus, because consensus with a weaker
retriever is exactly what you are trying to avoid.

Worth doing the arithmetic once on the exercise's own data, because the
behaviour reverses. Document `d1` is first in the lexical run and absent from
the dense one; `d2` is second in the lexical run and second in the dense one.
At `k = 60` with a lexical weight of 0.7, `d1` scores `0.7/61 = 0.0115` while
`d2` scores `0.7/62 + 0.3/62 = 0.0161`, so the document both retrievers found
wins. At `k = 1` the same two documents score `0.7/2 = 0.35` and
`0.7/3 + 0.3/3 = 0.333`, and the ordering flips. Nothing about the retrievers
changed; a constant did.

**Weighting.** Giving the runs unequal weights `w_r` restores the information
RRF discarded — not the scores, but the standing prior that one retriever is
better. The tuned weight here is
0.7 <!-- computed: hybrid_fusion.best_w_lex --> on the lexical run, which is
close to the ratio of the two retrievers' recall and is what you would guess if
you were guessing.

**The alternative.** Score normalisation — min-max or z-score each run, then
add — uses more information and is more fragile. A single outlying score
compresses everything else, the normalisation depends on how deep a list you
took, and the mapping has to be recomputed per query because BM25 scores are
not comparable across queries either. RRF's indifference to scores is a genuine
robustness property, bought at the price of the very information that would
have told it which retriever deserved the larger ballot.

??? question "Both folds chose k = 1 but different weights. Which part of the tuning would you keep?"
    The `k`. A parameter that two independent halves of the benchmark agree on
    is supported by evidence; one they disagree on is being chosen by noise.
    Keeping the replicated part and defaulting the unstable part is more
    defensible than either accepting both or discarding both — and it is the
    same reasoning you would apply to a feature that appeared in every
    cross-validation fold against one that appeared in half of them.

## D · From data science to LLM systems

This is ensembling, and the intuition transfers almost intact. Averaging
uncorrelated errors is why bagging works, and lesson 3.3's measurement that
each retriever answers queries the other misses is exactly the decorrelation
condition an ensemble needs. If you have ever seen a blend of two models beat
both, you have seen this mechanism.

Three differences matter here, and the third is the one that produced the
result above.

**There is no training signal, so the weights are hyperparameters rather than
fitted quantities.** A stacking ensemble learns its blend weights from
out-of-fold predictions; hybrid retrieval has a handful of labelled queries at
best, and the weight is chosen by search over a small grid. That makes the
search itself a modelling decision with all the usual consequences.

**Rank is not probability.** Ensembling classifiers averages calibrated
scores; RRF averages positions, which discards magnitude entirely. There is no
retrieval equivalent of a well-calibrated probability to average, which is why
this field reaches for rank fusion at all.

**The evaluation set is tiny, so tuning on it is far more dangerous than you
are used to.** With a hundred thousand rows, selecting among 25 configurations
on the test set inflates your estimate slightly. With
176 <!-- computed: hybrid_fusion.n_queries --> queries it inflated it by
2.8 <!-- computed: hybrid_fusion.optimism_pts --> points and turned a p-value
of 0.0002 into one of 0.1153. The habit that protects you — never report a
number measured on the data you used to choose it — is a habit you already
have, and retrieval work routinely abandons it because the query set feels too
small to split. That is exactly backwards.

## E · Minimal implementation

Weighted RRF in full:

```python
from collections import defaultdict

def wrrf(runs, weights, k=60, limit=10):
    """Fuse ranked lists of document ids into one ranking."""
    scores = defaultdict(float)
    for run, w in zip(runs, weights):
        for rank, doc in enumerate(run, start=1):   # rank is 1-based
            scores[doc] += w / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [doc for doc, _ in ranked[:limit]]
```

Two details carry more weight than their size suggests. `enumerate(run, start=1)`
matters because a zero-based rank makes the top document's contribution
`w / k` rather than `w / (k + 1)`, which at `k = 1` is a fifty per cent error on
the single most important term and at `k = 60` is invisible — a bug that hides
at the default and appears when you tune.

The tie-break on `kv[0]` matters for the same reason it did in lesson 3.3.
Documents that appear in neither list at the same position collide constantly
under RRF, because the score is a sum of a small number of rationals and exact
ties are common rather than rare. Without a deterministic tie-break the fused
ranking depends on dictionary insertion order and the experiment is not
reproducible.

??? question "Why fuse the top 100 from each retriever rather than the top 10, when you only report recall@10?"
    Because fusion re-orders and cannot retrieve. A gold document sitting at
    rank 40 in one run is reachable by fusion only if that run was read to
    depth 40. Fusing the top 10 would cap the union ceiling near the
    individual retrievers' own recall@10 and guarantee a null result, which
    would then get blamed on the fusion rule rather than on the depth.

## F · Production practice

**Do not ship `k = 60` with equal weights because it is the default.** On this
corpus that configuration is measurably worse than deleting the dense retriever
entirely. The default is right when your retrievers are of comparable quality;
check whether yours are before accepting it.

**Fuse deep lists.** Fusion cannot recover a document neither retriever
retrieved, so the candidate depth bounds everything. This experiment uses 100
per retriever; using 10 would have capped the ceiling far below the
0.858 <!-- computed: hybrid_fusion.union_ceiling --> that makes fusion worth
attempting.

**Tune on queries you then throw away.** Split the labelled set, choose on one
part, report on the other. If the set is too small to split, it is too small to
tune on, and the right move is to ship the better single retriever and go and
label more queries.

**Keep the individual runs observable.** When fused quality drops, the question
is always which component moved, and that is unanswerable if only the fused
output is logged.

??? question "If score normalisation uses more information than RRF, why is it not obviously better?"
    Because the extra information is not trustworthy. Min-max normalisation is
    set by the single highest and lowest scores in the list, so one outlier
    compresses everything else; the result depends on how deep a list you
    normalised; and BM25 scores are not comparable across queries, so the
    mapping has to be refitted every time. RRF gives up real information in
    exchange for depending on nothing fragile.

## G · Experiment

`python experiments/hybrid_fusion.py`, over the same 176 answerable queries,
fusing the top 100 from each retriever.

| configuration | recall@10 | against lexical alone |
|---|---:|---|
| dense alone | 0.614 | — |
| **lexical alone** | **0.705** | — |
| RRF, `k=60`, equal weights *(the default)* | 0.642 | **6.3 points worse** |
| RRF, `k=10`, equal weights | 0.688 | worse |
| best of 25 configurations, in-sample | 0.778 | won 13, lost 0, p = 0.0002 |
| **the same tuning, held out** | **0.750** | won 14, lost 6, **p = 0.1153** |
| either one finds it *(ceiling)* | 0.858 | — |

**The default is not a starting point, it is a regression.** Six points below
the component it was added to, from a parameter choice that looks like a
formality.

**The tuned result is real but smaller than it appears.** Both folds
independently selected `k = 1`, so that part of the tuning is stable; the weight
moved between 0.7 and
0.6 <!-- computed: hybrid_fusion.fold2_w_lex -->, and the two held-out scores
were 0.784 <!-- computed: hybrid_fusion.fold1_heldout --> and
0.716 <!-- computed: hybrid_fusion.fold2_heldout --> — a spread of nearly seven
points between two halves of the same benchmark, which is itself a useful
measure of how much this query set can resolve.

**And the honest verdict is "not demonstrated".** Held out, fusion wins 14
queries and loses 6. That is the direction you would hope for and it is not a
significant split on 20 discordant pairs. Fusion is probably helping a little;
this benchmark cannot show it. Saying so is more useful than reporting 0.778,
which is the number a less careful version of this lesson would have printed.

**What it would take to settle this.** McNemar's power depends only on the
discordant pairs, and there were
20 <!-- computed: hybrid_fusion.discordant_pairs --> of them —
11.4% <!-- computed: hybrid_fusion.discordant_rate_pct -->% of the query set.
Holding that rate and that 14-to-6 split, reaching 80% power at the 0.05 level
needs roughly
430 <!-- computed: hybrid_fusion.queries_for_80pct_power --> labelled queries,
against the 176 available. That is the actionable form of a null result:
not "fusion does not work", but "this benchmark is about a factor of two and a
half too small to tell, and here is the number to aim at".

**Fusion captured
29% <!-- computed: hybrid_fusion.headroom_captured_pct -->% of the available
headroom** even at its tuned best. The union ceiling assumes an oracle that
knows which retriever to trust per query, and no fixed weighting can be that
oracle — which is precisely the job of the reranker in lesson 3.6.

??? question "The held-out result is +4.5 points at p = 0.1153. Do you ship the fusion?"
    Reasonably, yes — with the caveat stated. The direction is right, the cost
    is a few milliseconds, and the risk of a change that is probably slightly
    positive is small. What you must not do is *report* it as an improvement,
    or use it as evidence when deciding the next thing. "Shipped, not
    demonstrated" is a coherent position; "shipped, and it gave us 7.3 points"
    is not.

## H · Failure modes and cost traps

**Shipping the paper's constant.** `k = 60` comes from a 2009 evaluation on
TREC runs of broadly similar quality. It encodes an assumption about your
retrievers that you can check in an afternoon.

**Reporting the tuned configuration's score on the tuning queries.** Measured
here: 2.8 points of optimism, and the difference between p = 0.0002 and
p = 0.1153. This is the single most common way retrieval improvements are
overstated, and it requires no bad faith at all — only a query set that felt
too small to split.

**Concluding from complementarity that fusion will help.** Lesson 3.3's
measurement that 22 queries are answered only by lexical retrieval is a
necessary condition for fusion to work and not a sufficient one. The headroom
was real; most of it is still there.

**Fusing shallow candidate lists.** Fusion re-orders; it cannot retrieve. Ten
candidates per retriever puts a low ceiling on the whole exercise, and the
resulting null result gets attributed to fusion rather than to the depth.

**Weighting by intuition rather than by measurement.** The tuned weight here
happened to land near the retrievers' recall ratio, which is reassuring, and it
is not a rule — the fold that chose 0.6 did about as well on its own tuning
half as the fold that chose 0.7.

**A mistake made writing this lesson.** The first version of this experiment
reported 0.778 and the p = 0.0002 that goes with it, and I nearly wrote the
lesson around a decisive win for hybrid retrieval. The held-out split was added
because the same session had already been burned by a significance test run on
a broken benchmark, and it turned the headline result into a null one. The
in-sample number was not wrong — it was an answer to a question nobody should
ask.

## I · Graded practice

<quiz-bank src="ret-l4"></quiz-bank>

<code-exercise src="ret-l4-rrf"></code-exercise>

<code-exercise src="ret-l4-optimism"></code-exercise>

## J · Annotated references

- **Cormack, Clarke and Buettcher (2009), "Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods"** — the source of the method
  and of `k = 60`. Worth reading for what their runs looked like, which is the
  context the constant assumes.
- **Bruch, Gai and Ingber (2023), "An Analysis of Fusion Functions for Hybrid
  Retrieval"** — a careful comparison of rank-based and score-based fusion,
  including when normalisation is defensible.
- **Wang et al. (2021), "BERT-based Dense Retrievers Require Interpolation with
  BM25"** — the result that dense retrieval generally needs a lexical partner,
  and the interpolation weight that implies.
- **Cawley and Talbot (2010), "On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation"** — not a retrieval
  paper, and the most relevant reference in this list to §G's second half.

## K · Extension

*Off-platform, half a day.* Repeat §G on your own corpus with your own labelled
queries, and report three numbers: the default configuration, the tuned
configuration in-sample, and the tuned configuration held out. The gap between
the second and third is your benchmark's optimism, and it is worth knowing
before you use that benchmark to make any other decision. If the gap is large,
the finding is about your query set rather than about fusion.
