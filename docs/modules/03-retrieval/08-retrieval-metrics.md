---
status: Verified
last_verified: 2026-08-13
volatility: low
pyodide: true
prereqs: ["3.6"]
---

# 3.8 · Retrieval metrics, and which one decides your answer

## A · Why this matters

Every system built in this module has been scored, until now, on recall@10.
This lesson scores all five of them on the five metrics people actually report,
and the table disagrees with itself.

| system | recall | MRR | nDCG |
|---|---:|---:|---:|
| lexical | **0.705** <!-- computed: retrieval_metrics.lexical_recall --> | 0.287 <!-- computed: retrieval_metrics.lexical_mrr --> | 0.377 <!-- computed: retrieval_metrics.lexical_ndcg --> |
| dense | 0.614 <!-- computed: retrieval_metrics.dense_recall --> | **0.356** <!-- computed: retrieval_metrics.dense_mrr --> | **0.407** <!-- computed: retrieval_metrics.dense_ndcg --> |

Lexical retrieval wins on recall and loses on every rank-aware metric, so a
team comparing BM25 against embeddings reaches opposite conclusions depending
on a choice that is usually made months earlier by whoever wrote the evaluation
harness first. Across all five systems,
2 <!-- computed: retrieval_metrics.distinct_winners --> different ones take the
top spot under the five metrics, and the mean rank correlation between metrics
is 0.600 <!-- computed: retrieval_metrics.mean_rank_correlation --> — they
agree about the ordering of systems rather less than half the time.

The second result is about a number that looks like a disaster and is not.
Lexical precision@10 is
0.076 <!-- computed: retrieval_metrics.lexical_precision -->, which reads as a
system getting ninety-two per cent of its results wrong. In fact
145 <!-- computed: retrieval_metrics.queries_single_gold --> of the
176 <!-- computed: retrieval_metrics.n_queries --> queries have exactly one
correct document, so nine of ten slots are *necessarily* wrong and precision@10
cannot exceed 0.118 <!-- computed: retrieval_metrics.precision_ceiling --> on
this benchmark however good the system is. Lexical is at
64% <!-- computed: retrieval_metrics.lexical_precision_pct_of_ceiling -->% of
the attainable maximum. A precision@k quoted without its ceiling is not a
measurement of anything.

!!! info "Terms used in this lesson"
    **Set metric** — one that asks only *whether* correct documents are in the
    top `k`, ignoring their positions. Recall@k and precision@k.

    **Rank metric** — one that credits position, so moving a correct answer up
    improves it. MRR, MAP, nDCG.

    **Recall@k** — the fraction of queries with at least one correct document
    in the top `k`. Used throughout this module.

    **Precision@k** — the fraction of the `k` returned documents that are
    correct. Bounded above by `(number of correct documents) / k`.

    **MRR** — mean reciprocal rank: average of `1/position` of the first
    correct result. Cares only about the first one.

    **MAP** — mean average precision: averages precision at each correct
    result, so it cares about *all* of them.

    **nDCG** — discounted cumulative gain, normalised by the best achievable
    ordering. The only one here that extends naturally to graded relevance.

## B · Mental model

**A metric is a claim about what the user does with the results, and choosing
one is a product decision disguised as a measurement decision.**

Three questions sort the family, and answering them for your product picks the
metric rather than leaving it to convention:

**Does the user see one result or ten?** If your interface shows a single
answer, or a model consumes only the top passage, then a correct document at
rank 8 is worth nothing and MRR is the honest metric. If a human scans a list,
position matters less and recall@10 is closer to what they experience.

**Is there one right answer or many?** With one correct document per query,
precision@10 is capped at 0.1 and MAP collapses to MRR, so reporting either
without saying so invites misreading. With many, MAP and nDCG start to earn
their complexity.

**Are all correct answers equally correct?** Everything in this module treats
relevance as binary, which the corpus supports because gold labels were
recorded as sets. Graded relevance — this document is perfect, that one is
adequate — needs nDCG and needs labels this benchmark does not have.

The reason the choice bites is that the metrics are not noisy versions of one
another; they measure genuinely different things, and a system can be built to
win one at the expense of another. Lesson 3.6 showed that directly: reranking
moved recall@10 by
2.2 points and moved MRR from 0.287 to
0.459 <!-- computed: retrieval_metrics.lexical_rerank_mrr -->, because
reordering is exactly what rank metrics reward and set metrics ignore.

??? question "If recall@10 and MRR disagree, why not just report both and let the reader decide?"
    You should report both, and it does not dissolve the problem, because
    somebody eventually has to ship one system rather than another. The
    decision needs a metric chosen in advance from what the product does with
    the results; reporting a family is how you stay honest about the tradeoff,
    not how you avoid making it. Choosing after seeing the numbers is how a
    measurement stops constraining anything.

## C · Mechanism

For a query with gold set `G` and results `r₁ … r_k`, write `hᵢ = 1` when `rᵢ`
is in `G`.

**Recall@k** is `1` if any `hᵢ = 1`, averaged over queries. It is binary per
query, which is why every comparison in this module could use an exact McNemar
test: each query contributes a single bit, and the paired test only needs the
queries where two systems disagree.

**Precision@k** is `(Σ hᵢ) / k`. The ceiling is `min(|G|, k) / k`, so with one
correct document and `k = 10` the best possible score is 0.1. This is the
metric most often reported without its ceiling, and the resulting number
invariably looks alarming.

**MRR** is `1 / (position of the first hit)`, or zero if there is none. It
ignores everything after the first correct result, which makes it the right
metric when the consumer reads one document and the wrong one when a user scans
a list and wants all the relevant items.

**MAP** averages precision at each hit position:

$$
\text{AP} = \frac{1}{\min(|G|, k)} \sum_{i:\, h_i = 1} \frac{\sum_{j \le i} h_j}{i}
$$

so it rewards getting *all* the correct documents high, not just the first.
With a single gold document it reduces exactly to reciprocal rank, which is why
MAP and MRR track each other so closely in §G — 145 of these 176 queries have
one gold document, so for most of the benchmark they are the same number
wearing different names.

**nDCG** discounts by the logarithm of position and normalises by the ideal
ordering:

$$
\text{nDCG} = \frac{\sum_i h_i / \log_2(i+1)}{\sum_{i=1}^{\min(|G|,k)} 1 / \log_2(i+1)}
$$

The normaliser is what makes it comparable across queries with different
numbers of correct documents, and it is also why nDCG is the only metric here
that extends to graded relevance without modification: replace `hᵢ` with a
relevance grade and the formula is unchanged.

**Continuous metrics need a different test.** Recall is binary so McNemar
applies, but MRR and nDCG are continuous per query, so the paired comparison
needs a bootstrap over the per-query differences instead. §G uses both, and the
distinction matters because applying McNemar to a continuous metric requires
thresholding it first, which throws away most of the information the metric
exists to capture.

**Worth working one query by hand, because the identity behind §G's MAP
column is easier to see than to take on trust.** Suppose a query has three
correct documents and the system returns exactly one of them, at position 2.
Precision at that hit is one half, and average precision divides the sum of
precisions-at-hits by `min(3, 10) = 3`, giving 0.167 — whereas reciprocal rank
is simply 0.5, because it stops at the first hit and never asks what else
existed. Now suppose the same query had only one correct document, found in the
same place: average precision becomes `0.5 / 1`, which is exactly the
reciprocal rank, and the two metrics have collapsed into one number.

Since 145 of these 176 queries are single-gold, that collapse governs most of
the benchmark, which is why the MAP and MRR columns move together everywhere in
§G rather than occasionally. The divisor is also where the most common
implementation bug lives, because dividing by the number of correct documents
*retrieved* rather than the number that *existed* scores the three-document
query at 0.5 instead of 0.167 — full marks for finding a third of the answer,
and a bug that cannot fire at all on a single-answer benchmark.

## D · From data science to LLM systems

You have made this choice before, under the name of picking between accuracy,
precision, recall, F1 and AUC — and the reasoning is identical, because each
encodes a different assumption about what an error costs. If you have ever
watched a model with excellent accuracy turn out to be useless on the minority
class, you already know what happens when the metric and the product disagree.

Two differences matter here.

**Rank replaces threshold.** A classifier's metrics are parameterised by an
operating threshold, and the ROC curve exists to show the whole family at once.
Retrieval has no threshold — lesson 3.1 established that absolute similarity
scores are not comparable across queries — so the free parameter is the cutoff
`k`, and there is no equivalent of AUC that summarises across all cutoffs in a
way anyone reports.

**The labels are sets rather than classes, and their size drives the ceiling.**
A classifier's metrics do not change when you relabel how many positives exist
per example, because there is one label per example. Here the number of correct
documents per query directly bounds precision, so two benchmarks with identical
systems and different labelling conventions produce different numbers. That is
why §A insists on the ceiling: it is a property of the *benchmark*, not of the
system, and it is invisible in the metric's name.

The habit that transfers unchanged is the important one. You would never report
a classifier's accuracy without the class balance, and precision@k without the
gold-set size is exactly the same omission.

??? question "If nDCG subsumes the others, why report anything else?"
    Because it answers a question your product may not be asking, and because
    its normaliser makes it harder to reason about than it looks. A score of
    0.5 tells you the ordering achieved half the discounted gain of a perfect
    one, which is not a quantity anyone has intuition for, whereas recall@3
    means something a product manager can act on. The practical compromise is
    to gate on the metric that matches the interface and report nDCG alongside
    for comparability with published work.

## E · Minimal implementation

The five metrics, computed per query so differences can be tested rather than
eyeballed:

```python
def score(hits, n_gold, k):
    """hits[i] is 1 when the i-th result is correct. Returns one query's scores."""
    recall = 1.0 if any(hits) else 0.0
    precision = sum(hits) / k

    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1 / i
            break

    found, acc = 0, 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            found += 1
            acc += found / i
    # The divisor is min(n_gold, k), not the number of hits found: a system
    # that retrieves one of three correct documents must be penalised for the
    # two it missed, and dividing by `found` would score it perfectly.
    ap = acc / min(n_gold, k)

    dcg = sum(h / math.log2(i + 1) for i, h in enumerate(hits, start=1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(n_gold, k) + 1))
    return recall, precision, rr, ap, dcg / idcg
```

Returning per-query values rather than an average is the decision that makes
everything downstream possible. A function returning a single mean cannot
support a paired test, a bootstrap interval, or a breakdown by query type, and
those are the three things you will want within a week of having the harness.
Averaging is a one-line operation the caller can do; recovering the per-query
values from an average is not possible at all.

??? question "Is precision@k ever the right metric for a retrieval system?"
    It is, once queries genuinely have many correct answers and the user reads
    several of them, since it then measures the density of useful material in
    what was returned rather than being pinned by the labelling convention. On
    a benchmark where nearly every query has one answer it carries no
    information that recall@k does not, and it carries a bound that invites
    misreading, so the honest choice there is to omit it.

## F · Production practice

**Choose the metric from the product, and write down the reasoning.** One
answer shown to a user means MRR; a list a human scans means recall@k at the
list length; a model consuming the top three means recall@3. The reasoning is
what makes the choice reviewable later.

**Report the ceiling with precision@k, always.** Or do not report precision@k
on a single-answer benchmark at all, since it carries no information that
recall@k does not.

**Report an interval, not a point.** §G's MRR difference is
-0.069 <!-- computed: retrieval_metrics.mrr_diff --> with a 95% interval of
[-0.120 <!-- computed: retrieval_metrics.mrr_ci_low -->,
-0.018 <!-- computed: retrieval_metrics.mrr_ci_high -->], and that interval is
what tells you the difference is real. A point estimate on 176 queries invites
a precision the benchmark cannot support.

**Use the right paired test for the metric's type.** McNemar for binary
recall; bootstrap over per-query differences for MRR, MAP and nDCG.

**Keep the per-query scores.** They cost nothing to store and they are what
lets you break a regression down by query type, which is how the near-duplicate
result in lesson 3.3 was found.

??? question "The recall difference sits at p = 0.0722. Should that be reported as 'no difference'?"
    No, because failing to reach a threshold is not evidence of absence, and
    43 against 27 is a real imbalance that a larger query set might well
    resolve. The accurate phrasing is that the difference is not demonstrated
    at this sample size, ideally with the number of queries that would
    demonstrate it — which lesson 3.4 worked out for a comparable case and put
    at roughly 430.

## G · Experiment

`python experiments/retrieval_metrics.py`, over the five systems built in this
module.

| system | recall | precision | MRR | MAP | nDCG |
|---|---:|---:|---:|---:|---:|
| lexical | 0.705 | 0.076 | 0.287 | 0.281 | 0.377 |
| dense | 0.614 | 0.062 | 0.356 | 0.341 | 0.407 |
| hybrid | 0.767 | 0.080 | 0.314 | 0.299 | 0.403 |
| lexical + rerank | 0.727 | 0.078 | **0.459** | **0.426** | 0.497 |
| hybrid + rerank | **0.801** | **0.081** | 0.454 | 0.412 | **0.500** |

**The metrics disagree about the best system, and about the best of the two
base retrievers.** Two systems share the five first places, and the mean rank
correlation across metric pairs is
0.600 <!-- computed: retrieval_metrics.mean_rank_correlation --> with a minimum
of 0.300 <!-- computed: retrieval_metrics.min_rank_correlation -->.

**But the two halves of the lexical-versus-dense flip are not equally
supported, and that is the more useful finding.** On recall, lexical wins
43 <!-- computed: retrieval_metrics.recall_lexical_won --> queries and loses
27 <!-- computed: retrieval_metrics.recall_lexical_lost -->, at
p = 0.0722 <!-- computed: retrieval_metrics.p_recall_flip --> — suggestive, and
not significant at the conventional threshold. On MRR, dense's advantage has a
bootstrap interval excluding zero. So the honest summary is not the comfortable
"it depends on your metric" but something stronger: **dense retrieval is
demonstrably better at ranking, and lexical retrieval is not demonstrably
better at finding.** Only computing both, with intervals, distinguishes those.

**MAP and MRR barely differ anywhere in the table**, because with one gold
document per query average precision reduces exactly to reciprocal rank, and
145 of 176 queries are single-gold. Reporting both suggests two independent
pieces of evidence where the benchmark provides one.

??? question "Two systems tie on every metric you report. Is there anything left to measure?"
    Usually yes, because aggregate agreement often hides per-query-type
    disagreement of the kind lesson 3.3 found when stopword removal moved
    multi-hop queries from 0.00 to 1.00 while pushing near-duplicates down by
    fourteen points. Breaking each metric down by query phenomenon is cheap
    once per-query scores are kept, and it frequently shows that two systems
    with identical averages are failing on entirely different inputs.

## H · Failure modes and cost traps

**Reporting precision@k on a single-answer benchmark.** The ceiling here is
0.118, so a "good" system scores under 0.1 and looks broken. Either report the
ceiling beside it or omit the metric.

**Choosing the metric after seeing the results.** Every system in the table
wins under some metric, so a metric selected afterwards can justify any of
them. Choose from the product, in advance, and write down why.

**Evaluating a reranker on a set metric.** Lesson 3.6 measured this: the same
change is overwhelmingly significant at rank one and invisible at rank ten.
Reordering is what rank metrics see and set metrics discard.

**Reporting MAP and MRR as independent evidence** when the benchmark is mostly
single-gold, since they are then the same quantity.

**Averaging away the per-query scores.** They are needed for paired tests,
intervals and breakdowns, and they cannot be recovered afterwards.

**Applying McNemar to a continuous metric.** It requires thresholding first,
which discards the positional information the metric exists to measure. Use a
bootstrap over per-query differences instead.

**A conclusion this module nearly reached.** Every earlier lesson reported
recall@10, and on that metric lexical retrieval is the better base retriever by
nine points. Adding rank-aware metrics reverses the comparison, and adding
intervals shows the reversal is the better-supported half. Had the harness been
written to compute one metric — which is the usual case, because the first
metric someone implements becomes the standard — this module would have shipped
a defensible and wrong recommendation.

## I · Graded practice

<quiz-bank src="ret-l8"></quiz-bank>

<code-exercise src="ret-l8-metrics"></code-exercise>

<code-exercise src="ret-l8-ceiling"></code-exercise>

## J · Annotated references

- **Manning, Raghavan and Schütze (2008), *Introduction to Information
  Retrieval*, ch. 8** — still the clearest treatment of the family and of what
  each metric assumes about the user.
- **Järvelin and Kekäläinen (2002), "Cumulated Gain-based Evaluation of IR
  Techniques"** — where nDCG comes from, and the argument for the logarithmic
  discount that is usually quoted without its justification.
- **Fuhr (2018), "Some Common Mistakes In IR Evaluation, And How They Can Be
  Avoided"** — short, sharp, and directly about the errors in §H.
- **Sakai (2006), "Evaluating Evaluation Metrics based on the Bootstrap"** —
  the method §G uses for the continuous metrics, and why it is preferable to a
  t-test on scores that are not normally distributed.

## K · Extension

*Off-platform, half a day.* Take your own retrieval evaluation and add two
things: the ceiling for every bounded metric you report, and a bootstrap
interval for every difference you act on. Then re-read your last three
conclusions and check whether any of them survives. The interesting outcome is
not that a conclusion fails, but which one — the differences that vanish under
an interval are usually the ones that were cheapest to obtain.
