---
status: Verified
last_verified: 2026-08-12
volatility: low
pyodide: true
prereqs: ["3.1"]
---

# 3.2 · Exact and approximate search

## A · Why this matters

The default architecture diagram for a retrieval system has a vector database
in it, and for most systems that box is premature. Exact search over the corpus
this module uses is one matrix multiply:
928,896 <!-- computed: ann_tradeoff.exact_multiply_adds --> multiply-adds and
3.72 <!-- computed: ann_tradeoff.exact_mb_float32 --> MB of memory traffic per
query, which on an ordinary laptop takes roughly a tenth of a millisecond. No
index, no build step, no extra service, and by construction it returns the
correct answer every time.

The cost of adopting an approximate index before you need one is not just the
operational weight of another dependency. It is that approximate search is
**lossy in a way nobody measures correctly**. The standard quality number for
these systems is agreement with exact search, and this lesson's experiment
shows agreement of
0.92 <!-- computed: ann_tradeoff.raw_agree_p16 --> — which sounds nearly
lossless — arriving with end-task recall down from
0.489 <!-- computed: ann_tradeoff.raw_exact_recall --> to
0.398 <!-- computed: ann_tradeoff.raw_recall_p16 -->. That is a fifth of the
system's answers gone, reported as 92% quality retained.

There is also a result here that reorders the whole decision. An approximate
index built on the transformed vectors from lesson 3.1 reaches recall
0.511 <!-- computed: ann_tradeoff.tf_recall_p4 --> while scanning
7.4% <!-- computed: ann_tradeoff.tf_scanned_p4 --> of the corpus — beating
**exact** search on the untransformed vectors, which manages 0.489 while
scanning all of it. Fixing the geometry was worth more than being exact.

!!! info "Terms used in this lesson"
    **Exact search** — score every document and take the best `k`. Also called
    brute force or flat search. It has no error, and its cost is linear in
    corpus size.

    **Approximate nearest neighbour (ANN)** — any structure that returns
    *probably* the best `k` while touching a fraction of the corpus, trading
    correctness for speed.

    **IVF (inverted file index)** — cluster the corpus once, assign each
    document to its nearest centroid, and at query time score only the
    documents belonging to the `nprobe` nearest centroids.

    **`nprobe`** — how many clusters to open. The single knob that moves an
    IVF index along its speed-versus-quality curve.

    **HNSW** — a navigable graph of vectors, searched greedily from an entry
    point. Different mechanism, same shape of tradeoff, exposed through a
    parameter usually called `ef_search`.

    **Agreement@k** — how much of exact search's top `k` the index also
    returned. This is what the ANN literature means by "recall", and it is not
    the same thing as this curriculum's recall@k, which is measured against
    labelled answers.

## B · Mental model

**An approximate index is a bet that the answer is near the query in a space
you have already partitioned. Its error is not random — it is concentrated
exactly where partitioning is hardest.**

The comfortable way to think about ANN error is as a small uniform tax: you
lose a percent or two of quality everywhere. That picture is wrong in a way
that matters, because the documents an IVF index misses are the ones sitting
near a cluster boundary, and a document near a boundary is one whose
relationship to the query was already marginal. Lesson 3.1 established that the
margin between a right answer and an arbitrary document is
0.072 <!-- computed: embedding_geometry.query_gold_margin -->, so a great many
correct answers sit exactly in that fragile region. Approximation eats the
hardest queries first, not a random selection of them.

That leads to the practical framing for the whole lesson. There are two
different quality questions, and conflating them is the most common mistake in
this area:

| Question | Measured against | What it tells you |
|---|---|---|
| Agreement@k | Exact search's own results | How faithfully the index reproduces brute force |
| Recall@k | Labelled correct answers | Whether the user gets an answer |

The second is the only one a user experiences. The first is the only one most
benchmarks report, because it needs no labelled data — which is precisely why
it is so widely available and so misleading.

??? question "Why would agreement@10 of 0.92 correspond to a much larger drop in end-task recall?"
    Because the two documents the index dropped are not a random two. Exact
    search's tenth-ranked result is usually irrelevant anyway, so losing it
    costs nothing; but the gold document is frequently *also* down in that
    fragile region, and it is drawn from the same population of
    near-boundary vectors that the index is worst at. The missing 8% and the
    answers are correlated.

## C · Mechanism

IVF has three phases and only the third runs per query.

**Build.** Run k-means over the corpus to obtain `nlist` centroids. The
conventional starting point is `nlist ≈ √n`, which for
2,419 <!-- computed: ann_tradeoff.n_documents --> documents gives about 49; this
experiment uses 64 <!-- computed: ann_tradeoff.nlist -->. This is the expensive
step, it is done once, and it is why adding documents to an IVF index degrades
it gradually — the centroids describe the corpus as it was when they were
fitted.

**Assign.** Every document is placed in the list of its nearest centroid. The
lists are not equal in size, and their imbalance is a property of the data
rather than something you control: on the raw vectors here the largest holds
120 <!-- computed: ann_tradeoff.raw_largest_list --> documents against a
standard deviation of
23.0 <!-- computed: ann_tradeoff.raw_list_size_sd -->, and one list ends up
holding nothing at all.

**Search.** Score the query against the `nlist` centroids, take the `nprobe`
nearest, and score the query against only the documents in those lists. Total
work is `nlist + (documents in the probed lists)` instead of `n`, and the
correctness loss is entirely accounted for by one thing: a document whose
centroid was not probed cannot be returned, no matter how well it matches.

**Where `√n` comes from.** The rule is quoted everywhere and derived almost
nowhere, and the derivation is two lines that also tell you when to ignore it.
Work per query is the centroid scan plus the documents in the opened lists:

$$
\text{work}(L) = L + \text{nprobe} \cdot \frac{n}{L}
$$

for `L` lists over `n` documents. Differentiate with respect to `L`, set to
zero, and the minimum sits at $L = \sqrt{\text{nprobe} \cdot n}$. At
`nprobe = 1` that is the familiar $\sqrt{n}$, which for
2,419 <!-- computed: ann_tradeoff.n_documents --> documents gives 49. The two
terms are in tension in a way worth seeing concretely: at 16 lists the scan
costs 16 centroids plus about 151 documents, at 49 it is 49 plus 49, and at 512
it is 512 centroids plus about 5 documents. The last configuration does
ninety-nine percent of its work deciding where to look.

The rule breaks the moment `nprobe` is large, which is the regime any system
tuned for quality ends up in. At `nprobe = 32`, the optimum is
$\sqrt{32 \times 2419} \approx 278$ lists rather than 49 — so a system tuned
for recall and a system tuned by the folk rule are configured differently, and
the folk rule was derived for the case nobody actually operates in.

The connection to lesson 3.1 is not decorative. k-means partitions by Euclidean
distance, and lesson 3.1 showed these vectors carry a large shared offset that
places the whole cloud far from the origin. Clustering a cloud dominated by a
common direction partitions it mostly along that direction, which is the one
axis carrying no information about topic. Removing it before clustering is not
an optimisation of the index; it changes what the index is organised by.

HNSW replaces this partition with a graph. Each vector is a node linked to some
of its neighbours across several layers, and search descends greedily from a
sparse top layer to a dense bottom one, keeping a candidate list of size
`ef_search`. The failure mode has the same shape as IVF's — a greedy walk can
settle in a local basin and never reach the true neighbour — and the same knob
exists to trade work against that risk. Everything this lesson measures about
`nprobe` applies to `ef_search` with the numbers changed.

??? question "If IVF only ever returns documents brute force also scored, how can it beat exact search?"
    On the same vectors it cannot — it is bounded above by exact search, since
    its results are a subset of what brute force ranked. The comparison in §A
    crosses two *different* vector sets: approximate search over transformed
    vectors against exact search over raw ones. Within either set, exactness
    still wins.

## D · From data science to LLM systems

You know this tradeoff as approximate versus exact nearest neighbours, and you
have probably used a k-d tree or a ball tree without thinking of it as
approximation, because in low dimensions those structures are exact. That is
the first thing that breaks. Space-partitioning trees degrade to brute force
somewhere around ten to twenty dimensions, so at
384 <!-- computed: embedding_geometry.dim --> the entire family of exact
spatial indexes is unavailable, and every practical structure is approximate by
necessity rather than by choice.

The second break is what "tuning" means. In modelling you tune against a
validation metric that reflects the task. Here the parameter is tuned against
agreement with exact search, because that is what the index library reports and
it needs no labels — so the default workflow optimises a proxy, and the
experiment below shows how loose the coupling between that proxy and the task
is. If you have a labelled query set, `nprobe` should be chosen against it.
Most teams have no labelled query set, which is the actual reason this goes
wrong.

The third is that the index is stateful in a way models are not. A fitted
transformer applied to new data is unchanged by it. An IVF index absorbs new
documents into centroids fitted on the old distribution, so quality decays as
the corpus drifts, silently and without any error being raised — the failure
looks like the retrieval slowly getting worse, which is easy to attribute to
almost anything else.

## E · Minimal implementation

IVF search is short enough to write out, and the version below is the one the
experiment runs.

```python
import numpy as np

def build(vectors, centroids):
    """Assign every vector to its nearest centroid."""
    # Squared euclidean expanded into a single matmul. The |v|^2 term is the
    # same for every centroid and cannot change the argmin, but it is kept so
    # the values are true distances if you ever print them.
    d = ((vectors ** 2).sum(1)[:, None]
         - 2 * vectors @ centroids.T
         + (centroids ** 2).sum(1))
    assign = d.argmin(axis=1)
    return [np.flatnonzero(assign == c) for c in range(len(centroids))]

def search(vectors, centroids, lists, q, nprobe, k=10):
    """Top k from the nprobe nearest lists."""
    order = np.argsort(-(centroids @ q))[:nprobe]
    candidates = np.concatenate([lists[c] for c in order])
    scores = vectors[candidates] @ q
    top = np.argpartition(-scores, k)[:k]
    return candidates[top[np.argsort(-scores[top])]]
```

The detail worth noticing is that `search` ranks *centroids* by dot product and
then ranks *documents* by dot product, and those are two different questions.
A centroid is an average of its members, so it is closer to the origin than any
of them and its score is systematically smaller. That does not matter for
choosing which lists to open, since only the ordering is used — but it is the
reason you cannot compare a centroid score against a document score, or use one
threshold for both.

??? question "Why can a centroid's score not be compared against a document's score?"
    A centroid is the mean of its members, and averaging vectors that point in
    slightly different directions produces a shorter vector. Its dot product
    with a query is therefore systematically smaller than its members' would
    be. Only the *ordering* of centroid scores is used, which is unaffected —
    but a threshold applied to both would silently reject every list.

## F · Production practice

**Do not build an index you do not need.** At a million documents, exact search
over float32 vectors of this width scans
1.54 <!-- computed: ann_tradeoff.mb_float32_at_1m_docs --> GB per query, which
is where brute force genuinely stops being viable. Below roughly a hundred
thousand documents it is a numpy call taking single-digit milliseconds, and it
is exact, and it has no build step to go stale. Quantise before you index: the
same corpus in int8 is
0.93 <!-- computed: ann_tradeoff.exact_mb_int8 --> MB, and lesson 3.1 measured
the fidelity cost of that at 0.99984.

**Tune `nprobe` against labelled queries, not against agreement.** If you have
no labels, build a small set — a hundred queries with known answers is enough
to see the difference between the two curves in §G, and it is a day of work
that pays for itself the first time someone proposes lowering `nprobe` to save
money.

**Apply the geometry transform before clustering, not after.** The centroids
are fitted to whatever vectors you hand them. Fitting on raw vectors and then
transforming at query time gives you a partition organised by the shared
direction and a query embedded in a different space, which is worse than either
choice made consistently.

**Rebuild on corpus drift, and monitor it.** Track the fraction of queries
whose nearest centroid distance exceeds what it was at build time. That number
rising is the earliest available signal that the partition no longer describes
the corpus, and it is available without labels.

??? question "Your corpus has 8,000 documents and someone proposes a vector database. What do you measure first?"
    The exact baseline: 8,000 by your vector width is a single matmul, and on
    ordinary hardware it lands in the low milliseconds. If that is inside
    budget, an index adds a build step that can go stale, a partition that
    ages with the corpus, and a quality loss nobody on the team is currently
    measuring — in exchange for reducing a cost that was not the problem.

## G · Experiment

`python experiments/ann_tradeoff.py` builds an IVF index over the recorded
fixture twice — once on the raw vectors and once after lesson 3.1's transform —
and measures both quality questions at every probe count.

| probes | scanned | agreement@10 | recall@10 (raw) | | scanned | agreement@10 | recall@10 (transformed) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1.5% | 0.401 | 0.176 | | 2.6% | 0.631 | 0.381 |
| 4 | 6.3% | 0.712 | 0.312 | | 7.4% | 0.834 | **0.511** |
| 16 | 26.4% | 0.920 | 0.398 | | 27.3% | 0.958 | 0.585 |
| 32 | 51.2% | 0.973 | 0.460 | | 50.9% | 0.991 | 0.614 |
| 64 (exact) | 100% | 1.000 | 0.489 | | 100% | 1.000 | 0.614 |

**Agreement is a badly calibrated proxy for quality.** At 16 probes on raw
vectors the index reproduces 92% of exact search's results and delivers 81% of
its end-task recall. At 4 probes it reproduces 71% and delivers 64%. The proxy
is optimistic at every point on the curve, and the gap is largest exactly where
people operate — the region where agreement still looks respectable.

**The transform beats exactness.** Four probes on transformed vectors, scanning
7.4% <!-- computed: ann_tradeoff.tf_scanned_p4 --> of the corpus, returns
0.511 <!-- computed: ann_tradeoff.tf_recall_p4 --> against exact search on raw
vectors returning
0.489 <!-- computed: ann_tradeoff.raw_exact_recall --> after scanning
everything. Two lines of post-processing were worth more than thirteen times
the compute.

**Better geometry also makes a better partition.** At one probe, the
transformed index already agrees with exact search
0.631 <!-- computed: ann_tradeoff.tf_agree_p1 --> of the time against
0.401 <!-- computed: ann_tradeoff.raw_agree_p1 --> for raw, and it leaves
0 <!-- computed: ann_tradeoff.tf_empty_lists --> empty lists against
1 <!-- computed: ann_tradeoff.raw_empty_lists -->. Clustering a cloud whose
dominant axis carries no topical information wastes most of the partition on
that axis.

**And the honest conclusion for a corpus this size is not to do any of it.**
Transformed IVF needs
32 <!-- computed: ann_tradeoff.tf_probes_for_parity --> probes and
50.9% <!-- computed: ann_tradeoff.tf_scanned_at_parity -->% of the corpus to
match its own exact baseline; raw IVF never matches it before scanning
everything. Halving the work for zero loss is a real saving in principle and
worth nothing here, because the thing being halved is a tenth of a millisecond.

??? question "Why does the transformed index need 32 probes to match its own exact baseline, when 16 probes already reaches 0.585 of 0.614?"
    Because the last few queries are the ones whose answers sit furthest from
    any centroid, and they are only reachable once most of the partition is
    open. The curve's shape is the general lesson: most of the recovery is
    cheap and the final fraction is not, so "lossless" configurations tend to
    scan so much of the corpus that they give up the reason the index existed.

## H · Failure modes and cost traps

**Reporting agreement as though it were quality.** The measured gap above is
the whole argument. If a vendor benchmark, a library default, or a colleague
quotes "95% recall" for an ANN index, that is agreement with brute force and
says nothing about your task until you measure it against labels.

**Tuning `nprobe` down to hit a latency target.** It is the most available
knob and the damage is invisible in every metric except the one nobody has.
Latency improves, error rates do not move, and answer quality degrades on
exactly the hard queries that mattered.

**Clustering raw vectors.** Measured above: it wastes the partition on the
shared direction, produces an empty list, and needs a full scan to match its
own exact baseline.

**Assuming a k-d tree will do.** It will, up to about twenty dimensions, and
then it degrades to a full scan with extra bookkeeping. At 384 dimensions there
is no exact spatial index worth having.

**Letting the index age.** Centroids fitted on last quarter's corpus partition
this quarter's badly, and nothing raises an error. The symptom is a slow
decline that gets attributed to the model, the prompt, or the users.

**Building the index before measuring the baseline.** At
2,419 <!-- computed: ann_tradeoff.n_documents --> documents the exact search is
928,896 <!-- computed: ann_tradeoff.exact_multiply_adds --> multiply-adds. It is
worth knowing that number for your own corpus before adding a service to avoid
paying it.

## I · Graded practice

<quiz-bank src="ret-l2"></quiz-bank>

<code-exercise src="ret-l2-ivf"></code-exercise>

<code-exercise src="ret-l2-agreement"></code-exercise>

## J · Annotated references

- **Malkov and Yashunin (2018), "Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs"** — the HNSW
  paper. Read §4 for the parameter that plays `nprobe`'s role.
- **Jégou, Douze and Schmid (2011), "Product Quantization for Nearest Neighbor
  Search"** — where IVF and its quantised variants come from, and still the
  clearest account of why the coarse quantiser exists.
- **Johnson, Douze and Jégou (2019), "Billion-scale similarity search with
  GPUs"** — the FAISS paper, and a useful calibration of the scale at which
  these structures become necessary rather than optional.
- **Bernhardsson, "ANN Benchmarks"** — the standard comparison suite. Read it
  with §G in mind: the y-axis is agreement with brute force, not task quality.

## K · Extension

*Off-platform, an afternoon.* Install a real index library, build both an IVF
and an HNSW index over the shipped fixture, and reproduce §G's two curves
against `nprobe` and `ef_search`. Two things are worth checking: whether the
gap between agreement and end-task recall has the same shape for a graph index
as for a partition index, and whether the geometry transform helps HNSW as much
as it helps IVF. The second has a plausible argument in both directions, which
is what makes it worth measuring rather than reasoning about.
