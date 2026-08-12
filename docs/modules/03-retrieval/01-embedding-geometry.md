---
status: Verified
last_verified: 2026-08-12
volatility: low
pyodide: true
prereqs: ["2.2"]
---

# 3.1 · Embedding geometry: the space is not shaped how you think

## A · Why this matters

Everything in this module rests on a single assumption, which is that distance
between embedding vectors means something about relatedness of text. That
assumption is sound enough to build on, and it is also considerably less
straightforward than the diagrams suggest, because the space those vectors
occupy has a shape that nobody draws.

Here is the measurement that should unsettle you. Take two documents at random
from the corpus this module uses, embed both, and compute the cosine similarity
between them. The average answer is
0.841 <!-- computed: embedding_geometry.random_pair_cosine_mean -->, and across
twenty thousand random pairs the *least* similar two documents anyone managed
to find still scored
0.582 <!-- computed: embedding_geometry.random_pair_cosine_min -->. Nothing in
this corpus is dissimilar to anything else. A shipment record about a pallet in
Glasgow and a clause about lithium batteries are, by this measure, 84% alike.

The practical cost of misreading that geometry is a system that silently
retrieves rubbish. Anyone who has written `if similarity > 0.8: use_it()` has
written a filter that, on this model, admits essentially every document in the
corpus. The threshold looks principled, it passes review, and it does nothing.

The reward for reading the geometry correctly is unusually concrete for a topic
this abstract. Two lines of arithmetic applied after the model has run — no
retraining, no larger model, no prompt changes — move recall@10 on this corpus
from 0.489 <!-- computed: embedding_geometry.recall_raw --> to
0.614 <!-- computed: embedding_geometry.recall_strip1 -->, a gain of
12.5 <!-- computed: embedding_geometry.recall_gain_pts --> points that survives
a paired significance test at
p = 0.0001 <!-- computed: embedding_geometry.p_first_component -->. Very little
else in this module is that cheap.

!!! info "Terms used in this lesson"
    **Embedding** — a fixed-length vector of floats produced from a span of
    text by a model trained so that related text lands in nearby directions.
    Here, 384 <!-- computed: embedding_geometry.dim --> of them per document.

    **Cosine similarity** — the cosine of the angle between two vectors, equal
    to their dot product once both have been scaled to unit length. It measures
    direction only and ignores magnitude entirely.

    **L2 normalisation** — dividing a vector by its own length so it sits on
    the unit sphere. After this step the dot product and the cosine are the
    same number, which is why almost every retrieval system does it once at
    indexing time.

    **Anisotropy** — the property of a set of vectors being concentrated in
    some directions rather than spread evenly. An isotropic set of random
    directions in high dimensions has near-zero average pairwise cosine; this
    model's outputs have 0.84.

    **Principal component** — a direction in the space, obtained from the
    variance structure of the data, along which the vectors vary most. The
    first one here carries
    29.4% <!-- computed: embedding_geometry.top_component_share_pct --> of all
    the variance by itself.

    **Recall@k** — the fraction of queries for which at least one correct
    document appears in the top `k` results. The retrieval metric this lesson
    optimises; §3.8 covers the family properly.

## B · Mental model

**The vectors do not fill the sphere. They occupy a narrow cone, and retrieval
happens in the small angular differences within that cone.**

The picture in most explanations shows arrows spread across a globe, with
related concepts clustered and unrelated ones pointing away from each other.
The reality for this model, and for most sentence encoders, is that every arrow
points into roughly the same octant, and the entire business of retrieval is
conducted in the residual wobble between them.

That reframing has three immediate consequences, and each one is a mistake you
will otherwise make.

The first is that **absolute similarity scores carry no information you can
act on**. A score of 0.84 is not "similar" here, because 0.84 is the
background. What matters is exclusively how a score compares to the other
scores for the same query, which means the only trustworthy operations are
ranking and comparison against a per-query baseline.

The second is that **the margin doing the real work is tiny**. A query scores
0.731 <!-- computed: embedding_geometry.query_gold_cosine_mean --> against the
document that actually answers it and
0.659 <!-- computed: embedding_geometry.query_random_cosine_mean --> against a
document drawn at random, so the entire signal separating a right answer from
an arbitrary one is
0.072 <!-- computed: embedding_geometry.query_gold_margin -->. Everything this
module does later — hybrid retrieval, reranking, filtering — is an attempt to
widen that margin or to avoid relying on it alone.

The third is subtler and catches people who are otherwise careful. Notice that
a query sitting on its own gold document (0.731) scores *lower* than two
unrelated documents score against each other (0.841). Short text and long text
occupy different parts of the cone, so a similarity is not comparable across
different kinds of input. Comparing a query-document score to a
document-document score is meaningless, and any threshold tuned on one will be
wrong for the other.

??? question "If unrelated documents score 0.84, why does retrieval work at all?"
    Because ranking only needs the *ordering* to be right, not the scale. The
    gold document does not have to score high in absolute terms; it has to
    score higher than the other 2,418 candidates. That is a much weaker
    requirement, and it is why the system works despite a margin of 0.072 —
    and also why it degrades so ungracefully, since a margin that small is
    easily swamped by a document that happens to share surface features.

## C · Mechanism

A sentence embedding model is a transformer that runs over the tokens and
produces one vector per token position, followed by a **pooling** step that
collapses those into a single vector. Two pooling choices dominate: take the
vector at the special first position, which is CLS pooling, or take the
attention-mask-weighted average over all positions, which is mean pooling. The
model used here specifies CLS in its own configuration, and §G of the fixture's
own record shows why you should verify that claim rather than accept it.

The output is then L2-normalised, which puts every document on the unit sphere
and makes the dot product identical to the cosine. This is worth doing once at
indexing time rather than per query, because a dot product over a normalised
matrix is a single matrix multiply and a cosine computed from scratch is not.

The anisotropy has a well-understood cause. Token embeddings learned by
frequency-weighted objectives end up with a large shared component that
reflects token frequency rather than meaning, and pooling preserves it: every
document contains common words, so every document inherits the same dominant
direction. The result is a mean vector far from the origin, and cosine
similarity computed around a distant mean measures mostly the shared offset.

This is why the fix works and why it is so simple. Subtracting the corpus mean
recentres the cloud on the origin, which alone takes the average random-pair
cosine from 0.841 to
+0.003 <!-- computed: embedding_geometry.centred_pair_cosine_mean -->. Then
projecting out the leading principal direction removes what remains of the
shared component:

$$
\hat{v} = \frac{(v - \mu) - \big((v - \mu) \cdot u_1\big)\, u_1}{\lVert \cdot \rVert}
$$

where $\mu$ is the mean over the corpus vectors and $u_1$ is the first right
singular vector of the centred matrix. The same $\mu$ and $u_1$ must be applied
to queries, because a transform applied to one side of a comparison and not the
other is not a transform, it is a bug.

The variance spectrum explains how much room there is for this to help. Of the
nominal 384 dimensions, half the variance lies in just
3 <!-- computed: embedding_geometry.dims_for_50pct_variance --> directions,
90% lies in 38 <!-- computed: embedding_geometry.dims_for_90pct_variance -->,
and 95% in 73 <!-- computed: embedding_geometry.dims_for_95pct_variance -->.
The vectors are 384 numbers wide and roughly forty numbers deep.

??? question "Why renormalise after projecting out a component, when the input was already unit length?"
    Because the projection removes a different amount from each vector.
    Documents that carried a lot of the shared direction lose a lot of length
    and documents that carried little lose little, so the outputs have unequal
    magnitudes. A dot product between vectors of unequal length is not a
    cosine, and it systematically favours whichever documents happened to lie
    furthest from the removed direction.

## D · From data science to LLM systems

You have done this before, under a different name. Centring a feature matrix
and removing dominant components is standardisation followed by PCA, and the
motivation is identical: a feature whose variance dwarfs everything else
dominates any distance metric computed over the raw data, so you remove it and
let the remaining structure become visible. If you have ever been surprised by
a k-nearest-neighbours model that keyed entirely on one unscaled column, you
have met this exact failure in its classical form.

The analogue is close enough to be useful and it breaks in three places, all of
which matter in production.

**You cannot refit per query.** In ordinary modelling you fit the transform on
training data and apply it at inference, and both happen in your process. Here
$\mu$ and $u_1$ are derived from the corpus, stored beside the index, and
applied to every incoming query — which means they are a piece of *state* your
retrieval service now owns and must version alongside the vectors.

**The two sides of the comparison have different distributions.** Queries are
short and documents are long, so they populate different regions of the cone,
and the statistics you compute from documents are not the statistics of
queries. Fitting $\mu$ on the documents and applying it to queries is what this
lesson does and what the literature does, but it is an approximation rather than
a principled whitening of a single population.

**Refitting invalidates the index.** In classical modelling, recomputing a
transform costs one pass over the data. Here the vectors have already been
written into a search index, possibly quantised, possibly built into a graph
structure that assumed those exact coordinates, so changing $\mu$ means
rebuilding all of it. That is an operational cost with no counterpart in the
sklearn version of this idea, and it is the reason people leave a suboptimal
transform in place for years.

??? question "You add 50,000 documents of a type the corpus did not previously contain. Does the transform need refitting?"
    Almost certainly yes, because the mean is a property of what is in the
    corpus and a new document type moves it. Adding more of what is already
    there does not. The reason this is a real decision rather than a
    formality is that refitting invalidates the index built from the old
    coordinates, so the answer is not automatically "refit on a schedule".

## E · Minimal implementation

The whole technique is eight lines. `docs` is the matrix of corpus vectors and
`queries` the matrix of query vectors, both already L2-normalised.

```python
import numpy as np

def fit_transform(docs: np.ndarray, n_components: int = 1):
    """Return (mu, components) to be applied to both sides of the comparison."""
    mu = docs.mean(axis=0)
    # Right singular vectors of the CENTRED matrix. Centring first is not
    # optional: the SVD of an uncentred matrix returns the mean direction as
    # its first component, so you would remove the same thing twice and lose
    # a genuine axis of variation on the second pass.
    _, _, components = np.linalg.svd(docs - mu, full_matrices=False)
    return mu, components[:n_components]

def apply(v: np.ndarray, mu: np.ndarray, components: np.ndarray) -> np.ndarray:
    v = v - mu
    v = v - (v @ components.T) @ components
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
```

The line that repays attention is the re-normalisation at the end. Projecting
out a component shortens the vector by an amount that differs per document,
because documents differ in how much of that shared direction they carried, and
skipping the renormalisation leaves you comparing dot products between vectors
of unequal length — which is no longer cosine similarity, and quietly favours
whichever documents happened to lie furthest from the removed direction.

## F · Production practice

Store the vectors int8-quantised with a per-vector scale. The fixture behind
this lesson does exactly that, and the cosine fidelity between the quantised
and float32 versions is 0.99984, which costs nothing measurable in retrieval
quality while cutting storage by a factor of four. Per-vector scales rather
than one global scale matter here, because a vector concentrated in a few
dimensions has a much larger peak than a flat one and a shared scale quantises
the flat ones far too coarsely.

Pin the model by commit, not by name and not by tag. The fixture records
`bge-small-en-v1.5` at revision `5c38ec7c`, because a repository can be updated
in place and a tag can be moved, and an embedding that changes underneath a
stored index produces a system that is subtly and unfixably wrong — old vectors
and new queries in the same space, with nothing to indicate it.

Treat $\mu$ and $u_1$ as versioned artefacts stored with the index, and record
which one built it. The failure mode when these drift apart is that retrieval
quality degrades without a single error being raised anywhere, since every
operation still succeeds and merely returns worse answers.

Recompute the transform when the corpus composition changes materially, not on
a schedule. $\mu$ is a property of what is in the corpus, so adding a large new
document type shifts it, whereas adding more of what is already there does not.

??? question "Retrieval quality has degraded and no errors are being logged. What would you check first?"
    Whether the mean and component stored with the index are the ones the
    query path is applying. Every one-sided or mismatched transform still
    produces plausible numbers in a plausible range, so nothing raises, and
    the only symptom is worse answers. This is the argument for a test that
    asserts a recall floor against a fixed query set rather than relying on
    review.

## G · Experiment

`python experiments/embedding_geometry.py` measures the shipped fixture. Every
comparison is an exact paired McNemar test over the
176 <!-- computed: embedding_geometry.n_queries_answerable --> answerable
queries, for the reason lesson 0.3 gives at length: these differences are a
handful of queries changing hands, and the aggregate cannot distinguish a real
effect from a coin flip.

| configuration | recall@10 | won | lost | p |
|---|---:|---:|---:|---:|
| raw | 0.489 | — | — | — |
| centred | 0.534 | 15 | 7 | 0.1338 |
| centred, minus 1 component | 0.614 | 14 | 0 | 0.0001 |
| centred, minus 2 components | 0.540 | 5 | 18 | 0.0106 |

Three things in that table are worth more than the headline.

**Centring on its own does not survive the test.** It looks like a gain of four
and a half points, which in a blog post would be reported as an improvement and
adopted. Fifteen queries improve, seven get worse, and
p = 0.1338 <!-- computed: embedding_geometry.p_centring --> says that split is
unremarkable. If the only tool applied here had been a comparison of aggregate
recall, the wrong conclusion was available and comfortable.

**The full transform is unambiguous.** Against the centred baseline, removing
the first component wins
14 <!-- computed: embedding_geometry.won_first_component --> queries and loses
0 <!-- computed: embedding_geometry.lost_first_component -->. A clean sweep is
rare in retrieval experiments and it is what a genuine effect looks like when
the mechanism is real rather than incidental.

**The optimum is sharp and it is at one.** Removing a second component gives
back most of the gain, and does so significantly at
p = 0.0106 <!-- computed: embedding_geometry.p_second_component -->. The first
direction is shared junk; the second is already carrying signal. Anyone who
reads "remove the dominant components" as licence to remove several will make
things worse and, without a paired test, will not be able to tell.

??? question "Recall improved from 0.489 to 0.534 and the paired test returned p = 0.13. What do you report?"
    That the change is not supported, and ideally what it would take to
    detect an effect of that size — roughly, how many more queries. Reporting
    "a 4.5-point improvement" is the tempting version and it is not what the
    data says. The measurement is not that centring fails; it is that this
    benchmark cannot distinguish a gain that small from noise.

## H · Failure modes and cost traps

**Thresholding on absolute similarity.** This is the big one and it is
everywhere. `if score > 0.8` is a filter that admits the entire corpus on this
model and admits nothing at all on a model whose outputs are better spread. The
number is not portable across models, across pooling choices, or between
query-document and document-document comparisons. Rank, or calibrate the
threshold against a measured distribution, and re-derive it whenever the model
changes.

**Applying the transform to one side only.** Fit $\mu$ on documents, forget to
subtract it from queries, and every operation still runs. Similarities remain
plausible numbers in a plausible range and the results are quietly much worse.
There is no exception and no warning, which is why this belongs in a test that
asserts a recall floor rather than in a code review.

**Fitting the transform on a sample and applying it to everything.** $\mu$
estimated from a few hundred documents of one type is not the corpus mean, and
the resulting recentring pushes that type toward the origin relative to the
rest. If the sample is not representative the transform will actively harm the
under-represented documents.

**Assuming more components is more better.** Measured above: the second
component costs you significantly. This is the failure most likely to be
introduced by someone who half-remembers the technique.

**Forgetting that the index encodes the geometry.** Changing $\mu$ or $u_1$
after building an approximate index is not a configuration change; it is a
rebuild. Approximate structures are built from the coordinates they were given,
and feeding them differently-transformed queries afterwards degrades recall in
a way that looks like the index is broken rather than misused.

**Assuming a relevant document scores well in absolute terms.** Take the query
in the tier-3 exercise below and look up where its correct answer falls in the
background distribution of query-document similarities: the 40th percentile. A
genuinely relevant document, scoring *below* the median pairing of unrelated
text. Calibration does not repair that — it makes it visible, where the raw
score of 0.637 did not. Any design that assumes the right answer is
identifiable from its score alone has assumed something this model does not
provide, and lessons 3.4 and 3.6 exist because of it.

**A mistake made writing this lesson.** The first version of the underlying
corpus produced a pooling comparison in which mean pooling beat CLS pooling by
ten queries to nil, at p = 0.0020. That result was decisive, reproducible, and
entirely an artefact of a defect in the corpus — every policy document shared
one body of text. On the corrected corpus the same comparison is twelve to five
at p = 0.1435, and the model card's own choice stands. A significance test run
against a broken benchmark measures the breakage with great confidence.

## I · Graded practice

<quiz-bank src="ret-l1"></quiz-bank>

<code-exercise src="ret-l1-cone"></code-exercise>

<code-exercise src="ret-l1-abtt"></code-exercise>

## J · Annotated references

- **Ethayarajh (2019), "How Contextual are Contextualized Word
  Representations?"** — the paper that measured anisotropy carefully and showed
  how extreme it is in contextual encoders. Read §4 for the geometry.
- **Mu and Viswanath (2018), "All-but-the-Top"** — the origin of the
  centre-and-project technique implemented here, on static word vectors. Their
  recommendation of roughly `d/100` components is worth contrasting with the
  measurement in §G, which finds one.
- **Reimers and Gurevych (2019), "Sentence-BERT"** — why pooled transformer
  outputs need training designed for similarity before cosine over them means
  anything, and the source of the mean-versus-CLS question in §C.
- **Robertson and Zaragoza (2009), "The Probabilistic Relevance Framework"** —
  the lexical counterpart, and the baseline lesson 3.3 builds. Dense retrieval
  is not obviously better than this until you measure both.

## K · Extension

*Off-platform, needs a GPU or patience and about two hours.* Take a second
embedding model of a different size and repeat §G's measurement end to end.
Two questions are worth the time: does the sharp optimum at one component hold,
or is the right number a property of the model rather than a universal? And
does the anisotropy shrink with model size, as is sometimes claimed? Report the
paired test rather than the aggregate, and treat any result that does not
survive it as absent.
