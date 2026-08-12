---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 2.2 · Few-shot selection is a retrieval problem

## A · Why this matters

Almost every discussion of few-shot prompting is about how many examples to
include and how to format them, and almost none of it is about *which* examples
to include — which is odd, because that is the only part of the decision with a
correct answer you can compute.

Choosing `k` examples from a pool is a retrieval problem, and you already own
the machinery for it. The consequences are measurable without a model anywhere
in the loop, which makes this one of the few genuinely checkable questions in a
module about prompting. Measured on a pool of forty labelled support tickets,
selecting examples at random leaves
22.5% <!-- computed: fewshot_selection.random_missing_own_label_pct --> of
prompts containing no example of the label the query actually needs, while any
informed strategy takes that to essentially zero.

That is the easy result. The interesting one, in §G, is that the strategy which
*sounds* safest is the one that fails hardest under a condition nobody checks.

!!! info "Terms used in this lesson"
    **Example pool** — the labelled examples available to choose from. It is a
    retrieval corpus, and thinking of it that way is most of this lesson.

    **Relevance** — how similar a chosen example is to the query. Maximising
    it alone is the `top-k` strategy.

    **Redundancy** — how similar the chosen examples are *to each other*.
    High redundancy means spending several example slots to show the model one
    thing.

    **Label coverage** — how many distinct labels appear among the chosen
    examples, as a fraction of the most you could have shown.

    **MMR (maximal marginal relevance)** — a selection rule that scores each
    candidate as its relevance to the query minus its similarity to whatever
    has already been chosen, trading a little of the former for the latter.

## B · Mental model

**The example pool is a retrieval corpus and the prompt is a results page.**

Once you accept that framing, the whole vocabulary of Module 3 arrives early
and for free: there is a query, a corpus, a similarity function, a budget of
`k` slots, and a question about what a good result set looks like. And the
answer to that last question is the same one search engines reached decades
ago, which is that a good result set is not simply the `k` most similar items,
because the most similar items tend to resemble each other as much as they
resemble the query.

Three properties compete for the same `k` slots, and no strategy maximises all
three at once:

| Property | Maximised by | What it costs |
|---|---|---|
| Relevance to the query | top-k | Slots spent on near-duplicates |
| Diversity of the set | MMR | A little relevance |
| Coverage of the label space | round-robin balance | Relevance, and sometimes correctness |

The reason this matters more than it sounds is that `k` is small and the slots
are expensive. At `k = 5` on the pool below, top-k selection shows the model
only 2.8 <!-- computed: fewshot_selection.topk_distinct_of_k --> distinct
labels, which means roughly two of the five slots are spent restating something
the model has already been shown.

??? question "Why would you ever want examples that are *less* similar to the query?"
    Because the examples are doing two jobs at once, and similarity only
    serves the first. They demonstrate the output format and they demonstrate
    the decision boundary, and a set of five near-identical examples shows the
    boundary from one side only. If every example you supply has the same
    label, you have shown the model what that label looks like and nothing
    about when *not* to apply it.

## C · Mechanism

**Similarity.** Represent each example and the query as vectors and use cosine
similarity between them. In §G that representation is TF-IDF computed from the
pool itself, which is deliberate: it involves no model at all, so the results
are properties of the text and the arithmetic rather than of anything learned,
and Module 3 replaces it with real embeddings once the corpus exists.

**Top-k** takes the `k` highest-similarity examples and stops. It maximises
mean relevance by construction, and it has no mechanism whatsoever for noticing
that its picks resemble one another.

**MMR** builds the set one element at a time, and at each step scores every
remaining candidate as

$$
\text{score}(i) = \lambda \cdot \text{sim}(i, q) \;-\; (1 - \lambda)\,\max_{j \in S}\ \text{sim}(i, j)
$$

where `S` is what has been chosen so far. The first term is relevance to the
query and the second is redundancy against the existing set, so `λ = 1`
degenerates to top-k and `λ = 0` ignores the query entirely. The greedy
construction matters: each pick changes the penalty applied to every remaining
candidate, which is why this cannot be computed as a single ranking.

**Round-robin balance** sorts the pool by similarity, then walks the labels in
turn taking the best unused example of each. It guarantees label coverage when
`k` is at least the number of labels, and §G shows what it does when `k` is
smaller — which is the part of this lesson worth remembering.

??? question "MMR's redundancy term is a `max` over the chosen set rather than a mean. Why does that choice matter?"
    Because a `max` penalises a candidate for being close to *any* single item
    already chosen, whereas a mean lets a candidate that duplicates one
    existing pick hide behind its distance from the others. With five slots
    and one tight cluster, the mean would happily admit a second and third
    member of that cluster, since each is far from everything outside it. The
    `max` is what makes the penalty bite on exactly the case the strategy
    exists to prevent.

## D · From data science to LLM systems

This is a problem you have solved before, several times, under different names.

| You know | Here |
|---|---|
| k-nearest neighbours | Selecting examples by similarity to the query |
| Diversity in a recommendation slate | MMR over the example pool |
| Stratified sampling | Round-robin balance over labels |
| Class imbalance in training data | Label skew in the selected examples |
| Feature selection under a budget | Example selection under a token budget |

The transfer is unusually clean, and the one adjustment is a matter of scale
rather than of kind. Your instincts about stratification were formed on
training sets of thousands of rows, where a stratified sample of a few hundred
comfortably represents every class. Here the "sample" has five slots and the
label space may have more classes than that, so a stratification rule that is
harmless at `n = 500` becomes an active liability at `n = 5` — which is exactly
the failure §G measures.

The second adjustment is that the selection runs *per query*, at request time,
inside your latency budget. A selection strategy that requires an expensive
computation over the whole pool for every request is not the same proposition
as one you run once offline, and MMR's greedy loop is `O(k · |pool|)`
similarity lookups rather than a single sort.

??? question "Your pool has 12 labels and your token budget allows 4 examples. What does that force?"
    It forces you to give up on covering the label space, and therefore to
    decide *which* labels matter for this query rather than pretending you can
    show them all. Coverage-driven selection is simply unavailable below
    `k = 12`, so the sensible strategies are relevance-driven ones — top-k or
    MMR — possibly with a rule that reserves one slot for the single most
    likely label. The mistake is applying a balance rule anyway and assuming it
    is the conservative choice.

## E · Minimal implementation

Both informed strategies fit in a few lines each. Top-k is a sort:

```python
def select_topk(sim, k):
    return list(np.argsort(-sim)[:k])
```

MMR is the same idea with a penalty that updates as the set grows:

```python
def select_mmr(sim, pairwise, k, lam=0.5):
    chosen, candidates = [], list(range(len(sim)))
    while len(chosen) < k and candidates:
        best, best_score = None, -math.inf
        for i in candidates:
            redundancy = max((pairwise[i, j] for j in chosen), default=0.0)
            score = lam * sim[i] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        candidates.remove(best)
    return chosen
```

The `default=0.0` in that `max` is doing quiet work: on the first iteration
nothing has been chosen, so the redundancy term is zero and the first pick is
simply the most relevant example. Without the default the expression raises on
an empty sequence, and the natural-looking fix of seeding `chosen` with the
top-1 result before the loop gives the same answer while making the special
case harder to see.

## F · Production practice

Compute the pool's embeddings once, offline, and cache them, because they do
not depend on the query and recomputing them per request is the single
most common performance mistake in this part of a system. Only the query
embedding and the similarity lookup belong on the request path.

Keep the pool versioned alongside the prompt template, since changing which
examples exist changes the model's behaviour exactly as much as changing the
instructions does, and a pool edited in place is an undocumented deployment.
Lesson 0.1's cohort key should include the pool version for the same reason it
includes the prompt hash.

Set `k` from the token budget rather than from a round number, using
[1.3](../01-tokens/03-context-windows.md)'s accounting: examples compete with
retrieved context and with the answer for one window, and five long examples
can cost more than the document you were trying to reason about.

And re-derive the selection strategy when the pool changes shape. The results
below are specific to a pool whose labels are lexically well separated, and
§H explains why that assumption is load-bearing.

## G · Experiment

```bash
python experiments/fewshot_selection.py
```

Forty labelled support tickets across five intents, `k = 5`, leave-one-out over
every example so that each query has a known correct label and never sees
itself. No model is involved: similarity is cosine over TF-IDF vectors computed
from the pool.

| Strategy | Relevance | Redundancy | Label coverage | Own label present |
|---|---|---|---|---|
| random | 0.058 <!-- computed: fewshot_selection.random_relevance --> | 0.055 <!-- computed: fewshot_selection.random_redundancy --> | 69.5% <!-- computed: fewshot_selection.random_coverage_pct --> | 77.5% <!-- computed: fewshot_selection.random_label_hit_pct --> |
| top-k | 0.172 <!-- computed: fewshot_selection.top_k_relevance --> | 0.093 <!-- computed: fewshot_selection.top_k_redundancy --> | 57.0% <!-- computed: fewshot_selection.top_k_coverage_pct --> | 100.0% <!-- computed: fewshot_selection.top_k_label_hit_pct --> |
| MMR | 0.142 <!-- computed: fewshot_selection.mmr_relevance --> | 0.034 <!-- computed: fewshot_selection.mmr_redundancy --> | 72.0% <!-- computed: fewshot_selection.mmr_coverage_pct --> | 97.5% <!-- computed: fewshot_selection.mmr_label_hit_pct --> |
| balanced | 0.141 <!-- computed: fewshot_selection.balanced_relevance --> | 0.069 <!-- computed: fewshot_selection.balanced_redundancy --> | 100.0% <!-- computed: fewshot_selection.balanced_coverage_pct --> | 100.0% <!-- computed: fewshot_selection.balanced_label_hit_pct --> |

??? question "Before reading the table: which strategy do you expect to have the *highest* mean relevance, and is that a good thing?"
    Top-k, necessarily — it is defined as the argmax of relevance, so nothing
    else can beat it and any strategy that ties has reduced to it. Whether
    that is good depends on what the slots are for: high mean relevance with
    high redundancy means the set is tightly clustered around the query, which
    demonstrates the format well and the decision boundary barely at all.
    Relevance is a constraint to satisfy rather than a quantity to maximise.

**Selection matters at all**, which is worth establishing before anything
subtler. Random selection leaves nearly a quarter of prompts without a single
example of the label the query needs, and every informed strategy fixes that
completely at this `k`.

**MMR is an unusually good trade.** It cuts redundancy by
63.4% <!-- computed: fewshot_selection.mmr_redundancy_drop_pct --> for
0.03 <!-- computed: fewshot_selection.mmr_relevance_cost --> of mean
similarity, which is the kind of ratio that makes a default. Top-k's five slots
show only 2.8 <!-- computed: fewshot_selection.topk_distinct_of_k --> distinct
labels, so two of them are effectively repeats.

### The result that contradicted my prediction

I built this experiment expecting to show that top-k frequently omits an
example of the query's own label, because that is the standard argument for
diversity-aware selection. **It does not: top-k scores
100.0% <!-- computed: fewshot_selection.top_k_label_hit_pct -->.** The
prediction was wrong, and the reason it was wrong is more useful than the
result I expected. These five intents use largely distinct vocabulary, so an
example's nearest neighbours are nearly always same-label, and nearest-neighbour
selection is already label-correct without being asked to be. On a pool whose
labels *share* vocabulary the argument would look quite different — which makes
"how separable are my labels?" a question to answer on your own pool rather
than a premise to inherit from a blog post.

### Where the safe-sounding strategy breaks

Sweeping `k` produces the finding that survives:

<figure class="llm-fig" markdown>
![The percentage of prompts containing an example of the query's own label, plotted against k for four strategies. The balanced curve starts far below the others and crosses them at k equal to the number of labels, after which all informed strategies sit at 100%.](../../assets/generated/figures/fewshot-selection-light.svg){.fig-light}
![The percentage of prompts containing an example of the query's own label, plotted against k for four strategies. The balanced curve starts far below the others and crosses them at k equal to the number of labels, after which all informed strategies sit at 100%.](../../assets/generated/figures/fewshot-selection-dark.svg){.fig-dark}
<figcaption markdown>How often the query's own label appears among the selected examples, against `k`. The vertical line is `k = 5`, the number of labels in the pool.</figcaption>
</figure>

| | k = 3 coverage | k = 3 own label | k = 8 own label |
|---|---|---|---|
| top-k | 70.0% <!-- computed: fewshot_selection.k3_top_k_coverage_pct --> | 95.0% <!-- computed: fewshot_selection.k3_top_k_label_hit_pct --> | 100.0% <!-- computed: fewshot_selection.k8_top_k_label_hit_pct --> |
| MMR | 81.7% <!-- computed: fewshot_selection.k3_mmr_coverage_pct --> | 92.5% <!-- computed: fewshot_selection.k3_mmr_label_hit_pct --> | 100.0% <!-- computed: fewshot_selection.k8_mmr_label_hit_pct --> |
| **balanced** | **100.0% <!-- computed: fewshot_selection.k3_balanced_coverage_pct -->** | **60.0% <!-- computed: fewshot_selection.k3_balanced_label_hit_pct -->** | 100.0% <!-- computed: fewshot_selection.k8_balanced_label_hit_pct --> |

At `k = 3` with five labels, the balanced strategy achieves perfect coverage of
the labels it *can* reach and supplies an example of the query's own label only
60% <!-- computed: fewshot_selection.k3_balanced_label_hit_pct --> of the
time — markedly worse than the strategy that ignores balance entirely. Forcing
coverage when there are not enough slots to cover anything spends scarce slots
on labels chosen without reference to the query, and two prompts in five then
contain no example of the answer being asked for.

**Balanced selection is safe when `k ≥ the number of labels`, and actively
harmful below it.** That is a condition anybody can check in one line, and it
is not a condition anybody usually states.

??? question "Balanced selection has perfect label coverage at k = 3 and the worst own-label rate. How can both be true?"
    Because coverage counts how many *distinct* labels appear, and it is
    computed against the most you could have shown — three, when `k = 3`. A
    round-robin over five labels does produce three distinct ones every time,
    so coverage is a perfect 100%. It simply has no reason to make one of them
    the query's label, since the round-robin order does not consult the query.
    The metric is measuring exactly what it says and it is the wrong metric to
    optimise alone.

## H · Failure modes and cost traps

**Selecting examples at random because "the model does not really care".** It
does: at `k = 5` a random pool leaves 22.5% of prompts with no example of the
needed label, and at `k = 3` that rises to two thirds.

**Optimising coverage without checking `k` against the label count.** The
finding above. Below `k = number of labels` a balance rule sacrifices the
metric you care about in order to satisfy one you introduced.

**Assuming these numbers transfer to your pool.** They will not, and the reason
is stated rather than hidden: the intents here are lexically well separated, so
similarity alone is already label-correct. Run the same leave-one-out check on
your own pool before adopting any strategy; it is thirty lines and it is the
only version of this comparison that describes your data.

**Recomputing pool embeddings on every request.** They do not depend on the
query. Cache them once and keep the request path to a single query embedding
and a similarity lookup.

**Editing the pool in place.** Changing which examples exist changes behaviour
exactly as much as editing the instructions, so an unversioned pool is an
undeployed deployment nobody can roll back.

**Treating this as evidence about answer quality.** It is not, and this is the
limit of what the experiment can support. Every metric here concerns the
*composition* of the example set, which is fully checkable without a model;
whether a better-composed set produces better answers is a question for
[lesson 0.3](../00-transition/03-evaluation-breaks.md)'s machinery on your own
task, and no amount of selection arithmetic substitutes for it.

**Letting examples crowd out the context.** Five verbose examples can cost more
than the document the question is actually about, and both come out of one
window.

## I · Graded practice

<code-exercise src="prm-l2-mmr"></code-exercise>

<code-exercise src="prm-l2-balanced"></code-exercise>

<quiz-bank src="prm-l2"></quiz-bank>

## J · Annotated references

- **Carbonell & Goldstein (1998), *The Use of MMR…*.** The original
  formulation, written for document summarisation and retrieval. Two pages of
  it are enough, and the framing transfers without modification.
- **Liu et al. (2021), on what makes good in-context examples.** The paper
  that established retrieval-based example selection as better than random
  selection; findable by author and year, and its title names a model, which
  is why it is not reproduced here. Worth reading against §G's caveat about
  label separability.
- **Any treatment of diversity in recommender result sets.** The same
  relevance-versus-redundancy trade, with a decade more thought behind it than
  the prompting literature has had.

## K · Extension

**Run the leave-one-out check on your own pool.** Take whatever examples you
currently paste into prompts, label them, embed them with anything to hand, and
compute the four numbers in §G. The one to look at first is the own-label rate,
because it is the only one that corresponds to something going visibly wrong.

Then answer the question that decides your strategy: **how separable are your
labels?** If nearest-neighbour selection is already label-correct, as it was
here, then diversity is buying you decision-boundary coverage rather than
correctness, and MMR is a refinement. If it is not — if your labels share
vocabulary, which is common for intents that differ by nuance rather than by
topic — then relevance alone is not enough and the argument for explicit
balance gets much stronger. That single measurement changes which of §G's rows
you should be reading.

**And check what your examples are costing you.** Count the tokens in your
current example set with [1.3](../01-tokens/03-context-windows.md)'s
accounting, then compare that against the tokens available for retrieved
context in the same request. Few-shot examples and retrieved documents compete
for one window, and the comparison frequently shows a system spending more of
its budget demonstrating the output format than supplying the facts the answer
depends on. If that is where you land, the cheapest fix is usually fewer and
shorter examples rather than a cleverer selection rule — which is worth knowing
before you implement MMR.
