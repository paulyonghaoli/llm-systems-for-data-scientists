---
status: Verified
last_verified: 2026-08-13
volatility: medium
pyodide: true
prereqs: ["4.3"]
---

# 4.4 · Agent memory

## A · Why this matters

An agent re-reads its whole history before every model call, so a twenty-step
episode pays for its context twenty times. The observations in the episode
below come to
479 <!-- computed: agent_memory.episode_tokens --> tokens; carrying them costs
5,078 <!-- computed: agent_memory.keep_all_carried -->, which is
10.6× <!-- computed: agent_memory.carry_multiple --> the text itself. That
multiple is why memory management exists, and it grows with the length of the
episode rather than staying put.

So something gets dropped. Five retention policies over the same episode:

| policy | carried | gross saving | re-fetches | net | net saving | given back |
|---|---:|---:|---:|---:|---:|---:|
| keep_all | 5078 | 0.0% | 0 | 5078 | 0.0% | — |
| recency | 2539 | **50.0%** | 14 | 4768 | **6.1%** | 87.8% |
| truncated | 3589 | **29.3%** | 11 | 5316 | **−4.7%** | 116.0% |
| keyed | 960 | 81.1% | 5 | 1683 | 66.9% | 17.6% |
| keyed+recency | 3499 | 31.1% | 5 | 4222 | 16.9% | 45.8% |

**The gross column is the one everybody reports and the net column is the one
you pay.** A dropped fact has to be fetched again, and the fresh observation is
then carried for every remaining step, so the saving comes back as a cost with
interest. A sliding window that looks like it halves the bill delivers
6.1% <!-- computed: agent_memory.recency_net_pct -->, having given back
87.8% <!-- computed: agent_memory.recency_given_back_pct --> of what it saved.

**And truncating old observations is worse than doing nothing.** It appears to
save 29.3% <!-- computed: agent_memory.truncated_gross_pct --> and its net is
-4.7% <!-- computed: agent_memory.truncated_net_pct -->: it costs more than
keeping everything, because it keeps enough of each observation to still be
paying for it and not enough for the facts to survive. A compaction that is
lossy *and* not very compact is the worst of both, and nothing in the gross
column says so.

!!! info "Terms used in this lesson"
    **Working memory** — what is in the context window right now. Bounded, and
    re-read every step.

    **Episodic memory** — the record of what happened in this episode: the
    trajectory. Lesson 4.6 grades it.

    **Semantic memory** — facts extracted out of the episode and kept as
    structured data. The `keyed` policy below.

    **Compaction** — replacing old observations with something smaller.
    Summarisation is the usual form; truncation is the measurable one here.

    **Retention policy** — the rule deciding what is still in context at step
    *n*. Every memory design is one of these.

    **Need-distance** — how many steps separate a fact being established from
    a step needing it. The axis every recall number here is reported against.

    **Silent miss** — a fact the agent no longer has and does not know it no
    longer has.

## B · Mental model

**Agent memory is a cache whose misses are silent.**

An ordinary cache tells you when it missed, and the miss triggers a fetch. An
agent that has dropped a fact receives no signal at all: the context simply
does not contain it, the model proceeds with what is there, and the outcome is
a re-fetch if you are lucky and a confident wrong answer if you are not. Every
number in §A's re-fetch column is the lucky case, and the whole reason to
measure recall directly is that the failure has no other symptom.

The second half of the model is the question that replaces "how small can I
make this". A policy's job is to still be holding the fact a later step needs,
so the quantity to measure is **recall at a given need-distance**, and size is
merely the budget it has to do that in. Reporting compression without recall is
reporting one side of a trade.

??? question "Isn't a re-fetch a perfectly good recovery? Why care what was dropped?"
    A re-fetch is the good case and it is not free — §A's re-fetch column costs
    recency 87.8% of its gross saving. The bad case is the one with no column:
    the agent proceeds without the fact, and there is no signal distinguishing
    "the model decided this was irrelevant" from "the model never saw it". A
    policy evaluated only on the episodes that recovered is being evaluated on
    its best behaviour.

## C · Mechanism

**Tag retained text with where it came from.** This is the one implementation
detail that decides whether the rest of the measurement means anything:

```python
@dataclass
class Retained:
    texts: dict[int, str]        # origin index -> the text still held
    keyed: dict[str, str]

    def holds(self, fact):
        if fact.key is not None and fact.key in self.keyed:
            return True
        return fact.needle in self.texts.get(fact.origin, "")
```

The obvious version asks whether the fact's text appears *anywhere* in context.
That is wrong, and it is wrong in the flattering direction: a repeated search
re-establishes the same sentence later, so a window scores credit for a fact it
dropped. The first version of this experiment gave a six-step window 0.70
recall on facts fifteen steps old, which is impossible, and only the origin tag
makes the number real.

**Distinguish facts that arrived with a key from facts stated in prose.** A
structured result — `{"shipment": ..., "status": ..., "depot": ...}` — has
field names, which is exactly the condition under which an extractor can
capture it. A sentence in a file has no key, so nothing knows to look for it
unless somebody anticipated it. That distinction is not a modelling
convenience; it is the actual reason semantic memory works for some facts and
not others.

**Charge a re-fetch properly.** Per missing fact, not per observation, and
including the carrying cost of the re-fetched observation for the rest of the
episode:

```python
refetch_tokens += obs[i].tokens() * (n - step + 1)
```

Charging per observation lets a policy that dropped twenty facts pay for five,
and the policy with the worst recall then looks cheapest. That is a
benchmark-design error of the same family as lesson 4.3's, and it happened here
before the per-fact version was written.

??? question "The keyed/prose split sounds like an artefact of this sandbox. Does it hold outside it?"
    It holds wherever tool results are structured and documents are not, which
    is most systems: an API returns fields with names and a knowledge base
    returns text, so the same extractor that captures every identifier for free
    captures nothing at all from the paragraph explaining what the identifier
    means. What the sandbox does control is the *ratio* — this episode has
    15 <!-- computed: agent_memory.n_keyed_facts --> keyed facts against
    10 <!-- computed: agent_memory.n_prose_facts --> prose ones, and a
    document-heavy workload would shift that balance and with it how much the
    keyed column is worth. The shape of the failure does not shift, only how
    much of your episode falls on each side of it.

**Assert the invariant.** A policy that keeps everything must lose nothing:

```python
assert net_cost(obs, keep_all)["refetches"] == 0, "keep_all dropped a fact"
```

This assertion failed on the first run, which is how the origin-tagging bug was
found. An invariant that cannot fail is not worth writing; this one could and
did.

## D · From data science to LLM systems

This is cache eviction, and you have the vocabulary already: a bounded store, a
policy deciding what to evict, a hit rate. A sliding window is LRU with the
recency test made explicit, and semantic extraction is a materialised view.

The transferable part is the discipline of reporting hit rate alongside size.
The part that does *not* transfer is the assumption underneath LRU, and it is
worth being precise about because it inverts.

**LRU works because of temporal locality: recently used predicts soon used.**
An agent episode has the opposite property. The facts a later step depends on
are the ones established *earliest* — lesson 4.3's discovered dependencies are
early-produced and late-consumed by construction, because that is what makes
them dependencies. So the age of a fact is, if anything, *positively*
correlated with its importance, and a policy that evicts by age is evicting by
a signal pointing the wrong way. The measured shape of that is §G's cliff:
recall 1.000 everywhere inside the window and 0.000 everywhere outside it.

The second difference is the silent miss from §B. A cache miss is an event you
can count without instrumenting anything else; an agent's is invisible unless
you go looking for the fact you already know it needed.

??? question "Could you evict by predicted importance instead of by age?"
    That is the right direction and it costs something the eviction policy
    cannot spend cheaply. Predicting importance means either a model call per
    eviction — on the budget you are trying to protect — or a heuristic, and
    the honest heuristic available here is exactly the keyed/prose split: keep
    what arrived with a key. §G shows how far that gets you, which is the whole
    keyed column, and where it stops, which is every prose fact in the episode.

## E · Minimal implementation

Four policies, all one-liners over the same tagged store:

```python
def keep_all(obs, step):
    return Retained(texts={i: obs[i].text for i in range(step)})

def recency(obs, step, window=6):
    return Retained(texts={i: obs[i].text for i in range(max(0, step - window), step)})

def keyed_only(obs, step):
    return Retained(keyed=extract(obs, step))       # keys survive forever

def keyed_plus_recency(obs, step, window=6):
    return Retained(texts={i: obs[i].text for i in range(max(0, step - window), step)},
                    keyed=extract(obs, step))
```

The composite is four lines and buys the whole keyed column at every distance.
It is worth noticing how little code the useful policy is, relative to how
often the shipped default is the bare window.

The cost of an episode is then a sum over steps rather than a single number,
which is the arithmetic that makes carrying expensive:

```python
def carried_tokens(obs, policy):
    return sum(policy(obs, step).tokens() for step in range(1, len(obs) + 1))
```

## F · Production practice

**Report recall at a need-distance, never compression alone.** A policy with a
compression figure and no recall figure has reported the half of the trade that
always looks good.

**Extract structured facts at write time and keep them forever.** They are
tiny — the keyed store carries
960 <!-- computed: agent_memory.keyed_carried --> tokens against
5,078 <!-- computed: agent_memory.keep_all_carried --> — and they are exactly
the identifiers later steps depend on.

**Do not ship a bare sliding window.** It is the default in most frameworks,
its recall is a cliff rather than a slope, and the cliff sits where the
important facts are.

**Combine, do not choose.** `keyed+recency` has full recall of keyed facts at
every distance and of everything inside the window, for four lines of code.

**Instrument the miss.** Log when a re-fetch happens and what was missing.
Without it the only visible symptom of a bad retention policy is a slightly
worse answer.

**Compact once, in the middle, rather than continuously.** §G puts the number
on it.

## G · Experiment

`python experiments/agent_memory.py`, over a
20 <!-- computed: agent_memory.n_steps -->-step episode whose every observation
is a real `llmlab.tools` envelope, so the sizes are measured rather than
invented. The facts a later step needs are authored, and the need-distance is
swept rather than picked.

**Recall by need-distance, keyed facts then prose facts:**

| policy | d=1 | d=3 | d=5 | d=8 | d=12 |
|---|---:|---:|---:|---:|---:|
| recency (keyed) | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** |
| truncated (keyed) | 1.000 | 1.000 | 1.000 | 0.333 | 0.333 |
| keyed | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| keyed+recency | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| recency (prose) | 1.000 | 1.000 | 1.000 | **0.000** | **0.000** |
| keyed (prose) | **0.000** | **0.000** | **0.000** | 0.000 | 0.000 |
| keyed+recency (prose) | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

**Recency's recall is a cliff.** Not a decay, not a gradient — 1.000 at every
distance inside the window and 0.000 at every distance outside. There is no
regime where it is degrading and you might notice.

**The keyed store and the window fail on disjoint classes**, which is the
argument for the composite and also its limit. Structured extraction is perfect
on keyed facts at any distance and scores
0.000 <!-- computed: agent_memory.keyed_prose_d1 --> on prose at
*distance one* — it never had them. The composite closes the keyed half
exactly and leaves prose recall a cliff, so nothing here short of keeping
everything has full recall at every distance. `truncated`'s 0.333 on keyed
facts is the accidental result of a 40-character prefix reaching the first
field of a JSON result and not the second or third, which is a good picture of
what truncation retains: whatever happened to be at the front.

**Compaction value peaks in the middle, and I predicted otherwise.** The value
of compacting once at step *k* is (tokens accumulated by then) × (steps
remaining to re-read them), a rising term times a falling one:

| compact at step | tokens saved |
|---:|---:|
| 3 | 901 |
| 5 | 1080 |
| 10 | 1360 |
| **11** | **1449** |
| 15 | 1110 |
| 18 | 498 |

I expected earlier to be strictly better, on the reasoning that a saving
multiplies by more remaining steps. That is true and it is only one of the two
factors: compacting at step 3 has almost nothing to compact. The peak is at
step 11 <!-- computed: agent_memory.compact_best_step --> of twenty —
0.55 <!-- computed: agent_memory.compact_best_frac --> of the way through —
worth 1.61× <!-- computed: agent_memory.compact_peak_over_early --> compacting
at step 3 and
2.91× <!-- computed: agent_memory.compact_peak_over_late --> compacting at step
18. The closed form is checked against a full re-run of both episodes at every
step, and they agree exactly.

The practical reading is not "compact at 55%" — that fraction depends on how
observation sizes are distributed through the episode — but that **the curve
has an interior maximum at all**, so both "compact as early as possible" and
"compact when you run out of room" are wrong for the same reason, from opposite
ends.

??? question "Why does the peak land near the middle rather than somewhere else?"
    Because both factors are close to linear in this episode: tokens
    accumulate at a roughly steady rate as steps go by, and the steps remaining
    fall at exactly one per step, so their product is close to a downward
    parabola whose maximum sits near the midpoint. An episode whose expensive
    observations all arrive at the end would push the peak later, and one that
    dumps a large document at step 2 would pull it much earlier — which is the
    reason to compute the curve for your own workload rather than adopt 0.55 as
    a number.

??? question "If compaction has a best moment, why not compact repeatedly at every good moment?"
    Because the value of the second pass is computed against what the first one
    already removed, so the rising factor has been reset while the falling one
    has not, and each additional pass captures less than the one before it
    while paying the same loss again. That is the argument against continuous
    compaction in §F, and it is the same shape as lesson 4.3's re-planning
    premium: the repeated version of a good idea pays its full cost every time
    and collects a diminishing share of the benefit.

## H · Failure modes and cost traps

**Reporting compression without recall.** Half of a trade, and always the
flattering half.

**Testing retention by searching all of context for the text.** Gives credit
for facts a repeated observation re-established, and inflates every windowed
policy. Tag by origin.

**Charging a re-fetch per observation instead of per fact.** Lets the policy
with the worst recall look cheapest — it did here, before the accounting was
fixed.

**A compaction that is lossy but not compact.** The `truncated` row: −4.7% net.
It pays to carry what it kept and pays again for what it lost.

**Assuming eviction by age is conservative.** It is the opposite in an agent
episode, where the oldest facts are the discovered dependencies everything
later rests on.

**Shipping the framework default.** The bare sliding window is the default
nearly everywhere and is the second-worst policy measured here.

**Compacting on a schedule.** Continuous compaction pays the loss repeatedly
while the value curve says one well-timed pass captures most of the saving.

## I · Graded practice

<quiz-bank src="agt-l4"></quiz-bank>

<code-exercise src="agt-l4-recall"></code-exercise>

<code-exercise src="agt-l4-timing"></code-exercise>

## J · Annotated references

- **Park et al., "Generative Agents" (2023)** — the memory stream with
  recency, importance and relevance scored together. Read the retrieval
  function and note that importance is the term this lesson cannot compute
  cheaply.
- **Packer et al., "MemGPT" (2023)** — paging between a bounded context and
  external storage, argued explicitly as an operating-systems problem. The
  closest thing to a systems treatment of §B.
- **Denning, "The Working Set Model for Program Behavior" (1968)** — where
  temporal locality as a justification for eviction-by-age comes from, and
  therefore the right place to see why §D's inversion matters.
- **Any framework's default memory class** — read what it evicts and when.
  Most are the bare window in §G's second-worst row.

## K · Extension

*Off-platform, an hour.* Take an agent transcript you have and pick three facts
its final answer depended on. For each, find the step that established it and
the step that used it, and write down the distance. Then check your memory
policy's window against those three distances. If any exceeds it, the agent
either re-fetched — look for the duplicate call — or answered without the fact,
which is the case with no evidence in the log and the reason to instrument the
miss.
