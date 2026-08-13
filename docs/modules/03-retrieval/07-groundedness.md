---
status: Verified
last_verified: 2026-08-13
volatility: low
pyodide: true
prereqs: ["3.6"]
---

# 3.7 · Groundedness and citation

## A · Why this matters

Retrieval put documents in front of a model. This lesson is about the only
question that follows: did the answer it produced actually come from them?

The framing that matters is which half of the problem you own. The generator is
usually someone else's model behind an API, and you cannot inspect it or change
how it attributes. What you own completely is the layer that reads the answer
and its citations and decides whether to ship it, and that layer needs no model
at all — four of the five checks below are set operations over metadata you
already have.

Measured on 55 <!-- computed: groundedness.n_answers --> answers with defects
planted at known positions, those checks catch
100% <!-- computed: groundedness.detection_rate_pct -->% of the defective answers
at a false-positive rate of
0.0 <!-- computed: groundedness.false_positive_rate_pct -->% on the grounded ones.
That combination is only interesting because both numbers are reported: a
checker that flags everything also detects everything, and the first question
to ask about any groundedness metric is what it does to answers that are fine.

The result worth carrying is the one about a failure none of the content checks
can see. An answer that quotes revision 1 of a policy accurately, and cites
revision 1, is internally consistent — the number it states really is in the
document it names. Nothing in the text distinguishes it from a correct answer.
Only a `superseded` flag stored beside the document does, and a system without
that field misses
5 <!-- computed: groundedness.stale_missed_without_provenance --> such answers
here while reporting itself as
88.6% <!-- computed: groundedness.detection_without_provenance_pct -->% effective.

!!! info "Terms used in this lesson"
    **Groundedness** — whether an answer's claims are supported by the
    documents it was given. Distinct from correctness: an answer can be true
    and ungrounded, which is worse than it sounds, because you have no way to
    tell it from a lucky guess.

    **Citation** — a reference from a span of an answer to a document id.

    **Attribution** — the general problem of establishing which source
    supports which claim. Citation is its cheapest observable form.

    **Context** — the documents actually passed to the generator. A citation
    outside it cannot be a source, whatever the document says.

    **Provenance** — metadata about a document rather than its content:
    version, effective date, supersession, origin. The only thing that catches
    a stale citation.

    **Abstention** — declining to answer. The corpus's
    unanswerable queries exist to measure it, and Capstone I scores it.

## B · Mental model

**Grade the citation before grading the claim, because the cheap checks are
also the strict ones.**

Verification questions form a ladder, and it is worth walking it in order of
cost because the early rungs are free and rule out whole classes of failure:

| | question | needs |
|---|---|---|
| 1 | Does the cited id exist? | the corpus index |
| 2 | Was it in the context? | the request log |
| 3 | Does every claim carry a citation? | a claim detector |
| 4 | Does the cited document support the claim? | the document text |
| 5 | Is the cited document the *right version*? | provenance metadata |

Rungs 1, 2 and 5 are set membership tests costing microseconds. Rung 4 is the
only one that reads document text, and rung 3 is where the difficulty is
hiding: "every claim" requires deciding what a claim is, which in general needs
a model. This lesson uses numeric statements as the proxy, because a sentence
containing a figure is unambiguously making a checkable assertion, and says
plainly that the general case is harder.

The ordering also encodes a hierarchy of badness. A fabricated citation is
loud and embarrassing and completely harmless once detected. A stale citation
is quiet, plausible, and gets acted on.

??? question "Why check that a citation was in the context, when you could just check whether the document supports the claim?"
    Because it is a stronger test and a cheaper one. A model that cites a
    document it was never shown did not read it, so any apparent support is
    coincidence — the claim happens to match text the model reproduced from
    training rather than from your corpus. Content checking would pass that
    answer, and the system would be attributing to a source it never
    consulted.

## C · Mechanism

The five checks, as implemented in `experiments/groundedness.py`.

**Citation resolves.** Every cited id appears in the corpus. Catches the
fabricated reference — a model producing something with the shape of a
citation rather than a real one.

**Citation in context.** Every cited id was among the documents passed in.
Requires logging the context per request, which is worth doing anyway and is
frequently not done.

**Every claim cited.** Every sentence containing a number carries at least one
citation. The numeric proxy is what makes this free; a general claim detector
is a model, with its own error rate, and at that point the checker has become
a thing needing evaluation of its own.

**Numbers supported.** Every number in a sentence appears in a document that
sentence cites. This is the one content check and the only one that catches an
answer whose citation resolves, was in context, and states the wrong figure —
the failure mode a reader cannot see. Note what it does *not* do: it verifies
that a number appears in the document, not that the document asserts it *of the
thing the query asked about*. A policy mentioning both 24 and 48 hours supports
either claim under this check.

**No superseded source.** No cited document is marked as replaced. Pure
provenance, and the only check that catches the stale citation.

The last two are worth comparing directly, because they fail in opposite
directions. The number check is content-based and catches a claim its own
citation contradicts. The provenance check is metadata-based and catches a
claim its citation *supports perfectly* — where the document is simply no
longer the governing one. No amount of reading the text can produce the second
judgement, which is the argument for storing version and supersession
alongside every document at ingestion time rather than hoping to reconstruct
them later.

**What the rungs cost, concretely.** Checks 1, 2 and 5 are set membership
tests against data already in memory, so they run in microseconds and can be
applied to every generation without a budget conversation. Check 4 reads the
text of each cited document, which means either holding the corpus in the
verification service or fetching a handful of documents per answer, and at
that point the check has an infrastructure dependency rather than merely a
cost. Check 3 is the one that looks cheapest and is not, because deciding what
counts as a claim in general requires a model, and a model-based checker
inherits an error rate that then has to be evaluated against human labels
before its output means anything.

That ordering suggests a deployment sequence rather than a menu. Ship the three
metadata checks first, since they catch
86.4% <!-- computed: groundedness.detection_metadata_only_pct -->% of these
defects with no dependencies at all; add the number check when you can reach
document text at verification time; and reach for a judge only for the residue
those two cannot see, which is where Course IV begins.

## D · From data science to LLM systems

The closest thing you have built is data validation — schema checks, range
checks, referential integrity — and the reasoning transfers directly. Cheap
assertions that run on everything catch the failures that would otherwise be
found by a user, and the reason they are worth writing is that they are
mechanical and complete rather than clever.

Three differences matter.

**The false-positive rate is the whole story, and it is not usually reported.**
A groundedness metric quoted as "94% of answers are grounded" without the rate
at which good answers are rejected is uninterpretable, in exactly the way a
classifier's accuracy is uninterpretable without its base rate. This lesson
reports both because one without the other is not a measurement.

**Some checks need data you must decide to keep.** Referential integrity needs
foreign keys; a stale-citation check needs a supersession field. If retrieval
returns text and nothing else, the entire provenance rung is unavailable, and
the loss is invisible — the system reports high groundedness because it cannot
ask the question that would lower it.

**The generator is not yours to fix.** In a data pipeline, a failing validation
gets traced back to the producer. Here the producer is a model you cannot
retrain, so the checker's output feeds a decision — ship, regenerate, abstain
— rather than a bug report. That changes what the checker is for: it is a
control, not a diagnostic.

??? question "Could you skip the citation-resolves check, since an id that does not exist will fail the content check anyway?"
    It would usually fail it, and you would learn the wrong thing, because the
    reported defect would be "the document does not support this claim" when
    the truth is "there is no such document". Each check answering exactly one
    question is what lets the table in §G localise a weakness, and collapsing
    two failures into one column costs precisely that. It is also the reason
    the number check skips unresolvable ids rather than treating them as
    unsupported.

## E · Minimal implementation

The number check, which is the one with any content in it:

```python
NUMBER = re.compile(r"\b\d+\b")

def numbers_supported(answer, docs):
    """Every figure in a sentence appears in a document that sentence cites."""
    for s in answer["sentences"]:
        nums = set(NUMBER.findall(s["text"]))
        # A sentence with no numbers makes no checkable numeric claim, and one
        # with no citations is the *other* check's problem. Conflating them
        # here would report a single failure for two distinct defects.
        if not nums or not s["cites"]:
            continue
        available = set()
        for c in s["cites"]:
            if c in docs:                       # unresolvable ids: check 1's job
                available |= set(NUMBER.findall(docs[c]["text"]))
        if nums - available:
            return False
    return True
```

Each check answering exactly one question is what makes the table in §G
readable: every defect is attributable to the rung that caught it, and a
change in one column has an obvious cause. A single `is_grounded()` returning
one boolean would score identically and tell you nothing about where a system
is weak.

The `\b` word boundaries are not decoration. Without them `24` matches inside
`2024`, so a document mentioning a year silently supports a claim about hours.

## F · Production practice

**Log the retrieved context with every generation.** Without it, check 2 is
impossible and check 4 is guesswork. This is the single highest-value thing to
add to a RAG system that does not have it.

**Store provenance at ingestion.** Version, effective date, supersession,
source system. Reconstructing it later is usually impossible, and its absence
costs silently: the checks that need it simply never run.

**Report the false-positive rate alongside the detection rate, always.**
Neither number means anything alone.

**Prefer the cheap rungs.** Metadata-only checking catches
86.4% <!-- computed: groundedness.detection_metadata_only_pct -->% of these
defects with no document text and no model. Reach for content checking, and
eventually for a judge, only for what the cheap rungs cannot see.

**Treat an uncited claim as a failure rather than a warning.** It is the
easiest defect for a generator to produce under pressure, and once tolerated it
is what every ungrounded answer looks like.

??? question "An answer states a figure in words — 'forty-eight hours' — rather than digits. What happens?"
    Every numeric check passes it silently, since there are no digits to
    extract and the sentence therefore makes no claim these checks can see.
    That is a genuine blind spot shared by the token comparison and the
    substring version alike, and the honest response is to state it rather
    than to imply the checker covers numeric claims in general. Whether it
    matters depends on your generator, which is worth measuring on real output
    rather than assuming.

## G · Experiment

`python experiments/groundedness.py`, over the authored answer set.

| defect | n | flagged | resolves | in context | claim cited | numbers | provenance |
|---|---:|---:|---:|---:|---:|---:|---:|
| grounded | 11 | **0.00** | — | — | — | — | — |
| fabricated_citation | 11 | 1.00 | 1.00 | 1.00 | — | 1.00 | — |
| unretrieved_citation | 11 | 1.00 | — | 1.00 | — | — | — |
| uncited_claim | 11 | 1.00 | — | — | 1.00 | — | — |
| wrong_number | 6 | 1.00 | — | — | — | 1.00 | — |
| stale_source | 5 | 1.00 | — | — | — | — | 1.00 |

**Every defect is caught, and by the rung that should catch it.** The clean
diagonal is the point: each check answers one question, so the table localises
a weakness rather than merely scoring one.

**Removing the content check costs one class.** Metadata-only detection is
86.4% <!-- computed: groundedness.detection_metadata_only_pct -->%, losing exactly
the 6 <!-- computed: groundedness.n_wrong_number --> `wrong_number` answers —
the ones whose citation is real and whose figure is not.

**Removing provenance costs a quieter one.** Detection without the supersession
field is
88.6% <!-- computed: groundedness.detection_without_provenance_pct -->%, and the
5 <!-- computed: groundedness.n_stale_source --> answers it loses are the ones
that quote a real document accurately. No content check can recover them,
because there is nothing wrong with the text.

**A defect this experiment found in its own inputs.** The first run reported a
false-positive rate of 0.083 — one "grounded" answer failing the number check.
The checker was right and the *answer builder* was wrong: it stated a policy's
`hold_hours` fact for a topic whose rules are written in days, so the figure
was in the document's metadata and not in its text. A fact recorded about a
document is not necessarily a fact stated in it, which is the same confusion
this lesson exists to catch, one level up. `tools/build_answers.py` now
verifies that every answer it labels grounded passes every check.

??? question "Why is a false-positive rate of zero here not as reassuring as it sounds?"
    Because the grounded answers were authored by the same generator that
    authored the defective ones, so they exercise a narrow and tidy slice of
    what real output looks like. A checker tested only against defects someone
    deliberately planted has not met the awkward middle — an answer that
    paraphrases a figure, cites two documents where one would do, or states
    something true that no single document supports. §K asks you to run these
    checks against real answers precisely because that is the population the
    false-positive rate has to hold on.

## H · Failure modes and cost traps

**Reporting groundedness without a false-positive rate.** "94% grounded" is
not a measurement. A checker that rejects everything scores 100% detection.

**Checking content without checking provenance.** The stale citation passes
every content test that exists, because its text genuinely says what the answer
claims. Measured here: five answers, invisible, in a system reporting 88.6%
effectiveness.

**Not logging the context.** Makes the strictest and cheapest check
impossible, and there is no way to reconstruct it after the fact.

**Regex without word boundaries.** `24` inside `2024` makes a document
mentioning a year support a claim about hours. One of the quieter ways a
checker reports success it has not earned.

**Treating "the number appears in the document" as support.** It establishes
that the figure is present, not that the document asserts it about the thing
being asked. A policy stating both 24 and 48 hours supports either claim under
this check, and closing that gap needs a claim-level judge with an error rate
of its own.

**Assuming a generator that cites is a generator that is grounded.** Citations
are cheap to produce and are exactly the surface a fluent model imitates well.
That is why rung 1 exists at all.

??? question "Should an uncited claim block the answer, or merely warn?"
    Block, because it is the easiest defect for a generator to produce under
    pressure and the one that degrades fastest once tolerated. A system that
    warns on uncited claims accumulates them until every ungrounded answer
    looks normal, whereas one that rejects them forces the generator's
    behaviour to stay within what the verification layer can check. That is a
    control decision rather than a measurement one, and it is worth taking
    deliberately rather than by default.

## I · Graded practice

<quiz-bank src="ret-l7"></quiz-bank>

<code-exercise src="ret-l7-numbers"></code-exercise>

<code-exercise src="ret-l7-provenance"></code-exercise>

## J · Annotated references

- **Rashkin et al. (2021), "Measuring Attribution in Natural Language
  Generation"** — the AIS framework, and the clearest definition of what
  "supported by" should mean before you try to automate it.
- **Bohnet et al. (2022), "Attributed Question Answering"** — a large study of
  automatic attribution metrics against human judgement, which is the
  calibration this lesson's cheap checks do not have.
- **Gao et al. (2023), "Enabling Large Language Models to Generate Text with
  Citations"** — ALCE, the benchmark, and a useful account of how often models
  cite documents that do not support the claim.
- **Es et al. (2023), "RAGAS"** — the widely used framework. Read §3 with §H in
  mind: its faithfulness metric uses a judge, so it inherits the judge's error
  rate, which is the thing Course IV spends a module on.

## K · Extension

*Off-platform, a day.* Take a hundred real answers from a system you run and
apply the five checks. Then hand-label the same hundred for groundedness and
compare, because the number this lesson cannot give you is how well mechanical
checks agree with a careful reader. Report the disagreements in both
directions: the answers the checks passed and you would not, which is the
interesting set, and the ones the checks flagged and you would ship, which
tells you what the checker costs to run in production.
