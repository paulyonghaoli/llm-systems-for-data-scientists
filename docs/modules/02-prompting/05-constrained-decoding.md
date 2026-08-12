---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 2.5 · Constrained decoding: the logit mask

## A · Why this matters

[Lesson 2.4](04-structured-output.md) spent its length on repairing malformed
output, and ended by recommending that you avoid needing to. This lesson is
that recommendation's mechanism.

Constrained decoding replaces "ask for JSON and fix what comes back" with
"make invalid JSON unrepresentable". Before each token is sampled, a grammar
declares which tokens could legally come next; everything else has its logit
set to negative infinity, so the softmax assigns it exactly zero probability
and the sampler cannot choose it. Validity stops being a rate you measure and
becomes a property of the sampler.

That is worth the whole of 2.4's repair ladder, and it comes with a
consequence nobody mentions. Measured on the sixty-two-character record from
the previous lesson, the grammar allows exactly one character at
52 <!-- computed: constrained_decoding.forced_positions --> of
62 <!-- computed: constrained_decoding.document_chars --> positions —
83.9% <!-- computed: constrained_decoding.forced_pct --> of the document. **The
model chooses ten characters; the grammar writes the rest.**

!!! info "Terms used in this lesson"
    **Logit mask** — a vector added to the logits before the softmax, holding
    zero for permitted tokens and negative infinity for forbidden ones.
    Forbidden tokens then receive exactly zero probability.

    **Grammar** — a formal description of the permitted language. Here a
    schema, expressed as an automaton that can say which characters may follow
    any given prefix.

    **Prefix viability** — whether a partial output can still be extended into
    a valid document. The mask is precisely the set of characters that keep it
    viable.

    **Forced position** — a position where the grammar permits exactly one
    continuation, so the model's distribution is irrelevant.

## B · Mental model

**The grammar and the model each get a vote, and the grammar's is a veto.**

The model proposes a distribution over the whole vocabulary. The mask deletes
everything illegal. Whatever survives is renormalised and sampled from. So the
model can only ever choose among continuations the grammar already approves,
and the two failure modes follow directly from that arrangement.

If the grammar permits exactly one continuation, the model's opinion does not
enter into it — which is fine when the continuation is a closing brace and
worth thinking about when it is not. And if the grammar permits several but the
model's probability mass sits almost entirely on tokens that were masked away,
you are sampling from a renormalised tail: the output is valid, and it is not
what the model would have said.

This is the honest cost of the technique, and it is easy to overlook because
the visible failure mode disappears completely. **Constrained decoding
guarantees shape and says nothing whatever about content.**

??? question "A masked decoder returns a perfectly valid record with a plausible but wrong `record_id`. Which layer of 2.4's stack caught it?"
    None of them, and none of them could. Parsing and schema validation both
    pass by construction under a mask, so the entire first two layers of
    lesson 2.4 have become vacuous — they now assert something the sampler
    already guaranteed. The third layer, correctness, is untouched and is the
    only one left doing work. Constraining the output moves *all* of your
    remaining risk into the layer that needs an evaluation.

## C · Mechanism

**The mask itself is two lines.** Given the set of permitted token ids:

```python
mask = np.full(vocab_size, -np.inf)
mask[list(permitted)] = 0.0
probs = softmax(logits + mask)
```

Adding negative infinity before the softmax sends `exp` to zero exactly, so
forbidden tokens receive no probability at all rather than a very small
amount. Doing it after the softmax — zeroing entries and renormalising — is
arithmetically equivalent and numerically worse, because the forbidden mass has
already been computed and subtracted from everything else.

**Computing the permitted set is the whole problem.** For a fixed schema this
is a state machine: consume the prefix, work out where in the document you are,
and return the alphabet available at that position. After `{` only `"` is
legal. Inside a hex field, sixteen characters are. After `"status": "`, exactly
three are — one per permitted value, which means **an enum is enforced during
generation rather than detected after it.**

**Prefix viability is the general formulation.** A character is permitted when
the prefix plus that character can still be extended into some valid document.
Real implementations compile the grammar into an automaton and track its state
incrementally, because recomputing viability from scratch at every position is
the difference between a usable decoder and a very slow one.

**Token-level masking is harder than it looks.** A BPE token may span several
characters, so a token is permitted only if *every* character it contributes
keeps the prefix viable — and a single token can straddle the boundary between
a field and the literal that follows it. This is where real implementations
spend their complexity, and it is why the experiment below works per character
and says so.

**The grammar has to be right, and nothing else will tell you if it is not.**
A validator can be wrong and merely fail to catch things; a *grammar* that is
wrong actively produces the wrong output, because it is no longer describing
the language you meant. If it forbids a character that a valid document
requires, generation walks into a dead end and either stops early or is forced
down a path that satisfies the grammar you wrote rather than the schema you
intended. The failure is confident, well-formed and entirely convincing, which
is why the automaton behind §G is checked against nine hand-written cases —
including three prefixes that have already left the language — before a single
number is quoted from it.

??? question "A validator with a bug usually fails safe. Why does a grammar with a bug fail dangerously?"
    Because their roles are opposite. A validator observes output that already
    exists, so a bug makes it miss problems — the output is no worse than it
    would have been without the check. A grammar *determines* output, so a bug
    changes what gets generated: forbid a character the schema requires and the
    model cannot produce the correct document at all, but it will produce
    something, and that something satisfies your grammar perfectly. You have
    replaced a detectable error with an undetectable one.

## D · From data science to LLM systems

The nearest thing in your existing practice is constrained optimisation, and
the analogy is exact in the part that matters.

| You know | Here |
|---|---|
| Feasible region in an optimisation | The set of grammatically legal continuations |
| Projecting onto a constraint set | Masking the logits |
| A constraint that binds at the optimum | A forced position, where the model has no choice |
| Rejection sampling | Repair-and-retry, which the mask replaces |
| Validation after fitting | Schema validation, made vacuous by the mask |

Rejection sampling is the row worth dwelling on. Lesson 2.4's repair ladder is
rejection sampling with extra steps: generate, test, discard or fix, repeat.
Constrained decoding is the standard improvement on rejection sampling — build
the constraint into the sampler so that every draw is feasible by
construction — and it brings the standard caveat with it. **A feasible draw is
not an optimal one**, and forcing feasibility can push you somewhere the
unconstrained distribution had almost no mass.

The habit that transfers badly is the assumption that a satisfied constraint is
evidence of a good solution. In an optimisation you still check the objective;
here the objective is correctness, nothing in the mask addresses it, and the
disappearance of the visible failures makes it easy to stop looking.

??? question "Your parse-failure rate drops to zero the day you enable constrained decoding. What should you check next, and why is nobody going to ask you to?"
    The accuracy of the extracted values, because the mask has removed your
    only visible symptom while leaving the underlying question untouched.
    Nobody will ask because every dashboard just turned green: the parse-error
    graph is flat at zero, the schema-violation graph is flat at zero, and both
    are now measuring a property of your sampler rather than of your system.
    That is a genuine improvement and it is also exactly when a quality
    regression becomes invisible.

??? question "You constrain generation to a schema and quality drops. Give two mechanisms that would explain it."
    First, the mask is renormalising a tail: at some position the model's mass
    sat almost entirely on tokens the grammar forbade, so the sampler drew from
    what little remained, and one poor choice early in a document propagates
    through everything conditioned on it. Second, the grammar may be forcing a
    field order or a format the model finds unnatural — models are trained on
    text and a schema that reads awkwardly is harder for them, which is a real
    effect and not a mystical one. Both are consistent with valid output and
    neither shows up in any validity metric.

## E · Minimal implementation

The grammar as an automaton over a fixed schema, and the mask that follows from
it:

```python
def legal_next(prefix):
    """Every character that could legally follow `prefix`."""
    # Walk the grammar, consuming prefix, and return the alphabet available
    # at the position it lands on. An empty set means the document is done.
    ...


def masked_probs(logits, prefix, vocab):
    permitted = legal_next(prefix)
    mask = np.full(len(vocab), -np.inf)
    for i, ch in enumerate(vocab):
        if ch in permitted:
            mask[i] = 0.0
    return softmax(logits + mask)
```

An empty permitted set is worth handling deliberately rather than letting it
produce a vector of `nan`. It means the document is complete, and the right
response is to stop generating — not to sample from nothing. A decoder that
treats "no legal continuation" as an error rather than as termination will
report failures on every successfully completed document.

The automaton in the experiment is checked against nine hand-written cases
before any number is quoted from it, because a grammar that is subtly wrong
produces confidently invalid output and the mask makes it look authoritative.

## F · Production practice

Use the provider's structured-output mode where one exists. It is the same
mechanism, implemented against the actual token vocabulary by people who have
handled the multi-character-token problem, and hand-rolling it against an API
that only returns sampled text is not possible anyway — masking requires access
to the logits, which most hosted endpoints do not expose.

Keep the schema in one place and generate both the constraint and the validator
from it. Two hand-written descriptions of the same contract will diverge, and
the failure is silent in the direction that matters: a validator laxer than the
grammar never fires, so you will not learn it has drifted.

Retain the validator even under a mask. It costs microseconds, it is the only
thing that catches a bug in your own grammar, and a mask built from the wrong
schema is precisely the failure that produces confident nonsense.

And measure quality after enabling it, not just validity. §D's self-check is
the whole argument: the visible failure mode disappears on the day you switch
it on, and the invisible one does not.

## G · Experiment

```bash
python experiments/constrained_decoding.py
```

A character-level grammar for lesson 2.4's record, over a
98 <!-- computed: constrained_decoding.vocab_size -->-character printable
vocabulary. The automaton is verified against nine known cases — empty prefix,
mid-field, enum disambiguation, completed document, and prefixes that have
already left the language — before anything is measured from it.

| | |
|---|---|
| Document length | 62 <!-- computed: constrained_decoding.document_chars --> characters |
| Positions with exactly one legal character | 52 <!-- computed: constrained_decoding.forced_positions --> (83.9% <!-- computed: constrained_decoding.forced_pct -->) |
| Positions where the model has a choice | 10 <!-- computed: constrained_decoding.free_positions --> |
| Mean legal characters per position | 3.0 <!-- computed: constrained_decoding.mean_options --> (3.0% <!-- computed: constrained_decoding.mean_allowed_pct --> of the vocabulary) |
| …when the model does have a choice | 13.1 <!-- computed: constrained_decoding.mean_options_when_free --> |
| Widest choice anywhere | 16 <!-- computed: constrained_decoding.max_options --> (16.3% <!-- computed: constrained_decoding.max_options_pct -->) |

**The grammar writes most of the document.** Eighty-four per cent of positions
admit exactly one character, which means the model is not being guided at those
positions so much as bypassed. That is the correct behaviour — nobody wants a
model exercising judgement about whether a closing brace belongs — and it
reframes what the model is contributing. Its entire contribution to a
structured output is the field values, which is exactly the part no
mask can check.

**Even where the model has a say, the choice is narrow.** At the ten free
positions it averages
13.1 <!-- computed: constrained_decoding.mean_options_when_free --> legal
characters, and the widest choice anywhere in the document is
16 <!-- computed: constrained_decoding.max_options -->, which is
16.3% <!-- computed: constrained_decoding.max_options_pct --> of the
vocabulary. Across the document the mean is
3.0% <!-- computed: constrained_decoding.mean_allowed_pct -->. A mask this
tight is doing something much stronger than nudging.

**The enum is the case that repays the effort most.** After `"status": "` the
grammar permits
3 <!-- computed: constrained_decoding.status_first_char_options --> characters,
one per permitted value, so an invalid status is not merely detected later — it
cannot be generated. Lesson 2.4 could only catch that after the fact, and only
if somebody remembered to check the enum.

**Validity is 100% <!-- computed: constrained_decoding.validity_pct -->, and
that is not a measurement.** It is a property of the construction, in the same
way that lesson 0.5's structural policy scored zero breakouts by construction.
Reporting it as though it were an experimental result would be the same
category error, so it is stated as what it is.

??? question "Eighty-four per cent of positions are forced. Does that mean the model is doing only sixteen per cent of the work?"
    No, and the framing is worth resisting. The forced positions are the
    punctuation and the key names — the parts that carry no information about
    *this particular record*. All of the information content sits in the ten
    free positions, so the model is doing approximately all of the work that
    matters and none of the work that does not. What the number really shows
    is how much of a structured output is scaffolding, and therefore how
    little of it a validity check was ever examining.

One more thing the numbers make concrete. The mean legal set across the
document is 3.0 <!-- computed: constrained_decoding.mean_options --> characters
out of 98 <!-- computed: constrained_decoding.vocab_size -->, so on average the
mask deletes about ninety-seven per cent of the vocabulary before the model is
consulted. Reading that alongside the forced-position count gives the honest
summary of what constrained decoding is: not a hint, not a preference, but a
narrow channel through which the model is permitted to express exactly the
information the schema has room for. Whether that information is *correct* is
the question the entire mechanism leaves untouched.

## H · Failure modes and cost traps

**Believing validity now implies correctness.** The mask guarantees shape.
Every remaining risk has moved into the content, where only an evaluation
reaches it, and the dashboards that used to show a problem now show zero by
construction.

**A grammar that does not match the validator.** Two hand-written descriptions
of one contract diverge, and a validator laxer than the grammar never fires, so
the drift is undetectable. Generate both from one schema.

**Masking after the softmax.** Zeroing probabilities and renormalising is
arithmetically equivalent and numerically worse; the forbidden mass has already
been computed and subtracted from everything else. Add `-inf` to the logits.

**Treating an empty permitted set as an error.** It means the document is
finished. A decoder that raises there reports a failure on every successful
completion.

**Assuming the technique is available.** Masking needs access to the logits.
Most hosted endpoints do not expose them, so in practice you are using the
provider's structured-output mode or you are not doing this at all.

**Forgetting that tokens span characters.** A BPE token is permitted only if
every character it contributes keeps the prefix viable, and tokens straddling a
field boundary are where real implementations get this wrong.

**Dropping the validator once the mask is in place.** It is microseconds, and
it is the only thing standing between a bug in your grammar and confident
nonsense.

## I · Graded practice

<code-exercise src="prm-l5-mask"></code-exercise>

<code-exercise src="prm-l5-grammar"></code-exercise>

<quiz-bank src="prm-l5"></quiz-bank>

This lesson closes Module 2's teaching content. The module's graded artifact —
an extraction harness scored on schema conformance, repair rate and cost — is
still to be built, and will put 2.4's repair ladder and this lesson's validator
into one pipeline.

## J · Annotated references

- **Willard & Louf (2023), *Efficient Guided Generation for Large Language
  Models*.** The paper behind `outlines`, and the clearest account of compiling
  a grammar into an automaton so that masking costs a state lookup rather than
  a re-parse.
- **The `llama.cpp` GBNF grammar documentation.** A working, readable grammar
  format you can experiment with locally, and the fastest way to see the
  token-versus-character problem in practice.
- **Any treatment of rejection sampling versus constrained sampling.** The
  statistical framing in §D, worked out properly, including why a feasible
  draw is not a representative one.

## K · Extension

**Write the grammar for a schema you actually use**, on paper, and count the
forced positions. The ratio is usually higher than people expect, and it tells
you how much of your output the model was never really choosing — which in turn
tells you how much a validity metric was ever measuring.

**Then check the thing the mask hides.** If you have structured output enabled
anywhere, take fifty outputs and check the *values* against ground truth rather
than against the schema. That number is the one the mask left untouched, it is
the only one still capable of moving, and it is almost certainly not on a
dashboard anywhere.
