---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 0.1 · What changes when the model is someone else's

## A · Why this matters

Every habit you have about models rests on an assumption so basic that it is
rarely stated: that you fitted the model yourself. Because you fitted it, you
chose the training data and can therefore reason about what the model has and
has not seen; you can retrain when the data shifts; you can inspect the
coefficients when a prediction surprises you; and given the same input the
thing returns the same output today, tomorrow, and in eighteen months when
somebody asks you to reproduce a result.

You are about to give up all four of those in exchange for a capability you
could not realistically have built, and for most applications that is a trade
worth making. What is not worth making is the discovery of the terms one
production incident at a time, which is what happens when the differences are
treated as details rather than as the design constraints they are.

Five properties change together, and each one quietly invalidates a habit that
has served you well until now.

| Property | The habit it breaks | Where it bites |
|---|---|---|
| Stochastic by default | "Run it again and check" | A single confirming run is a sample, not a confirmation |
| Frozen — you cannot refit | "The vectorizer learned something odd; fix the preprocessing" | Adaptation happens in the context window or not at all |
| Versioned by someone else | "My baseline from March is still my baseline" | Comparisons silently span two different models |
| Metered, in both directions | "Inference cost is a rounding error" | Unit economics become a product decision |
| Opaque | "Look at the feature importances" | The only introspection available is behavioural |

The rest of this lesson takes each property in turn, says precisely what
mechanism produces it, and then — in §E and §F — describes the one artefact
that makes all five survivable, which is a properly designed record of every
call you make.

!!! info "Terms used in this lesson"
    **Decoding parameters** — the settings that turn the model's probability
    distribution over next tokens into a single chosen token: `temperature`,
    `top_p`, `top_k`, any repetition penalty, and a seed where one is offered.
    They are not part of the model; they configure a small piece of ordinary
    code that runs after it, and lesson 1.2 implements that code.

    **Refusal** — a successful call whose output declines to do what was
    asked. The infrastructure worked; the answer is simply not the one you
    wanted, which is why a refusal belongs in quality metrics and not in
    reliability metrics.

    **Cohort key** — the tuple of properties that two runs must share before
    their results may legitimately be compared: the resolved model version,
    the decoding parameters, and the rendered prompt.

## B · Mental model

The most useful way to hold all of this is to stop thinking of the model as
something you deploy and start thinking of it as **a dependency that sometimes
disagrees with itself**.

Everything you actually control sits around that dependency rather than inside
it: what you put in, what you do with what comes back, what happens when the
call fails, and what you write down so that a future version of you can
reconstruct what happened. The model itself is one line in the middle of that
arrangement, and it is the only line you cannot change, which means that all
of your engineering effort necessarily lands on the lines around it.

This shape is familiar from elsewhere, and the analogy is a good one because it
transfers a whole set of instincts intact. **You are integrating a third-party
API whose service level nobody else will measure for you.** No provider
publishes the accuracy of your task on your data, because no provider knows
what your task is; that measurement is yours to build, and Module 0's other
three lessons are largely about building it well enough to trust.

There is one respect in which the API analogy undersells the problem. An
ordinary API returns the same answer for the same request, so when it starts
returning something different you know that something changed. Here the answer
varies between two identical requests as a matter of routine, which means the
signal you would normally use to detect change is buried inside noise that was
always there. Distinguishing the two is a statistical problem rather than an
engineering one, and it is why lesson 0.3 exists.

??? question "Which of these can you still do once the model is a rented service: retrain, refit preprocessing, change decoding parameters, inspect coefficients, version-pin?"
    Only two survive: you can change the decoding parameters, and you can pin
    a version where the provider exposes one. Retraining and refitting are
    gone entirely, because you have neither the weights nor the corpus.
    "Inspect coefficients" is replaced by behavioural introspection, in which
    you learn what the model does by giving it inputs and observing outputs —
    exactly the position an experimentalist occupies with respect to a natural
    system, and the reason that careful experimental design suddenly matters
    more than it did.

## C · Mechanism

**Stochastic by default.** At each position the model produces a score for
every token in its vocabulary, and a sampler converts those scores into a
single choice. Because that conversion involves a random draw whenever the
temperature is above zero, two identical requests take two different paths
through the same distribution.

Disabling the sampling does not buy you determinism either, for a reason that
has nothing to do with the model. The arithmetic runs in floating point on
hardware that batches many requests together, and floating-point addition is
not associative, so summing the same numbers in a different order can give a
slightly different result. A request batched alongside different neighbours can
therefore produce logits that differ in their last bits, and occasionally that
difference is enough to change which token comes out on top. Lesson 1.2 takes
the sampler apart properly and measures how often the top choice actually wins.

**Frozen.** The weights and the tokenizer were fixed before you arrived, using
a corpus you cannot inspect, and there is no `fit` method anywhere in the
interface. The only adaptation available at this level is choosing what to put
in the context window, which is why Modules 2, 3 and 4 — prompting, retrieval
and agents — are collectively the discipline that replaces feature engineering.

**Versioned by someone else.** A provider can change what sits behind a name
without changing the name, so a baseline you measured in March may not be
reproducible in August even though every line of your code is identical. This
single fact is why the record in §E stamps a *resolved* version rather than the
alias you requested, and it is why the sentence "we measured 82%" is
incomplete in a way that "we measured 82% on version X in March" is not.

**Metered, in both directions.** You pay for the tokens you send and for the
tokens that come back, at different rates, on every request forever. Cost
therefore scales with usage instead of sitting in a fixed training bill that
somebody approved once, and the practical consequence is that the person
choosing the design is now also the person choosing the unit economics.
Lesson 0.4 does that arithmetic.

**Opaque.** There are no coefficients to read, no feature importances to rank,
and no partial dependence plots to draw. What remains is behavioural probing:
perturb the input, observe the output, and be disciplined about the statistics,
because with a stochastic system and a small sample the temptation to
over-read a single observation is enormous.

## D · From data science to LLM systems

The mapping from your existing practice is close enough to be genuinely
dangerous, in the sense that most of it works and the parts that do not fail
quietly rather than loudly.

| You had | You now have | Where the analogy breaks |
|---|---|---|
| `model.fit(X, y)` | nothing | There is no fit. Adaptation happens in the context window, or not at all |
| `model.predict(X)` | a metered network call that may fail | Prediction can now time out, cost money, or refuse |
| A pinned model artefact | a name that maps to a model *today* | Your artefact is not immutable unless you pinned a version and recorded it |
| Deterministic scoring | a distribution over outputs | A single run is a sample, not a measurement |
| Feature importances | behavioural probing | You can only learn from the outside |
| Training cost, paid once | inference cost, paid per request | Unit economics become a product decision |

The habit that transfers best is one you may not think of as a habit at all:
**you already refuse to trust a number without knowing how many observations
it rests on.** That instinct is worth more here than it was in your previous
work, because evaluation items are expensive enough that sample sizes are
small, and because the field around you frequently reports single runs as
findings. Lesson 0.3 is entirely about pointing this instinct at the right
target.

The habit that transfers worst is "run it again and see". In a deterministic
pipeline, re-running a failing case after a change is a legitimate check,
because a change in behaviour can only have come from the change you made.
Here, re-running gives you a second draw from a distribution whose mean you do
not know, and if the item succeeded thirty per cent of the time before your
change then a single post-change success was always reasonably likely.

??? question "Your colleague says a prompt change 'clearly fixed it' because they re-ran the failing example and it worked. What is the sample size of that claim?"
    One, against a system whose per-item behaviour is a biased coin with
    unknown bias. If the item previously succeeded thirty per cent of the
    time, then a single success after the change had probability 0.3 of
    occurring with no improvement whatsoever, which means the observation is
    almost uninformative. This is the most common false conclusion in the
    field and it costs nothing to avoid: run the item twenty times before and
    twenty times after, and compare the rates rather than the anecdotes.

??? question "Why does 'the model is frozen' make prompt construction more important than it looks?"
    Because it is the only remaining channel through which task-specific
    information can reach the model. In your previous work, information about
    the task entered through the training data and the features; here the
    weights are fixed, so every fact the model needs about *your* problem has
    to arrive in the context window on every single request. That reframes
    prompting from a matter of phrasing into a question of what information to
    select and how to fit it in a budget, which is what Modules 2 and 3
    actually teach.

## E · Minimal implementation

The smallest thing worth building on the first day is not a wrapper around the
API, because a wrapper can be added at any time and every SDK ships one. It is
the record you keep of each call, because without it none of the analysis in
the rest of this module is possible, and because a record cannot be
reconstructed after the fact for calls you have already made.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RunRecord:
    request_id: str
    model_version: str    # the resolved version, not the alias you asked for
    params: dict          # temperature, top_p, max_tokens, seed if offered
    prompt_hash: str      # of the fully rendered prompt, not the template
    in_tokens: int
    out_tokens: int
    latency_ms: float
    ttft_ms: float | None
    outcome: str          # "ok" | "refused" | "timeout" | "invalid" | "error"
```

Each field earns its place by making a specific later question answerable. The
resolved version is what lets you tell whether two results are comparable at
all; the parameters and the prompt hash complete that picture; the token counts
are the only honest input to a cost model; splitting latency into
time-to-first-token and total separates queueing problems from generation-length
problems, which have different fixes; and a categorised outcome keeps refusals
out of your reliability metrics, where they would be mistaken for infrastructure
failures.

The single function that turns those records into something usable is short:

```python
def cohort_key(r: RunRecord) -> tuple:
    """Two runs may only be compared if all of this matches."""
    return (r.model_version, params_signature(r.params), r.prompt_hash)
```

Two results carrying different cohort keys are not two measurements of one
thing; they are one measurement each, of two different things, and averaging
them produces a number that describes neither. The first exercise below
implements this and the diagnostic that goes with it.

## F · Production practice

Log every call from the first day, and log these fields in particular.

**The resolved model version**, not the alias you requested, because an alias
is a promise about capability rather than about identity. If the provider does
not return a resolved version, record the wall-clock time instead and treat the
version boundary as unknown rather than as absent.

**Every decoding parameter, including the defaults you did not set.** A default
you never chose is still a parameter you depended on, and defaults change
between library versions without anybody announcing it; recording the effective
values costs nothing and converts an unanswerable question into a lookup.

**A hash of the fully rendered prompt**, rather than the template and its
variables. Rendering logic is code, code changes, and the combination of an old
template with new rendering does not reconstruct what was actually sent.

**Token counts taken from the provider's response**, not estimated locally.
Lesson 1.1's extension explains why a local estimate and a billed count differ
by a stable amount, and lesson 1.3 explains where that difference comes from.

**Latency split into time-to-first-token and total**, because those two numbers
have different causes and different remedies, and a single figure conceals
which one you are looking at.

**The outcome, categorised.** "It failed" is not a category. A timeout, a
refusal, a schema violation and a server error call for four different
responses, and collapsing them into one field guarantees that the most
interesting of the four never gets investigated.

None of this requires a vendor or a platform. A JSONL file with one record per
line will carry you a surprisingly long way, and the discipline of writing the
record is worth more than the sophistication of where it is stored.

## G · Experiment

Retries look like a pure reliability lever: they cost a little latency and they
convert some failures into successes. Their arithmetic is short enough to do by
hand, and the result is not what most people expect.

```bash
python experiments/service_economics.py
```

Write `f` for the probability that a single attempt fails and `K` for the
maximum number of attempts. Because a request only reaches attempt *k* if the
preceding *k−1* attempts all failed, the expected number of attempts spent on a
request is a geometric sum, and the probability that the request eventually
succeeds is one minus the probability that every attempt failed:

$$
\text{attempts per request} = \sum_{k=1}^{K} f^{\,k-1} = \frac{1 - f^K}{1 - f},
\qquad
\text{success rate} = 1 - f^K
$$

Dividing the first expression by the second gives the quantity that actually
appears on an invoice, which is the number of attempts spent per *successful*
request — and the two factors of $1 - f^K$ cancel, leaving `K` nowhere in the
result:

$$
\text{attempts per success} = \frac{1 - f^K}{(1 - f)\,(1 - f^K)} = \frac{1}{1 - f}
$$

Measured, at a twenty per cent failure rate, raising the retry limit from three
attempts to ten:

| | limit 3 | limit 10 |
|---|---|---|
| Attempts per success | 1.25 <!-- computed: service_economics.apc_f20_k3 --> | 1.25 <!-- computed: service_economics.apc_f20_k10 --> |
| Requests that eventually succeed | 99.2% <!-- computed: service_economics.success_rate_f20_pct --> | 100.0% <!-- computed: service_economics.success_rate_f20_k10_pct --> |

**Raising the retry limit does not change what a successful request costs
you.** It buys a higher success rate, and it buys a worse tail latency, and it
changes the cost per success by exactly nothing. The lever that reduces spend
is therefore the failure rate itself rather than the retry policy, which is a
useful thing to know before a meeting in which somebody proposes reducing the
retry limit to save money.

At more ordinary failure rates the overhead is small in absolute terms —
2.0% <!-- computed: service_economics.cost_overhead_f2_pct --> at a two per cent
failure rate and 8.7% <!-- computed: service_economics.cost_overhead_f8_pct -->
at eight per cent — but it is worth noticing that this overhead is invisible in
any accounting that counts only successful calls.

??? question "Given that, when *is* raising the retry limit the right move?"
    When the failures are independent and transient, and when the additional
    tail latency is acceptable to whoever is waiting. Under those conditions a
    higher limit converts a visible failure into a slow success at no extra
    cost per success, which is an unusually good trade. It is the wrong move
    when failures are correlated — a malformed prompt, a schema the model
    cannot satisfy, an overloaded provider — because then each retry
    faithfully reproduces the same failure while you pay for every attempt and
    the failure rate stays exactly where it was.

## H · Failure modes and cost traps

**Comparing across versions without noticing.** Two numbers measured six weeks
apart, with the provider having updated the model in between, presented as an
improvement. This is the most common invalid comparison in the field, and the
reason it survives is that nothing about it looks wrong: the code is unchanged,
the harness is unchanged, and the only thing that moved is invisible unless you
recorded it.

**Trusting a parameter you never set.** A default temperature is still a
temperature, and when a library changes its default your outputs change while
your code does not. Recording the effective parameters turns this from a
mystery into a diff.

**Letting a cache hide the variance.** A response cache makes a stochastic
system look deterministic throughout development, because every repeated
request during your testing returns the stored answer. The true behaviour then
appears in production, with cold keys and real traffic, at the least convenient
possible moment.

**Treating a refusal as an error.** A refusal is a successful call with an
output you did not want. Filing it under reliability hides a quality problem
inside an infrastructure metric, where the people who would recognise it will
never look for it.

**Estimating tokens locally and billing on the estimate.** The estimate is
wrong by the size of the chat template, consistently, on every request. Lesson
1.3 measures the gap.

**"Run it again and see."** Covered in §D, and repeated here because everybody
does it, including people who can explain exactly why it does not work.

??? question "You log the model alias you requested rather than the version the provider resolved. Six months later, what can you no longer answer?"
    Whether any two results in the log came from the same model. Because the
    alias is a constant string throughout the file, the log looks perfectly
    consistent, and that appearance of consistency is worse than an obvious
    gap would be — nothing about it prompts you to doubt the comparison you
    are about to make. If the provider does not return a resolved version, the
    timestamp is the only handle you have, which is precisely why §E records
    it.

## I · Graded practice

<code-exercise src="tr-l1-cohorts"></code-exercise>

<code-exercise src="tr-l1-retries"></code-exercise>

<quiz-bank src="tr-l1"></quiz-bank>

## J · Annotated references

- **Google SRE Workbook, the chapters on SLOs and error budgets.** Written
  about ordinary services, and an LLM feature is an ordinary service in every
  respect that matters here. The error-budget framing is the cleanest available
  way to decide how much reliability is worth buying.
- **Hyrum's Law.** A single sentence, and it explains why a provider changing a
  default breaks somebody: with a sufficient number of users, every observable
  behaviour of a system is depended upon by someone.
- **Any good treatment of idempotency keys in payment APIs.** The problem of
  retrying an expensive, non-idempotent operation safely was solved a decade
  earlier and solved well; lesson 1.6 borrows the solution directly.

## K · Extension

**Write the record before you write the wrapper.** Take whatever LLM-adjacent
code you already have — or a ten-line script that calls anything at all — and
add the `RunRecord` from §E to it, appending one JSON object per line to a
file. This takes about twenty minutes and it is the only step in this module
that cannot be done retroactively.

Then answer three questions using nothing but the log. How many distinct model
versions does it contain? What fraction of your calls used a parameter that you
never explicitly set? And can you find two runs that you would have compared
without hesitation, but which turn out to have different cohort keys? Most logs
answer "more than one", "most of them", and "yes" respectively, which is the
point of the exercise.
