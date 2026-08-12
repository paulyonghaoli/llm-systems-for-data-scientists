---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 0.2 · Anatomy of an LLM application

## A · Why this matters

The model call is one line of code. Everything around that line is yours to
build, and essentially every failure you will investigate originates in one of
those surrounding lines rather than in the call itself.

This is the part of the work that surprises people arriving from modelling,
because the ratio is so lopsided. You will spend far more time on context
assembly, output validation and graceful degradation than on anything that
resembles modelling, and the reason is not that the model is simple. It is that
the model is the single component you cannot debug, cannot patch and cannot
improve, so all of your engineering effort is displaced onto the components
that surround it.

There is a second reason the surrounding code deserves this much attention,
and it is quantitative rather than temperamental. The stages run in series, so
a request has to survive all of them, which means the end-to-end success rate
is the *product* of the individual stage rates rather than something close to
the worst of them. That product is considerably less forgiving than intuition
suggests, and §G measures exactly how much.

!!! info "Terms used in this lesson"
    **Stage** — one step in the pipeline that a request passes through, such
    as context assembly or output validation. Each has its own failure rate,
    and a request must survive every one of them.

    **Serial reliability** — the end-to-end success rate of a chain of
    independent stages, equal to the product of their individual rates. Adding
    a stage always lowers it.

    **Fallback** — what the user receives when the primary path did not
    produce a usable answer. A fallback is *graceful* if the caller can tell
    it happened and *silent* if they cannot.

    **Retry amplification** — the failure mode in which a struggling service
    receives more traffic because its clients are retrying, which converts a
    partial outage into a total one.

## B · Mental model

**Seven stages, in series, each of which can fail on its own terms.**

```
input → context assembly → prompt construction → the call →
        output validation → fallback → observability
```

The word doing the work in that diagram is *series*. Because a request must
pass every stage to succeed, the probabilities multiply, and multiplication
punishes long chains far more aggressively than addition would:

| Per stage | End to end (7 stages) |
|---|---|
| 99% | 93.2% <!-- computed: service_economics.end_to_end_990_pct --> |
| 99.5% | 96.6% <!-- computed: service_economics.end_to_end_995_pct --> |
| 99.9% | 99.3% <!-- computed: service_economics.end_to_end_999_pct --> |

To reach a ninety-nine per cent end-to-end success rate across seven stages,
each individual stage has to manage
99.857% <!-- computed: service_economics.per_stage_needed_for_99_pct -->, which
means that "three nines each" is not a comfortable engineering margin here but
roughly the minimum that gets you to a respectable headline number.

Two consequences follow, and both are counterintuitive enough to be worth
stating explicitly. Improving an *average* stage barely moves the total, since
you are replacing one factor slightly closer to one. Removing a stage
altogether removes a factor entirely, which is almost always the largest
available win and almost never appears on the list of options anybody
considers.

??? question "Your pipeline is at 93% end to end and you have time to make exactly one stage perfect. How much does that buy?"
    If all seven stages sit at 99%, making one of them flawless takes the
    product from 0.99⁷ to 0.99⁶, or roughly 93.2% to 94.1% — a single
    percentage point for a stage that now never fails. Serial reliability
    rewards lifting the *worst* stage rather than perfecting an average one,
    and it rewards deleting a stage most of all, which is why "do we need this
    step?" is a more valuable question than "how do we make this step better?"

## C · Mechanism

**1 · Input handling.** Whatever arrives from the user or the calling service,
together with the decisions about length limits, encoding and what you refuse
to process at all. This stage fails by accepting an input so large that nothing
else can fit alongside it in the context window, which converts a validation
problem into a truncation problem three stages later.

**2 · Context assembly.** Retrieval results, conversation history and tool
output — everything that enters the window besides your instructions. It fails
in two opposite directions: by retrieving nothing and proceeding anyway, which
produces a confident answer from the model's memory, or by retrieving so much
that the instructions are pushed out of the window entirely. Modules 3 and 4
are about doing this stage well.

**3 · Prompt construction.** Rendering the template, ordering the pieces and
budgeting the tokens. Its characteristic failure is silent truncation, and
mini-project 1 is devoted entirely to getting this stage right.

**4 · The call.** Timeout, bounded retries and a budget guard. It fails by
having no timeout at all, by retrying without bound, or by retrying an
operation that was not idempotent and therefore performing its side effect
twice.

??? question "Which of the seven stages can fail without raising an exception?"
    All of them, which is the single most important structural fact in this
    lesson. Truncation silently drops text; retrieval silently returns an
    empty list; the call silently returns a refusal with a 200 status;
    validation silently accepts a parseable object carrying the wrong fields;
    and fallback silently substitutes a worse answer for a better one. Only
    the timeout reliably throws. A pipeline built on the assumption that
    errors announce themselves will therefore look healthy for as long as it
    is failing.

**5 · Output validation.** Parsing what came back, checking it against a
schema, and deciding whether it is usable. This stage fails by accepting
anything that parses, and the distinction matters because a syntactically
valid JSON object containing a hallucinated field is not a success in any sense
your users would recognise.

**6 · Fallback.** What the user gets when stages one through five did not
produce an answer. It fails by not existing, and — more insidiously — by being
silent, because a degraded answer presented as a normal one is worse than an
error: it teaches users to trust something they should not, and it removes the
signal you would have used to discover how often the primary path fails.

**7 · Observability.** The record from lesson 0.1. It fails by being added
after the first incident rather than before it, which is by definition too late
for the incident you would have learned the most from.

## D · From data science to LLM systems

The closest thing in your existing practice is `sklearn.pipeline.Pipeline`, and
the structural similarity is real: named stages, applied in order, each
transforming what the next one sees. Three differences matter enough to
enumerate.

| `Pipeline` | An LLM application |
|---|---|
| Stages are deterministic transforms | One stage is a stochastic remote service |
| A stage either works or raises | A stage can *partially* work — a truncated context, a half-valid object |
| `fit` then `transform` | No fit. The pipeline is all there is |
| Failure surfaces as an exception | Failure often surfaces as a plausible answer |

That last row deserves more than a table cell, because it inverts a debugging
instinct that has probably served you well for years. In a modelling pipeline a
broken stage throws, so the stack trace tells you which stage broke and you
start there. Here a broken stage usually produces something that looks
entirely reasonable: retrieval that returns nothing yields a fluent answer
drawn from the model's memory, and truncation that drops your final instruction
yields a careful answer to a different question. **Nothing raises**, which is
why validation has to be a stage in its own right rather than an assertion you
add after something goes wrong.

The second difference worth dwelling on is partial success. A `transform` that
half-worked is a bug; a context that half-fitted is Tuesday. Because the
degradation is continuous rather than binary, you cannot rely on the presence
or absence of an exception to tell you whether the stage did its job, and the
only remaining option is to measure the stage's output against something you
can check — which is what makes retrieval metrics and schema conformance the
load-bearing measurements of Modules 3 and 2 respectively.

## E · Minimal implementation

The call wrapper, with the three policies that stop stage four from taking the
rest of the system down with it:

```python
def call_with_policy(transport, request, *, timeout_s, max_attempts, budget_tokens):
    spent = 0
    for attempt in range(1, max_attempts + 1):
        if spent + request.estimated_tokens > budget_tokens:
            raise BudgetExceeded(spent, budget_tokens)
        try:
            response = transport(request, timeout_s=timeout_s)
        except (Timeout, TransientError):
            spent += request.estimated_tokens   # a failed attempt is still billed
            if attempt == max_attempts:
                raise
            continue
        spent += response.total_tokens
        return response, spent
```

Two details are easy to omit and expensive to omit, and the first exercise
below is built around both. A failed attempt still consumes budget, because the
provider read your input before the failure occurred and charged you for
reading it; and the budget is checked *before* an attempt rather than after,
because a check that runs afterwards is reporting an overrun you have already
paid for rather than preventing one.

The reason these are worth stating as invariants rather than as coding advice
is that neither produces an error when you get it wrong. A wrapper that bills
only successes under-reports spend by exactly the failure rate, and a wrapper
that checks the budget afterwards guards nothing while appearing, in code
review, to guard something.

## F · Production practice

Real stacks add several components around the seven stages, and each addresses
a specific failure from §C. A request queue with backpressure prevents stage
one from accepting more work than the pipeline can carry. A circuit breaker
stops a struggling provider from being hammered, which addresses retry
amplification directly. Structured tracing with one span per stage turns the
serial-reliability arithmetic from a model into a measurement, because you can
finally see which factor is the small one. And a prompt registry versions stage
three the way you already version code.

Frameworks exist for all of this, and this curriculum covers them at
**awareness level only**: you build the loop yourself, because the loop is what
you will be debugging at two in the morning. Their value is real and largely
organisational — shared conventions, somebody else's integrations, and an
obvious place to put each piece — but their cost is that they conceal exactly
the stage boundaries this lesson is trying to make visible.

The single most useful thing to log, once the record from lesson 0.1 exists, is
**which stage a failed request died in**. That field is cheap to populate,
because each stage already knows its own name, and it converts the serial
reliability model from arithmetic you believe into a distribution you can read
off a dashboard. Without it, every failure looks like a failure of the model
call, since that is the only stage anybody instruments by default; with it, the
usual discovery is that the model call is among the more reliable components
and that the retrieval or validation stage on either side of it accounts for
most of the loss.

## G · Experiment

```bash
python experiments/service_economics.py
```

The pipeline table in §B is the output of a two-line calculation, and it is
worth running rather than reading because the shape of the result is
unintuitive enough that most people's estimate is wrong in the same direction.
Try it with your own stage count: a system with four stages at ninety-nine per
cent sits at 96.1%, and one with a dozen sits at 88.6%.

The practical conclusion is worth stating in the strongest available form.
**Removing a stage is usually a bigger reliability win than improving one**,
because deleting a factor from a product beats nudging a factor towards one,
and it is the option that essentially never appears on a design review agenda.

??? question "Given serial reliability, what is wrong with adding a second model call to check the first one's output?"
    It adds a stage, so the end-to-end rate is multiplied by that stage's own
    reliability before you count any benefit. Worse, the checker fails in ways
    that correlate with the thing it is checking, since both are the same
    model with the same blind spots, so the errors it catches are
    disproportionately the ones you would have caught anyway. You have bought
    some error detection and paid for it in reliability, latency and cost —
    sometimes a good trade, never a free one, and rarely costed before it is
    made.

??? question "A circuit breaker opens during a provider outage and every request now fails instantly. Is that better or worse than before?"
    Better for the provider, better for your latency, and neutral to worse for
    the user unless stage six exists. A circuit breaker converts slow failures
    into fast ones, which is genuinely valuable because it stops one slow
    upstream from consuming every worker you have; but it does not create an
    answer. Without a fallback you have made the failure cheaper without
    making it any less of a failure, and it is worth being clear with yourself
    about which of those two things you just achieved.

## H · Failure modes and cost traps

**No timeout.** The default in most HTTP clients is to wait indefinitely, and a
single slow upstream will then consume every worker in your pool while your own
service appears, from the outside, to have simply stopped.

**Unbounded retries during an outage.** The provider is struggling, so your
clients send more traffic, so the provider struggles more. This is how a
partial outage becomes a total one, and lesson 1.4 measures how much extra load
a naive policy generates.

**Retrying a non-idempotent operation.** If the call had a side effect — a
record written, a message sent, a payment taken — the retry performs it a
second time, and the client believes it made one call. Lesson 1.4 covers
idempotency keys, which are the standard solution and were solved properly by
the payments industry a decade ago.

**Validation that only checks syntax.** `json.loads` succeeding tells you the
model produced JSON. It tells you nothing about whether the fields are the ones
you asked for, whether their types are usable, or whether the values bear any
relation to reality.

**Silent fallback.** A degraded answer that is indistinguishable from a normal
one teaches users to trust it and hides the primary path's failure rate from
you simultaneously. Graceful degradation is good practice; *silent* degradation
is not, and the distinguishing question is simply whether the caller can tell.

**Observability added after the first incident.** Too late, by definition, for
the incident that would have taught you the most.

**Budget checked after the call.** The overrun has already been paid for, so
the check is a metric wearing the costume of a control.

## I · Graded practice

<code-exercise src="tr-l2-policy"></code-exercise>

<code-exercise src="tr-l2-validate"></code-exercise>

<quiz-bank src="tr-l2"></quiz-bank>

## J · Annotated references

- **Nygard, *Release It!*** — timeouts, circuit breakers and bulkheads,
  written about databases and web services long before any of this existed.
  Every pattern in it applies here unchanged, which is itself the lesson.
- **Google SRE Book, "Handling Overload" and "Addressing Cascading
  Failures".** The retry-amplification failure mode above, described properly
  by people who have watched it happen at a scale where it takes a region down.
- **The JSON Schema specification.** Worth an hour of your time, because stage
  five is a schema problem and the vocabulary for describing it already exists
  rather than needing to be invented per project.

## K · Extension

**Draw your own seven stages and put a number on each.** Take any system you
have, whether or not it involves a model, write down its stages in order, and
then write down the failure rate you *believe* each one has. Multiply them
together and compare the result with whatever end-to-end number you currently
report.

The interesting part is not the product but the gaps. Find the two stages for
which you have no measurement at all, because those are the ones to instrument
first, and in practice there are almost always exactly two: the one everybody
assumes is fine, and the one nobody owns.
