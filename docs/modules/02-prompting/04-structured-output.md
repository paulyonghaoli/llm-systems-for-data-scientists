---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 2.4 · Structured output: schema, validation, repair

## A · Why this matters

You asked for JSON and you received something JSON-shaped. The distance
between those two descriptions is where most of the engineering in an
extraction pipeline lives, and it is unusually tractable: almost all of it can
be fixed by code you write once, and the part that cannot be is identifiable in
advance.

Measured on 600 <!-- computed: structured_output.n --> outputs with a stated
mix of realistic malformations, a bare `json.loads` succeeds
61.7% <!-- computed: structured_output.raw_pct --> of the time. Two
deterministic repair stages — neither of which involves calling anything — take
that to 96.2% <!-- computed: structured_output.after_mechanical_pct -->. The
residual 3.8% <!-- computed: structured_output.residual_pct --> is the only
part where an expensive model-based repair has anything to earn.

The more important number is a different one.
7.8% <!-- computed: structured_output.parses_but_fails_schema_pct --> of all
outputs parse perfectly and violate the schema, which means a pipeline that
treats successful parsing as success is wrong about one output in thirteen and
has no signal that it is.

!!! info "Terms used in this lesson"
    **Schema** — the contract an output must satisfy: which keys are required,
    what types their values take, and which values are permitted.

    **Repair ladder** — an ordered sequence of increasingly expensive attempts
    to turn a malformed output into a usable one, cheapest first.

    **Mechanical repair** — deterministic string fixes for known failure
    modes, such as removing a trailing comma. Free, and it either works or it
    does not.

    **Model-based repair** — sending the broken output back and asking for a
    corrected version. A full extra call, with its own failure rate.

    **Silent truncation** — output cut off at `max_tokens` that nonetheless
    parses and satisfies the schema, so no check anywhere reports a problem.

It is worth being precise about why this deserves a lesson at all, given that
none of the individual techniques is difficult. The reason is that the failures
here are almost entirely *quiet*. A malformed response raises, which is
inconvenient and self-announcing; a well-formed response with a missing key, a
stringified number or a silently truncated tail flows into whatever consumes it
and surfaces days later as a data-quality problem nobody can trace back to its
origin. The engineering effort therefore goes not into handling errors, which
is easy, but into converting quiet failures into loud ones, which is the whole
of the design.

## B · Mental model

**Three layers, each a weaker claim than the last.**

```
does it parse?  →  does it match the schema?  →  is it right?
```

They get collapsed into one constantly, and the collapse is expensive because
each layer catches a different class of failure. Parsing catches syntax.
Schema validation catches a well-formed object with the wrong shape — a missing
key, a number that arrived as a string, a status outside the permitted set.
Neither says anything at all about whether the values correspond to reality.

The third layer is genuinely outside what any of this can check, and this
lesson says so rather than gesturing at it: establishing correctness needs an
evaluation on labelled data, which is
[lesson 0.3](../00-transition/03-evaluation-breaks.md)'s machinery, and no
amount of validation substitutes for it.

The useful discipline is therefore to know which layer you are standing on at
any moment. "It parsed" is a much smaller claim than most code treats it as.

??? question "Your extraction pipeline has run clean for a week with no parse errors. What have you established?"
    That the output has been syntactically well-formed, and nothing else. On
    the mix measured in §G that still leaves roughly one output in thirteen
    violating the schema, plus an unknown number that satisfy the schema and
    are factually wrong. A week without parse errors is evidence about your
    parser, not about your data.

There is a fourth question that is easy to forget because it sits outside the
sequence entirely: **did you receive the whole output at all?** A truncated
response can satisfy the first two layers while being an incomplete record, so
completeness is not a stricter version of validity but an orthogonal property
with its own signal. That signal is the finish reason, it costs nothing to
read, and §G measures how often the rest of the machinery misses what it would
have caught.

??? question "Rank these four checks by cost, cheapest first: schema validation, reading the finish reason, a model-based repair call, `json.loads`."
    Reading the finish reason is free — it arrived in the response you already
    paid for. `json.loads` and schema validation are microseconds of local
    compute, so effectively free too. The model-based repair is a full extra
    request, several orders of magnitude more expensive than the other three
    combined. Given that ordering it is worth noticing how often systems
    implement the fourth and skip the first.

## C · Mechanism

**The repair ladder, cheapest first.** Each rung is tried only when the
previous one failed, so the expensive rungs run rarely.

| Rung | Cost | Fixes |
|---|---|---|
| `json.loads` | free | nothing; it is the test |
| Strip fences and preamble | free | ```` ```json ```` blocks, "Here is the JSON you requested:" |
| Mechanical fixes | free | trailing commas, `NaN`, single quotes, unquoted keys |
| Ask the model to repair it | **a full call** | genuinely malformed structure |
| Fail loudly | free | everything else |

Ordering matters for cost rather than for correctness: every rung that runs
before the model call is a chance to avoid paying for it, and §G measures how
often that chance pays off.

**Extraction before repair.** Stripping the fence and the conversational
padding must happen before the mechanical fixes, because those fixes operate on
what should be a JSON document and will otherwise be applied to prose. The
usual implementation takes the substring from the first `{` to the last `}`,
which is crude, deterministic, and correct far more often than it has any right
to be.

**Schema validation, and the trap inside it.** Checking types in Python
contains one genuine hazard: `bool` is a subclass of `int`, so
`isinstance(True, int)` is `True` and a quantity of `true` passes an integer
check unchallenged. Any validator written with bare `isinstance` has this hole,
and it is the second exercise below.

**`NaN` is not a parse failure, which is worse.** Python's `json.loads`
accepts `NaN`, `Infinity` and `-Infinity` **by default**, even though none of
them is valid JSON. Left alone it hands you a float that is not equal to
itself and that `json.dumps` re-serialises back into invalid JSON, so the
failure propagates silently into whatever consumes it rather than announcing
itself at the boundary. The fix is not a string repair but the documented
opt-out:

```python
def reject(name):
    raise ValueError(f"{name} is not valid JSON")

json.loads(text, parse_constant=reject)
```

Every parser in this curriculum uses it, and
2.2% <!-- computed: structured_output.silently_accepted_pct --> of the outputs
in §G would otherwise slip through.

**Truncation is the case that defeats the ladder.** A response cut off at
`max_tokens` sometimes lands after a complete field, and if the remaining keys
were optional it parses *and* satisfies the schema. Nothing in this lesson
catches that; the finish reason from
[lesson 1.4](../01-tokens/04-api-contract.md) is the only reliable signal, which
is why that field keeps reappearing.

**Extraction is deliberately crude, and that is a design choice rather than an
oversight.** Taking everything between the first `{` and the last `}` will do
the wrong thing on a response containing two separate objects, or on prose that
happens to contain a brace. A more careful implementation would track nesting
depth and return the first balanced object, which is perhaps fifteen lines and
worth writing once your outputs justify it. The crude version is presented here
because it is correct on the overwhelming majority of real responses and
because its failure mode is loud: it produces something that does not parse,
which the ladder then reports, rather than something that parses into the wrong
object.

That distinction — loud failure over quiet wrongness — is the principle
governing every rung. Each repair is permitted to fail; none is permitted to
succeed incorrectly. It is why the single-quote fix carries a guard, why
extraction prefers to return garbage that fails parsing over a plausible
fragment, and why the `NaN` opt-out matters more than the `NaN` string
substitution ever did.

??? question "Extraction takes everything between the first `{` and the last `}`. What does it do to a response containing two JSON objects?"
    It returns both plus whatever sat between them, which will not parse, and
    the ladder reports a failure. That is the desired outcome: the alternative
    designs either return the first object silently — discarding data the model
    produced, with nobody informed — or attempt to reconcile the two, which
    requires guessing at intent. A repair that cannot tell which object you
    wanted should decline rather than choose.

## D · From data science to LLM systems

Everything here is data validation, a discipline you already have opinions
about.

| You know | Here |
|---|---|
| Schema validation on ingest | Schema validation on model output |
| `pandera` / `great_expectations` checks | Required keys, types, permitted values |
| Coercing a column that arrived as strings | Coercing a field that arrived as a string |
| Quarantining bad rows | The residual that fails every repair |
| A parse error is loud; a semantic error is quiet | Exactly the same, and worse |

The habit that transfers is quarantine. You already know not to drop a bad row
silently, and the same instinct applies to the
3.8% <!-- computed: structured_output.residual_pct --> that survives every
repair: it should be recorded and countable rather than swallowed by a
`try/except` that returns `None`.

The habit that needs adjusting is where you place the validation. In a data
pipeline, validation sits at the boundary and everything downstream may assume
it passed. Here the "source" is nondeterministic, so the same input can produce
a valid output on Monday and an invalid one on Tuesday — which means validation
is not a one-off gate at ingest but a per-request property with a *rate*, and
that rate belongs in your metrics next to latency and cost.

??? question "You add a `try/except json.JSONDecodeError` that returns None and move on. What have you just made unmeasurable?"
    The failure rate, and therefore any change in it. A swallowed exception
    turns a countable event into an absence, so a provider update that doubles
    your malformation rate shows up as slightly more missing data and nothing
    else. Catch the exception by all means, and increment a counter in the
    same breath — the difference between the two versions is one line and
    about a week of eventual confusion.

There is a second adjustment, and it concerns coercion. In a data pipeline you
routinely coerce a column that arrived as strings, because the source is a
system with a fixed if annoying contract and the coercion is a permanent
adapter. Here coercion is more dangerous, because the "source" can change its
behaviour between requests: a pipeline that silently accepts `"17"` and casts
it will keep working when the model starts returning strings for every numeric
field, and you will never learn that something changed. Validating strictly
and counting the failures preserves the signal; coercing quietly destroys it.

That does not mean never coerce. It means the coercion should be a *recorded*
repair, sitting on the same ladder as the string fixes and counted the same
way, rather than an invisible `int(value)` somewhere in the consumer. The
distinction costs one counter and is the difference between noticing a drift
in week one and noticing it in month six.

??? question "A column arrives as strings in your warehouse and you cast it. The same field arrives as a string from a model and you cast it. Why is the second worse?"
    Because the warehouse's contract is fixed and annoying, whereas the model's
    is neither. A cast in the warehouse is a permanent adapter to a source that
    will behave the same way tomorrow, so the coercion carries no information
    once written. A cast on model output is an adapter to a source that can
    change between requests, so the *rate* at which it fires is a signal about
    the model — and casting silently is precisely what destroys that signal.
    The remedy is not to refuse coercion but to put it on the ladder where it
    is counted, which costs one counter.

## E · Minimal implementation

The ladder, in the order it should run:

```python
def parse(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def extract(text):
    """Strip a code fence and any conversational padding."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def mechanical(text):
    text = re.sub(r",\s*([}\]])", r"\1", text)             # trailing comma
    text = re.sub(r"\bNaN\b|\bInfinity\b", "null", text)    # not valid JSON
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')                       # single-quoted
    return re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', text)  # unquoted keys
```

The guard on the single-quote fix is worth pausing on. Replacing every `'` with
`"` unconditionally destroys any apostrophe inside a legitimately
double-quoted string, turning `"note": "customer's address"` into something
that no longer parses — so the repair only fires when the document contains no
double quotes at all and is therefore single-quoted throughout. A repair that
can corrupt valid input is worse than no repair, because it converts a visible
failure into a different visible failure while looking helpful.

The ladder itself is four lines and its shape carries the cost argument:

```python
def repair(text):
    for candidate in (text, extract(text), mechanical(extract(text))):
        parsed = parse(candidate)
        if parsed is not None:
            return parsed
    return None          # the residual: log it, count it, do not swallow it
```

Each candidate is strictly more processed than the last, and the loop returns
at the first success, so clean input never touches a repair and the expensive
work only happens for the inputs that need it. Extending the ladder with a
model-based rung means appending one more candidate — which is exactly where
it belongs, both in the code and in the reasoning about what it costs.

The `return None` at the end is the part worth defending in review. It would
be easy to raise instead, and easier still to return an empty dict, but a
`None` that the caller must handle explicitly is what keeps the residual
visible. An empty dict flows onward and fails somewhere else, several
functions away from anything that could explain it.

## F · Production practice

Use the provider's structured-output mode when one exists, because a
constrained decoder cannot emit invalid JSON in the first place and makes the
whole first rung of the ladder unnecessary. Lesson 2.5 implements the mask that
makes this work. Keep the ladder anyway for the modes and providers that do not
offer it.

Count each rung. A dashboard showing raw parse rate, post-repair rate and
residual tells you when a provider update has changed the shape of the output,
and it is three counters. Without them the first sign is usually a downstream
consumer complaining about missing fields.

Validate before use, always, and prefer a real schema library to hand-rolled
checks once the shape is more than a few keys deep — the hand-rolled version
acquires the `bool`/`int` hole and several like it, and none of them announce
themselves.

And check the finish reason before any of this. A truncated response that
happens to parse is the one failure the ladder cannot see, and the finish
reason is the only thing that reports it.

Record which rung succeeded, not merely that something did. The distribution
across rungs is a leading indicator: a system that used to satisfy most
requests at the first rung and now needs the mechanical fixes has experienced
a change in model behaviour, and that shift is visible days before it shows up
as a downstream complaint. It is one extra field on the record from
[lesson 0.1](../00-transition/01-what-changes.md).

Keep the schema itself under version control and treat a change to it as a
deployment, because widening a type or adding a permitted status alters what
your pipeline accepts just as surely as changing the prompt does. A schema
edited to make a failing case pass is a decision worth reviewing rather than a
quick fix, and it is one of the few places where the temptation to loosen a
check is immediately rewarded and slowly punished.

Finally, prefer a real schema library once the shape exceeds a handful of
keys. Hand-rolled validators are fine for three fields and acquire the
`bool`/`int` hole, the nested-object problem and the optional-versus-null
distinction as they grow, none of which announce themselves.

## G · Experiment

```bash
python experiments/structured_output.py
```

600 <!-- computed: structured_output.n --> outputs, malformations drawn from
the fixed mix stated in the script's docstring — fenced blocks, preambles,
trailing commas, single quotes, unquoted keys, truncation, `NaN` literals, and
schema violations that parse cleanly.

| Stage | Parses | Gain |
|---|---|---|
| Raw `json.loads` | 61.7% <!-- computed: structured_output.raw_pct --> | |
| + strip fences and preamble | 80.5% <!-- computed: structured_output.after_extract_pct --> | +18.8 <!-- computed: structured_output.gain_extract_pts --> |
| + mechanical fixes | 96.2% <!-- computed: structured_output.after_mechanical_pct --> | +15.7 <!-- computed: structured_output.gain_mechanical_pts --> |

<figure class="llm-fig" markdown>
![Two stacked bar charts. The upper shows parse rate rising across three repair stages from 63.8% to 96.2%. The lower shows three layers — parses, parses and matches schema, and correct — each shorter than the last, with the final bar greyed and labelled as not knowable without an evaluation.](../../assets/generated/figures/structured-output-light.svg){.fig-light}
![Two stacked bar charts. The upper shows parse rate rising across three repair stages from 63.8% to 96.2%. The lower shows three layers — parses, parses and matches schema, and correct — each shorter than the last, with the final bar greyed and labelled as not knowable without an evaluation.](../../assets/generated/figures/structured-output-dark.svg){.fig-dark}
<figcaption markdown>Above: what each rung of the ladder recovers. Below: why "recovered" is a weaker claim than it sounds, with the third bar deliberately empty because this experiment cannot fill it.</figcaption>
</figure>

**Thirty-two points of recovery, at zero marginal cost.** The two free rungs
between them turn a system that fails nearly two requests in five into one that
fails one in twenty-six, without a single additional call. Any argument for
model-based repair has to be made against the
3.8% <!-- computed: structured_output.residual_pct --> that remains, not
against the 38.3% you started with — and a repair call that costs a full
request to recover 3.8% of them is a very different proposition from one that
recovers a third.

**Parsing is not validation.**
7.8% <!-- computed: structured_output.parses_but_fails_schema_pct --> of all
outputs parse cleanly and fail the schema: a required key missing, or a
quantity that arrived as `"17"` rather than `17`. Every one of those would sail
through a pipeline whose only check is a successful `json.loads`.

**The `NaN` case is not a repair problem at all.** I wrote a mechanical
`NaN` → `null` fix and then discovered it never ran, because a default
`json.loads` had already accepted the document at the first rung.
2.2% <!-- computed: structured_output.silently_accepted_pct --> of outputs
carry one of these literals, and without `parse_constant` every one of them
parses into a float that fails every subsequent comparison — including
`value == value`. The experiment now uses a strict parser, which is why the
raw rate above is 61.7% rather than the 63.8% an unguarded one reports; the
difference is exactly the outputs that were being silently corrupted rather
than rejected.

**And the case nothing here catches.** Of the
41 <!-- computed: structured_output.truncated_n --> truncated outputs,
43.9% <!-- computed: structured_output.truncated_parses_pct --> still parse as
JSON, and
9.8% <!-- computed: structured_output.truncated_passes_schema_pct --> parse
*and* satisfy the schema. Those are incomplete records that every check in this
lesson pronounces healthy, and the only thing that reports them is the finish
reason.

??? question "The residual is 3.8%. Under what conditions is a model-based repair call worth adding?"
    When the value of recovering those requests exceeds the cost of the extra
    call *times the rate at which you make it*, which is the easy half. The
    harder half is that the repair call has its own failure rate, so it does
    not recover the full 3.8% — and the outputs reaching it are by
    construction the strangest ones, on which a model is least reliable. In
    most systems the honest answer is to fail loudly on the residual and spend
    the effort on the finish-reason check instead, which costs nothing and
    addresses a failure the ladder cannot reach at all.

A note on reading these numbers. The malformation mix is stated in the script's
docstring precisely because it is a choice rather than a measurement: 55% of
the generated outputs are clean, and if your prompt or provider produces a very
different distribution then every row shifts. What does *not* shift is the
ordering — extraction before mechanical fixes before a model call — because
that follows from the costs rather than from the mix, and a free rung is worth
trying ahead of an expensive one regardless of how often it succeeds.

The one figure worth transplanting directly is the residual, because it is the
input to a decision you will actually face. If your own residual is a fraction
of a percent then a model-based repair rung is almost certainly not worth
building; if it is ten percent it probably is, and the calculation takes a few
minutes against your traffic and your per-call cost.

## H · Failure modes and cost traps

**Treating a successful parse as a successful extraction.** Measured at 7.8%
wrong on this mix, with no signal.

**Hand-rolled type checks and the `bool`/`int` hole.** `isinstance(True, int)`
is `True` in Python, so a boolean passes an integer check. This is not an
obscure edge case; a model emitting `"quantity": true` is exactly the kind of
thing that happens when a field name is ambiguous.

**A single-quote repair with no guard.** Replacing every `'` with `"` corrupts
apostrophes inside valid strings. A repair that can damage well-formed input is
worse than no repair.

**Running mechanical fixes before extraction.** They will be applied to the
conversational preamble as well, which is at best wasted and at worst
introduces quotes into prose that then confuses the extractor.

**Swallowing the parse failure.** `except: return None` makes the failure rate
unmeasurable, so the only symptom of a provider change is a slow drift in
missing data.

**Reaching for a model-based repair first.** It is the most expensive rung and,
on this mix, the two free rungs above it handle nearly ten times as much.

**Leaving `parse_constant` unset.** `NaN` and `Infinity` are accepted by
Python's default JSON parser and are not valid JSON. A permissive parse turns
a rejectable document into a float that is not equal to itself, which is a
considerably worse outcome than an exception.

**Assuming truncation announces itself.** Nearly half of truncated outputs
parse, and a tenth satisfy the schema too. Check the finish reason.

**Validating once at the boundary and trusting it downstream.** The source is
nondeterministic, so validity is a per-request property with a rate rather than
a fact about the pipeline.

**Coercing silently instead of counting.** Casting `"17"` to `17` deep in a
consumer keeps the pipeline working and destroys the signal that the model's
behaviour changed. Coerce on the ladder, where it is recorded, or not at all.

**Loosening the schema to make a failing case pass.** It is immediately
rewarded and slowly punished, and it is a deployment rather than a fix. Widen
a type only with the same deliberation you would apply to changing the prompt.

**Returning an empty object from the residual.** `{}` flows onward and fails
several functions away from anything that can explain it; `None` forces the
caller to decide, which is the point.

## I · Graded practice

<code-exercise src="prm-l4-repair"></code-exercise>

<code-exercise src="prm-l4-schema"></code-exercise>

<quiz-bank src="prm-l4"></quiz-bank>

## J · Annotated references

- **The JSON Schema specification.** Worth an hour, because the vocabulary for
  describing an output contract already exists and reinventing a subset of it
  per project is how the `bool`/`int` hole gets rediscovered.
- **Pydantic's validation documentation.** The most widely used way to express
  these contracts in Python, and its handling of strict versus coercive types
  is directly relevant to the "17" versus 17 case.
- **Your provider's structured-output or JSON-mode documentation.** Read it
  alongside lesson 2.5, which implements the mechanism underneath it. Where it
  is available it removes an entire class of failure rather than repairing it.

- **CPython's `json` module documentation, specifically `parse_constant`.**
  Short, and it documents the `NaN` behaviour that §C treats as the lesson's
  sharpest trap. Worth reading once so the default stops being a surprise.
- **Postel's law, and the several decades of argument against it.** "Be liberal
  in what you accept" produced a generation of systems that could not tell
  which of their inputs were wrong. The repair ladder is deliberately liberal
  *and* counted, which is the compromise the argument eventually reached.

## K · Extension

**Measure your own mix.** Take a few hundred logged outputs, run the three
rungs over them, and record the same three numbers. The mix in §G is stated
precisely so it can be replaced: your malformation distribution depends on your
prompt, your provider and your schema, and the only interesting version of this
table is yours.

Then do the cheaper thing first. Count how many of your outputs arrive with a
finish reason other than a natural stop, because that number is free to obtain
and it bounds a failure mode the entire repair ladder is blind to. If it is
non-trivial, raising `max_tokens` or shortening the requested output will do
more for your extraction rate than any repair you could write.
**And instrument the rung, not just the outcome.** Once the ladder is in
place, record which rung produced each success. That distribution is a cheap
early-warning system: a shift from "most requests parse at the first rung" to
"most need the mechanical fixes" means the model's output shape has changed,
and you will see it days before anything downstream complains. It is one extra
string on a record you are already writing.

