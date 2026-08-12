---
status: Verified
last_verified: 2026-08-09
volatility: low
pyodide: true
---

# 1.4 · The API contract

## A · Why this matters

A model call is an HTTP request, and almost everything that goes wrong with it
is ordinary distributed-systems trouble wearing an unfamiliar hat.

Two things make it unfamiliar. **Success is not binary** — a 200 response can
contain a truncated answer, a refusal, or an empty string, and only a
secondary field tells you which. And **rate limits reject rather than queue**,
so what happens next is entirely your client's decision. Get that decision
wrong and you can put
218.66× <!-- computed: rate_limits.immediate_load_multiplier --> more load on a
service than the work requires, which is what retrying immediately does.

Both of those properties invert an instinct that has served you reliably
elsewhere. You are used to a status code that means what it says, so that a 200
ends the question of whether the call worked; and you are used to a rate limit
that queues rather than rejects, so that exceeding it costs latency rather than
requiring a decision. Neither holds here, and the cost of discovering that
during an incident is considerably higher than the cost of reading this
section.

!!! info "Terms used in this lesson"
    **Finish reason** — the field saying *why* generation stopped: a natural
    ending, truncation at `max_tokens`, a refusal, or a pause for a tool call.
    It is the only reliable way to distinguish an answer from a fragment.

    **Token bucket** — the usual shape of a rate limit: a bucket holding up to
    `C` requests that refills at `R` per second, where a request consumes one
    and an empty bucket means rejection rather than queueing.

    **Full jitter** — a retry delay drawn uniformly from zero up to an
    exponentially growing bound, rather than set to the bound itself.

    **Idempotency key** — an identifier you generate per *logical* operation so
    that a server can recognise a retry and avoid repeating a side effect.

## B · Mental model

**The contract has four parts, and most people read one.**

| Part | What it says | Usually ignored? |
|---|---|---|
| Request shape | messages, decoding parameters, limits | No |
| Response shape | the text — *and the finish reason* | The finish reason, yes |
| Failure taxonomy | which errors are worth retrying | Yes |
| Rate policy | what happens when you exceed it | Almost always |

A useful reframing: you are not calling a function, you are **submitting work
to a shared service that is allowed to say no**. Everything below follows from
taking "allowed to say no" seriously.

??? question "The API returns HTTP 200 with a non-empty string. Name three ways that response can still be a failure."
    It was truncated at `max_tokens` (finish reason `length`); it is a refusal
    rather than an answer (finish reason `content_filter`, or just a polite
    decline in the text); or it is valid text that does not satisfy your
    schema. All three are "successful" calls, and only the first two are
    visible without parsing.

## C · Mechanism

**Streaming.** The response arrives as a sequence of events rather than one
body. It does not make generation faster — the same tokens are produced at the
same rate — but the user sees the first one after TTFT instead of after the
whole completion, which for a long answer is most of the perceived wait
([0.4 §G](../00-transition/04-cost-and-latency.md) measured TTFT at about a
sixth of median total time). The cost is that error handling moves *into* the
stream: a failure can now arrive after you have already shown the user half an
answer.

**Stop conditions and finish reasons.** Generation ends for one of several
reasons, and the response says which:

| Finish reason | Means | Correct response |
|---|---|---|
| `stop` | The model ended naturally, or hit a stop sequence | Accept |
| `length` | Cut off at `max_tokens` | **Not a complete answer.** Retry with a larger budget, or continue |
| `content_filter` | Refused | A quality event, not a reliability one ([0.1 §H](../00-transition/01-what-changes.md)) |
| `tool_calls` | The model wants a tool run | Dispatch, then continue |

**The failure taxonomy** divides responses into those worth another attempt and
those that will fail identically however many times you send them. A 429, a 5xx
or a timeout is transient and therefore worth retrying, whereas a 400, a 401 or
a 422 describes something about the request itself that will not have changed by
the time the retry arrives. Retrying a request the server has already rejected
on its merits is pure cost, since it will be rejected in exactly the same way
and lesson 0.1's arithmetic confirms you pay for every attempt regardless of its
outcome.

**Rate limits are usually token buckets**, meaning a bucket holds up to `C`
requests and refills at `R` per second, so that each request consumes one token
and an empty bucket produces a rejection rather than a queue. The burst
allowance `C` is why your first several requests always succeed while the tenth
suddenly does not, and it is also why a naive benchmark that fires ten requests
and reports success tells you almost nothing about sustained behaviour.

**Idempotency.** If a request had a side effect and you retry it, the effect
happens twice. An idempotency key — a unique id you generate and the server
remembers — lets the server recognise the retry and return the original
result instead of doing the work again. This is a solved problem borrowed
wholesale from payments.

??? question "Should the idempotency key be generated per attempt or per operation?"
    Per operation, so every retry of the same logical request carries the same
    key. A key generated per attempt is a fresh key each time, which is
    exactly equivalent to having no key at all — and it looks correct in code
    review, because the key is unmistakably there.

## D · From data science to LLM systems

| You know | Here |
|---|---|
| `requests.get(...)` with a timeout | Same, and the default timeout is "forever" |
| Retrying a flaky download | Retrying a *metered* call, where every attempt is billed |
| HTTP status codes | Status codes **plus** a finish reason inside a 200 |
| Rate limits on a data API | Rate limits that reject, with a burst allowance |
| Idempotent GETs | Requests that may have side effects, needing explicit keys |

The instinct that transfers cleanly is your existing discipline about retries
on flaky infrastructure, since you have almost certainly written a download
loop that backs off and gives up. What changes is that every attempt is now
metered, so a retry policy is a spending decision as much as a reliability one,
and the arithmetic in [0.1 §G](../00-transition/01-what-changes.md) applies
directly: the cost per successful request depends on the failure rate rather
than on the retry limit, which means a policy tuned to reduce spend by
retrying less is optimising the wrong variable.

The instinct that fails is treating the status code as the verdict. In most
APIs you have integrated, a 200 means the server did what you asked, so
checking the code is sufficient and checking the body is defensive
programming. Here a 200 means only that the server *responded*, and whether it
did what you asked lives in a separate field that a client library will happily
let you ignore. Every downstream check in [0.2](../00-transition/02-anatomy.md)
exists because that field is so easy not to read.

## E · Minimal implementation

Full-jitter backoff, which is four lines and the subject of §G:

```python
def delay_ms(attempt, base=200, cap=30_000, rng=random):
    bound = min(cap, base * 2 ** (attempt - 1))
    return rng.uniform(0, bound)          # not `bound`, and not `bound ± 10%`
```

And the decision the response actually requires:

```python
def classify(status, finish_reason=None):
    if status == 429:              return "retry_after"
    if status >= 500:              return "retry"
    if status >= 400:              return "fail"        # our fault; retrying repeats it
    if finish_reason == "length":  return "truncated"   # a 200 that is not an answer
    if finish_reason == "content_filter": return "refused"
    return "accept"
```

Five lines, and it is the difference between a system that degrades and one
that lies.

Neither function is complicated, and both are worth writing out because their
mistakes are so quiet. The delay is drawn from the whole interval rather than
perturbed around its top, since a small wobble around a common value is still a
common value and leaves clients that collided at the same instant colliding
again. The classifier checks the status before the finish reason, and treats
429 separately from the other client errors, because 429 is the one 4xx worth
retrying while the rest describe a request the server has already judged on its
merits.

What both share is that getting them wrong produces no error. A backoff without
jitter still retries and still eventually succeeds; a classifier that returns
`accept` for a truncated response still returns something the caller can use.
The failure appears later, as an unexplained load pattern or an answer that
stops mid-sentence, at which point the cause is several layers away from the
symptom.

## F · Production practice

Honour `Retry-After` when the server sends it — it is the one piece of
information you have about when capacity will exist, and it beats any policy
you could compute. Read the rate-limit headers if the provider publishes
remaining quota; proactive throttling is cheaper than reactive retrying.

Send an idempotency key on anything with a side effect, and generate it from
the *logical* operation rather than from the attempt, so retries share a key.

Set a client timeout explicitly. Use streaming for anything a user waits on,
and make sure your error path can handle a failure arriving mid-stream after
partial output has been displayed.

And put a cap on total attempts *and* total elapsed time. A policy bounded
only by attempt count can, with exponential backoff, keep a request alive for
an hour.

Bound the total elapsed time as well as the attempt count, because those are
different limits and only one of them is usually stated. With exponential
backoff and a thirty-second cap, a policy allowing eight attempts can keep a
single request alive for minutes, which is indistinguishable from a hang to
whoever is waiting on it; a deadline expressed in seconds is what the caller
actually cares about, and it is the one most retry configurations omit.

Generate the idempotency key from the logical operation rather than from the
attempt, so that every retry of the same request carries the same key. A key
generated per attempt is a fresh key each time, which is exactly equivalent to
having no key at all while looking entirely correct in review — the kind of bug
that survives indefinitely because the mechanism is visibly present.

## G · Experiment

```bash
python experiments/rate_limits.py
```

50 <!-- computed: rate_limits.clients --> clients, 20 requests each —
1,000 <!-- computed: rate_limits.total_requests --> in total — against a
bucket refilling at 20 <!-- computed: rate_limits.refill_per_s --> per second.
The fastest anyone could finish is
49.5 <!-- computed: rate_limits.floor_s --> seconds.

| Policy | Finished in | vs floor | Attempts per success |
|---|---|---|---|
| Retry immediately | 49.5 <!-- computed: rate_limits.immediate_completed_s -->s | 1.00× <!-- computed: rate_limits.immediate_vs_floor --> | 218.66× <!-- computed: rate_limits.immediate_attempts_per_success --> |
| Fixed 200 ms | 49.6 <!-- computed: rate_limits.fixed_completed_s -->s | 1.00× <!-- computed: rate_limits.fixed_vs_floor --> | 13.05× <!-- computed: rate_limits.fixed_attempts_per_success --> |
| Exponential, no jitter | 114.4 <!-- computed: rate_limits.exponential_completed_s -->s | 2.31× <!-- computed: rate_limits.exponential_vs_floor --> | 2.44× <!-- computed: rate_limits.exponential_attempts_per_success --> |
| **Exponential + full jitter** | **67.2 <!-- computed: rate_limits.full_jitter_completed_s -->s** | **1.36× <!-- computed: rate_limits.full_jitter_vs_floor -->** | **2.94× <!-- computed: rate_limits.full_jitter_attempts_per_success -->** |

**This is not the result I expected, and the disagreement is the useful part.**

The familiar advice is "add jitter, it reduces retries". On these numbers it
does not: plain exponential backoff generates *fewer* attempts per success
(2.44 against 2.94). It achieves that by sleeping — it finishes in
2.31× <!-- computed: rate_limits.exponential_vs_floor --> the minimum possible
time, with clients still asleep while capacity sits unused.

So "fewer retries" is the wrong objective. **A policy can minimise retries by
declining to do the work.** The quantity that matters is time to complete under
the constraint, and there jitter wins clearly:
1.7× <!-- computed: rate_limits.jitter_vs_exponential_speedup --> faster than
plain exponential, at 1.36× the theoretical floor.

And at the other extreme, retrying immediately also finishes at the floor —
because it grabs every token the instant it appears — while putting
74.4× <!-- computed: rate_limits.immediate_over_jitter_load --> more load on
the service than full jitter does. That is the behaviour that turns a busy
provider into an unavailable one.

Full jitter is the balanced choice: nearly the throughput of hammering, at
nearly the politeness of deep backoff.

??? question "Plain exponential backoff generated the fewest attempts. Why is it still the worst policy here?"
    Because it took 2.31× the minimum time to do the same work. Its low
    attempt count is a symptom of clients being asleep while the bucket had
    tokens to give. Optimising the number of retries optimises a cost, not an
    outcome — and the outcome is when the work finishes.

??? question "Retrying immediately finished at the theoretical floor. Why is that not an argument for it?"
    Because the floor is set by the service's capacity, and the simulation
    grants that capacity regardless of load. A real service under 218×
    amplification degrades, and the policy that caused it is competing with
    every other client doing the same thing. It is fast in a model where
    nobody else exists.

## H · Failure modes and cost traps

**Treating 200 as success.** A truncated answer, a refusal, and a real answer
all arrive with the same status code.

**Retrying a 400.** The server has already judged the request. Retrying it
costs money and changes nothing.

**Exponential backoff without jitter.** Clients that collided stay
synchronised and keep colliding, and the deep sleeps cost you throughput —
2.31× the floor above.

**Jitter of ±10% instead of full.** A small perturbation around a common value
is still a common value. Full jitter means uniform over the whole interval.

**Bounding retries by count only.** With exponential backoff, "up to 8
attempts" can mean an hour. Bound elapsed time as well.

**Retrying a side-effecting request without an idempotency key.** The effect
happens twice, and the second one is invisible in your logs because the client
believes it made one call.

**Ignoring `Retry-After`.** It is the only reliable information you will get
about when capacity returns.

**Forgetting that a stream can fail after partial output.** The user has
already read half an answer that is now not going to be finished.

**Reading the rate limit as a throughput target.** A bucket that refills at
twenty per second does not mean you may sustain twenty per second, because your
own retries consume tokens from the same bucket and every rejected request has
already cost you an attempt. The sustainable rate is therefore lower than the
advertised one by roughly the failure rate, and a client tuned to sit exactly
at the limit will oscillate between saturation and rejection rather than
settling.

**Streaming without a plan for a mid-stream failure.** Once the first tokens
have been shown, the error path has to decide between leaving a truncated
answer on screen, replacing it, or appending an apology to it — and whichever
you choose is a product decision that will be made by default if you do not
make it deliberately. The one option unavailable to you is the one
non-streaming code takes for granted, which is to fail before the user has seen
anything.

## I · Graded practice

<code-exercise src="tok-l4-backoff"></code-exercise>

<code-exercise src="tok-l4-classify"></code-exercise>

<quiz-bank src="tok-l4"></quiz-bank>

## J · Annotated references

- **Brooker, "Exponential Backoff And Jitter" (AWS Architecture Blog).** The
  original simulation this lesson's §G is modelled on. Worth reading against
  these numbers, because the objective it optimises is not the same one.
- **Stripe's idempotency documentation.** The clearest statement of the
  idempotency-key pattern anywhere, and it transfers unchanged.
- **The Server-Sent Events specification.** What streaming actually is,
  including reconnection semantics people usually discover by accident.
- **Google SRE Book, "Handling Overload".** Client-side throttling, and why a
  well-behaved client is part of a service's capacity planning.

## K · Extension

**Measure your own provider's bucket.** Send requests as fast as they will be
accepted and record when the rejections start. The number accepted before the
first 429 is roughly the burst capacity; the steady rate afterwards is the
refill rate. Ten minutes, and you now have the two parameters §G's model needs.

**Then check your client.** Find the retry policy in whatever SDK you use and
answer three questions: is there jitter, is it full or proportional, and is
there a bound on total elapsed time as well as attempts? In most libraries at
least one of those answers is unwelcome.

**Then check what your retry policy would do to a struggling provider.** Take
the policy your SDK ships with, work out how many attempts it makes in the
first ten seconds of a total outage, and multiply by your request rate. That
number is the additional load your service contributes at precisely the moment
the provider can least absorb it, and comparing it against §G's table tells you
whether your client is a well-behaved participant or part of the problem.

The corresponding question on the response side takes about the same time.
Search your codebase for every place a model response is consumed and count how
many of them inspect the finish reason before using the text. In most systems
the answer is zero, and the fix is a single shared helper rather than a change
at each call site — which makes it one of the cheapest reliability improvements
available anywhere in this curriculum.
