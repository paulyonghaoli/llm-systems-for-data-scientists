---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 0.4 · Cost and latency are features

## A · Why this matters

In your previous work the expensive part happened once. Training consumed a
budget that somebody approved, and inference afterwards was cheap enough that
nobody costed it per prediction. That relationship has inverted completely:
training cost is now somebody else's problem entirely, and **you pay per
request, forever, in two currencies at once** — money and time.

The inversion changes who has to care. A model that is two per cent better and
three times more expensive is not a better model; it is a different product
decision, and you are now the person holding both halves of it. Neither number
is difficult to compute, and both are routinely guessed at, usually in the
direction that makes the design under discussion look good.

No prices appear anywhere in this lesson. They change monthly and would be
stale before you read this, and in any case the useful quantities turn out to
be ratios rather than absolutes — see [the living doc](../../living/models.md)
for why every volatile figure in this curriculum is confined to a single page.

!!! info "Terms used in this lesson"
    **Price ratio** — how much more an output token costs than an input token.
    It reflects the fact that generating a token requires a full forward pass
    while reading one does not, so it is structural and moves slowly even as
    absolute prices fall.

    **Time to first token (TTFT)** — how long a request waits before any
    output appears. Driven by queueing and by prompt length, because the whole
    prompt is processed before the first output token exists.

    **Time per output token (TPOT)** — how quickly the remainder arrives once
    generation has started. Independent of prompt length.

    **Percentile (p50, p95, p99)** — the value below which that percentage of
    requests fall. p50 is the median; p95 is the experience of the unluckiest
    one request in twenty.

    **Unit economics** — cost per unit of the thing the business counts: per
    request, per conversation, per resolved ticket. Cost per *successful*
    request is usually the honest denominator.

## B · Mental model

**Two different distributions, and neither is well described by its mean.**

Cost is driven by token counts in two directions at two different prices, so
the shape of your traffic decides where the money goes — and where it goes is
usually not where people look, because the intuition "output tokens are
expensive" is true per token and misleading in aggregate. Latency is
heavy-tailed, so the average request is not the average experience, and the
tail is not the rare case it sounds like when you call it a tail.

The single most useful reframing in this lesson is short enough to remember:
**the mean is a number about your system, and the percentile is a number about
your users.** A dashboard showing mean latency is describing a request that
most people never make, which is why the first thing to do with any latency
panel you inherit is to check what statistic it is plotting.

??? question "Your p95 latency is 4.3 seconds and your mean is 2.3 seconds. Which number goes in the design conversation?"
    The p95, and then the p99 immediately after it. The mean says a typical
    request is fine, which is true and not the question; the p95 says that one
    request in twenty takes nearly twice as long, and for a user making twenty
    requests in a session that slow request is close to a certainty rather
    than an edge case. Design for the experience people actually have, which
    means designing for the tail of the distribution rather than its centre.

## C · Mechanism

**Cost.** Per request, the arithmetic is a single line:

$$
\text{cost} = n_{\text{in}} \cdot p_{\text{in}} + n_{\text{out}} \cdot p_{\text{out}}
$$

Output tokens are priced several times higher than input tokens; call that
multiple the **price ratio** `r`. Because the ratio reflects a real difference
in the work done — generating a token requires a full forward pass through the
model, whereas reading one is processed in parallel with all the others — it
moves slowly even when absolute prices fall sharply. Everything below is
therefore expressed in terms of `r`, which means the conclusions survive the
next repricing while a figure in currency would not.

**Latency.** Two components, experienced quite differently by the person
waiting. **Time to first token** is dominated by queueing and by the length of
the prompt, because the model must process the entire prompt before the first
output token can exist. **Time per output token** governs how fast the rest
arrives, and multiplies by the number of tokens generated. Total time is
approximately TTFT plus TPOT times output length.

With streaming, the user experiences TTFT as responsiveness and the rest as
reading speed; without it, they wait for the whole thing and experience the
total. Lesson 1.4 covers the streaming mechanics, and Module 11 explains where
these two numbers come from inside a serving engine.

The practical consequence of that split is that the two components respond to
completely different interventions. Shortening the prompt reduces TTFT and does
nothing to TPOT, while asking for a shorter answer reduces the TPOT term and
does nothing to TTFT; adding capacity reduces the queueing part of TTFT and
leaves generation speed exactly where it was. Reporting a single latency figure
therefore does not merely lose information — it makes the next optimisation
decision unanswerable, because the number cannot tell you which of three
unrelated levers would have moved it.

## D · From data science to LLM systems

| Your habit | What it becomes |
|---|---|
| Cost is a training-time budget, approved once | A per-request unit cost that scales with success |
| Latency is a deployment detail for someone else | A product constraint you design around |
| Report the mean | Report p50, p95 and p99; the mean summarises the system, not anyone's experience |
| Optimise the model | Optimise the *request* — most of the bill is input tokens you chose to send |

There is an instinct you already possess that transfers perfectly, even if you
have never applied it here. You know that a right-skewed distribution should
not be summarised by its mean, and you apply that automatically to income, to
house prices and to session durations without needing to be reminded. Latency
is the same shape and deserves the same treatment; the only reason it usually
does not get it is that monitoring dashboards default to averages and nobody
changes the default.

The habit that needs adjusting is subtler. In modelling work, the expensive
resource was compute during training, and it was consumed once by a process you
controlled end to end. Here the scarce resource is a per-request token budget
with a hard ceiling, consumed by every user interaction, and the quantity you
are optimising is not the model's efficiency but the *request's* — which is a
design question rather than a modelling one, and which is answered in Modules
1 and 3 rather than here.

## E · Minimal implementation

```python
def cost_units(records, price_ratio):
    """Cost in units of one input token, so no prices are needed."""
    return sum(r["in_tokens"] + r["out_tokens"] * price_ratio for r in records)


def percentile(values, q):
    """Nearest-rank percentile. Exact, no interpolation, no surprises."""
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]
```

Expressing cost in "input-token units" is a small trick that repays itself
repeatedly. It lets you compare two designs, run a regression on spend, and put
a defensible number in a design document without hard-coding a price that will
be wrong next quarter; converting to currency is one multiplication whenever
somebody actually needs money rather than a ratio. It also makes the two
providers you are choosing between directly comparable, since their price
ratios usually differ less than their absolute prices do, and a design that
wins on token units tends to win on both bills.

The nearest-rank percentile is worth using in preference to an interpolating
one for a reason that has nothing to do with accuracy: it is exactly the value
of an observation you actually made, so when you go looking for the request
that produced your p95 you will find it in the log rather than finding two
requests that bracket a number nobody experienced.

## F · Production practice

Take token counts from the provider's response rather than from a local
estimate, for the reasons lesson 1.1's extension sets out and lesson 1.3
quantifies. Break latency into TTFT and total, because those two numbers have
different causes and different fixes and a single figure conceals which one you
are looking at.

Track cost per *successful* request rather than per call, since failures and
retries are real spend that no one bills you separately for — lesson 0.1's
retry arithmetic gives the multiplier. Set a budget per request and enforce it
before the call rather than after, per lesson 0.2. And put the p95 rather than
the mean on the dashboard people actually look at, because the statistic on the
default panel is the one that will be quoted in every subsequent meeting.

## G · Experiment

```bash
python experiments/service_economics.py
```

A simulated retrieval-augmented workload of twenty thousand requests, averaging
1897 <!-- computed: service_economics.mean_in_tokens --> input tokens and
164 <!-- computed: service_economics.mean_out_tokens --> output tokens. The
traffic model is stated in full in the script; these are its numbers rather
than a measurement of any provider, and what transfers is the shape of the
conclusion rather than the absolute figures.

**Where the money goes.** Input accounts for
92.1% <!-- computed: service_economics.input_share_of_tokens_pct --> of all
tokens, but its share of the *bill* depends on the price ratio:

| Output costs | Output's share of spend |
|---|---|
| 3× input | 20.6% <!-- computed: service_economics.output_share_of_cost_r3_pct --> |
| 4× input | 25.7% <!-- computed: service_economics.output_share_of_cost_r4_pct --> |
| 5× input | 30.1% <!-- computed: service_economics.output_share_of_cost_r5_pct --> |

At a fourfold ratio, therefore, roughly **three quarters of the bill is the
prompt** — the part you wrote, rather than the part the model generated.
Cutting thirty per cent from prompts saves
22.3% <!-- computed: service_economics.saving_from_30pct_shorter_prompts_pct -->
of total spend, while cutting thirty per cent from answers saves
7.7% <!-- computed: service_economics.saving_from_30pct_shorter_answers_pct -->.

This is the opposite of where attention usually goes, and the reason is a
reasonable intuition applied to the wrong quantity. "Output tokens are
expensive" is true per token and misleading in aggregate, because a
retrieval-augmented request sends far more than it receives, and four times a
small number remains the smaller of the two. Which lever is larger is an
empirical question about *your* traffic rather than a general rule, and
answering it takes about ten minutes.

**What the latency number means.**

| | ms |
|---|---|
| mean | 2260 <!-- computed: service_economics.latency_mean_ms --> |
| p50 | 1982 <!-- computed: service_economics.latency_p50_ms --> |
| p95 | 4274 <!-- computed: service_economics.latency_p95_ms --> |
| p99 | 7031 <!-- computed: service_economics.latency_p99_ms --> |

Only 37.2% <!-- computed: service_economics.pct_slower_than_mean --> of requests
are slower than the mean, which is the signature of a right-skewed
distribution: a small number of very slow requests drag the average above the
typical experience, so "average latency" describes a request that most users
never make. And time to first token is just
15.4% <!-- computed: service_economics.ttft_share_of_latency_p50_pct --> of
median total time, which is the entire argument for streaming — you can hand
the user something to read in roughly a sixth of the time it takes to finish
generating.

??? question "Given that input dominates the bill, why is 'just retrieve fewer documents' not automatically the right call?"
    Because retrieval quality is what makes the answer correct, and the cost of
    a wrong answer does not appear anywhere on this page. Cutting context is a
    quality-for-cost trade rather than a free saving, which means it has to be
    measured with lesson 0.3's tools, on the same items, with an interval
    attached. Cheaper and worse is easy to achieve accidentally; the
    engineering is in knowing which of the two you got.

??? question "Why express costs in input-token units rather than in currency, when the finance team wants currency?"
    Because the analysis outlives the price list. Every conclusion in §G is a
    share or a ratio — which side dominates, what a thirty per cent cut is
    worth, how the mean relates to the p95 — and all of those survive a
    repricing unchanged. Give the finance team currency by all means; keep the
    design document in ratios, because it is the document that will still be
    read in six months.

## H · Failure modes and cost traps

**Optimising output length when input dominates.** The most common misdirected
effort in this area, and §G quantifies the misdirection: at a fourfold price
ratio a thirty per cent cut in prompts is worth roughly three times a thirty
per cent cut in answers.

**Putting the mean on the dashboard.** It hides the tail by construction, and
the tail is what your users complain about.

**Measuring latency without splitting TTFT from total.** The two have different
causes — queueing and prompt length on one side, generation length on the other
— so a single number cannot tell you which to fix, and the fixes are unrelated.

**Estimating p95 as mean plus two standard deviations.** That assumes a normal
distribution, and heavy-tailed latency is the canonical example of data that is
not. On the fixture in this lesson's second exercise the shortcut overshoots
the true p95 by about seventy per cent, and you will not discover the error
until an SLO is breached.

**Costing per call rather than per success.** Failures and retries are billed,
so the denominator matters; lesson 0.1's arithmetic supplies the multiplier.

**Pricing a design against today's prices.** Absolute prices are the
fastest-decaying fact in this field. Ratios are not.

**Forgetting that a longer prompt is also slower.** Prompt length appears in
TTFT as well as in the bill, so context bloat is paid for twice, once in money
and once in the wait before anything appears on screen.

??? question "You halve your prompt length. Which of cost, TTFT and TPOT improve?"
    Cost and TTFT. Time per output token is the speed of generating each token
    of the answer and does not care how long the prompt was, so a shorter
    prompt shortens the wait before the answer starts without changing how
    fast it then arrives. This asymmetry is why prompt bloat is worth
    attacking first: it is the only quantity in the system that you are
    charged for twice.

??? question "A vendor cuts prices by 40% and your traffic doubles. Which of this lesson's conclusions change?"
    None of them. Every figure here is a share or a ratio — which side
    dominates the bill, what a thirty per cent cut is worth, how the mean
    relates to the p95 — so absolute spend changes while the shape does not.
    That is the entire argument for costing designs in input-token units: the
    analysis outlives the price list, and converting to money is one
    multiplication performed whenever somebody actually needs a figure in
    currency.

## I · Graded practice

<code-exercise src="tr-l4-cost"></code-exercise>

<code-exercise src="tr-l4-latency"></code-exercise>

<quiz-bank src="tr-l4"></quiz-bank>

## J · Annotated references

- **Gil Tene, "How NOT to Measure Latency".** A talk about coordinated
  omission and why almost every latency number you have seen is optimistic.
  The single most useful hour available on this subject, and it will change
  how you read every dashboard afterwards.
- **Dean & Barroso (2013), *The Tail at Scale*.** Why tail latency dominates
  user experience once a request touches several components, which every LLM
  application does by the time it reaches production.
- **Google SRE Book, the chapter on Service Level Objectives.** How to turn a
  percentile into a commitment that somebody can be held to, which is the step
  between measuring latency and managing it.

## K · Extension

**Compute your own version of the §G tables.** Take a day of real traffic, or a
hundred requests you make deliberately, and produce two things: the share of
spend that is input versus output at your provider's price ratio, and your p50,
p95 and p99 latency split into TTFT and total.

Then answer the only question that really matters. If you had to cut spend by
thirty per cent, which lever would you pull, and what would it cost you in
quality? The second half of that question is what makes it an engineering
answer rather than a finance one, and answering it properly requires lesson
0.3's machinery rather than this lesson's.
