---
status: Verified
last_verified: 2026-08-13
volatility: low
pyodide: true
prereqs: ["4.1"]
---

# 4.2 · The loop, and when to stop it

## A · Why this matters

The loop itself is four lines. A model proposes a call, the harness validates
and executes it, the result goes back into the context, and the whole thing
repeats until something says stop. Everything expensive about running agents in
production lives in that last clause, because an agent with no stopping rule
except a step limit reaches the step limit every time it is going nowhere, and
pays full price for each step on the way.

So people add rules. Thirty episodes across
6 <!-- computed: agent_loop.n_behaviours --> authored behaviours, under six of
them:

| rule | solved | tokens | saved | wasted | lost |
|---|---:|---:|---:|---:|---:|
| max_steps | 16 | 22205 | 0.0% | 17622 | 0 |
| budget | 16 | 16811 | 24.3% | 12228 | 0 |
| repeat_stop | 11 | 10764 | 51.5% | 7891 | **5** |
| no_progress | 16 | 13961 | 37.1% | 9378 | 0 |
| novelty | 16 | 11047 | 50.2% | 6464 | 0 |
| composite (naive cache) | 16 | 16364 | **26.3%** | 11921 | 0 |
| composite | 16 | 10334 | **53.5%** | 5891 | 0 |

The baseline spends
17,622 <!-- computed: agent_loop.det_max_steps_wasted --> of its
22,205 <!-- computed: agent_loop.det_max_steps_tokens --> tokens on episodes
that never succeed, which is 79% of the bill for nothing. Every rule improves
on that, and the two interesting rows are the ones where an improvement went
wrong.

**`repeat_stop` looks like the best rule in the table and it is the worst.** It
saves 51.5% <!-- computed: agent_loop.det_repeat_stop_saving_pct --> and loses
5 <!-- computed: agent_loop.det_repeat_stop_lost --> of the sixteen solved
episodes, because refusing to let an agent repeat itself also refuses the agent
that repeats itself once and then does the right thing.

**And the composite is worse than its own best component, until one line is
added.** Serving a repeated call from cache is an obvious saving, and the
version that does so naively saves
26.3% <!-- computed: agent_loop.det_composite_naive_saving_pct --> — against
50.2% <!-- computed: agent_loop.det_novelty_saving_pct --> for the novelty
detector on its own. The cache starves the detector of the observations it runs
on. Counting a cache hit as a step that learned nothing restores it to
53.5% <!-- computed: agent_loop.det_composite_saving_pct -->.

!!! info "Terms used in this lesson"
    **Step** — one iteration: a proposal from the model, a validated call, a
    result appended to the context. Charged at ~90 tokens plus the result.

    **Termination condition** — anything that ends the episode. A step limit is
    one; solving the task is another; the rest are what this lesson is about.

    **Budget** — a cap on spend rather than on iterations, which is the unit
    the bill is actually denominated in.

    **No-progress detector** — a rule that stops when the last K steps taught
    the agent nothing. What counts as "nothing" is the entire design.

    **Novelty** — whether a step produced a result not already seen this
    episode. The stronger definition of progress, and the one that survives §G.

    **Blind spot** — a behaviour a rule structurally cannot see, as opposed to
    one it merely handles badly.

## B · Mental model

**A stopping rule is a classifier, and it has both error types.**

That framing is worth taking literally, because it is the one that makes the
table above readable. A rule looks at a partial episode and predicts whether
continuing is worth paying for. A false positive stops an episode that would
have succeeded, and costs a task. A false negative lets a doomed episode run,
and costs tokens. `repeat_stop` has a
5 <!-- computed: agent_loop.det_repeat_stop_lost -->-episode false-positive
count that the token column completely hides, which is exactly what happens
when a classifier is evaluated on one of its two error types.

The second half of the model is that these rules do not look at the task at
all — they look at the *trace*. None of them knows what the agent was asked or
whether it is close, only what it has called and what came back. That is a real
limitation and also the reason they generalise: a rule that reasoned about the
task would need a model, and a model is the thing whose behaviour you are
trying to bound.

??? question "Why not just ask the model whether it is making progress?"
    Because the failure you are bounding is the model's judgement, and a model
    that has convinced itself the next call will work is exactly the one
    proposing the twelfth identical call. Self-report is also not free: it is
    another generation per step, on the step budget you are trying to protect.
    Lesson 4.6 does use a model to *grade* trajectories, but after the fact and
    outside the loop, where its cost is bounded and its answer is not steering
    the thing it is judging.

## C · Mechanism

Six rules, and the differences between them are all in what evidence they read.

**`max_steps`** counts iterations. It is the only rule that is not optional,
because it is the one that guarantees termination — every other rule is a
prediction, and this one is a fact. What it cannot do is bound *spend*, since
steps vary in cost by an order of magnitude depending on how much a tool
returns.

**`budget`** counts tokens and stops when they run out. It bounds the thing you
are billed for, which is the right unit, and it stops nothing early: on these
episodes it saves
24.3% <!-- computed: agent_loop.det_budget_saving_pct --> and loses nothing.
The threshold is the whole design, and it is set here from what a budget is
*for* rather than from what these episodes cost — the longest legitimate
solution path is four steps, and a budget that cannot absorb two unproductive
steps on top of that is cutting correct work, so the cap is six steps of
headroom at
840 <!-- computed: agent_loop.budget_tokens --> tokens.

**`repeat_stop`** ends the episode when the agent proposes a call it has
already made. It is the cheapest rule to implement and the one with the sharpest
edge, for reasons §G measures.

**`no_progress`** stops after
3 <!-- computed: agent_loop.no_progress_k --> identical results in a row. K is
set from meaning again: two identical results could be a coincidence — a search
that legitimately returns the same document twice — and three consecutive is a
loop.

**`novelty`** stops after K steps in a row that produced no result the episode
had not already seen. The difference from `no_progress` is one word, and §G
shows it is the difference between catching a two-cycle and not.

**`composite`** is budget plus novelty plus serving repeated calls from cache
instead of executing them. The line that makes it work rather than fail:

```python
if caches_repeats and key in seen_calls:
    ep.cached += 1
    if counts_cache_hits:
        ep.novel.append(False)      # a cache hit learned nothing, by definition
        if len(ep.novel) >= k and not any(ep.novel[-k:]):
            return stop("no_novelty")
    continue
```

Without those three lines the `continue` skips the detector entirely, so an
agent that repeats itself becomes invisible to the rule written to catch
exactly that.

??? question "Why does the cache make things worse rather than merely not better?"
    Because it changes the price of a step without changing the number of them.
    A cached step still costs a model call — the proposal was generated and
    paid for — so the agent spins just as long while each turn of the spin
    looks cheaper, and the budget that would eventually have stopped it takes
    correspondingly longer to bite. That is the
    9.8 <!-- computed: agent_loop.naive_flaky_steps --> against
    3.4 <!-- computed: agent_loop.composite_flaky_steps --> in §G, and it is
    the general shape of an optimisation applied to work that should not be
    happening at all.

## D · From data science to LLM systems

Early stopping is a technique you already have. A training loop with patience
watches a validation metric, stops when it has not improved for K epochs, and
carries the same two errors: stop too early and you leave accuracy on the
table, too late and you waste compute. The vocabulary transfers directly, and
K is the patience parameter under a different name.

Two things do not transfer, and both change the design.

**There is no validation metric.** Early stopping works because the quantity
being watched is the quantity you care about, measured on held-out data. An
agent loop has no such signal mid-episode: nothing tells you whether the agent
is closer to the answer, only whether the last call returned something new.
Novelty is a *proxy* for progress, and like every proxy it can be satisfied
without the thing it stands for — an agent making steadily new and steadily
useless calls looks maximally productive to it. That is `runaway`, and it is
why a budget still has to sit underneath.

**And the process is adversarial to your rule in a way a training loop is not.**
Gradient descent does not respond to your patience setting. A model that has
been told it is looping will paraphrase its next call rather than repeat it,
which defeats a call-level check while changing nothing about the behaviour. It
is not doing this on purpose, but the effect is the same as if it were, and it
is the general reason to define these rules over *results* rather than over
*calls*: results are what the world returns, and the model does not author them.

??? question "Does that mean the step limit is doing nothing once the other rules are in?"
    It is doing the one thing none of them can promise. Every other rule fires
    on evidence, so every other rule has some input on which it never fires,
    and `runaway` is that input for the repeat check. The step limit is the
    only guarantee of termination in the system, which is why it stays even
    when it never binds — you keep it for the case you did not think of, not
    for the cases in the table.

## E · Minimal implementation

The loop, with the rules factored out into predicates over the trace:

```python
def episode(propose, tools, rules, max_steps=12):
    trace, tokens, seen_results = [], 0, set()
    for step in range(max_steps):
        tool, args = propose(step, trace)
        tokens += STEP_TOKENS

        if rules.stop_on_repeat and (tool, args) in {(c, a) for c, a, _ in trace}:
            return Episode(trace, tokens, stopped_by="repeat")

        result = tools.call(tool, args)
        tokens += len(str(result)) // 4
        trace.append((tool, args, result))

        if result["ok"] and (tool, args) == goal:
            return Episode(trace, tokens, solved=True)

        novel = digest(result) not in seen_results
        seen_results.add(digest(result))
        if rules.k and len(trace) >= rules.k and not any(novelty[-rules.k:]):
            return Episode(trace, tokens, stopped_by="no_novelty")
        if rules.budget and tokens >= rules.budget:
            return Episode(trace, tokens, stopped_by="budget")
    return Episode(trace, tokens, stopped_by="max_steps")
```

Two details are load-bearing. The budget check comes *after* the call, because
a budget that refuses to start a step it cannot finish will stop one step early
forever and you will never see why. And the novelty check runs on a digest of
the whole result envelope rather than on its `ok` field, so two different
failures count as two different observations — an agent that fails in a new way
each time is learning something, even if it does not look like it.

## F · Production practice

**Bound spend, not steps, and keep both.** The step limit guarantees
termination; the budget bounds the bill. Neither substitutes for the other, and
the composite in §G carries both.

**Define progress over results, not over calls.** This is the whole difference
between `no_progress` and `novelty` in the table, and it is also what makes the
rule robust to a model that rephrases rather than repeats.

**Count a cache hit as a step that learned nothing.** Otherwise the saving you
just added removes the signal your detector runs on, which is the
26.3% <!-- computed: agent_loop.det_composite_naive_saving_pct --> row.

**Record why every episode stopped.** `stopped_by` is one string and it is the
only thing that distinguishes "the agent finished" from "we cut it off", which
is the first question anyone asks about an agent's success rate. Lesson 4.6
grades trajectories, and this field is what makes them gradeable.

**Never stop on a repeated call alone.** It is the rule with the worst
false-positive rate in the table, it is the one most often shipped, and the
behaviour it punishes — try, get an unhelpful result, try the same thing once
more — is not pathological.

**Set K from the tool's failure rate, not from taste.** §G gives the
arithmetic.

## G · Experiment

`python experiments/agent_loop.py`, over
30 <!-- computed: agent_loop.n_episodes --> episodes:
6 <!-- computed: agent_loop.n_behaviours --> behaviours across
5 <!-- computed: agent_loop.n_instances --> instances. The behaviours are
**authored, not sampled from a model**, and what that buys and costs is worth
being explicit about. It costs the ability to say how often a real agent
oscillates. It does not cost the result, because what is measured is the
*interaction* — which rule stops which behaviour, and at what price — and that
follows from the rules once the behaviours are fixed. For the same reason there
are no p-values here: a significance test asks whether a result would replicate
on new samples from the same population, and the population is my own
authorship.

Mean steps per episode, deterministic sandbox:

| behaviour | max_steps | budget | repeat_stop | no_progress | novelty | composite |
|---|---:|---:|---:|---:|---:|---:|
| direct | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| exploratory | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| duplicator | 3.0 | 3.0 | **1.0** | 3.0 | 3.0 | 3.0 |
| flaky_retry | 9.8 | 6.6 | 1.0 | 2.6 | 3.4 | 3.4 |
| oscillator | 12.0 | 8.0 | 2.0 | **12.0** | 5.0 | 5.0 |
| runaway | 12.0 | 9.0 | **12.0** | 3.0 | 4.0 | 4.0 |

**Three blind spots, one per rule, and each is structural rather than a matter
of tuning.**

`repeat_stop` runs `runaway` to the full
12 <!-- computed: agent_loop.repeat_stop_runaway_steps --> steps, because every
call it makes really is new — the check has nothing to fire on. The same rule
solves
0 <!-- computed: agent_loop.repeat_stop_duplicator_solved --> of five
`duplicator` episodes, cutting each at step 1 when the answer was at step 3.
A rule that is both blind to the worst behaviour and fatal to a benign one is
not a rule to tighten; it is one to replace.

`no_progress` runs `oscillator` to
12.0 <!-- computed: agent_loop.no_progress_oscillator_steps --> steps. Two
calls alternating produce two results alternating, so there are never three
identical results in a row and the detector never fires. Novelty stops the same
behaviour at
5.0 <!-- computed: agent_loop.novelty_oscillator_steps --> steps, on the same
K, for a one-word change in what "no progress" means. This is the finding I was
least expecting, and it generalises: any cycle longer than one defeats a rule
written against repetition.

And the naive cache runs `flaky_retry` to
9.8 <!-- computed: agent_loop.naive_flaky_steps --> steps where the corrected
version stops at
3.4 <!-- computed: agent_loop.composite_flaky_steps -->, because every repeat
after the first was served from cache and never recorded, so the novelty window
stayed one observation long forever.

**The cost that has no free side.** Everything above holds in a sandbox where
retrying is always pointless, which flatters every detector — three identical
failures really do mean a fourth is worthless. Run the flaky task with an
independent draw per attempt, which is what a real service does, over
4,000 <!-- computed: agent_loop.sweep_n --> episodes:

| K | cuts a solvable episode | f<sup>K+1</sup> predicts |
|---|---:|---:|
| 2 | 6.1% | 6.4% |
| 3 | 2.7% | 2.6% |
| 4 | 1.2% | 1.0% |

At K = 3 the detector cuts
2.7% <!-- computed: agent_loop.sweep_cut_pct --> of episodes a retry would have
solved, against
2.6% <!-- computed: agent_loop.sweep_predicted_pct --> from the closed form.
It is **K+1** and not K because the first failure is the first time that result
has been seen, so it counts as novel and the run of uninformative steps starts
after it — I had this wrong when I wrote the sweep, and the arithmetic
disagreeing with the measurement by a factor of two is what found it.

That makes K a quantity you compute rather than tune. Given a tool that fails
independently at rate *f*, patience of K costs you f<sup>K+1</sup> of the
episodes that tool appears in, so a 40%-failure service and a 1% tolerance
gives K = 5 <!-- computed: agent_loop.patience_f40_tol1pct --> — with
K = 4 landing at
1.02% <!-- computed: agent_loop.cut_pct_f40_k4 -->, over the line by an amount
that matters only if you have decided it does. Relax the tolerance to 2% and
the answer is
4 <!-- computed: agent_loop.patience_f40_tol2pct -->; take the same 1%
tolerance to a tool that fails a tenth of the time and it is
2 <!-- computed: agent_loop.patience_f10_tol1pct -->. No amount of staring at
traces produces those numbers faster, and the version of this I wrote by hand
had the first one wrong.

??? question "The sweep and the closed form agree. Doesn't that make the sweep redundant?"
    They agree *now*. They disagreed by roughly a factor of two on the first
    run, and the disagreement is what showed the closed form was written as
    f<sup>K</sup> when the loop implements f<sup>K+1</sup>. A derivation and a
    measurement that agree are two independent routes to the same number; one
    of them alone is a number with no error-detection on it, and this exact
    check is what found the off-by-one.

??? question "Every rule in the first table costs at most a few thousand tokens. Why care?"
    Because the table is 30 episodes and the ratios are what scale, not the
    totals. The baseline puts 79% of its spend on episodes that never succeed,
    and that fraction is a property of the behaviour mix rather than of the
    volume — at a million episodes it is the same 79%. The other reason is
    latency, which the token column stands in for: the difference between
    12.0 <!-- computed: agent_loop.no_progress_oscillator_steps --> steps and
    5.0 <!-- computed: agent_loop.novelty_oscillator_steps --> is a user
    waiting more than twice as long for the same eventual failure.

## H · Failure modes and cost traps

**Stopping on a repeated call.** The worst false-positive rate here, and blind
to the behaviour that costs the most.

**Defining progress as "the same result twice".** Cannot see a two-cycle, which
is the most common non-trivial loop there is.

**Caching repeats without recording them.** Turns a working detector off, and
the resulting system saves half what the detector saved alone.

**A budget that refuses to start a step it cannot finish.** Stops one step
early on every episode, permanently, and looks like a tuning problem.

**Reporting an agent's success rate without `stopped_by`.** "68% success" means
something different when the other 32% are budget cuts than when they are wrong
answers, and the number alone cannot tell you which.

**Choosing K by watching traces.** The traces you watch are the ones you
noticed, which are the pathological ones; the episodes K costs you are the ones
that quietly failed. f<sup>K+1</sup> is available and does not have this
problem.

**Assuming a deterministic test harness measures a stopping rule fairly.** Every
number in the first table is generous to the detectors, because a retry can
never pay there. The second table is the correction, and it exists because the
first one looked too good.

## I · Graded practice

<quiz-bank src="agt-l2"></quiz-bank>

<code-exercise src="agt-l2-novelty"></code-exercise>

<code-exercise src="agt-l2-patience"></code-exercise>

## J · Annotated references

- **Yao et al., "ReAct" (2022)** — the interleaved reason/act loop this lesson
  bounds. Read it for the loop shape and note that termination is a footnote,
  which is roughly the weight it gets in most implementations too.
- **Prechelt, "Early Stopping — But When?" (1998)** — the patience parameter,
  worked through carefully in a setting where a validation metric exists. The
  contrast with an agent loop, which has no such metric, is the useful part.
- **Anthropic, "Building effective agents" (2024)** — the argument that most
  agent problems are better solved by a fixed workflow. A workflow terminates
  by construction, which is the strongest version of everything above.
- **Any postmortem of a runaway agent bill** — these are common, public, and
  uniformly about a missing budget rather than a missing step limit.

## K · Extension

*Off-platform, an hour.* Take an agent loop you have run and compute, from its
logs alone, how many episodes ended in each `stopped_by` state. If the field
does not exist, that is the finding. Then estimate *f* for your least reliable
tool from the same logs and work out what patience you have been running
implicitly — most loops have one, expressed as a retry count somewhere, and it
is rarely the number f<sup>K+1</sup> would have chosen.
