---
status: Verified
last_verified: 2026-08-13
volatility: low
pyodide: true
prereqs: ["4.2"]
---

# 4.3 · Planning and decomposition

## A · Why this matters

Lesson 4.2 bounded a loop that decides one call at a time. The obvious
improvement is to decide all of them at once — write the plan up front, then
execute it — and the obvious improvement is enormous, because the executions
after a plan need no model call at all.

Twenty tasks, four kinds, three strategies:

| strategy | solved | model calls | tokens | saved | waits | saved |
|---|---:|---:|---:|---:|---:|---:|
| no_plan | **20** | 61 | 4810 | 0.0% | 122 | 0.0% |
| static_plan | **14** | 20 | 1375 | **71.4%** | 59 | **51.6%** |
| replan | **20** | 31 | 2985 | 37.9% | 82 | 32.8% |

A static plan costs
1,375 <!-- computed: planning.static_plan_tokens --> tokens against
4,810 <!-- computed: planning.no_plan_tokens -->, and
20 <!-- computed: planning.static_plan_model_calls --> model calls against
61 <!-- computed: planning.no_plan_model_calls -->. It also solves
14 <!-- computed: planning.static_plan_solved --> of twenty tasks where
deciding one step at a time solves
20 <!-- computed: planning.no_plan_solved --> — and the six it loses are not
spread evenly. On one task kind it scores
0 <!-- computed: planning.static_plan_discovered_solved --> of five.

**That failure is not a quality problem and no better planner fixes it.** The
task is: look up a shipment, then read the notes file for whichever depot it
turns out to be at. The filename is a fact that execution produces, so a plan
written before execution cannot contain it. A planner has to guess, and the
guess is wrong four times in five by construction — not because models guess
badly, but because the information does not exist yet.

The strategy invented to fix that is re-planning, and the third row is where it
gets interesting. On this exact task kind, replanning costs
276 <!-- computed: planning.replan_discovered_tokens --> tokens against
191 <!-- computed: planning.no_plan_discovered_tokens --> for having no plan at
all, and takes
6.2 <!-- computed: planning.replan_discovered_waits --> sequential round-trips
against
5.2 <!-- computed: planning.no_plan_discovered_waits -->. **On the task
re-planning exists for, it is both more expensive and slower than never
planning.**

!!! info "Terms used in this lesson"
    **Plan** — a list of calls decided before any of them run. Its value is
    that executing it needs no further model calls.

    **Decomposition** — splitting a goal into steps. The plan is the artefact;
    decomposition is the act.

    **Task graph** — the steps plus their dependencies. Its **depth** is the
    longest chain, its **width** is steps divided by depth.

    **Discovered dependency** — a step whose *argument*, not merely whose
    ordering, comes from an earlier step's result. The distinction is the whole
    lesson.

    **Wait** — one sequential round-trip. A model call is a wait; a batch of
    tool calls dispatched together is one wait however many calls it holds.

    **Re-planning** — producing a fresh plan from the current state after a
    failure, which costs a full plan call.

## B · Mental model

**A plan is a cache of decisions, and like every cache it is only valid while
the thing it summarises has not changed.**

That framing gets the economics and the failure mode in one. The saving is real
and large — decisions computed once and replayed cost nothing to replay, which
is the entire 71.4% column. The invalidation condition is what people skip: a
plan is stale the moment execution produces a fact the plan assumed. On tasks
where execution produces no such facts, the cache never invalidates and
planning is free money. On tasks where it does, you are serving a stale entry
and the tool call fails on an argument nobody checked.

The second half of the model is that **depth and width are different questions
and planning answers only one of them**. Depth is a floor on how many
sequential round-trips any strategy needs, and nothing can beat it. Width is
how much can go at once, and only a strategy that knows more than one next step
can exploit it. That is why the latency column and the token column in §A
disagree with each other about which strategy wins, and why §G measures them
separately.

??? question "If a plan is a cache, is re-planning just cache invalidation?"
    Almost, and the gap is where the cost sits. Invalidating a cache is free;
    re-planning pays a full model call that has to re-emit every step still to
    come. So the analogy holds for *when* to invalidate and breaks for *what it
    costs*, which is exactly why the discovered-dependency row goes the way it
    does — an interleaved agent reads the fact off the previous result for the
    price of one step, and a re-planner pays for the whole remainder to learn
    the same thing.

## C · Mechanism

**Decompose.** Turn the goal into steps and, crucially, into *dependencies*
between them. Most plan formats record only an order, which throws away the
difference between "B must follow A" and "B happens to be written after A" —
and that difference is the entire width column.

**Cost a plan call honestly.** The cost model here has two constants and the
important results do not depend on either:

```python
def step_cost(observations):            # read the history, emit one call
    return OBS * observations + SPEC

def plan_cost(remaining, observations): # read the same history, emit the rest
    return OBS * observations + SPEC * remaining
```

Both read the whole history, which is why an episode's cost grows with the
square of its length whichever strategy you pick — that is a property of
carrying observations forward, not of planning. The only structural difference
is the output: a plan writes down every step still to come, so a plan call is
roughly *remaining* times the size of a step call.

**Execute in waves.** Steps whose dependencies are satisfied go together:

```python
def waves(steps, done):
    out, placed = [], set(done)
    remaining = [i for i in range(len(steps)) if i not in placed]
    while remaining:
        batch = [i for i in remaining if set(steps[i].after) <= placed]
        if not batch:
            return out                  # a cycle, or an unsatisfiable dep
        out.append(batch)
        placed |= set(batch)
        remaining = [i for i in remaining if i not in placed]
    return out
```

The `if not batch: return out` is not defensive padding. A plan with a
dependency cycle is a plan a model can and does emit, and without that check
this loop does not terminate — a planning bug becoming a hang rather than an
error.

**Mark what was guessed.** A step whose argument came from the goal and a step
whose argument the planner invented are different objects, and only one of them
is safe to execute without re-checking. The experiment carries a `discovered`
flag and a separate `guess` for exactly this reason, and a production planner
that cannot tell you which of its arguments are guesses cannot be re-planned
selectively.

**Re-plan on the failure, not on the schedule.** Re-planning after every step
is the same as having no plan, but more expensive — §G puts a number on it.

## D · From data science to LLM systems

You have made this tradeoff before, under the name **query planning**. A
database builds an execution plan once and runs it over millions of rows,
because planning per row would cost more than the query; and the plan goes
wrong exactly when its cardinality estimates were wrong, which is a fact about
the data that planning time did not have. Adaptive query execution — replanning
mid-query once the real row counts are known — is the same third strategy, with
the same justification and the same overhead objection.

Two things differ, and both push toward planning less than you would in a
database.

**The planner and the executor are the same expensive component.** A query
planner is cheap relative to execution, so planning is nearly free. Here the
plan call and the step call come from the same model at the same price, so the
plan has to save more step calls than it costs, and §G's break-even is a real
constraint rather than a formality.

**And a wrong plan fails differently.** A bad query plan is slow; a bad agent
plan calls a file that does not exist, or worse, one that does. That asymmetry
is the argument for the guess-marking in §C: a plan that cannot distinguish its
assumptions from its facts gives you no way to fail safely on the assumptions.

??? question "Isn't this just the interpreter-versus-compiler tradeoff?"
    It is the same shape, and the analogy is worth pushing until it breaks. A
    compiler amortises analysis over many executions, which is why compiling is
    worth it — and an agent plan is executed **once**. There is no amortisation
    here at all, so the saving comes entirely from not re-deciding within a
    single run, which is a much thinner margin than a compiler enjoys. That is
    the reason the break-even matters and why "always plan" is wrong.

## E · Minimal implementation

The static planner is four lines, and its whole content is the branch:

```python
for batch in waves(task.steps, done):
    for i in batch:
        step = task.steps[i]
        args = step.guess if step.discovered else step.args
        result = tools.call(step.tool, args)
        if not result["ok"]:
            return Result(failed_at=step.tool)   # nothing recovers from this
        done.add(i)
```

Re-planning changes one thing — the failure path produces a new plan instead of
ending the episode — and gains one subtlety that is easy to get wrong:

```python
known |= {i for i in range(len(steps))
          if steps[i].discovered and set(steps[i].after) <= done}
```

A re-plan can only resolve a discovered argument whose *dependencies have
actually completed*. Writing this as "after any re-plan, all discovered
arguments are known" makes the strategy look better than it is, because it
hands the re-planner facts it has not observed yet. That is a benchmark-design
error rather than a code bug, and it is the kind that produces a flattering
number nobody questions.

## F · Production practice

**Plan when the task graph is knowable, and say how you know.** "Knowable"
means no step's argument comes from another step's result. That is a property
you can check by reading the goal, and it is worth checking explicitly rather
than discovering in the failure log.

**Record dependencies, not just order.** Without them the width of the graph is
invisible, and width is where the latency saving lives —
8.0 <!-- computed: planning.no_plan_parallel_waits --> waits down to
2.0 <!-- computed: planning.static_plan_parallel_waits --> on four independent
lookups.

**Mark guessed arguments as guesses.** It is the only thing that lets you
re-plan the wrong half of a plan instead of all of it.

**Do not re-plan on a schedule.** Re-planning after every step costs
250 <!-- computed: planning.premium_n5 --> tokens more than never planning on a
five-step task, and the premium grows with the square of the length.

**Treat a cyclic plan as an ordinary output.** Models emit them; the wave
builder must terminate on one rather than hang.

**Budget the plan call.** It scales with the number of steps, so a planner
asked to decompose a large task produces a large output, and a step limit
expressed in *steps* does not bound it. Lesson 4.2's budget is in tokens for
this reason among others.

## G · Experiment

`python experiments/planning.py`, over
20 <!-- computed: planning.n_tasks --> tasks:
4 <!-- computed: planning.n_kinds --> kinds across
5 <!-- computed: planning.n_instances --> instances. The strategies are
authored, exactly as in 4.2, so what is measured is the arithmetic of the three
shapes rather than how well any particular model plans.

Solved, and model calls per task:

| | steps | depth | width | no_plan | static_plan | replan |
|---|---:|---:|---:|---:|---:|---:|
| static | 3 | 3 | 1.00 | 5 · 3.0 | 5 · 1.0 | 5 · 1.0 |
| discovered | 2 | 2 | 1.00 | 5 · 2.6 | **0** · 1.0 | 5 · 2.6 |
| parallel | 4 | 1 | 4.00 | 5 · 4.0 | 5 · 1.0 | 5 · 1.0 |
| flaky | 2 | 2 | 1.00 | 5 · 2.6 | **4** · 1.0 | 5 · 1.6 |

**Planning's latency saving tracks width and nothing else.** Waits per task:

| | width | no_plan | static_plan | speedup |
|---|---:|---:|---:|---:|
| parallel | 4.00 | 8.0 | 2.0 | 4.0× |
| static | 1.00 | 6.0 | 4.0 | 1.5× |

This is structural rather than incidental. An interleaved agent pays two waits
per step — decide, then call — so a graph of width *w* and depth *d* costs it
2*wd*, while a plan pays one wait to think and then one per wave, so *d* + 1.
The ratio is roughly 2*w*, so **a plan over a chain saves almost nothing in
latency however good the plan is**, and a plan over four independent lookups
saves fourfold. Nothing about the token column predicts this: the tokens depend
on the number of steps, and the latency on their shape.

??? question "Could a planner avoid the discovered-dependency failure by planning conditionally — "read whichever file step 1 names"?"
    That is the right instinct and it moves the problem rather than removing
    it, because a conditional plan has to be *interpreted* at execution time by
    something that can read step 1's result and substitute it. If that
    something is code, you have written a workflow and the model is no longer
    planning; if it is the model, you have paid a model call at the point of
    substitution and re-invented interleaving. The useful version is the first
    one, and it is why the strongest advice in this area is usually to replace
    the agent with a fixed workflow wherever the graph is known.

??? question "Why does the `parallel` task have four steps but a depth of one?"
    Because none of its steps depends on any other, so the longest chain
    through the graph is a single node. Depth counts the longest path rather
    than the number of nodes, which is exactly the distinction that makes it a
    latency floor: four independent calls can all be in flight at once, so the
    time is one call's time and not four. Width, steps divided by depth, is
    4.00 <!-- computed: planning.width_parallel --> here and
    1.00 <!-- computed: planning.width_static --> for every other kind.

**Re-planning loses to having no plan on the tasks it was invented for.** On
`discovered`, replan takes
276 <!-- computed: planning.replan_discovered_tokens --> tokens and
6.2 <!-- computed: planning.replan_discovered_waits --> waits, against
191 <!-- computed: planning.no_plan_discovered_tokens --> and
5.2 <!-- computed: planning.no_plan_discovered_waits --> for deciding one step
at a time. The mechanism is plain once stated: an interleaved agent reads the
depot off the previous result as part of an ordinary step, and the re-planner
pays a whole plan call to learn the same fact. Re-planning is still the best
overall strategy in the top table — it is the only one that solves everything —
but it is not a free upgrade, and the row where it is worst is not a corner
case.

**How much re-planning is too much, and how much of that answer is assumed.**
Re-planning after every step costs exactly

$$\mathrm{SPEC} \cdot \frac{n(n-1)}{2}$$

more than never planning, and that expression has no OBS term because both
strategies read the same history — every observation cost cancels. The whole
premium is re-emitting steps you will emit again. At *n* = 5 the closed form
gives 250 <!-- computed: planning.premium_predicted_n5 --> and the loop gives
250 <!-- computed: planning.premium_measured_n5 -->; they disagreed on the
first run because the loop was re-planning after the *final* step, where there
is nothing left to plan.

The break-even surprise rate does depend on the assumed constants, so it is
reported with them halved and doubled rather than quoted as if measured:

| steps | premium | break-even | SPEC ÷ 2 | SPEC × 2 |
|---:|---:|---:|---:|---:|
| 3 | 75 | 0.667 | 0.667 | 0.333 |
| 5 | 250 | 0.600 | 0.800 | 0.400 |
| 8 | 700 | 0.625 | 0.750 | 0.500 |

At five steps, re-planning stops paying somewhere between 40% and 80% of steps
surprising you, depending on how expensive writing a step down really is. That
range is wide, and reporting the middle number alone would have been the
dishonest version.

## H · Failure modes and cost traps

**Planning a task with a discovered dependency.** Scores zero, and the saving
that motivated the plan is still visible in the token column, which is what
makes it survive review.

**Recording order instead of dependencies.** Discards the width of the graph
and with it the only latency saving planning actually offers.

**Re-planning on a schedule rather than on a failure.** Costs SPEC·n(n−1)/2
more than not planning, and grows quadratically.

**Not marking which arguments were guessed.** Forces re-planning to be
all-or-nothing when it could have been partial.

**A plan with a dependency cycle.** Hangs a wave-building loop that lacks the
no-progress check, turning a model error into an unresponsive process.

**Assuming a re-plan knows things it has not observed.** A benchmark bug rather
than a code bug: allow it and re-planning looks strictly better than
interleaving, which is the result §A shows is false.

**Budgeting a planner in steps.** A plan call's size scales with the number of
steps it emits, so "at most 12 steps" bounds nothing about the plan itself.

## I · Graded practice

<quiz-bank src="agt-l3"></quiz-bank>

<code-exercise src="agt-l3-waves"></code-exercise>

<code-exercise src="agt-l3-breakeven"></code-exercise>

## J · Annotated references

- **Wang et al., "Plan-and-Solve Prompting" (2023)** — the plan-then-execute
  shape, argued for on reasoning quality. Read it alongside §G's table and note
  that the tasks it evaluates on have no discovered dependencies.
- **Yao et al., "ReAct" (2022)** — the interleaved alternative, and the paper
  worth re-reading once you have the width argument, because its examples are
  chains and chains are where interleaving loses least.
- **Graefe, "Query Evaluation Techniques for Large Databases" (1993)** — plan
  once, execute over many rows. The classic version of this tradeoff, with the
  cost asymmetry that agents do not get.
- **Any adaptive-query-execution design doc** (Spark AQE is the readable one) —
  re-planning mid-execution, including the accounting for when the re-plan is
  not worth it.

## K · Extension

*Off-platform, an hour.* Take a multi-step agent task you have run and draw its
graph: nodes for steps, edges for real dependencies, and a mark on every
argument that came from another step's output. Then compute its depth and
width. If the width is 1 you have a chain, planning will not make it faster,
and any speedup you were expecting was going to come from somewhere else. If
any argument is marked, note which step produces it — that step is the earliest
point at which a plan for the rest could be correct.
