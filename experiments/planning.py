"""When a plan pays for itself, and when it cannot exist.

    python experiments/planning.py
    python experiments/planning.py --json

Three ways to organise an agent's work:

    no_plan      decide the next call each turn, using everything seen so far
    static_plan  decide every call up front, then execute the list
    replan       plan up front, execute, and re-plan whenever a call fails

The limit on planning is not a quality problem, it is an information one. A
plan written before execution cannot contain anything execution produces, so on
a task whose second step takes its argument from the first step's result a
planner has to guess -- not because models guess badly, but because the fact
does not exist yet. This measures what that costs; it does not try to prove it,
because it is a logical claim rather than an empirical one.

The cost model has exactly two constants, and the important ratios do not
depend on either:

    SPEC   tokens to write down one step
    OBS    tokens for one observation carried in context

A step call reads the history and emits one step. A plan call reads the same
history and emits every step still to come. That is the only difference, and
everything below follows from it -- including the result that re-planning at
every step costs exactly SPEC * n(n-1)/2 more than never planning at all,
whatever OBS is.

Three results worth the lesson:

  * A static plan solves 0 of 5 discovered-dependency tasks and 5 of 5 of
    everything else, at one model call per task against the interleaved
    strategy's 2.6 to 4.0. The failure is total and the saving is large, which
    is a bad combination to discover in production.

  * Planning buys latency only in proportion to how WIDE the task graph is, and
    the token saving is unrelated. On four independent lookups a plan halves
    the round-trips; on a chain it adds one and saves nothing.

  * Re-planning is not a free upgrade to planning. It is slower than never
    planning on exactly the tasks it exists for, because it pays a plan call to
    learn what an interleaved agent would have read off the previous result.

The strategies are AUTHORED, like Module 4.2's behaviours, and the same caveat
applies: what is measured is the arithmetic of the three shapes, not how well
any particular model plans.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llmlab.tools import Sandbox  # noqa: E402

#: Tokens to write one step of a plan, and to carry one observation. These are
#: assumed rather than measured, which is why the sensitivity sweep exists --
#: but note that only their RATIO can matter, and the headline result below is
#: independent of it.
SPEC_TOKENS = 25
OBS_TOKENS = 45


def step_cost(observations: int, spec: int = SPEC_TOKENS, obs: int = OBS_TOKENS) -> int:
    """Read the history, emit one call."""
    return obs * observations + spec


def plan_cost(remaining: int, observations: int,
              spec: int = SPEC_TOKENS, obs: int = OBS_TOKENS) -> int:
    """Read the same history, emit every step still to come."""
    return obs * observations + spec * remaining


# --- the tasks --------------------------------------------------------------

DEPOTS = ["north-east", "south-west", "central", "north-west", "south-east"]

#: One notes file per depot, so "read the notes for this shipment's depot" has
#: an answer that cannot be written down before the shipment is looked up.
DEPOT_FILES = {f"notes/depot-{d}.txt": f"{d} depot. Exceptions bay is bay 4."
               for d in DEPOTS}
DEPOT_FILES["notes/policy.txt"] = "A hold lasts up to 48 hours."
DEPOT_FILES["notes/readme.txt"] = "Operational notes. Do not distribute."

SHIPMENTS = [f"TL-{4471 + 37 * i}" for i in range(5)]

#: A step that fails this many times running is treated as unrecoverable.
MAX_ATTEMPTS = 6


@dataclass
class Step:
    tool: str
    args: dict
    #: True when this step's argument is only knowable from an earlier result.
    discovered: bool = False
    #: What a planner writing before execution would put here instead.
    guess: dict | None = None
    #: Indices that must complete first. Empty means it can go in round one.
    after: tuple[int, ...] = ()


@dataclass
class Task:
    kind: str
    instance: int
    steps: list[Step]

    def depth(self) -> int:
        """Longest dependency chain: the fewest execution waves any strategy needs."""
        best = [0] * len(self.steps)
        for i, s in enumerate(self.steps):
            best[i] = 1 + max((best[j] for j in s.after), default=0)
        return max(best, default=0)

    def width(self) -> float:
        return round(len(self.steps) / self.depth(), 2)


def tasks() -> list[Task]:
    out = []
    for i, (shipment, depot) in enumerate(zip(SHIPMENTS, DEPOTS, strict=True)):
        # Everything the goal needs is in the goal. A plan is exactly right.
        out.append(Task("static", i, [
            Step("read_file", {"path": "notes/policy.txt"}),
            Step("read_file", {"path": "notes/readme.txt"}, after=(0,)),
            Step("calculator", {"expression": "48 / 2"}, after=(1,)),
        ]))

        # Step 1 tells you which file step 2 must read. No plan can know it.
        out.append(Task("discovered", i, [
            Step("shipment_status", {"shipment": shipment}),
            Step("read_file", {"path": f"notes/depot-{depot}.txt"},
                 discovered=True,
                 guess={"path": "notes/depot.txt"},   # the plausible wrong answer
                 after=(0,)),
        ]))

        # Four independent lookups. Nothing depends on anything.
        out.append(Task("parallel", i, [
            Step("read_file", {"path": "notes/policy.txt"}),
            Step("read_file", {"path": "notes/readme.txt"}),
            Step("search", {"query": "redelivery attempts"}),
            Step("calculator", {"expression": "3 * (4 + 5)"}),
        ]))

        # A chain that runs through the flaky tool.
        out.append(Task("flaky", i, [
            Step("read_file", {"path": "notes/policy.txt"}),
            Step("shipment_status", {"shipment": shipment}, after=(0,)),
        ]))
    return out


def box_for(task: Task, attempt: int = 0) -> Sandbox:
    """A sandbox where retries are independent, so patience can pay."""
    return Sandbox(seed=1000 * (1 + task.instance) + attempt,
                   files=dict(DEPOT_FILES),
                   depots=dict(zip(SHIPMENTS, DEPOTS, strict=True)))


# --- the three strategies ---------------------------------------------------
#
# `waits` is the latency unit, and it is counted the same way for all three: a
# model call is one sequential wait, and a batch of tool calls dispatched
# together is one more, however many calls it contains. Counting a batch as one
# is the whole reason a plan can be faster than adapting.

@dataclass
class Result:
    solved: bool = False
    model_calls: int = 0
    tokens: int = 0
    executed: int = 0
    waits: int = 0
    replans: int = 0
    failed_at: str = ""
    notes: list[str] = field(default_factory=list)


def run_no_plan(task: Task) -> Result:
    """One model call per step, deciding with everything seen so far.

    Adapts perfectly -- a discovered argument is read off the previous result --
    and cannot batch, because it only ever knows the next step.
    """
    r = Result()
    attempt = 0
    for step in task.steps:
        while True:
            r.model_calls += 1
            r.tokens += step_cost(r.executed)
            r.waits += 2                       # decide, then call
            res = box_for(task, attempt).call(step.tool, step.args)
            r.executed += 1
            if res["ok"]:
                break
            attempt += 1
            if attempt >= MAX_ATTEMPTS:
                r.failed_at = step.tool
                return r
    r.solved = True
    return r


def _waves(task: Task, done: set[int]) -> list[list[int]]:
    """Remaining steps grouped into batches that can be dispatched together."""
    out, placed = [], set(done)
    remaining = [i for i in range(len(task.steps)) if i not in placed]
    while remaining:
        batch = [i for i in remaining if set(task.steps[i].after) <= placed]
        if not batch:
            return out
        out.append(batch)
        placed |= set(batch)
        remaining = [i for i in remaining if i not in placed]
    return out


def run_static_plan(task: Task) -> Result:
    """One plan call, then execute the list without further thought.

    Executions cost no model calls at all, which is the entire argument for
    planning, and independent steps go out together. A step whose argument was
    not knowable at plan time carries the planner's guess, and nothing recovers.
    """
    r = Result()
    r.model_calls = 1
    r.tokens = plan_cost(len(task.steps), 0)
    r.waits = 1

    done: set[int] = set()
    for batch in _waves(task, done):
        r.waits += 1
        for i in batch:
            step = task.steps[i]
            args = step.guess if step.discovered else step.args
            res = box_for(task).call(step.tool, args)
            r.executed += 1
            if not res["ok"]:
                r.failed_at = step.tool
                r.notes.append(f"{step.tool} {args} -> {res['error'][:60]}")
                return r
            done.add(i)
    r.solved = len(done) == len(task.steps)
    return r


def run_replan(task: Task) -> Result:
    """Plan, execute in waves, and re-plan from the current state on any failure.

    A re-plan can resolve what the first plan could not, because by then the
    observation is in the history -- but only for steps whose dependencies have
    actually completed. It pays a full plan call to do so.
    """
    r = Result()
    r.model_calls = 1
    r.tokens = plan_cost(len(task.steps), 0)
    r.waits = 1

    done: set[int] = set()
    known: set[int] = set()        # discovered args this plan has resolved
    attempt = 0

    while len(done) < len(task.steps):
        waves = _waves(task, done)
        if not waves:
            break
        batch = waves[0]
        r.waits += 1
        failed = False
        for i in batch:
            step = task.steps[i]
            args = step.args if (not step.discovered or i in known) else step.guess
            res = box_for(task, attempt).call(step.tool, args)
            r.executed += 1
            if res["ok"]:
                done.add(i)
            else:
                failed = True
        if not failed:
            continue

        attempt += 1
        if attempt >= MAX_ATTEMPTS:
            r.failed_at = task.steps[batch[0]].tool
            return r
        # Re-plan: a full plan call over what is left, which can now resolve
        # any discovered argument whose dependencies have completed.
        r.replans += 1
        r.model_calls += 1
        r.waits += 1
        r.tokens += plan_cost(len(task.steps) - len(done), r.executed)
        known |= {i for i in range(len(task.steps))
                  if task.steps[i].discovered and set(task.steps[i].after) <= done}
    r.solved = len(done) == len(task.steps)
    return r


STRATEGIES = {"no_plan": run_no_plan, "static_plan": run_static_plan, "replan": run_replan}


# --- what the arithmetic says, independently of the simulation ---------------

def replan_every_step_premium(n: int, spec: int = SPEC_TOKENS) -> int:
    """How much more re-planning after every step costs than never planning.

    Both strategies read the same history, so every OBS term cancels and the
    difference is spec * n * (n - 1) / 2 -- the cost of re-emitting steps you
    will emit again. Independent of OBS, which is why this is the one number
    here that does not move when the cost model does.
    """
    return spec * n * (n - 1) // 2


def breakeven_surprise_rate(n: int, spec: int = SPEC_TOKENS, obs: int = OBS_TOKENS) -> float:
    """The fraction of steps that may surprise you before no_plan is cheaper.

    Unlike the premium above, this DOES depend on spec/obs, so the sweep
    reports it under the ratio halved and doubled.
    """
    interleaved = sum(step_cost(i, spec, obs) for i in range(n))
    for s in range(n):
        cost = plan_cost(n, 0, spec, obs)
        cost += sum(plan_cost(n - i, i, spec, obs) for i in range(1, s + 1))
        if cost > interleaved:
            return round(s / n, 3)
    return 1.0


def compute() -> dict[str, float]:
    ts = tasks()
    kinds = ["static", "discovered", "parallel", "flaky"]
    out: dict[str, float] = {
        "n_tasks": len(ts), "n_kinds": len(kinds), "n_instances": len(SHIPMENTS),
        "spec_tokens": SPEC_TOKENS, "obs_tokens": OBS_TOKENS,
    }

    for name, fn in STRATEGIES.items():
        rs = [fn(t) for t in ts]
        out[f"{name}_solved"] = sum(1 for r in rs if r.solved)
        out[f"{name}_model_calls"] = sum(r.model_calls for r in rs)
        out[f"{name}_tokens"] = sum(r.tokens for r in rs)
        out[f"{name}_waits"] = sum(r.waits for r in rs)
        for kind in kinds:
            g = [fn(t) for t in ts if t.kind == kind]
            out[f"{name}_{kind}_solved"] = sum(1 for r in g if r.solved)
            out[f"{name}_{kind}_calls"] = round(sum(r.model_calls for r in g) / len(g), 1)
            out[f"{name}_{kind}_waits"] = round(sum(r.waits for r in g) / len(g), 1)
            out[f"{name}_{kind}_tokens"] = round(sum(r.tokens for r in g) / len(g), 1)

    base = out["no_plan_tokens"]
    for name in STRATEGIES:
        out[f"{name}_saving_pct"] = round(100 * (base - out[f"{name}_tokens"]) / base, 1)
        out[f"{name}_wait_saving_pct"] = round(
            100 * (out["no_plan_waits"] - out[f"{name}_waits"]) / out["no_plan_waits"], 1)

    for kind in kinds:
        first = next(t for t in ts if t.kind == kind)
        out[f"depth_{kind}"] = first.depth()
        out[f"width_{kind}"] = first.width()
        out[f"n_steps_{kind}"] = len(first.steps)

    for n in (3, 5, 8):
        out[f"premium_n{n}"] = replan_every_step_premium(n)
        out[f"breakeven_n{n}"] = breakeven_surprise_rate(n)
        out[f"breakeven_n{n}_half"] = breakeven_surprise_rate(n, spec=SPEC_TOKENS // 2)
        out[f"breakeven_n{n}_double"] = breakeven_surprise_rate(n, spec=SPEC_TOKENS * 2)

    # Cross-check: the closed form against the loop, on a chain of five steps
    # with a surprise at every one.
    n = 5
    interleaved = sum(step_cost(i) for i in range(n))
    every = plan_cost(n, 0) + sum(plan_cost(n - i, i) for i in range(1, n))
    out["premium_measured_n5"] = every - interleaved
    out["premium_predicted_n5"] = replan_every_step_premium(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    kinds = ["static", "discovered", "parallel", "flaky"]
    print(f"{int(r['n_tasks'])} tasks: {int(r['n_kinds'])} kinds "
          f"x {int(r['n_instances'])} instances\n")
    print(f"{'strategy':<13}{'solved':>7}{'model calls':>13}{'tokens':>9}{'saved':>8}"
          f"{'waits':>8}{'saved':>8}")
    for name in STRATEGIES:
        print(f"{name:<13}{int(r[f'{name}_solved']):>7}"
              f"{int(r[f'{name}_model_calls']):>13}"
              f"{int(r[f'{name}_tokens']):>9}{r[f'{name}_saving_pct']:>7.1f}%"
              f"{int(r[f'{name}_waits']):>8}{r[f'{name}_wait_saving_pct']:>7.1f}%")

    for label, suffix, fmt in (("solved, by task kind", "solved", "d"),
                               ("model calls per task", "calls", "f"),
                               ("waits per task (latency)", "waits", "f")):
        print(f"\n{label}")
        print(f"  {'strategy':<13}" + "".join(f"{k:>13}" for k in kinds))
        for name in STRATEGIES:
            cells = "".join(
                (f"{int(r[f'{name}_{k}_{suffix}']):>13}" if fmt == "d"
                 else f"{r[f'{name}_{k}_{suffix}']:>13.1f}") for k in kinds)
            print(f"  {name:<13}{cells}")

    print(f"\n  {'steps':<13}" + "".join(f"{int(r[f'n_steps_{k}']):>13}" for k in kinds))
    print(f"  {'depth':<13}" + "".join(f"{int(r[f'depth_{k}']):>13}" for k in kinds))
    print(f"  {'width':<13}" + "".join(f"{r[f'width_{k}']:>13.2f}" for k in kinds))

    print("\nre-planning after every step, against never planning at all")
    print(f"  {'steps':<8}{'premium':>10}{'break-even surprise rate':>26}"
          f"{'spec/2':>9}{'spec*2':>9}")
    for n in (3, 5, 8):
        print(f"  {n:<8}{int(r[f'premium_n{n}']):>10}{r[f'breakeven_n{n}']:>26.3f}"
              f"{r[f'breakeven_n{n}_half']:>9.3f}{r[f'breakeven_n{n}_double']:>9.3f}")
    print(f"\n  closed form vs loop at n=5: predicted {int(r['premium_predicted_n5'])}, "
          f"measured {int(r['premium_measured_n5'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
