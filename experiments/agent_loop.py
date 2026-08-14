"""What a termination condition catches, what it misses, and what it costs.

    python experiments/agent_loop.py
    python experiments/agent_loop.py --json

An agent loop that only counts steps runs a broken episode to the step limit
every time, so people add stopping rules: a token budget, a repeated-call
check, a no-progress detector. Each saves money on the episodes that were going
nowhere. Each can also cut an episode that was about to succeed, and each has a
blind spot that the behaviour it was written for does not reveal.

Six behaviours drive the loop, and they are AUTHORED rather than sampled from a
model. That is the honest limitation and the lesson states it: what is measured
is the arithmetic of the stopping rules given those behaviours, not how often a
real model behaves that way. What authorship cannot fake is the *interaction* --
which rule stops which behaviour and at what cost -- because that follows from
the rules once the behaviours are fixed.

For the same reason there are no p-values below. A significance test answers
"would this replicate on new samples from the same population", and the
population here is my own authorship.

Three results the lesson is built on, none of which I expected:

  * A repeated-call check cannot see `runaway`, whose calls are all distinct
    and all useless, and it kills `duplicator`, which repeats itself once and
    then does the right thing.
  * A no-progress detector defined as "K identical results in a row" cannot see
    a two-cycle at all: `oscillator` runs to the step limit under it. Defining
    progress as *novelty over a window* fixes that.
  * Serving a repeated call from cache -- an obvious saving -- starves the
    novelty detector of the observations it runs on, and the composite that
    does this ends up saving LESS than the detector did on its own. Counting a
    cache hit as a step that learned nothing is the whole fix.

And the tradeoff that has no free side: a novelty detector cuts a *solvable*
episode whenever a genuinely flaky tool fails K+1 times in a row -- K+1 and not
K, because the first failure is the first time that result has been seen and so
counts as novel. That is f**(k+1), arithmetic rather than a hyperparameter to
tune, and the sweep below checks the loop agrees with it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llmlab.tools import Sandbox  # noqa: E402

#: A step is one model proposal. Tool results are charged on top, by length.
STEP_TOKENS = 90

#: The step limit everyone ships, and the only rule the baseline has.
MAX_STEPS = 12

#: Set from what a budget is FOR, not from what these episodes cost: the
#: longest legitimate solution path here is four steps, and a budget that
#: cannot absorb two unproductive steps on top of it would be cutting correct
#: work. Six steps of headroom, at ~140 tokens a step.
BUDGET_TOKENS = 6 * 140

#: Two identical results could be a coincidence -- a search that legitimately
#: returns the same document twice. Three consecutive is a loop.
NO_PROGRESS_K = 3

#: The flaky tool's failure rate, from llmlab.tools.Sandbox.
FAILURE_RATE = 0.4

RULES = ("max_steps", "budget", "repeat_stop", "no_progress", "novelty",
         "composite_naive", "composite")

SHIPMENTS = ["TL-4471", "TL-9002", "TL-1150", "TL-3318", "TL-6207"]


# --- the behaviours ---------------------------------------------------------
#
# None of them can see the stopping rule, which is the point: a rule is only
# worth having if it works on an agent that is not cooperating with it.

def direct(step: int, goal: tuple) -> tuple:
    """Proposes the goal immediately."""
    return goal


def exploratory(step: int, goal: tuple) -> tuple:
    """Three different, valid, unproductive calls, then the goal."""
    warmup = [
        ("search", {"query": "redelivery attempts"}),
        ("search", {"query": "claims window"}),
        ("read_file", {"path": "notes/readme.txt"}),
    ]
    return warmup[step] if step < len(warmup) else goal


def duplicator(step: int, goal: tuple) -> tuple:
    """Repeats one call once before doing the right thing."""
    return ("search", {"query": "hold policy"}) if step < 2 else goal


def flaky_retry(step: int, goal: tuple) -> tuple:
    """Proposes the goal forever. Whether that is stubbornness or patience
    depends on something the agent cannot see."""
    return goal


def oscillator(step: int, goal: tuple) -> tuple:
    """Alternates between two calls and never converges."""
    pair = [("search", {"query": "hold policy"}), ("read_file", {"path": "notes/policy.txt"})]
    return pair[step % 2]


def runaway(step: int, goal: tuple) -> tuple:
    """Never repeats itself and never gets anywhere -- the case a repeated-call
    check cannot see, because every call really is new."""
    return ("search", {"query": f"depot exceptions bay {step}"})


BEHAVIOURS = {
    "direct": direct,
    "exploratory": exploratory,
    "duplicator": duplicator,
    "flaky_retry": flaky_retry,
    "oscillator": oscillator,
    "runaway": runaway,
}


def scenarios(n_instances: int = len(SHIPMENTS)) -> list[dict]:
    """One task per behaviour per instance: a goal call that ends the episode."""
    out = []
    for i in range(n_instances):
        shipment = SHIPMENTS[i % len(SHIPMENTS)]
        for name in BEHAVIOURS:
            goal = (("shipment_status", {"shipment": shipment}) if name == "flaky_retry"
                    else ("read_file", {"path": "notes/depot.txt"}))
            out.append({"instance": i, "behaviour": name, "goal": goal, "shipment": shipment})
    return out


# --- the loop ---------------------------------------------------------------

@dataclass
class Episode:
    solved: bool = False
    steps: int = 0
    tokens: int = 0
    stopped_by: str = "max_steps"
    executed: int = 0
    cached: int = 0
    novel: list[bool] = field(default_factory=list)


def run(task: dict, rule: str, retry_pays: bool, k: int = NO_PROGRESS_K) -> Episode:
    """One episode under one stopping rule.

    `retry_pays` decides what a retry means. False keeps `llmlab`'s
    deterministic sandbox, where the same lookup fails the same way forever, so
    retrying is always pointless. True gives each attempt an independent draw,
    which is what a real service does and what makes patience rational.
    """
    ep = Episode()
    behaviour = BEHAVIOURS[task["behaviour"]]
    box = Sandbox(seed=1 + task["instance"])
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    recent: list[str] = []

    budgeted = rule in ("budget", "composite_naive", "composite")
    stops_on_repeat = rule == "repeat_stop"
    watches_identical = rule == "no_progress"
    watches_novelty = rule in ("novelty", "composite_naive", "composite")
    caches_repeats = rule in ("composite_naive", "composite")
    # The naive composite skips execution on a repeat and records nothing,
    # so the step never reaches the novelty detector at all.
    counts_cache_hits = rule == "composite"

    for step in range(MAX_STEPS):
        tool, args = behaviour(step, task["goal"])
        key = f"{tool}:{json.dumps(args, sort_keys=True)}"

        if stops_on_repeat and key in seen_calls:
            ep.stopped_by = "repeat_stop"
            return ep

        ep.steps += 1
        ep.tokens += STEP_TOKENS

        if caches_repeats and key in seen_calls:
            # The proposal was still generated and still paid for; only the
            # execution is skipped. A cache saves tool time, not model calls --
            # and a cache hit is by definition a step that learned nothing, so
            # it counts as one. Leaving it out is what starved the detector.
            ep.cached += 1
            if counts_cache_hits:
                ep.novel.append(False)
                if watches_novelty and len(ep.novel) >= k and not any(ep.novel[-k:]):
                    ep.stopped_by = "no_novelty"
                    return ep
            continue

        if retry_pays:
            # An independent draw per attempt, without touching llmlab: the
            # flaky tool seeds on (seed, argument), so varying the seed varies
            # the attempt.
            box = Sandbox(seed=1000 * (1 + task["instance"]) + step)
        result = box.call(tool, args)
        seen_calls.add(key)
        ep.executed += 1
        ep.tokens += len(str(result)) // 4

        digest = json.dumps(result, sort_keys=True, default=str)
        is_novel = digest not in seen_results
        seen_results.add(digest)
        ep.novel.append(is_novel)
        recent.append(digest)

        if result["ok"] and (tool, args) == task["goal"]:
            ep.solved = True
            ep.stopped_by = "solved"
            return ep

        # "K identical results in a row" -- the definition that reads most
        # naturally and cannot see a two-cycle.
        if watches_identical and len(recent) >= k and len(set(recent[-k:])) == 1:
            ep.stopped_by = "no_progress"
            return ep

        # "K steps in a row that produced nothing not already seen."
        if watches_novelty and len(ep.novel) >= k and not any(ep.novel[-k:]):
            ep.stopped_by = "no_novelty"
            return ep

        if budgeted and ep.tokens >= BUDGET_TOKENS:
            ep.stopped_by = "budget"
            return ep

    return ep


def summarise(tasks: list[dict], rule: str, retry_pays: bool) -> dict:
    eps = [run(t, rule, retry_pays) for t in tasks]
    return {
        "solved": sum(1 for e in eps if e.solved),
        "tokens": sum(e.tokens for e in eps),
        "steps": sum(e.steps for e in eps),
        "wasted": sum(e.tokens for e in eps if not e.solved),
        "episodes": eps,
    }


def flaky_sweep(n: int = 4000, k: int = NO_PROGRESS_K) -> dict[str, float]:
    """How often a novelty detector cuts an episode a retry would have solved.

    The flaky task alone, over many independent instances, with retries that
    pay. The closed form is f**k -- K consecutive independent failures -- and
    this checks the loop actually behaves like the arithmetic says.
    """
    solved_budget = 0
    solved_novelty = 0
    for i in range(n):
        task = {"instance": 10_000 + i, "behaviour": "flaky_retry",
                "shipment": SHIPMENTS[i % len(SHIPMENTS)],
                "goal": ("shipment_status", {"shipment": SHIPMENTS[i % len(SHIPMENTS)]})}
        solved_budget += run(task, "budget", True).solved
        solved_novelty += run(task, "novelty", True, k=k).solved
    return {
        "n": n,
        "budget": solved_budget,
        "novelty": solved_novelty,
        "cut_pct": round(100 * (solved_budget - solved_novelty) / n, 1),
    }


def patience_for(f: float, tolerance: float, k_max: int = 20) -> int:
    """The smallest K whose false-cut rate is within tolerance.

    A novelty detector with patience K cuts a solvable episode when the tool
    fails K+1 times in a row, so the rate is f**(k+1). Solving for K is one
    line and removes a hyperparameter from the list of things to tune.
    """
    for k in range(1, k_max + 1):
        if f ** (k + 1) <= tolerance:
            return k
    return k_max


def compute() -> dict[str, float]:
    tasks = scenarios()
    out: dict[str, float] = {
        "n_episodes": len(tasks),
        "n_behaviours": len(BEHAVIOURS),
        "n_instances": len(SHIPMENTS),
        "max_steps": MAX_STEPS,
        "budget_tokens": BUDGET_TOKENS,
        "no_progress_k": NO_PROGRESS_K,
    }

    for condition, retry_pays in (("det", False), ("retry", True)):
        runs = {rule: summarise(tasks, rule, retry_pays) for rule in RULES}
        base = runs["max_steps"]
        for rule, r in runs.items():
            out[f"{condition}_{rule}_solved"] = r["solved"]
            out[f"{condition}_{rule}_tokens"] = r["tokens"]
            out[f"{condition}_{rule}_steps"] = r["steps"]
            out[f"{condition}_{rule}_wasted"] = r["wasted"]
            out[f"{condition}_{rule}_saving_pct"] = round(
                100 * (base["tokens"] - r["tokens"]) / base["tokens"], 1)
            out[f"{condition}_{rule}_lost"] = sum(
                1 for a, b in zip(base["episodes"], r["episodes"], strict=True)
                if a.solved and not b.solved)

    # Which rule stops which behaviour, and after how many steps.
    for name in BEHAVIOURS:
        group = [t for t in tasks if t["behaviour"] == name]
        for rule in RULES:
            eps = [run(t, rule, False) for t in group]
            out[f"det_{name}_{rule}_steps"] = round(sum(e.steps for e in eps) / len(eps), 1)
            out[f"det_{name}_{rule}_solved"] = sum(1 for e in eps if e.solved)

    # The three blind spots, as single numbers the lesson can quote.
    out["repeat_stop_runaway_steps"] = out["det_runaway_repeat_stop_steps"]
    out["repeat_stop_duplicator_solved"] = out["det_duplicator_repeat_stop_solved"]
    out["no_progress_oscillator_steps"] = out["det_oscillator_no_progress_steps"]
    out["novelty_oscillator_steps"] = out["det_oscillator_novelty_steps"]

    # What caching a repeated call does to a spinning agent, with and without
    # counting the cache hit as a step that learned nothing.
    out["naive_flaky_steps"] = out["det_flaky_retry_composite_naive_steps"]
    out["composite_flaky_steps"] = out["det_flaky_retry_composite_steps"]
    out["naive_oscillator_steps"] = out["det_oscillator_composite_naive_steps"]

    # The cost of novelty detection where a retry can pay.
    sweep = flaky_sweep()
    out["sweep_n"] = sweep["n"]
    out["sweep_budget_solved"] = sweep["budget"]
    out["sweep_novelty_solved"] = sweep["novelty"]
    out["sweep_cut_pct"] = sweep["cut_pct"]
    # K+1, not K: the first failure is the first time that result has been
    # seen, so it counts as novel and the run of K starts after it.
    out["sweep_predicted_pct"] = round(100 * FAILURE_RATE ** (NO_PROGRESS_K + 1), 1)
    for k in (2, 4):
        s = flaky_sweep(k=k)
        out[f"sweep_cut_pct_k{k}"] = s["cut_pct"]
        out[f"sweep_predicted_pct_k{k}"] = round(100 * FAILURE_RATE ** (k + 1), 1)

    # K is a quantity you compute. These are the worked examples the lesson
    # quotes, computed here so gate 18 checks them rather than my arithmetic.
    out["patience_f40_tol1pct"] = patience_for(0.4, 0.01)
    out["patience_f40_tol2pct"] = patience_for(0.4, 0.02)
    out["patience_f10_tol1pct"] = patience_for(0.1, 0.01)
    out["cut_pct_f40_k4"] = round(100 * 0.4 ** 5, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"{int(r['n_episodes'])} episodes: {int(r['n_behaviours'])} behaviours "
          f"x {int(r['n_instances'])} instances")
    print(f"max_steps={int(r['max_steps'])}  budget={int(r['budget_tokens'])} tokens  "
          f"K={int(r['no_progress_k'])}\n")

    for condition, label in (("det", "deterministic sandbox: a retry never pays"),
                             ("retry", "independent draw per attempt: a retry can pay")):
        print(label)
        print(f"  {'rule':<14}{'solved':>7}{'tokens':>9}{'saved':>8}{'wasted':>9}{'lost':>6}")
        for rule in RULES:
            print(f"  {rule:<14}"
                  f"{int(r[f'{condition}_{rule}_solved']):>7}"
                  f"{int(r[f'{condition}_{rule}_tokens']):>9}"
                  f"{r[f'{condition}_{rule}_saving_pct']:>7.1f}%"
                  f"{int(r[f'{condition}_{rule}_wasted']):>9}"
                  f"{int(r[f'{condition}_{rule}_lost']):>6}")
        print()

    print("mean steps per episode, by behaviour (deterministic sandbox)")
    print(f"  {'behaviour':<14}" + "".join(f"{rule:>17}" for rule in RULES))
    for name in BEHAVIOURS:
        row = "".join(f"{r[f'det_{name}_{rule}_steps']:>17.1f}" for rule in RULES)
        print(f"  {name:<14}{row}")

    print("\nthe three blind spots")
    print(f"  repeat_stop runs runaway to {r['repeat_stop_runaway_steps']:.0f} steps "
          f"(every call is distinct) and solves "
          f"{int(r['repeat_stop_duplicator_solved'])}/{int(r['n_instances'])} duplicator")
    print(f"  no_progress runs oscillator to {r['no_progress_oscillator_steps']:.0f} steps; "
          f"novelty stops it at {r['novelty_oscillator_steps']:.0f}")
    print(f"  caching a repeat without recording it runs flaky_retry to "
          f"{r['naive_flaky_steps']:.1f} steps; counting the cache hit as a step "
          f"that learned nothing brings it back to {r['composite_flaky_steps']:.1f}")

    print(f"\nwhere novelty detection is not free ({int(r['sweep_n'])} flaky episodes, "
          f"retries pay)")
    for k, lbl in ((2, "K=2"), (int(r["no_progress_k"]), "K=3"), (4, "K=4")):
        key = "sweep_cut_pct" if k == int(r["no_progress_k"]) else f"sweep_cut_pct_k{k}"
        pred = ("sweep_predicted_pct" if k == int(r["no_progress_k"])
                else f"sweep_predicted_pct_k{k}")
        print(f"  {lbl}: cuts {r[key]:.1f}% of solvable episodes  "
              f"(f**K predicts {r[pred]:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
