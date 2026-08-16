"""What forgetting costs, and when compacting is worth most.

    python experiments/agent_memory.py
    python experiments/agent_memory.py --json

An agent re-reads its whole history before every model call, so an episode of
n steps pays for its context n times and the bill grows with the square of the
length. Something has to be dropped. Every memory design is a retention policy
over observations, and the question it has to answer is not "how small can I
make this" but "will the fact a later step needs still be here".

The observations are real: each is the envelope from an actual `llmlab.tools`
call, so the sizes are measured rather than invented. What is authored is which
facts a later step needs, and the need-distance is swept rather than picked, so
the result is a curve a reader can locate their own workload on.

Retention is measured against the EVIDENCE, not the string. A fact belongs to
the observation that established it, and it counts as retained only if that
observation's evidence for it survives. Checking whether the text appears
anywhere in context is the obvious implementation and it is wrong: a repeated
search re-establishes the same sentence later, which made a six-step window
score 0.70 on facts fifteen steps old.

Three results:

  * **Compaction value peaks in the middle of an episode, and the peak is not
    shallow.** Value is (tokens accumulated so far) x (steps remaining), a
    rising term times a falling one. I predicted earlier-is-always-better and
    the measurement says otherwise; the closed form now agrees with the loop
    exactly at every step.

  * **Recency's recall is a cliff, not a slope.** 1.000 at every distance
    inside the window and 0.000 at every distance outside it, with nothing in
    between to warn you -- and the facts most likely to matter are the ones
    established earliest, which is precisely where the cliff is.

  * **The gross saving is not the net saving, and one policy goes negative.**
    Truncating old observations appears to save 29.3%; counting the re-fetches
    it forces, it costs 4.7% MORE than keeping everything. Recency's 50.0%
    becomes 6.1%. A dropped fact has to be fetched again, and the fresh
    observation is then carried for the rest of the episode.

  * **A keyed store and a window fail on disjoint fact classes.** Structured
    extraction keeps what arrived with a key, forever, and loses every fact
    stated in prose; a window keeps both kinds and only inside itself. The
    composite closes the keyed half exactly and leaves prose recall a cliff, so
    nothing here short of keeping everything has full recall at every distance.

Two limits worth stating rather than burying. The re-fetch accounting fixes a
single need-distance, so it examines the facts established in the first half of
the episode; it is applied identically to every policy, so the comparison is
fair, and the absolute counts are not a population estimate.

And the lossy-compaction policy truncates old observations instead of
summarising them with a model, because no model runs in this harness. Real
summarisation retains far more per token. Nothing in the argument depends on
how good the compactor is -- what matters is that compaction is lossy, that
the loss is silent, and that its value is set by timing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llmlab.tools import Sandbox  # noqa: E402

#: Roughly four characters per token, used only to convert one measured length
#: into another.
CHARS_PER_TOKEN = 4

#: The sliding window most frameworks ship by default.
WINDOW = 6

#: How much of an old observation a truncating compactor keeps.
TRUNCATE_CHARS = 40

#: How far back the facts a step needs were established, for the cost table.
NEED_DISTANCE = 10

DEPOTS = ["north-east", "south-west", "central", "north-west", "south-east"]
SHIPMENTS = [f"TL-{4471 + 37 * i}" for i in range(5)]

#: Distinct content per file, so a fact belongs to one observation rather than
#: being re-established by the next one.
FILES = {
    "notes/policy.txt": "A hold lasts up to 48 hours",
    "notes/readme.txt": "Claims must be submitted within 28 days",
    "notes/hazmat.txt": "Lithium cells need a written declaration",
    "notes/bays.txt": "Bay 4 handles exceptions at every depot",
    "notes/redelivery.txt": "Redelivery is attempted three times",
}
FILES.update({f"notes/depot-{d}.txt": f"The {d} depot opens at 0600"
              for d in DEPOTS})

QUERIES = ["redelivery attempts", "claims window", "lithium declaration",
           "address verification hold", "consecutive working days"]


@dataclass
class Fact:
    """Something a later step might need, and the evidence that establishes it.

    `key` is set when the fact arrived with one -- a field name in a structured
    result -- which is exactly the condition under which an extractor can
    capture it. A fact stated in prose has no key, so no extractor knows to
    look for it unless somebody anticipated it.
    """

    origin: int
    needle: str
    key: str | None = None

    @property
    def kind(self) -> str:
        return "keyed" if self.key else "prose"


@dataclass
class Observation:
    text: str
    facts: list[Fact] = field(default_factory=list)

    def tokens(self) -> int:
        return max(1, len(self.text) // CHARS_PER_TOKEN)


def episode(n_steps: int = 20, seed: int = 1) -> list[Observation]:
    """A real trajectory: every observation is an actual tool envelope."""
    box = Sandbox(seed=seed, failure_rate=0.0, files=dict(FILES),
                  depots=dict(zip(SHIPMENTS, DEPOTS, strict=True)))
    paths = sorted(FILES)
    plan = []
    for i in range(n_steps):
        which = i % 4
        if which == 0:
            plan.append(("shipment_status", {"shipment": SHIPMENTS[i // 4 % len(SHIPMENTS)]}))
        elif which == 1:
            plan.append(("read_file", {"path": paths[i % len(paths)]}))
        elif which == 2:
            plan.append(("search", {"query": QUERIES[i // 4 % len(QUERIES)], "k": 1}))
        else:
            plan.append(("calculator", {"expression": f"{48 + i} / {2 + i % 5}"}))

    out: list[Observation] = []
    for i, (tool, args) in enumerate(plan):
        res = box.call(tool, args)
        value = res.get("value")
        text = f"{tool}({json.dumps(args, sort_keys=True)}) -> {json.dumps(value, default=str)}"
        facts: list[Fact] = []
        if isinstance(value, dict):
            for k, v in value.items():
                # The needle is how this value appears in the observation, so
                # retention is tested against the evidence rather than a guess.
                facts.append(Fact(origin=i, needle=json.dumps(str(v))[1:-1],
                                  key=f"{k}:{args.get('shipment', '')}"))
        elif isinstance(value, str):
            facts.append(Fact(origin=i, needle=value))
        elif isinstance(value, list):
            facts.extend(Fact(origin=i, needle=d["text"].split(".")[0].strip())
                         for d in value if d.get("text"))
        out.append(Observation(text=text, facts=facts))
    return out


# --- retention policies -----------------------------------------------------

@dataclass
class Retained:
    """What the agent actually has in context, tagged with where it came from.

    Tagging is what makes retention measurable. Without the origin index a
    fact re-established by a later observation looks like a fact the policy
    kept, and every windowed policy scores far better than it deserves.
    """

    texts: dict[int, str] = field(default_factory=dict)   # origin -> retained text
    keyed: dict[str, str] = field(default_factory=dict)

    def tokens(self) -> int:
        body = sum(max(1, len(t) // CHARS_PER_TOKEN) for t in self.texts.values())
        keys = sum(max(1, (len(k) + len(v)) // CHARS_PER_TOKEN)
                   for k, v in self.keyed.items())
        return body + keys

    def holds(self, fact: Fact) -> bool:
        if fact.key is not None and fact.key in self.keyed:
            return True
        return fact.needle in self.texts.get(fact.origin, "")


def _window_texts(obs: list[Observation], step: int, window: int) -> dict[int, str]:
    return {i: obs[i].text for i in range(max(0, step - window), step)}


def keep_all(obs: list[Observation], step: int) -> Retained:
    """No memory management. Perfect recall, quadratic bill."""
    return Retained(texts={i: obs[i].text for i in range(step)})


def recency(obs: list[Observation], step: int, window: int = WINDOW) -> Retained:
    """The last `window` observations, and nothing else."""
    return Retained(texts=_window_texts(obs, step, window))


def truncated(obs: list[Observation], step: int, window: int = WINDOW,
              prefix: int = TRUNCATE_CHARS) -> Retained:
    """Everything, but anything older than the window is cut down."""
    texts = {i: obs[i].text[:prefix] for i in range(max(0, step - window))}
    texts.update(_window_texts(obs, step, window))
    return Retained(texts=texts)


def _extract(obs: list[Observation], step: int) -> dict[str, str]:
    store: dict[str, str] = {}
    for i in range(step):
        for f in obs[i].facts:
            if f.key is not None:
                store[f.key] = f.needle
    return store


def keyed_only(obs: list[Observation], step: int) -> Retained:
    """Extract what has a key, keep it forever, drop the prose."""
    return Retained(keyed=_extract(obs, step))


def keyed_plus_recency(obs: list[Observation], step: int,
                       window: int = WINDOW) -> Retained:
    """The composite most frameworks ship."""
    return Retained(texts=_window_texts(obs, step, window), keyed=_extract(obs, step))


POLICIES = {
    "keep_all": keep_all,
    "recency": recency,
    "truncated": truncated,
    "keyed": keyed_only,
    "keyed+recency": keyed_plus_recency,
}


# --- recall as a function of how far back the fact was established ----------

def recall_at_distance(obs: list[Observation], policy, distance: int) -> dict[str, float]:
    """Of the facts established `distance` steps ago, how many are still held?"""
    hits: dict[str, int] = {"keyed": 0, "prose": 0}
    total: dict[str, int] = {"keyed": 0, "prose": 0}
    for i in range(len(obs) - distance):
        held = policy(obs, i + distance + 1)
        for f in obs[i].facts:
            total[f.kind] += 1
            hits[f.kind] += held.holds(f)
    return {k: (hits[k] / total[k] if total[k] else float("nan")) for k in hits}


# --- the cost of carrying, and of getting a dropped fact back ---------------

def carried_tokens(obs: list[Observation], policy) -> int:
    """Context tokens paid across the episode: the sum over every step."""
    return sum(policy(obs, step).tokens() for step in range(1, len(obs) + 1))


def net_cost(obs: list[Observation], policy, distance: int = NEED_DISTANCE) -> dict:
    """Carrying cost, plus the re-fetches that missing facts force.

    Charged per missing FACT, not per observation: a policy that drops twenty
    facts has to get twenty of them back. The re-fetched observation is then
    carried for the rest of the episode, which is the part that makes a gross
    saving and a net saving diverge.
    """
    carried = carried_tokens(obs, policy)
    n = len(obs)
    refetch_tokens, refetches = 0, 0
    for i in range(n - distance):
        step = i + distance + 1
        held = policy(obs, step)
        for f in obs[i].facts:
            if not held.holds(f):
                refetches += 1
                refetch_tokens += obs[i].tokens() * (n - step + 1)
    return {"carried": carried, "refetch_tokens": refetch_tokens,
            "refetches": refetches, "total": carried + refetch_tokens}


# --- when to compact --------------------------------------------------------

def compaction_value(obs: list[Observation], at_step: int,
                     prefix: int = TRUNCATE_CHARS) -> int:
    """Tokens saved by truncating everything before `at_step`, once.

    (tokens accumulated by then) x (steps remaining to re-read them). A rising
    term times a falling one, so the value peaks in the middle rather than at
    the start -- which is the opposite of what I expected before measuring it.
    """
    before = sum(o.tokens() for o in obs[:at_step])
    after = sum(max(1, len(o.text[:prefix]) // CHARS_PER_TOKEN) for o in obs[:at_step])
    return (before - after) * (len(obs) - at_step)


def measured_compaction_value(obs: list[Observation], at_step: int,
                              prefix: int = TRUNCATE_CHARS) -> int:
    """The same number by running both episodes and subtracting."""
    def plain(step: int) -> int:
        return Retained({i: obs[i].text for i in range(step)}).tokens()

    def compacted(step: int) -> int:
        if step <= at_step:
            return plain(step)
        texts = {i: obs[i].text[:prefix] for i in range(at_step)}
        texts.update({i: obs[i].text for i in range(at_step, step)})
        return Retained(texts).tokens()

    return (sum(plain(s) for s in range(1, len(obs) + 1))
            - sum(compacted(s) for s in range(1, len(obs) + 1)))


def compute() -> dict[str, float]:
    obs = episode()
    n = len(obs)
    distances = (1, 3, 5, 8, 12)
    out: dict[str, float] = {
        "n_steps": n,
        "window": WINDOW,
        "truncate_chars": TRUNCATE_CHARS,
        "need_distance": NEED_DISTANCE,
        "episode_tokens": sum(o.tokens() for o in obs),
        "n_keyed_facts": sum(1 for o in obs for f in o.facts if f.kind == "keyed"),
        "n_prose_facts": sum(1 for o in obs for f in o.facts if f.kind == "prose"),
    }

    baseline = carried_tokens(obs, keep_all)
    out["keep_all_carried"] = baseline
    out["carry_multiple"] = round(baseline / out["episode_tokens"], 2)

    # An invariant, not a result: a policy that keeps everything must lose
    # nothing. It failed on the first version of this file, which is how the
    # evidence-versus-string bug was found.
    assert net_cost(obs, keep_all)["refetches"] == 0, "keep_all dropped a fact"
    for d in distances:
        r = recall_at_distance(obs, keep_all, d)
        assert r["keyed"] == 1.0 and r["prose"] == 1.0, f"keep_all imperfect at d={d}"

    for name, policy in POLICIES.items():
        out[f"{name}_carried"] = carried_tokens(obs, policy)
        out[f"{name}_gross_pct"] = round(
            100 * (baseline - out[f"{name}_carried"]) / baseline, 1)
        for d in distances:
            r = recall_at_distance(obs, policy, d)
            out[f"{name}_keyed_d{d}"] = round(r["keyed"], 3)
            out[f"{name}_prose_d{d}"] = round(r["prose"], 3)
        net = net_cost(obs, policy)
        out[f"{name}_refetches"] = net["refetches"]
        out[f"{name}_net"] = net["total"]
        out[f"{name}_net_pct"] = round(100 * (baseline - net["total"]) / baseline, 1)
        out[f"{name}_given_back_pct"] = (
            round(100 * net["refetch_tokens"] / (baseline - out[f"{name}_carried"]), 1)
            if baseline > out[f"{name}_carried"] else 0.0)

    # When to compact. The closed form and the loop must agree at every step.
    values = {}
    for k in range(1, n):
        values[k] = compaction_value(obs, k)
        assert values[k] == measured_compaction_value(obs, k), f"mismatch at {k}"
    best = max(values, key=lambda k: values[k])
    out["compact_best_step"] = best
    out["compact_best_value"] = values[best]
    out["compact_best_frac"] = round(best / n, 2)
    for k in (3, 5, 10, 15, 18):
        out[f"compact_at_{k}"] = values[k]
    out["compact_peak_over_early"] = round(values[best] / values[3], 2)
    out["compact_peak_over_late"] = round(values[best] / values[18], 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    distances = (1, 3, 5, 8, 12)
    print(f"{int(r['n_steps'])}-step episode: {int(r['episode_tokens'])} tokens of "
          f"observations, {int(r['n_keyed_facts'])} keyed facts, "
          f"{int(r['n_prose_facts'])} prose facts")
    print(f"carrying all of it costs {int(r['keep_all_carried'])} tokens -- "
          f"{r['carry_multiple']}x the text itself, because context is re-read "
          f"every step\n")

    print(f"{'policy':<15}{'carried':>9}{'gross':>8}{'refetch':>9}{'net':>9}{'net':>8}"
          f"{'given back':>12}")
    for name in POLICIES:
        print(f"{name:<15}{int(r[f'{name}_carried']):>9}{r[f'{name}_gross_pct']:>7.1f}%"
              f"{int(r[f'{name}_refetches']):>9}{int(r[f'{name}_net']):>9}"
              f"{r[f'{name}_net_pct']:>7.1f}%{r[f'{name}_given_back_pct']:>11.1f}%")

    for kind in ("keyed", "prose"):
        print(f"\nrecall of {kind} facts, by how many steps ago they were established")
        print(f"  {'policy':<15}" + "".join(f"{'d=' + str(d):>9}" for d in distances))
        for name in POLICIES:
            print(f"  {name:<15}" +
                  "".join(f"{r[f'{name}_{kind}_d{d}']:>9.3f}" for d in distances))

    print("\nthe same compaction, applied at different steps")
    print(f"  {'at step':<10}{'tokens saved':>14}")
    for k in (3, 5, 10, 15, 18):
        print(f"  {k:<10}{int(r[f'compact_at_{k}']):>14}")
    print(f"\n  peak at step {int(r['compact_best_step'])} of {int(r['n_steps'])} "
          f"({r['compact_best_frac']} of the way through), worth "
          f"{int(r['compact_best_value'])} tokens")
    print(f"  {r['compact_peak_over_early']}x compacting at step 3, "
          f"{r['compact_peak_over_late']}x compacting at step 18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
