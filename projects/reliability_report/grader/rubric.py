"""The published rubric. Nothing is hidden from the learner.

Five criteria, 100 points, scored as the fraction of scenarios satisfying
each. Criterion E is the one that matters most and is failed most often: a
report that always produces a number is not more useful than one that
sometimes refuses, it is less.
"""

from __future__ import annotations

from collections.abc import Callable

from grader import reference
from grader.scenarios import edge_scenarios, scenario

CRITERIA = [
    ("A", "counts and cohorts", 20,
     "n and the number of distinct (version, params, prompt) cohorts"),
    ("B", "success rate and interval", 20,
     "success_rate and a Wilson 95% interval, or None where not permitted"),
    ("C", "latency percentiles", 20,
     "nearest-rank p50 and p95 over every record"),
    ("D", "cost accounting", 20,
     "cost_units at the given price ratio, and cost per SUCCESS"),
    ("E", "warnings and refusal", 20,
     "the right warnings, and no summary rate across model versions"),
]

Report = Callable[[list[dict], float], dict]
TOL = 1e-6


def _num_close(got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    try:
        return abs(float(got) - float(want)) <= TOL * max(1.0, abs(float(want)))
    except (TypeError, ValueError):
        return False


def _check_one(fn: Report, records: list[dict], price_ratio: float) -> dict[str, bool]:
    got = fn(records, price_ratio)
    want = reference.report(records, price_ratio)

    ci_ok = False
    if want["ci"] is None:
        ci_ok = got.get("ci") is None
    else:
        g = got.get("ci")
        ci_ok = (
            isinstance(g, (tuple, list))
            and len(g) == 2
            and _num_close(g[0], want["ci"][0])
            and _num_close(g[1], want["ci"][1])
        )

    return {
        "A": got.get("n") == want["n"] and got.get("cohorts") == want["cohorts"],
        "B": _num_close(got.get("success_rate"), want["success_rate"]) and ci_ok,
        "C": (_num_close(got.get("p50_ms"), want["p50_ms"])
              and _num_close(got.get("p95_ms"), want["p95_ms"])),
        "D": (_num_close(got.get("cost_units"), want["cost_units"])
              and _num_close(got.get("cost_per_success"), want["cost_per_success"])),
        "E": (sorted(got.get("warnings") or []) == want["warnings"]
              and (want["success_rate"] is not None or got.get("success_rate") is None)),
    }


def grade(fn: Report, seed: int, n_scenarios: int = 12) -> dict:
    per_criterion: dict[str, list[bool]] = {k: [] for k, *_ in CRITERIA}
    failures: list[str] = []
    labels = {k: name for k, name, *_ in CRITERIA}

    cases = [(f"seed {seed + i}", scenario(seed + i)) for i in range(n_scenarios)]
    cases += [(f"edge '{e['name']}'", e) for e in edge_scenarios()]

    for label, case in cases:
        try:
            res = _check_one(fn, case["records"], case["price_ratio"])
        except Exception as e:  # noqa: BLE001
            for key in per_criterion:
                per_criterion[key].append(False)
            failures.append(f"{label}: raised {type(e).__name__}: {e}")
            continue
        for key, ok in res.items():
            per_criterion[key].append(ok)
            if not ok:
                failures.append(f"{label}: {key} ({labels[key]}) failed")

    breakdown = []
    total = 0.0
    for key, name, points, _desc in CRITERIA:
        results = per_criterion[key]
        frac = sum(results) / len(results) if results else 0.0
        earned = round(points * frac, 1)
        total += earned
        breakdown.append({
            "key": key, "name": name, "points": points,
            "earned": earned, "passed": sum(results), "of": len(results),
        })

    return {"total": round(total, 1), "breakdown": breakdown, "failures": failures}
