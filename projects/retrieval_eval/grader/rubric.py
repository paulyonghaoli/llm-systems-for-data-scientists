"""The published rubric. Nothing here is hidden from the learner.

Six criteria, 100 points. Criteria B and D carry half the marks between them
and neither is about metric arithmetic, because getting the arithmetic right is
the part everyone does. Keeping the per-query values is what makes every later
question answerable, and choosing the test from the metric's *type* is what
stops a reranking improvement being reported as noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from grader import reference
from grader.scenarios import benchmark, degenerate_cases

CRITERIA = [
    ("A", "metric arithmetic", 25,
     "all five metrics match the reference, per query, on every benchmark"),
    ("B", "per-query retained", 20,
     "per-query scores are returned, not just means — nothing downstream "
     "works without them"),
    ("C", "precision ceiling", 15,
     "reports the highest precision@k the gold labels permit"),
    ("D", "right test per metric", 25,
     "McNemar for binary recall, a bootstrap interval for continuous "
     "metrics, and a verdict matching the reference's"),
    ("E", "breakdown by phenomenon", 10,
     "means reported per query type, since aggregate agreement hides "
     "per-type disagreement"),
    ("F", "degenerate cases", 5,
     "no queries, no gold, nothing retrieved, fewer results than k, more "
     "gold than k, everything correct"),
]

TOL = 1e-9


def _close(x: float, y: float) -> bool:
    if isinstance(x, float) and math.isnan(x):
        return isinstance(y, float) and math.isnan(y)
    return abs(float(x) - float(y)) < 1e-6


def _grade_one(evaluate: Callable, compare: Callable, bench: dict) -> dict:
    """Run both student functions over one benchmark and report what held."""
    out: dict = {}
    k, qrels = bench["k"], bench["qrels"]

    got = evaluate(bench["runs_a"], qrels, k, bench["phenomena"])
    ref = reference.evaluate(bench["runs_a"], qrels, k, bench["phenomena"])

    if not isinstance(got, dict):
        return {"fatal": f"evaluate returned {type(got).__name__}, expected dict"}

    # B — per-query values present and complete.
    pq = got.get("per_query")
    out["B"] = (isinstance(pq, dict)
                and set(pq) == set(ref["per_query"])
                and all(isinstance(v, dict) and set(v) >= set(reference.METRICS)
                        for v in pq.values()))

    # A — the arithmetic, checked per query rather than on the means, because
    # two different per-query vectors can share a mean.
    if out["B"]:
        out["A"] = all(
            _close(pq[q][m], ref["per_query"][q][m])
            for q in ref["per_query"] for m in reference.METRICS
        )
    else:
        means = got.get("means", {})
        out["A"] = isinstance(means, dict) and all(
            _close(means.get(m, -1), ref["means"][m]) for m in reference.METRICS
        )

    # C — the ceiling.
    out["C"] = _close(got.get("precision_ceiling", -1), ref["precision_ceiling"])

    # E — per-phenomenon means.
    bp = got.get("by_phenomenon", {})
    out["E"] = (isinstance(bp, dict)
                and set(bp) == set(ref["by_phenomenon"])
                and all(_close(bp[name].get(m, -1), ref["by_phenomenon"][name][m])
                        for name in ref["by_phenomenon"] for m in reference.METRICS))

    # D — the comparison, on the metric where the two systems differ and on
    # the one where they do not.
    got_b = evaluate(bench["runs_b"], qrels, k, bench["phenomena"])
    ref_b = reference.evaluate(bench["runs_b"], qrels, k, bench["phenomena"])
    d_ok = True
    details = []
    for metric in ("recall", "mrr"):
        try:
            c = compare(got.get("per_query", {}), got_b.get("per_query", {}), metric)
            r = reference.compare(ref["per_query"], ref_b["per_query"], metric)
        except Exception as e:  # noqa: BLE001
            d_ok = False
            details.append(f"compare({metric}) raised {type(e).__name__}: {e}")
            continue
        if not isinstance(c, dict):
            d_ok = False
            details.append(f"compare({metric}) returned {type(c).__name__}")
            continue
        if c.get("test") != r["test"]:
            d_ok = False
            details.append(f"{metric}: test '{c.get('test')}', expected '{r['test']}'")
        if bool(c.get("significant")) != r["significant"]:
            d_ok = False
            details.append(f"{metric}: significant={c.get('significant')}, "
                           f"reference says {r['significant']}")
        if not r["test"] == "mcnemar" and "ci_low" not in c:
            d_ok = False
            details.append(f"{metric}: a bootstrap comparison must report an interval")
    out["D"] = d_ok
    out["_details"] = details
    return out


def grade(evaluate: Callable, compare: Callable, seed: int,
          n_benchmarks: int = 5) -> dict:
    per: dict[str, list[bool]] = {key: [] for key, *_ in CRITERIA}
    failures: list[str] = []

    for i in range(n_benchmarks):
        bench = benchmark(seed + i)
        try:
            got = _grade_one(evaluate, compare, bench)
        except Exception as e:  # noqa: BLE001
            for key in "ABCDE":
                per[key].append(False)
            failures.append(f"benchmark {seed + i}: raised {type(e).__name__}: {e}")
            continue

        if "fatal" in got:
            for key in "ABCDE":
                per[key].append(False)
            failures.append(f"benchmark {seed + i}: {got['fatal']}")
            continue

        for key in "ABCDE":
            per[key].append(bool(got[key]))
            if not got[key]:
                detail = {
                    "A": "per-query metric values differ from the reference",
                    "B": "per_query missing, incomplete, or not a dict of dicts",
                    "C": "precision_ceiling wrong or absent",
                    "D": "; ".join(got.get("_details") or ["comparison wrong"]),
                    "E": "by_phenomenon wrong or absent",
                }[key]
                failures.append(f"benchmark {seed + i}: {key} — {detail}")

    for case in degenerate_cases():
        try:
            got = evaluate(case["runs"], case["qrels"], 10, case["phenomena"])
            ref = reference.evaluate(case["runs"], case["qrels"], 10, case["phenomena"])
            ok = (isinstance(got, dict)
                  and _close(got.get("precision_ceiling", -1), ref["precision_ceiling"])
                  and set(got.get("per_query", {})) == set(ref["per_query"]))
            if not ok:
                failures.append(f"degenerate '{case['name']}': wrong output shape or ceiling")
        except Exception as e:  # noqa: BLE001
            ok = False
            failures.append(f"degenerate '{case['name']}': raised {type(e).__name__}: {e}")
        per["F"].append(ok)

    breakdown, total = [], 0.0
    for key, name, points, _desc in CRITERIA:
        results = per[key]
        frac = sum(results) / len(results) if results else 0.0
        earned = round(points * frac, 1)
        total += earned
        breakdown.append({"key": key, "name": name, "points": points,
                          "earned": earned, "passed": sum(results), "of": len(results)})

    return {"total": round(total, 1), "breakdown": breakdown, "failures": failures}
