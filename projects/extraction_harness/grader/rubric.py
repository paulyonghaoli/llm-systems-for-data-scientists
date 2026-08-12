"""The published rubric. Nothing here is hidden from the learner.

Six criteria, 100 points. Criterion B is a safety property rather than a
quality one: a harness that emits a record violating the schema has done
something worse than failing, because the bad record now flows downstream
wearing the same clothes as a good one.
"""

from __future__ import annotations

from collections.abc import Callable

from grader import reference
from grader.scenarios import batch, degenerate_batches

CRITERIA = [
    ("A", "recovery", 25,
     "recovers at least as many valid records as the reference"),
    ("B", "no false accepts", 25,
     "never returns a record that fails the schema"),
    ("C", "truncation caught", 20,
     "every output with a non-stop finish reason is reported as truncated, "
     "even when it parses and validates"),
    ("D", "cost discipline", 15,
     "no more expensive repair calls than the reference makes"),
    ("E", "nothing dropped", 10,
     "one result per input, and anything not recovered carries a reason"),
    ("F", "degenerate cases", 5,
     "empty batch, all-truncated batch, a truncated output that looks valid, "
     "a content-filter refusal"),
]

Harness = Callable[..., list]


class _Counter:
    """Stands in for a model-based repair call, and counts how often it runs.

    Returns None: the point of the exercise is the cost of reaching for it,
    not what it would have returned. A harness that calls it for everything
    scores the same recovery and loses criterion D.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> None:
        self.calls += 1
        return None


def _schema_ok(obj: object) -> bool:
    return reference.schema_ok(obj)


def _check(harness: Harness, outputs: list[dict]) -> dict:
    counter = _Counter()
    results = harness(outputs, counter)

    if not isinstance(results, list) or len(results) != len(outputs):
        return {"fatal": f"expected {len(outputs)} results, got "
                         f"{len(results) if isinstance(results, list) else type(results).__name__}"}

    recovered = 0
    false_accepts = 0
    missing_reason = 0
    truncated_missed = 0

    for item, res in zip(outputs, results, strict=True):
        status = res.get("status")
        record = res.get("record")

        if status in ("ok", "repaired"):
            recovered += 1
            if not _schema_ok(record):
                false_accepts += 1
            if item["finish_reason"] != "stop":
                truncated_missed += 1
        else:
            if not res.get("reason"):
                missing_reason += 1
            if item["finish_reason"] != "stop" and status != "truncated":
                truncated_missed += 1

    return {
        "recovered": recovered,
        "false_accepts": false_accepts,
        "missing_reason": missing_reason,
        "truncated_missed": truncated_missed,
        "repair_calls": counter.calls,
    }


def grade(harness: Harness, seed: int, n_batches: int = 6) -> dict:
    per: dict[str, list[bool]] = {k: [] for k, *_ in CRITERIA}
    failures: list[str] = []

    for i in range(n_batches):
        outputs = batch(seed + i)
        got = _check(harness, outputs)
        ref = _check(reference.process, outputs)

        if "fatal" in got:
            for key in "ABCDE":
                per[key].append(False)
            failures.append(f"batch {seed + i}: {got['fatal']}")
            continue

        checks = {
            "A": got["recovered"] >= ref["recovered"],
            "B": got["false_accepts"] == 0,
            "C": got["truncated_missed"] == 0,
            "D": got["repair_calls"] <= ref["repair_calls"],
            "E": got["missing_reason"] == 0,
        }
        for key, ok in checks.items():
            per[key].append(ok)
            if not ok:
                detail = {
                    "A": f"recovered {got['recovered']} vs reference {ref['recovered']}",
                    "B": f"{got['false_accepts']} record(s) returned that fail the schema",
                    "C": f"{got['truncated_missed']} truncated output(s) not reported as truncated",
                    "D": (f"{got['repair_calls']} expensive repairs vs reference "
                          f"{ref['repair_calls']}"),
                    "E": f"{got['missing_reason']} non-recovered result(s) with no reason",
                }[key]
                failures.append(f"batch {seed + i}: {key} — {detail}")

    for case in degenerate_batches():
        try:
            got = _check(harness, case["outputs"])
            ok = "fatal" not in got and got["false_accepts"] == 0 and got["truncated_missed"] == 0
            if not ok:
                failures.append(f"degenerate '{case['name']}': "
                                f"{got.get('fatal') or 'false accept or missed truncation'}")
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
