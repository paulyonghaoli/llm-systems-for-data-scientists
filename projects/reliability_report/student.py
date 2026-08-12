"""Mini-project 0 - the reliability report.

Turn a raw run log into the numbers that decide whether a system is
shippable, and refuse to produce the ones the log cannot support.

    cd projects/reliability_report
    python -m grader --seed 1

The full specification and the rubric are in README.md. The starter below is
the report everybody writes first: every field populated, every question
answered, no warnings, no refusals. Run the grader against it before you
change anything - the breakdown is the lesson.
"""

from __future__ import annotations

import json
import statistics


def params_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def report(records: list[dict], price_ratio: float) -> dict:
    """Summarise a run log. See README.md for the required keys."""
    n = len(records)
    n_ok = sum(1 for r in records if r["outcome"] == "ok")
    rate = n_ok / n if n else 0.0
    latencies = [r["latency_ms"] for r in records]
    mean_latency = statistics.fmean(latencies) if latencies else 0.0
    sd = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    cost = sum(r["in_tokens"] + r["out_tokens"] for r in records)

    return {
        "n": n,
        "cohorts": len({r["prompt_hash"] for r in records}),
        "success_rate": rate,
        "ci": (rate - 1.96 * (rate * (1 - rate) / n) ** 0.5 if n else 0.0,
               rate + 1.96 * (rate * (1 - rate) / n) ** 0.5 if n else 1.0),
        "refusal_rate": sum(1 for r in records if r["outcome"] == "refused") / n if n else 0.0,
        "p50_ms": mean_latency,
        "p95_ms": mean_latency + 2 * sd,
        "cost_units": cost,
        "cost_per_success": cost / n if n else 0.0,
        "warnings": [],
    }
