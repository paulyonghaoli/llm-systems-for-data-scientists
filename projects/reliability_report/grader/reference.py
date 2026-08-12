"""Reference implementation of the reliability report.

The interesting part is not the arithmetic; it is the refusal. A log that
spans two model versions cannot be summarised by one success rate, and the
correct behaviour is to say so rather than to produce a number that reads like
a measurement and is not one.
"""

from __future__ import annotations

import json
import math

Z95 = 1.959963984540054
SMALL_SAMPLE_BELOW = 250  # lesson 0.3: the first n on the grid within +/-5 points


def params_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def cohort_key(r: dict) -> tuple:
    return (r["model_version"], params_signature(r["params"]), r["prompt_hash"])


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def report(records: list[dict], price_ratio: float) -> dict:
    n = len(records)
    versions = {r["model_version"] for r in records}
    param_sets = {params_signature(r["params"]) for r in records}

    warnings = []
    if len(versions) > 1:
        warnings.append("multiple_model_versions")
    if len(param_sets) > 1:
        warnings.append("multiple_param_sets")
    if 0 < n < SMALL_SAMPLE_BELOW:
        warnings.append("small_sample")
    if any(r["outcome"] == "refused" for r in records):
        warnings.append("refusals_present")
    if n == 0:
        warnings.append("empty_log")

    n_ok = sum(1 for r in records if r["outcome"] == "ok")
    n_refused = sum(1 for r in records if r["outcome"] == "refused")

    # A single rate across two model versions is not a measurement of
    # anything, so it is not produced.
    may_summarise = n > 0 and "multiple_model_versions" not in warnings
    success_rate = n_ok / n if may_summarise else None
    ci = wilson(n_ok, n) if may_summarise else None

    latencies = [r["latency_ms"] for r in records]
    cost_units = sum(r["in_tokens"] + r["out_tokens"] * price_ratio for r in records)

    return {
        "n": n,
        "cohorts": len({cohort_key(r) for r in records}),
        "success_rate": success_rate,
        "ci": ci,
        "refusal_rate": (n_refused / n) if n else None,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "cost_units": cost_units,
        "cost_per_success": (cost_units / n_ok) if n_ok else None,
        "warnings": sorted(warnings),
    }
