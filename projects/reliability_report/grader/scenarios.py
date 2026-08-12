"""Seeded run logs.

Deliberately varied along the axes that decide whether a summary is even
allowed: how many model versions the log spans, how many parameter sets, how
many records, and whether refusals are present.
"""

from __future__ import annotations

import random

OUTCOMES_OK = "ok"
VERSIONS = ["m-2026-03-01", "m-2026-08-01"]
PARAM_SETS = [
    {"temperature": 0.0, "top_p": 1.0},
    {"temperature": 0.7, "top_p": 0.9},
]


def _record(rng: random.Random, version: str, params: dict, refusal_rate: float,
            fail_rate: float) -> dict:
    roll = rng.random()
    if roll < refusal_rate:
        outcome = "refused"
    elif roll < refusal_rate + fail_rate:
        outcome = rng.choice(["timeout", "error", "invalid"])
    else:
        outcome = OUTCOMES_OK

    in_tokens = int(rng.lognormvariate(7.3, 0.5))
    out_tokens = 0 if outcome in ("timeout", "error") else int(rng.lognormvariate(4.9, 0.45))

    latency = 300 + 0.05 * in_tokens + 11 * out_tokens + rng.expovariate(1 / 250)
    if rng.random() < 0.05:
        latency += rng.expovariate(1 / 2500)

    return {
        "model_version": version,
        "params": params,
        "prompt_hash": f"p{rng.randrange(6)}",
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "latency_ms": round(latency, 1),
        "outcome": outcome,
    }


def scenario(seed: int) -> dict:
    rng = random.Random(seed)
    n = rng.choice([80, 150, 300, 600])
    n_versions = 1 if rng.random() < 0.6 else 2
    n_params = 1 if rng.random() < 0.7 else 2
    refusal_rate = rng.choice([0.0, 0.03, 0.10])
    fail_rate = rng.choice([0.01, 0.05, 0.12])

    records = [
        _record(
            rng,
            VERSIONS[rng.randrange(n_versions)],
            PARAM_SETS[rng.randrange(n_params)],
            refusal_rate,
            fail_rate,
        )
        for _ in range(n)
    ]
    return {"seed": seed, "records": records, "price_ratio": rng.choice([3.0, 4.0, 5.0])}


def edge_scenarios() -> list[dict]:
    """The cases a report written against the happy path gets wrong."""
    rng = random.Random(0)
    base = [_record(rng, VERSIONS[0], PARAM_SETS[0], 0.0, 0.0) for _ in range(40)]

    every_call_failed = []
    for r in (dict(x) for x in base[:30]):
        r["outcome"] = "error"
        r["out_tokens"] = 0
        every_call_failed.append(r)

    two_versions = [dict(x) for x in base]
    for r in two_versions[:20]:
        r["model_version"] = VERSIONS[1]

    all_refused = []
    for r in (dict(x) for x in base[:25]):
        r["outcome"] = "refused"
        all_refused.append(r)

    return [
        {"name": "empty log", "records": [], "price_ratio": 4.0},
        {"name": "not one success", "records": every_call_failed, "price_ratio": 4.0},
        {"name": "two model versions", "records": two_versions, "price_ratio": 4.0},
        {"name": "every call refused", "records": all_refused, "price_ratio": 4.0},
    ]
