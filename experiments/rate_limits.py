"""What a retry policy does to a rate-limited service.

    python experiments/rate_limits.py
    python experiments/rate_limits.py --json

A rate limit is not a queue. When you exceed it the request is *rejected*, and
what happens next is entirely your client's decision. Four policies, simulated
against the same token bucket with the same work to do:

- **immediate** — retry at once
- **fixed** — wait a constant interval
- **exponential** — double the wait each attempt, no randomness
- **full jitter** — wait a uniform random time between 0 and the exponential
  bound

The interesting comparison is the last two. Exponential backoff without
randomness makes every client that collided at time T retry at T + d, and then
at T + 3d, and so on: they stay synchronised, so they keep colliding. Adding
jitter breaks the synchronisation, which is the whole mechanism.

The simulation is discrete-time and seeded. It is a model of a token bucket,
not a measurement of any provider — what transfers is the ordering of the
policies and roughly how much separates them, not the absolute seconds.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

SEED = 20260809
STEP_MS = 10  # simulation resolution
HORIZON_MS = 600_000

N_CLIENTS = 50
REQUESTS_PER_CLIENT = 20
BUCKET_CAPACITY = 10.0  # burst allowance, in requests
REFILL_PER_S = 20.0  # sustained rate

BASE_MS = 200
CAP_MS = 30_000

POLICIES = ("immediate", "fixed", "exponential", "full_jitter")


def _delay_ms(policy: str, attempt: int, rng: np.random.Generator) -> float:
    """Wait before retry number `attempt` (1-based)."""
    if policy == "immediate":
        return 0.0
    if policy == "fixed":
        return float(BASE_MS)
    bound = min(CAP_MS, BASE_MS * (2 ** (attempt - 1)))
    if policy == "exponential":
        return float(bound)
    if policy == "full_jitter":
        return float(rng.uniform(0, bound))
    raise ValueError(policy)


def simulate(policy: str) -> dict[str, float]:
    rng = np.random.default_rng(SEED)

    tokens = BUCKET_CAPACITY
    refill_per_step = REFILL_PER_S * STEP_MS / 1000.0

    next_try = np.zeros(N_CLIENTS)  # ms
    remaining = np.full(N_CLIENTS, REQUESTS_PER_CLIENT)
    attempt = np.zeros(N_CLIENTS, dtype=int)  # consecutive rejections
    started = np.zeros(N_CLIENTS)  # when the current request first tried

    accepted = 0
    rejected = 0
    latencies: list[float] = []
    now = 0.0

    while remaining.sum() > 0 and now < HORIZON_MS:
        tokens = min(BUCKET_CAPACITY, tokens + refill_per_step)

        # Everyone whose wait has elapsed tries. Order within a tick is
        # shuffled: serving in client-index order would starve high-index
        # clients systematically, which is an artefact of the simulation
        # rather than a property of any retry policy.
        due = np.flatnonzero((remaining > 0) & (next_try <= now))
        rng.shuffle(due)
        for c in due:
            if tokens >= 1.0:
                tokens -= 1.0
                accepted += 1
                latencies.append(now - started[c])
                remaining[c] -= 1
                attempt[c] = 0
                next_try[c] = now
                started[c] = now
            else:
                rejected += 1
                attempt[c] += 1
                next_try[c] = now + _delay_ms(policy, int(attempt[c]), rng)
        now += STEP_MS

    lat = np.array(latencies) if latencies else np.array([0.0])
    return {
        "policy": policy,
        "completed_s": round(now / 1000.0, 1),
        "accepted": accepted,
        "rejected": rejected,
        "wasted_pct": round(100 * rejected / max(1, accepted + rejected), 1),
        "attempts_per_success": round((accepted + rejected) / max(1, accepted), 2),
        "p50_wait_s": round(float(np.percentile(lat, 50)) / 1000.0, 2),
        "p95_wait_s": round(float(np.percentile(lat, 95)) / 1000.0, 2),
        "finished": bool(remaining.sum() == 0),
    }


def compute() -> dict[str, float]:
    out: dict[str, float] = {
        "clients": N_CLIENTS,
        "requests_per_client": REQUESTS_PER_CLIENT,
        "total_requests": N_CLIENTS * REQUESTS_PER_CLIENT,
        "refill_per_s": REFILL_PER_S,
        # The bucket starts full, so the first `BUCKET_CAPACITY` requests are
        # free. Leaving that out would make the floor unreachable and every
        # policy look 1% better than physics allows.
        "floor_s": round(
            (N_CLIENTS * REQUESTS_PER_CLIENT - BUCKET_CAPACITY) / REFILL_PER_S, 1),
    }
    results = {p: simulate(p) for p in POLICIES}
    for p, r in results.items():
        for k, v in r.items():
            if k != "policy":
                out[f"{p}_{k}"] = v

    floor = out["floor_s"]
    for p in POLICIES:
        out[f"{p}_vs_floor"] = round(results[p]["completed_s"] / floor, 2)

    # The comparison that actually separates the policies.
    out["jitter_vs_exponential_speedup"] = round(
        results["exponential"]["completed_s"] / results["full_jitter"]["completed_s"], 2)
    out["immediate_load_multiplier"] = results["immediate"]["attempts_per_success"]
    out["jitter_load_multiplier"] = results["full_jitter"]["attempts_per_success"]
    out["immediate_over_jitter_load"] = round(
        results["immediate"]["attempts_per_success"]
        / results["full_jitter"]["attempts_per_success"], 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"{v['clients']:.0f} clients x {v['requests_per_client']:.0f} requests "
          f"= {v['total_requests']:.0f} against {v['refill_per_s']:.0f}/s "
          f"(floor: {v['floor_s']}s)\n")
    head = (f"{'policy':<14} {'done in':>9} {'vs floor':>9} {'attempts':>9} "
            f"{'per success':>12}")
    print(head)
    print("-" * len(head))
    for p in POLICIES:
        print(f"{p:<14} {v[f'{p}_completed_s']:>8.1f}s {v[f'{p}_vs_floor']:>8.2f}x "
              f"{v[f'{p}_rejected'] + v[f'{p}_accepted']:>9.0f} "
              f"{v[f'{p}_attempts_per_success']:>11.2f}x")
    print(f"\nfull jitter finishes {v['jitter_vs_exponential_speedup']}x faster than plain")
    print(f"exponential, and puts {v['immediate_over_jitter_load']}x less load on the")
    print("service than retrying immediately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
