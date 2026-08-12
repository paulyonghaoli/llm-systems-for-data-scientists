"""How badly can one slice of traffic fail while the headline number looks fine?

    python experiments/aggregate_masking.py
    python experiments/aggregate_masking.py --json

Two questions, both of which decide whether a dashboard can catch an incident
at all:

1. **Masking.** With a healthy success rate `p_h` and a broken subgroup that is
   share `s` of traffic, the aggregate is `(1-s)*p_h + s*p_s`. Invert it and
   you get the worst the subgroup can be while the aggregate still clears a
   stated floor. Below a certain share the answer is "arbitrarily bad" — the
   subgroup can fail *every single request* and the aggregate stays above the
   line.

2. **Detection.** Even when you do slice, the subgroup is small, so its own
   interval is wide. Detecting the failure is a sample-size problem, and the
   sample that matters is the subgroup's, not the total.

Everything here is arithmetic plus one seeded simulation. Re-checked against
lesson 0.5 by gate 18.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

Z95 = 1.959963984540054
SEED = 20260809
TRIALS = 8000

HEALTHY = 0.95
FLOOR = 0.90
SHARES = (0.05, 0.10, 0.20, 0.30)


def wilson(successes: np.ndarray, n: np.ndarray, z: float = Z95):
    """Vectorised Wilson interval; n may be an array with zeros."""
    n = np.maximum(n, 1)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def worst_subgroup_rate(share: float, healthy: float = HEALTHY,
                        floor: float = FLOOR) -> float | None:
    """Lowest subgroup success rate that still leaves the aggregate >= floor.

    Returns None when even a completely broken subgroup cannot pull the
    aggregate below the floor — the case worth knowing about.
    """
    p_s = healthy - (healthy - floor) / share
    return None if p_s < 0 else p_s


def detection_probability(n_total: int, share: float, p_subgroup: float,
                          healthy: float = HEALTHY, trials: int = TRIALS) -> float:
    """How often a slice-and-compare catches the broken subgroup.

    Rule: flag when the subgroup's Wilson interval does not overlap the rest
    of traffic's. Simple, conservative, and what a dashboard can actually do.
    """
    rng = np.random.default_rng(SEED)
    n_sub = rng.binomial(n_total, share, size=trials)
    n_rest = n_total - n_sub
    ok_sub = rng.binomial(np.maximum(n_sub, 0), p_subgroup)
    ok_rest = rng.binomial(np.maximum(n_rest, 0), healthy)

    sub_lo, sub_hi = wilson(ok_sub, n_sub)
    rest_lo, _ = wilson(ok_rest, n_rest)
    flagged = (sub_hi < rest_lo) & (n_sub > 0)
    return float(flagged.mean())


def narrative_incident() -> dict[str, float]:
    """One concrete incident, drawn rather than asserted.

    A provider rolls a new version out to a slice of traffic and it is much
    worse at this task. The dashboard stays green throughout.
    """
    rng = np.random.default_rng(SEED + 1)
    n = 600
    share_new = 0.06
    is_new = rng.random(n) < share_new
    ok = np.where(
        is_new,
        rng.random(n) < 0.40,
        rng.random(n) < 0.95,
    )
    n_new = int(is_new.sum())
    return {
        "narr_n": n,
        "narr_new_items": n_new,
        "narr_new_share_pct": round(100 * n_new / n, 1),
        "narr_overall_pct": round(100 * float(ok.mean()), 1),
        "narr_old_pct": round(100 * float(ok[~is_new].mean()), 1),
        "narr_new_pct": round(100 * float(ok[is_new].mean()), 1),
        "narr_gap_pts": round(100 * float(ok[~is_new].mean() - ok[is_new].mean()), 1),
    }


def compute() -> dict[str, float]:
    out: dict[str, float] = {"healthy_pct": HEALTHY * 100, "floor_pct": FLOOR * 100}

    for share in SHARES:
        tag = str(int(share * 100))
        worst = worst_subgroup_rate(share)
        # Aggregate when the subgroup fails every request.
        out[f"aggregate_if_dead_s{tag}_pct"] = round(100 * (1 - share) * HEALTHY, 2)
        out[f"worst_subgroup_s{tag}_pct"] = -1.0 if worst is None else round(100 * worst, 1)
        out[f"aggregate_drop_if_dead_s{tag}_pts"] = round(100 * share * HEALTHY, 2)

    # The headline: below this share, a totally dead subgroup cannot breach
    # the floor at all.
    out["share_below_which_masking_is_total_pct"] = round(
        100 * (HEALTHY - FLOOR) / HEALTHY, 2)

    # Detection, for a subgroup at 10% of traffic. Two severities: an outright
    # break (45%) and a mild degradation (80%), which is what drift looks like.
    for n in (200, 500, 2000, 5000):
        out[f"detect_pct_n{n}"] = round(100 * detection_probability(n, 0.10, 0.45), 1)
        out[f"detect_mild_pct_n{n}"] = round(100 * detection_probability(n, 0.10, 0.80), 1)

    # And the subgroup's own interval at those totals.
    for n in (200, 500, 2000):
        n_sub = int(round(0.10 * n))
        lo, hi = wilson(np.array([round(0.45 * n_sub)]), np.array([n_sub]))
        out[f"subgroup_n{n}_items"] = n_sub
        out[f"subgroup_halfwidth_n{n}_pts"] = round(100 * float(hi[0] - lo[0]) / 2, 1)

    out.update(narrative_incident())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"Healthy traffic succeeds {HEALTHY:.0%}; the dashboard floor is {FLOOR:.0%}.\n")
    print(f"{'subgroup share':>15}  {'worst it can be':>16}  {'aggregate if it dies':>21}")
    for share in SHARES:
        t = str(int(share * 100))
        worst = v[f"worst_subgroup_s{t}_pct"]
        worst_s = "anything at all" if worst < 0 else f"{worst:.1f}%"
        print(f"{share:>14.0%}  {worst_s:>16}  {v[f'aggregate_if_dead_s{t}_pct']:>20.2f}%")
    print(f"\nBelow {v['share_below_which_masking_is_total_pct']:.2f}% of traffic, a subgroup "
          f"that fails EVERY request cannot pull the aggregate under the floor.")

    print("\nOnce you DO slice, how often is it caught? Subgroup at 10% of traffic,")
    print("against 95% healthy:")
    print(f"  {'total n':>8}  {'subgroup n':>11}  {'broken (45%)':>13}  {'degraded (80%)':>15}")
    for n in (200, 500, 2000, 5000):
        items = v.get(f"subgroup_n{n}_items", 0.10 * n)
        print(f"  {n:>8}  {items:>11.0f}  {v[f'detect_pct_n{n}']:>12.1f}%  "
              f"{v[f'detect_mild_pct_n{n}']:>14.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
