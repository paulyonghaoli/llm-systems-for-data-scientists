"""How big does an LLM evaluation set have to be before it can tell you anything?

    python experiments/eval_power.py
    python experiments/eval_power.py --json

Everything here is ordinary binomial statistics, which is the point: the
reader already owns these tools. What is new is the setting. Evaluation items
are expensive, so n is small; and both systems are run on the *same* items, so
the comparison is paired and almost nobody analyses it that way.

## The generative model, and why it is not the obvious one

Items differ in difficulty. A hard item tends to defeat both systems, so their
scores are correlated, and the *difference* between them is far less noisy
than either score on its own. That correlation is the whole reason a paired
analysis wins, so the simulation has to produce it honestly.

The first version of this experiment added a shared Gaussian shift to each
system's success probability and clipped to [0, 1]. That was wrong, and
quietly so: clipping collapses the two probabilities together at the top of
the range, so a nominal five-point gap was worth much less than five points
and every power figure came out too low. Lesson 0.3 §H keeps the mistake.

This version uses a latent-difficulty model with a logistic link:

    P(success on item i) = logistic(theta_system - d_i),   d_i ~ N(0, sigma)

and solves for each `theta` by quadrature so the marginal success rate is
*exactly* the rate claimed. Nothing is clipped.

That fixed the effect size and revealed a second modelling error. Drawing the
two systems' outcomes independently *given* the item makes them barely
correlated, and pairing then buys almost nothing (measured: no reduction in
required sample size at all). Real variants are not independent — change a
prompt and the new version behaves identically on most items and differs on a
few. So there is an explicit agreement parameter `rho`: with probability rho
the two systems consume the same random draw for an item, and otherwise draw
independently.

**How much pairing buys you is therefore a property of your systems, not a
universal constant**, which is why the results below report a sensitivity to
rho rather than a single number.

Seeded; stable across runs; re-checked against the lesson by gate 18.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

Z95 = 1.959963984540054  # two-sided 95%

SEED = 20260809
TRIALS = 8000  # simulations per power estimate
SEARCH_TRIALS = 3000  # simulations per point of the sample-size search
CHUNK = 1000  # trials per vectorised block, to bound memory
ITEM_SD = 1.0  # spread of item difficulty on the latent scale
RHO = 0.5  # base case: the two systems behave identically on half the items

# Quadrature grid over item difficulty, for solving theta.
_D = np.linspace(-8.0, 8.0, 4001)


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _marginal_rate(theta: float, sigma: float) -> float:
    """Success rate averaged over item difficulty."""
    w = np.exp(-0.5 * (_D / sigma) ** 2)
    w /= w.sum()
    return float((_logistic(theta - _D) * w).sum())


def solve_theta(target: float, sigma: float = ITEM_SD) -> float:
    """Find theta whose marginal success rate is exactly `target`."""
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _marginal_rate(mid, sigma) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. Behaves at small n and near 0 or 1, which is
    exactly where the textbook normal-approximation interval falls apart."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_halfwidth(p: float, n: int) -> float:
    lo, hi = wilson(round(p * n), n)
    return (hi - lo) / 2


def power(n: int, p_a: float, p_b: float, trials: int = TRIALS,
          sigma: float = ITEM_SD, rho: float = RHO) -> dict[str, float]:
    """Fraction of simulated evaluations in which each test calls the
    difference significant.

    `rho` is the probability that the two systems consume the same random
    draw on an item — how often the change simply does not alter behaviour.
    """
    theta_a, theta_b = solve_theta(p_a, sigma), solve_theta(p_b, sigma)
    rng = np.random.default_rng(SEED)
    unpaired = paired = 0

    done = 0
    while done < trials:
        m = min(CHUNK, trials - done)
        d = rng.normal(0.0, sigma, size=(m, n))
        u = rng.random((m, n))
        v = np.where(rng.random((m, n)) < rho, u, rng.random((m, n)))
        a = u < _logistic(theta_a - d)
        b = v < _logistic(theta_b - d)

        sa = a.sum(1)
        sb = b.sum(1)

        # Two-proportion z-test: what people reach for by default.
        pool = (sa + sb) / (2 * n)
        se = np.sqrt(np.maximum(2 * pool * (1 - pool) / n, 1e-12))
        unpaired += int((np.abs(sb / n - sa / n) / se > Z95).sum())

        # McNemar: only the items where the systems disagree carry any
        # information about which one is better.
        b_only = (b & ~a).sum(1)
        a_only = (a & ~b).sum(1)
        disagree = b_only + a_only
        se_p = np.sqrt(np.maximum(disagree, 1e-12))
        paired += int(((disagree > 0) & (np.abs(b_only - a_only) / se_p > Z95)).sum())

        done += m

    return {"unpaired": unpaired / trials, "paired": paired / trials}


GRID = (25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000)


def smallest_n(p_a: float, p_b: float, target: float, key: str,
               rho: float = RHO) -> int:
    for n in GRID:
        if power(n, p_a, p_b, trials=SEARCH_TRIALS, rho=rho)[key] >= target:
            return n
    return -1


def compute() -> dict[str, float]:
    out: dict[str, float] = {"item_sd": ITEM_SD}

    # 1. How wide is a success rate measured on n items?
    for n in (20, 50, 100, 200, 500, 1000):
        out[f"halfwidth_p80_n{n}"] = round(100 * wilson_halfwidth(0.80, n), 1)
    for n in (100, 150, 200, 250, 300, 400, 500, 750, 1000):
        if wilson_halfwidth(0.80, n) <= 0.05:
            out["n_for_5pt_interval"] = n
            break

    # 2. Can you see a real five-point improvement, 80% -> 85%?
    for n in (50, 100, 200, 500):
        p = power(n, 0.80, 0.85)
        out[f"power_unpaired_n{n}"] = round(100 * p["unpaired"], 1)
        out[f"power_paired_n{n}"] = round(100 * p["paired"], 1)

    out["n_unpaired_80pct_power"] = smallest_n(0.80, 0.85, 0.80, "unpaired")
    out["n_paired_80pct_power"] = smallest_n(0.80, 0.85, 0.80, "paired")
    out["paired_sample_saving"] = round(
        out["n_unpaired_80pct_power"] / max(1, out["n_paired_80pct_power"]), 1)

    # 2b. How much pairing buys depends on how alike the two systems are.
    for rho in (0.0, 0.5, 0.9):
        tag = str(int(rho * 100))
        p = power(200, 0.80, 0.85, rho=rho)
        out[f"power_unpaired_n200_rho{tag}"] = round(100 * p["unpaired"], 1)
        out[f"power_paired_n200_rho{tag}"] = round(100 * p["paired"], 1)
        out[f"n_paired_80pct_rho{tag}"] = smallest_n(0.80, 0.85, 0.80, "paired", rho=rho)
        # And with no real difference: is each test calibrated at 5%?
        nullp = power(200, 0.80, 0.80, rho=rho)
        out[f"false_alarm_unpaired_rho{tag}"] = round(100 * nullp["unpaired"], 1)
        out[f"false_alarm_paired_rho{tag}"] = round(100 * nullp["paired"], 1)

    # 3. Sanity check: with no real difference, how often does each test
    #    call a winner anyway? Both should sit near 5%.
    null = power(200, 0.80, 0.80)
    out["false_alarm_unpaired_n200"] = round(100 * null["unpaired"], 1)
    out["false_alarm_paired_n200"] = round(100 * null["paired"], 1)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()

    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print("A success rate measured on n items, observed 80% (Wilson 95%):")
    for n in (20, 50, 100, 200, 500, 1000):
        print(f"  n = {n:>5}   80% +/- {v[f'halfwidth_p80_n{n}']:.1f} points")
    print(f"  first n on the grid within +/-5 points: {v['n_for_5pt_interval']:.0f}")

    print(f"\nDetecting a real 80% -> 85% improvement ({TRIALS} simulations):")
    print(f"  {'n':>5}  {'unpaired':>9}  {'paired':>7}")
    for n in (50, 100, 200, 500):
        print(f"  {n:>5}  {v[f'power_unpaired_n{n}']:>8.1f}%  {v[f'power_paired_n{n}']:>6.1f}%")
    print(f"  n for 80% power: unpaired {v['n_unpaired_80pct_power']:.0f}, "
          f"paired {v['n_paired_80pct_power']:.0f} "
          f"({v['paired_sample_saving']}x fewer items)")

    print("\nHow alike are the two systems? (rho = fraction of items where the")
    print("change makes no difference at all) -- power at n=200, and n for 80%:")
    print(f"  {'rho':>5}  {'unpaired':>9}  {'paired':>7}  {'n80 paired':>11}  "
          f"{'false alarm (u/p)':>18}")
    for rho in (0.0, 0.5, 0.9):
        t = str(int(rho * 100))
        alarm = (f"{v[f'false_alarm_unpaired_rho{t}']:.1f}% / "
                 f"{v[f'false_alarm_paired_rho{t}']:.1f}%")
        print(f"  {rho:>5}  {v[f'power_unpaired_n200_rho{t}']:>8.1f}%  "
              f"{v[f'power_paired_n200_rho{t}']:>6.1f}%  "
              f"{v[f'n_paired_80pct_rho{t}']:>11.0f}  {alarm:>18}")

    print("\nWith no real difference at all (both should be near 5%):")
    print(f"  unpaired {v['false_alarm_unpaired_n200']}%   paired {v['false_alarm_paired_n200']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
