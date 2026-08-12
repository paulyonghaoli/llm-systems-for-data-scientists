"""What voting over k samples buys, and what decomposition costs.

    python experiments/self_consistency.py
    python experiments/self_consistency.py --json

Three pieces of arithmetic, all exact rather than simulated, because the
underlying model is binomial and closed forms are available:

1. **Self-consistency.** Sample k answers and take the majority. This is
   Condorcet's jury theorem, and it has a hard precondition that gets omitted
   whenever the technique is recommended.
2. **Correlated samples.** Repeated samples from one model are *not*
   independent jurors. A correlation parameter `rho` — the probability that
   the model is committed to one answer and every sample agrees — bounds what
   voting can achieve, and at high `rho` it achieves nothing while costing k
   times as much.
3. **Decomposition.** Splitting one hard step into n easier ones multiplies n
   accuracies together, so it wins only if the per-step accuracy rises enough
   to beat the chain. The break-even is exact and is usually higher than
   people guess.

The independence assumption is the whole ballgame, so it is a parameter here
rather than an unstated premise.
"""

from __future__ import annotations

import argparse
import json
from math import comb

P_VALUES = (0.40, 0.55, 0.70, 0.85)
K_VALUES = (1, 3, 5, 11, 21)
RHO_VALUES = (0.0, 0.3, 0.6)

#: Cost is dominated by the k sampled answers; a vote itself is free.
BASE_P = 0.70


def majority_correct(p: float, k: int) -> float:
    """Probability that a majority of k independent samples is correct.

    Binary case: each sample is correct with probability p, and the majority
    is correct when more than half of them are. Even k is not used here — a
    tie has no winner and every practical implementation uses odd k.
    """
    need = k // 2 + 1
    return sum(comb(k, i) * p**i * (1 - p) ** (k - i) for i in range(need, k + 1))


def majority_correlated(p: float, k: int, rho: float) -> float:
    """Majority accuracy when samples are not independent.

    With probability `rho` the model is committed: every sample returns the
    same answer, so voting cannot help and accuracy is just p. Otherwise the
    k samples are independent and the jury theorem applies.
    """
    return rho * p + (1 - rho) * majority_correct(p, k)


def chain_accuracy(q: float, n: int) -> float:
    """End-to-end accuracy of n independent steps each correct with prob q."""
    return q**n


def breakeven_step_accuracy(p: float, n: int) -> float:
    """Per-step accuracy a chain of n steps needs to match one step at p."""
    return p ** (1 / n)


def compute() -> dict[str, float]:
    out: dict[str, float] = {"base_p": BASE_P}

    # 1. Voting, independent samples.
    for p in P_VALUES:
        tag = str(int(p * 100))
        for k in K_VALUES:
            out[f"vote_p{tag}_k{k}"] = round(100 * majority_correct(p, k), 1)
        out[f"vote_gain_p{tag}_k5"] = round(
            100 * (majority_correct(p, 5) - p), 1)

    # The precondition: below one half, voting makes things worse.
    out["vote_p40_k1"] = round(100 * 0.40, 1)
    out["vote_p40_k21_loss"] = round(100 * (0.40 - majority_correct(0.40, 21)), 1)

    # 2. What each additional accuracy point costs, at the base rate.
    #
    # "Samples per correct answer" (k / accuracy) looks like the natural
    # metric and is useless: cost rises linearly in k while accuracy is
    # bounded by 1, so it is minimised at k = 1 for every p and can never
    # recommend anything. The marginal version is the one that answers the
    # question, and it is where the diminishing return actually lives —
    # the raw accuracy gain does not diminish at all here.
    out["k5_over_k1_gain_pts"] = round(
        100 * (majority_correct(BASE_P, 5) - majority_correct(BASE_P, 1)), 1)
    out["k21_over_k5_gain_pts"] = round(
        100 * (majority_correct(BASE_P, 21) - majority_correct(BASE_P, 5)), 1)
    for a, b in zip(K_VALUES, K_VALUES[1:], strict=False):
        gain = 100 * (majority_correct(BASE_P, b) - majority_correct(BASE_P, a))
        out[f"marginal_cost_k{a}_to_k{b}"] = round((b - a) / gain, 2)
    out["error_k1_pct"] = round(100 * (1 - majority_correct(BASE_P, 1)), 1)
    out["error_k5_pct"] = round(100 * (1 - majority_correct(BASE_P, 5)), 1)
    out["error_k21_pct"] = round(100 * (1 - majority_correct(BASE_P, 21)), 1)

    # 3. Correlation bounds the benefit.
    for rho in RHO_VALUES:
        tag = str(int(rho * 100))
        out[f"corr_rho{tag}_k5"] = round(100 * majority_correlated(BASE_P, 5, rho), 1)
        out[f"corr_rho{tag}_k21"] = round(100 * majority_correlated(BASE_P, 21, rho), 1)
    out["rho60_k21_gain_pts"] = round(
        100 * (majority_correlated(BASE_P, 21, 0.6) - BASE_P), 1)

    # 4. Chains and the cost of decomposition.
    for n in (2, 3, 5):
        out[f"chain_q90_n{n}"] = round(100 * chain_accuracy(0.90, n), 1)
        out[f"breakeven_n{n}"] = round(100 * breakeven_step_accuracy(BASE_P, n), 1)
    out["decomp_n3_lift_needed_pts"] = round(
        100 * (breakeven_step_accuracy(BASE_P, 3) - BASE_P), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print("Majority vote over k independent samples (accuracy %):\n")
    head = f"{'per-sample':>11}" + "".join(f"{'k=' + str(k):>8}" for k in K_VALUES)
    print(head)
    print("-" * len(head))
    for p in P_VALUES:
        t = str(int(p * 100))
        row = "".join(f"{v[f'vote_p{t}_k{k}']:>7.1f}%" for k in K_VALUES)
        print(f"{p:>10.0%} {row}")
    print(f"\nBelow 50% the theorem runs backwards: at p=0.40, k=21 is "
          f"{v['vote_p40_k21_loss']} points WORSE than a single sample.")

    print(f"\nAt p={BASE_P:.0%}, the accuracy gain does NOT diminish:")
    print(f"  k=1 -> k=5  : +{v['k5_over_k1_gain_pts']} points")
    print(f"  k=5 -> k=21 : +{v['k21_over_k5_gain_pts']} points  (identical)")
    print(f"  error: {v['error_k1_pct']}% -> {v['error_k5_pct']}% -> {v['error_k21_pct']}%")
    print("\n  ...but the cost of each point does. Extra samples per accuracy point:")
    for a, b in zip(K_VALUES, K_VALUES[1:], strict=False):
        print(f"    k={a:>2} -> k={b:<2}  {v[f'marginal_cost_k{a}_to_k{b}']:>5}")

    print(f"\nCorrelated samples at p={BASE_P:.0%} (rho = P(model is committed)):")
    print(f"  {'rho':>5} {'k=5':>8} {'k=21':>8}")
    for rho in RHO_VALUES:
        t = str(int(rho * 100))
        print(f"  {rho:>5.1f} {v[f'corr_rho{t}_k5']:>7.1f}% {v[f'corr_rho{t}_k21']:>7.1f}%")
    print(f"  at rho=0.6, twenty-one samples buy {v['rho60_k21_gain_pts']} points")

    print(f"\nDecomposing one step at {BASE_P:.0%} into n steps needs each step at:")
    for n in (2, 3, 5):
        print(f"  n={n}: {v[f'breakeven_n{n}']}%  (a chain of 90% steps gives "
              f"{v[f'chain_q90_n{n}']}%)")
    print(f"  splitting into 3 requires +{v['decomp_n3_lift_needed_pts']} points "
          f"per step merely to break even")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
