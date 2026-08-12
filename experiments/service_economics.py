"""What a request actually costs, and what it actually takes.

    python experiments/service_economics.py
    python experiments/service_economics.py --json

Three pieces of arithmetic that decide whether an LLM feature is viable, none
of which is difficult and all of which are usually skipped:

1. **Where the money goes.** Output tokens are priced several times higher
   than input tokens, so the token counts and the cost shares are different
   distributions. Optimising the wrong one is common.
2. **What the latency number means.** Response times are heavy-tailed, so the
   mean is not a typical experience and the tail is not a rare one.
3. **What reliability costs.** Retries multiply spend as well as latency, and
   the multiplier is not small once you count the requests that fail *after*
   generating output.

No market prices appear here or in any lesson — they change monthly and would
be wrong by the time you read this. Everything is expressed as a **price
ratio** (how much more an output token costs than an input token), which is
structural and moves slowly, so the conclusions survive. Supply your own
provider's numbers to get absolute figures.

Traffic is generated from a fixed seed; gate 18 re-checks the lesson against
whatever this prints.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

SEED = 20260809
N_REQUESTS = 20000

#: Output tokens cost this many times what input tokens cost. A ratio, not a
#: price. Three values, because providers differ and the conclusion should
#: not depend on picking one.
PRICE_RATIOS = (3.0, 4.0, 5.0)
BASE_RATIO = 4.0

#: Fraction of requests that wait behind someone else's generation.
QUEUE_FRACTION = 0.05


def traffic(rng: np.random.Generator, n: int = N_REQUESTS) -> dict[str, np.ndarray]:
    """A plausible retrieval-augmented workload.

    Prompts are dominated by retrieved context, so input is log-normal with a
    long right tail; answers are short and much less variable. Latency is
    modelled as a fixed overhead plus a per-output-token cost plus a
    heavy-tailed component for queueing.
    """
    in_tokens = np.clip(rng.lognormal(mean=7.4, sigma=0.55, size=n), 50, None)
    out_tokens = np.clip(rng.lognormal(mean=5.0, sigma=0.45, size=n), 5, None)

    ttft_ms = 180 + 0.045 * in_tokens + rng.lognormal(mean=3.6, sigma=0.85, size=n)
    tpot_ms = 11 + rng.normal(0, 1.2, size=n).clip(-4, 8)

    # Queueing. A small fraction of requests arrive when the server is busy
    # and wait behind other people's generations. This is a mixture, not a
    # fatter lognormal, because that is what actually happens: most requests
    # are unaffected and a few are affected a lot. Without it the tail is far
    # too well behaved and the "the mean is not typical" argument does not
    # survive its own evidence.
    queued = rng.random(n) < QUEUE_FRACTION
    queue_ms = np.where(queued, rng.lognormal(mean=7.6, sigma=0.9, size=n), 0.0)

    latency_ms = ttft_ms + queue_ms + tpot_ms * out_tokens
    return {
        "in": in_tokens,
        "out": out_tokens,
        "ttft_ms": ttft_ms + queue_ms,
        "latency_ms": latency_ms,
    }


def retry_economics(fail_rate: float, max_attempts: int) -> dict[str, float]:
    """What a bounded retry policy costs.

    attempts_per_request  = sum_{k=1..K} f^(k-1) = (1 - f^K) / (1 - f)
    success_rate          = 1 - f^K
    attempts_per_success  = the ratio of those two

    The ratio simplifies to **1 / (1 - f), with K cancelling out entirely**.
    That is worth sitting with: raising the retry limit does not change what a
    successful request costs you on average. It buys a higher success rate and
    a worse tail latency, and nothing else. The algebra is three lines and it
    contradicts the intuition that "more retries means more waste".
    """
    f = fail_rate
    attempts_per_request = (1 - f**max_attempts) / (1 - f) if f < 1 else max_attempts
    success_rate = 1 - f**max_attempts
    return {
        "attempts_per_request": attempts_per_request,
        "success_rate": success_rate,
        "attempts_per_success": (
            attempts_per_request / success_rate if success_rate > 0 else float("inf")
        ),
    }


def compute() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    t = traffic(rng)
    out: dict[str, float] = {}

    tin, tout = t["in"].sum(), t["out"].sum()
    out["mean_in_tokens"] = round(float(t["in"].mean()))
    out["mean_out_tokens"] = round(float(t["out"].mean()))
    out["input_share_of_tokens_pct"] = round(100 * tin / (tin + tout), 1)

    # 1. Where the money goes, at each price ratio.
    for r in PRICE_RATIOS:
        tag = str(int(r))
        cost_in, cost_out = tin, tout * r
        out[f"output_share_of_cost_r{tag}_pct"] = round(
            100 * cost_out / (cost_in + cost_out), 1)

    r = BASE_RATIO
    total = tin + tout * r
    # 2. Which 30% cut is worth more?
    out["saving_from_30pct_shorter_prompts_pct"] = round(
        100 * (0.30 * tin) / total, 1)
    out["saving_from_30pct_shorter_answers_pct"] = round(
        100 * (0.30 * tout * r) / total, 1)

    # 3. Latency: the mean is not a typical experience.
    lat = t["latency_ms"]
    out["latency_mean_ms"] = round(float(lat.mean()))
    out["latency_p50_ms"] = round(float(np.percentile(lat, 50)))
    out["latency_p95_ms"] = round(float(np.percentile(lat, 95)))
    out["latency_p99_ms"] = round(float(np.percentile(lat, 99)))
    out["p95_over_p50"] = round(float(np.percentile(lat, 95) / np.percentile(lat, 50)), 1)
    out["pct_slower_than_mean"] = round(100 * float((lat > lat.mean()).mean()), 1)
    out["ttft_share_of_latency_p50_pct"] = round(
        100 * float(np.percentile(t["ttft_ms"], 50) / np.percentile(lat, 50)), 1)

    # 4. What reliability costs.
    for fr in (0.02, 0.08, 0.20):
        tag = str(int(fr * 100))
        e3 = retry_economics(fr, 3)
        out[f"attempts_per_success_f{tag}"] = round(e3["attempts_per_success"], 2)
        out[f"cost_overhead_f{tag}_pct"] = round(100 * (e3["attempts_per_success"] - 1), 1)
        out[f"attempts_per_request_f{tag}"] = round(e3["attempts_per_request"], 3)
        out[f"success_rate_f{tag}_pct"] = round(100 * e3["success_rate"], 2)
    out["give_up_rate_f20_pct"] = round(100 * 0.20**3, 1)

    # The K-independence, stated as a number so the lesson cannot drift from
    # it: same failure rate, retry limits 3 and 10, identical cost per success.
    out["apc_f20_k3"] = round(retry_economics(0.20, 3)["attempts_per_success"], 4)
    out["apc_f20_k10"] = round(retry_economics(0.20, 10)["attempts_per_success"], 4)
    out["success_rate_f20_k10_pct"] = round(
        100 * retry_economics(0.20, 10)["success_rate"], 3)

    # 5. Serial reliability. An LLM feature is a pipeline of stages, and a
    #    request has to survive all of them.
    stages = 7
    out["pipeline_stages"] = stages
    for per_stage in (0.99, 0.995, 0.999):
        tag = str(per_stage).replace("0.", "").ljust(3, "0")
        out[f"end_to_end_{tag}_pct"] = round(100 * per_stage**stages, 1)
    out["per_stage_needed_for_99_pct"] = round(100 * 0.99 ** (1 / stages), 3)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"Workload: {N_REQUESTS} requests, mean {v['mean_in_tokens']:.0f} in / "
          f"{v['mean_out_tokens']:.0f} out tokens")
    print(f"  input is {v['input_share_of_tokens_pct']}% of all tokens\n")

    print("Share of spend that is OUTPUT tokens, by price ratio:")
    for r in PRICE_RATIOS:
        print(f"  output costs {r:.0f}x input  ->  "
              f"{v[f'output_share_of_cost_r{int(r)}_pct']}% of the bill")

    print(f"\nCutting 30% at a {BASE_RATIO:.0f}x price ratio:")
    print(f"  30% shorter prompts saves {v['saving_from_30pct_shorter_prompts_pct']}%")
    print(f"  30% shorter answers saves {v['saving_from_30pct_shorter_answers_pct']}%")

    print("\nLatency:")
    print(f"  mean {v['latency_mean_ms']:.0f} ms | p50 {v['latency_p50_ms']:.0f} | "
          f"p95 {v['latency_p95_ms']:.0f} | p99 {v['latency_p99_ms']:.0f}")
    print(f"  p95/p50 = {v['p95_over_p50']}x; only "
          f"{v['pct_slower_than_mean']}% of requests are slower than the mean")
    print(f"  time to first token is {v['ttft_share_of_latency_p50_pct']}% "
          f"of median total latency")

    print("\nRetries (up to 3 attempts):")
    for fr in (0.02, 0.08, 0.20):
        t = str(int(fr * 100))
        print(f"  {fr:.0%} failure -> {v[f'attempts_per_success_f{t}']} attempts per success "
              f"(+{v[f'cost_overhead_f{t}_pct']}% spend), "
              f"{v[f'success_rate_f{t}_pct']}% eventually succeed")
    print(f"  at 20% failure, {v['give_up_rate_f20_pct']}% of requests still give up")
    print(f"  raising the limit 3 -> 10 at 20% failure: cost per success "
          f"{v['apc_f20_k3']} -> {v['apc_f20_k10']} (unchanged), "
          f"success {v['success_rate_f20_pct']}% -> {v['success_rate_f20_k10_pct']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
