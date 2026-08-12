"""Three designs for the same task, costed before any of them is built.

    python experiments/design_costs.py
    python experiments/design_costs.py --json

Answer a question against a corpus of N chunks. Three architectures:

- **long context** — put the whole corpus in one prompt. One call.
- **retrieval** — retrieve the k best chunks. One call, independent of N.
- **map-reduce** — one call per chunk to summarise, then one call to combine.
  N + 1 calls.

Everything here is arithmetic on a stated workload model; nothing is measured
from a provider, and the absolute seconds are the model's. What transfers is
the *shape*: which design grows with N, which does not, and where the curves
cross.

## What this deliberately does not model

**Quality.** Retrieval is the cheapest design at every size and is worthless
if it retrieves the wrong chunks. A cost model tells you what a design costs,
never whether it works — that comparison needs lesson 0.3's machinery on your
own data. Treating the cheap column as the answer is the main way this kind of
analysis gets misused.
"""

from __future__ import annotations

import argparse
import json

# --- workload model ---------------------------------------------------------

CHUNK_TOKENS = 400
SYSTEM_TOKENS = 120
QUESTION_TOKENS = 25
ANSWER_TOKENS = 200
SUMMARY_TOKENS = 60  # what one map call emits per chunk

RETRIEVED_K = 6
PRICE_RATIO = 4.0  # output tokens cost this many input tokens
PARALLELISM = 8  # concurrent map calls

# Latency model, same shape as experiments/service_economics.py.
TTFT_BASE_MS = 200.0
TTFT_PER_INPUT_TOKEN_MS = 0.045
TPOT_MS = 11.0

#: Illustrative window size, in tokens. A parameter of the model, not a claim
#: about any product — substitute your own.
WINDOW_TOKENS = 100_000

SIZES = (10, 50, 100, 500, 1000)


def _call(in_tokens: float, out_tokens: float) -> dict[str, float]:
    return {
        "in": in_tokens,
        "out": out_tokens,
        "cost": in_tokens + out_tokens * PRICE_RATIO,
        "latency_ms": TTFT_BASE_MS + TTFT_PER_INPUT_TOKEN_MS * in_tokens + TPOT_MS * out_tokens,
    }


def long_context(n: int) -> dict[str, float]:
    c = _call(SYSTEM_TOKENS + n * CHUNK_TOKENS + QUESTION_TOKENS, ANSWER_TOKENS)
    return {"calls": 1, "in": c["in"], "out": c["out"], "cost": c["cost"],
            "latency_ms": c["latency_ms"], "fits": c["in"] + ANSWER_TOKENS <= WINDOW_TOKENS}


def retrieval(n: int) -> dict[str, float]:
    k = min(RETRIEVED_K, n)
    c = _call(SYSTEM_TOKENS + k * CHUNK_TOKENS + QUESTION_TOKENS, ANSWER_TOKENS)
    return {"calls": 1, "in": c["in"], "out": c["out"], "cost": c["cost"],
            "latency_ms": c["latency_ms"], "fits": True}


def map_reduce(n: int, parallelism: int = PARALLELISM) -> dict[str, float]:
    m = _call(SYSTEM_TOKENS + CHUNK_TOKENS + QUESTION_TOKENS, SUMMARY_TOKENS)
    r = _call(SYSTEM_TOKENS + n * SUMMARY_TOKENS + QUESTION_TOKENS, ANSWER_TOKENS)

    # Map calls run concurrently, in ceil(n / parallelism) rounds.
    rounds = -(-n // parallelism)
    return {
        "calls": n + 1,
        "in": n * m["in"] + r["in"],
        "out": n * m["out"] + r["out"],
        "cost": n * m["cost"] + r["cost"],
        "latency_ms": rounds * m["latency_ms"] + r["latency_ms"],
        "fits": r["in"] + ANSWER_TOKENS <= WINDOW_TOKENS,
    }


DESIGNS = {"long_context": long_context, "retrieval": retrieval, "map_reduce": map_reduce}


def compute() -> dict[str, float]:
    out: dict[str, float] = {
        "chunk_tokens": CHUNK_TOKENS,
        "retrieved_k": RETRIEVED_K,
        "price_ratio": PRICE_RATIO,
        "parallelism": PARALLELISM,
        "window_tokens": WINDOW_TOKENS,
    }

    results = {n: {name: fn(n) for name, fn in DESIGNS.items()} for n in SIZES}

    for n in SIZES:
        for name in DESIGNS:
            r = results[n][name]
            out[f"{name}_n{n}_cost"] = round(r["cost"])
            out[f"{name}_n{n}_calls"] = r["calls"]
            out[f"{name}_n{n}_latency_s"] = round(r["latency_ms"] / 1000, 1)
        out[f"longctx_over_rag_n{n}"] = round(
            results[n]["long_context"]["cost"] / results[n]["retrieval"]["cost"], 1)
        out[f"mapreduce_over_rag_n{n}"] = round(
            results[n]["map_reduce"]["cost"] / results[n]["retrieval"]["cost"], 1)

    # Where long context stops fitting at all.
    biggest = max((n for n in range(1, 5000) if long_context(n)["fits"]), default=0)
    out["long_context_max_chunks"] = biggest

    # Is map-reduce ever the faster option at this concurrency?
    crossover = next(
        (n for n in range(1, 5000)
         if map_reduce(n)["latency_ms"] < long_context(n)["latency_ms"]), -1)
    out["mapreduce_faster_from_n"] = crossover  # -1 means never, at PARALLELISM

    # If not, what concurrency would it take at the largest size modelled?
    big = SIZES[-1]
    target = long_context(big)["latency_ms"]
    out["parallelism_needed_at_max_n"] = next(
        (p for p in range(1, big + 1) if map_reduce(big, p)["latency_ms"] < target), -1)
    out["mapreduce_over_longctx_at_max_n"] = round(
        map_reduce(big)["cost"] / long_context(big)["cost"], 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"chunks of {CHUNK_TOKENS} tokens, k={RETRIEVED_K}, output priced at "
          f"{PRICE_RATIO:.0f}x input, {PARALLELISM} concurrent map calls\n")
    print(f"{'N':>6}  {'long ctx':>18}  {'retrieval':>18}  {'map-reduce':>20}")
    print(f"{'':>6}  {'cost / latency':>18}  {'cost / latency':>18}  "
          f"{'calls / cost / lat':>20}")
    print("-" * 72)
    for n in SIZES:
        lc = (f"{v[f'long_context_n{n}_cost']:.0f} / "
              f"{v[f'long_context_n{n}_latency_s']}s")
        rg = f"{v[f'retrieval_n{n}_cost']:.0f} / {v[f'retrieval_n{n}_latency_s']}s"
        mr = (f"{v[f'map_reduce_n{n}_calls']:.0f} / {v[f'map_reduce_n{n}_cost']:.0f} / "
              f"{v[f'map_reduce_n{n}_latency_s']}s")
        print(f"{n:>6}  {lc:>18}  {rg:>18}  {mr:>20}")

    print(f"\nlong context costs {v['longctx_over_rag_n100']}x retrieval at N=100, "
          f"{v['longctx_over_rag_n1000']}x at N=1000")
    print(f"long context stops fitting a {WINDOW_TOKENS:,}-token window beyond "
          f"N={v['long_context_max_chunks']:.0f}")
    if v["mapreduce_faster_from_n"] < 0:
        print(f"map-reduce is never faster at {PARALLELISM} concurrent calls; it would "
              f"need {v['parallelism_needed_at_max_n']:.0f}")
        print(f"to match long context at N={SIZES[-1]}, while costing "
              f"{v['mapreduce_over_longctx_at_max_n']}x as much")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
