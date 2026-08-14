"""The reference harness. Scores full marks against its own rubric.

Nothing here is hidden from the learner — README.md describes every one of
these behaviours. What the reference does not do is tell you *why* each one
matters, which is what the rubric's criteria are for.
"""

from __future__ import annotations

import math
import random

METRICS = ("recall", "precision", "mrr", "map", "ndcg")
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260813


def score_query(results: list[str], gold: list[str], k: int) -> dict[str, float]:
    """Every metric for one query."""
    gold_set = set(gold)
    hits = [1.0 if d in gold_set else 0.0 for d in results[:k]]

    # A query with no correct documents has no defined ranking quality. It is
    # excluded from the means rather than scored zero, because scoring it zero
    # would penalise a system for a query that cannot be answered.
    if not gold_set:
        return {m: float("nan") for m in METRICS}

    recall = 1.0 if any(hits) else 0.0
    precision = sum(hits) / k

    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1 / i
            break

    found, acc = 0, 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            found += 1
            acc += found / i
    ap = acc / min(len(gold_set), k)

    dcg = sum(h / math.log2(i + 1) for i, h in enumerate(hits, start=1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
    ndcg = dcg / idcg if idcg else 0.0

    return {"recall": recall, "precision": precision, "mrr": rr,
            "map": ap, "ndcg": ndcg}


def _mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else 0.0


def evaluate(runs: dict[str, list[str]], qrels: dict[str, list[str]],
             k: int = 10, phenomena: dict[str, str] | None = None) -> dict:
    """Score a run, keeping per-query values and reporting the ceiling."""
    per_query = {qid: score_query(runs.get(qid, []), qrels.get(qid, []), k)
                 for qid in qrels}

    means = {m: _mean([s[m] for s in per_query.values()]) for m in METRICS}

    # The bound precision@k cannot exceed on this benchmark. Computed over the
    # scored queries only, so it matches the population the means are over.
    scored = [q for q in qrels if qrels[q]]
    ceiling = (sum(min(len(qrels[q]), k) / k for q in scored) / len(scored)
               if scored else 0.0)

    by_phenomenon: dict[str, dict[str, float]] = {}
    if phenomena:
        groups: dict[str, list[str]] = {}
        for qid in per_query:
            groups.setdefault(phenomena.get(qid, "unknown"), []).append(qid)
        for name, qids in groups.items():
            by_phenomenon[name] = {
                m: _mean([per_query[q][m] for q in qids]) for m in METRICS
            }

    return {
        "per_query": per_query,
        "means": means,
        "precision_ceiling": ceiling,
        "n_scored": len(scored),
        "by_phenomenon": by_phenomenon,
    }


def _mcnemar_p(a: list[float], b: list[float]) -> float:
    """Exact two-sided McNemar over paired binary outcomes."""
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y > 0 and x == 0)
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x > 0 and y == 0)
    n = b_only + a_only
    if n == 0:
        return 1.0
    hi = max(b_only, a_only)
    tail = sum(math.comb(n, i) for i in range(hi, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]],
            metric: str) -> dict:
    """Compare two systems on one metric, choosing the test from its type.

    Recall is one bit per query, so an exact McNemar test applies and is
    exact. Everything else is continuous, so the paired comparison is a
    bootstrap over per-query differences — thresholding a continuous metric to
    force it into McNemar discards the magnitudes that are the whole reason
    for using that metric.
    """
    shared = sorted(set(a) & set(b))
    xs = [a[q][metric] for q in shared if not math.isnan(a[q][metric])]
    ys = [b[q][metric] for q in shared if not math.isnan(b[q][metric])]

    binary = metric == "recall"
    result: dict = {
        "metric": metric,
        "test": "mcnemar" if binary else "bootstrap",
        "n": len(xs),
        "mean_a": sum(xs) / len(xs) if xs else 0.0,
        "mean_b": sum(ys) / len(ys) if ys else 0.0,
    }
    result["diff"] = result["mean_b"] - result["mean_a"]

    if binary:
        p = _mcnemar_p(xs, ys)
        result["p_value"] = p
        result["significant"] = p < 0.05
        return result

    diffs = [y - x for x, y in zip(xs, ys, strict=True)]
    if not diffs:
        result.update({"ci_low": 0.0, "ci_high": 0.0, "significant": False})
        return result
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    n = len(diffs)
    for _ in range(BOOTSTRAP):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low = means[int(0.025 * BOOTSTRAP)]
    high = means[int(0.975 * BOOTSTRAP)]
    result["ci_low"] = low
    result["ci_high"] = high
    result["significant"] = low > 0 or high < 0
    return result
