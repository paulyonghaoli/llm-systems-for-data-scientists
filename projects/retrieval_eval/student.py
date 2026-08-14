"""Mini-project 3 - the retrieval evaluation harness.

Implement `evaluate` and `compare` below, then grade them:

    cd projects/retrieval_eval
    python -m grader --seed 1

The full specification and the rubric are in README.md. Benchmarks are
generated fresh from the seed on every run, so there is no fixed expected
output.

The starter below is the harness most people write first, and it is not
stupid: the metric arithmetic in it is correct. What it does is throw the
per-query values away and return only the means, which quietly forecloses
every question you will want to ask within a week — paired tests, confidence
intervals, and breakdowns by query type all need the values it discarded.

`compare` is then stuck. With only means available it can do nothing but
subtract them and guess, so it reports a difference with no test behind it and
calls anything non-zero significant. Both systems in the benchmark have
*identical* recall by construction, so that guess is wrong immediately.
"""

from __future__ import annotations

import math


def evaluate(runs, qrels, k=10, phenomena=None):
    """Score a run against gold labels.

    runs        {query_id: [doc_id, ...]}   ranked, best first
    qrels       {query_id: [doc_id, ...]}   the correct documents
    k           cutoff
    phenomena   {query_id: label} or None

    Returns a dict — see README.md for the required keys.
    """
    totals = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "map": 0.0, "ndcg": 0.0}
    n = 0

    for qid, gold in qrels.items():
        gold_set = set(gold)
        if not gold_set:
            continue
        hits = [1.0 if d in gold_set else 0.0 for d in runs.get(qid, [])[:k]]
        n += 1

        totals["recall"] += 1.0 if any(hits) else 0.0
        totals["precision"] += sum(hits) / k

        for i, h in enumerate(hits, start=1):
            if h:
                totals["mrr"] += 1 / i
                break

        found, acc = 0, 0.0
        for i, h in enumerate(hits, start=1):
            if h:
                found += 1
                acc += found / i
        totals["map"] += acc / min(len(gold_set), k)

        dcg = sum(h / math.log2(i + 1) for i, h in enumerate(hits, start=1))
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
        totals["ndcg"] += dcg / idcg if idcg else 0.0

    # Everything above is correct, and everything below throws away what makes
    # it useful: the per-query values are gone, there is no ceiling, and no
    # breakdown by query type.
    return {"means": {m: (v / n if n else 0.0) for m, v in totals.items()}}


def compare(a, b, metric):
    """Compare two systems on one metric.

    a, b     the `per_query` mappings returned by `evaluate`
    metric   one of recall, precision, mrr, map, ndcg

    Returns a dict — see README.md for the required keys.
    """
    # With no per-query values there is nothing to test, so this subtracts two
    # numbers and asserts a result. It names no test and applies none.
    return {"metric": metric, "test": "none", "significant": True}
