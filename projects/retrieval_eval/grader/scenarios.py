"""Synthetic benchmarks, generated fresh from a seed on every run.

Two systems are produced for each benchmark, and the relationship between them
is the point of the whole exercise. System B is system A with correct documents
moved *up* for some queries and nothing else changed, which is exactly what a
reranker does. So:

  * recall@k is **identical** between them, because the same documents are in
    the same top-k window;
  * MRR and nDCG are **better** for B, because position is all that moved.

A harness that reports only recall concludes the two systems are the same. A
harness that applies a binary test to MRR throws away the magnitudes that
carry the difference. Both mistakes are gradeable, and both were made by real
teams before they were made by this file.
"""

from __future__ import annotations

import random

N_QUERIES = 60
CORPUS = 300
K = 10
PHENOMENA = ("vocabulary_mismatch", "lexical_distractor", "multi_hop", "near_duplicate")


def benchmark(seed: int, n_queries: int = N_QUERIES) -> dict:
    """One benchmark: gold labels, two systems' ranked runs, and query types."""
    rng = random.Random(seed)
    qrels: dict[str, list[str]] = {}
    runs_a: dict[str, list[str]] = {}
    runs_b: dict[str, list[str]] = {}
    phenomena: dict[str, str] = {}

    for i in range(n_queries):
        qid = f"q{i:03d}"
        phenomena[qid] = PHENOMENA[i % len(PHENOMENA)]

        # One gold document usually, two occasionally: enough multi-gold
        # queries that MAP and MRR are not the same number everywhere.
        n_gold = 2 if rng.random() < 0.25 else 1
        gold = [f"d{rng.randrange(CORPUS):03d}" for _ in range(n_gold)]
        gold = list(dict.fromkeys(gold))
        qrels[qid] = gold

        # A ranked list of distinct documents, with the gold placed somewhere
        # in it — or nowhere, for the queries neither system answers.
        filler = []
        seen = set(gold)
        while len(filler) < K + 5:
            d = f"d{rng.randrange(CORPUS):03d}"
            if d not in seen:
                seen.add(d)
                filler.append(d)

        run = filler[:K]
        if rng.random() < 0.75:                       # this query is answerable
            pos = rng.randrange(0, K)
            run[pos] = gold[0]
            if len(gold) > 1 and rng.random() < 0.5:
                other = rng.randrange(0, K)
                if other != pos:
                    run[other] = gold[1]
        runs_a[qid] = list(run)

        # System B: same documents, correct ones promoted for some queries.
        promoted = list(run)
        if rng.random() < 0.45:
            for g in gold:
                if g in promoted:
                    at = promoted.index(g)
                    to = max(0, at - rng.randrange(2, 6))
                    promoted.insert(to, promoted.pop(at))
        runs_b[qid] = promoted

    return {"qrels": qrels, "runs_a": runs_a, "runs_b": runs_b,
            "phenomena": phenomena, "k": K}


def degenerate_cases() -> list[dict]:
    """The inputs that break a harness written only against the happy path."""
    return [
        {
            "name": "no queries",
            "qrels": {}, "runs": {}, "phenomena": {},
        },
        {
            "name": "query with no gold documents",
            "qrels": {"q0": []},
            "runs": {"q0": ["d1", "d2", "d3"]},
            "phenomena": {"q0": "unanswerable"},
        },
        {
            "name": "nothing retrieved",
            "qrels": {"q0": ["d1"]},
            "runs": {"q0": []},
            "phenomena": {"q0": "vocabulary_mismatch"},
        },
        {
            "name": "fewer results than k",
            "qrels": {"q0": ["d2"]},
            "runs": {"q0": ["d1", "d2"]},
            "phenomena": {"q0": "multi_hop"},
        },
        {
            "name": "more gold documents than k",
            "qrels": {"q0": [f"d{i}" for i in range(20)]},
            "runs": {"q0": [f"d{i}" for i in range(10)]},
            "phenomena": {"q0": "near_duplicate"},
        },
        {
            "name": "every result correct",
            "qrels": {"q0": ["d0", "d1"]},
            "runs": {"q0": ["d0", "d1"]},
            "phenomena": {"q0": "multi_hop"},
        },
    ]
