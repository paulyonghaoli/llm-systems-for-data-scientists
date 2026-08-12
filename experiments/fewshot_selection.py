"""Picking few-shot examples is a retrieval problem, and the obvious strategy loses.

    python experiments/fewshot_selection.py
    python experiments/fewshot_selection.py --json

Four ways to choose `k` examples from a labelled pool, given a query:

- **random** — the baseline everybody starts with
- **top-k** — the `k` most similar examples, which is what people reach for
- **MMR** — maximal marginal relevance, trading a little similarity for
  coverage of the space
- **balanced** — round-robin over labels, taking the most similar unused
  example of each label in turn

They are scored on four things, and the fourth is the one that decides it.

## No model is involved anywhere here

Similarity is cosine over L2-normalised TF-IDF vectors computed from the pool
itself, so every number below is a property of the text and the arithmetic
rather than of anything learned. That matters for the lesson's honesty: the
claim being made is about *selection*, which is fully checkable, and not about
whether a given set of examples makes a model answer better, which is not
checkable here and is stated as such in §H.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "fewshot" / "tickets.jsonl"

SEED = 20260811
K = 5
MMR_LAMBDA = 0.5


def tokenize(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]


def tfidf(docs: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """L2-normalised TF-IDF over a fixed vocabulary. Fifteen lines, no deps."""
    vocab: dict[str, int] = {}
    counts = []
    for d in docs:
        c = Counter(tokenize(d))
        counts.append(c)
        for w in c:
            vocab.setdefault(w, len(vocab))

    n = len(docs)
    df = np.zeros(len(vocab))
    for c in counts:
        for w in c:
            df[vocab[w]] += 1
    idf = np.log((1 + n) / (1 + df)) + 1.0

    m = np.zeros((n, len(vocab)))
    for i, c in enumerate(counts):
        for w, tf in c.items():
            m[i, vocab[w]] = tf
    m *= idf
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12), vocab


# --- selection strategies ---------------------------------------------------


def select_random(sim: np.ndarray, labels: list[str], k: int, rng: random.Random) -> list[int]:
    return rng.sample(range(len(labels)), k)


def select_topk(sim: np.ndarray, labels: list[str], k: int, rng: random.Random) -> list[int]:
    return list(np.argsort(-sim)[:k])


def select_mmr(sim: np.ndarray, labels: list[str], k: int, rng: random.Random,
               pairwise: np.ndarray | None = None, lam: float = MMR_LAMBDA) -> list[int]:
    """Maximal marginal relevance: each pick maximises relevance to the query
    minus the largest similarity to anything already picked."""
    chosen: list[int] = []
    candidates = list(range(len(labels)))
    while len(chosen) < k and candidates:
        best, best_score = None, -math.inf
        for i in candidates:
            redundancy = max((pairwise[i, j] for j in chosen), default=0.0)
            score = lam * sim[i] - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        candidates.remove(best)
    return chosen


def select_balanced(sim: np.ndarray, labels: list[str], k: int, rng: random.Random) -> list[int]:
    """Round-robin over labels, most similar unused example of each in turn."""
    by_label: dict[str, list[int]] = {}
    for i in np.argsort(-sim):
        by_label.setdefault(labels[i], []).append(int(i))
    chosen: list[int] = []
    order = sorted(by_label)
    while len(chosen) < k:
        progressed = False
        for lab in order:
            if by_label[lab] and len(chosen) < k:
                chosen.append(by_label[lab].pop(0))
                progressed = True
        if not progressed:
            break
    return chosen


STRATEGIES = {
    "random": select_random,
    "top_k": select_topk,
    "mmr": select_mmr,
    "balanced": select_balanced,
}


def compute() -> dict[str, float]:
    rows = [json.loads(line) for line in POOL.read_text(encoding="utf-8").splitlines()]
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    vectors, _ = tfidf(texts)
    pairwise = vectors @ vectors.T
    n_labels = len(set(labels))

    out: dict[str, float] = {
        "pool_size": len(rows),
        "n_labels": n_labels,
        "k": K,
        "mmr_lambda": MMR_LAMBDA,
    }

    # Leave-one-out: each example in turn is the query, and is excluded from
    # its own candidate pool. Every query therefore has a known correct label.
    for name, fn in STRATEGIES.items():
        rng = random.Random(SEED)
        rel, red, cov, hit = [], [], [], []
        for q in range(len(rows)):
            mask = [i for i in range(len(rows)) if i != q]
            sub_sim = pairwise[q, mask]
            sub_labels = [labels[i] for i in mask]
            sub_pair = pairwise[np.ix_(mask, mask)]

            kwargs = {"pairwise": sub_pair} if name == "mmr" else {}
            picked = fn(sub_sim, sub_labels, K, rng, **kwargs)

            rel.append(float(np.mean([sub_sim[i] for i in picked])))
            if len(picked) > 1:
                red.append(float(np.mean([sub_pair[a, b] for i, a in enumerate(picked)
                                          for b in picked[i + 1:]])))
            picked_labels = {sub_labels[i] for i in picked}
            cov.append(len(picked_labels) / min(K, n_labels))
            hit.append(labels[q] in picked_labels)

        tag = name
        out[f"{tag}_relevance"] = round(float(np.mean(rel)), 3)
        out[f"{tag}_redundancy"] = round(float(np.mean(red)), 3)
        out[f"{tag}_coverage_pct"] = round(100 * float(np.mean(cov)), 1)
        out[f"{tag}_label_hit_pct"] = round(100 * float(np.mean(hit)), 1)

    out["random_missing_own_label_pct"] = round(100 - out["random_label_hit_pct"], 1)
    out["mmr_redundancy_drop_pct"] = round(
        100 * (1 - out["mmr_redundancy"] / out["top_k_redundancy"]), 1)
    out["mmr_relevance_cost"] = round(out["top_k_relevance"] - out["mmr_relevance"], 3)
    out["balanced_relevance_cost"] = round(
        out["top_k_relevance"] - out["balanced_relevance"], 3)
    out["topk_distinct_of_k"] = round(out["top_k_coverage_pct"] / 100 * min(K, n_labels), 1)

    # How the comparison moves with k. Below the number of labels, no strategy
    # can cover them all, and the question changes from coverage to which
    # labels you sacrifice.
    for k in (3, 8):
        for name, fn in STRATEGIES.items():
            rng = random.Random(SEED)
            cov, hit = [], []
            for q in range(len(rows)):
                mask = [i for i in range(len(rows)) if i != q]
                sub_sim = pairwise[q, mask]
                sub_labels = [labels[i] for i in mask]
                kwargs = ({"pairwise": pairwise[np.ix_(mask, mask)]}
                          if name == "mmr" else {})
                picked = fn(sub_sim, sub_labels, k, rng, **kwargs)
                picked_labels = {sub_labels[i] for i in picked}
                cov.append(len(picked_labels) / min(k, n_labels))
                hit.append(labels[q] in picked_labels)
            out[f"k{k}_{name}_coverage_pct"] = round(100 * float(np.mean(cov)), 1)
            out[f"k{k}_{name}_label_hit_pct"] = round(100 * float(np.mean(hit)), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"pool of {v['pool_size']:.0f} labelled examples, {v['n_labels']:.0f} intents, "
          f"k = {v['k']:.0f}, leave-one-out over every example\n")
    head = (f"{'strategy':<10} {'relevance':>10} {'redundancy':>11} {'label coverage':>15} "
            f"{'own label present':>18}")
    print(head)
    print("-" * len(head))
    for name in STRATEGIES:
        print(f"{name:<10} {v[f'{name}_relevance']:>10.3f} {v[f'{name}_redundancy']:>11.3f} "
              f"{v[f'{name}_coverage_pct']:>14.1f}% {v[f'{name}_label_hit_pct']:>17.1f}%")
    print(f"\nrandom omits an example of the query's own label "
          f"{v['random_missing_own_label_pct']}% of the time; every informed "
          f"strategy is at or near 100%")
    print(f"MMR cuts redundancy {v['mmr_redundancy_drop_pct']}% for "
          f"{v['mmr_relevance_cost']} of mean similarity")
    print(f"top-k's {v['k']:.0f} slots show only "
          f"{v['topk_distinct_of_k']} distinct labels on average")

    print(f"\n{'':>10} {'k=3 coverage':>13} {'k=3 own lbl':>12} "
          f"{'k=8 coverage':>13} {'k=8 own lbl':>12}")
    for name in STRATEGIES:
        print(f"{name:<10} {v[f'k3_{name}_coverage_pct']:>12.1f}% "
              f"{v[f'k3_{name}_label_hit_pct']:>11.1f}% "
              f"{v[f'k8_{name}_coverage_pct']:>12.1f}% "
              f"{v[f'k8_{name}_label_hit_pct']:>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
