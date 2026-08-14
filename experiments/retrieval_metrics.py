"""Retrieval metrics: which system is best depends on which one you picked.

    python experiments/retrieval_metrics.py
    python experiments/retrieval_metrics.py --json

Five systems built across Module 3, scored on the same 176 queries with the
five metrics people actually report. The table is the lesson: the metrics do
not agree, and the disagreement is not a rounding artefact.

**Lexical retrieval beats dense on recall@10 and loses to it on every
rank-aware metric.** 0.705 against 0.614 on recall; 0.287 against 0.356 on
MRR. A team comparing BM25 with embeddings reaches opposite conclusions
depending on a choice usually made before any measurement, by whoever wrote the
evaluation harness first.

**And the two differences are not equally supported.** Lexical's recall
advantage is 43 queries won against 27, at p = 0.0722 — suggestive and not
significant. Dense's MRR advantage has a bootstrap confidence interval that
excludes zero. So the honest summary is not "it depends on the metric" but
something sharper: one of the two claims survives scrutiny and the other does
not, and only reporting both metrics reveals which.

**precision@10 cannot exceed 0.118 here.** 145 of the 176 queries have exactly
one correct document, so nine of ten slots are necessarily wrong and the metric
is capped near 0.1 by construction. Lexical's 0.076 looks like a catastrophe
and is 64% of the attainable maximum. A precision@k reported without its
ceiling is not a measurement of anything.

Everything reads the recorded fixtures, so there is no model and no network.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.bm25_behaviour import BM25, stopwords  # noqa: E402
from experiments.record_embeddings import mcnemar_exact  # noqa: E402
from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"
RR = ROOT / "data" / "fixtures" / "reranker"

K = 10
DEPTH = 50
SEED = 20260813
BOOTSTRAP = 2000
METRICS = ("recall", "precision", "mrr", "map", "ndcg")
SYSTEMS = ("lexical", "dense", "hybrid", "lexical+rerank", "hybrid+rerank")


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def build_runs() -> tuple[list[dict], dict]:
    docs, all_queries = load()
    queries = [q for q in all_queries if q["gold_doc_ids"]]
    order = [n for n, q in enumerate(all_queries) if q["gold_doc_ids"]]

    manifest = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    pooling = manifest["pooling"]
    dv = load_pair(f"documents_{pooling}")
    qv = load_pair(f"queries_{pooling}_bare")
    mu = dv.mean(axis=0)
    _, _, comp = np.linalg.svd(dv - mu, full_matrices=False)
    u1 = comp[:1]

    def strip(v: np.ndarray) -> np.ndarray:
        x = v - mu
        return normalise(x - (x @ u1.T) @ u1)

    dvs, qvs = strip(dv), strip(qv)
    ids = [d["doc_id"] for d in docs]
    bm = BM25(docs, stopwords(docs))

    lex, den = {}, {}
    for n, q in zip(order, queries, strict=True):
        lex[q["query_id"]] = bm.rank(q["text"], DEPTH)
        s = dvs @ qvs[n]
        top = np.argpartition(-s, DEPTH)[:DEPTH]
        den[q["query_id"]] = [ids[i] for i in top[np.argsort(-s[top])]]

    scores = {}
    for line in (RR / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            scores[row["query_id"]] = row["scores"]

    def fuse(qid: str) -> list[str]:
        sc: dict[str, float] = defaultdict(float)
        for rank, d in enumerate(lex[qid], start=1):
            sc[d] += 0.7 / (1 + rank)
        for rank, d in enumerate(den[qid], start=1):
            sc[d] += 0.3 / (1 + rank)
        return [d for d, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]

    def rerank(cands: list[str], qid: str) -> list[str]:
        s = scores[qid]
        return sorted(cands, key=lambda d: (-s.get(d, -99.0), d))

    runs = {
        "lexical": lambda q: lex[q],
        "dense": lambda q: den[q],
        "hybrid": fuse,
        "lexical+rerank": lambda q: rerank(lex[q], q),
        "hybrid+rerank": lambda q: rerank(fuse(q), q),
    }
    return queries, runs


def per_query(fn, queries: list[dict], metric: str, k: int = K) -> np.ndarray:
    """One score per query, so differences can be tested rather than eyeballed."""
    out = []
    for q in queries:
        gold = set(q["gold_doc_ids"])
        hits = [1.0 if d in gold else 0.0 for d in fn(q["query_id"])[:k]]
        if metric == "recall":
            out.append(1.0 if any(hits) else 0.0)
        elif metric == "precision":
            out.append(sum(hits) / k)
        elif metric == "mrr":
            v = 0.0
            for i, h in enumerate(hits, start=1):
                if h:
                    v = 1 / i
                    break
            out.append(v)
        elif metric == "map":
            found, acc = 0, 0.0
            for i, h in enumerate(hits, start=1):
                if h:
                    found += 1
                    acc += found / i
            out.append(acc / min(len(gold), k))
        elif metric == "ndcg":
            dcg = sum(h / math.log2(i + 1) for i, h in enumerate(hits, start=1))
            idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
            out.append(dcg / idcg)
    return np.array(out)


def compute() -> dict[str, float]:
    queries, runs = build_runs()
    rng = np.random.default_rng(SEED)

    out: dict[str, float] = {"n_queries": len(queries), "k": K}

    scores: dict[str, dict[str, np.ndarray]] = {}
    for name, fn in runs.items():
        scores[name] = {m: per_query(fn, queries, m) for m in METRICS}
        for m in METRICS:
            out[f"{name.replace('+', '_')}_{m}"] = round(float(scores[name][m].mean()), 3)

    # Which system wins under each metric? The disagreement is the point.
    winners = {}
    for m in METRICS:
        winners[m] = max(SYSTEMS, key=lambda s: scores[s][m].mean())
        out[f"winner_{m}"] = winners[m]
    out["distinct_winners"] = len(set(winners.values()))

    # The precision ceiling. With one correct document and ten slots, nine are
    # necessarily wrong, so the metric cannot approach 1 however good the
    # system is.
    sizes = Counter(len(q["gold_doc_ids"]) for q in queries)
    out["queries_single_gold"] = sizes[1]
    out["precision_ceiling"] = round(
        sum(min(len(q["gold_doc_ids"]), K) / K for q in queries) / len(queries), 3)
    out["lexical_precision_pct_of_ceiling"] = round(
        100 * out["lexical_precision"] / out["precision_ceiling"], 0)

    # The flip, tested. Recall is binary per query so McNemar applies; MRR is
    # continuous, so a bootstrap over paired differences is the right tool.
    t = mcnemar_exact(scores["dense"]["recall"], scores["lexical"]["recall"])
    out["recall_lexical_won"] = t["b_only"]
    out["recall_lexical_lost"] = t["a_only"]
    out["p_recall_flip"] = t["p_value"]

    diff = scores["lexical"]["mrr"] - scores["dense"]["mrr"]
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                     for _ in range(BOOTSTRAP)])
    out["mrr_diff"] = round(float(diff.mean()), 3)
    out["mrr_ci_low"] = round(float(np.percentile(boot, 2.5)), 3)
    out["mrr_ci_high"] = round(float(np.percentile(boot, 97.5)), 3)
    out["mrr_ci_excludes_zero"] = bool(out["mrr_ci_high"] < 0 or out["mrr_ci_low"] > 0)

    # How much do the metrics agree about system ordering? Spearman over the
    # five systems' means, averaged across metric pairs.
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        r = [0.0] * len(vals)
        for pos, i in enumerate(order, start=1):
            r[i] = pos
        return r

    means = {m: [scores[s][m].mean() for s in SYSTEMS] for m in METRICS}
    rho = []
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1:]:
            ra, rb = ranks(means[a]), ranks(means[b])
            n = len(ra)
            d2 = sum((x - y) ** 2 for x, y in zip(ra, rb, strict=True))
            rho.append(1 - 6 * d2 / (n * (n * n - 1)))
    out["mean_rank_correlation"] = round(float(np.mean(rho)), 3)
    out["min_rank_correlation"] = round(float(np.min(rho)), 3)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"{r['n_queries']} queries, cutoff {r['k']}\n")
    print(f"{'system':<16}" + "".join(f"{m:>10}" for m in METRICS))
    for s in SYSTEMS:
        key = s.replace("+", "_")
        print(f"{s:<16}" + "".join(f"{r[f'{key}_{m}']:>10.3f}" for m in METRICS))
    print("\nbest system by metric:")
    for m in METRICS:
        print(f"  {m:<10} {r[f'winner_{m}']}")
    print(f"  -> {r['distinct_winners']} different systems win under 5 metrics")
    print(f"\nprecision@{r['k']} ceiling {r['precision_ceiling']:.3f} "
          f"({r['queries_single_gold']} of {r['n_queries']} queries have one gold "
          f"document); lexical reaches {r['lexical_precision_pct_of_ceiling']:.0f}% of it")
    print("\nlexical vs dense")
    print(f"  recall@{r['k']}: lexical won {r['recall_lexical_won']}, "
          f"lost {r['recall_lexical_lost']}, p={r['p_recall_flip']}")
    print(f"  MRR@{r['k']}   : diff {r['mrr_diff']:+.3f}, "
          f"95% CI [{r['mrr_ci_low']:+.3f}, {r['mrr_ci_high']:+.3f}]"
          f"{'  excludes zero' if r['mrr_ci_excludes_zero'] else ''}")
    print(f"\nrank correlation between metrics: mean {r['mean_rank_correlation']:.3f}, "
          f"min {r['min_rank_correlation']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
