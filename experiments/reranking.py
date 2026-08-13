"""Reranking: a large effect that recall@10 cannot see.

    python experiments/reranking.py
    python experiments/reranking.py --json

A cross-encoder scores each (query, document) pair with both texts in the model
at once, so it can compare them directly instead of comparing two vectors
produced in ignorance of each other. It is far more accurate and far too
expensive to run over a corpus, so it reranks a first stage's candidates.

The headline is not the improvement. It is *where* the improvement is.

    metric        lexical   + rerank
    recall@1        0.051      0.250     won 38, lost 3
    recall@3        0.466      0.670     won 41, lost 5
    recall@10       0.705      0.727     won  5, lost 1   -- not significant

Reranking makes the right answer *first* five times as often, and barely
changes whether it is in the top ten at all. That is exactly what a reordering
step should do, and it means the metric a team happens to have chosen decides
whether they can see the effect. A team reporting recall@10 would measure this
reranker, find nothing, and conclude cross-encoders are not worth the latency.

Everything reads `data/fixtures/reranker/`, recorded once by
`experiments/record_reranker.py`, so there is no model and no network here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.bm25_behaviour import BM25, stopwords  # noqa: E402
from experiments.record_embeddings import mcnemar_exact  # noqa: E402
from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"
RR = ROOT / "data" / "fixtures" / "reranker"

DEPTH = 50
CUTOFFS = (1, 3, 5, 10)
#: The fusion configuration lesson 3.4 selected, held out.
RRF_K, RRF_W = 1, 0.7


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def p_floor(p_text: str) -> str:
    """A readable bound for a p-value, for quoting in prose.

    Scientific notation cannot be inlined in a lesson: gate 18 parses the
    number preceding a `computed:` marker and `1e-08` is not one. Rounding to
    four decimals turns these into a flat 0.0, which reads as certainty. A
    bound is both honest and legible.
    """
    p = float(p_text)
    for bound in (0.0001, 0.001, 0.01, 0.05):
        if p < bound:
            return f"{bound:g}"
    return f"{p:.4g}"


def exact_p(a: np.ndarray, b: np.ndarray) -> str:
    """Two-sided exact McNemar p at a precision small values survive.

    `mcnemar_exact` rounds to four decimals, which prints 0.0 for the effects
    measured here and reads as a claim of certainty rather than of a very
    small number.
    """
    t = mcnemar_exact(a, b)
    n, hi = t["discordant"], max(t["b_only"], t["a_only"])
    if n == 0:
        return "1"
    tail = sum(comb(n, i) for i in range(hi, n + 1)) / (2 ** n)
    return f"{min(1.0, 2 * tail):.2g}"


def build() -> tuple[list[dict], dict, dict, dict]:
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
    return queries, lex, den, scores


def compute() -> dict[str, float]:
    queries, lex, den, scores = build()
    manifest = json.loads((RR / "manifest.json").read_text(encoding="utf-8"))

    def fuse(qid: str) -> list[str]:
        sc: dict[str, float] = defaultdict(float)
        for rank, d in enumerate(lex[qid], start=1):
            sc[d] += RRF_W / (RRF_K + rank)
        for rank, d in enumerate(den[qid], start=1):
            sc[d] += (1 - RRF_W) / (RRF_K + rank)
        return [d for d, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]

    def rerank(qid: str, cands: list[str]) -> list[str]:
        s = scores[qid]
        return sorted(cands, key=lambda d: (-s.get(d, -99.0), d))

    runs = {
        "lexical": lambda qid: lex[qid],
        "dense": lambda qid: den[qid],
        "hybrid": fuse,
    }
    reranked = {
        "lexical": lambda qid: rerank(qid, lex[qid]),
        "dense": lambda qid: rerank(qid, den[qid]),
        "hybrid": lambda qid: rerank(qid, fuse(qid)),
    }

    def rec(fn, k: int) -> np.ndarray:
        return np.array([float(bool(set(fn(q["query_id"])[:k]) & set(q["gold_doc_ids"])))
                         for q in queries])

    def mrr(fn, k: int = 10) -> float:
        total = 0.0
        for q in queries:
            gold = set(q["gold_doc_ids"])
            for i, d in enumerate(fn(q["query_id"])[:k], start=1):
                if d in gold:
                    total += 1 / i
                    break
        return total / len(queries)

    out: dict[str, float] = {
        "n_queries": len(queries),
        "depth_per_retriever": DEPTH,
        "model": manifest["model"],
        "n_pairs": manifest["n_pairs"],
        "pairs_per_query": round(manifest["n_pairs"] / len(queries), 1),
    }

    for name in runs:
        for k in CUTOFFS:
            base, after = rec(runs[name], k), rec(reranked[name], k)
            out[f"{name}_at{k}"] = round(float(base.mean()), 3)
            out[f"{name}_rerank_at{k}"] = round(float(after.mean()), 3)
            t = mcnemar_exact(base, after)
            out[f"{name}_won_at{k}"] = t["b_only"]
            out[f"{name}_lost_at{k}"] = t["a_only"]
            out[f"{name}_p_at{k}"] = exact_p(base, after)
            # A bound, for quoting inline. See p_floor.
            out[f"{name}_pfloor_at{k}"] = p_floor(out[f"{name}_p_at{k}"])
        out[f"{name}_mrr"] = round(mrr(runs[name]), 3)
        out[f"{name}_rerank_mrr"] = round(mrr(reranked[name]), 3)

    # What the reranker could reach if it were perfect: is the gold document
    # anywhere in the candidates it was given?
    covered = np.array([float(bool(set(scores[q["query_id"]]) & set(q["gold_doc_ids"])))
                        for q in queries])
    out["candidate_ceiling"] = round(float(covered.mean()), 3)
    out["candidate_ceiling_pct"] = round(100 * float(covered.mean()), 1)
    out["reranker_shortfall_pts"] = round(
        100 * (out["candidate_ceiling"] - out["hybrid_rerank_at10"]), 1)

    out["at1_gain_x"] = round(out["lexical_rerank_at1"] / out["lexical_at1"], 1)
    out["at10_gain_pts"] = round(
        100 * (out["lexical_rerank_at10"] - out["lexical_at10"]), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not (RR / "manifest.json").exists():
        print("reranker fixture missing; run experiments/record_reranker.py "
              "(needs the network)", file=sys.stderr)
        return 1

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"{r['model']}, {r['n_pairs']} recorded pairs "
          f"({r['pairs_per_query']} per query), {r['n_queries']} queries\n")
    for name in ("lexical", "dense", "hybrid"):
        print(f"{name}")
        print(f"  {'cutoff':>8} {'before':>8} {'after':>8} {'won':>5} {'lost':>5} {'p':>9}")
        for k in CUTOFFS:
            print(f"  {'@' + str(k):>8} {r[f'{name}_at{k}']:>8.3f} "
                  f"{r[f'{name}_rerank_at{k}']:>8.3f} {r[f'{name}_won_at{k}']:>5} "
                  f"{r[f'{name}_lost_at{k}']:>5} {r[f'{name}_p_at{k}']:>9}")
        print(f"  {'MRR@10':>8} {r[f'{name}_mrr']:>8.3f} {r[f'{name}_rerank_mrr']:>8.3f}\n")
    print(f"candidate ceiling {r['candidate_ceiling']:.3f} — the gold document is "
          f"somewhere in the candidates this often;")
    print(f"the reranked hybrid reaches {r['hybrid_rerank_at10']:.3f}, "
          f"{r['reranker_shortfall_pts']:.1f} points short of it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
