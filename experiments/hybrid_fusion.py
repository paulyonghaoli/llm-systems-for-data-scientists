"""Hybrid retrieval: what fusion actually recovers, and what tuning it costs.

    python experiments/hybrid_fusion.py
    python experiments/hybrid_fusion.py --json

Lesson 3.3 established the opportunity: lexical retrieval and dense retrieval
fail on different queries, and either-one-finds-it reaches far above the better
of the two. This measures how much of that gap fusion closes, using the same
176 answerable queries.

Three results, and the first two point in opposite directions.

**Reciprocal rank fusion with the usual defaults makes things worse.** With
`k = 60` from the original paper and equal weights, fusion scores 0.642 against
0.705 for the lexical retriever on its own. The mechanism is not subtle: RRF
ignores scores entirely and treats both retrievers as equally trustworthy, so
when one is meaningfully better than the other, the weaker one's confident
mistakes are given the same voice as the stronger one's correct answers.

**Weighted RRF with a tuned rank constant does help.** The best configuration
found reaches 0.778, and against lexical alone it wins 13 queries and loses
none.

**And that 0.778 is an overestimate, because it was chosen on the queries it is
reported on.** Twenty-five configurations were compared on 176 queries; picking
the best of them and quoting its score is the benchmark equivalent of reporting
training accuracy. Tuning on half the queries and evaluating on the other half
gives 0.750 — the honest number, and 2.8 points below the in-sample one. That
gap is the measurement this script exists for as much as the fusion result is.

Everything reads the recorded fixture, so there is no network and no model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.bm25_behaviour import BM25, stopwords  # noqa: E402
from experiments.record_embeddings import mcnemar_exact  # noqa: E402
from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"

K = 10
#: How deep each retriever's list goes before fusion. Fusion can only combine
#: what it is given, so this bounds the ceiling.
DEPTH = 100
SEED = 20260812
#: The rank constant from Cormack et al. Its role is to damp the influence of
#: top ranks: large k flattens the contribution of rank 1 against rank 20.
DEFAULT_RRF_K = 60
K_GRID = (1, 5, 10, 20, 60)
W_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def build_runs() -> tuple[list[dict], dict, dict]:
    """Both retrievers' ranked lists, computed once."""
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
    # The stopword-filtered index, because lesson 3.3 measured it as the better
    # lexical retriever and fusing with a known-worse component would be
    # answering a question nobody has.
    bm = BM25(docs, stopwords(docs))

    lex, den = {}, {}
    for n, q in zip(order, queries, strict=True):
        lex[q["query_id"]] = bm.rank(q["text"], DEPTH)
        s = dvs @ qvs[n]
        top = np.argpartition(-s, DEPTH)[:DEPTH]
        den[q["query_id"]] = [ids[i] for i in top[np.argsort(-s[top])]]
    return queries, lex, den


def wrrf(qid: str, lex: dict, den: dict, k: int, w_lex: float) -> list[str]:
    """Weighted reciprocal rank fusion over two ranked lists."""
    scores: dict[str, float] = defaultdict(float)
    for rank, doc in enumerate(lex[qid], start=1):
        scores[doc] += w_lex / (k + rank)
    for rank, doc in enumerate(den[qid], start=1):
        scores[doc] += (1.0 - w_lex) / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in ranked[:K]]


def hits(queries: list[dict], fn) -> np.ndarray:
    return np.array([float(bool(set(fn(q)) & set(q["gold_doc_ids"]))) for q in queries])


def compute() -> dict[str, float]:
    queries, lex, den = build_runs()
    out: dict[str, float] = {"n_queries": len(queries), "k": K, "depth": DEPTH}

    lex_hits = hits(queries, lambda q: lex[q["query_id"]][:K])
    den_hits = hits(queries, lambda q: den[q["query_id"]][:K])
    out["lexical_recall"] = round(float(lex_hits.mean()), 3)
    out["dense_recall"] = round(float(den_hits.mean()), 3)
    out["union_ceiling"] = round(
        float(np.maximum(lex_hits, den_hits).mean()), 3)
    out["headroom_pts"] = round(
        100 * (out["union_ceiling"] - max(out["lexical_recall"], out["dense_recall"])), 1)

    # --- the defaults, which are what most systems ship --------------------
    for k in (DEFAULT_RRF_K, 10):
        f = hits(queries, lambda q, k=k: wrrf(q["query_id"], lex, den, k, 0.5))
        out[f"rrf_equal_k{k}"] = round(float(f.mean()), 3)
    default = hits(queries, lambda q: wrrf(q["query_id"], lex, den, DEFAULT_RRF_K, 0.5))
    t = mcnemar_exact(default, lex_hits)
    out["p_default_vs_lexical"] = t["p_value"]
    out["default_loses_to_lexical_by_pts"] = round(
        100 * (out["lexical_recall"] - out[f"rrf_equal_k{DEFAULT_RRF_K}"]), 1)

    # --- the full grid, scored in-sample ------------------------------------
    grid = [(k, w) for k in K_GRID for w in W_GRID]
    best_score, best_cfg = -1.0, None
    for k, w in grid:
        f = hits(queries, lambda q, k=k, w=w: wrrf(q["query_id"], lex, den, k, w))
        if f.mean() > best_score:
            best_score, best_cfg = float(f.mean()), (k, w)
    out["grid_size"] = len(grid)
    out["best_insample"] = round(best_score, 3)
    out["best_k"] = best_cfg[0]
    out["best_w_lex"] = best_cfg[1]
    best_hits = hits(queries, lambda q: wrrf(q["query_id"], lex, den, *best_cfg))
    t = mcnemar_exact(lex_hits, best_hits)
    out["insample_won"] = t["b_only"]
    out["insample_lost"] = t["a_only"]
    out["p_insample_vs_lexical"] = t["p_value"]

    # --- the same tuning, scored honestly -----------------------------------
    # Two folds: choose the configuration on one half, score it on the other.
    # This is the only number here that estimates what fusion would do on
    # queries it was not tuned against.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(queries))
    half = len(queries) // 2
    pooled_fusion = np.zeros(len(queries))
    pooled_lex = np.zeros(len(queries))
    fold_rows = []
    for fold, (tune_i, test_i) in enumerate(((idx[:half], idx[half:]),
                                             (idx[half:], idx[:half])), start=1):
        tune = [queries[i] for i in tune_i]
        test = [queries[i] for i in test_i]
        pick, pick_score = None, -1.0
        for k, w in grid:
            s = hits(tune, lambda q, k=k, w=w: wrrf(q["query_id"], lex, den, k, w)).mean()
            if s > pick_score:
                pick_score, pick = float(s), (k, w)
        test_hits = hits(test, lambda q, pick=pick: wrrf(q["query_id"], lex, den, *pick))
        test_lex = hits(test, lambda q: lex[q["query_id"]][:K])
        pooled_fusion[test_i] = test_hits
        pooled_lex[test_i] = test_lex
        fold_rows.append((fold, pick, round(pick_score, 3),
                          round(float(test_hits.mean()), 3),
                          round(float(test_lex.mean()), 3)))
        out[f"fold{fold}_k"] = pick[0]
        out[f"fold{fold}_w_lex"] = pick[1]
        out[f"fold{fold}_tune"] = round(pick_score, 3)
        out[f"fold{fold}_heldout"] = round(float(test_hits.mean()), 3)

    out["heldout_recall"] = round(float(pooled_fusion.mean()), 3)
    out["heldout_gain_pts"] = round(100 * float(pooled_fusion.mean() - pooled_lex.mean()), 1)
    out["optimism_pts"] = round(100 * (out["best_insample"] - out["heldout_recall"]), 1)
    t = mcnemar_exact(pooled_lex, pooled_fusion)
    out["heldout_won"] = t["b_only"]
    out["heldout_lost"] = t["a_only"]
    out["p_heldout_vs_lexical"] = t["p_value"]
    # How large a query set would resolve the held-out difference? McNemar's
    # power depends only on the discordant pairs, so this follows from the
    # observed discordant rate and how lopsided the split is. Reported because
    # "not significant" is only actionable alongside "and here is what would
    # settle it".
    discordant = out["heldout_won"] + out["heldout_lost"]
    rate = discordant / len(queries)
    share = out["heldout_won"] / discordant if discordant else 0.5
    z_alpha, z_beta = 1.96, 0.84                       # two-sided 0.05, 80% power
    needed_discordant = (z_alpha + z_beta) ** 2 / (4 * (share - 0.5) ** 2)
    out["discordant_pairs"] = discordant
    out["discordant_rate_pct"] = round(100 * rate, 1)
    out["queries_for_80pct_power"] = int(round(needed_discordant / rate, -1))

    out["headroom_captured_pct"] = round(
        100 * out["heldout_gain_pts"] / out["headroom_pts"], 0)
    out["_folds"] = fold_rows
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    r = compute()
    if args.json:
        r.pop("_folds", None)
        print(json.dumps(r, indent=2))
        return 0

    print(f"{r['n_queries']} answerable queries, recall@{r['k']}, "
          f"fusing top {r['depth']} from each retriever\n")
    print(f"  lexical alone            {r['lexical_recall']:.3f}")
    print(f"  dense alone              {r['dense_recall']:.3f}")
    print(f"  either one finds it      {r['union_ceiling']:.3f}   "
          f"({r['headroom_pts']:+.1f} pts of headroom)\n")
    print(f"  RRF, k=60, equal weights {r['rrf_equal_k60']:.3f}   "
          f"WORSE than lexical alone by {r['default_loses_to_lexical_by_pts']:.1f} pts")
    print(f"  RRF, k=10, equal weights {r['rrf_equal_k10']:.3f}\n")
    print(f"  best of {r['grid_size']} configurations  {r['best_insample']:.3f} "
          f"at k={r['best_k']}, w_lex={r['best_w_lex']}  "
          f"(won {r['insample_won']}, lost {r['insample_lost']}, "
          f"p={r['p_insample_vs_lexical']})")
    print("    ...but chosen on the queries it is scored on\n")
    for fold, pick, tune, held, lexr in r["_folds"]:
        print(f"  fold {fold}: tuned k={pick[0]}, w_lex={pick[1]} on half "
              f"({tune:.3f}) -> held-out {held:.3f}  (lexical {lexr:.3f})")
    print(f"\n  held-out fusion          {r['heldout_recall']:.3f}   "
          f"({r['heldout_gain_pts']:+.1f} pts over lexical, won {r['heldout_won']}, "
          f"lost {r['heldout_lost']}, p={r['p_heldout_vs_lexical']})")
    print(f"  optimism from tuning on the test set: {r['optimism_pts']:+.1f} pts")
    print(f"  headroom captured: {r['headroom_captured_pct']:.0f}%")
    print(f"  {r['discordant_pairs']} discordant pairs ({r['discordant_rate_pct']}% of queries); "
          f"~{r['queries_for_80pct_power']} queries would give 80% power")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
