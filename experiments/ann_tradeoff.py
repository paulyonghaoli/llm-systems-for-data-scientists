"""What an approximate index actually costs, measured twice on the same corpus.

    python experiments/ann_tradeoff.py
    python experiments/ann_tradeoff.py --json

Exact search over this corpus is one matrix multiply: 2,419 documents by 384
dimensions, and every candidate is scored. An approximate index gives that up
in exchange for touching fewer vectors, and the question is always what the
exchange rate is.

The usual way to quote it is **agreement with exact search** — of the ten
documents brute force would have returned, how many did the index find. That
number is easy to measure, appears in every benchmark, and is not the thing
you care about. What you care about is whether the *answer* is still in the
results, which is end-task recall against the labelled queries. This script
measures both, on the same runs, so the gap between them is visible rather
than assumed.

The index here is IVF: cluster the corpus once with k-means, assign every
document to its nearest centroid, and at query time score only the documents
in the `nprobe` nearest clusters. It is the approximate structure that is
simple enough to write out in full, and the tradeoff it exposes — probe more
lists, scan more vectors, lose less — is the same one HNSW's `ef_search`
exposes through a different mechanism.

Everything runs against the recorded fixture, so no network and no index
library. Two vector sets are compared: the raw embeddings, and the same
embeddings after lesson 3.1's centre-and-project transform, because k-means
partitions by Euclidean distance and lesson 3.1 established that these vectors
sit in a narrow cone with a large shared offset. Whether that offset hurts the
clustering is a measurable question and this answers it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"

SEED = 20260812
#: Roughly sqrt(n), the usual starting point for IVF list count.
NLIST = 64
KMEANS_ITERS = 25
K = 10
PROBES = (1, 2, 4, 8, 16, 32, 64)


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def kmeans(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Lloyd's algorithm, seeded and fixed-iteration so the result is stable.

    Initialised by sampling distinct rows rather than at random in space,
    because a centroid that starts outside the cone attracts nothing and stays
    empty for every iteration.
    """
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(len(vectors), size=k, replace=False)].copy()
    for _ in range(KMEANS_ITERS):
        # Squared euclidean, expanded so it is one matmul rather than a loop.
        d = (vectors ** 2).sum(1)[:, None] - 2 * vectors @ centroids.T + (centroids ** 2).sum(1)
        assign = d.argmin(axis=1)
        for c in range(k):
            members = vectors[assign == c]
            if len(members):
                centroids[c] = members.mean(axis=0)
    return centroids


def build_lists(vectors: np.ndarray, centroids: np.ndarray) -> list[np.ndarray]:
    d = (vectors ** 2).sum(1)[:, None] - 2 * vectors @ centroids.T + (centroids ** 2).sum(1)
    assign = d.argmin(axis=1)
    return [np.flatnonzero(assign == c) for c in range(len(centroids))]


def exact_top_k(docs: np.ndarray, q: np.ndarray, k: int = K) -> np.ndarray:
    scores = docs @ q
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


def ivf_top_k(docs: np.ndarray, centroids: np.ndarray, lists: list[np.ndarray],
              q: np.ndarray, nprobe: int, k: int = K) -> tuple[np.ndarray, int]:
    """Top k from the nprobe nearest lists, and how many vectors were scanned."""
    order = np.argsort(-(centroids @ q))[:nprobe]
    candidates = np.concatenate([lists[c] for c in order]) if len(order) else np.array([], int)
    if len(candidates) == 0:
        return np.array([], dtype=int), 0
    scores = docs[candidates] @ q
    n = min(k, len(candidates))
    top = np.argpartition(-scores, n - 1)[:n] if n < len(scores) else np.arange(len(scores))
    top = top[np.argsort(-scores[top])]
    return candidates[top], len(candidates)


def evaluate(docs: np.ndarray, queries: np.ndarray, gold: list[set[int]],
             label: str) -> dict:
    centroids = kmeans(docs, NLIST, SEED)
    lists = build_lists(docs, centroids)
    sizes = [len(x) for x in lists]

    exact = [exact_top_k(docs, q) for q in queries]
    exact_recall = float(np.mean([bool(g & set(e.tolist()))
                                  for e, g in zip(exact, gold, strict=True)]))

    rows = {}
    for nprobe in PROBES:
        agree = []
        hits = []
        scanned = []
        for n, q in enumerate(queries):
            got, seen = ivf_top_k(docs, centroids, lists, q, nprobe)
            agree.append(len(set(got.tolist()) & set(exact[n].tolist())) / K)
            hits.append(bool(gold[n] & set(got.tolist())))
            scanned.append(seen)
        rows[nprobe] = {
            "agreement_at_10": round(float(np.mean(agree)), 3),
            "recall_at_10": round(float(np.mean(hits)), 3),
            "scanned_pct": round(100 * float(np.mean(scanned)) / len(docs), 1),
        }
    return {
        "label": label,
        "exact_recall_at_10": round(exact_recall, 3),
        "empty_lists": int(sum(1 for s in sizes if s == 0)),
        "largest_list": int(max(sizes)),
        "smallest_list": int(min(sizes)),
        "list_size_sd": round(float(np.std(sizes)), 1),
        "probes": rows,
    }


def compute() -> dict[str, float]:
    manifest = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    pooling = manifest["pooling"]
    doc_rows, query_rows = load()
    dv = load_pair(f"documents_{pooling}")
    qv = load_pair(f"queries_{pooling}_bare")

    index = {d["doc_id"]: i for i, d in enumerate(doc_rows)}
    order = [n for n, q in enumerate(query_rows) if q["gold_doc_ids"]]
    gold = [{index[g] for g in query_rows[n]["gold_doc_ids"]} for n in order]
    qa = qv[order]

    mu = dv.mean(axis=0)
    centred = dv - mu
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    u1 = components[:1]

    def strip(v: np.ndarray) -> np.ndarray:
        x = v - mu
        return normalise(x - (x @ u1.T) @ u1)

    raw = evaluate(dv, qa, gold, "raw")
    transformed = evaluate(strip(dv), strip(qa), gold, "centred, minus 1 component")

    out: dict[str, float] = {
        "n_documents": len(doc_rows),
        "n_queries": len(gold),
        "nlist": NLIST,
        "k": K,
    }
    for name, r in (("raw", raw), ("tf", transformed)):
        out[f"{name}_exact_recall"] = r["exact_recall_at_10"]
        out[f"{name}_empty_lists"] = r["empty_lists"]
        out[f"{name}_largest_list"] = r["largest_list"]
        out[f"{name}_list_size_sd"] = r["list_size_sd"]
        for p, row in r["probes"].items():
            out[f"{name}_agree_p{p}"] = row["agreement_at_10"]
            out[f"{name}_recall_p{p}"] = row["recall_at_10"]
            out[f"{name}_scanned_p{p}"] = row["scanned_pct"]

    # Deterministic cost of the exact baseline. Wall-clock is deliberately not
    # recorded here: gate 18 re-runs this and diffs the output, and a timing
    # would differ on every machine and every run. Multiply-adds and bytes are
    # properties of the arithmetic, so they are the same everywhere.
    out["exact_multiply_adds"] = len(doc_rows) * dv.shape[1]
    out["exact_mb_float32"] = round(len(doc_rows) * dv.shape[1] * 4 / 1e6, 2)
    out["exact_mb_int8"] = round(len(doc_rows) * dv.shape[1] / 1e6, 2)
    out["mb_float32_at_1m_docs"] = round(1_000_000 * dv.shape[1] * 4 / 1e9, 2)

    # The headline: the probe count at which end-task recall is indistinguishable
    # from exact, and what fraction of the corpus that scans.
    for p in PROBES:
        if out[f"tf_recall_p{p}"] >= out["tf_exact_recall"] - 0.005:
            out["tf_probes_for_parity"] = p
            out["tf_scanned_at_parity"] = out[f"tf_scanned_p{p}"]
            break
    for p in PROBES:
        if out[f"raw_recall_p{p}"] >= out["raw_exact_recall"] - 0.005:
            out["raw_probes_for_parity"] = p
            out["raw_scanned_at_parity"] = out[f"raw_scanned_p{p}"]
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not (EMB / "manifest.json").exists():
        print("embedding fixture missing; run experiments/record_embeddings.py",
              file=sys.stderr)
        return 1

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"{r['n_documents']} documents, {r['n_queries']} queries, "
          f"IVF with {r['nlist']} lists\n")
    for name, title in (("raw", "raw vectors"),
                        ("tf", "centred, minus 1 component")):
        print(f"{title}")
        print(f"  exact recall@10 {r[f'{name}_exact_recall']:.3f}   "
              f"empty lists {r[f'{name}_empty_lists']}   "
              f"largest {r[f'{name}_largest_list']}   "
              f"size sd {r[f'{name}_list_size_sd']}")
        print(f"  {'probes':>7} {'scanned':>8} {'agreement@10':>13} {'recall@10':>10}")
        for p in PROBES:
            print(f"  {p:>7} {r[f'{name}_scanned_p{p}']:>7.1f}% "
                  f"{r[f'{name}_agree_p{p}']:>13.3f} {r[f'{name}_recall_p{p}']:>10.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
