"""Chunking, measured: a cost this corpus can show and a benefit it cannot.

    python experiments/chunking_tradeoff.py
    python experiments/chunking_tradeoff.py --json

Every guide to retrieval-augmented generation tells you to chunk. This
measures what chunking does to retrieval quality on the shipped fixture, at
four sizes, against the same documents left whole.

**It is worse at every size, monotonically.** Document-level retrieval scores
0.614; 510-token chunks 0.585; 128-token chunks 0.358. Smaller chunks are worse
than larger ones and every chunking is worse than not chunking.

**The mechanism is candidate inflation.** A document is scored by the best of
its chunks, and a maximum over more samples is larger for *every* document
including the wrong ones. Splitting 2,419 documents into 7,082 chunks does not
give the right document a better chance of matching; it gives 2,418 wrong
documents more chances each.

**And the benefit chunking exists for is nearly absent here.** Chunking pays
when a document does not fit the embedding model's context, because the tail
of a long document is otherwise simply discarded. This corpus has 31 documents
over the 510-token budget out of 2,419. The 40 queries whose gold document is
truncated were measured separately, and chunking did not help them either —
so the honest conclusion is not "chunking is bad" but "this corpus can show
chunking's costs and cannot show its benefits, and a corpus of long documents
would say something different".

That distinction is the lesson. A result measured where the mechanism does not
apply is not evidence about the mechanism.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"

K = 10
#: The model's usable content budget: 512 positions less [CLS] and [SEP].
LIMIT = 510
#: How many top units to pool from. Large enough that pooling, not truncation,
#: decides the result.
POOL_DEPTH = 400
CHUNKINGS = ("fixed_510", "fixed_256_ov64", "fixed_256", "fixed_128")


def fit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lesson 3.1's transform, fitted to whatever units are being indexed.

    Fitted per granularity rather than reused from the document vectors: the
    mean of 7,082 chunks is not the mean of 2,419 documents, and applying one
    population's transform to another is the mismatch lesson 3.1 warns about.
    """
    mu = v.mean(axis=0)
    _, _, comp = np.linalg.svd(v - mu, full_matrices=False)
    return mu, comp[:1]


def apply(v: np.ndarray, mu: np.ndarray, u1: np.ndarray) -> np.ndarray:
    x = v - mu
    y = x - (x @ u1.T) @ u1
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)


def hits(units: np.ndarray, qvecs: np.ndarray, owner: list[str],
         order: list[int], queries: list[dict], pooling: str = "max") -> np.ndarray:
    """Per-query hit flags, pooling unit scores up to their document."""
    out = []
    for n, q in zip(order, queries, strict=True):
        s = units @ qvecs[n]
        top = np.argpartition(-s, min(POOL_DEPTH, len(s) - 1))[:POOL_DEPTH]
        if pooling == "max":
            best: dict[str, float] = defaultdict(lambda: -9.9)
            for j in top:
                best[owner[j]] = max(best[owner[j]], float(s[j]))
        else:
            best = defaultdict(float)
            for j in top:
                best[owner[j]] += float(s[j])
        ranked = sorted(best, key=lambda d: -best[d])[:K]
        out.append(float(bool(set(ranked) & set(q["gold_doc_ids"]))))
    return np.array(out)


def compute() -> dict[str, float]:
    docs, all_queries = load()
    queries = [q for q in all_queries if q["gold_doc_ids"]]
    order = [n for n, q in enumerate(all_queries) if q["gold_doc_ids"]]
    manifest = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    pooling = manifest["pooling"]
    qv = load_pair(f"queries_{pooling}_bare")

    # True model-token length per document, read off the widest chunking's spans.
    meta510 = [json.loads(x) for x in
               (EMB / "chunks_fixed_510_meta.jsonl").read_text(encoding="utf-8").splitlines() if x]
    tok_len: dict[str, int] = defaultdict(int)
    for m in meta510:
        tok_len[m["doc_id"]] = max(tok_len[m["doc_id"]], m["end"])
    truncated = {d for d, n in tok_len.items() if n > LIMIT}
    long_q = [i for i, q in enumerate(queries) if set(q["gold_doc_ids"]) & truncated]
    short_q = [i for i, q in enumerate(queries) if not (set(q["gold_doc_ids"]) & truncated)]

    out: dict[str, float] = {
        "n_documents": len(docs),
        "n_queries": len(queries),
        "k": K,
        "limit_tokens": LIMIT,
        "docs_over_limit": len(truncated),
        "queries_gold_truncated": len(long_q),
        "queries_gold_fits": len(short_q),
    }

    dv = load_pair(f"documents_{pooling}")
    mu, u1 = fit(dv)
    owner_docs = [d["doc_id"] for d in docs]
    doc_hits = hits(apply(dv, mu, u1), apply(qv, mu, u1), owner_docs, order, queries)
    out["units_document"] = len(docs)
    out["recall_document"] = round(float(doc_hits.mean()), 3)
    out["recall_document_truncated"] = round(float(doc_hits[long_q].mean()), 3)
    out["recall_document_fits"] = round(float(doc_hits[short_q].mean()), 3)

    for name in CHUNKINGS:
        meta = [json.loads(x) for x in
                (EMB / f"chunks_{name}_meta.jsonl").read_text(encoding="utf-8").splitlines() if x]
        cv = load_pair(f"chunks_{name}")
        owner = [m["doc_id"] for m in meta]
        m2, u2 = fit(cv)
        cvt, qvt = apply(cv, m2, u2), apply(qv, m2, u2)
        h = hits(cvt, qvt, owner, order, queries)
        out[f"units_{name}"] = len(cv)
        out[f"recall_{name}"] = round(float(h.mean()), 3)
        out[f"recall_{name}_truncated"] = round(float(h[long_q].mean()), 3)
        out[f"recall_{name}_fits"] = round(float(h[short_q].mean()), 3)
        out[f"recall_{name}_sumpool"] = round(
            float(hits(cvt, qvt, owner, order, queries, "sum").mean()), 3)
        out[f"units_per_doc_{name}"] = round(len(cv) / len(docs), 2)

    out["best_chunking"] = max(CHUNKINGS, key=lambda n: out[f"recall_{n}"])
    out["cost_of_best_chunking_pts"] = round(
        100 * (out["recall_document"] - out[f"recall_{out['best_chunking']}"]), 1)
    out["cost_of_128_pts"] = round(
        100 * (out["recall_document"] - out["recall_fixed_128"]), 1)
    out["pct_docs_over_limit"] = round(100 * len(truncated) / len(docs), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"{r['n_documents']} documents, {r['n_queries']} answerable queries, "
          f"recall@{r['k']}, max-pooled to document level\n")
    print(f"{'granularity':<17} {'units':>7} {'per doc':>8} {'recall':>7} "
          f"{'sum-pool':>9} {'gold cut':>9} {'gold fits':>10}")
    print(f"{'document':<17} {r['units_document']:>7} {1.0:>8.2f} "
          f"{r['recall_document']:>7.3f} {'-':>9} "
          f"{r['recall_document_truncated']:>9.3f} {r['recall_document_fits']:>10.3f}")
    for name in CHUNKINGS:
        print(f"{name:<17} {r[f'units_{name}']:>7} {r[f'units_per_doc_{name}']:>8.2f} "
              f"{r[f'recall_{name}']:>7.3f} {r[f'recall_{name}_sumpool']:>9.3f} "
              f"{r[f'recall_{name}_truncated']:>9.3f} {r[f'recall_{name}_fits']:>10.3f}")
    print(f"\nbest chunking is {r['best_chunking']}, still "
          f"{r['cost_of_best_chunking_pts']:.1f} points below leaving documents whole")
    print(f"128-token chunks cost {r['cost_of_128_pts']:.1f} points")
    print(f"\ndocuments over the {r['limit_tokens']}-token budget: "
          f"{r['docs_over_limit']} of {r['n_documents']} ({r['pct_docs_over_limit']}%) — "
          f"the condition chunking exists for is nearly absent here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
