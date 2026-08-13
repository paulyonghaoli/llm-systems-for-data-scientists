"""BM25 on the corpus: what its parameters do, and what it is for.

    python experiments/bm25_behaviour.py
    python experiments/bm25_behaviour.py --json

Four measurements, on the same 176 answerable queries used everywhere else in
Module 3.

**Stopwords.** The plain implementation in `tools/verify_corpus.py` does no
stopword removal, which is the textbook definition and a poor system. Removing
terms that appear in more than a third of the corpus takes overall recall@10
from 0.551 to 0.705, and takes the multi-hop queries from **zero** to one. The
mechanism is worth understanding rather than memorising: those queries are
mostly function words, the shortest documents in the corpus are FAQs, and BM25's
length normalisation rewards short documents for matching anything at all — so
a query full of common words retrieves whichever documents are shortest.

**And it is not free.** Near-duplicate queries get *worse*, 0.37 to 0.23,
because what distinguished six near-identical depot handbooks was partly the
common words. This is the honest shape of the result and it is why the number
reported is a distribution over query types rather than a single average.

**Parameters.** `b` controls how hard length normalisation bites and `k1`
controls how fast term frequency saturates. Both are swept here rather than
described, because their effect depends on the corpus and this corpus has
documents spanning 126 to 1,088 tokens.

**Complementarity.** BM25 reaches 0.551 and dense retrieval 0.614, but the two
disagree constantly: 22 queries are found only by BM25 and 33 only by dense.
Either-one-finds-it reaches 0.739. That gap is the entire argument for lesson
3.4, and it is measured here rather than asserted there.

Everything reads the recorded fixture, so there is no network and no model.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.record_embeddings import mcnemar_exact  # noqa: E402
from tools.verify_corpus import load, tokenize  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"

K = 10
#: A term in more than this fraction of documents carries almost no information
#: about which document you want. Derived from the corpus rather than imported
#: as a word list, so it adapts to the domain -- "consignment" is a stopword
#: here and would not be in a general collection.
STOPWORD_DF = 0.35
B_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
K1_SWEEP = (0.5, 1.2, 1.5, 2.0, 4.0)


class BM25:
    """Okapi BM25 with optional stopword removal.

    Written out rather than imported from a library so every number below
    depends only on this repository, and so the parameters are visible instead
    of being defaults someone else chose.
    """

    def __init__(self, docs: list[dict], stop: frozenset[str] = frozenset(),
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b, self.stop = k1, b, stop
        self.ids = [d["doc_id"] for d in docs]
        self.toks = [[w for w in tokenize(d["text"]) if w not in stop] for d in docs]
        self.lens = [len(t) for t in self.toks]
        self.avg = sum(self.lens) / len(self.lens)
        self.tf = [Counter(t) for t in self.toks]
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.toks):
            for w in set(t):
                self.postings[w].append(i)

    def rank(self, query: str, limit: int = K) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for w in tokenize(query):
            if w in self.stop:
                continue
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings[w]:
                f = self.tf[i][w]
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                scores[i] += idf * f * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [self.ids[i] for i, _ in top]


def stopwords(docs: list[dict]) -> frozenset[str]:
    df: Counter = Counter()
    for d in docs:
        df.update(set(tokenize(d["text"])))
    return frozenset(w for w, c in df.items() if c > STOPWORD_DF * len(docs))


def recall(bm: BM25, queries: list[dict]) -> float:
    hits = sum(bool(set(bm.rank(q["text"])) & set(q["gold_doc_ids"])) for q in queries)
    return hits / len(queries)


def compute() -> dict[str, float]:
    docs, all_queries = load()
    queries = [q for q in all_queries if q["gold_doc_ids"]]
    stop = stopwords(docs)

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_kind[q["phenomenon"]].append(q)

    out: dict[str, float] = {
        "n_documents": len(docs),
        "n_queries": len(queries),
        "k": K,
        "n_stopwords": len(stop),
        "stopword_df_threshold_pct": int(100 * STOPWORD_DF),
    }

    plain = BM25(docs)
    filtered = BM25(docs, stop)

    # Paired test on the headline claim, and on the b sweep's largest gap.
    # 176 queries means one query is 0.57 points, so a few points of movement
    # is a handful of queries changing hands and the aggregate cannot tell
    # that from noise.
    def flags(bm):
        return np.array([float(bool(set(bm.rank(q["text"])) & set(q["gold_doc_ids"])))
                         for q in queries])

    f_plain, f_filtered = flags(plain), flags(filtered)
    t_stop = mcnemar_exact(f_plain, f_filtered)
    out["p_stopwords"] = t_stop["p_value"]
    # mcnemar_exact rounds to four decimals, which turns this p-value into a
    # flat 0.0 and reads as a claim of certainty rather than of a small number.
    # Recomputed here at a precision the result actually supports, rather than
    # changing the shared helper and churning every p-value already cited in
    # lessons 3.1 and 3.2.
    n_disc = t_stop["discordant"]
    hi = max(t_stop["b_only"], t_stop["a_only"])
    tail = sum(comb(n_disc, i) for i in range(hi, n_disc + 1)) / (2 ** n_disc)
    out["p_stopwords_precise"] = f"{min(1.0, 2 * tail):.7f}"
    out["won_stopwords"] = t_stop["b_only"]
    out["lost_stopwords"] = t_stop["a_only"]

    t_b = mcnemar_exact(flags(BM25(docs, stop, b=0.25)), flags(BM25(docs, stop, b=0.75)))
    out["p_b"] = t_b["p_value"]
    out["won_b"] = t_b["b_only"]
    out["lost_b"] = t_b["a_only"]

    out["recall_plain"] = round(recall(plain, queries), 3)
    out["recall_stopworded"] = round(recall(filtered, queries), 3)
    out["stopword_gain_pts"] = round(
        100 * (out["recall_stopworded"] - out["recall_plain"]), 1)

    for kind, qs in by_kind.items():
        out[f"plain_{kind}"] = round(recall(plain, qs), 2)
        out[f"stopworded_{kind}"] = round(recall(filtered, qs), 2)

    # Parameter sweeps, both on the stopworded index so the two effects are
    # not confounded.
    for b in B_SWEEP:
        out[f"recall_b{str(b).replace('.', '')}"] = round(
            recall(BM25(docs, stop, b=b), queries), 3)
    for k1 in K1_SWEEP:
        out[f"recall_k1_{str(k1).replace('.', '')}"] = round(
            recall(BM25(docs, stop, k1=k1), queries), 3)

    # Document length spread, which is what makes b matter here at all.
    lens = sorted(len(tokenize(d["text"])) for d in docs)
    out["doc_len_p10"] = lens[len(lens) // 10]
    out["doc_len_median"] = lens[len(lens) // 2]
    out["doc_len_p90"] = lens[9 * len(lens) // 10]

    # --- complementarity with dense retrieval ------------------------------
    manifest = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    pooling = manifest["pooling"]
    dv = load_pair(f"documents_{pooling}")
    qv = load_pair(f"queries_{pooling}_bare")
    mu = dv.mean(axis=0)
    _, _, comp = np.linalg.svd(dv - mu, full_matrices=False)
    u1 = comp[:1]

    def strip(v: np.ndarray) -> np.ndarray:
        x = v - mu
        y = x - (x @ u1.T) @ u1
        return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)

    dvs, qvs = strip(dv), strip(qv)
    order = [n for n, q in enumerate(all_queries) if q["gold_doc_ids"]]

    lex_hit, dense_hit = [], []
    for n, q in zip(order, queries, strict=True):
        gold = set(q["gold_doc_ids"])
        lex_hit.append(bool(set(plain.rank(q["text"])) & gold))
        top = np.argpartition(-(dvs @ qvs[n]), K)[:K]
        dense_hit.append(bool({docs[j]["doc_id"] for j in top} & gold))

    both = sum(1 for a, b_ in zip(lex_hit, dense_hit, strict=True) if a and b_)
    lex_only = sum(1 for a, b_ in zip(lex_hit, dense_hit, strict=True) if a and not b_)
    dense_only = sum(1 for a, b_ in zip(lex_hit, dense_hit, strict=True) if b_ and not a)
    n = len(queries)
    out["dense_recall"] = round(sum(dense_hit) / n, 3)
    out["both"] = both
    out["lexical_only"] = lex_only
    out["dense_only"] = dense_only
    out["neither"] = n - both - lex_only - dense_only
    out["union_ceiling"] = round((both + lex_only + dense_only) / n, 3)
    out["headroom_pts"] = round(
        100 * (out["union_ceiling"] - max(out["recall_plain"], out["dense_recall"])), 1)
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
          f"recall@{r['k']}\n")
    print(f"stopwords: {r['n_stopwords']} terms above "
          f"{r['stopword_df_threshold_pct']}% document frequency")
    print(f"  plain BM25        {r['recall_plain']:.3f}")
    print(f"  stopwords removed {r['recall_stopworded']:.3f}   "
          f"({r['stopword_gain_pts']:+.1f} points, won {r['won_stopwords']} "
          f"lost {r['lost_stopwords']}, p={r['p_stopwords_precise']})\n")
    print(f"  {'phenomenon':<22} {'plain':>7} {'filtered':>9}")
    for kind in sorted(k[6:] for k in r if k.startswith("plain_")):
        print(f"  {kind:<22} {r['plain_' + kind]:>7.2f} {r['stopworded_' + kind]:>9.2f}")
    print(f"\ndocument length: p10 {r['doc_len_p10']}, median {r['doc_len_median']}, "
          f"p90 {r['doc_len_p90']} tokens")
    print("  b (length normalisation): " + "  ".join(
        f"{b}={r['recall_b' + str(b).replace('.', '')]:.3f}" for b in B_SWEEP))
    print("  k1 (tf saturation):       " + "  ".join(
        f"{k1}={r['recall_k1_' + str(k1).replace('.', '')]:.3f}" for k1 in K1_SWEEP))
    print(f"  b=0.25 against b=0.75: won {r['won_b']} lost {r['lost_b']}, "
          f"p={r['p_b']:.4f}")
    print(f"\nagainst dense retrieval ({r['dense_recall']:.3f}):")
    print(f"  both {r['both']}   lexical only {r['lexical_only']}   "
          f"dense only {r['dense_only']}   neither {r['neither']}")
    print(f"  either finds it: {r['union_ceiling']:.3f} "
          f"({r['headroom_pts']:+.1f} points over the better single retriever)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
