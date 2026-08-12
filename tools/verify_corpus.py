"""Measure that the corpus contains the phenomena it claims to contain.

PLAN section 10a commits to seven planted phenomena, "each to be verified
rather than asserted". This is that verification, and the distinction is the
whole point: `build_corpus.py` *intends* to plant a vocabulary mismatch when it
writes a query in customer register against a policy in operational register,
but intent is not evidence. A template that quietly shares one rare word
between query and gold document produces a query that BM25 answers easily,
which is no longer a vocabulary mismatch no matter what the label says.

So every phenomenon here is checked by measuring a property that the phenomenon
*implies*, using a real lexical retriever:

    vocabulary_mismatch   BM25 should FAIL -- the gold document should not rank
                          near the top, because query and answer share almost
                          no words.
    lexical_distractor    BM25's top hit should be the WRONG document -- some
                          other document shares the rare term.
    near_duplicate        The gold document should have a very similar sibling.
    superseded            The superseded version should exist, be highly
                          similar to the current one, and not be gold.
    multi_hop             No single gold document should suffice on its own.
    contradiction         A non-authoritative document should state a
                          numerically different answer.
    unanswerable          No document should score well.

Thresholds are set from what each phenomenon *means*, not from what this
corpus happens to produce. If a check fails the corpus gets fixed, not the
threshold -- otherwise this file measures nothing and merely records history.

Run directly, or as gate 25 via `tools/verify.py`.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

# BM25 parameters. The Robertson defaults; nothing here is tuned, because a
# tuned retriever would make "BM25 fails on this query" a statement about the
# tuning rather than about the corpus.
K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    """Plain BM25 over the document collection.

    Written out rather than imported so the numbers below depend on nothing
    outside this repository, and so Module 3 has a reference implementation to
    compare a learner's against.
    """

    def __init__(self, docs: list[dict]) -> None:
        self.ids = [d["doc_id"] for d in docs]
        self.toks = [tokenize(d["text"]) for d in docs]
        self.lens = [len(t) for t in self.toks]
        self.avg = sum(self.lens) / len(self.lens)
        self.tf: list[Counter] = [Counter(t) for t in self.toks]
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(docs)
        self.idf = {
            w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.toks):
            for w in set(t):
                self.postings[w].append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for w in tokenize(query):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings[w]:
                f = self.tf[i][w]
                denom = f + K1 * (1 - B + B * self.lens[i] / self.avg)
                out[i] += idf * f * (K1 + 1) / denom
        return out

    def rank(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return [(self.ids[i], v) for i, v in top]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def gold_rank(ranked: list[tuple[str, float]], gold: set[str]) -> int | None:
    """1-based rank of the first gold document, or None if outside the list."""
    for pos, (doc_id, _) in enumerate(ranked, start=1):
        if doc_id in gold:
            return pos
    return None


def load() -> tuple[list[dict], list[dict]]:
    docs = [json.loads(x) for x in
            (CORPUS / "documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [json.loads(x) for x in
               (CORPUS / "queries.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return docs, queries


def main() -> int:
    if not (CORPUS / "documents.jsonl").exists():
        print("corpus not built; run tools/build_corpus.py", file=sys.stderr)
        return 1

    docs, queries = load()
    by_id = {d["doc_id"]: d for d in docs}
    bm25 = BM25(docs)
    words = {d["doc_id"]: set(tokenize(d["text"])) for d in docs}

    # --- referential integrity, before anything else -----------------------
    # A gold id naming a document that does not exist used to raise KeyError
    # halfway through the phenomenon checks -- found by the mutation test in
    # `test_verify_corpus.py`, which deletes documents. A dangling reference is
    # a corpus defect and deserves a clean report, not a traceback, because it
    # is exactly the failure a future edit to the generator would introduce.
    dangling = sorted({g for q in queries for g in q["gold_doc_ids"] if g not in by_id})
    if dangling:
        print(f"{len(dangling)} gold reference(s) name documents that do not exist:",
              file=sys.stderr)
        for d in dangling[:10]:
            print(f"  - {d}", file=sys.stderr)
        return 1

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        by_kind[q["phenomenon"]].append(q)

    # Ranked results, computed once and shared by every check below.
    ranked = {q["query_id"]: bm25.rank(q["text"], limit=20) for q in queries}

    failures: list[str] = []
    rows: list[tuple[str, int, str, str, bool]] = []

    def record(kind: str, measured: str, expect: str, ok: bool) -> None:
        rows.append((kind, len(by_kind[kind]), measured, expect, ok))
        if not ok:
            failures.append(f"{kind}: measured {measured}, expected {expect}")

    # --- vocabulary_mismatch: BM25 should struggle -------------------------
    # The query is in customer register and the answer in operational register.
    # If BM25 finds it easily, the registers are not actually disjoint and the
    # phenomenon is mislabelled.
    qs = by_kind["vocabulary_mismatch"]
    missed = 0
    for q in qs:
        r = gold_rank(ranked[q["query_id"]], set(q["gold_doc_ids"]))
        missed += r is None or r > 5
    pct = 100 * missed / len(qs)
    record("vocabulary_mismatch", f"{pct:.0f}% have gold outside BM25 top-5", ">=50%", pct >= 50)

    # Overlap should also be low in absolute terms.
    overlaps = []
    for q in qs:
        qt = set(tokenize(q["text"]))
        for g in q["gold_doc_ids"]:
            overlaps.append(len(qt & words[g]) / max(1, len(qt)))
    mean_overlap = sum(overlaps) / len(overlaps)
    record("vocabulary_mismatch", f"mean query-term coverage {mean_overlap:.2f}", "<=0.55",
           mean_overlap <= 0.55)

    # --- lexical_distractor: BM25's top hit should be wrong ----------------
    qs = by_kind["lexical_distractor"]
    fooled = 0
    for q in qs:
        top = ranked[q["query_id"]]
        fooled += bool(top) and top[0][0] not in set(q["gold_doc_ids"])
    pct = 100 * fooled / len(qs)
    record("lexical_distractor", f"{pct:.0f}% have a non-gold BM25 top-1", ">=50%", pct >= 50)

    # --- near_duplicate: gold should have a very similar sibling -----------
    qs = by_kind["near_duplicate"]
    have_twin = 0
    sims = []
    for q in qs:
        best = 0.0
        for g in q["gold_doc_ids"]:
            same_type = [d for d in docs
                         if d["type"] == by_id[g]["type"] and d["doc_id"] != g]
            for other in same_type:
                best = max(best, jaccard(words[g], words[other["doc_id"]]))
        sims.append(best)
        if best >= 0.7:
            have_twin += 1
    pct = 100 * have_twin / len(qs)
    record("near_duplicate", f"{pct:.0f}% have a sibling at Jaccard>=0.70 "
           f"(mean best {sum(sims)/len(sims):.2f})", "100%", pct == 100)

    # --- superseded: old version present, similar, and not gold ------------
    qs = by_kind["superseded"]
    ok_count = 0
    for q in qs:
        gold = set(q["gold_doc_ids"])
        good = True
        for g in gold:
            old_id = by_id[g]["doc_id"].replace("-v2", "-v1")
            old = by_id.get(old_id)
            if old is None or old_id in gold:
                good = False
                break
            if jaccard(words[g], words[old_id]) < 0.5:
                good = False
        ok_count += good
    pct = 100 * ok_count / len(qs)
    record("superseded", f"{pct:.0f}% have a similar non-gold predecessor", "100%", pct == 100)

    # --- multi_hop: no single gold document should suffice -----------------
    # Operationalised as: each gold document, scored alone against the query,
    # is missing content the other supplies -- so the union of two documents
    # covers query terms that neither covers by itself.
    qs = by_kind["multi_hop"]
    genuine = 0
    for q in qs:
        gold = q["gold_doc_ids"]
        if len(gold) < 2:
            continue
        qt = set(tokenize(q["text"]))
        cover = [len(qt & words[g]) for g in gold]
        union = len(qt & set().union(*(words[g] for g in gold)))
        if union > max(cover):
            genuine += 1
    pct = 100 * genuine / len(qs)
    record("multi_hop", f"{pct:.0f}% need both documents for term coverage", ">=90%", pct >= 90)

    # --- contradiction: a non-gold document states a different number ------
    qs = by_kind["contradiction"]
    conflicting = 0
    for q in qs:
        gold_nums: set[str] = set()
        for g in q["gold_doc_ids"]:
            gold_nums |= set(re.findall(r"\b\d+\b", by_id[g]["text"]))
        rival = [d for d in docs if d["type"] == "faq"
                 and d["doc_id"] not in set(q["gold_doc_ids"])]
        if any(set(re.findall(r"\b\d+\b", d["text"])) - gold_nums for d in rival):
            conflicting += 1
    pct = 100 * conflicting / len(qs)
    record("contradiction", f"{pct:.0f}% have a rival document with different numbers",
           ">=90%", pct >= 90)

    # --- unanswerable: nothing should score well ---------------------------
    # Compared against answerable queries rather than an absolute cutoff,
    # because BM25 scores have no meaningful scale of their own.
    qs = by_kind["unanswerable"]
    answerable = [q for q in queries if q["gold_doc_ids"]]
    top_un = [ranked[q["query_id"]][0][1] if ranked[q["query_id"]] else 0.0 for q in qs]
    top_an = [ranked[q["query_id"]][0][1] if ranked[q["query_id"]] else 0.0 for q in answerable]
    mu, ma = sum(top_un) / len(top_un), sum(top_an) / len(top_an)
    record("unanswerable", f"mean top-1 score {mu:.1f} vs {ma:.1f} answerable",
           "strictly lower", mu < ma)
    empty = all(not q["gold_doc_ids"] for q in qs)
    record("unanswerable", f"gold list empty for all {len(qs)}", "100%", empty)

    # --- report ------------------------------------------------------------
    width = max(len(r[0]) for r in rows)
    print(f"corpus: {len(docs)} documents, {len(queries)} queries\n")
    print(f"{'phenomenon':<{width}}  {'n':>3}  {'measured':<52} {'expected':<14} ")
    print("-" * (width + 78))
    for kind, n, measured, expect, ok in rows:
        mark = "ok  " if ok else "FAIL"
        print(f"{kind:<{width}}  {n:>3}  {measured:<52} {expect:<14} {mark}")

    if failures:
        print("\nphenomena not present as claimed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nall seven phenomena measured present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
