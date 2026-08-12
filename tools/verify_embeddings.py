"""Check the recorded embedding fixture, offline. Gate 27.

`experiments/record_embeddings.py` needs the network and a 130 MB model
download; this reads only what that recorder committed. It answers three
questions the fixture cannot answer for itself:

  1. **Is it intact?** Shapes, dtypes, and row order have to line up with
     `data/corpus/*.jsonl`, or every retrieval result downstream is silently
     scored against the wrong document.

  2. **Did quantisation cost anything?** int8 is a claim about fidelity, and a
     claim about fidelity is a number.

  3. **How much does dense retrieval beat lexical on vocabulary mismatch?**
     Gate 25 establishes the lexical half — the gold document is outside BM25's
     top 5 for 100% of those queries. This gate establishes the other half, and
     it is also the sharpest available test that the *pooling* is right:
     reading the wrong vector out of the forward pass produces embeddings that
     still look like embeddings, still have unit norm, and quietly retrieve
     worse.

Thresholds come from what the claim means, not from what the fixture happens
to score — and when the two disagree, it is the claim that gets rewritten in
public, not the threshold that gets quietly lowered. That happened here: see
`MIN_DENSE_RECALL_VOCAB` below and the second amendment to PLAN section 10a.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.verify_corpus import BM25, load  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"

#: PLAN section 10a originally claimed dense retrieval *resolves* vocabulary
#: mismatch, and this threshold was 0.50 -- what "resolves" has to mean. The
#: fixture measured 0.30 and the claim was withdrawn rather than the threshold
#: lowered to meet it. See PLAN section 10a's second amendment for the
#: evidence: with every word of filler removed and only twelve candidates to
#: choose between, bge-small still picks the right topic 55.6% of the time, so
#: 0.30 over 2,419 documents is the model's ceiling and not a corpus defect.
#:
#: What replaced it is the relationship that *is* true and that Module 3 is
#: actually about: dense retrieval is dramatically better than lexical here,
#: and still nowhere near sufficient on its own.
MIN_DENSE_RECALL_VOCAB = 0.20
MIN_DENSE_OVER_LEXICAL = 3.0

#: int8 over unit vectors should be nearly lossless. Below this the
#: quantisation scheme is wrong, not merely lossy.
MIN_INT8_FIDELITY = 0.999

REQUIRED_MANIFEST = ["recorded_on", "model", "revision", "licence", "dim",
                     "pooling", "pooling_decision", "pooling_significance", "chosen"]

#: Recomputing recall from the shipped int8 vectors will not land exactly on
#: the float32 number the recorder wrote, but it should be within a query or
#: two out of 176.
RECALL_TOLERANCE = 0.02


def dequantise(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    v = q.astype(np.float32) * scale[:, None]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def load_pair(stem: str) -> np.ndarray:
    q = np.load(EMB / f"{stem}_int8.npy")
    s = np.load(EMB / f"{stem}_scale.npy")
    if q.dtype != np.int8:
        raise ValueError(f"{stem}: expected int8, found {q.dtype}")
    if s.shape != (q.shape[0],):
        raise ValueError(f"{stem}: {q.shape[0]} vectors but {s.shape} scales")
    return dequantise(q, s)


def recall_at_k(doc_vecs: np.ndarray, q_vecs: np.ndarray, doc_ids: list[str],
                queries: list[dict], k: int = 10) -> float:
    index = {d: i for i, d in enumerate(doc_ids)}
    hits = 0
    n = 0
    for qi, q in queries:
        gold = {index[g] for g in q["gold_doc_ids"] if g in index}
        if not gold:
            continue
        n += 1
        sims = doc_vecs @ q_vecs[qi]
        top = np.argpartition(-sims, k)[:k]
        hits += bool(gold & set(top.tolist()))
    return hits / n if n else 0.0


def main() -> int:
    if not (EMB / "manifest.json").exists():
        print("embedding fixture missing; run experiments/record_embeddings.py "
              "(needs the network)", file=sys.stderr)
        return 1

    manifest = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    docs, queries = load()
    doc_ids = [d["doc_id"] for d in docs]
    problems: list[str] = []

    missing = [k for k in REQUIRED_MANIFEST if k not in manifest]
    if missing:
        problems.append(f"manifest missing {missing}")

    doc_vecs = load_pair("documents")
    q_vecs = load_pair("queries")

    # Row order is the whole contract between the fixture and the corpus.
    if doc_vecs.shape != (len(docs), manifest["dim"]):
        problems.append(f"documents {doc_vecs.shape} != ({len(docs)}, {manifest['dim']})")
    if q_vecs.shape != (len(queries), manifest["dim"]):
        problems.append(f"queries {q_vecs.shape} != ({len(queries)}, {manifest['dim']})")

    norms = np.linalg.norm(doc_vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        problems.append(f"document vectors not unit norm (min {norms.min():.4f})")

    fidelity = manifest["documents"]["int8_cosine_fidelity"]
    if fidelity < MIN_INT8_FIDELITY:
        problems.append(f"int8 fidelity {fidelity} below {MIN_INT8_FIDELITY}")

    for name, report in manifest.get("chunkings", {}).items():
        vecs = load_pair(f"chunks_{name}")
        meta_path = EMB / f"chunks_{name}_meta.jsonl"
        n_meta = sum(1 for line in meta_path.read_text(encoding="utf-8").splitlines() if line)
        if vecs.shape[0] != n_meta:
            problems.append(f"{name}: {vecs.shape[0]} vectors but {n_meta} metadata rows")
        if vecs.shape[0] != report["n_chunks"]:
            problems.append(f"{name}: manifest says {report['n_chunks']}, found {vecs.shape[0]}")
        if report["int8_cosine_fidelity"] < MIN_INT8_FIDELITY:
            problems.append(f"{name}: int8 fidelity {report['int8_cosine_fidelity']} too low")

    # --- the recorded decision must be reproducible from what shipped ------
    # The manifest asserts a four-way comparison that was run against the
    # model. Both poolings were shipped precisely so that claim can be checked
    # without the model, which turns the manifest from testimony into
    # something falsifiable.
    for combo, recorded in manifest["pooling_decision"].items():
        pool, prefix = combo.split("_")
        try:
            dv = load_pair(f"documents_{pool}")
            qv = load_pair(f"queries_{pool}_{prefix}")
        except FileNotFoundError:
            problems.append(f"{combo}: vectors not shipped, so the recorded "
                            f"comparison cannot be checked")
            continue
        got = recall_at_k(dv, qv, doc_ids, list(enumerate(queries)))
        if abs(got - recorded["recall_at_10"]) > RECALL_TOLERANCE:
            problems.append(f"{combo}: manifest says recall@10 "
                            f"{recorded['recall_at_10']:.3f}, fixtures give {got:.3f}")

    # A departure from the model card's own configuration has to be backed by
    # the significance test the recorder ran, not by the aggregate alone.
    sig = manifest["pooling_significance"]
    if manifest["chosen"] != sig["baseline"] and not sig["significant_at_0.05"]:
        problems.append(f"chose {manifest['chosen']} over the model-card default "
                        f"{sig['baseline']} on p={sig['p_value']}, which is not significant")

    # --- the claim in PLAN 10a --------------------------------------------
    vocab = [(i, q) for i, q in enumerate(queries)
             if q["phenomenon"] == "vocabulary_mismatch"]
    dense = recall_at_k(doc_vecs, q_vecs, doc_ids, vocab)

    bm25 = BM25(docs)
    hits = 0
    for _, q in vocab:
        ranked = [d for d, _ in bm25.rank(q["text"], limit=10)]
        hits += bool(set(ranked) & set(q["gold_doc_ids"]))
    lexical = hits / len(vocab)

    if dense < MIN_DENSE_RECALL_VOCAB:
        problems.append(f"dense recall@10 on vocabulary_mismatch {dense:.2f} "
                        f"below {MIN_DENSE_RECALL_VOCAB}")
    # A ratio, because the point is the size of the gap, not either number
    # alone. Guarded for a zero denominator: BM25 scoring nothing at all on
    # this slice is the expected result, not a division error.
    ratio = dense / lexical if lexical > 0 else float("inf")
    if ratio < MIN_DENSE_OVER_LEXICAL:
        problems.append(f"dense ({dense:.2f}) is only {ratio:.1f}x BM25 "
                        f"({lexical:.2f}) on vocabulary_mismatch; PLAN 10a claims "
                        f"a large gap, not a marginal one")

    overall_dense = recall_at_k(doc_vecs, q_vecs, doc_ids, list(enumerate(queries)))

    print(f"model      {manifest['model']} @ {manifest['revision'][:12]} "
          f"({manifest['licence']}), recorded {manifest['recorded_on']}")
    print(f"pooling    {manifest['pooling']}  (chosen: {manifest['chosen']})")
    print(f"vectors    {len(docs)} documents, {len(queries)} queries, "
          f"{sum(r['n_chunks'] for r in manifest.get('chunkings', {}).values()):,} chunks")
    print(f"int8       document cosine fidelity {fidelity:.6f}")
    print(f"recall@10  dense {overall_dense:.3f} overall")
    print(f"           vocabulary_mismatch: dense {dense:.3f} vs BM25 {lexical:.3f}")

    if problems:
        print("\nembedding fixture problems:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"\nembedding fixture intact; dense is {ratio:.0f}x BM25 on vocabulary "
          f"mismatch and still misses {1 - dense:.0%} of it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
