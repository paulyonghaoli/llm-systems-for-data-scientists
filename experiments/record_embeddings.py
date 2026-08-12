"""Record real embeddings for the retrieval corpus. RUN BY HAND ONLY.

    pip install -e ".[record]"
    python experiments/record_embeddings.py

Writes data/fixtures/embeddings/ — int8 vectors plus a dated, revision-pinned
manifest. Everything downstream reads those files, never the model, so the test
suite stays offline (gate 13) and no number in a lesson can move because a hub
download changed under us. Same contract as experiments/record_tiktoken.py.

Model: BAAI/bge-small-en-v1.5, MIT licensed, pinned to an explicit commit sha
rather than `main`. 384 dimensions, 512 max positions, ONNX export driven
through onnxruntime so no torch install is needed.

Two choices this script makes by measurement rather than by assumption:

  * **Pooling.** The model card specifies CLS pooling, and most embedding code
    defaults to mean pooling. A three-sentence probe disagreed with the card,
    which is exactly the kind of result that is too small to act on. Both
    poolings are read off the *same* forward pass, so evaluating them on the
    real 200-query benchmark costs one extra matrix slice.
  * **The query instruction prefix.** bge recommends prefixing queries (not
    passages) with an instruction for short-query retrieval. Whether it helps
    on this corpus is measurable, so it is measured.

Both results are written into the manifest so the lessons can cite them
offline.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "fixtures" / "embeddings"

REPO = "BAAI/bge-small-en-v1.5"
#: Pinned by commit, not by `main`. A tag can move; a sha cannot.
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
LICENCE = "mit"
DIM = 384

#: The model has 512 position embeddings, and [CLS] and [SEP] occupy two of
#: them, so the usable content budget is 510 rather than 512. Chunk sizes are
#: named for the content they hold, because that is the number a learner is
#: reasoning about when they choose one.
MAX_CONTENT_TOKENS = 510

#: bge's recommended prefix for the *query* side of short-query retrieval.
#: Passages are never prefixed.
INSTRUCTION = "Represent this sentence for searching relevant passages: "

#: Chunkings the Module 3 lesson compares. Fixed-size in model tokens, because
#: the 510-token ceiling above is denominated in model tokens and no other unit
#: makes that constraint visible.
CHUNKINGS = {
    "fixed_128": (128, 0),
    "fixed_256": (256, 0),
    "fixed_256_ov64": (256, 64),
    "fixed_510": (510, 0),
}

BATCH = 32


# --- model ----------------------------------------------------------------

class Embedder:
    """bge-small over onnxruntime, returning both poolings from one pass."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(REPO, "onnx/model.onnx", revision=REVISION)
        tok_path = hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
        self.tok = Tokenizer.from_file(tok_path)
        self.tok.enable_truncation(512)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.ort_version = ort.__version__

    def encode_ids(self, text: str) -> list[int]:
        """Content token ids, without [CLS]/[SEP]. Used for chunking."""
        self.tok.no_truncation()
        self.tok.no_padding()
        ids = self.tok.encode(text, add_special_tokens=False).ids
        self.tok.enable_truncation(512)
        self.tok.enable_padding()
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def _forward(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        am = np.array([e.attention_mask for e in enc], dtype=np.int64)
        tt = np.zeros_like(ids)
        hidden = self.sess.run(
            None, {"input_ids": ids, "attention_mask": am, "token_type_ids": tt}
        )[0]
        cls = hidden[:, 0]
        mask = am[..., None].astype(np.float32)
        mean = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        return cls, mean

    def embed(self, texts: list[str], label: str = "") -> tuple[np.ndarray, np.ndarray]:
        """Both poolings, L2-normalised, for a list of texts.

        Sorted by length so each batch pads to its own longest member rather
        than to the longest in the corpus; on a corpus whose documents span
        150-1200 tokens that is most of the runtime.
        """
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        cls_out = np.zeros((len(texts), DIM), dtype=np.float32)
        mean_out = np.zeros((len(texts), DIM), dtype=np.float32)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            c, m = self._forward([texts[i] for i in idx])
            cls_out[idx] = c
            mean_out[idx] = m
            if label and (start // BATCH) % 20 == 0:
                done = min(start + BATCH, len(order))
                print(f"  {label}: {done}/{len(texts)}", flush=True)
        return normalise(cls_out), normalise(mean_out)


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# --- int8 ------------------------------------------------------------------

def quantise(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-vector int8, with the scales kept alongside.

    Per-vector rather than one global scale: the vectors are L2-normalised, so
    a vector whose mass sits in few dimensions has a much larger peak than one
    that spreads evenly, and a global scale would quantise the flat ones far
    too coarsely.
    """
    scale = np.abs(v).max(axis=1) / 127.0
    scale = np.maximum(scale, 1e-12).astype(np.float32)
    q = np.rint(v / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def dequantise(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return normalise(q.astype(np.float32) * scale[:, None])


# --- retrieval scoring -----------------------------------------------------

def per_query_hits(doc_vecs: np.ndarray, query_vecs: np.ndarray, doc_ids: list[str],
                   queries: list[dict], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Per-query hit flags and reciprocal ranks, over the answerable queries.

    Returned per query rather than aggregated because choosing between two
    configurations on aggregate recall alone is the mistake lesson 0.3 is
    about: the two systems are scored on the *same* queries, so the pairing is
    information and throwing it away wastes most of the power available.
    """
    index = {d: i for i, d in enumerate(doc_ids)}
    hits: list[float] = []
    rrs: list[float] = []
    for qi, q in enumerate(queries):
        gold = {index[g] for g in q["gold_doc_ids"] if g in index}
        if not gold:
            continue
        sims = doc_vecs @ query_vecs[qi]
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        found = [pos for pos, d in enumerate(top, start=1) if d in gold]
        hits.append(1.0 if found else 0.0)
        rrs.append(1.0 / found[0] if found else 0.0)
    return np.array(hits), np.array(rrs)


def evaluate(doc_vecs: np.ndarray, query_vecs: np.ndarray, doc_ids: list[str],
             queries: list[dict], k: int = 10) -> dict:
    """Recall@k and MRR@k over the answerable queries."""
    hits, rrs = per_query_hits(doc_vecs, query_vecs, doc_ids, queries, k)
    n = len(hits)
    return {"n": n, "recall_at_10": float(hits.mean()), "mrr_at_10": float(rrs.mean())}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired hit/miss flags.

    Exact rather than the chi-square or normal approximation because the
    discordant count here is around ten, and the approximation is unreliable
    below roughly 25. Only the queries where the two systems disagree carry
    any information about which is better — the ones they both get right, or
    both get wrong, cancel.
    """
    from math import comb

    b_only = int(((b > 0) & (a == 0)).sum())
    a_only = int(((a > 0) & (b == 0)).sum())
    n = b_only + a_only
    if n == 0:
        return {"b_only": 0, "a_only": 0, "discordant": 0, "p_value": 1.0}
    hi = max(b_only, a_only)
    tail = sum(comb(n, i) for i in range(hi, n + 1)) / (2 ** n)
    return {
        "b_only": b_only,
        "a_only": a_only,
        "discordant": n,
        "p_value": round(min(1.0, 2 * tail), 4),
    }


# --- chunking --------------------------------------------------------------

def chunk_document(ids: list[int], size: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) token spans for one document."""
    if size > MAX_CONTENT_TOKENS:
        raise ValueError(f"chunk size {size} exceeds the {MAX_CONTENT_TOKENS}-token budget")
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than the chunk size")
    spans = []
    start = 0
    while start < len(ids):
        spans.append((start, min(start + size, len(ids))))
        if start + size >= len(ids):
            break
        start += step
    return spans or [(0, 0)]


def main() -> int:
    try:
        import onnxruntime  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        print('recording deps missing. `pip install -e ".[record]"` first.', file=sys.stderr)
        return 2

    if not (CORPUS / "documents.jsonl").exists():
        print("corpus not built; run tools/build_corpus.py", file=sys.stderr)
        return 1

    docs = [json.loads(x) for x in
            (CORPUS / "documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [json.loads(x) for x in
               (CORPUS / "queries.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    doc_ids = [d["doc_id"] for d in docs]
    print(f"{len(docs)} documents, {len(queries)} queries")

    emb = Embedder()
    OUT.mkdir(parents=True, exist_ok=True)

    # --- decide pooling and prefix, on the real benchmark -------------------
    print("\nembedding documents (truncated to 512) for the pooling decision")
    doc_cls, doc_mean = emb.embed([d["text"] for d in docs], label="docs")

    q_text = [q["text"] for q in queries]
    q_cls_bare, q_mean_bare = emb.embed(q_text, label="queries")
    q_cls_instr, q_mean_instr = emb.embed([INSTRUCTION + t for t in q_text], label="queries+instr")

    combos = {
        "cls_bare": (doc_cls, q_cls_bare),
        "cls_instruction": (doc_cls, q_cls_instr),
        "mean_bare": (doc_mean, q_mean_bare),
        "mean_instruction": (doc_mean, q_mean_instr),
    }
    table = {name: evaluate(dv, qv, doc_ids, queries) for name, (dv, qv) in combos.items()}
    print(f"\n{'combination':<18} {'recall@10':>10} {'mrr@10':>8}")
    for name, r in table.items():
        print(f"{name:<18} {r['recall_at_10']:>10.3f} {r['mrr_at_10']:>8.3f}")

    best = max(table, key=lambda k: (table[k]["recall_at_10"], table[k]["mrr_at_10"]))

    # The model card specifies CLS with no query prefix, so that is the
    # baseline any departure from it has to beat -- and beat by more than
    # sampling noise. A difference of a few points on 176 queries is a handful
    # of queries changing hands, which is exactly the situation where an
    # aggregate comparison misleads.
    baseline = "cls_bare"
    hits_base, _ = per_query_hits(*combos[baseline], doc_ids, queries)
    hits_best, _ = per_query_hits(*combos[best], doc_ids, queries)
    test = mcnemar_exact(hits_base, hits_best)
    significant = test["p_value"] < 0.05
    print(f"\n{best} vs {baseline} (model card default): "
          f"{test['b_only']} queries won, {test['a_only']} lost, "
          f"p = {test['p_value']:.4f}")

    if not significant:
        print(f"not significant at 0.05 — keeping the model card default "
              f"({baseline}) rather than chasing an unreplicated difference")
        best = baseline

    pooling, prefix_mode = best.split("_")
    use_instruction = prefix_mode == "instruction"
    print(f"chosen: {best}")

    doc_vecs = doc_cls if pooling == "cls" else doc_mean
    if pooling == "cls":
        q_vecs = q_cls_instr if use_instruction else q_cls_bare
    else:
        q_vecs = q_mean_instr if use_instruction else q_mean_bare

    # Both poolings are kept, not just the winner. They cost under 2 MB
    # together, they make the comparison above reproducible with no network,
    # and Module 3 gets to hand the learner a real measurement to redo rather
    # than a number to take on trust.
    for name, vecs in (("documents_cls", doc_cls), ("documents_mean", doc_mean),
                       ("queries_cls_bare", q_cls_bare),
                       ("queries_mean_bare", q_mean_bare),
                       ("queries_cls_instruction", q_cls_instr),
                       ("queries_mean_instruction", q_mean_instr)):
        qi, si = quantise(vecs)
        np.save(OUT / f"{name}_int8.npy", qi)
        np.save(OUT / f"{name}_scale.npy", si)

    # --- chunk and embed ----------------------------------------------------
    token_ids = [emb.encode_ids(d["text"]) for d in docs]
    print(f"\ncorpus is {sum(len(t) for t in token_ids):,} model tokens")

    chunk_report = {}
    for name, (size, overlap) in CHUNKINGS.items():
        texts: list[str] = []
        meta: list[dict] = []
        for d, ids in zip(docs, token_ids, strict=True):
            for start, end in chunk_document(ids, size, overlap):
                texts.append(emb.decode(ids[start:end]))
                meta.append({"doc_id": d["doc_id"], "start": start, "end": end})
        print(f"\n{name}: {len(texts):,} chunks")
        c_cls, c_mean = emb.embed(texts, label=name)
        vecs = c_cls if pooling == "cls" else c_mean

        q, scale = quantise(vecs)
        np.save(OUT / f"chunks_{name}_int8.npy", q)
        np.save(OUT / f"chunks_{name}_scale.npy", scale)
        (OUT / f"chunks_{name}_meta.jsonl").write_text(
            "".join(json.dumps(m) + "\n" for m in meta), encoding="utf-8")

        fidelity = float((normalise(vecs) * dequantise(q, scale)).sum(axis=1).mean())
        chunk_report[name] = {
            "size": size, "overlap": overlap, "n_chunks": len(texts),
            "int8_cosine_fidelity": round(fidelity, 6),
        }
        print(f"  int8 fidelity {fidelity:.6f}")

    # --- document and query vectors ----------------------------------------
    dq, ds = quantise(doc_vecs)
    qq, qs = quantise(q_vecs)
    np.save(OUT / "documents_int8.npy", dq)
    np.save(OUT / "documents_scale.npy", ds)
    np.save(OUT / "queries_int8.npy", qq)
    np.save(OUT / "queries_scale.npy", qs)

    doc_fidelity = float((doc_vecs * dequantise(dq, ds)).sum(axis=1).mean())
    after = evaluate(dequantise(dq, ds), dequantise(qq, qs), doc_ids, queries)

    manifest = {
        "recorded_on": date.today().isoformat(),
        "model": REPO,
        "revision": REVISION,
        "licence": LICENCE,
        "dim": DIM,
        "max_position_embeddings": 512,
        "max_content_tokens": MAX_CONTENT_TOKENS,
        "onnxruntime_version": emb.ort_version,
        "pooling": pooling,
        "query_instruction": INSTRUCTION if use_instruction else None,
        "note": (
            "Real embeddings for data/corpus, recorded once by "
            "experiments/record_embeddings.py and shipped int8. Never called at "
            "test time. Pooling and query-prefix were chosen by measuring all "
            "four combinations on the 200-query benchmark, not by assumption."
        ),
        "pooling_decision": table,
        "pooling_significance": {
            "baseline": "cls_bare",
            "test": "exact two-sided McNemar on recall@10 hit flags",
            **test,
            "significant_at_0.05": significant,
        },
        "chosen": best,
        "documents": {
            "n": len(docs),
            "order": "matches data/corpus/documents.jsonl line order",
            "int8_cosine_fidelity": round(doc_fidelity, 6),
        },
        "queries": {
            "n": len(queries),
            "order": "matches data/corpus/queries.jsonl line order",
        },
        "retrieval_after_quantisation": after,
        "chunkings": chunk_report,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\ndocument int8 fidelity {doc_fidelity:.6f}")
    print(f"dense recall@10 after quantisation {after['recall_at_10']:.3f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
