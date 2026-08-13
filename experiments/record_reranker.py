"""Record cross-encoder reranker scores for the corpus. RUN BY HAND ONLY.

    pip install -e ".[record]"
    python experiments/record_reranker.py

Writes data/fixtures/reranker/ — one dated, revision-pinned file of scores.
Everything downstream reads it, never the model, so the suite stays offline
(gate 13). Same contract as experiments/record_tiktoken.py and
experiments/record_embeddings.py.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2, Apache-2.0, pinned to an explicit
commit rather than `main`. It is a *cross*-encoder: query and document go into
the model together and one relevance logit comes out, so it can compare them
directly instead of comparing two independently-produced vectors. That is why
it is accurate and also why it cannot be indexed — there is no document vector
to store, and every candidate costs a forward pass at query time.

Candidates are the union of each retriever's top 50, so the recorded scores
support reranking a lexical run, a dense run, or a fusion of them, without
re-recording. Scoring the whole corpus for every query would be 425,744 pairs
and is exactly the thing a reranker is too expensive to do.
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.bm25_behaviour import BM25, stopwords  # noqa: E402
from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"
OUT = ROOT / "data" / "fixtures" / "reranker"

REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
LICENCE = "apache-2.0"
#: How deep into each retriever's list to take candidates. A reranker's cost is
#: linear in this, so it is the main cost dial in a production system.
DEPTH = 50
BATCH = 32
MAX_TOKENS = 512


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def main() -> int:
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError:
        print('recording deps missing. `pip install -e ".[record]"` first.', file=sys.stderr)
        return 2

    docs, all_queries = load()
    queries = [q for q in all_queries if q["gold_doc_ids"]]
    order = [n for n, q in enumerate(all_queries) if q["gold_doc_ids"]]
    by_id = {d["doc_id"]: d for d in docs}
    ids = [d["doc_id"] for d in docs]

    # First-stage candidates: the same two retrievers the lessons use.
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
    bm = BM25(docs, stopwords(docs))

    candidates: dict[str, list[str]] = {}
    for n, q in zip(order, queries, strict=True):
        lex = bm.rank(q["text"], DEPTH)
        s = dvs @ qvs[n]
        top = np.argpartition(-s, DEPTH)[:DEPTH]
        den = [ids[i] for i in top[np.argsort(-s[top])]]
        # Union, in a deterministic order: lexical first, then dense additions.
        seen, merged = set(), []
        for d in lex + den:
            if d not in seen:
                seen.add(d)
                merged.append(d)
        candidates[q["query_id"]] = merged

    total = sum(len(v) for v in candidates.values())
    print(f"{len(queries)} queries, {total} query-document pairs to score "
          f"(top {DEPTH} from each retriever, deduplicated)")

    model_path = hf_hub_download(REPO, "onnx/model.onnx", revision=REVISION)
    tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json", revision=REVISION))
    tok.enable_truncation(MAX_TOKENS)
    tok.enable_padding()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def score(pairs: list[tuple[str, str]]) -> np.ndarray:
        enc = tok.encode_batch(pairs)
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
        }
        return sess.run(None, feed)[0][:, 0]

    OUT.mkdir(parents=True, exist_ok=True)
    lines, done = [], 0
    for q in queries:
        cand = candidates[q["query_id"]]
        scores: list[float] = []
        for start in range(0, len(cand), BATCH):
            batch = [(q["text"], by_id[d]["text"]) for d in cand[start:start + BATCH]]
            scores.extend(float(x) for x in score(batch))
        lines.append(json.dumps({
            "query_id": q["query_id"],
            # Rounded to four decimals: the ranking is unaffected and the file
            # is a third the size.
            "scores": {d: round(s, 4) for d, s in zip(cand, scores, strict=True)},
        }))
        done += len(cand)
        if len(lines) % 40 == 0:
            print(f"  {len(lines)}/{len(queries)} queries, {done}/{total} pairs")

    (OUT / "scores.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "recorded_on": date.today().isoformat(),
        "model": REPO,
        "revision": REVISION,
        "licence": LICENCE,
        "kind": "cross-encoder relevance logit",
        "candidate_depth_per_retriever": DEPTH,
        "max_tokens": MAX_TOKENS,
        "onnxruntime_version": ort.__version__,
        "n_queries": len(queries),
        "n_pairs": total,
        "note": (
            "Relevance logits for the union of each retriever's top "
            f"{DEPTH} candidates, recorded once by experiments/record_reranker.py. "
            "Never called at test time. Higher is more relevant; the scale is "
            "unbounded and not comparable across models."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {(OUT / 'scores.jsonl').relative_to(ROOT)} ({total} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
