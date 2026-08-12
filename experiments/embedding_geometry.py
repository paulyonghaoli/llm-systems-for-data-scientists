"""Embedding space is not shaped the way the cosine-similarity story implies.

    python experiments/embedding_geometry.py
    python experiments/embedding_geometry.py --json

Everything here is measured on the recorded fixture in
`data/fixtures/embeddings/`, so it needs no network and no model: these are
properties of 2,419 real 384-dimension vectors from `bge-small-en-v1.5`, not of
a simulation. `experiments/record_embeddings.py` produced them once, by hand.

Three results, in increasing order of how much they should change what you do.

**Nothing is dissimilar.** Two documents drawn at random have a cosine
similarity around 0.84, and the *least* similar pair in twenty thousand draws
is still above 0.55. The vectors do not spread over the sphere; they occupy a
narrow cone in one corner of it. Any intuition that "cosine 0.8 means closely
related" is calibrated against a geometry this model does not have.

**384 dimensions is not 384 dimensions.** Half the variance across the corpus
lies in *three* directions, and the single largest component holds nearly a
third of it on its own. Most of the nominal dimensionality is doing very little
work.

**One line of post-processing buys more than most prompt engineering.**
Subtracting the mean and projecting out the single dominant direction lifts
recall@10 from 0.489 to 0.614 on the corpus benchmark. Removing a *second*
direction significantly undoes it. The optimum is sharp, it is at one, and
neither the improvement nor the sharpness is something you would guess.

Every comparison is an exact paired McNemar test, because these differences are
a handful of queries changing hands out of 176 and the aggregate alone cannot
tell a real effect from a coin flip. That distinction does real work here:
centering *by itself* looks like a 4.5-point gain and does not survive the test.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.record_embeddings import mcnemar_exact  # noqa: E402
from tools.verify_corpus import load  # noqa: E402
from tools.verify_embeddings import load_pair  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "fixtures" / "embeddings"

#: Fixed so the sampled pairs are the same on every run. Gate 11 re-runs this
#: and diffs the output.
SEED = 20260812
PAIRS = 20_000
K = 10


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def random_pair_cosines(vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cosine between randomly drawn distinct pairs."""
    i = rng.integers(0, len(vectors), PAIRS)
    j = rng.integers(0, len(vectors), PAIRS)
    keep = i != j
    return (vectors[i[keep]] * vectors[j[keep]]).sum(axis=1)


def hit_flags(docs: np.ndarray, queries: np.ndarray, gold: list[set[int]],
              k: int = K) -> np.ndarray:
    """1.0 where a gold document appears in the top k, per query."""
    out = np.zeros(len(gold), dtype=float)
    for n, g in enumerate(gold):
        top = np.argpartition(-(docs @ queries[n]), k)[:k]
        out[n] = float(bool(g & set(top.tolist())))
    return out


def compute() -> dict[str, float]:
    manifest = json.loads((EMB / "manifest.json").read_text(encoding="utf-8"))
    pooling = manifest["pooling"]
    doc_rows, query_rows = load()
    dv = load_pair(f"documents_{pooling}")
    qv = load_pair(f"queries_{pooling}_bare")

    index = {d["doc_id"]: i for i, d in enumerate(doc_rows)}
    answerable = [q for q in query_rows if q["gold_doc_ids"]]
    order = [n for n, q in enumerate(query_rows) if q["gold_doc_ids"]]
    gold = [{index[g] for g in q["gold_doc_ids"]} for q in answerable]
    qa = qv[order]

    rng = np.random.default_rng(SEED)
    out: dict[str, float] = {
        "n_documents": len(doc_rows),
        "n_queries_answerable": len(answerable),
        "dim": int(dv.shape[1]),
        "k": K,
    }

    # --- 1. the cone --------------------------------------------------------
    raw_pairs = random_pair_cosines(dv, rng)
    out["random_pair_cosine_mean"] = round(float(raw_pairs.mean()), 3)
    out["random_pair_cosine_min"] = round(float(raw_pairs.min()), 3)
    out["random_pair_cosine_sd"] = round(float(raw_pairs.std()), 3)

    # A query sitting on its own gold document scores *lower* than two
    # unrelated documents score against each other. Absolute cosine carries no
    # meaning across different kinds of text.
    gold_sims = np.array([qa[n] @ dv[i] for n, g in enumerate(gold) for i in g])
    qi = rng.integers(0, len(qa), PAIRS)
    di = rng.integers(0, len(dv), PAIRS)
    rand_sims = (qa[qi] * dv[di]).sum(axis=1)
    out["query_gold_cosine_mean"] = round(float(gold_sims.mean()), 3)
    out["query_random_cosine_mean"] = round(float(rand_sims.mean()), 3)
    out["query_gold_margin"] = round(float(gold_sims.mean() - rand_sims.mean()), 3)

    # --- 2. effective dimensionality ---------------------------------------
    centred = dv - dv.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    share = singular ** 2 / (singular ** 2).sum()
    cumulative = np.cumsum(share)
    out["top_component_share_pct"] = round(100 * float(share[0]), 1)
    for frac in (0.5, 0.9, 0.95):
        n = int(np.searchsorted(cumulative, frac) + 1)
        out[f"dims_for_{int(frac * 100)}pct_variance"] = n

    # --- 3. centring and component removal ---------------------------------
    mu = dv.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)

    def strip(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Centre, then project out the first n principal directions."""
        d, q = dv - mu, qa - mu
        if n:
            p = components[:n]
            d = d - (d @ p.T) @ p
            q = q - (q @ p.T) @ p
        return normalise(d), normalise(q)

    flags = {"raw": hit_flags(dv, qa, gold)}
    for n in (0, 1, 2):
        d, q = strip(n)
        flags[f"strip{n}"] = hit_flags(d, q, gold)

    for name, f in flags.items():
        out[f"recall_{name}"] = round(float(f.mean()), 3)

    centred_pairs = random_pair_cosines(strip(0)[0], np.random.default_rng(SEED))
    out["centred_pair_cosine_mean"] = round(float(centred_pairs.mean()), 3)

    # Paired tests. The aggregate gain from centring alone is not the point;
    # whether it survives one of these is.
    for a, b, key in (("raw", "strip0", "centring"),
                      ("strip0", "strip1", "first_component"),
                      ("strip1", "strip2", "second_component")):
        t = mcnemar_exact(flags[a], flags[b])
        out[f"p_{key}"] = t["p_value"]
        out[f"won_{key}"] = t["b_only"]
        out[f"lost_{key}"] = t["a_only"]

    out["recall_gain_pts"] = round(
        100 * (out["recall_strip1"] - out["recall_raw"]), 1)
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

    print(f"{r['n_documents']} documents, {r['dim']} dimensions, "
          f"{r['n_queries_answerable']} answerable queries\n")
    print("the cone")
    print(f"  random document pair cosine   {r['random_pair_cosine_mean']:.3f} "
          f"(sd {r['random_pair_cosine_sd']:.3f}, min {r['random_pair_cosine_min']:.3f})")
    print(f"  query to its gold document    {r['query_gold_cosine_mean']:.3f}")
    print(f"  query to a random document    {r['query_random_cosine_mean']:.3f}")
    print(f"  margin that has to carry it   {r['query_gold_margin']:.3f}")
    print(f"  after centring                {r['centred_pair_cosine_mean']:+.3f}\n")
    print("effective dimensionality")
    print(f"  largest single component      {r['top_component_share_pct']:.1f}% of variance")
    for frac in (50, 90, 95):
        print(f"  directions for {frac}% of variance {r[f'dims_for_{frac}pct_variance']:>4} "
              f"of {r['dim']}")
    print("\nretrieval, recall@10")
    print(f"  raw                           {r['recall_raw']:.3f}")
    print(f"  centred                       {r['recall_strip0']:.3f}  "
          f"won {r['won_centring']}, lost {r['lost_centring']}, p={r['p_centring']:.4f}")
    print(f"  centred, minus 1 component    {r['recall_strip1']:.3f}  "
          f"won {r['won_first_component']}, lost {r['lost_first_component']}, "
          f"p={r['p_first_component']:.4f}")
    print(f"  centred, minus 2 components   {r['recall_strip2']:.3f}  "
          f"won {r['won_second_component']}, lost {r['lost_second_component']}, "
          f"p={r['p_second_component']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
