"""Groundedness checks: what each one catches, and what none of them can.

    python experiments/groundedness.py
    python experiments/groundedness.py --json

The usual framing puts a model in the loop — generate an answer, ask a judge
whether it is supported. This runs the other half, the half you own: given an
answer with citations and the context it was produced from, decide
mechanically whether the citations hold up.

Five checks, over 55 authored answers with defects planted at known positions
(`tools/build_answers.py`). Four of the five need no model, no embeddings and
no network; they are string and set operations over metadata you already have.

The results worth knowing before writing any of this:

  * The cheap checks catch the loud failures completely. A citation naming a
    document that does not exist, or one the retriever never returned, is
    caught 100% of the time by a set membership test.
  * A number check catches the answer that cites the right document and states
    the wrong figure — the failure that looks correct to a reader.
  * **No content check catches a stale citation.** An answer quoting revision 1
    accurately, and citing revision 1, is internally consistent: the number it
    states really is in the document it names. Only provenance metadata
    distinguishes it from a correct answer, which is why the corpus records
    `superseded` per document and why a system without that field cannot detect
    this class at all.
  * And the false-positive rate on grounded answers is the number that decides
    whether any of this is usable, because a checker that flags everything
    detects everything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

NUMBER = re.compile(r"\b\d+\b")

CHECKS = ("citation_resolves", "citation_in_context", "every_claim_cited",
          "numbers_supported", "no_superseded_source")


def load() -> tuple[dict, list[dict]]:
    docs = {}
    for line in (CORPUS / "documents.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            docs[d["doc_id"]] = d
    answers = [json.loads(x) for x in
               (CORPUS / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return docs, answers


def check(answer: dict, docs: dict) -> dict[str, bool]:
    """Run every check. True means the check passed, i.e. found no problem."""
    cited = [c for s in answer["sentences"] for c in s["cites"]]
    context = set(answer["context"])

    # 1. Every citation names a document that exists. Catches a fabricated id,
    #    which is what a model does when it is pattern-matching the shape of a
    #    reference rather than reporting one.
    resolves = all(c in docs for c in cited)

    # 2. Every citation was in the context. A real document the model was never
    #    shown cannot be the source of anything it wrote, so this is a stronger
    #    and cheaper test than any content comparison.
    in_context = all(c in context for c in cited)

    # 3. Every sentence stating a number carries at least one citation. Numeric
    #    claims are used as the proxy for "factual" here because they are
    #    unambiguous; the general version needs a claim detector and is where
    #    this check stops being free.
    cited_claims = all(s["cites"] for s in answer["sentences"] if NUMBER.search(s["text"]))

    # 4. Every number in a sentence appears in a document that sentence cites.
    #    The one content check, and the only one that catches an answer whose
    #    citation is real and whose figure is not.
    supported = True
    for s in answer["sentences"]:
        nums = set(NUMBER.findall(s["text"]))
        if not nums or not s["cites"]:
            continue
        available: set[str] = set()
        for c in s["cites"]:
            if c in docs:
                available |= set(NUMBER.findall(docs[c]["text"]))
        if nums - available:
            supported = False

    # 5. No citation points at a superseded document. Pure provenance: nothing
    #    in the text of a revision says it has been replaced.
    fresh = all(not docs[c]["superseded"] for c in cited if c in docs)

    return {
        "citation_resolves": resolves,
        "citation_in_context": in_context,
        "every_claim_cited": cited_claims,
        "numbers_supported": supported,
        "no_superseded_source": fresh,
    }


def compute() -> dict[str, float]:
    docs, answers = load()
    by_defect: dict[str, list[dict]] = defaultdict(list)
    for a in answers:
        by_defect[a["defect"]].append(a)

    results = {a["answer_id"]: check(a, docs) for a in answers}

    out: dict[str, float] = {
        "n_answers": len(answers),
        "n_checks": len(CHECKS),
        "n_grounded": len(by_defect["grounded"]),
    }

    # Detection: an answer is flagged if any check fails.
    for defect, group in sorted(by_defect.items()):
        flagged = sum(1 for a in group if not all(results[a["answer_id"]].values()))
        out[f"flagged_{defect}"] = round(flagged / len(group), 3)
        out[f"n_{defect}"] = len(group)
        # Which check does the work for this defect?
        for name in CHECKS:
            fails = sum(1 for a in group if not results[a["answer_id"]][name])
            out[f"{defect}__{name}"] = round(fails / len(group), 3)

    defective = [a for a in answers if a["defect"] != "grounded"]
    caught = sum(1 for a in defective if not all(results[a["answer_id"]].values()))
    out["detection_rate"] = round(caught / len(defective), 3)
    out["false_positive_rate"] = out["flagged_grounded"]

    # What survives if the one content check is removed — the configuration a
    # team without document text, or without the budget to compare it, is left
    # with.
    metadata_only = [c for c in CHECKS if c != "numbers_supported"]
    caught_meta = sum(1 for a in defective
                      if not all(results[a["answer_id"]][c] for c in metadata_only))
    out["detection_metadata_only"] = round(caught_meta / len(defective), 3)

    # And what a system with no provenance field can do.
    no_prov = [c for c in CHECKS if c != "no_superseded_source"]
    caught_np = sum(1 for a in defective
                    if not all(results[a["answer_id"]][c] for c in no_prov))
    out["detection_without_provenance"] = round(caught_np / len(defective), 3)
    out["stale_missed_without_provenance"] = out["n_stale_source"]

    # Percentage twins for every rate. Prose quotes percentages and gate 18
    # compares the number in front of the marker literally, so a fraction-valued
    # key cannot back a "%" sentence. Third time this has bitten; adding them
    # for all four rather than one at a time.
    for key in ("detection_rate", "false_positive_rate",
                "detection_metadata_only", "detection_without_provenance"):
        out[f"{key}_pct"] = round(100 * out[key], 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not (CORPUS / "answers.jsonl").exists():
        print("answer set missing; run tools/build_answers.py", file=sys.stderr)
        return 1

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    print(f"{r['n_answers']} answers, {r['n_checks']} checks\n")
    defects = sorted(k[8:] for k in r if k.startswith("flagged_"))
    width = max(len(d) for d in defects)
    print(f"{'defect':<{width}} {'n':>3} {'flagged':>8}   " +
          " ".join(f"{c[:12]:>13}" for c in CHECKS))
    for d in defects:
        row = " ".join(f"{r[f'{d}__{c}']:>13.2f}" for c in CHECKS)
        print(f"{d:<{width}} {r[f'n_{d}']:>3} {r[f'flagged_{d}']:>8.2f}   {row}")
    print(f"\noverall detection      {r['detection_rate']:.3f}")
    print(f"false positives        {r['false_positive_rate']:.3f} "
          f"on {r['n_grounded']} grounded answers")
    print(f"metadata checks only   {r['detection_metadata_only']:.3f}")
    print(f"without provenance     {r['detection_without_provenance']:.3f} "
          f"({r['stale_missed_without_provenance']} stale citations become invisible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
