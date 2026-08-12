"""Adversarial test of the corpus verifier.

`verify_corpus.py` passed all nine checks the first time it was run. That is
exactly when a check is least trustworthy: a check that has only ever passed
has not been tested, and one that can never fail is decoration. Silent-success
bugs are easy to write here -- an empty candidate list, a `continue` that skips
every item, a percentage over a denominator that happens to be zero -- and all
of them look like a green tick.

So each phenomenon gets a mutation that destroys it, and the corresponding
check must report FAIL. Run directly; exits non-zero if any mutation slips
past. This is gate 26.
"""

from __future__ import annotations

import contextlib
import copy
import io
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools import verify_corpus as vc  # noqa: E402


def run_with(docs: list[dict], queries: list[dict]) -> tuple[int, str]:
    """Run the verifier against supplied data, returning (code, stdout)."""
    original = vc.load
    vc.load = lambda: (docs, queries)  # type: ignore[assignment]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = vc.main()
    finally:
        vc.load = original  # type: ignore[assignment]
    return code, out.getvalue()


def failed_rows(stdout: str) -> set[str]:
    """Phenomena with at least one FAIL row."""
    return {line.split()[0] for line in stdout.splitlines() if line.rstrip().endswith("FAIL")}


# --- mutations ------------------------------------------------------------
# Each takes (docs, queries) already deep-copied, and destroys one phenomenon.

def break_vocabulary_mismatch(docs, queries):
    """Quote the gold document verbatim, so BM25 finds it trivially."""
    by_id = {d["doc_id"]: d for d in docs}
    for q in queries:
        if q["phenomenon"] == "vocabulary_mismatch" and q["gold_doc_ids"]:
            q["text"] = " ".join(by_id[q["gold_doc_ids"][0]]["text"].split()[:40])
    return docs, queries


def break_lexical_distractor(docs, queries):
    """Same trick: make the gold document the obvious lexical match."""
    by_id = {d["doc_id"]: d for d in docs}
    for q in queries:
        if q["phenomenon"] == "lexical_distractor" and q["gold_doc_ids"]:
            q["text"] = " ".join(by_id[q["gold_doc_ids"][0]]["text"].split()[:40])
    return docs, queries


def break_near_duplicate(docs, queries):
    """Give every handbook disjoint vocabulary, so no siblings remain."""
    for i, d in enumerate(docs):
        if d["type"] == "handbook":
            d["text"] = " ".join(f"zz{i}tok{j}" for j in range(300))
    return docs, queries


def break_superseded(docs, queries):
    """Delete the superseded versions, leaving the current ones untouched.

    Deleting every `-v1` document was too crude: topics that were never revised
    are current *at* v1, so that mutation orphaned live gold references and
    tripped the integrity check instead of the superseded check. Only remove a
    v1 that actually has a v2 successor.
    """
    ids = {d["doc_id"] for d in docs}
    return [d for d in docs
            if not (d["doc_id"].endswith("-v1")
                    and d["doc_id"][:-3] + "-v2" in ids)], queries


def break_integrity(docs, queries):
    """Point a gold reference at a document that does not exist."""
    for q in queries:
        if q["gold_doc_ids"]:
            q["gold_doc_ids"] = ["policy-does-not-exist-v9"]
            break
    return docs, queries


def break_multi_hop(docs, queries):
    """Keep only the first gold document, so one document suffices."""
    for q in queries:
        if q["phenomenon"] == "multi_hop":
            q["gold_doc_ids"] = q["gold_doc_ids"][:1]
    return docs, queries


def break_contradiction(docs, queries):
    """Strip digits from the FAQs, so no rival states a different number."""
    for d in docs:
        if d["type"] == "faq":
            d["text"] = re.sub(r"\d", "", d["text"])
    return docs, queries


def break_unanswerable_scores(docs, queries):
    """Make the unanswerable queries quote a real document."""
    for q in queries:
        if q["phenomenon"] == "unanswerable":
            q["text"] = " ".join(docs[0]["text"].split()[:40])
    return docs, queries


def break_unanswerable_gold(docs, queries):
    """Give an unanswerable query a gold document."""
    for q in queries:
        if q["phenomenon"] == "unanswerable":
            q["gold_doc_ids"] = [docs[0]["doc_id"]]
            break
    return docs, queries


# `None` means the verifier is expected to bail out before printing the table
# at all, so there is no per-phenomenon row to look for -- a non-zero exit is
# the whole signal.
MUTATIONS = [
    ("vocabulary_mismatch", break_vocabulary_mismatch),
    ("lexical_distractor", break_lexical_distractor),
    ("near_duplicate", break_near_duplicate),
    ("superseded", break_superseded),
    ("multi_hop", break_multi_hop),
    ("contradiction", break_contradiction),
    ("unanswerable", break_unanswerable_scores),
    ("unanswerable", break_unanswerable_gold),
    (None, break_integrity),
]


def main() -> int:
    base_docs, base_queries = vc.load()

    code, out = run_with(copy.deepcopy(base_docs), copy.deepcopy(base_queries))
    if code != 0:
        print("baseline corpus does not pass; fix that before testing mutations",
              file=sys.stderr)
        print(out, file=sys.stderr)
        return 1
    print(f"baseline passes ({len(base_docs)} docs, {len(base_queries)} queries)\n")

    bad = 0
    for name, mutate in MUTATIONS:
        docs, queries = mutate(copy.deepcopy(base_docs), copy.deepcopy(base_queries))
        code, out = run_with(docs, queries)
        caught = True if name is None else name in failed_rows(out)
        label = f"{mutate.__name__}"
        if code != 0 and caught:
            print(f"  caught   {label}")
        else:
            reason = "verifier returned 0" if code == 0 else f"{name} row still ok"
            print(f"  MISSED   {label}  ({reason})")
            bad += 1

    if bad:
        print(f"\n{bad} mutation(s) went undetected", file=sys.stderr)
        return 1
    print(f"\nall {len(MUTATIONS)} mutations detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
