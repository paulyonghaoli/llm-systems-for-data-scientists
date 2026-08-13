"""Build the cited-answer set that lesson 3.7 checks.

    python tools/build_answers.py

Groundedness is usually taught with a generator in the loop, which this
curriculum cannot run: there is no API key, and gate 12 forbids grading
learner text against a mock. So the artefact here is the **checker**, not the
generator, and the thing it checks is an authored set of answers with defects
planted at known positions.

That is not a compromise so much as a relocation of the interesting part. In a
production system the generator is someone else's model and you cannot change
it; what you own is the verification layer that decides whether to ship what it
produced. This file writes the inputs that layer has to cope with.

Six kinds of answer, each anchored in something the corpus already contains:

    grounded              every sentence cited, citations retrieved, numbers agree
    fabricated_citation   cites a document id that does not exist
    unretrieved_citation  cites a real document that was not in the context
    uncited_claim         states a number with no citation at all
    wrong_number          cites the right policy, states the FAQ's conflicting number
    stale_source          cites the superseded revision, whose numbers differ

The last two are the dangerous ones, because the citation is real and resolves.
An answer citing `policy-damage_claims-v1` looks perfectly supported until you
notice v2 exists and says 24 hours where v1 said 72.

Deterministic from seed 20260813. Writes data/corpus/answers.jsonl.
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = CORPUS / "answers.jsonl"

SEED = 20260813
CONTEXT_SIZE = 5


def load_corpus() -> tuple[list[dict], list[dict]]:
    docs = [json.loads(x) for x in
            (CORPUS / "documents.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = [json.loads(x) for x in
               (CORPUS / "queries.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return docs, queries


def build() -> list[dict]:
    docs, queries = load_corpus()
    by_id = {d["doc_id"]: d for d in docs}
    rng = random.Random(SEED)

    policies = [d for d in docs if d["type"] == "policy"]
    current = {d["facts"]["topic"]: d for d in policies if not d["superseded"]}
    superseded = {d["facts"]["topic"]: d for d in policies if d["superseded"]}
    conflict_faq = {d["facts"]["topic"]: d for d in docs
                    if d["type"] == "faq" and not d["facts"]["authoritative"]}
    others = [d["doc_id"] for d in docs if d["type"] in ("incident", "shipment")]

    # Topics where the revision actually moved a number: only these can carry a
    # stale_source defect that a number check could ever catch.
    stale_topics = [t for t, cur in current.items()
                    if t in superseded
                    and (superseded[t]["facts"]["hold_hours"] != cur["facts"]["hold_hours"]
                         or superseded[t]["facts"]["claim_days"] != cur["facts"]["claim_days"])]
    conflict_topics = sorted(conflict_faq)

    number = re.compile(r"\b\d+\b")

    def numbers_in(doc_id: str) -> set[str]:
        return set(number.findall(by_id[doc_id]["text"]))

    answers: list[dict] = []

    def add(query: dict, context: list[str], sentences: list[dict], defect: str,
            note: str) -> None:
        answers.append({
            "answer_id": f"a{len(answers):03d}",
            "query_id": query["query_id"],
            "context": context,
            "sentences": sentences,
            "defect": defect,
            "note": note,
        })

    def context_for(gold: str) -> list[str]:
        """The gold document plus filler, as a retriever would have returned."""
        filler = rng.sample(others, CONTEXT_SIZE - 1)
        ctx = [gold, *filler]
        rng.shuffle(ctx)
        return ctx

    topic_queries: dict[str, dict] = {}
    for q in queries:
        for g in q["gold_doc_ids"]:
            topic = by_id[g]["facts"].get("topic") if g in by_id else None
            if topic and topic not in topic_queries and by_id[g]["type"] == "policy":
                topic_queries[topic] = q

    for topic, q in sorted(topic_queries.items()):
        cur = current.get(topic)
        if cur is None:
            continue
        gold = cur["doc_id"]
        # Only state figures the document actually contains. Not every topic's
        # rules mention an hour count -- address verification talks in days --
        # so a fact recorded in `facts` is not necessarily a fact stated in the
        # text. The checker found this by flagging a "grounded" answer, which
        # is the builder's bug rather than the checker's.
        present = numbers_in(gold)
        hours = cur["facts"]["hold_hours"]
        days = cur["facts"]["claim_days"]
        if str(hours) not in present or str(days) not in present:
            continue
        ctx = context_for(gold)

        # 1. grounded
        add(q, ctx, [
            {"text": f"A hold lasts up to {hours} hours.", "cites": [gold]},
            {"text": f"Representations must be made within {days} days.", "cites": [gold]},
        ], "grounded", "every sentence cited, citations in context, numbers match")

        # 2. fabricated citation
        add(q, ctx, [
            {"text": f"A hold lasts up to {hours} hours.", "cites": [gold]},
            {"text": f"Representations must be made within {days} days.",
             "cites": [f"policy-{topic}-v9"]},
        ], "fabricated_citation", "the second citation names a document that does not exist")

        # 3. real citation, not in the context the model was given
        outside = next(d["doc_id"] for d in policies
                       if d["doc_id"] != gold and d["doc_id"] not in ctx
                       and not d["superseded"])
        add(q, ctx, [
            {"text": f"A hold lasts up to {hours} hours.", "cites": [gold]},
            {"text": "Exceptions are granted in writing by a regional manager.",
             "cites": [outside]},
        ], "unretrieved_citation",
            "the cited document exists but was never retrieved, so the model "
            "could not have read it")

        # 4. an uncited numeric claim
        add(q, ctx, [
            {"text": f"A hold lasts up to {hours} hours.", "cites": [gold]},
            {"text": f"Representations must be made within {days} days.", "cites": []},
        ], "uncited_claim", "a factual sentence with no citation at all")

    # 5. wrong_number: cites the governing policy, states the FAQ's number
    for topic in conflict_topics:
        cur, faq = current.get(topic), conflict_faq[topic]
        if cur is None or topic not in topic_queries:
            continue
        stated = faq["facts"]["hold_hours"]
        if stated == cur["facts"]["hold_hours"]:
            continue
        # The defect is that the number is absent from the cited policy; if it
        # happens to appear there anyway the answer is not actually wrong.
        if str(stated) in numbers_in(cur["doc_id"]):
            continue
        q = topic_queries[topic]
        gold = cur["doc_id"]
        add(q, context_for(gold), [
            {"text": f"A hold lasts up to {stated} hours.", "cites": [gold]},
        ], "wrong_number",
            f"cites the governing policy, which says "
            f"{cur['facts']['hold_hours']} hours, and states the "
            f"non-authoritative FAQ's {stated}")

    # 6. stale_source: cites the superseded revision
    for topic in stale_topics:
        if topic not in topic_queries:
            continue
        old = superseded[topic]
        q = topic_queries[topic]
        if str(old["facts"]["hold_hours"]) not in numbers_in(old["doc_id"]):
            continue
        add(q, context_for(old["doc_id"]), [
            {"text": f"A hold lasts up to {old['facts']['hold_hours']} hours.",
             "cites": [old["doc_id"]]},
        ], "stale_source",
            f"cites revision 1, which says {old['facts']['hold_hours']} hours, "
            f"where the current revision says {current[topic]['facts']['hold_hours']}")

    return answers


def main() -> int:
    if not (CORPUS / "documents.jsonl").exists():
        print("corpus not built; run tools/build_corpus.py", file=sys.stderr)
        return 1
    answers = build()

    # A "grounded" answer that any check rejects is a defect in this file, not
    # a finding. Verified here rather than asserted, for the same reason the
    # corpus phenomena are.
    sys.path.insert(0, str(ROOT))
    from experiments.groundedness import check
    from experiments.groundedness import load as load_checked
    docs_by_id, _ = load_checked()
    for a in answers:
        if a["defect"] != "grounded":
            continue
        failed = [k for k, ok in check(a, docs_by_id).items() if not ok]
        if failed:
            print(f"{a['answer_id']} is labelled grounded but fails {failed}",
                  file=sys.stderr)
            return 1
    OUT.write_text("".join(json.dumps(a) + "\n" for a in answers), encoding="utf-8")

    counts: dict[str, int] = {}
    for a in answers:
        counts[a["defect"]] = counts.get(a["defect"], 0) + 1
    print(f"{len(answers)} answers")
    for k in sorted(counts):
        print(f"    {k:<22} {counts[k]:>3}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
