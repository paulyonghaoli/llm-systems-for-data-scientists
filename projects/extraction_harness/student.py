"""Mini-project 2 - the extraction harness.

Implement `process` below, then grade it:

    cd projects/extraction_harness
    python -m grader --seed 1

The full specification and the rubric are in README.md. Batches are generated
fresh from the seed on every run, so there is no fixed expected output.

The starter below is a harness that does the obvious thing: parse, and if it
parses, accept it. It is not stupid — it is what most first implementations
look like — and the rubric shows precisely which three claims it cannot
support.
"""

from __future__ import annotations

import json


def process(outputs: list[dict], expensive_repair) -> list[dict]:
    """Turn raw model outputs into validated records.

    outputs           list of {"text": str, "finish_reason": str}
    expensive_repair  callable(text) -> str | None. Stands in for a
                      model-based repair call. Every invocation is counted
                      and charged against criterion D, so reach for it last.

    Returns one result per input, each a dict with:
      status   "ok" | "repaired" | "truncated" | "rejected"
      record   the validated dict, or None
      reason   why it was not recovered (required unless status is ok/repaired)
    """
    results = []
    for item in outputs:
        try:
            record = json.loads(item["text"])
        except ValueError:
            results.append({"status": "rejected", "record": None,
                            "reason": "did not parse"})
            continue
        results.append({"status": "ok", "record": record, "reason": None})
    return results
