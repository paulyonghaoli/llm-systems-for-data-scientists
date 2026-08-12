"""Reference implementation of the extraction harness.

Four things it gets right, in the order they matter:

1. The finish reason is checked **first**. A truncated output is reported as
   truncated even when it parses and satisfies the schema, because that case
   is real and nothing downstream can detect it.
2. The repair ladder runs cheapest-first, so the expensive rung is reached
   only by outputs the free rungs could not fix.
3. Validation happens after every repair, and a record that fails it is
   rejected rather than returned.
4. Nothing is silently dropped: every input produces exactly one result, and
   anything not recovered carries a reason.
"""

from __future__ import annotations

import json
import re

ALLOWED_STATUS = {"cancelled", "fulfilled", "pending"}
REQUIRED = {"record_id": str, "status": str, "quantity": int}


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def parse(text: str) -> object | None:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        return None


def extract(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def mechanical(text: str) -> str:
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bNaN\b|\bInfinity\b", "null", text)
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', text)


def schema_ok(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    for key, typ in REQUIRED.items():
        if key not in obj:
            return False
        value = obj[key]
        if typ is int and isinstance(value, bool):
            return False
        if not isinstance(value, typ):
            return False
    return obj["status"] in ALLOWED_STATUS


def process(outputs: list[dict], expensive_repair) -> list[dict]:
    results = []
    for item in outputs:
        text, finish = item["text"], item["finish_reason"]

        # 1. Completeness is orthogonal to validity, and only this field
        #    reports it. Checking it first is what stops a truncated record
        #    that happens to parse from being accepted.
        if finish != "stop":
            results.append({"status": "truncated", "record": None,
                            "reason": f"finish_reason={finish}"})
            continue

        # 2. The free rungs.
        obj, repairs = parse(text), 0
        if obj is None:
            obj, repairs = parse(extract(text)), 1
        if obj is None:
            obj, repairs = parse(mechanical(extract(text))), 2

        # 3. The expensive rung, only for what the free ones could not fix.
        if obj is None:
            obj = parse(expensive_repair(text) or "")
            repairs = 3

        if obj is None:
            results.append({"status": "rejected", "record": None,
                            "reason": "unparseable after every repair"})
            continue

        # 4. Validation after repair, always.
        if not schema_ok(obj):
            results.append({"status": "rejected", "record": None,
                            "reason": "parsed but failed the schema"})
            continue

        results.append({
            "status": "ok" if repairs == 0 else "repaired",
            "record": obj,
            "reason": None,
            "repairs": repairs,
        })
    return results
