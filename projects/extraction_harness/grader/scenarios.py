"""Seeded batches of model output, malformed the way models malform things.

Every batch is generated from the seed, so there is no fixed expected output
to memorise, and the malformation mix is the one stated in lesson 2.4 §G.
"""

from __future__ import annotations

import json
import random
import re

STATUSES = ("cancelled", "fulfilled", "pending")

#: Failure modes and their weights. The residual is well-formed output.
MALFORMATIONS = {
    "clean": 0.46,
    "fenced": 0.12,
    "preamble": 0.08,
    "trailing_comma": 0.05,
    "single_quotes": 0.04,
    "unquoted_keys": 0.03,
    "nan_literal": 0.03,
    "schema_violation": 0.07,
    "truncated": 0.09,
    "hopeless": 0.03,
}


def _record(rng: random.Random) -> dict:
    return {
        "record_id": f"{rng.randrange(0x100000, 0xFFFFFF):06x}",
        "status": rng.choice(STATUSES),
        "quantity": rng.randrange(1, 40),
        "note": rng.choice(["", "address unverified", "partial shipment"]),
    }


def _malform(obj: dict, kind: str, rng: random.Random) -> tuple[str, str]:
    """Return (text, finish_reason)."""
    text = json.dumps(obj)
    if kind == "clean":
        return text, "stop"
    if kind == "fenced":
        return f"```json\n{text}\n```", "stop"
    if kind == "preamble":
        return f"Here is the JSON you requested:\n\n{text}", "stop"
    if kind == "trailing_comma":
        return text[:-1] + ",}", "stop"
    if kind == "single_quotes":
        return text.replace('"', "'"), "stop"
    if kind == "unquoted_keys":
        return re.sub(r'"(\w+)":', r"\1:", text), "stop"
    if kind == "nan_literal":
        return text.replace(str(obj["quantity"]), "NaN", 1), "stop"
    if kind == "schema_violation":
        broken = dict(obj)
        if rng.random() < 0.5:
            broken["quantity"] = str(broken["quantity"])
        else:
            del broken["status"]
        return json.dumps(broken), "stop"
    if kind == "truncated":
        # Cut at max_tokens. Half the time the model had "closed" the object
        # as it ran out, so the result parses and may satisfy the schema —
        # which is exactly the case only the finish reason reveals.
        cut = rng.randrange(int(len(text) * 0.45), int(len(text) * 0.92))
        head = text[:cut]
        if rng.random() < 0.5:
            head = head.rsplit(",", 1)[0] + "}"
        return head, "length"
    if kind == "hopeless":
        return rng.choice([
            "I am afraid I cannot help with that request.",
            "Could you clarify which record you mean?",
        ]), "stop"
    raise ValueError(kind)


def batch(seed: int, n: int = 120) -> list[dict]:
    """One seeded batch of raw model outputs."""
    rng = random.Random(seed * 7919)
    kinds = list(MALFORMATIONS)
    weights = [MALFORMATIONS[k] for k in kinds]
    out = []
    for _ in range(n):
        kind = rng.choices(kinds, weights)[0]
        text, finish = _malform(_record(rng), kind, rng)
        out.append({"text": text, "finish_reason": finish, "_kind": kind})
    return out


def degenerate_batches() -> list[dict]:
    """The edge cases a harness written against the happy path gets wrong."""
    return [
        {"name": "empty batch", "outputs": []},
        {
            "name": "every output truncated",
            "outputs": [{"text": '{"record_id": "a1f3c9", "status": "pending"',
                         "finish_reason": "length", "_kind": "truncated"}] * 5,
        },
        {
            # Parses, satisfies the schema, and was cut off. Only the finish
            # reason distinguishes it from a good record.
            "name": "truncated but valid-looking",
            "outputs": [{"text": '{"record_id": "a1f3c9", "status": "pending", "quantity": 3}',
                         "finish_reason": "length", "_kind": "truncated"}],
        },
        {
            "name": "content filter refusal",
            "outputs": [{"text": "", "finish_reason": "content_filter", "_kind": "hopeless"}],
        },
    ]
