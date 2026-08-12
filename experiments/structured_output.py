"""Which repairs actually recover malformed output, and what each one costs.

    python experiments/structured_output.py
    python experiments/structured_output.py --json

A model asked for JSON returns something JSON-shaped. The gap between those is
where this lesson lives, and it has three distinct layers that get collapsed
into one:

    does it parse?  →  does it match the schema?  →  is it right?

Only the first two are checkable here. The third needs an evaluation and is
named as out of scope rather than quietly folded in.

## The malformation model, stated

There is no model in this experiment, so the malformed outputs are generated
from valid objects by applying failure modes drawn from a fixed distribution.
Those failure modes are real — fenced code blocks, conversational preambles,
trailing commas, single quotes, unquoted keys, truncation at `max_tokens`,
`NaN` literals, and output that parses perfectly while violating the schema —
but their *mix* is a choice, stated here so the recovery rates below are read
as "what these repairs do to this mix" rather than as a universal constant.
Run it against your own logged failures to get numbers about your system.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SEED = 20260811
N = 600

#: How often each failure mode occurs. The residual is well-formed output.
MALFORMATIONS = {
    "clean": 0.55,
    "fenced": 0.12,
    "preamble": 0.08,
    "trailing_comma": 0.05,
    "single_quotes": 0.04,
    "unquoted_keys": 0.03,
    "truncated": 0.07,
    "nan_literal": 0.02,
    "schema_violation": 0.04,
}

#: The contract the output is supposed to satisfy.
REQUIRED = {"record_id": str, "status": str, "quantity": int}
ALLOWED_STATUS = {"fulfilled", "pending", "cancelled"}

ROOT = Path(__file__).resolve().parent.parent


def _valid_object(rng: random.Random) -> dict:
    return {
        "record_id": f"{rng.randrange(0x100000, 0xFFFFFF):06x}",
        "status": rng.choice(sorted(ALLOWED_STATUS)),
        "quantity": rng.randrange(1, 40),
        "note": rng.choice(["", "address unverified", "partial shipment"]),
    }


def _malform(obj: dict, kind: str, rng: random.Random) -> str:
    text = json.dumps(obj)
    if kind == "clean":
        return text
    if kind == "fenced":
        return f"```json\n{text}\n```"
    if kind == "preamble":
        return f"Here is the JSON you requested:\n\n{text}"
    if kind == "trailing_comma":
        return text[:-1] + ",}"
    if kind == "single_quotes":
        return text.replace('"', "'")
    if kind == "unquoted_keys":
        return re.sub(r'"(\w+)":', r"\1:", text)
    if kind == "truncated":
        # Cut at max_tokens. Sometimes this lands mid-object and fails to
        # parse; sometimes it lands after a complete field and the remaining
        # keys were optional, in which case it parses and is quietly wrong.
        cut = rng.randrange(int(len(text) * 0.45), int(len(text) * 0.9))
        head = text[:cut]
        if rng.random() < 0.5:  # the model "closed" the object as it ran out
            head = head.rsplit(",", 1)[0] + "}"
        return head
    if kind == "nan_literal":
        return text.replace(str(obj["quantity"]), "NaN", 1)
    if kind == "schema_violation":
        broken = dict(obj)
        if rng.random() < 0.5:
            broken["quantity"] = str(broken["quantity"])  # right key, wrong type
        else:
            del broken["status"]  # required key missing
        return json.dumps(broken)
    raise ValueError(kind)


# --- the repair ladder, cheapest first ---------------------------------------


def _reject_constant(name: str) -> None:
    """Refuse NaN / Infinity / -Infinity.

    Python's json.loads accepts all three **by default**, even though none is
    valid JSON. Left alone it hands you a float that is not equal to itself
    and that json.dumps re-serialises into invalid JSON, so the failure is
    silent data corruption rather than a parse error. `parse_constant` is the
    documented opt-out and every parser in this curriculum uses it.
    """
    raise ValueError(f"{name} is not valid JSON")


def parse(text: str) -> object | None:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_permissive(text: str) -> object | None:
    """What you get without the opt-out, for the comparison in §G."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def extract(text: str) -> str:
    """Strip fences and conversational padding around a JSON object."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def mechanical(text: str) -> str:
    """Deterministic fixes for the malformations a model actually produces."""
    text = re.sub(r",\s*([}\]])", r"\1", text)  # trailing comma
    text = re.sub(r"\bNaN\b|\bInfinity\b", "null", text)  # non-JSON literals
    if '"' not in text and "'" in text:  # single-quoted throughout
        text = text.replace("'", '"')
    text = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', text)  # unquoted keys
    return text


def schema_ok(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    for key, typ in REQUIRED.items():
        if key not in obj:
            return False
        # bool is a subclass of int, and a bool where a count belongs is wrong.
        if typ is int and isinstance(obj[key], bool):
            return False
        if not isinstance(obj[key], typ):
            return False
    return obj["status"] in ALLOWED_STATUS


def compute() -> dict[str, float]:
    rng = random.Random(SEED)
    kinds = list(MALFORMATIONS)
    weights = [MALFORMATIONS[k] for k in kinds]

    stats = {
        "raw": 0, "after_extract": 0, "after_mechanical": 0,
        "schema_after_repair": 0,
    }
    truncated_total = truncated_parses = truncated_passes_schema = 0
    silently_accepted = 0
    parses_but_fails_schema = 0

    for _ in range(N):
        kind = rng.choices(kinds, weights)[0]
        text = _malform(_valid_object(rng), kind, rng)

        raw = parse(text)
        if raw is not None:
            stats["raw"] += 1

        step1 = raw if raw is not None else parse(extract(text))
        if step1 is not None:
            stats["after_extract"] += 1

        step2 = step1 if step1 is not None else parse(mechanical(extract(text)))
        if step2 is not None:
            stats["after_mechanical"] += 1
            if schema_ok(step2):
                stats["schema_after_repair"] += 1
            else:
                parses_but_fails_schema += 1

        if parse(text) is None and parse_permissive(text) is not None:
            silently_accepted += 1

        if kind == "truncated":
            truncated_total += 1
            if step2 is not None:
                truncated_parses += 1
                if schema_ok(step2):
                    truncated_passes_schema += 1

    out: dict[str, float] = {"n": N}
    for key, count in stats.items():
        out[f"{key}_pct"] = round(100 * count / N, 1)
    out["gain_extract_pts"] = round(out["after_extract_pct"] - out["raw_pct"], 1)
    out["gain_mechanical_pts"] = round(
        out["after_mechanical_pct"] - out["after_extract_pct"], 1)
    out["residual_pct"] = round(100 - out["after_mechanical_pct"], 1)
    out["parses_but_fails_schema_pct"] = round(100 * parses_but_fails_schema / N, 1)

    out["silently_accepted_pct"] = round(100 * silently_accepted / N, 1)
    out["truncated_n"] = truncated_total
    out["truncated_parses_pct"] = round(100 * truncated_parses / truncated_total, 1)
    out["truncated_passes_schema_pct"] = round(
        100 * truncated_passes_schema / truncated_total, 1)
    out["truncated_silent_pct"] = out["truncated_passes_schema_pct"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"{v['n']:.0f} outputs, malformation mix stated in the module docstring\n")
    print(f"{'stage':<34} {'parses':>8} {'gain':>7}")
    print("-" * 51)
    print(f"{'raw json.loads':<34} {v['raw_pct']:>7.1f}% {'':>7}")
    print(f"{'+ strip fences and preamble':<34} {v['after_extract_pct']:>7.1f}% "
          f"{v['gain_extract_pts']:>+7.1f}")
    print(f"{'+ mechanical fixes':<34} {v['after_mechanical_pct']:>7.1f}% "
          f"{v['gain_mechanical_pts']:>+7.1f}")
    print(f"\n{v['residual_pct']}% still does not parse — that residual is what a "
          f"model-based\nrepair call would have to earn its cost against.")

    print(f"\nParsing is not validation: {v['parses_but_fails_schema_pct']}% of all "
          f"outputs parse\ncleanly and fail the schema.")
    print(f"\n{v['silently_accepted_pct']}% carry a NaN or Infinity literal, which a "
          f"DEFAULT json.loads\naccepts without complaint — silent corruption rather "
          f"than a parse error.")

    print(f"\nOf the {v['truncated_n']:.0f} truncated outputs:")
    print(f"  {v['truncated_parses_pct']}% still parse as JSON")
    print(f"  {v['truncated_passes_schema_pct']}% parse AND satisfy the schema — "
          f"a silently\n     incomplete record that no check here can catch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
