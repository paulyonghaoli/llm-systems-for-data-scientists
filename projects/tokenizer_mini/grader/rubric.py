"""The published rubric. Nothing here is hidden from the learner.

Five criteria, 100 points. Each is scored as the fraction of scenarios that
satisfy it, so a packer that is right on English prose and wrong on Japanese
loses points rather than failing outright.
"""

from __future__ import annotations

from collections.abc import Callable

from grader import reference
from grader.scenarios import degenerate_scenarios, scenario, tokenizer

CRITERIA = [
    ("A", "budget respected", 30,
     "total_tokens + reserved_output never exceeds context_limit"),
    ("B", "accounting consistent", 25,
     "reported total_tokens equals a fresh count of the prompt actually returned"),
    ("C", "selection matches reference", 20,
     "the same documents are included, in the same order"),
    ("D", "truncation on a token boundary", 15,
     "truncated_text is a true character-prefix of the document and re-encodes "
     "to a prefix of its token sequence"),
    ("E", "degenerate cases", 10,
     "no documents / budget below the system prompt / one oversized document"),
]

Packer = Callable[..., dict]


def _check_one(pack: Packer, sc: dict) -> dict[str, bool]:
    tok = tokenizer()
    out = pack(
        tok,
        sc["system_prompt"],
        sc["documents"],
        sc["context_limit"],
        sc["reserved_output"],
    )
    ref = reference.pack(
        tok,
        sc["system_prompt"],
        sc["documents"],
        sc["context_limit"],
        sc["reserved_output"],
    )

    prompt = out.get("prompt", "")
    real_tokens = tok.count(prompt)

    results = {
        "A": real_tokens + sc["reserved_output"] <= sc["context_limit"],
        "B": out.get("total_tokens") == real_tokens,
        "C": list(out.get("included", [])) == ref["included"],
        "D": True,
    }

    trunc = out.get("truncated_text", "") or ""
    if trunc:
        idx = len(out.get("included", []))
        source = sc["documents"][idx] if idx < len(sc["documents"]) else ""
        on_boundary = (
            source.startswith(trunc)
            and tok.encode(trunc) == tok.encode(source)[: len(tok.encode(trunc))]
        )
        results["D"] = on_boundary
    return results


def grade(pack: Packer, seed: int, n_scenarios: int = 12) -> dict:
    per_criterion: dict[str, list[bool]] = {k: [] for k, *_ in CRITERIA}
    failures: list[str] = []

    for i in range(n_scenarios):
        sc = scenario(seed + i)
        try:
            res = _check_one(pack, sc)
        except Exception as e:  # noqa: BLE001
            for key in ("A", "B", "C", "D"):
                per_criterion[key].append(False)
            failures.append(f"seed {sc['seed']}: raised {type(e).__name__}: {e}")
            continue
        for key, ok in res.items():
            per_criterion[key].append(ok)
            if not ok:
                label = dict((k, n) for k, n, *_ in CRITERIA)[key]
                failures.append(f"seed {sc['seed']}: {key} ({label}) failed")

    for sc in degenerate_scenarios():
        wants_refusal = sc.get("expect") == "raises"
        try:
            res = _check_one(pack, sc)
        except ValueError:
            ok = wants_refusal
            if not ok:
                failures.append(
                    f"degenerate '{sc['name']}': refused a request it should have packed"
                )
        except Exception as e:  # noqa: BLE001
            ok = False
            failures.append(f"degenerate '{sc['name']}': raised {type(e).__name__}: {e}")
        else:
            if wants_refusal:
                ok = False
                failures.append(
                    f"degenerate '{sc['name']}': returned a packed prompt instead of raising "
                    f"ValueError - the system prompt must not be silently trimmed"
                )
            else:
                ok = all(res.values())
                if not ok:
                    bad = ", ".join(k for k, v in res.items() if not v)
                    failures.append(f"degenerate '{sc['name']}': failed {bad}")
        per_criterion["E"].append(ok)

    breakdown = []
    total = 0.0
    for key, name, points, _desc in CRITERIA:
        results = per_criterion[key]
        frac = sum(results) / len(results) if results else 0.0
        earned = round(points * frac, 1)
        total += earned
        breakdown.append({
            "key": key, "name": name, "points": points,
            "earned": earned, "passed": sum(results), "of": len(results),
        })

    return {"total": round(total, 1), "breakdown": breakdown, "failures": failures}
