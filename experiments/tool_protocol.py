"""What argument validation catches, and the class it cannot touch.

    python experiments/tool_protocol.py
    python experiments/tool_protocol.py --json

Between a model proposing a tool call and a program executing it there is a
boundary. This measures what each layer of validation stops, using proposed
calls with defects planted at known positions, against the deterministic
sandbox in `llmlab.tools`.

Four validators of increasing strictness:

    name_only        the tool exists
    + required       every required argument is present
    + types          every argument has its declared type
    + no_extras      no argument the schema never described  (the full check)

The structural result is the one people expect: each layer catches its own
class completely, and the full validator executes nothing malformed.

The result worth the lesson is the other one. **Adversarial values pass every
layer.** `{"path": "notes/../../etc/passwd"}` is a string where a string was
declared, and `{"expression": "__import__('os').system(...)"}` is a
syntactically perfect argument. A schema constrains *type*, and these attacks
are carried in *content*, so validation cannot see them at all — the tool's own
guard is the only thing between them and the operation they name. Every one of
them executes, and is stopped inside the tool or not at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llmlab.tools import Sandbox, validate_call  # noqa: E402

LAYERS = ("name_only", "required", "types", "no_extras")


def proposed_calls() -> list[dict]:
    """Calls a model might emit, each labelled with what is wrong with it.

    Authored rather than sampled: the point is coverage of the defect classes,
    and a random generator would produce ten of the easy ones for every
    interesting one.
    """
    return [
        # --- well formed -------------------------------------------------
        {"tool": "calculator", "args": {"expression": "3 * (4 + 5)"}, "defect": "none"},
        {"tool": "search", "args": {"query": "hold policy", "k": 2}, "defect": "none"},
        {"tool": "read_file", "args": {"path": "notes/depot.txt"}, "defect": "none"},
        {"tool": "shipment_status", "args": {"shipment": "TL-4471"}, "defect": "none"},

        # --- the tool does not exist -------------------------------------
        {"tool": "web_search", "args": {"query": "hold"}, "defect": "unknown_tool"},
        {"tool": "calculater", "args": {"expression": "1+1"}, "defect": "unknown_tool"},

        # --- a required argument is absent -------------------------------
        {"tool": "calculator", "args": {}, "defect": "missing_required"},
        {"tool": "read_file", "args": {"mode": "r"}, "defect": "missing_required"},

        # --- the argument is the wrong type ------------------------------
        {"tool": "search", "args": {"query": "hold", "k": "2"}, "defect": "wrong_type"},
        {"tool": "calculator", "args": {"expression": 42}, "defect": "wrong_type"},
        # bool is a subclass of int, so an unguarded isinstance check accepts
        # this and True then behaves as 1 everywhere downstream.
        {"tool": "search", "args": {"query": "hold", "k": True}, "defect": "bool_for_int"},

        # --- an argument the schema never described ----------------------
        {"tool": "read_file", "args": {"path": "notes/depot.txt", "encoding": "utf-8"},
         "defect": "extra_argument"},
        {"tool": "search", "args": {"query": "hold", "k": 2, "rerank": True},
         "defect": "extra_argument"},

        # --- schema-perfect, and hostile ---------------------------------
        {"tool": "read_file", "args": {"path": "notes/../../etc/passwd"},
         "defect": "adversarial_value"},
        {"tool": "read_file", "args": {"path": "/etc/shadow"},
         "defect": "adversarial_value"},
        {"tool": "calculator", "args": {"expression": "__import__('os').system('echo pwned')"},
         "defect": "adversarial_value"},
        {"tool": "calculator", "args": {"expression": "9 ** 9 ** 9"},
         "defect": "adversarial_value"},
    ]


def check(sandbox: Sandbox, call: dict, layer: str) -> bool:
    """True when this validator would let the call through to execution."""
    spec = sandbox.spec(call["tool"])
    if spec is None:
        return False                      # every layer checks the name exists
    if layer == "name_only":
        return True

    args = call["args"]
    problems = validate_call(spec, args)

    def kind(p: str) -> str:
        if p.startswith("missing required"):
            return "required"
        if p.startswith("unexpected argument"):
            return "no_extras"
        return "types"

    enforced = {"required": {"required"},
                "types": {"required", "types"},
                "no_extras": {"required", "types", "no_extras"}}[layer]
    return not any(kind(p) in enforced for p in problems)


def compute() -> dict[str, float]:
    calls = proposed_calls()
    by_defect: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_defect[c["defect"]].append(c)

    out: dict[str, float] = {
        "n_calls": len(calls),
        "n_defect_classes": len(by_defect) - 1,      # "none" is not a defect
    }
    for name, group in by_defect.items():
        out[f"n_{name}"] = len(group)

    # How many malformed calls each validator admits, and how many good ones
    # it wrongly rejects.
    for layer in LAYERS:
        sandbox = Sandbox(seed=1)
        admitted_bad = sum(1 for c in calls
                           if c["defect"] not in ("none", "adversarial_value")
                           and check(sandbox, c, layer))
        rejected_good = sum(1 for c in calls
                            if c["defect"] == "none" and not check(sandbox, c, layer))
        out[f"{layer}_admits_malformed"] = admitted_bad
        out[f"{layer}_rejects_wellformed"] = rejected_good
        for defect, group in by_defect.items():
            caught = sum(1 for c in group if not check(sandbox, c, layer))
            out[f"{layer}__{defect}"] = round(caught / len(group), 3)

    # The adversarial class, against the strictest validator, and then against
    # the tool itself.
    strict = Sandbox(seed=1)
    adversarial = by_defect["adversarial_value"]
    passed_validation = sum(1 for c in adversarial if check(strict, c, "no_extras"))
    out["adversarial_passing_validation"] = passed_validation
    out["adversarial_pct_passing_validation"] = round(
        100 * passed_validation / len(adversarial), 1)

    stopped_by_tool = 0
    for c in adversarial:
        res = strict.call(c["tool"], c["args"])
        if not res["ok"]:
            stopped_by_tool += 1
    out["adversarial_stopped_by_tool"] = stopped_by_tool
    out["adversarial_stopped_by_validation"] = len(adversarial) - passed_validation

    # And the invariant that matters most: nothing that failed validation ran.
    audit = Sandbox(seed=1)
    for c in calls:
        audit.call(c["tool"], c["args"])
    out["total_calls_made"] = len(audit.calls)
    out["total_executed"] = len(audit.executed_calls())
    out["rejected_before_execution"] = out["total_calls_made"] - out["total_executed"]
    out["executed_despite_failing_validation"] = sum(
        1 for c in audit.calls
        if c["executed"] and str(c.get("error", "")).startswith("invalid arguments")
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = compute()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0

    defects = ["unknown_tool", "missing_required", "wrong_type", "bool_for_int",
               "extra_argument", "adversarial_value"]
    print(f"{r['n_calls']} proposed calls, {r['n_defect_classes']} defect classes\n")
    print(f"{'defect':<20} {'n':>3}   " + "".join(f"{ly:>12}" for ly in LAYERS))
    for d in defects:
        row = "".join(f"{r[f'{ly}__{d}']:>12.2f}" for ly in LAYERS)
        print(f"{d:<20} {r[f'n_{d}']:>3}   {row}")
    print(f"{'(well formed)':<20} {r['n_none']:>3}   " +
          "".join(f"{r[f'{ly}__none']:>12.2f}" for ly in LAYERS))
    print("\n  fraction of each class the validator REJECTS; the bottom row is "
          "the false-positive rate\n")

    for ly in LAYERS:
        print(f"  {ly:<12} admits {r[f'{ly}_admits_malformed']} malformed, "
              f"wrongly rejects {r[f'{ly}_rejects_wellformed']} well-formed")

    print("\nadversarial values, against the strictest validator:")
    print(f"  pass validation entirely   {r['adversarial_passing_validation']}"
          f" of {r['n_adversarial_value']}  "
          f"({r['adversarial_pct_passing_validation']}%)")
    print(f"  stopped by the tool itself {r['adversarial_stopped_by_tool']}"
          f" of {r['n_adversarial_value']}")
    print(f"\ninvariant: {r['rejected_before_execution']} of {r['total_calls_made']} "
          f"calls rejected before execution; "
          f"{r['executed_despite_failing_validation']} executed despite failing validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
