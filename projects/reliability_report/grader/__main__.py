from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import reference, rubric  # noqa: E402

PASS_MARK = 80.0


def _report(result: dict, label: str) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    for row in result["breakdown"]:
        bar = f"{row['passed']}/{row['of']}"
        print(f"  {row['key']}  {row['name']:<28} {bar:>7}   "
              f"{row['earned']:>5.1f} / {row['points']}")
    print(f"  {'':<3} {'TOTAL':<28} {'':>7}   {result['total']:>5.1f} / 100")
    if result["failures"]:
        print(f"\n  first failures ({len(result['failures'])} total):")
        for f in result["failures"][:8]:
            print(f"    - {f}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="grader")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenarios", type=int, default=12)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        for seed in range(1, args.sweep + 1):
            result = rubric.grade(reference.report, seed=seed, n_scenarios=4)
            if result["total"] < 100.0:
                _report(result, f"reference, seed {seed}")
                print(f"\nFAIL: reference scored {result['total']} on seed {seed}")
                return 1
        print(f"sweep of {args.sweep} seeds: reference scores 100.0 on every one")
        return 0

    if args.reference:
        result = rubric.grade(reference.report, seed=args.seed, n_scenarios=args.scenarios)
        _report(result, f"reference solution, seed {args.seed}")
        if result["total"] < 100.0:
            print("\nFAIL: the reference must score full marks against its own rubric")
            return 1
        return 0

    import student

    result = rubric.grade(student.report, seed=args.seed, n_scenarios=args.scenarios)
    _report(result, f"student.py, seed {args.seed}")
    print()
    if result["total"] >= PASS_MARK:
        print(f"PASS ({result['total']}/100, pass mark {PASS_MARK})")
        return 0
    print(f"not yet ({result['total']}/100, pass mark {PASS_MARK})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
