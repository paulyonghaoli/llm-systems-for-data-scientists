"""Run one exercise's solution and starter, with output.

    python tools/check_one.py tok-l1-merge

Iterating on a single exercise through the full validator is slow, and a
failing exercise's *printed output* is usually the fastest route to
understanding why.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.content_lib import load_all, run_exercise_solution, run_exercise_starter  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    cs = load_all()
    if target not in cs.exercises:
        print(f"no such exercise: {target}")
        print("available:", ", ".join(sorted(cs.exercises)))
        return 2

    spec = cs.exercises[target]
    rc = 0
    sol = run_exercise_solution(spec)
    if sol:
        print(f"FAIL solution  {target}\n  {sol}")
        rc = 1
    else:
        print(f"ok   solution  {target}")

    start = run_exercise_starter(spec)
    if start:
        print(f"FAIL starter   {target}\n  {start}")
        rc = 1
    else:
        print(f"ok   starter   {target} (fails its tests, as it must)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
