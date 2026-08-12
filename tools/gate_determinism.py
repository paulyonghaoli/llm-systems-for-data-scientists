"""Gate 11 — the same seed twice produces the identical score.

    python tools/gate_determinism.py

A grader with hidden nondeterminism (a set iteration order, an unseeded
shuffle, a dict that used to be ordered) gives two learners different marks
for identical work, and gives the author a flaky CI they will eventually
learn to ignore. Cheap to check, so it is checked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "projects"


def main() -> int:
    problems: list[str] = []
    checked = 0

    for project in sorted(PROJECTS.iterdir()):
        if not (project / "grader" / "__init__.py").exists():
            continue
        sys.path.insert(0, str(project))
        try:
            import grader  # noqa: PLC0415

            first = json.dumps(grader.run_reference(seed=3, n_scenarios=5), sort_keys=True)
            second = json.dumps(grader.run_reference(seed=3, n_scenarios=5), sort_keys=True)
            if first != second:
                problems.append(f"{project.name}: two runs at seed 3 disagree")
            checked += 1
        except Exception as e:  # noqa: BLE001
            problems.append(f"{project.name}: {type(e).__name__}: {e}")
        finally:
            sys.path.remove(str(project))
            for mod in [m for m in sys.modules if m == "grader" or m.startswith("grader.")]:
                del sys.modules[mod]

    print(f"graders checked for determinism: {checked}")
    if problems:
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
