"""Gate 13 — the whole content suite runs with the network switched off.

    python tools/no_network.py

Blocks the socket layer, then runs everything that executes learner-facing
code: every exercise setup/solution/starter/test, every experiment that
lessons quote numbers from, and the project graders.

Why it is a gate and not a convention: an exercise or a fixture that quietly
calls a live API works perfectly on the author's machine, works in CI while
the key is present, and then breaks for every learner — who has no key, no
account, and no idea why the page is spinning. The rule in PLAN.md is
"cassettes or nothing"; this is what enforces it.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class NetworkAccessError(RuntimeError):
    pass


def _blocked(*args: object, **kwargs: object) -> None:
    raise NetworkAccessError(
        "network access attempted while gate 13 was running - content must be "
        "reproducible offline (see PLAN.md §9)"
    )


def block_network() -> None:
    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]


def main() -> int:
    block_network()
    problems: list[str] = []

    # 1. Every exercise, executed exactly as the browser executes it.
    from tools.content_lib import load_all, run_exercise_solution, run_exercise_starter

    cs = load_all()
    for eid, spec in sorted(cs.exercises.items()):
        try:
            err = run_exercise_solution(spec) or run_exercise_starter(spec)
        except NetworkAccessError as e:
            problems.append(f"exercise {eid}: {e}")
            continue
        if err:
            problems.append(f"exercise {eid}: {err}")
    print(f"  exercises offline: {len(cs.exercises)}")

    # 2. Every experiment a lesson quotes numbers from.
    n_exp = 0
    for script in sorted((ROOT / "experiments").glob("*.py")):
        if script.stem.startswith("record_"):
            continue  # recorders are run by hand, with the network, on purpose
        try:
            module = __import__(f"experiments.{script.stem}", fromlist=["compute"])
            if hasattr(module, "compute"):
                module.compute()
                n_exp += 1
        except NetworkAccessError as e:
            problems.append(f"experiments/{script.name}: {e}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"experiments/{script.name}: {type(e).__name__}: {e}")
    print(f"  experiments offline: {n_exp}")

    # 3. Every project grader, in reference mode.
    n_proj = 0
    for project in sorted((ROOT / "projects").iterdir()):
        if not (project / "grader" / "__init__.py").exists():
            continue
        sys.path.insert(0, str(project))
        try:
            import grader  # noqa: PLC0415

            grader.run_reference(seed=1, n_scenarios=3)
            n_proj += 1
        except NetworkAccessError as e:
            problems.append(f"{project.name}: {e}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{project.name}: {type(e).__name__}: {e}")
        finally:
            sys.path.remove(str(project))
            for mod in [m for m in sys.modules if m == "grader" or m.startswith("grader.")]:
                del sys.modules[mod]
    print(f"  graders offline: {n_proj}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
