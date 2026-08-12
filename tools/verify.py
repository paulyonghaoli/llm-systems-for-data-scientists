"""Run every gate CI runs, locally, unpiped, and fail loudly.

    python tools/verify.py

**Never pipe a gate into `tail`, `head` or `grep` when its exit status is what
you are relying on.** A shell pipeline's exit status is the *last* command's.
In the sibling robotics curriculum that exact mistake — running
`python tools/validate_content.py | tail -2` inside an `&&` chain — masked
seven failing exercises across five commits while CI was red.

The 19 gates are listed in PLAN.md §7. Gate numbers below match that table.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TOK_MINI = ROOT / "projects" / "tokenizer_mini"
EXTRACTION = ROOT / "projects" / "extraction_harness"
REL_MINI = ROOT / "projects" / "reliability_report"

GATES: list[tuple[str, list[str], Path]] = [
    ("01 ruff", [PY, "-m", "ruff", "check", "."], ROOT),
    ("02 pytest", [PY, "-m", "pytest", "-q"], ROOT),
    # 03 solutions pass, 04 starters fail, 05 schema
    ("03-05 content", [PY, "tools/validate_content.py"], ROOT),
    # 06 strict build (also catches every broken internal link)
    ("06 mkdocs strict", [PY, "-m", "mkdocs", "build", "--strict"], ROOT),
    # 08 graders run against their reference and score full marks
    ("08 grader:tokenizer_mini", [PY, "-m", "grader", "--reference", "--seed", "1"], TOK_MINI),
    ("08 grader:reliability_report", [PY, "-m", "grader", "--reference", "--seed", "1"], REL_MINI),
    # 10 the thresholds hold across seeds, not just the one they were tuned on
    ("10 sweep:tokenizer_mini", [PY, "-m", "grader", "--sweep", "30"], TOK_MINI),
    ("10 sweep:reliability_report", [PY, "-m", "grader", "--sweep", "30"], REL_MINI),
    ("08 grader:extraction_harness",
     [PY, "-m", "grader", "--reference", "--seed", "1"], EXTRACTION),
    ("10 sweep:extraction_harness", [PY, "-m", "grader", "--sweep", "30"], EXTRACTION),
    ("11 determinism", [PY, "tools/gate_determinism.py"], ROOT),
    ("12-17 subject gates", [PY, "tools/gates.py"], ROOT),
    ("13 no network", [PY, "tools/no_network.py"], ROOT),
    ("18 computed numbers", [PY, "tools/gate_numbers.py"], ROOT),
    # 22 committed figures match a fresh render from the same code
    ("22 figures", [PY, "tools/figures.py", "--check"], ROOT),
    # 25 the retrieval corpus contains the phenomena it claims to contain,
    # and 26 checks that gate 25 can actually fail (it passed all nine of its
    # own checks on the first run, which is when a check is least trustworthy)
    ("25 corpus phenomena", [PY, "tools/verify_corpus.py"], ROOT),
    ("26 corpus verifier", [PY, "tools/test_verify_corpus.py"], ROOT),
    # 07 nav coverage and orphans live here too
    ("19 audit", [PY, "tools/audit.py"], ROOT),
]


def capstone_gates() -> list[tuple[str, list[str], Path]]:
    """Gate 09. Capstone I lands at the end of P1; until then, say so rather
    than passing silently, because a gate with nothing to check looks
    identical to a gate that is working."""
    found = sorted(p for p in (ROOT / "projects").glob("capstone_*") if p.is_dir())
    return [
        (f"09 capstone:{p.name}", [PY, "-m", "eval", "run", "--seed", "1"], p) for p in found
    ]


def main() -> int:
    gates = GATES + capstone_gates()
    caps = [g for g in gates if g[0].startswith("09")]

    failures = []
    for name, cmd, cwd in gates:
        print(f"--- {name} ", end="", flush=True)
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.returncode == 0:
            print("ok")
        else:
            print("FAIL")
            failures.append(name)
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-30:]
            for line in tail:
                print(f"    {line}")

    if not caps:
        print("--- 09 capstone: none built yet (Capstone I closes P1) - nothing to run")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(gates)} gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
