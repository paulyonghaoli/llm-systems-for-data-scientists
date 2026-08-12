"""Gate 18 — every quoted number came from code, and still does.

    python tools/gate_numbers.py

Contract. When a lesson states a measured number, it is followed immediately
by a marker naming the experiment and the key that produced it:

    English prose costs 0.206 <!-- computed: bpe_compression.tpc_english_cl100k -->
    tokens per character.

This gate re-runs `experiments/<name>.py --json` and checks the literal in the
prose still matches the value the code produces, at the precision the prose
quotes it to. Only closing punctuation may sit between the number and its
marker.

It exists because of a failure that is invisible to every other check: a
number that was true when it was written. Re-deriving it by hand is exactly
the work nobody does, so it is a gate.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXPERIMENTS = ROOT / "experiments"

MARKER = re.compile(
    r"([-+]?\d[\d,]*(?:\.\d+)?)"  # the number as written
    r"([^\d<\n]{0,10})"  # closing markup / units, no digits
    r"<!--\s*computed:\s*([A-Za-z_][\w]*)\.([\w]+)\s*-->"
)


def run_experiment(name: str, cache: dict[str, dict]) -> dict | None:
    if name in cache:
        return cache[name]
    script = EXPERIMENTS / f"{name}.py"
    if not script.exists():
        cache[name] = {}
        return None
    proc = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        cache[name] = {}
        print(f"  experiment {name} failed:\n{proc.stderr.strip()[-800:]}", file=sys.stderr)
        return None
    cache[name] = json.loads(proc.stdout)
    return cache[name]


def main() -> int:
    problems: list[str] = []
    cache: dict[str, dict] = {}
    checked = 0

    for md in sorted(DOCS.rglob("*.md")):
        rel = md.relative_to(ROOT).as_posix()
        text = md.read_text(encoding="utf-8")
        for m in MARKER.finditer(text):
            literal_raw, _, exp_name, key = m.groups()
            lineno = text.count("\n", 0, m.start()) + 1
            values = run_experiment(exp_name, cache)
            if values is None:
                problems.append(f"{rel}:{lineno}: no runnable experiments/{exp_name}.py")
                continue
            if key not in values:
                problems.append(
                    f"{rel}:{lineno}: {exp_name} produces no key {key!r} "
                    f"(has {len(values)} keys)"
                )
                continue
            checked += 1
            literal = literal_raw.replace(",", "")
            decimals = len(literal.split(".")[1]) if "." in literal else 0
            want = f"{float(literal):.{decimals}f}"
            got = f"{float(values[key]):.{decimals}f}"
            if want != got:
                problems.append(
                    f"{rel}:{lineno}: prose says {literal_raw} for {exp_name}.{key}, "
                    f"code now produces {got}"
                )

    print(f"computed numbers checked: {checked} (across {len(cache)} experiment(s))")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
