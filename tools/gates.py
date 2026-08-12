"""Gates 12, 14, 15, 16 and 17 — the ones specific to this subject.

    python tools/gates.py

12  no exercise grades learner-authored prompt text against a model client
14  every recorded fixture/cassette carries a date and a version; no secrets
15  no volatile literal (model name, price, context limit) outside docs/living/
16  every lesson declares status + last_verified; volatile lessons must be fresh
17  exercise code imports only what Pyodide actually has

Each is here because of a specific way this curriculum could rot without any
other check noticing.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.content_lib import PYODIDE_ALLOWED_IMPORTS, load_all  # noqa: E402

DOCS = ROOT / "docs"
LIVING = DOCS / "living"

#: How stale a `volatility: high` lesson may get before CI complains.
STALE_DAYS = 180


# --- 12 ---------------------------------------------------------------------

CLIENT_CALL = re.compile(r"\b(MockClient|CassetteClient|\.generate\(|\.complete\(|\.chat\()")
PROMPT_VAR = re.compile(r"^\s*(?:system_)?prompt\s*=\s*[\"']", re.M)


def gate_12(problems: list[str]) -> None:
    """PLAN.md §5c: never score a learner's prompt text by feeding it to a mock.

    That grades their ability to reverse-engineer a fake we wrote. It is the
    most tempting shortcut in this subject, so it is a gate rather than a
    convention.
    """
    cs = load_all()
    for eid, spec in sorted(cs.exercises.items()):
        authors_prompt = PROMPT_VAR.search(spec.get("starter_code", "") or "")
        tests_call_model = CLIENT_CALL.search(spec.get("tests", "") or "")
        if authors_prompt and tests_call_model:
            problems.append(
                f"exercise {eid}: starter asks the learner to author prompt text and the "
                f"tests invoke a model client - that grades the mock, not the skill "
                f"(PLAN.md §5c)"
            )


# --- 14 ---------------------------------------------------------------------

SECRET = re.compile(
    r"\bsk-[A-Za-z0-9]{8,}"
    r"|\bapi[_-]?key\s*[:=]\s*[\"'][^\"']{8,}"
    r"|\bBearer\s+[A-Za-z0-9._-]{20,}"
)


def gate_14(problems: list[str]) -> None:
    """A recorded fixture without a date and a version is an undated claim
    about the world, which is exactly what this curriculum must not make."""
    recorded = sorted((ROOT / "data").glob("**/*.json"))
    if not recorded:
        problems.append("data/: no recorded fixtures found at all")
    for path in recorded:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if SECRET.search(text):
            problems.append(f"{rel}: looks like it contains a credential")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            problems.append(f"{rel}: invalid JSON: {e}")
            continue
        if not isinstance(payload, dict):
            continue
        if "recorded_on" not in payload:
            problems.append(f"{rel}: recorded fixture has no 'recorded_on' date")
        else:
            try:
                date.fromisoformat(str(payload["recorded_on"]))
            except ValueError:
                problems.append(f"{rel}: 'recorded_on' is not an ISO date")
        if not any(k.endswith("_version") or k == "model" for k in payload):
            problems.append(f"{rel}: recorded fixture pins no version or model")


# --- 15 ---------------------------------------------------------------------

VOLATILE = [
    (re.compile(r"\bgpt-[0-9]", re.I), "a model name"),
    (re.compile(r"\bclaude-[a-z0-9]", re.I), "a model name"),
    (re.compile(r"\bgemini-[0-9]", re.I), "a model name"),
    (re.compile(r"\bllama[- ][0-9]", re.I), "a model name"),
    (re.compile(r"\$\s*[\d.]+\s*(?:/|per)\s*(?:1?[MK]|million|thousand)\s*tokens", re.I),
     "a price"),
    (re.compile(r"\b\d[\d,]*\s*k?\s*[- ]token context", re.I), "a context limit"),
    (re.compile(r"\bcontext window of\s+\d", re.I), "a context limit"),
]


def gate_15(problems: list[str]) -> None:
    """Model names, prices and context limits change monthly. If they are
    scattered through 88 lessons, the curriculum decays everywhere at once
    and there is no single place to re-audit. They live in docs/living/."""
    # Quiz and exercise text is rendered into pages too, so scanning only
    # docs/ would leave a hole exactly the size of the question banks.
    sources = [p for p in DOCS.rglob("*.md") if LIVING not in p.parents]
    sources += sorted((ROOT / "curriculum").rglob("*.yaml"))

    for md in sources:
        rel = md.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for pattern, what in VOLATILE:
                m = pattern.search(line)
                if m:
                    problems.append(
                        f"{rel}:{lineno}: {what} ({m.group(0).strip()!r}) outside docs/living/"
                    )


# --- 16 ---------------------------------------------------------------------

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def gate_16(problems: list[str]) -> None:
    import yaml

    today = date.today()
    for md in sorted((DOCS / "modules").rglob("*.md")):
        rel = md.relative_to(ROOT).as_posix()
        m = FRONT_MATTER.match(md.read_text(encoding="utf-8"))
        if not m:
            problems.append(f"{rel}: no YAML front matter (needs status + last_verified)")
            continue
        meta = yaml.safe_load(m.group(1)) or {}
        status = meta.get("status")
        if status not in {"Draft", "Reviewed", "Verified", "Reproducible"}:
            problems.append(f"{rel}: status {status!r} is not one of the four declared states")
        raw = meta.get("last_verified")
        if raw is None:
            problems.append(f"{rel}: no last_verified date")
            continue
        try:
            verified = raw if isinstance(raw, date) else date.fromisoformat(str(raw))
        except ValueError:
            problems.append(f"{rel}: last_verified {raw!r} is not an ISO date")
            continue
        if verified > today:
            problems.append(f"{rel}: last_verified {verified} is in the future")
        age = (today - verified).days
        if meta.get("volatility") == "high" and age > STALE_DAYS:
            problems.append(f"{rel}: volatile lesson last verified {age} days ago (> {STALE_DAYS})")


# --- 17 ---------------------------------------------------------------------


def gate_17(problems: list[str]) -> None:
    """Exercise code that imports `torch` or `requests` looks fine in CI and
    dies in the browser, where the learner meets it."""
    cs = load_all()
    for eid, spec in sorted(cs.exercises.items()):
        for field_name in ("setup_code", "starter_code", "solution", "tests"):
            src = spec.get(field_name) or ""
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                problems.append(f"exercise {eid}.{field_name}: syntax error: {e}")
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    if root not in PYODIDE_ALLOWED_IMPORTS:
                        problems.append(
                            f"exercise {eid}.{field_name}: imports {root!r}, which Pyodide "
                            f"does not have (allowlist is in tools/content_lib.py)"
                        )


GATES = [
    ("12 prompt-not-graded-by-mock", gate_12),
    ("14 fixture integrity", gate_14),
    ("15 volatility containment", gate_15),
    ("16 lesson freshness", gate_16),
    ("17 pyodide imports", gate_17),
]


def main() -> int:
    all_problems: list[str] = []
    for name, fn in GATES:
        problems: list[str] = []
        fn(problems)
        print(f"  {name}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
        all_problems.extend(problems)
    if all_problems:
        print()
        for p in all_problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
