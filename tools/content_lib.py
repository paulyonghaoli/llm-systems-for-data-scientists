"""Shared loader/validator for interactive content (quiz banks, code exercises).

Authoring format is YAML under `curriculum/<module>/questions/*.yaml` and
`curriculum/<module>/exercises/*.yaml`. At mkdocs build time these are
converted to JSON under `docs/assets/generated/` for the front-end components;
in CI every exercise's reference solution is executed against its own tests
and every starter is executed to confirm it does *not* pass.
"""

from __future__ import annotations

import ast
import inspect
import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CURRICULUM = REPO_ROOT / "curriculum"
GENERATED = REPO_ROOT / "docs" / "assets" / "generated"

QUESTION_TYPES = {"single", "multi", "numeric"}

#: Modules that in-browser exercise code may import. Pyodide ships CPython
#: plus a fixed package set; anything outside this list either does not exist
#: in the browser or reaches the network. Enforced by gate 17.
PYODIDE_ALLOWED_IMPORTS = {
    # stdlib subset actually used by exercises
    "collections", "dataclasses", "itertools", "json", "math", "random", "re",
    "statistics", "string", "textwrap", "unicodedata", "functools", "heapq",
    "bisect", "typing", "__future__", "hashlib", "array", "copy", "enum",
    # available in Pyodide
    "numpy", "scipy", "sklearn", "pandas",
    # ours
    "llmlab",
}


@dataclass
class ContentError:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ContentSet:
    quizzes: dict[str, dict] = field(default_factory=dict)  # id -> bank
    exercises: dict[str, dict] = field(default_factory=dict)  # id -> spec
    errors: list[ContentError] = field(default_factory=list)


def _validate_quiz(bank: dict, path: str, errors: list[ContentError]) -> None:
    if not bank.get("id"):
        errors.append(ContentError(path, "quiz bank missing 'id'"))
        return
    if not bank.get("lesson"):
        errors.append(ContentError(path, "quiz bank missing 'lesson'"))
    questions = bank.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append(ContentError(path, "quiz bank needs a non-empty 'questions' list"))
        return
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "?")
        loc = f"{path}#{qid}"
        if qid in seen:
            errors.append(ContentError(loc, "duplicate question id"))
        seen.add(qid)
        qtype = q.get("type", "single")
        if qtype not in QUESTION_TYPES:
            errors.append(ContentError(loc, f"unknown type {qtype!r}"))
        if not q.get("prompt"):
            errors.append(ContentError(loc, "missing prompt"))
        if qtype == "numeric":
            if not isinstance(q.get("answer"), (int, float)):
                errors.append(ContentError(loc, "numeric question needs a numeric 'answer'"))
            if not isinstance(q.get("tolerance"), (int, float)):
                errors.append(ContentError(loc, "numeric question needs a numeric 'tolerance'"))
        else:
            opts = q.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                errors.append(ContentError(loc, "needs >= 2 options"))
                continue
            n_correct = sum(1 for o in opts if o.get("correct"))
            if qtype == "single" and n_correct != 1:
                errors.append(
                    ContentError(loc, f"single-choice needs exactly 1 correct, has {n_correct}")
                )
            if qtype == "multi" and n_correct < 1:
                errors.append(ContentError(loc, "multi-choice needs >= 1 correct option"))
            for i, o in enumerate(opts):
                if not o.get("text"):
                    errors.append(ContentError(loc, f"option {i} missing text"))
                # A distractor without an explanation is a missed teaching
                # opportunity; the component renders one per option.
                if not o.get("explanation"):
                    errors.append(ContentError(loc, f"option {i} missing explanation"))


def _validate_exercise(spec: dict, path: str, errors: list[ContentError]) -> None:
    for key in ("id", "title", "lesson", "starter_code", "tests", "solution"):
        if not spec.get(key):
            errors.append(ContentError(path, f"exercise missing '{key}'"))
    hints = spec.get("hints")
    if hints is not None and not isinstance(hints, list):
        errors.append(ContentError(path, "'hints' must be a list"))
    provided = spec.get("provided")
    if provided is not None and not isinstance(provided, dict):
        errors.append(ContentError(path, "'provided' must be a mapping of name -> {summary, ...}"))


def _exec(src: str, ns: dict) -> None:
    """Run exercise code the way the browser does.

    `exec` inherits __future__ flags from the calling frame, and this module
    uses `from __future__ import annotations`. Left alone, that silently runs
    every exercise under a different language mode than Pyodide uses — most
    visibly turning annotations into strings. `dont_inherit=True` removes the
    difference.
    """
    exec(compile(src or "", "<exercise>", "exec", dont_inherit=True), ns)  # noqa: S102


def _summarize_value(value: object, compact: bool = False) -> str:
    """One-line description of a value the learner is handed."""
    if isinstance(value, bool | int | str | type(None)):
        return repr(value)
    if isinstance(value, float):
        if not compact:
            return repr(value)
        s = f"{value:.6g}"
        return s if any(c in s for c in ".en") else s + ".0"
    shape = getattr(value, "shape", None)
    if shape is not None:  # numpy array
        return f"array{tuple(shape)}"
    if isinstance(value, list | tuple | set | dict):
        if len(value) <= 6 and isinstance(value, list | tuple):
            inner = ", ".join(_summarize_value(x, compact=True) for x in value)
            return f"({inner})" if isinstance(value, tuple) else f"[{inner}]"
        return f"{type(value).__name__} of {len(value)}"
    return type(value).__name__


def _type_name(t: object) -> str:
    """Short, readable name for an annotation."""
    if t is inspect.Parameter.empty:
        return ""
    return getattr(t, "__name__", str(t)).replace("numpy.", "np.")


def _render_signature(name: str, value: object) -> str:
    try:
        sig = inspect.signature(value)
    except (TypeError, ValueError):
        return f"{name}(...)"
    parts = []
    for p in sig.parameters.values():
        s = p.name
        ann = _type_name(p.annotation)
        if ann:
            s += f": {ann}"
        if p.default is not inspect.Parameter.empty:
            s += f" = {p.default!r}" if ann else f"={p.default!r}"
        parts.append(s)
    out = f"{name}({', '.join(parts)})"
    ret = _type_name(sig.return_annotation)
    return f"{out} -> {ret}" if ret else out


def _used_names(spec: dict) -> set[str]:
    """Names the learner's starter and the reference solution actually read."""
    used: set[str] = set()
    for src in (spec.get("starter_code", ""), spec.get("solution", "")):
        try:
            tree = ast.parse(src or "")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
    return used


def build_provided(spec: dict, errors: list[str] | None = None) -> list[dict]:
    """Describe the hidden objects `setup_code` hands the learner.

    The learner never sees `setup_code`, so a name appearing in the starter
    with no contract leaves them guessing at the signature — which argument is
    which, what the units are, whether a call has a side effect. Everything
    derivable is derived here (exact signature, constant values) so it cannot
    drift from the code; the author supplies only what introspection cannot
    know, through an optional `provided:` block in the YAML:

        provided:
          wilson:
            summary: 95% score interval for a proportion.
            notes:
              - Returns a (lo, hi) pair, already clamped to [0, 1].
            example: wilson(40, 50)

    `example` is evaluated at build time and its real output is baked in, so a
    worked call cannot go stale either. Set `hide: true` to withhold an object
    whose docstring would give away the answer.
    """
    setup = spec.get("setup_code", "")
    if not (setup or "").strip():
        return []
    ns: dict = {}
    try:
        _exec(setup, ns)
    except Exception as e:  # noqa: BLE001
        if errors is not None:
            errors.append(f"setup_code failed: {type(e).__name__}: {e}")
        return []

    authored = spec.get("provided") or {}
    if not isinstance(authored, dict):
        authored = {}
    for name in authored:
        if name not in ns and errors is not None:
            errors.append(f"'provided' documents {name!r}, which setup_code never defines")

    used = _used_names(spec)
    out: list[dict] = []
    for name, value in ns.items():  # insertion order == definition order
        if name.startswith("__") or isinstance(value, types.ModuleType):
            continue
        extra = authored.get(name) or {}
        # Author-documented names are always shown; otherwise only what the
        # learner's own code touches, so the panel is a reference rather than
        # a dump of the exercise's internals.
        if name not in used and name not in authored:
            continue
        if extra.get("hide"):
            continue

        # Imports are not "provided by the exercise". `from collections import
        # Counter` would otherwise appear in the panel carrying the standard
        # library's own docstring, which is noise the learner can look up and
        # which crowds out the objects the author actually wrote. Objects
        # defined inside setup_code have __module__ of None (functions) or
        # "builtins" (classes), because exec runs in a namespace with no
        # __name__; anything else came from an import.
        if callable(value) and name not in authored:
            origin = getattr(value, "__module__", None)
            if origin not in (None, "builtins"):
                continue

        entry: dict = {"name": name}
        if callable(value):
            entry["kind"] = "class" if isinstance(value, type) else "function"
            entry["signature"] = _render_signature(name, value)
        else:
            entry["kind"] = "constant"
            entry["value"] = _summarize_value(value)

        # An authored summary wins over the docstring, which is written for
        # someone reading the source and may name the bug outright. Constants
        # get no docstring fallback: inspect.getdoc(1.0) returns float's own
        # docstring, which is nonsense here.
        summary = extra.get("summary") or ""
        if not summary and entry["kind"] == "function":
            summary = inspect.getdoc(value) or ""
        elif not summary and entry["kind"] == "class":
            # `inspect.getdoc` walks the MRO for classes, so an undocumented
            # `class Timeout(Exception)` silently inherits "Common base class
            # for all non-exit exceptions." — which passes the gate while
            # telling the learner nothing about *this* exception. Own
            # docstring only; a confidently wrong summary is worse than none.
            summary = (value.__dict__.get("__doc__") or "") if isinstance(value, type) else ""
        if summary:
            entry["summary"] = " ".join(summary.split())
        if extra.get("notes"):
            entry["notes"] = list(extra["notes"])
        if extra.get("example"):
            entry["example"] = extra["example"]
            try:
                entry["example_out"] = _summarize_value(
                    eval(extra["example"], dict(ns)), compact=True)  # noqa: S307
            except Exception as e:  # noqa: BLE001
                if errors is not None:
                    errors.append(
                        f"provided example for {name!r} raised {type(e).__name__}: {e}")
        out.append(entry)
    return out


def run_exercise_solution(spec: dict) -> str | None:
    """Execute setup + reference solution + tests. Returns error text or None.

    Same namespace-exec model the Pyodide worker uses, run in local CPython.
    This is what stops in-browser exercises rotting silently.
    """
    ns: dict = {}
    try:
        _exec(spec.get("setup_code", ""), ns)
        _exec(spec["solution"], ns)
        _exec(spec["tests"], ns)
    except AssertionError as e:
        return f"reference solution FAILS its own tests: {e}"
    except Exception as e:  # noqa: BLE001
        return f"error running solution: {type(e).__name__}: {e}"
    return None


def run_exercise_starter(spec: dict) -> str | None:
    """Return None if the starter FAILS its tests (correct), else a message.

    An exercise whose starter already passes asks the learner for nothing.
    It is invisible to every other check in the toolchain and to eye review.
    """
    ns: dict = {}
    try:
        _exec(spec.get("setup_code", ""), ns)
        _exec(spec["starter_code"], ns)
    except Exception:  # noqa: BLE001
        return None  # a starter that cannot even run is a failing starter
    try:
        _exec(spec["tests"], ns)
    except Exception:  # noqa: BLE001
        return None
    return "starter PASSES its own tests - the exercise asks for nothing"


def load_all() -> ContentSet:
    cs = ContentSet()
    for path in sorted(CURRICULUM.glob("*/questions/*.yaml")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            bank = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            cs.errors.append(ContentError(rel, f"YAML parse error: {e}"))
            continue
        _validate_quiz(bank, rel, cs.errors)
        if bank.get("id"):
            if bank["id"] in cs.quizzes:
                cs.errors.append(ContentError(rel, f"duplicate quiz bank id {bank['id']!r}"))
            cs.quizzes[bank["id"]] = bank
    for path in sorted(CURRICULUM.glob("*/exercises/*.yaml")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            cs.errors.append(ContentError(rel, f"YAML parse error: {e}"))
            continue
        _validate_exercise(spec, rel, cs.errors)
        if spec.get("id"):
            if spec["id"] in cs.exercises:
                cs.errors.append(ContentError(rel, f"duplicate exercise id {spec['id']!r}"))
            cs.exercises[spec["id"]] = spec
    return cs


def _write_if_changed(path: Path, content: str) -> bool:
    """Avoid touching unchanged files: `mkdocs serve` watches docs/ and would
    otherwise rebuild in a loop."""
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def export_json(cs: ContentSet) -> int:
    """Convert loaded content to docs/assets/generated/*.json. Returns count written."""
    written = 0
    for bank_id, bank in cs.quizzes.items():
        if _write_if_changed(GENERATED / "quizzes" / f"{bank_id}.json", json.dumps(bank, indent=1)):
            written += 1
    for ex_id, spec in cs.exercises.items():
        public = {
            k: spec.get(k)
            for k in (
                "id", "title", "description", "starter_code", "setup_code",
                "tests", "hints", "solution",
            )
        }
        public["provided"] = build_provided(spec)
        out = json.dumps(public, indent=1)
        if _write_if_changed(GENERATED / "exercises" / f"{ex_id}.json", out):
            written += 1
    return written
