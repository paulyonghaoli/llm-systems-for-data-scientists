"""Gates 3, 4 and 5: schema, reference solutions pass, starters fail.

- Schema-validates every quiz bank and exercise spec.
- Executes every exercise's reference solution against its own tests.
- Executes every exercise's STARTER and requires it to fail.
- Exports the JSON consumed by the front-end components.

Exit code 0 = everything valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Gate 23. Shorter than this and the message is not saying anything.
MIN_FIRST_FAILURE_CHARS = 25

from tools.content_lib import (  # noqa: E402
    ContentError,
    build_provided,
    export_json,
    load_all,
    run_exercise_solution,
    run_exercise_starter,
    starter_first_failure,
)


def main() -> int:
    cs = load_all()
    failures = list(cs.errors)

    for ex_id, spec in sorted(cs.exercises.items()):
        err = run_exercise_solution(spec)
        if err:
            failures.append(ContentError(f"exercise {ex_id}", err))
        err = run_exercise_starter(spec)
        if err:
            failures.append(ContentError(f"exercise {ex_id}", err))

        # Gate 20. The learner cannot read setup_code, so anything it hands
        # them has to carry its own contract: a name and a signature do not
        # say what the arguments mean or what comes back. Give the function a
        # docstring, or describe it under `provided:` in the YAML.
        perrs: list[str] = []
        for item in build_provided(spec, perrs):
            if item["kind"] != "constant" and not item.get("summary"):
                failures.append(ContentError(
                    f"exercise {ex_id}",
                    f"{item['signature']} is handed to the learner with no "
                    f"description: give it a docstring or a `provided:` summary"))
        failures.extend(ContentError(f"exercise {ex_id}", e) for e in perrs)

        # Gate 24. A starter must fail its TESTS, not fail to compile. Gate 4
        # accepts a starter that cannot run at all, on the reasonable ground
        # that it is certainly not passing — but a SyntaxError in scaffolding
        # the author wrote shows the learner an error in code they did not
        # write, about something the exercise is not teaching. Found by
        # noticing a starter that printed nothing.
        for field in ("setup_code", "starter_code", "solution", "tests"):
            src = spec.get(field) or ""
            try:
                compile(src, f"<{ex_id}.{field}>", "exec")
            except SyntaxError as e:
                failures.append(ContentError(
                    f"exercise {ex_id}",
                    f"{field} does not compile: line {e.lineno}: {e.msg}"))

        # Gate 23. The first assertion a failing starter trips is the message
        # the learner actually reads, so it has to say something.
        first = starter_first_failure(spec)
        if first is not None and len(first.strip()) < MIN_FIRST_FAILURE_CHARS:
            failures.append(ContentError(
                f"exercise {ex_id}",
                f"the starter's first failure shows {first.strip()!r} — a bare "
                f"`assert` with no message. That line is the whole teaching "
                f"surface of the exercise; say what went wrong and why"))

    n = export_json(cs)
    print(f"quiz banks: {len(cs.quizzes)}  exercises: {len(cs.exercises)}  json rewritten: {n}")

    if failures:
        print(f"\n{len(failures)} content error(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("content OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
