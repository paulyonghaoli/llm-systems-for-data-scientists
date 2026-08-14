"""A deterministic sandboxed toolset, and the contract around calling it.

Module 4 needs tools an agent can actually call, and it needs them to behave
identically in CI, in the browser and on the tenth run of the same seed. So
nothing here touches the network, the clock, or the real filesystem: the
"filesystem" is a dict, the "search index" is a fixed document list, and the
tool that fails does so on a seeded schedule rather than at random.

The part worth reading closely is `validate_call`. Between a model proposing a
call and a program executing it there is a boundary, and everything an agent
can be made to do wrong crosses it. The validator is not a formality that
catches typos — it is the only thing standing between a generated string and a
function call.
"""

from __future__ import annotations

import ast
import operator
import random
from dataclasses import dataclass, field

#: Argument types a tool may declare. Deliberately small: a schema language
#: rich enough to be interesting is rich enough to disagree with itself, and
#: the lesson is about the boundary rather than about JSON Schema.
TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told a tool accepts.

    `parameters` maps an argument name to `{"type": ..., "required": bool,
    "description": str}`. This is the *published* contract; `validate_call`
    enforces it, and the two being the same object is the point — a schema the
    validator does not read is documentation, not a boundary.
    """

    name: str
    description: str
    parameters: dict[str, dict]

    def required(self) -> set[str]:
        return {k for k, v in self.parameters.items() if v.get("required")}


def validate_call(spec: ToolSpec, args: dict) -> list[str]:
    """Every problem with a proposed call. Empty means safe to execute.

    Returns *all* problems rather than the first, because a model that gets
    three arguments wrong should be told about three, not asked three times.
    """
    problems: list[str] = []

    if not isinstance(args, dict):
        return [f"arguments must be an object, got {type(args).__name__}"]

    for name in sorted(spec.required() - set(args)):
        problems.append(f"missing required argument '{name}'")

    # Unexpected arguments are rejected rather than ignored. Most validators
    # ignore them, and it is the quietest way a call becomes something the
    # published schema never described.
    for name in sorted(set(args) - set(spec.parameters)):
        problems.append(f"unexpected argument '{name}'")

    for name, value in sorted(args.items()):
        declared = spec.parameters.get(name)
        if declared is None:
            continue
        want = TYPES[declared["type"]]
        # bool is a subclass of int in Python, so an unguarded isinstance check
        # accepts True where an integer was declared -- and True indexes,
        # arithmetics and compares like 1 without ever looking wrong.
        if declared["type"] in ("integer", "number") and isinstance(value, bool):
            problems.append(f"argument '{name}' is a boolean, not a {declared['type']}")
        elif not isinstance(value, want):
            problems.append(
                f"argument '{name}' should be {declared['type']}, "
                f"got {type(value).__name__}"
            )
    return problems


# --- the calculator ---------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic without `eval`.

    `eval` on a model-generated string is the shortest path from a tool call to
    arbitrary code execution, so the expression is parsed to an AST and only
    the node types below are honoured. Anything else -- a name, a call, an
    attribute, a comprehension -- raises rather than being ignored.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse expression: {e.msg}") from None

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            # A model asked for a product will occasionally ask for 9 ** 9 ** 9,
            # which is not an error and will not return this decade.
            if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 1e6):
                raise ValueError("exponent out of range")
            return _BINOPS[type(node.op)](left, right)
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return walk(tree)


# --- the sandbox ------------------------------------------------------------

DEFAULT_FILES = {
    "notes/depot.txt": "North-east depot. Exceptions bay is bay 4.",
    "notes/policy.txt": "A hold lasts up to 48 hours.",
    "notes/readme.txt": "Operational notes. Do not distribute.",
}

DEFAULT_DOCS = [
    ("d1", "An address verification hold suspends movement for up to 48 hours."),
    ("d2", "Claims for loss must be submitted within 28 days of despatch."),
    ("d3", "Redelivery is attempted three times on consecutive working days."),
    ("d4", "Lithium cells may not be carried without a written declaration."),
]


@dataclass
class Sandbox:
    """Four tools, a recorded trajectory, and no source of nondeterminism.

    `seed` fixes the flaky tool's failure schedule, so an episode that failed
    once fails identically forever -- which is what makes a flaky tool
    teachable rather than merely annoying.
    """

    seed: int = 1
    files: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FILES))
    docs: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_DOCS))
    failure_rate: float = 0.4
    calls: list[dict] = field(default_factory=list)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("calculator", "Evaluate an arithmetic expression.",
                     {"expression": {"type": "string", "required": True,
                                     "description": "e.g. '3 * (4 + 5)'"}}),
            ToolSpec("search", "Search the operations notes.",
                     {"query": {"type": "string", "required": True,
                                "description": "words to search for"},
                      "k": {"type": "integer", "required": False,
                            "description": "how many results, default 2"}}),
            ToolSpec("read_file", "Read a file under notes/.",
                     {"path": {"type": "string", "required": True,
                               "description": "e.g. 'notes/depot.txt'"}}),
            ToolSpec("shipment_status", "Look up a consignment's status.",
                     {"shipment": {"type": "string", "required": True,
                                   "description": "e.g. 'TL-4471'"}}),
        ]

    def spec(self, name: str) -> ToolSpec | None:
        return next((s for s in self.specs() if s.name == name), None)

    def call(self, name: str, args: dict) -> dict:
        """Validate, then execute. Returns an envelope, never raises.

        A tool that raises into the agent loop is a tool that ends the episode,
        so every outcome -- unknown tool, invalid arguments, execution failure
        -- comes back as a result the agent can read and act on. That is the
        difference between an error the model can recover from and a crash.
        """
        record = {"tool": name, "args": args}
        spec = self.spec(name)

        if spec is None:
            known = ", ".join(s.name for s in self.specs())
            out = {"ok": False, "error": f"no such tool '{name}'; available: {known}"}
            self.calls.append({**record, **out, "executed": False})
            return out

        problems = validate_call(spec, args)
        if problems:
            out = {"ok": False, "error": "invalid arguments: " + "; ".join(problems)}
            # `executed` is the field that matters for grading: a call that
            # failed validation must never have run.
            self.calls.append({**record, **out, "executed": False})
            return out

        try:
            value = self._execute(name, args)
            out = {"ok": True, "value": value}
        except Exception as e:  # noqa: BLE001 - the envelope is the point
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.calls.append({**record, **out, "executed": True})
        return out

    def _execute(self, name: str, args: dict) -> object:
        if name == "calculator":
            return safe_eval(args["expression"])

        if name == "search":
            k = args.get("k", 2)
            if k < 1:
                raise ValueError("k must be at least 1")
            terms = {w for w in args["query"].lower().split() if len(w) > 2}
            scored = [
                (sum(1 for w in terms if w in text.lower()), did, text)
                for did, text in self.docs
            ]
            hits = sorted((s for s in scored if s[0] > 0),
                          key=lambda r: (-r[0], r[1]))[:k]
            return [{"id": d, "text": txt} for _, d, txt in hits]

        if name == "read_file":
            path = args["path"]
            # Path traversal is the reason this tool takes a path rather than a
            # file id, and the reason it refuses one. `notes/../secrets` is a
            # perfectly ordinary string until it reaches a filesystem.
            if ".." in path or path.startswith("/") or "\\" in path:
                raise PermissionError(f"path escapes the sandbox: {path!r}")
            if not path.startswith("notes/"):
                raise PermissionError(f"only notes/ is readable, got {path!r}")
            if path not in self.files:
                raise FileNotFoundError(f"no such file: {path!r}")
            return self.files[path]

        if name == "shipment_status":
            shipment = args["shipment"]
            # Seeded on the argument, so the same lookup fails the same way for
            # the same seed however many times the agent retries it. A flaky
            # tool that succeeds on retry teaches retrying; one that does not
            # teaches giving up, and an agent needs both.
            rng = random.Random(f"{self.seed}:{shipment}")
            if rng.random() < self.failure_rate:
                raise TimeoutError(f"status service timed out for {shipment}")
            return {"shipment": shipment, "status": "in transit", "depot": "north-east"}

        raise AssertionError(f"unreachable: {name}")

    def executed_calls(self) -> list[dict]:
        return [c for c in self.calls if c["executed"]]
