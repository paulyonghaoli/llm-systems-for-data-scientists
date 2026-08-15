"""The tool boundary, held in place.

Lesson 4.1 makes two claims that are only worth making if they stay true: that
no call failing validation ever executes, and that the guards inside the tools
stop what validation cannot see. Both are properties of `llmlab.tools` rather
than of the experiment that measures them, so they are asserted here.
"""

from __future__ import annotations

import pytest

from llmlab.tools import Sandbox, ToolSpec, safe_eval, validate_call

SPEC = ToolSpec(
    "search",
    "Search the operations notes.",
    {
        "query": {"type": "string", "required": True, "description": "words"},
        "k": {"type": "integer", "required": False, "description": "how many"},
    },
)


# --- validate_call ----------------------------------------------------------

def test_wellformed_call_has_no_problems():
    assert validate_call(SPEC, {"query": "hold", "k": 2}) == []
    assert validate_call(SPEC, {"query": "hold"}) == [], "k is optional"


def test_bool_is_not_an_integer():
    """bool subclasses int, so an unguarded isinstance check accepts True."""
    problems = validate_call(SPEC, {"query": "hold", "k": True})
    assert len(problems) == 1
    assert "boolean" in problems[0]


def test_unexpected_arguments_are_rejected_not_ignored():
    problems = validate_call(SPEC, {"query": "hold", "rerank": True})
    assert problems == ["unexpected argument 'rerank'"]


def test_every_problem_is_reported():
    problems = validate_call(SPEC, {"k": True, "rerank": 1})
    assert len(problems) == 3, problems


def test_report_is_independent_of_argument_order():
    a = validate_call(SPEC, {"k": True, "rerank": 1})
    b = validate_call(SPEC, {"rerank": 1, "k": True})
    assert a == b


def test_non_object_arguments():
    problems = validate_call(SPEC, ["query", "hold"])
    assert len(problems) == 1
    assert "object" in problems[0]


# --- safe_eval --------------------------------------------------------------

@pytest.mark.parametrize(
    ("expression", "expected"),
    [("3 * (4 + 5)", 27.0), ("-2 + 10", 8.0), ("7 % 4", 3.0), ("2 ** 10", 1024.0)],
)
def test_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "[x for x in range(3)]",
        "9 ** 9 ** 9",
        "1 / 0",
        "3 +",
    ],
)
def test_hostile_or_unbounded_expressions_raise(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


# --- the sandbox ------------------------------------------------------------

ADVERSARIAL = [
    ("read_file", {"path": "notes/../../etc/passwd"}),
    ("read_file", {"path": "/etc/shadow"}),
    ("read_file", {"path": "notes\\..\\secret.txt"}),
    ("calculator", {"expression": "__import__('os').system('echo pwned')"}),
    ("calculator", {"expression": "9 ** 9 ** 9"}),
]


@pytest.mark.parametrize(("tool", "args"), ADVERSARIAL)
def test_adversarial_values_pass_validation_and_are_stopped_by_the_tool(tool, args):
    """The division of labour lesson 4.1 measures, asserted rather than asserted about."""
    box = Sandbox(seed=1)
    spec = box.spec(tool)
    assert validate_call(spec, args) == [], "these are schema-perfect by construction"

    result = box.call(tool, args)
    assert result["ok"] is False, result
    assert box.calls[-1]["executed"] is True, "it reached the tool; the guard is what refused"


def test_nothing_failing_validation_ever_executes():
    box = Sandbox(seed=1)
    for tool, args in [
        ("search", {}),                                   # missing required
        ("search", {"query": "hold", "k": True}),          # bool for int
        ("search", {"query": "hold", "rerank": True}),     # unexpected
        ("web_search", {"query": "hold"}),                 # no such tool
    ]:
        box.call(tool, args)
    assert box.executed_calls() == []
    assert all(c["ok"] is False for c in box.calls)


def test_call_never_raises():
    box = Sandbox(seed=1)
    for tool, args in [*ADVERSARIAL, ("nope", {}), ("read_file", {"path": 3})]:
        assert box.call(tool, args)["ok"] is False


def test_read_file_reaches_only_notes():
    box = Sandbox(seed=1)
    assert box.call("read_file", {"path": "notes/depot.txt"})["ok"] is True
    assert box.call("read_file", {"path": "notes/absent.txt"})["ok"] is False


def test_flaky_tool_is_deterministic():
    """A failure that reproduces is a failure a lesson can be written about."""
    a = [Sandbox(seed=1).call("shipment_status", {"shipment": s})["ok"]
         for s in ("TL-4471", "TL-9002", "TL-1150", "TL-3318")]
    b = [Sandbox(seed=1).call("shipment_status", {"shipment": s})["ok"]
         for s in ("TL-4471", "TL-9002", "TL-1150", "TL-3318")]
    assert a == b
    assert len(set(a)) == 2, "the fixture is useless if every lookup goes the same way"

    box = Sandbox(seed=1)
    repeats = [box.call("shipment_status", {"shipment": "TL-4471"})["ok"] for _ in range(5)]
    assert len(set(repeats)) == 1, "retrying must not change the outcome within a seed"


def test_depot_defaults_are_unchanged():
    """Lessons 4.1 and 4.2 see one depot; only 4.3 populates the mapping."""
    box = Sandbox(seed=1, failure_rate=0.0)
    assert box.call("shipment_status", {"shipment": "TL-4471"})["value"]["depot"] == "north-east"


def test_depots_make_a_dependency_discoverable():
    """4.3 needs an answer that is not knowable before the lookup happens."""
    box = Sandbox(seed=1, failure_rate=0.0,
                  depots={"TL-4471": "south-west", "TL-9002": "central"})
    assert box.call("shipment_status", {"shipment": "TL-4471"})["value"]["depot"] == "south-west"
    assert box.call("shipment_status", {"shipment": "TL-9002"})["value"]["depot"] == "central"
    # Anything unmapped keeps the default, so the extension is purely additive.
    assert box.call("shipment_status", {"shipment": "TL-0000"})["value"]["depot"] == "north-east"


def test_specs_are_self_describing():
    """Gate 20's rule, applied to the model: nothing is handed over undescribed."""
    for spec in Sandbox().specs():
        assert spec.description.strip()
        for name, declared in spec.parameters.items():
            assert declared.get("description"), f"{spec.name}.{name} has no description"
            assert declared["type"] in {"string", "integer", "number", "boolean"}
