"""Autograder for the reliability-report mini-project.

    python -m grader --seed 1                 grade student.py
    python -m grader --reference --seed 1     grade the reference (CI does this)
    python -m grader --sweep 30               30 consecutive seeds, reference only
"""

from __future__ import annotations


def run_reference(seed: int = 1, n_scenarios: int = 12) -> dict:
    """Entry point used by tools/verify.py and the no-network gate."""
    from grader import reference, rubric

    return rubric.grade(reference.report, seed=seed, n_scenarios=n_scenarios)
