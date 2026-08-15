---
status: Draft
last_verified: 2026-08-13
volatility: medium
---

# Module 4 · Agents and tool use

An agent is a program that lets a language model choose which of your functions
to run and with what arguments. Every lesson in this module is downstream of
that sentence: the loop, the memory, the permissions and the evaluation all
exist because a call the model proposed has to be turned into a call your
program makes.

This is the module where the curriculum stops being about a system that answers
and starts being about a system that acts, which changes what a mistake costs.
A retrieval system that ranks badly returns a worse answer; an agent that calls
badly reads a file it should not have, spends a budget it did not have, or
loops until someone notices.

Everything here runs against `llmlab.tools` — four tools, a recorded
trajectory, and no source of nondeterminism. There is no network, no clock and
no real filesystem: the "filesystem" is a dict, the "search index" is a fixed
document list, and the tool that fails does so on a seeded schedule rather than
at random, so an episode that failed once fails identically on the tenth run.

!!! warning "What a deterministic sandbox cannot show you"
    A seeded failure schedule makes retry behaviour teachable and measurable,
    and it says nothing about the failure *distribution* of a real service.
    Every lesson that reports a number from the sandbox says so, and lesson 4.6
    is explicit about which agent behaviours a replay harness can grade and
    which it can only record.

## Volatility

This module is marked `medium` rather than `low`, and the distinction is worth
stating. The mechanisms — validating a proposed call, bounding a loop, deciding
what to keep in a context window — are stable, and the lessons are written
about those. The wire formats and the named protocols around them are not, so
where a lesson names one it says what problem the format solves, which is the
part that will still be true when the format is not.

## Lessons

1. [4.1 The tool-calling protocol](01-tool-calling-protocol.md) — **available**
2. [4.2 The loop, and when to stop it](02-loop-and-termination.md) — **available**
3. [4.3 Planning and decomposition](03-planning-and-decomposition.md) — **available**
4. 4.4 Agent memory: working, episodic, semantic, and compaction — *planned*
5. 4.5 MCP as a protocol — *planned*
6. 4.6 Agent evaluation: trajectories, replay and determinism — *planned*
7. 4.7 Skills and progressive context disclosure — *planned*
8. 4.8 Sandboxing, permissions and the tool-output trust boundary — *planned*
9. 4.9 Multi-agent: when it helps and when it is just latency — *planned*
10. 4.10 The agent failure lab — *planned*

**Module 4 is in progress.** The mini-project is specified in `PLAN.md` §5 and
is built once the lessons it grades are written.
