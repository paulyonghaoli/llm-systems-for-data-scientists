# Contributing

## The one command

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,docs]"
python tools/verify.py     # every gate CI runs, unpiped
```

`tools/verify.py` is the only local check to trust. **Never pipe a gate into
`tail`, `head` or `grep` when its exit status is what you rely on** — a shell
pipeline's exit status is the *last* command's. In the sibling robotics
curriculum, running `python tools/validate_content.py | tail -2` inside an
`&&` chain masked seven failing exercises across five commits while CI was red.

To iterate on one exercise:

```bash
python tools/check_one.py tok-l1-encode
```

It runs the solution *and* the starter, and prints what each one produced.

## Content status

Every lesson declares one of four states in its front matter, plus a
`last_verified` date and a `volatility` flag:

| State | Means |
|---|---|
| **Draft** | Written, not checked |
| **Reviewed** | Re-read on a different day; claims checked against sources |
| **Verified** | Every code path runs; every quoted number produced by a script |
| **Reproducible** | The above, plus regenerated from scratch in CI on every commit |

`volatility: high` lessons fail CI once they pass 180 days without
re-verification. This field is not decoration — it is why gate 16 exists.

## Lesson schema

Every lesson uses the same `##` headings, in this order. Gate 19 checks it
mechanically, which is the entire reason the schema is uniform.

**A** Why this matters · **B** Mental model · **C** Mechanism ·
**D** From data science to LLM systems · **E** Minimal implementation ·
**F** Production practice · **G** Experiment ·
**H** Failure modes and cost traps · **I** Graded practice ·
**J** Annotated references · **K** Extension

Reference example: [docs/modules/01-tokens/01-bpe.md](docs/modules/01-tokens/01-bpe.md).

## Rules that produced this repository

**Verify, don't assert.** Run the thing. When a number appears in a lesson it
comes from code that produced it — literally, via a
`<!-- computed: experiment.key -->` marker that gate 18 re-checks on every
commit.

**Wrong assumptions are the content.** When something surprises you,
instrument it before theorising, and when you turn out to be wrong, the fix
goes *into the lesson* rather than being quietly corrected. Two examples
already in this repository: comparing serialization formats per character
gave the opposite answer to comparing them per unit of information (lesson 1.1
§H), and the context-packer's reference implementation failed its own rubric
on a case the spec never defined (mini-project 1, rule 5).

**Helpers go in `setup_code`, never `starter_code`.** Otherwise the reference
solution raises `NameError` when the learner's edits replace them.

**Sweep seeds before shipping a grader.** Thresholds that hold on seed 1
routinely fail on seed 3. `python -m grader --sweep 30` is a gate.

**Never grade a learner's prompt text against a mock model.** That grades
their ability to reverse-engineer a fake we wrote. Gate 12.

**Never use destructive git commands to undo a small edit.**

## Authoring content

Quiz banks live in `curriculum/<module>/questions/*.yaml`, exercises in
`curriculum/<module>/exercises/*.yaml`. A MkDocs hook converts them to JSON at
build time; the front-end components read that.

Every option in a multiple-choice question needs an `explanation`, including
the correct one — a distractor without a reason is a missed lesson, and the
schema check enforces it.

**A starter's demo loop must not crash.** Starters usually print a few worked
cases before the tests run, and an unguarded loop that raises partway through
shows the learner a traceback from scaffolding rather than the behaviour the
exercise is about. Wrap the loop so a raising case prints `raised TypeError`
and the rest still run — the partial output *is* the diagnosis. Gate 24 catches
a starter that will not compile; this one is a convention because a starter
that raises at runtime is sometimes exactly right.

**The starter's first failure is the exercise.** Whichever assertion a failing
starter trips first is the only message most learners will read, so it has to
say what went wrong and why — gate 23 rejects a bare `assert` in that position.
Later assertions may be bare when a messaged one has already explained the
idea; the gate deliberately checks the first failure rather than all of them,
because requiring a message everywhere produces noise instead of teaching.

Every exercise starter must **fail** its own tests. An exercise whose starter
passes asks for nothing, is invisible to every other check in the toolchain,
and eye review does not catch it. Gate 4.

**Every object `setup_code` hands the learner needs a contract.** The learner
cannot open `setup_code`, so a name appearing in the starter with only a
signature leaves them guessing at what the arguments mean and what comes back.
Give the function a docstring — which improves the source at the same time — or
describe it under `provided:` in the YAML, which also covers constants and
supports notes and a worked example whose output is computed at build time.
Gate 20.

Imports are excluded from the panel automatically: `from collections import
Counter` is not something your exercise provides, and the standard library's
own docstring is noise. Classes use their *own* docstring only, because
`inspect.getdoc` walks the MRO and would let an undocumented
`class Timeout(Exception)` inherit "Common base class for all non-exit
exceptions." — a summary that passes the gate and tells the learner nothing.


## Lesson depth

Lessons are 2,500–4,000 words of prose, and gate 21 enforces the floor along
with two style measures: mean sentence length at least 22 words, and no more
than 18% of sentences under eleven words. The targets are 25–30 and under 15%
respectively; the gate leaves headroom so it fails on drift rather than on
rounding.

Length alone is not the point. Depth means **defining every term at first use**
in a `!!! info "Terms used in this lesson"` box and adding it to
[the glossary](docs/glossary.md), attaching a worked number to each claim,
preferring a table over a sentence that gestures at a list, and answering the
objection a sceptical reader is about to raise. Padding will pass the word
count and fail the reader.

The style measures exist because the failure they catch is invisible otherwise:
a stream of short sentences with bolded one-line reveals reads as notes rather
than as a chapter. Join clauses with explicit connectives — *because, since, so
that, which means* — so the reasoning sits inside the sentences rather than in
the gaps between them.

`tools/audit.py` carries a `DEEPENING_BACKLOG` of lessons written before the
gate existed. It may only shrink: a lesson on the list that now passes is
itself reported as an error, so the exemption cannot outlive its reason.


## Figures

Figures are generated, never drawn: `tools/figures.py` renders each one light
and dark from the same experiment code the prose quotes, so a figure cannot
disagree with the table beside it. Embed with `md_in_html` inside
`<figure class="llm-fig" markdown>` using two markdown images tagged
`{.fig-light}` and `{.fig-dark}` — never a raw `<img>`, which mkdocs passes
through verbatim so `--strict` never validates the path and a broken figure
404s silently.

Determinism is load-bearing: `svg.hashsalt` is pinned and the SVG date
metadata stripped, so a re-render of unchanged code is byte-identical. Gate 22
relies on that to diff committed SVGs against a fresh render, which is the
pictorial version of gate 18. After changing a figure function, run
`python tools/figures.py` and commit what it writes.

## Deploying

The site is served from Cloudflare Workers static assets. There is no Worker
code, so nothing runs server-side and nothing can.

```bash
python tools/verify.py && .venv/Scripts/mkdocs build && npx wrangler deploy
```

Deploy is **manual and deliberate**: no push publishes anything, and the build
step is separate from the upload so a red gate stops a release rather than
shipping alongside it.

The site is in **soft launch** — reachable, and excluded from search engines by
`docs/robots.txt` and a `noindex, nofollow` meta tag injected through
`overrides/main.html`. `python tools/launch.py` toggles both together, because
the two guards drifting apart is the failure that makes a soft launch
imaginary:

```bash
python tools/launch.py --status     # which state am I in?
python tools/launch.py --go         # allow indexing
python tools/launch.py --unlaunch   # restore the guards
```

Repository visibility is not scripted anywhere; that stays a manual decision.

## The gates

Twenty-four, listed in `PLAN.md` §7 and run by `tools/verify.py`. Seven are
specific to this subject: no prompt-graded-by-mock, no network at test time,
fixture integrity, volatility containment, lesson freshness, Pyodide import
allowlist, and computed-number verification.

When you check something by hand, add it to `tools/audit.py` so it never has
to be checked by hand again.
