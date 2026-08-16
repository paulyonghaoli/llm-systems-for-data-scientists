# Living doc · pinned versions and volatile facts

**Last audited: 2026-08-09.**

Everything in this curriculum that decays lives here, and nowhere else. That
is a CI gate, not a convention: `tools/gates.py` fails the build if a model
name, a price or a context limit appears in a lesson.

The reason is arithmetic. Scattered through 88 lessons, a price change means
88 places to re-check and 88 chances to miss one. Confined to one page, it
means one page — and a reader who wants to know how stale the claim is can
look at the date at the top rather than guessing.

## What is pinned right now

| Thing | Pin | Recorded | Used by |
|---|---|---|---|
| `tiktoken` | 0.13.0 | 2026-08-09 | `data/fixtures/tiktoken_counts.json` |
| `cl100k_base` vocabulary | 100,277 symbols | 2026-08-09 | lesson 1.1 §A, §G |
| `o200k_base` vocabulary | 200,019 symbols | 2026-08-09 | lesson 1.1 §G |
| Pyodide | 0.26.4 | 2026-08-09 | the in-browser exercise runner |
| Python | ≥ 3.11 | — | everything |

## How the pins work

No exercise, experiment or grader in this repository ever calls a live API or
downloads a vocabulary. Real behaviour is captured once by a recorder script
that is run by hand — `experiments/record_tiktoken.py` — and everything
downstream reads the recorded fixture. That is what lets the entire test suite
run offline, which is itself gate 13.

The cost of that choice is honest and worth stating: **a recorded number
describes the world on the day it was recorded.** If you are reading this long
after the date above, re-record and see what moved. Instructions are in the
recorder's docstring.

## Re-recording is content, not maintenance

When a re-record changes a number, the diff goes into the lesson rather than
quietly replacing the old value. Watching a fixed prompt's token count shift
across tokenizer generations is one of the few ways to *show* rather than
assert that this field decays. The 2026-08-09 recording already contains one
such observation: moving from `cl100k_base` to `o200k_base` changed English,
Python and numeric text by nothing at all, and Japanese by roughly a quarter.

## Model names, prices and context limits

Absent from the *lessons*, deliberately, and that is the CI gate above. This
curriculum teaches mechanisms that outlive a model generation, and a table of
prices would be wrong within months of being written. When a lesson needs
concrete figures for cost arithmetic, it uses figures the learner supplies for
whichever provider they actually use — which is also the only version of the
exercise that is worth anything.

Where the state of the field genuinely has to be discussed, it is discussed
here in `docs/living/`, dated and with its sources marked:

- **[The frontier](frontier.md)** — evaluation, agent reliability, effective
  context length, cost and the open-weight gap, and how each maps onto a
  module. Researched 2026-08-16.
- **[The role field](roles.md)** — where this work sits in the job market, and
  a blunt note on how much of that literature is untraceable. Researched
  2026-08-16.

Both carry a re-audit checklist rather than a promise to stay current.
