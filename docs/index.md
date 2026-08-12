# LLM Systems for Data Scientists

An interactive, project-based curriculum for people who already know Python,
pandas, sklearn, statistics and offline evaluation — and who have never
trained a transformer, shipped a retrieval system, sized a KV cache, or
written an evaluation that had to catch a regression in a model they do not
control.

This is self-study, written up and shared. It is not a course you enrol in and
there is nobody to email. Everything runs in your browser; nothing is
transmitted anywhere.

## Where this is

**Phase P1, in progress.** Course I is being built. Five lessons exist:

- [**Module 0 · From data science to LLM systems**](modules/00-transition/index.md)
  — what changes when the model is someone else's, the anatomy of an LLM
  application, why your evaluation habits break, cost and latency as features,
  and a diagnostic lab. Five lessons, ten exercises, and a graded mini-project.
- [**Module 1 · Tokens, sampling and the API contract**](modules/01-tokens/index.md)
  — BPE from scratch, the sampler you actually control, what a chat template
  really sends, what an API promises, and how to cost a design before you
  build it. Five lessons, eleven exercises, and a graded mini-project.
- [**Module 2 · Prompting and structured output**](modules/02-prompting/index.md)
  — deliberately not a book of phrasings: the machinery around the wording.
  One of five lessons so far.

Start with Module 0 if you are new to this material; start with
[1.1](modules/01-tokens/01-bpe.md) if you want to see what a lesson here looks
like at full depth.

The [curriculum map](curriculum.md) shows the full plan and what is and is not
built. Nothing on it is claimed as available until it is.

## How it works

Four tiers of practice, escalating:

1. **Inline checks** inside the prose — click to reveal.
2. **Graded quizzes**, with an explanation for every wrong option as well as
   the right one.
3. **In-browser coding exercises**. Real CPython, running locally in your
   browser via Pyodide. Every starter contains a real bug rather than a blank;
   run it first and read what it prints.
4. **Autograded artifacts** — one mini-project per module, one capstone per
   course, scored against a published rubric on seeded inputs.

Progress is stored in your browser's `localStorage` and can be exported as
JSON from the [progress page](progress.md). Not cookies: a cookie is
transmitted on every request, which is pure waste against a static site and is
exactly the thing that makes consent banners necessary. Nothing here leaves
your browser, so there is nothing to consent to.

## What it will not do

No accounts, no backend, no certificates, no cost. No live API calls anywhere
in the material — behaviour from real models is *recorded*, dated and
version-pinned, so every exercise runs offline and forever. No multi-GPU work
and no CUDA kernels; the infrastructure material is taught as cost models,
which is both what fits in a browser and what the job actually asks for.

Where something genuinely cannot be checked by a machine, it says so and tells
you what to do instead, rather than substituting a quiz for a skill.

## Honest status

Content decays fast in this field. Every lesson carries a status, a
last-verified date, and a volatility flag; anything version-dependent lives in
[one living document](living/models.md) rather than scattered through the
lessons, and CI enforces that. If you are reading this a year from now, check
the dates.
