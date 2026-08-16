# The frontier: where LLM systems are going (and where this curriculum sits)

**Living document · Researched 2026-08-16.** Frontier content churns fast:
names here are coordinates, not gospel. Claims are marked **[V]** verified
against a primary source, **[C]** company claim unverified externally, **[U]**
single or secondary source — treat with suspicion.

This page exists because of a CI gate. `tools/gates.py` fails the build if a
model name, a price or a context limit appears in a lesson, so everything that
decays is confined to `docs/living/`. That is a maintenance decision — one page
to re-audit rather than a hundred — and it is also the reason you should read
the date above before trusting anything below.

---

## The one-sentence summary

**The capability curve and the reliability curve have separated.** Models keep
clearing harder benchmarks while the systems built on them keep failing in
production for reasons that have nothing to do with model quality — evaluation
that does not measure what it claims, agents that are excellent per step and
poor per episode, and context windows whose advertised size is not their usable
size. Every one of those is a measurement problem, which is unusually good news
for someone arriving from data science.

---

## 1. The evaluation crisis

The benchmarks the field ran on two years ago are largely spent. MMLU,
HumanEval, HellaSwag and the original GSM8K are effectively retired by a
combination of saturation and contamination [U], and SWE-bench Verified went
from roughly 60% to near-100% in a single year [V, [AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)].
A benchmark everyone passes has stopped carrying information.

Contamination is not merely a historical problem with old test sets. Audits
have found frontier models able to reproduce verbatim gold patches or problem
statements for some SWE-bench Verified tasks [U] — on a benchmark specifically
built to be harder to contaminate.

The response has been to make benchmarks time-aware rather than static:
LiveCodeBench scores models only on problems published after their training
cutoff, LiveBench rotates fresh questions in monthly. There is also a live
research thread on *construct validity* — whether a benchmark measures the
thing its name claims — which is the question a statistician would have asked
first.

**Why this matters here.** Course IV is built on this, and Module 3 already
demonstrates the local version twice. In lesson 3.6 the same reranker looks
transformative at recall@1 (0.051 → 0.250, p < 0.0001) and statistically
invisible at recall@10 (p = 0.22) — a team standardised on one metric would
have measured a real effect and found nothing. Lesson 3.8 then shows two
different systems taking first place across five metrics on the same runs.

---

## 2. The agent reliability gap

This is the most decision-relevant number in the whole area, and it is not a
capability number.

**τ-bench introduced pass^k — the probability an agent succeeds on *all* k
attempts — and the gap is enormous: 61% pass@1 against 25% pass@8** on retail
agent tasks [V, τ-bench]. The single-run score that gets published is roughly
2.5× the number that describes what a user experiences across a working week.
Reported pass^4 scores commonly sit 15–25 points below pass^1 [U].

The compounding arithmetic is the other half, and it needs no source because it
is multiplication: three chained steps at 70% each succeed 34% of the time.
Long-horizon agents are not failing because any one step is bad.

**METR's time-horizon work** is the best-constructed capability measure in the
area: the length of task (measured by human expert completion time) an agent
completes 50% of the time. It doubled roughly every seven months over
2019–2025 and about every four months more recently, reaching somewhere above
fourteen hours in 2026 [U on the current figure]. Read METR's own caveats
before quoting it: **measurements above 16 hours are unreliable with their
current task suite** [V, [METR](https://metr.org/time-horizons/)], the suite is
about 230 mostly-coding tasks, and METR itself notes that a single year's data
gives a less robust doubling estimate than the long run.

**Why this matters here.** Module 4 is downstream of the compounding
arithmetic. Lesson 4.2's stopping rules exist because episodes fail rather than
steps, and its `pass^k`-shaped observation — that a rule saving 51.5% of tokens
also silently destroys 5 of 16 successes — is the same class of finding.

---

## 3. Context: the advertised window is not the usable window

Effective context length is substantially shorter than the number on the box.
Benchmarks that strip out easy lexical shortcuts (NoLiMa) report frontier
models falling from near-perfect short-context accuracy to roughly 70% by
32k tokens, with most models below half their short-context baseline at the
same length [U]. RULER's evaluation across 17 long-context models confirms the
older lost-in-the-middle result: recall for facts placed centrally drops
sharply against facts at either end [U].

The practical consensus in 2026 is hybrid rather than either-or — retrieve
first, then reason over a large but curated window — for a cost reason as much
as a quality one, since retrieval uses a small fraction of the tokens and
time-to-first-token on a very long prompt is seconds rather than
sub-second [U].

**Why this matters here.** "Long context killed RAG" was a real prediction and
it did not happen. Module 3 is not legacy content, and Module 4.4's finding
that carrying a 20-step history costs 10.6× the text itself is the
agent-shaped version of the same economics.

---

## 4. Cost, and the open-weight convergence

**Inference cost for a GPT-3.5-class capability fell roughly 280× in about two
years** [V, AI Index 2026]. That is the single most consequential number for
anyone designing systems, because it means an architecture chosen on cost
grounds has a short shelf life.

**Open-weight models now lag closed frontier models by about four months, or
8 points of Epoch's capability index** [V, [Epoch AI](https://epoch.ai/data-insights/open-closed-eci-gap),
May 2026] — down from six to ten months through most of 2025. Epoch's own
caveats apply: confidence in individual placements varies, and the index is one
aggregation among possible others.

The AI Index's surrounding figures are worth holding onto for scale: US AI
investment of $285.9bn in 2025, data-centre capacity around 29.6 GW, and
organisational adoption at 88% [V].

---

## 5. Reality check: what is actually deployed

| Claim in circulation | What the source actually says |
|---|---|
| "95% of enterprise AI pilots fail" | MIT Project NANDA's *GenAI Divide* (Jul 2025): 95% of pilots delivered **no measurable P&L impact**, from 52 executive interviews, 153 survey responses and 300 public deployments. That is a claim about attributable profit, not about whether anything shipped. [V as a report; widely misquoted] |
| "88% enterprise adoption" | AI Index 2026, organisational adoption of AI in some form [V]. Not the same population as the sentence above, and the two are routinely juxtaposed as if they were. |
| "Agents succeed 56.6% of the time in production, across 6,259 deployments" | Traced to a single blog with no published methodology. **Not used on this page.** [U] |
| "AI job demand exceeds supply 3.2 : 1 / 1.6M open roles" | Same problem — repeated across several marketing sites, no primary source located. **Not used.** [U] |

The last two rows are the point of the table. This subject has a large
secondary literature written for search traffic, in which confident numbers
circulate with no traceable origin. Two of the most quotable figures I found
while researching this page could not be traced to any methodology at all, and
excluding them is more informative than including them with a caveat.

---

## 6. What this means for the curriculum

The frontier's own diagnosis — measurement, reliability, cost — maps onto this
programme better than a capability-led story would.

| Frontier problem | Where it lives here |
|---|---|
| Benchmarks that measure the wrong thing | Module 0.3, Module 3.8, all of Course IV |
| Contamination and time-aware evaluation | Course IV, Module 13 |
| Agents good per step and poor per episode | Module 4.2, 4.6; Capstone I |
| Retrieval still necessary despite long context | Module 3 |
| Context cost growing quadratically with episode length | Module 4.4 |
| Cost arithmetic outliving any particular price | Modules 1.5, 10, 11 |
| Open weights making self-hosting a real option | Course III |

**The honest counter-position**, which the curriculum should hold itself to: it
is possible that the reliability gap closes on its own as models improve, in
which case a lot of careful evaluation machinery becomes unnecessary. The
evidence against that is that the gap has persisted across several capability
generations while the benchmarks kept rising — but it is a bet, not a fact, and
this page should be re-read against that question at each audit.

---

## What is deliberately absent

No table of prices, context limits or per-model leaderboard positions. Those
change monthly, and a curriculum that quotes them is wrong within a quarter.
Where a lesson needs concrete figures for cost arithmetic, the learner supplies
figures for whichever provider they actually use — which is the only version of
that exercise worth doing.

See also [pinned versions](models.md) for the fixtures and library versions
this repository actually depends on, and
[the role field](roles.md) for where this work sits in the job market.

---

## Re-audit checklist

For whoever revisits this page — including me:

1. Has the evaluation crisis resolved, or just moved to new benchmarks?
2. Has pass^k or an equivalent reliability measure become standard reporting?
3. Is METR's time horizon still doubling, and is their task suite still the
   binding constraint on measuring it?
4. Did the open/closed gap keep narrowing?
5. Which claims on this page are now [U] that were [V], because the source
   moved or stopped being maintained?
