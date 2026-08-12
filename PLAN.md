# LLM Systems for Data Scientists — Master Plan

**Version:** 0.3 · **Date:** 2026-08-09 · **Status:** governing document. Later
sessions treat this file as authoritative and amend it when scope changes.

**Where things stand:** **P0 complete**; **P1 (Course I) under way.**
**Modules 0, 1 and 2 are complete** — five lessons and a graded
mini-project each. Module 3 is next. The deepening backlog was cleared on
2026-08-11 and **every lesson meets gate 21** — a median of 2,556 words against
the 2,500 floor, mean sentence length 25.7 words, 11.8% short sentences, up
from 1,574 / 20.3 / 20.5% before the pass.

Fifteen lessons, 32 exercises, 15 question banks (122 questions), 3 autograded
mini-projects, 14 experiments, 8 generated figures, 24 gates green and
browser-verified. See §11–14 for the build logs. Next: Module 3, on the corpus
decided in §10a.

**Audience.** Data scientists who are fluent in Python, pandas, NumPy, sklearn,
statistics, experiment design and offline evaluation — and who have never
trained a transformer, shipped an LLM feature, sized a KV cache, or run an eval
that had to catch a regression in a system they don't control.

**Tone.** This is self-study, written up and shared with peers. It is not a
resume project and must never read like one. No promotion on any channel.

---

## 1. Scale calibration, in honest units

### The unit

US graduate **semester credit-hours**, because that is the unit this audience
already calibrates on, plus a second unit that is more useful in practice:
**weeks to first production LLM feature**.

| Unit | Standard | This project |
|---|---|---|
| 1 graduate credit | ≈ 45 hrs total learner work | — |
| 3-credit course | ≈ 135 hrs (reading + practice + project) | 1 course = 3–5 modules + 1 capstone |
| 12 credits | ≈ 540 hrs learner work | 4 courses, 16 modules, ~87 lessons, 4 capstones |
| Major textbook | 600–900 pages | ~87 lessons × 2,500–4,000 words ≈ 800 page-equivalents |

**Second unit.** Course I alone (~135 hrs, ~10 weeks at 12 hrs/week) is
calibrated to "you can now build, evaluate and cost a retrieval or agent feature
that survives contact with a real user." That is the claim Course I has to earn,
and it is the one most learners actually want. Courses II–IV extend to "you can
reason about, adapt, serve and gate a model," which is a different and longer
claim.

### Authoring cost, honestly

Estimated from the shape of the `robotics-for-ai-engineers` build, which used
the same workflow (AI-assisted drafting, human verification, every gate run):

| Item | Unit cost | Count | Hours |
|---|---:|---:|---:|
| Lesson, all four tiers, verified | 8–14 hrs | 87 | 700–1,230 |
| Module mini-project (autograded) | 12–20 hrs | 16 | 190–320 |
| Capstone (rubric, seeds, reference, ablations) | 50–80 hrs | 4 | 200–320 |
| Platform: components, schemas, Pyodide worker, CI | — | — | 60–90 |
| Corpus + cassette + embedding fixtures | — | — | 40–70 |
| **Total** | | | **≈ 1,200–2,000 hrs** |

At 10–15 hrs/week that is roughly **two to three years** for all four courses.
Course I on its own is ~600 hrs — about **nine months to a year**. Those numbers
are the reason for the phasing rule below.

### The phasing rule

**Every phase must be independently shippable.** Nothing in this plan may
depend on finishing everything. If the project stops after Course I, what exists
is a complete, coherent, CI-green course on building and evaluating LLM
applications — not a stub of something larger. Course numbers are fixed at the
outset (see §4) precisely so that build order can change without breaking a
single link, exercise id, or nav path. That lesson is inherited: the robotics
curriculum was restructured twice after ~50 lessons existed and nothing broke,
because module *numbers* never moved.

---

## 2. Lesson schema

Every lesson uses the same named sections, in the same order. Uniformity is not
an aesthetic choice — it is what makes drift **mechanically detectable**. With
one schema, a script can find every lesson missing its failure-modes section
across 87 files. With bespoke structure, nothing is detectable.

| | Section | Contents |
|---|---|---|
| **A** | Why this matters — and what it costs | The problem, and its dollar/latency/quality price. Cost is a first-class quantity in this subject and appears from lesson one. |
| **B** | Mental model | The picture to hold. One diagram or one worked scale. |
| **C** | Mechanism | The actual math or protocol. Attention, BPE merges, RRF, ZeRO partitioning, bootstrap CI — written out, not gestured at. |
| **D** | **From data science to LLM systems** *(bridge — required)* | Explicit mapping from what the reader already knows. "This is your train/test contamination problem, except the test set is on the public internet and inside the model." Every lesson names the analogue and, crucially, **where the analogy breaks**. |
| **E** | Minimal implementation | From scratch, pure Python/NumPy, runnable in the browser. |
| **F** | Production practice | How the real stack does it, with **pinned versions and a date**. Framework names live here and nowhere else. |
| **G** | Experiment | A seeded run whose numbers appear in the prose. Every number in a lesson comes from code that produced it. |
| **H** | **Failure modes and cost traps** *(required)* | What breaks, what it looks like when it breaks, what it costs. Includes the mistakes made while authoring the lesson. |
| **I** | **Graded practice** *(required)* | Tier-1 checks, the tier-2 bank, the tier-3 exercises. This section is the assessment contract for the lesson. |
| **J** | Annotated references | Dated. Papers, docs, source. Each with one line on why it's worth reading. |
| **K** | Extension | Optional off-platform work — the parts a browser cannot do (§5). Always labeled with its real cost. |

Front matter carries: `status` (Draft → Reviewed → Verified → Reproducible),
`last_verified` (date), `volatility` (low | high), `pyodide` (true | false),
`prereqs` (lesson ids). The staleness gate (§7) reads these.

---

## 3. Assessment ladder — four tiers

### Tier 1 — Inline self-checks
Short questions inside the prose, answer behind a click (`pymdownx.details`).
Zero infrastructure. **Target: 4–8 per lesson.**

### Tier 2 — Graded quizzes
`<quiz-bank>` web component. Instant grading, per-option explanations including
why each distractor is wrong, retry, shuffle. Banks authored as YAML in
`curriculum/<module>/questions/*.yaml`. Types: `single`, `multi`, `numeric`
(with tolerance — this subject is full of arithmetic worth drilling).
Progress in `localStorage`, JSON export/import. **Target: 8–12 per lesson, plus
a 25–40 question module exam.** ≈ 1,300 questions at full scale.

### Tier 3 — In-browser coding exercises
`<code-exercise>`: CodeMirror + **Pyodide** in a Web Worker. Spec in YAML:
`setup_code` (hidden; all provided helpers go here or solutions raise
`NameError`), `starter_code`, `tests` (assert-style, run client-side),
`hints[]`, `solution`. **Target: 2–3 per lesson (~200 exercises).**

This tier is where the real design decision lives — see §5.

### Tier 4 — Autograded artifacts
Local-first, no backend. Each is a folder with `README.md`, a `student.py` with
a documented spec, and `grader/` with a `--reference` mode that CI runs.
`python -m grader` prints a scored rubric breakdown.

- **One mini-project per module** (16).
- **One capstone per course** (4) — see §6.

**Anti-memorization without hidden tests:** inputs are randomized per run but
seeded; the **rubric is published, the expected output is not fixed**. Hidden
tests stay out of scope until there is a cohort to protect.

---

## 4. Content inventory

Course numbers and module numbers are **fixed now and never renumbered**. Build
order (§8) is a separate thing.

### Course I — Working with LLMs (3 cr ≈ 135 hrs)

| Module | Lessons | Hrs |
|---|---:|---:|
| **0 · From data science to LLM systems** — what changes when the model is stochastic and someone else's; the anatomy of an LLM application; why your offline-eval habits partly break; cost and latency as first-class metrics; roles field guide; diagnostic lab | 5 | 10 |
| **1 · Tokens, sampling and the API contract** — BPE from scratch; tokenizer pathologies (numbers, code, non-English, the leading space); context windows and truncation policy; temperature / top-k / top-p / repetition penalty; determinism and why you still don't get it; latency arithmetic (TTFT vs TPOT) and cost arithmetic; structured output modes; rate limits, retries, idempotency | 5 | 18 |
| **2 · Prompting and structured output** — instruction structure; few-shot example *selection* as a retrieval problem; decomposition and chaining; self-consistency and its cost curve; JSON schema, validation and repair loops; constrained decoding (implement the logit mask) | 5 | 16 |
| **3 · Embeddings, retrieval and RAG** — embedding geometry; exact vs approximate search (HNSW/IVF concepts against an exact baseline); BM25; hybrid fusion and RRF; chunking as an optimization problem; reranking; groundedness and citation; retrieval metrics; long-context vs retrieval; **RAG failure gallery lab** | 8 | 28 |
| **4 · Agents and tool use** — the tool-calling protocol; the loop (ReAct-shaped) and its termination conditions; planning and decomposition; memory and state; multi-agent — when it helps and when it is just latency; sandboxing tool execution; **agent failure lab** | 6 | 18 |
| Mini-projects (5) + Course I exam bank | — | — |
| **Capstone I · Grounded assistant** | — | 45 |

### Course II — Training and adaptation (3 cr ≈ 135 hrs)

| Module | Lessons | Hrs |
|---|---:|---:|
| **5 · Transformer internals** — embeddings and positional encoding (learned, RoPE, ALiBi); attention forward pass in NumPy; multi-head, GQA/MQA; the MLP block; normalization and residuals; the full forward pass, checked against pinned reference tensors; MoE routing | 6 | 30 |
| **6 · Training dynamics and memory arithmetic** — backprop through the block; optimizers and optimizer *state*; **why a 7B model needs ~84 GB to train in fp32** and how mixed precision, gradient checkpointing and accumulation change that; LR schedules and warmup; loss curves and what each pathology looks like; numerical stability | 6 | 32 |
| **7 · Adaptation** — when *not* to fine-tune (the decision tree, with costs); SFT and data formatting; LoRA implemented and merged in NumPy, with exact parameter counts; QLoRA and what quantized base weights change; catastrophic forgetting, measured | 5 | 28 |
| **8 · Preference optimization and scaling laws** — reward models; RLHF's moving parts; DPO and why it's the default now; evaluation of alignment; Chinchilla compute-optimal arithmetic; inference-compute tradeoffs | 4 | 20 |
| Mini-projects (4) | — | — |
| **Capstone II · Train it, then adapt it** | — | 25 |

### Course III — AI infrastructure (3 cr ≈ 135 hrs)

| Module | Lessons | Hrs |
|---|---:|---:|
| **9 · Data pipelines and curation** — corpus construction; near-duplicate detection with MinHash/LSH; **decontamination against a held-out eval set**; quality filters and their biases; PII handling; provenance, licensing and dataset lineage; mixture weights | 6 | 30 |
| **10 · Distributed training as a cost model** — data / tensor / pipeline / sequence parallelism; ZeRO stages 1–3 and FSDP; memory per device and communication volume per step, derived; pipeline bubbles (GPipe vs 1F1B); collective-communication cost; what actually fails at scale | 5 | 28 |
| **11 · Inference serving** — prefill vs decode; KV cache size arithmetic; static vs continuous batching; paged attention; prefill/decode disaggregation; speculative decoding and acceptance rates; **the throughput / p95-latency / cost surface** | 6 | 34 |
| **12 · Quantization and compression** — int8/int4, group-wise scales, outliers; PTQ vs QAT; measured error vs measured savings; distillation; pruning; what each buys on the latency budget | 4 | 22 |
| Mini-projects (4) | — | — |
| **Capstone III · Serving simulator** | — | 21 |

### Course IV — Evaluation and production (3 cr ≈ 135 hrs)

| Module | Lessons | Hrs |
|---|---:|---:|
| **13 · Eval design** — what "accuracy" means when output is text; task-grounded metrics; building a harness; **statistical rigor at small n** (bootstrap CIs, paired tests, minimum detectable effect, multiple comparisons); benchmark contamination and how to detect it; annotation and inter-rater agreement | 6 | 32 |
| **14 · Judges and regression gates** — LLM-as-judge; its documented failure modes (position bias, length bias, self-preference, sycophancy), each reproduced and measured; calibrating a judge against human labels; regression gates in CI; canary sets and drift detection | 5 | 30 |
| **15 · Production** — rollout and rollback; observability, tracing and what to log; caching (prompt, semantic, KV) and its correctness hazards; unit economics and the cost model; abuse, prompt injection and the tool-execution trust boundary; incident forensics | 5 | 28 |
| Mini-projects (3) | — | — |
| **Capstone IV · The eval harness that catches a planted regression** | — | 45 |

**Program totals (targets):** 16 modules · **87 lessons** · ~1,300 quiz questions
(870 lesson + 16 module exams) · ~200 in-browser exercises · 16 mini-projects ·
4 capstones · **540 learner-hours** (4 × 135, capstones included) ⇒ **12 credits
defensible; Course I alone ⇒ 3.**

---

## 5. The executable core

> This is the crux of the whole design. Everything else is bookkeeping.

**Constraint:** a static site, no backend, no accounts, no API key, no GPU, no
network at grade time. Pyodide gives CPython + NumPy + SciPy + scikit-learn +
pandas in a Web Worker. It does **not** give PyTorch, CUDA, `tiktoken` (Rust),
or outbound HTTP to a model provider.

Everything below is designed against that constraint.

### 5a. Three fixture types make the whole thing work

**1. Pinned real embeddings.** Embeddings for the shipped corpora are computed
**offline, once, with a real named model on a recorded date**, quantized to
int8 with stored scales, and shipped as `.npy`. Retrieval exercises therefore
run against *real* embedding geometry with no model in the browser. Recall@k
and nDCG numbers are real numbers, not toy ones.

**2. Cassettes — recorded real model responses.** Request/response pairs
captured from real APIs against pinned model versions on a recorded date, keyed
by a hash of (model, params, normalized prompt), stored as JSON. `CassetteClient`
replays them. Exercises that need genuine model behavior get genuine model
behavior, deterministically, offline, in CI, forever. A cassette miss is a hard
error, never a live call.

**3. `MockClient` — a synthetic model with knobs.** A deterministic fake whose
*mechanisms* are real and inspectable: a real BPE tokenizer, real
temperature/top-p sampling over a logit table, a real context limit that
truncates, real tool-call emission, configurable latency and price, and
switchable pathologies (position bias, length bias, JSON-invalidity rate,
refusal rate, hallucination-under-missing-evidence). It exists to make failure
modes *reproducible on demand* — the same reason the robotics curriculum needed
a simulator.

All three ship inside `llmlab`, a pure-Python wheel installed into Pyodide with
`micropip` and importable in plain CPython so the same tests run in CI.

### 5b. What a machine can genuinely check

| Area | What is graded, concretely |
|---|---|
| Tokenization | Learner implements BPE train/encode/decode; checked against a pinned vocabulary and round-trip invariants. Token counting and cost arithmetic on real text. |
| Sampling | Implement temperature / top-k / top-p / repetition penalty; checked distributionally across seeds against a reference sampler. |
| Structured output | Implement schema validation, a repair loop, and a **grammar-constrained logit mask**. Graded on: valid-JSON rate over a seeded stream of deliberately-malformed generations, and that the mask never permits an invalid token. |
| Retrieval | Real int8 embeddings + a labeled corpus. Graded on **recall@k, nDCG@10, MRR** — real metrics, real numbers, deterministic. |
| Chunking | Graded *by downstream retrieval quality on the labeled set*, never by string equality against a reference chunker. There are many right chunkings; there is one measurable outcome. |
| RAG | Gold-evidence oracle: each query's answer requires a known span. Graded on retrieval of the gold chunk, citation correctness, groundedness (no claim without supporting retrieved text), abstention on the deliberately-unanswerable subset, and cost per query. |
| Agents | A deterministic sandboxed toolset (calculator, mock search over a fixed corpus, fake filesystem, a flaky tool with a seeded failure rate). Graded on task success across seeded episodes, tool-call count, loop detection, and recovery from tool errors. |
| Transformer internals | Attention, MHA/GQA, the block, the full forward pass — checked against pinned reference tensors within tolerance. It's NumPy. |
| Training math | Backprop on a small MLP against numerical gradients; memory arithmetic for params + grads + optimizer state + activations; gradient accumulation equivalence; LR schedule shapes. |
| LoRA | Implement forward and merge; exact parameter-count checks; post-merge numerical equivalence to the reference. |
| Distributed strategies | Memory-per-device and comms-volume-per-step for DP/TP/PP/ZeRO-1/2/3, and pipeline bubble fraction for GPipe vs 1F1B. Pure arithmetic with an exact reference — and exactly what the interview asks. |
| Serving | A queueing simulator: seeded arrival process, prefill/decode cost model, KV-cache capacity, batching policy. The learner implements the **policy**; scored on throughput vs p95 TTFT/TPOT against a published envelope. |
| Quantization | Implement int8/int4 quant–dequant with group-wise scales; measured reconstruction error and memory saving against reference bounds. |
| Data pipelines | MinHash/LSH near-dup detection scored on a labeled duplicate set (precision/recall); decontamination scored on planted contamination; filter-bias measurement. |
| Evaluation | Harness construction; bootstrap CIs; paired tests; minimum detectable effect at given n; contamination detection; judge-bias measurement against `MockClient`'s switchable biases; regression gates scored on **detection rate against planted regressions and false-alarm rate on null runs**. |
| Unit economics | Cost/latency models for a described workload, checked against a reference calculation. |

### 5c. What CANNOT be machine-checked — stated plainly

**No exercise anywhere in this curriculum scores the learner's prompt text by
feeding it to `MockClient` and checking the output.** That would grade the
learner's ability to reverse-engineer a fake I wrote. It is the most tempting
shortcut in this subject and it is banned. Enforced by CI gate 12 (§7).

Genuinely out of reach, with the honest alternative for each:

| Not checkable | Why | What the learner does instead |
|---|---|---|
| Whether a prompt is *good* on a current frontier model | Behavior is model-, version- and date-specific; a mock can't stand in | **Off-platform lab, bring your own key.** Published task, labeled set, and my own recorded run against pinned models on a stated date. The learner runs the same script and compares. Cost: cents to a few dollars. Never in CI. |
| Open-ended generation quality | No client-side oracle; a judge model needs a model | Constrained proxies are graded on-platform (extraction accuracy on labeled data, schema validity, abstention). Quality judgment is explicitly named as a human skill, with a calibration exercise: score 20 outputs, then compare against published labels and inter-rater agreement. |
| Whether a fine-tune actually improves a real model | Needs a GPU | **Optional rented-GPU lab.** A nanoGPT-scale run ≈ 1 GPU-hour; a LoRA fine-tune ≈ $5–20. Published expected loss curves with tolerances so the learner can tell success from silence. Course II's capstone is designed to be complete *without* it. |
| Real multi-node distributed training | Nobody has 64 GPUs at home | The cost model is the graded artifact — and is the transferable skill. Stated as such, not as a substitute pretending to be the real thing. |
| Real serving throughput on real hardware | Needs GPUs and vLLM/TRT-LLM | The simulator is graded; a published measured-on-real-hardware reference table is provided for calibration, dated, with the gap between simulator and reality discussed rather than hidden. |
| Production judgment — incidents, on-call, stakeholder tradeoffs | Not a code problem | Written case studies with published post-hoc analyses. Ungraded, and labeled ungraded. |

A curriculum that names its own limits is more trustworthy than one that
substitutes a quiz for a skill.

---

## 6. Capstones — one per course

Each is a real artifact, run locally with `python -m grader` (or `-m eval`),
scored against a **published rubric** on **randomized-but-seeded** inputs. No
fixed expected output.

**Capstone I · Grounded assistant.** Build a retrieval + agent system over a
shipped corpus. Scored across N seeded query sets: answer correctness against
gold evidence (30), citation correctness and groundedness (20), abstention on
the unanswerable subset (15), cost per query under budget (15), p95 latency
under budget (10), code quality gates (10).

**Capstone II · Train it, then adapt it.** Train a small transformer from
scratch at CPU-feasible scale, then adapt it, with measured before/after on a
held-out task. Scored on: reaching a target held-out loss (25), a correct and
*verified* memory/compute budget report (25), adaptation delta over base (25),
ablation quality — the learner must show which choices mattered (15), code
quality (10). The optional GPU extension is scored separately and never gates
the capstone.

**Capstone III · Serving simulator.** Implement admission, batching, and cache
policies for a simulated engine. Scored on hitting a published throughput /
p95-TTFT / p95-TPOT / cost-per-1k-tokens envelope across seeded traffic
profiles, including a burst profile and a long-context profile designed to break
naive policies.

**Capstone IV · The harness that catches the regression.** Build an eval harness
and a CI gate. Scored on: detection rate against a set of planted regressions of
varying effect size (35), **false-alarm rate on null-change runs (25)** — the
half that most eval work gets wrong, contamination detection (15), statistical
calibration of the reported CIs (15), report quality against a rubric (10).

---

## 7. Content-integrity CI

One command runs every gate, locally, unpiped:

```bash
python tools/verify.py
```

**Never pipe a gate into `tail`, `head`, or `grep` when its exit status is what
you rely on.** A pipeline's exit status is the *last* command's. In the robotics
repo this exact mistake masked seven failing exercises across five commits while
CI was red. `tools/check_one.py <exercise-id>` is the way to iterate on one item.

| # | Gate | Catches |
|---:|---|---|
| 1 | `ruff check .` | style/lint |
| 2 | `pytest -q` on `llmlab/` | library regressions |
| 3 | Every exercise's **reference solution runs and PASSES** its own tests | broken exercises |
| 4 | Every exercise's **STARTER FAILS** its own tests | **an exercise that asks for nothing.** Highest-value gate in the set; nothing else catches it, and eye review does not |
| 5 | Every quiz bank and exercise spec **schema-validates** | malformed content |
| 6 | `mkdocs build --strict` | broken internal links, bad refs |
| 7 | **Nav covers every page; no orphans** | pages nobody can reach |
| 8 | Every mini-project grader runs against its reference and scores full marks | broken graders |
| 9 | Every capstone runs against its reference on ≥2 seeds | rubric drift |
| 10 | **Seed sweep** — graders re-run across 20–30 seeds; thresholds must hold | thresholds tuned to seed 1 |
| 11 | **Determinism** — same seed twice ⇒ byte-identical score | hidden nondeterminism |
| 12 | **No prompt-graded-by-mock** — no exercise's tests may assert on `MockClient` output derived from learner-authored prompt text | the banned shortcut (§5c) |
| 13 | **No network** — full suite runs with outbound network disabled; any live-API call is a failure | accidental live dependency |
| 14 | **Cassette integrity** — every cassette records model id + date + request hash; no orphans; no unused entries; no secrets in fixtures | undated or fabricated "real" outputs |
| 15 | **Volatility containment** — no model name, price, or context-limit literal appears outside `docs/living/` | decay scattered through 87 lessons |
| 16 | **Staleness** — every lesson has `status` and `last_verified`; `volatility: high` lessons fail past a threshold age | silent rot |
| 17 | **Pyodide compatibility** — exercise code imports only from an allowlist (no `torch`, no `requests`) | exercises that can't run in the browser |
| 18 | **Computed numbers** — every number tagged `<!-- computed: <script> -->` has a runnable producer whose output still matches | a number that was once true |
| 19 | `tools/audit.py` | every check ever done by hand, so it is never done by hand twice |
| 20 | **Provided-object contracts** — every non-constant object `setup_code` hands the learner carries a docstring or a `provided:` summary | a name and a signature that say nothing about what the arguments mean, in code the learner cannot open |
| 21 | **Lesson depth and prose style** — ≥2,500 words, mean sentence ≥22, ≤18% of sentences under 11 words | a "textbook-scale" claim that the text does not support, and notes-style prose that reads as an essay rather than a chapter |
| 22 | **Figures match their code** — every committed SVG is byte-identical to a fresh render by `tools/figures.py` | a figure that has drifted from the experiment it illustrates — the pictorial version of the stale number gate 18 catches |
| 23 | **The starter's first failure carries a message** — the first assertion a failing starter trips must explain itself | `an assertion failed`, which is the only thing a learner sees when the teaching surface of the exercise is a bare `assert` |
| 24 | **Every exercise field compiles** — setup, starter, solution and tests must all parse | a `SyntaxError` in author-written scaffolding, which shows the learner an error in code they did not write about something the exercise is not teaching |

Gates 12–18 are specific to this subject; 1–11, 19–20 and 22 are inherited; 21 and 23 are ours from the sibling robotics curriculum.

---

## 8. Phased roadmap

Reader order is I → II → III → IV. **Build order is not the same thing**, and
because course/module numbers are fixed (§4), changing build order costs
nothing.

| Phase | Contents | Exit criterion |
|---|---|---|
| **P0 · Platform + exemplar** ✅ **COMPLETE 2026-08-09** | `quiz-bank` and `code-exercise` components, YAML schemas, Pyodide worker, `llmlab` v0 (**tokenizer only** — see §11 amendment 1), all 19 gates wired. Lesson 1.1 *BPE from scratch* with all four tiers | A stranger reads 1.1, answers the quiz, writes and runs code in the page, and locally grades a mini-project. **Verified in a browser, by looking at it** — met; see §11 |
| **P1 · Course I** *(in progress)* | Modules 0–4, 5 mini-projects, exam bank, Capstone I, corpus + embedding + cassette fixtures. **Module 0 complete 2026-08-09** | Course I ≈ 135 learner-hrs, all gates green. **Shippable and complete on its own** |
| **P2 · Course IV** | Modules 13–15, 3 mini-projects, Capstone IV | Evaluation and production complete. Built second, not last: it is the least crowded skill area, it builds on Course I rather than on II–III, and a learner who stops here has the most employable pair of courses |
| **P3 · Course II** | Modules 5–8, 4 mini-projects, Capstone II, optional GPU lab | Training and adaptation complete; 9 credits defensible. Precedes Course III because Module 11 (KV cache, batching) is unteachable without Module 5's attention |
| **P4 · Course III** | Modules 9–12, 4 mini-projects, Capstone III | 12-credit program complete |
| **Continuous** | Question-bank growth, `docs/living/` re-audit per model release, cassette re-recording with dated diffs | — |

**Re-recording cassettes is content, not maintenance.** When a re-record against
a newer model changes an output, the diff goes into the lesson. Watching a
pinned prompt's behavior shift across model versions is one of the few ways to
*show* rather than assert that this field decays.

---

## 9. Explicit non-goals

Deliberately not building:

- **Accounts, backends, databases, an LMS, certificates, forums, monetization.**
  Progress is `localStorage` with JSON export/import. Nothing leaves the browser,
  so there is nothing to disclose and no consent banner to write. (`localStorage`,
  never cookies — a cookie is transmitted on every request, which is pointless
  against a static site and is precisely what makes banners necessary.)
- **Any live API dependency at grade time.** Cassettes or nothing.
- **Multi-node cluster work, or anything requiring more than one GPU.** Cost
  models instead, named as such.
- **CUDA kernel authoring.** The valuable, transferable thing for this audience
  is the performance model — memory bandwidth, arithmetic intensity, batch-1
  decode — not kernel syntax.
- **A prompt cookbook.** Prompt patterns date in months. Mechanisms don't.
- **Agent-framework tutorials.** LangChain / LlamaIndex / DSPy and friends are
  covered at awareness level in section F only. The learner builds the loop.
- **RLHF at scale.** DPO-class methods and the concepts, not a training run.
- **A model-release news feed.** `docs/living/frontier.md` is re-audited at
  phase boundaries, not maintained as journalism.
- **Safety or policy advocacy.** Prompt injection and the tool-execution trust
  boundary are covered as engineering (Module 15). Positions are not.
- **Video on the critical path.** Optional recorded demos at phase boundaries;
  never a dependency.
- **Promotion.** No channel, no launch post.

---

## 10. Open questions

Recorded so they are decided deliberately rather than by accident:

1. ~~**Corpus choice for retrieval/RAG.**~~ **Resolved 2026-08-09 — see §10a.**
2. **Which models get cassetted.** Recording against 2–3 models across two size
   tiers makes cross-model comparison teachable, and multiplies recording cost.
   Recommendation: 2 models for P1, expand only if a lesson needs it.
3. **Repo public or private, and when.** Robotics went public with a soft-launch
   `noindex` guard toggled by a script. Same approach recommended, decided at P1.
4. **Does the GPU lab get published expected-loss references?** It should, but
   they cost real GPU money to produce and must be re-verified as drivers and
   library versions move. Deferred to P3.

## 10a. Resolved — the retrieval corpus (2026-08-09)

**Decision: an authored synthetic corpus, with 200 labelled queries.**

### What decided it

Two constraints, and corpus *size* is not one of them. At 384-dimension int8
embeddings, 20 MB holds ~9,500 chunks of 400 tokens — 3.8M tokens, far more
than the material needs.

**The relevance judgments are the expensive part.** Metric precision depends on
the number of labelled queries, not the size of the corpus:

| Labelled queries | Wilson 95% half-width on recall@k |
|---|---|
| 50 | ±12.3 points |
| 200 | ±6.3 points |
| 500 | ±4.0 points |

Whatever the text, someone authors ~200 judgments. Authoring them *with* the
corpus — query, answer and gold span produced together — is far cheaper than
retrofitting them onto found text, and that is the decisive practical
argument.

**Memorization is fatal for the capstone and irrelevant for Module 3.**
Retrieval metrics involve no model, so a famous corpus is fine there. Capstone
I scores groundedness and abstention: if the model can answer from memory you
cannot tell whether retrieval worked, and the unanswerable subset stops being
unanswerable. That rules out Wikipedia and the public IR benchmarks for
Course I's capstone, whatever their licences say.

### The shape

- **Domain:** the fictional logistics/support company already present in the
  Module 0 fixtures (shipment `TL-4471`, the north-east depot, holds and
  verification). Continuity is free and the world already half exists.
- **Size:** 6,000–9,000 chunks of ~400 tokens. Comfortably inside 20 MB with
  embeddings; sized by what the phenomena need, not by the budget.
- **Queries:** **200**, each with a gold span, plus a deliberately
  unanswerable subset for abstention scoring.
- **Planted phenomena, each to be verified rather than asserted:** vocabulary
  mismatch that dense retrieval resolves and BM25 does not; lexical-overlap
  distractors that fool BM25; superseded document versions with dates;
  near-duplicates; multi-hop questions requiring two chunks; contradictions
  between documents.
- **Realistic mess is a requirement, not a nicety:** inconsistent terminology,
  dated supersessions, tables, and formatting drift. A corpus that is too
  clean teaches a retrieval problem nobody has.
- **Embeddings:** a 384-dimension permissively licensed model
  (`bge-small-en-v1.5` or `all-MiniLM-L6-v2` — licence to be confirmed at
  recording time), computed once by a hand-run recorder, int8-quantized with
  stored scales, shipped as `.npy`. Same contract as the tiktoken fixture:
  dated, version-pinned, never called at test time.

### The honesty cost, and how it is paid

A corpus built to contain a phenomenon can overstate it. Three mitigations,
all of which belong in the lessons rather than in a footnote: report effect
sizes plainly; state in §G of each retrieval lesson that the corpus was
constructed to contain the effect; and cite published BEIR-style numbers as an
external calibration point **without shipping those datasets**, so the reader
can see whether our effect sizes are in a plausible range.

This is the same discipline as lesson 1.2's synthetic logit distribution: the
generative choice *is* the claim, so it is stated rather than buried.

---

---

## 11. Build log — P0 (2026-08-09)

### What shipped

| | |
|---|---|
| Lesson | 1.1 · Tokens are not words: BPE from scratch (11 schema sections, 5 inline self-checks) |
| Quiz bank | 10 questions — 8 multiple-choice, 2 numeric with tolerance; every option carries an explanation |
| In-browser exercises | 3 (`tok-l1-pairs`, `tok-l1-merge`, `tok-l1-encode`) |
| Autograded artifact | `projects/tokenizer_mini` — the context packer, published rubric, seeded scenarios |
| Library | `llmlab.tokenizer` — byte-level BPE, 29 tests |
| Experiments | `bpe_compression.py` (quoted by the lesson, re-run by gate 18), `record_tiktoken.py` (by hand only) |
| Fixtures | `data/corpus.txt`, six held-out samples, `data/fixtures/tiktoken_counts.json` (recorded 2026-08-09, `tiktoken` 0.13.0) |
| Gates | 19 numbered gates, run as 11 commands by `tools/verify.py` |

**Browser verification, done by looking:** all seven pages return 200 with
correct headings; Pyodide loads and runs the exercises; the `tok-l1-encode`
starter fails in-page with its intended diagnostic and the reference solution
passes; quiz grading marks the chosen distractor *and* the correct option and
renders both explanations; the numeric question grades within tolerance;
progress writes to `localStorage` under `llmds.*`; the computed-number markers
are present in the HTML and invisible in the rendered text.

**Gates verified adversarially, not just observed green.** Eight deliberate
breakages were introduced one at a time and every one was caught: a starter
that passes its own tests (4), a lesson number that no longer matches the code
(18), a model name outside the living doc (15), a stale volatile lesson (16), a
missing schema section (19), an exercise importing `torch` (17), an exercise
grading prompt text against a mock (12), a page missing from the nav (7), and a
quiz option with no explanation (5).

### Amendment 1 — `llmlab` v0 is the tokenizer only

`MockClient` and `CassetteClient` move to **P1**, alongside the first lesson
that needs them.

A cassette is a *recording*: it carries a model id and a date, and gate 14
enforces that. Building the replay machinery now would mean either shipping no
cassettes (dead code, and a gate passing vacuously) or shipping synthetic
fixtures dressed as recordings — which is precisely the dishonesty §5c exists
to prevent. The *pattern* is proven instead by
`experiments/record_tiktoken.py`: a hand-run recorder, a dated version-pinned
fixture, and everything downstream reading the fixture rather than the
network. That is exactly the cassette contract, exercised end to end, on data
that really was recorded.

### Amendment 2 — gate 09 has nothing to run, and says so

No capstones exist until P1. `verify.py` prints that explicitly rather than
reporting a pass, because a gate with nothing to check looks identical to a
gate that is working.

### Two wrong assumptions that became content

Both are in the material now rather than quietly fixed, per CONTRIBUTING.md.

1. **Tokens per character is the wrong unit for comparing serialization
   formats.** The first version of `bpe_compression.py` compared the JSON
   sample against the numeric-table sample and found JSON 27% cheaper. Both
   numbers were right and the conclusion was meaningless — the two files carry
   different information. Measured on the *same* seven records, JSON costs
   **2.66×** what a tab-separated table does. Now lesson 1.1 §H, with the
   mistake shown.
2. **The context-packer spec had a hole, and the rubric found it.** The
   reference implementation scored 96.7/100 against its own rubric, failing the
   one case the spec never defined: what to do when the system prompt alone
   exceeds the budget. Silently trimming it is the worst answer available.
   Refusing is now rule 5, and the story is in the project README.

### Carried forward to P1

- The `data/samples/` corpus is deliberately tiny (6.7 KB training corpus).
  Module 3's retrieval work needs the real shipped corpus decided in §10 Q1.
- `llmlab` is not yet packaged as a wheel for `micropip`; P0 exercises are
  self-contained, with helpers in `setup_code`. That holds until an exercise
  needs more code than a `setup_code` block should carry.
- Two 404s in the dev server are `/favicon.ico`. Cosmetic; fix when the site
  gets a favicon.

---

## 12. Build log — P1, Module 0 (2026-08-09)

### What shipped

Four lessons (0.1 what changes · 0.2 anatomy · 0.3 evaluation · 0.4 cost and
latency), 8 in-browser exercises, 4 question banks (32 questions), the
`reliability_report` mini-project, and a roles appendix. Two new experiments:
`eval_power.py` and `service_economics.py`. Verified in a browser — MathJax
renders, self-checks expand, the `tr-l1-retries` starter fails in-page with its
diagnostic and the reference solution passes.

**Lesson 0.5 (the diagnostic lab) was deferred and then built** — see §12b.

### Three findings that came out of running things

1. **Retry limits are not a cost lever.** Attempts per success is
   `((1-f^K)/(1-f)) / (1-f^K) = 1/(1-f)`, and `K` cancels. Raising the retry
   limit buys success rate and tail latency and changes cost per success by
   nothing. This was found by doing the algebra for an exercise whose planned
   "bug" turned out to be the correct answer. Now lesson 0.1 §G and its
   exercise.
2. **The paired-vs-unpaired gap is much larger than expected, and the unpaired
   test is miscalibrated rather than merely weak.** On correlated variants
   (rho = 0.9) at n = 200, the unpaired test detects a real five-point
   improvement 9.0% of the time against the paired test's 80.5%, and its
   false-alarm rate under the null collapses to 0.0% instead of 5%. A test that
   never rejects looks exactly like a system that never improves. Lesson 0.3 §G.
3. **Two modelling errors in a row, both producing plausible tables.** The
   power simulation first clipped success probabilities to [0, 1], which
   collapsed the effect size near the ceiling; the fix revealed a second error,
   drawing the two systems' outcomes independently given the item, which
   removed the correlation pairing exists to exploit. Only after adding an
   explicit agreement parameter did the numbers hold up. Kept in lesson 0.3 §H
   as the module's own worked example of "if a simulation is the evidence, the
   generative model is the claim".

### A defect only the browser could show

The `<!-- computed: … -->` markers were written *between* a number and its
unit, so `25.0 <!-- … -->%` rendered as "25.0 %" with a stray space. Every gate
passed; the page looked wrong. 38 markers were moved after their units across
5 lessons. Gate 18 still matches (66 numbers checked across 3 experiments).

This is the third time in this repository that browser verification has found
something no gate could. It stays a required step, not a nicety.

### Gate additions

`tools/audit.py` now checks the tier-1 self-check count per lesson (4–8, per
§3). It was added after a hand count found lesson 1.1 shipping with two, and
it immediately caught all four new lessons under-supplied.

## 12b. Build log — lesson 0.5, closing Module 0 (2026-08-09)

The diagnostic lab: a system that worked in the demo and fails in production,
with the log to diagnose it from. Two exercises (`tr-l5-slice` builds the
group-by; `tr-l5-diagnose` writes the runbook as code against four planted
root causes across seeded logs), an 8-question bank, and a fourth experiment,
`aggregate_masking.py`.

**Module 0 is now complete**: 5 lessons, a graded mini-project, and the roles
appendix.

### What the experiment established

1. **Masking is total below a computable share.** With 95% healthy traffic and
   a 90% floor, a slice smaller than
   **5.26%** of traffic can fail *every* request without the aggregate ever
   breaching the floor. There is no threshold on the headline number that
   catches it — this is arithmetic, not tuning.
2. **Detection, once you slice, is nearly free for outright breaks.** A 10%
   slice failing at 45% is caught 99.5% of the time on 20 slice items. So
   incidents that run for weeks are not a sample-size problem; **nobody ran
   the group-by.** That inverts the obvious remedy, and it is the conclusion
   the lesson leads with.
3. **Mild degradation is a genuine power problem.** A slice at 80% against 95%
   healthy is caught 41.3% of the time at n=200 and 80.1% at n=500 — lesson
   0.3's arithmetic, unchanged.

Point 2 was not the expected finding. The lab was drafted around "detection is
hard"; the simulation said otherwise, so the lesson's argument changed to match
it rather than the reverse. Keeping all three conclusions separate is what
makes §G worth running.

### A tuning pass that changed the story

The narrative incident was first drawn with an 8% slice, which put the
aggregate at 89.7% — just *below* a 90% floor, so the dashboard would have
alerted and the example would have argued against itself. Redrawn at a 6%
slice: 91.2% overall, a version failing 59% of the time, a 53-point gap, and a
green dashboard. The point of the example is that nothing fires, so the
parameters have to produce an example where nothing fires.

---

## 13. Build log — Module 1, lessons 1.2–1.3 (2026-08-09)

**1.2 · Sampling** — temperature, top-k, top-p, repetition penalty, and what
"deterministic" actually buys. Three exercises (`softmax` with the overflow and
`T=0` cases, `top_p_filter` including the token that crosses the threshold,
sign-aware `repetition_penalty`), an 8-question bank, and
`experiments/sampling_shape.py`.

**1.3 · Context windows and chat templates** — the chat template as the real
input contract, the stateless resend, and truncation policy. Two exercises
(`prompt_tokens`/`fits`, and a `trim_history` with stated invariants), an
8-question bank, and `experiments/chat_overhead.py` reading a newly extended
recorded fixture.

### Amendment — Module 1 is 5 lessons, not 6

Original plan: 1.4 latency arithmetic · 1.5 cost arithmetic · 1.6 rate limits.
That was written before Module 0 existed. Lesson **0.4 now covers cost and
latency as measurement discipline** — price ratios, percentiles, why the mean
is the wrong summary — and Module 11 owns serving internals, so two of the
three planned lessons would have been restatement.

What genuinely remains, restructured:

- **1.4 · The API contract** — streaming, stop conditions and finish reasons,
  structured-output modes, rate limits, retries and idempotency. (Merges the
  old 1.4 and 1.6.)
- **1.5 · Costing a design before you build it** — forward modelling of a
  *proposed* system (RAG versus long-context versus multi-call), which is a
  different activity from measuring an existing one.

Program totals move from 88 lessons to **87**. Module numbers are unchanged, so
nothing links or ids had to move — the reason §1 fixes them at the outset.

### What the two experiments established

1. **`top_p` is a policy, not a budget.** The same `top_p=0.90` admits 6
   candidates on a near-determined distribution and 31,394 on a wide-open one.
   Reported across three distribution shapes rather than one, because the
   variation *is* the finding.
2. **Filter order is not a detail.** Temperature-then-top-p leaves a nucleus of
   15 tokens; top-p-then-temperature leaves 2,171, with a total-variation
   distance of ~0.095 between the resulting distributions. Implementations
   differ, so identical parameters do not mean identical behaviour.
3. **The repetition-penalty sign bug, quantified.** Dividing a logit of −9.07
   by 1.2 makes the penalised token **4.53× more likely**. Most logits are
   negative, so this is the normal case, not an edge case.
4. **Low temperature is not determinism.** The argmax is drawn 92.7% of the
   time at `T=0.3` — per *token*. Over 200 tokens that is ~3 × 10⁻⁷ agreement
   with greedy. Per-token agreement is a badly misleading summary.
5. **Conversation cost is quadratic.** Measured on an 8-message conversation:
   38.0% of the final prompt is chat-template wrapper, and the four requests
   bill 2.54× the largest single prompt. Projected to 20 turns (linear fit,
   worst error 3.2%): 15,420 tokens billed against a 1,455-token final prompt
   — **8.9×**. Nobody sizes a conversational feature from the last request, and
   the last request is what the dashboard shows.

### Lessons 1.4–1.5, closing Module 1

**1.4 · The API contract** — streaming, finish reasons, the failure taxonomy,
rate limits as token buckets, idempotency. Two exercises (full-jitter backoff
honouring `Retry-After`; classifying a response into
accept/truncated/refused/retry/fail), and `experiments/rate_limits.py`.

**1.5 · Costing a design before you build it** — long context vs retrieval vs
map-reduce, costed from parameters known in advance. Two exercises (the cost
model; feasibility-then-cost selection), and `experiments/design_costs.py`.

**Module 1 is complete: 5 lessons, 11 exercises, 5 banks, 1 mini-project.**

### Two experiments that contradicted the advice I was about to write

1. **"Add jitter, it reduces retries" is false here.** Plain exponential
   backoff produced *fewer* attempts per success (2.44 vs 2.94) than full
   jitter. It achieves that by sleeping: it finishes in 2.31× the minimum
   possible time with clients idle while capacity goes unused. Full jitter
   finishes 1.7× faster, at 1.36× the floor. **Minimising retries is
   minimising a cost, not achieving an outcome** — a policy can reach zero
   retries by declining to work. The lesson leads with that reframing rather
   than the received advice. Retrying immediately also finishes at the floor
   while putting 74.4× more load on the service, which is the amplification
   failure from 0.2.
2. **Map-reduce is dominated on both axes**, not just cost. It costs 2.1× long
   context at N=1000 because the system prompt is re-sent once per chunk (the
   `N·s` term), and at 8 concurrent calls it is also *slower* — 115.7s against
   20.4s. Its real advantages are narrower than its reputation: no window
   ceiling, and a concurrency knob. Matching long context's latency at N=1000
   needs 59 concurrent calls, which is a rate-limit question, not a design one.

Two simulation artefacts were found and fixed before either result was
trusted: the rate-limit model served clients in index order, systematically
starving high-index clients for reasons unrelated to any policy; and its
"floor" ignored the bucket's initial burst, making the floor unreachable and
every policy look 1% better than physics allows.

### Note on the sampling experiment's honesty

There is no model here, so the next-token distribution is synthetic. The
choice of distribution *is* the claim, so it is stated in the docstring
(Zipfian, with the exponent standing in for model confidence) and every result
is reported across three exponents. A single number would have been a fact
about the exponent, not about sampling — the same discipline lesson 0.3 §H
arrived at the hard way.

---

## 14. Build log — Module 2, lesson 2.1 (2026-08-09)

**2.1 · Instruction structure and the trust boundary.** Two exercises
(escape-the-escape with a lossless assemble/parse round trip; an authorisation
check that derives authority from the request's origin rather than from what
appeared in the prompt), an 8-question bank, and
`experiments/prompt_assembly.py`.

### Module 2 is the module gate 12 exists for

Prompting is the subject where "grade the learner's prose against our mock" is
most tempting and most worthless. The module opener says so explicitly, and
every lesson is scoped to the machinery around the wording — assembly,
selection, voting arithmetic, schema conformance, logit masking — all of which
has correct answers that can be checked.

### The experiment separates two things that are constantly conflated

`prompt_assembly.py` counts **structural breakouts** (can a parser recover the
regions?) and **semantic payload** (does the content read as an instruction?)
separately, because policies that drive the first to zero do nothing to the
second:

| Policy | Structural breakouts |
|---|---|
| Naive code fence | 9.1% |
| Naive rare sentinel | 9.1% |
| Escaped sentinel | 0.0% |
| Structural (separate message) | 0.0% (by construction) |

27.3% of payloads carry instruction-like text, and **the same 27.3% survive
escaping unchanged**. Escaping is a complete solution to the parsing problem
and no solution at all to injection — a distinction worth a lesson because the
testable half is the one that gets measured, fixed, and declared victory over.

### A discarded first attempt

The lesson was first drafted around a *delimiter collision* experiment:
count how often each candidate delimiter occurs naturally in the shipped
documents. It was built, run, and thrown away — only 1 of 8 candidates
collided, because the seven frozen sample documents are too few and too clean
(no Markdown, so no code fences). It would have been an honest measurement of
nothing. The collision question moved to §K, where the learner runs it on a
corpus that can actually answer it.

### Also fixed

The structural policy's escape originally inserted a zero-width space inside
the marker. It works, and it is a bad habit — invisible in logs and diffs, and
undone by any component that normalises Unicode. Replaced with a visible
placeholder, and the reasoning is now §G's second self-check.

---

*Amend this file when scope changes. It is the record of what was decided and
why — including what was declined.*
