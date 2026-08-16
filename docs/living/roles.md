# The role field: where a data scientist fits in LLM/AI work

**Living document · Researched 2026-08-16.** Claims are marked **[V]** verified
against a primary source, **[C]** company or vendor claim, **[U]** single or
secondary source. This page is field research for a self-study curriculum, not
career advice, and it is not written to be persuasive about anything.

There is a companion page — [who does what](../modules/00-transition/roles.md) —
which describes the *work* rather than the market. That one ages slowly. This
one ages fast, which is why it lives here.

---

## Read this first: the evidence quality in this subject is unusually bad

Researching this page turned up more unsourced numbers than any other topic in
this curriculum. A representative sample of figures currently circulating, none
of which I could trace to a methodology:

- "AI engineering jobs are up 800%"
- "Global AI talent demand exceeds supply 3.2 : 1 — 1.6M open positions against
  518,000 qualified candidates"
- "AI engineer salaries rose $50K year over year"

Each appears on multiple sites, each is stated without a source, and several of
those sites sell recruitment services or courses. **None of them are used
below.** That is not fastidiousness for its own sake: a field where the
loudest numbers are untraceable is one where you should weight your own
observations heavily and other people's confidence lightly.

What follows uses employer-side or platform-side data where it exists, and says
so where it does not.

---

## The one-sentence summary

**The title "data scientist" is being unbundled rather than eliminated: the
statistical and evaluation half of the job is in more demand than ever and is
increasingly posted under other names, while the modelling half has moved
toward people who build systems on top of models they did not train.**

---

## 1. What the titles mean now

Titles in this field are unstable and mean different things at different
companies, so treat these as clusters of work rather than job specs.

**AI engineer.** Builds product features on top of models somebody else
trained: retrieval, agents, structured output, orchestration. LinkedIn ranked
it the **fastest-growing job title in the US for 2026, with postings up 143%
year over year**, and counted **639,000 AI-related US postings added between
2023 and 2025, of which about 75,000 were AI engineer roles** [V, LinkedIn Jobs
on the Rise 2026]. The commonly listed skills are orchestration frameworks,
RAG and PyTorch — which is to say the job is mostly systems work.

**ML engineer.** Training, fine-tuning, serving, infrastructure. Older, better
defined, and the closest thing to a stable title in the area.

**Applied scientist / research engineer.** Where the training-from-scratch work
actually sits, largely in frontier labs and large platforms. Small in absolute
numbers.

**Evaluation.** The newest of these clusters. Roles that were a bullet inside
a senior ML engineer posting are increasingly standalone postings, and frontier
labs run dedicated evaluation teams [U]. I could not find employer-side
posting counts for this specific title, so treat the trend as directional and
the magnitude as unknown.

**Data scientist.** Still hiring, and increasingly a legacy umbrella term [U].
The interesting signal is compositional rather than volumetric: NLP appears in
roughly 19% of data-scientist postings, up from about 5% in 2024 [U]. The
title is absorbing this work rather than being replaced by it.

---

## 2. Compensation, and why the average is misleading

The most useful thing to know about compensation here is that **the
distribution is bimodal, so any single average describes nobody**.

From levels.fyi, which is self-reported and skews toward large tech employers
[V as a source, with that caveat]:

| Role | Median total compensation |
|---|---|
| ML engineer | ~$261K |
| AI engineer (9,500+ profiles) | ~$211K |
| ML-focused data scientist | ~$185K |

Against that, frontier-lab software-engineer medians are reported in the
$600K–$795K range, mostly in equity [U]. Enterprise ML roles cluster far
lower. San Francisco, New York and Seattle sit well above the national median.

Two readings follow. First, the widely-quoted "AI salaries" figures are
usually the upper mode with the lower one omitted. Second, the gap between the
modes is much larger than the gap between adjacent titles, so *where* you do
this work dominates *what it is called*.

---

## 3. Which lanes are crowded, and which are not

This is the section most worth disagreeing with, because it is inference rather
than data.

**Crowded: prompt-and-wrapper application work.** The barrier is low by
construction, and the LinkedIn skill list for AI engineer — orchestration
framework, RAG, PyTorch — is exactly what every bootcamp now teaches.

**Less crowded: evaluation.** The argument is indirect but it has three legs.
Evaluation is where the [frontier page](frontier.md) locates the field's actual
bottleneck; it is the skill set a statistician already largely has; and it is
the one whose absence has a documented consequence — MIT's *GenAI Divide*
attributes stalled pilots to organisations lacking the measurement and
governance to scale trust, not to model capability [V as a report].

**Less crowded: cost and serving arithmetic.** Being able to say what a design
will cost before it is built is rare, unglamorous and immediately legible to
whoever approves budgets.

**Structurally small: training frontier models.** A real career and a narrow
door. Worth being honest that most of the material teaching it is not
preparing anyone for that job.

---

## 4. What transfers from data science, and what does not

**Transfers almost intact:** experimental design, significance and its abuse,
confidence intervals, held-out evaluation, selection effects, understanding
that a metric is a proxy. This curriculum's Module 3.4 finding — a tuned
fusion config that looks significant in-sample (p = 0.0002) and is not on
held-out queries (p = 0.1153) — is an entirely ordinary statistical result that
is repeatedly rediscovered in this field the hard way.

**Transfers with modification:** feature engineering becomes context
construction; model selection becomes provider and configuration selection;
error analysis becomes trajectory analysis.

**Does not transfer, and is the actual gap:** systems engineering. Latency
budgets, failure envelopes, idempotency, cost per request, what happens when a
dependency is slow. Module 0.2 is the shape of it, and it is the honest reason
a data scientist is not automatically an AI engineer.

**The uncomfortable one:** the modelling skill that took longest to acquire —
choosing and tuning architectures — is the least used of anything on this list
for most of these roles.

---

## 5. What this curriculum is and is not evidence of

Stated plainly because the framing matters: this is **self-study, written up
and shared with peers. It is not a resume project and is not promoted
anywhere.** Working through it is not a credential and nobody should present it
as one.

What it is good for is the thing §3 says is scarce: it produces measured,
reproducible results with stated uncertainty, including several where the
initial claim was wrong and was withdrawn rather than softened. That habit is
the transferable artefact, not the site.

---

## Re-audit checklist

1. Is "AI engineer" still the fastest-growing title, or has it stabilised the
   way "data scientist" did after 2015?
2. Has evaluation become a standard standalone title with countable postings?
3. Is the compensation distribution still bimodal, or did the frontier-lab mode
   compress?
4. Have any of the untraceable numbers in the preamble acquired a real source?
   If so, use them and say where they came from.
5. Has the systems-engineering gap in §4 narrowed, because tooling absorbed it?
