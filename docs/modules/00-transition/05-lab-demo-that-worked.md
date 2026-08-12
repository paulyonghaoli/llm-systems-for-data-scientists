---
status: Verified
last_verified: 2026-08-11
volatility: low
pyodide: true
---

# 0.5 · Lab: the demo that worked

## A · Why this matters

The demo worked. It worked on your machine, it worked in the meeting, and it
worked on the twenty examples you tried before the meeting. Then it went to
real users, the complaints started, and the dashboard says ninety-one per cent
success.

This lab is about the distance between those two sentences, which is neither
mysterious nor a modelling problem. **A production LLM failure is almost always
one stage failing for one slice of traffic, concealed inside an aggregate that
looks perfectly healthy** — and the aggregate is designed, by arithmetic rather
than by anybody's negligence, to conceal exactly that.

The four lessons before this one supplied the vocabulary: cohort keys and
categorised outcomes from 0.1, the seven stages from 0.2, intervals and power
from 0.3, and percentiles from 0.4. This lesson is the runbook that uses all of
them, and its central claim is quantitative rather than rhetorical: below a
computable share of traffic, a slice can fail *every single request* without
the headline number ever crossing a threshold.

!!! info "Terms used in this lesson"
    **Slice** — a subset of traffic sharing some property: a model version, a
    parameter set, an input-length band, a language, a tenant. The diagnostic
    unit of this lesson.

    **Masking** — the arithmetic by which a failure confined to a small slice
    moves the aggregate too little to trigger an alert, regardless of how
    completely that slice is broken.

    **Signature** — an observable pattern in a log that identifies *which*
    stage broke, as opposed to merely establishing that something did.

    **Discordance between slice and whole** — the situation where a slice's
    confidence interval does not overlap the rest of traffic's, which is the
    cheapest reliable trigger for an alert.

## B · Mental model

**An incident is a subgroup, not a system.**

When an entire system is broken you find out within minutes, because every
signal moves at once and somebody is paged. The failures that survive long
enough to become incidents worth writing up are precisely the ones the headline
number tolerates: a slice of traffic, a model version, an input shape, a
language, a single tenant. Anything that moved the aggregate decisively has
already been fixed by the time you would be reading a log.

The diagnostic question is therefore never "is it working", because the
aggregate has already answered that with a reassuring number. The question is
**"for whom is it not working"**, and the first action that follows from it is
always the same: slice the log, and compare the slices against each other
rather than against a threshold.

Comparing slices against each other rather than against a fixed number matters
more than it sounds. A threshold has to be chosen in advance, applies equally
to a slice of eight requests and a slice of eight thousand, and goes stale
whenever the system's baseline moves. A comparison between a slice and the rest
of traffic needs no threshold, adjusts automatically as the baseline drifts,
and — because both sides carry intervals — refuses to fire on a slice too small
to say anything about.

??? question "Why does the aggregate hide a subgroup failure rather than averaging it in visibly?"
    It does average it in, and that is precisely the problem. An average is a
    weighted sum, and a small weight makes even a total failure small: a slice
    that is five per cent of traffic contributes at most five points to the
    headline no matter how badly it fails. The same failure that would page
    everybody if it were global is therefore invisible when it is local, and
    the transition between those two situations is smooth rather than sudden,
    which is why nobody notices where it happened.

## C · Mechanism

**The masking arithmetic.** Write `p_h` for the success rate of healthy
traffic, `s` for the broken slice's share of traffic, and `p_s` for the slice's
own success rate. The aggregate is the weighted average:

$$
\text{aggregate} = (1-s)\,p_h + s\,p_s
$$

Set a dashboard floor `F` and invert the expression to find the worst the slice
can be while the aggregate still clears the floor:

$$
p_s = p_h - \frac{p_h - F}{s}
$$

When that expression comes out negative, no value of `p_s` — not even zero —
brings the aggregate below the floor, which means **the slice can fail every
single request and the alert never fires**. That happens whenever
`s < (p_h − F) / p_h`, and §G computes the threshold for a realistic pair of
values.

**The four signatures.** Establishing that *something* is wrong is the easy
half. Each of these patterns identifies *which* of lesson 0.2's stages broke,
which is what turns an observation into a diagnosis with an owner and a fix.

| Signature | Stage | What you see in the log |
|---|---|---|
| Version drift | the call | Two `model_version` values with very different success rates |
| Context truncation | prompt construction | A spike of records at one exact `in_tokens` value, succeeding far less often |
| Retry storm | the call | Mean `attempts` well above one, and a latency tail |
| Silent refusal | output handling | Refusals counted as errors, or not counted at all |

The truncation signature is the one people miss, and it is worth understanding
why it works. Natural text lengths are continuous: if you plot the input sizes
of real traffic you get a smooth distribution with no particular value
occurring more often than its neighbours. A hard edge at exactly 4096, with a
fifth of all traffic sitting precisely on it, cannot arise from the data — it
is a ceiling somebody imposed, and the records sitting on it are the ones that
were cut. Once you know to look for a spike at a round number, the diagnosis
takes seconds.

The retry-storm signature carries a subtlety that the exercise below encodes as
a priority rule. Elevated attempt counts are almost always *downstream* of the
real fault: something broke, and the client is faithfully retrying it. Fixing
the retry policy in that situation makes the symptom quieter while leaving the
cause in place, and lengthens the outage by removing the signal that would have
led you to it.

## D · From data science to LLM systems

You already do this, and you do it well. Subgroup analysis, slicing by segment,
checking whether an aggregate improvement holds within each stratum — this is
Simpson's paradox territory, and you have been trained to be suspicious of
exactly this shape of claim.

| Your habit | Here |
|---|---|
| Slice by demographic segment or cohort | Slice by model version, parameter set, input-length band, language |
| Check the model is not worse for a subgroup | Check the *system* is not broken for a subgroup |
| Watch for Simpson's paradox | The same paradox, with unfamiliar keys |
| Fairness auditing | The same machinery, a different motivation |

The genuinely new part is *which keys to slice by*, and they are not keys your
previous work would have suggested. Nothing in ordinary subgroup analysis
prompts you to group by "the exact number of input tokens", because in a normal
pipeline the input size is not a property that anything conditions on. Here it
is exactly where truncation hides, which makes it one of the highest-yield
slices available.

The second difference is that these keys change underneath you. A new model
version appears without any deploy on your side, so the slice you need to
examine did not exist yesterday and no dashboard was configured for it. That is
the concrete reason lesson 0.1 insisted on recording the *resolved* version on
every call: **you cannot slice by a field you did not log**, and by the time
you want the field it is too late to start recording it.

??? question "You slice by ten keys and one shows a significant difference at p < 0.05. What have you found?"
    Possibly nothing. Ten independent tests at a five per cent threshold give
    roughly a forty per cent chance of at least one false positive, so a
    single hit from a sweep is a hypothesis rather than a finding. Confirm it
    on fresh traffic or on a held-out period before acting on it. This is also
    why the four signatures in §C are worth more than a p-value: each one has
    a mechanism attached, so a match is evidence about a specific stage rather
    than evidence that some slice happened to look unusual.

## E · Minimal implementation

The whole diagnostic toolkit is one function:

```python
def success_by(records, key_fn):
    groups = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)
    out = {}
    for key, rows in groups.items():
        n = len(rows)
        ok = sum(1 for r in rows if r["outcome"] == "ok")
        out[key] = {"n": n, "rate": ok / n, "ci": wilson(ok, n)}
    return out
```

Note the interval, which is not decoration. A slice with eleven items has an
interval wide enough to contain almost any hypothesis, and reporting its rate
without one invites the opposite error to the aggregate's: chasing a subgroup
that is merely small. The interval is also what lets the comparison in §B work
without a threshold, because "these two intervals do not overlap" is a
statement that scales correctly with however much data each side happens to
have.

The other thing worth noticing is that `key_fn` is a parameter. The same nine
lines slice by version, by parameter signature, by input-length band and by
tenant, which means adding a new diagnostic view costs one lambda rather than
one dashboard.

The alert built on top of it is barely longer, and its shape is worth reading
carefully because the two guards it carries are the ones that separate a useful
alert from a noisy one:

```python
def suspicious_slices(records, key_fn, min_n=20):
    """Slices whose success interval does not overlap the rest of traffic's."""
    out = []
    for key, g in success_by(records, key_fn).items():
        if g["n"] < min_n:                      # too small to say anything
            continue
        rest = [r for r in records if key_fn(r) != key]
        if not rest:                            # one slice: nothing to compare with
            continue
        rest_ok = sum(1 for r in rest if r["outcome"] == "ok")
        rest_lo, _ = wilson(rest_ok, len(rest))
        if g["ci"][1] < rest_lo:                # slice ceiling below the rest's floor
            out.append((key, g))
    return out
```

`min_n` exists because a slice of six with a fifty per cent success rate is a
slice of six, and an alerting rule that fires on it will be switched off within
a week by whoever is carrying the pager. The comparison against *the rest of
traffic* rather than against a fixed threshold matters for a different reason:
thresholds have to be chosen, maintained, and re-chosen whenever the baseline
moves, whereas "this slice disagrees with everything else" needs no
maintenance and automatically tracks a system that is improving or degrading
as a whole.

What this deliberately does not do is decide *why*. Establishing that a slice
is broken and identifying which stage broke it are separate steps, and
conflating them produces alerts that assert a cause nobody has verified. The
signatures in §C are the second step, and the second exercise below implements
them as a function you can point at any log.

## F · Production practice

Slice by these, always, and log whatever is needed to make each one possible:
**resolved model version · parameter signature · input-length band · prompt
template version · language or locale · caller or tenant**. Each of those has
produced a real incident that the aggregate could not see.

Alert on the *slices* rather than only on the total. The cheapest useful alert
in this entire curriculum is "any slice with at least N records whose success
interval does not overlap the rest of traffic's"; it costs one group-by, it
needs no thresholds to be chosen or maintained, and it catches every signature
in §C's table.

Keep per-item outcomes rather than only aggregates. A harness that stores
totals has thrown away the ability to do any of this after the fact, and after
the fact is exactly when you will want to, because incidents are discovered
rather than scheduled.

**Run the slicing on a schedule, not only when something looks wrong.** This is
the single highest-value operational change this module recommends, and §G
explains why in numbers: an outright break in a ten-per-cent slice is detected
essentially always, even on twenty items, so the reason these incidents survive
for weeks is not that the data was insufficient but that nobody ran the
group-by. A cron job that computes §E's alert over yesterday's log and posts
the result costs an afternoon and converts a class of multi-week incidents into
next-morning ones.

**Choose the slicing keys in advance and write them down.** Slicing after an
incident means choosing which slice to examine *after* seeing the outcome,
which is how people end up confidently explaining noise; the keys above are a
reasonable starting set precisely because they were chosen before anything went
wrong. Adding a key later is fine, but it should be added to the standing list
rather than tried once during an investigation and forgotten.

**Retain long enough to see a version boundary.** Model versions change on the
provider's schedule rather than yours, and a comparison across a version
boundary is the invalid comparison from lesson 0.1. If your retention window is
shorter than the interval between provider updates, you will routinely be
unable to answer whether a change in behaviour coincided with a change in
model — which is usually the first question worth asking.

## G · Experiment

```bash
python experiments/aggregate_masking.py
```

**How much can hide.** Healthy traffic succeeding ninety-five per cent of the
time, against a dashboard floor of ninety per cent:

| Broken slice is… | …and can fail this badly | Aggregate if it dies completely |
|---|---|---|
| 5% of traffic | *anything at all* | 90.25% <!-- computed: aggregate_masking.aggregate_if_dead_s5_pct --> |
| 10% of traffic | down to 45.0% <!-- computed: aggregate_masking.worst_subgroup_s10_pct --> | 85.50% <!-- computed: aggregate_masking.aggregate_if_dead_s10_pct --> |
| 20% of traffic | down to 70.0% <!-- computed: aggregate_masking.worst_subgroup_s20_pct --> | 76.00% <!-- computed: aggregate_masking.aggregate_if_dead_s20_pct --> |
| 30% of traffic | down to 78.3% <!-- computed: aggregate_masking.worst_subgroup_s30_pct --> | 66.50% <!-- computed: aggregate_masking.aggregate_if_dead_s30_pct --> |

Below **5.26% <!-- computed: aggregate_masking.share_below_which_masking_is_total_pct -->**
of traffic, a slice that fails *every* request cannot pull the aggregate under
the floor. There is no threshold you can set on the headline number that
catches it, because the headline number never moves far enough — this is a
property of the arithmetic rather than a tuning problem with a better setting
waiting to be found.

The shaded region below is that impossibility, drawn:

<figure class="llm-fig" markdown>
![Aggregate success rate against the share of traffic in a broken slice. Two curves, one for a slice failing every request and one for a slice succeeding 45% of the time, both falling away from 95%. A horizontal line marks the 90% dashboard floor, and a shaded band on the left marks the region where neither curve can reach it.](../../assets/generated/figures/aggregate-masking-light.svg){.fig-light}
![Aggregate success rate against the share of traffic in a broken slice. Two curves, one for a slice failing every request and one for a slice succeeding 45% of the time, both falling away from 95%. A horizontal line marks the 90% dashboard floor, and a shaded band on the left marks the region where neither curve can reach it.](../../assets/generated/figures/aggregate-masking-dark.svg){.fig-dark}
<figcaption markdown>Healthy traffic at 95%, dashboard floor at 90%. In the shaded band no alert is possible at any severity, because the slice's contribution to the aggregate is bounded by its share. Rendered by `tools/figures.py` from the same arithmetic the table quotes.</figcaption>
</figure>

Two features of that picture are worth more than the curve itself. The shaded
band has a hard right-hand edge rather than fading out, because the bound is
exact rather than probabilistic; and the two curves converge as the slice
grows, which is why a *large* broken slice is a problem your existing alerting
already handles and a small one is not.

**One concrete incident, drawn rather than asserted.** A provider rolls a new
version out to a slice of traffic overnight.
600 <!-- computed: aggregate_masking.narr_n --> requests, of which
34 <!-- computed: aggregate_masking.narr_new_items --> —
5.7% <!-- computed: aggregate_masking.narr_new_share_pct --> — hit the new
version:

| | Success rate |
|---|---|
| Old version | 94.2% <!-- computed: aggregate_masking.narr_old_pct --> |
| New version | 41.2% <!-- computed: aggregate_masking.narr_new_pct --> |
| **Overall (the dashboard)** | **91.2% <!-- computed: aggregate_masking.narr_overall_pct -->** |

A 53.0 <!-- computed: aggregate_masking.narr_gap_pts -->-point gap between the
two versions, and the dashboard is green. Nobody is paged, no threshold is
crossed, and the only way anybody finds out is if somebody runs the group-by.

**Once you do slice, is it obvious?** Detection here means the slice's
confidence interval failing to overlap the rest of traffic's, for a slice at
ten per cent of traffic against ninety-five per cent healthy:

| Total n | Slice items | Outright break (45%) | Mild degradation (80%) |
|---|---|---|---|
| 200 | 20 <!-- computed: aggregate_masking.subgroup_n200_items --> | 99.5% <!-- computed: aggregate_masking.detect_pct_n200 --> | 41.3% <!-- computed: aggregate_masking.detect_mild_pct_n200 --> |
| 500 | 50 <!-- computed: aggregate_masking.subgroup_n500_items --> | 100.0% <!-- computed: aggregate_masking.detect_pct_n500 --> | 80.1% <!-- computed: aggregate_masking.detect_mild_pct_n500 --> |
| 2000 | 200 <!-- computed: aggregate_masking.subgroup_n2000_items --> | 100.0% <!-- computed: aggregate_masking.detect_pct_n2000 --> | 100.0% <!-- computed: aggregate_masking.detect_mild_pct_n2000 --> |

Three separate conclusions live in that table, and keeping them apart is what
makes it useful. An outright break is caught essentially always, even on twenty
slice items, **which means the reason these incidents run for weeks is not
insufficient data but that nobody ran the group-by** — and that inverts the
usual remedy, because the highest-value change is a scheduled query rather than
a larger evaluation set. A *mild* degradation is a genuine statistical problem:
at two hundred requests you find it forty-one per cent of the time, which is a
coin flip, and lesson 0.3's power arithmetic applies unchanged. And the
aggregate finds neither of them, at any sample size, ever.

??? question "Which of those three conclusions changes what you do on Monday?"
    The first. Outright breaks are essentially free to catch and nobody is
    catching them, so the highest-value change available is a scheduled
    slice-and-compare rather than a bigger sample or a better metric. The
    second conclusion tells you what *cannot* be fixed by slicing alone —
    gradual drift needs volume or a longer window — and the third tells you
    why the dashboard you currently have will never raise either one, which is
    the argument you will need when asking for the first.

??? question "All four signatures could be described as 'success rate is lower for some subset'. What does each one add beyond that?"
    A stage, and therefore an owner and a fix. "Some requests fail more" is
    true of every incident and actionable in none of them. The value of a
    signature is that it points at prompt construction rather than at the
    call, or at output handling rather than at retrieval, which is the
    difference between a diagnosis and an observation — and which determines
    whose afternoon is about to change.

## H · Failure modes and cost traps

**Alerting only on the aggregate.** The arithmetic in §G says this cannot work
for small slices. It is not a tuning problem with a better threshold waiting to
be discovered; there is no threshold that succeeds.

**Slicing only after an incident.** By then you are choosing which slice to
examine *after* having seen the outcome, which is how people end up confidently
explaining noise. Define the slices in advance, and let them run on a schedule.

**Slices too small to say anything.** A group of eight with a fifty per cent
success rate is a group of eight. Report the interval, and set a minimum group
size before anything is allowed to page a human.

**Slicing by too many keys.** Ten slices at five per cent give roughly a forty
per cent chance of a false alarm somewhere, so prefer a small set of keys with
mechanisms attached over an exhaustive sweep of everything you happen to have
logged.

**Not logging the key you need.** You cannot group by resolved model version if
you recorded only the alias, and lesson 0.1's record exists for precisely this
moment.

**Treating the retry storm as the cause.** Elevated attempts are usually
downstream of the real fault, so fixing the retry policy makes the symptom
quieter and the outage longer. This is why the exercise below specifies a
priority order that reports the version before the storm.

**Assuming a spike at a round number is a coincidence.** Natural lengths are
continuous. A hard edge at exactly 4096 is a ceiling somebody imposed, and the
records sitting on it are the ones that got cut.

**Reading a signature as a diagnosis without checking the mechanism.** The four
signatures in §C are patterns, and a pattern can arise for more than one
reason: a version gap can also appear because the two versions serve different
*traffic* rather than serving it differently, which is a routing question
rather than a model question. The check is cheap — compare the input-length
distributions of the two slices before concluding anything about quality — and
skipping it produces confident bug reports filed against the wrong team.

**Averaging slices back together to report a single number.** Having done the
work to separate the traffic, it is tempting to report the mean of the slice
rates as though it summarised them. It does not, because the slices have wildly
different sizes, and an unweighted mean over-represents the small ones while a
weighted mean reconstructs precisely the aggregate that concealed the problem
in the first place. Report the slices.

**Letting the minimum group size hide a growing slice.** A `min_n` guard is
necessary, and it has a failure mode of its own: a new model version that
arrives on one per cent of traffic sits below the threshold for as long as the
rollout is small, which is exactly the period during which catching it would
have been cheapest. The mitigation is to track slices that are *below* the
threshold separately, as a list of things you cannot yet say anything about,
rather than filtering them out silently.

**Treating an interval that overlaps as evidence of no problem.** Overlapping
intervals mean the data cannot distinguish the slice from the rest, which is a
statement about your sample size rather than about the system. Lesson 0.3's
power arithmetic applies unchanged: at twenty items in a slice, a mild
degradation is invisible, and "we checked and it was fine" is a considerably
stronger claim than the evidence supports.

## I · Graded practice

Two exercises. The first builds the tool; the second turns §C's table into a
runbook that executes.

<code-exercise src="tr-l5-slice"></code-exercise>

<code-exercise src="tr-l5-diagnose"></code-exercise>

<quiz-bank src="tr-l5"></quiz-bank>

Then the module's graded artifact,
[**Mini-project 0 · the reliability report**](project-reliability.md), which
asks for the same discipline in a different shape: produce the numbers a log
can support, and refuse to produce the ones it cannot.

## J · Annotated references

- **Simpson (1951), and any modern treatment of the paradox.** The statistical
  content of this lab is over seventy years old; only the slicing keys are new,
  which is worth remembering when the field presents these problems as
  unprecedented.
- **Google SRE Book, "Monitoring Distributed Systems".** The four golden
  signals, and the argument for alerting on symptoms rather than on causes.
  Worth reading against §C's table, which argues for a specific exception.
- **Sculley et al. (2015), *Hidden Technical Debt in Machine Learning
  Systems*.** The original statement that the model is a small box in a large
  diagram. Lesson 0.2's seven stages are that diagram, redrawn for this
  setting.
- **Nushi, Kamar & Horvitz (2018), *Towards Accountable AI*.** On decomposing
  system failures by component rather than reporting a single end-to-end
  number, which is the same argument this lab makes from a different direction.

## K · Extension

**Run the group-by you have never run.** Take any log you have — from your own
work, from the mini-project, from a script you wrote last week — and compute a
success rate with an interval for every value of every key you happen to have
recorded. Do not filter to the keys you think are interesting, because the
point is to find the one you would not have chosen.

Two things usually fall out of that exercise. There is at least one slice you
did not know existed, and there is at least one key you now wish you had
logged. Write both lists down; the second is more valuable than the first,
because it is the only version of this exercise that improves next month's
incident rather than explaining last month's.

**Then compute your own masking threshold.** The 5.26% in §G is specific to a
healthy rate of ninety-five per cent and a floor of ninety, and yours will
differ. The general form is $(p_h - F) / p_h$, so a system running at
ninety-nine per cent against a ninety-five per cent floor has a threshold of
roughly four per cent, while one running at eighty per cent against a
seventy-five per cent floor has one of about six. Put your two numbers in, and
you have the exact size below which your current alerting is blind — which is
a more useful thing to bring to a review than any general argument about
slicing.

**And find out how long your smallest interesting slice takes to reach `min_n`.**
Divide your daily request volume by the share of traffic a new model version or
a new tenant typically represents, and you have the number of days before that
slice can say anything at all. If the answer is longer than the interval
between provider updates, the honest conclusion is that per-day slicing cannot
detect version drift for you, and you need either a longer window or a
deliberately larger canary — both of which are decisions somebody should make
on purpose rather than discover during an incident.
