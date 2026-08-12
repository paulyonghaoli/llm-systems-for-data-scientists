# Glossary

Every term is defined where it is first used, inside a **Terms used in this
lesson** box at the top of the section that needs it. This page collects those
definitions so you can look one up without hunting for the lesson that
introduced it.

Terms are listed alphabetically. The lesson in brackets is where the term is
introduced and where the fuller explanation lives.

## A–C

**Abstention** *(0.3, Capstone I)* — declining to answer rather than guessing.
A system that abstains on questions its context cannot support scores better
on groundedness than one that always produces something, and measuring
abstention requires deliberately unanswerable questions in the evaluation set.

**Chat template** *(1.3)* — the string format a model was trained to expect,
wrapping each message in role markers. It is applied to your list of messages
before tokenization, so the tokens you are billed for include the wrapper.

**Cohort key** *(0.1)* — the tuple of properties two runs must share before
their results may be compared: resolved model version, decoding parameters and
rendered prompt. Two results with different cohort keys are one measurement
each, of two different things.

**Context window** *(1.3)* — the maximum number of tokens a request may
contain, counting the prompt and the generated answer together. It is a
budget shared between input and output, not a limit on input alone.

## D–L

**Decoding parameters** *(0.1, 1.2)* — the settings that turn the model's
probability distribution into a chosen token: temperature, top-p, top-k,
repetition penalty, and any seed. They are not part of the model; they
configure a small piece of ordinary code that runs after it.

**Finish reason** *(1.4)* — the field in a response saying why generation
stopped: `stop` for a natural ending, `length` for truncation at `max_tokens`,
`content_filter` for a refusal, `tool_calls` for a paused turn. A response can
be HTTP 200 and still not be an answer, and this field is the only reliable
way to tell.

**Groundedness** *(0.3, Capstone I)* — whether the claims in an answer are
supported by the retrieved context rather than by the model's memory. It is
measurable only if the evaluation corpus is one the model plausibly has not
memorised.

**Idempotency key** *(1.4)* — a unique identifier you generate per *logical*
operation and send with the request, so that a server can recognise a retry
and return the original result instead of performing the side effect twice.

**Logits** *(1.2)* — the raw, unnormalised scores the model produces for every
token in the vocabulary at a given position. Softmax turns them into
probabilities; almost all of them are negative.

## M–R

**Nucleus** *(1.2)* — the set of tokens top-p sampling keeps: the shortest
list of highest-probability tokens whose combined probability reaches `p`. Its
size is not fixed, and varies by orders of magnitude with how confident the
model is.

**Prefill** *(0.4, 1.3, Module 11)* — the phase in which the model processes
the whole prompt before producing the first output token. Its cost grows with
prompt length, which is why a longer prompt raises time-to-first-token as well
as the bill.

**Refusal** *(0.1)* — a successful call whose output declines the request. It
belongs in quality metrics rather than reliability metrics, because the
infrastructure worked exactly as intended.

**Retry amplification** *(0.2, 1.4)* — the failure mode in which a struggling
service receives *more* traffic because its clients are retrying, converting a
partial outage into a total one.

## S–Z

**Structural breakout** *(2.1)* — content that escapes the region a prompt
placed it in, so that parsing the prompt by its delimiters no longer recovers
the intended regions. It is a parsing problem with a correct solution, and it
is distinct from prompt injection.

**Time to first token (TTFT)** *(0.4, 1.4)* — how long a request waits before
any output appears. Driven by queueing and by prompt length. With streaming it
is what the user experiences as responsiveness.

**Time per output token (TPOT)** *(0.4)* — how quickly the rest of the answer
arrives once generation has started. Independent of prompt length.

**Token** *(1.1)* — the unit a model reads and writes, and the unit you are
billed in. Not a word and not a character: a compression scheme fitted to a
corpus before the model was trained, and frozen thereafter.

**Trust lattice** *(2.1)* — the ordering of text sources by how much authority
each may grant: system prompt, then authenticated user, then retrieved
documents and tool output. Capabilities are gated on the *origin* of a
request, never on its content.

**Wilson interval** *(0.3)* — a confidence interval for a proportion that
behaves correctly at small sample sizes and near 0 or 1, where the textbook
normal-approximation interval does not.
