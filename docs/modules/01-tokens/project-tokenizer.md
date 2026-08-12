---
status: Verified
last_verified: 2026-08-09
volatility: low
---

# Mini-project 1 · The context packer

Fit as much of a document set into a context window as will actually fit, and
be exactly right about how much that is.

```bash
cd projects/tokenizer_mini
python -m grader --seed 1
```

Documents are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric is published in full — nothing is
hidden — in
[`projects/tokenizer_mini/README.md`](https://github.com/paulyonghaoli/llm-systems-for-data-scientists/blob/main/projects/tokenizer_mini/README.md).

## What you implement

One function, in `student.py`:

```python
def pack(tok, system_prompt, documents, context_limit, reserved_output) -> dict
```

returning `included`, `truncated_text`, `prompt` and `total_tokens`, subject
to five rules:

1. The prompt is `"\n\n".join([system_prompt, *parts])`. The separators cost
   tokens; count them.
2. `total_tokens + reserved_output <= context_limit`. Always.
3. Documents are taken in order; the first that does not fit whole is
   truncated and packing stops.
4. Truncation happens on a **token** boundary, and the result must still be a
   genuine character-prefix of the source document.
5. If the system prompt alone does not fit, **raise `ValueError`**.

## The rubric

| | Criterion | Points |
|---|---|---:|
| **A** | Budget respected | 30 |
| **B** | Accounting consistent — reported count equals a fresh count of what you returned | 25 |
| **C** | Selection matches the reference | 20 |
| **D** | Truncation on a token boundary | 15 |
| **E** | Degenerate cases | 10 |

Pass mark **80**. The shipped starter — four characters per token — scores
**26.6**, and the breakdown tells you exactly which of the five claims that
estimate cannot support.

## Where rule 5 came from

It was not in the first draft. The reference implementation scored 96.7 out of
100 against its own rubric, failing the one edge case the specification never
addressed: what should a packer do when the *system prompt itself* does not
fit? Trimming it silently is the worst available answer — the request
succeeds, the response looks plausible, and the model has simply stopped being
told what to do. Refusing loudly is the only defensible behaviour, so it
became rule 5.

The rubric found the hole in the spec. That is what rubrics are for, and it is
why the reference is graded in CI on every commit.
