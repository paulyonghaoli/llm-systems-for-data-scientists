# Mini-project 1 · The context packer

Fit as much of a document set into a context window as will actually fit, and
be exactly right about how much that is.

```bash
cd projects/tokenizer_mini
python -m grader --seed 1
```

Documents are generated fresh from the seed on every run, so there is no fixed
expected output to memorise. The rubric below is the whole rubric — nothing is
hidden.

## What you implement

One function, in `student.py`:

```python
def pack(tok, system_prompt, documents, context_limit, reserved_output) -> dict
```

Return a dict with exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `included` | `list[int]` | indices of documents included **in full**, in original order |
| `truncated_text` | `str` | the partial tail of the first document that did not fit whole; `""` if none |
| `prompt` | `str` | the assembled prompt, exactly as it would be sent |
| `total_tokens` | `int` | the token count of `prompt` |

Rules:

1. The prompt is `"\n\n".join([system_prompt, *parts])`. The separators cost
   tokens; count them.
2. **Hard requirement:** `total_tokens + reserved_output <= context_limit`.
3. Documents are considered in order. The first one that does not fit whole is
   truncated to the longest prefix that does, and packing stops there.
4. Truncation happens on a **token** boundary, not a character boundary. The
   result must still be a genuine character-prefix of the original document.
5. If the system prompt alone does not fit in
   `context_limit - reserved_output`, **raise `ValueError`**. Do not trim it.

Rule 5 was not in the first draft of this project. The reference
implementation scored 96.7/100 against its own rubric because the spec never
said what a packer should do when the instructions themselves do not fit, and
"silently return a prompt with the instructions trimmed" is the worst
available answer — the request succeeds, the model just stops being told what
to do. The rule exists because the grader found the hole.

## Rubric — 100 points

Each criterion is scored as the fraction of scenarios that satisfy it, over 12
seeded scenarios plus 3 fixed edge cases.

| | Criterion | Points | Passes when |
|---|---|---:|---|
| **A** | Budget respected | 30 | A fresh count of the returned `prompt`, plus `reserved_output`, never exceeds `context_limit` |
| **B** | Accounting consistent | 25 | The `total_tokens` you report equals a fresh count of the `prompt` you actually returned |
| **C** | Selection matches reference | 20 | The same documents are included, in the same order |
| **D** | Truncation on a token boundary | 15 | `truncated_text` is a character-prefix of the source document *and* its token sequence is a prefix of the source's token sequence |
| **E** | Degenerate cases | 10 | No documents · budget below the system prompt (must raise) · one document far larger than the window |

Pass mark: **80**.

Criterion B is the one worth thinking about. It is possible to pass A and fail
B by being accidentally conservative — budgeting against one string and
sending another. That bug does not show up as an error; it shows up months
later as unexplained cost, or as a request that overruns the window the first
time somebody sends Japanese.

## The starter

`student.py` ships with the estimate everybody reaches for first: four
characters per token. Run the grader against it before you change anything.
It scores **26.6/100** and the breakdown shows precisely which of the five
claims that estimate cannot support.

## Useful commands

```bash
python -m grader --seed 7 --scenarios 20    # more scenarios, different seed
python -m grader --reference --seed 1       # what full marks looks like
python -m grader --sweep 30                 # what CI runs
```
