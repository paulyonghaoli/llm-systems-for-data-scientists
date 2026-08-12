"""Mini-project 1 - the context packer.

Implement `pack` below, then grade it:

    cd projects/tokenizer_mini
    python -m grader --seed 1

The full specification and the rubric are in README.md. Read them; the grader
scores against the published rubric, not against a fixed expected output, and
the documents are generated fresh from the seed on every run.

The starter below is the estimate everybody reaches for first: four
characters per token. It is a reasonable rule of thumb and a completely
unreasonable budget, and the grader will show you exactly how it fails.
"""

from __future__ import annotations

from llmlab.tokenizer import BPETokenizer

SEPARATOR = "\n\n"


def pack(
    tok: BPETokenizer,
    system_prompt: str,
    documents: list[str],
    context_limit: int,
    reserved_output: int,
) -> dict:
    """Fit as many documents as possible into the context window.

    Returns a dict with keys:
      included        list[int]  indices of documents included in full, in order
      truncated_text  str        the partial tail of the first document that
                                 did not fit whole ("" if there is none)
      prompt          str        the assembled prompt, exactly as it would be sent
      total_tokens    int        the token count of `prompt`

    Hard requirement: total_tokens + reserved_output <= context_limit.
    """
    budget = context_limit - reserved_output
    used = len(system_prompt) // 4

    included: list[int] = []
    for i, doc in enumerate(documents):
        cost = len(doc) // 4
        if used + cost > budget:
            break
        included.append(i)
        used += cost

    parts = [system_prompt] + [documents[i] for i in included]
    prompt = SEPARATOR.join(parts)
    return {
        "included": included,
        "truncated_text": "",
        "prompt": prompt,
        "total_tokens": used,
    }
