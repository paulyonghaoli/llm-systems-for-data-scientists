"""Reference implementation of the context packer.

Two things make this correct where the four-characters-per-token estimate is
not. It counts the string it actually sends, including the separators; and it
truncates on a *token* boundary by encoding first and decoding a prefix,
rather than slicing characters and hoping.
"""

from __future__ import annotations

from llmlab.tokenizer import BPETokenizer

SEPARATOR = "\n\n"


def assemble(system_prompt: str, parts: list[str]) -> str:
    return SEPARATOR.join([system_prompt, *parts]) if parts else system_prompt


def longest_prefix_within(tok: BPETokenizer, text: str, max_tokens: int) -> str:
    """Longest prefix of `text` that costs at most `max_tokens` tokens.

    Binary search over token counts of decoded prefixes. Decoding a prefix of
    the token sequence is what keeps the cut on a token boundary; it can still
    land mid-word, which is correct - words are not the unit here.
    """
    if max_tokens <= 0:
        return ""
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text
    candidate = tok.decode(ids[:max_tokens])
    # Decoding a truncated id list can strand a partial UTF-8 sequence, which
    # `decode` renders as U+FFFD. Walk back until the prefix is genuinely a
    # prefix of the original string and still inside budget.
    while candidate and (not text.startswith(candidate) or tok.count(candidate) > max_tokens):
        candidate = candidate[:-1]
    return candidate


def pack(
    tok: BPETokenizer,
    system_prompt: str,
    documents: list[str],
    context_limit: int,
    reserved_output: int,
) -> dict:
    budget = context_limit - reserved_output
    # The system prompt is not negotiable. If it alone does not fit, the
    # caller has asked for something impossible and must be told so. Trimming
    # it instead is how instructions quietly disappear in production: the
    # request still succeeds, the model just stops being told what to do.
    if tok.count(system_prompt) > budget:
        raise ValueError(
            f"system prompt costs {tok.count(system_prompt)} tokens but the budget is "
            f"{budget} (context_limit {context_limit} - reserved_output {reserved_output})"
        )

    included: list[int] = []
    truncated_text = ""

    for i, doc in enumerate(documents):
        trial = assemble(system_prompt, [documents[j] for j in included] + [doc])
        if tok.count(trial) <= budget:
            included.append(i)
            continue
        # Does not fit whole. Spend whatever budget is left on a prefix of it.
        base = assemble(system_prompt, [documents[j] for j in included])
        room = budget - tok.count(base) - tok.count(SEPARATOR)
        candidate = longest_prefix_within(tok, doc, room)
        if candidate:
            trial = assemble(system_prompt, [documents[j] for j in included] + [candidate])
            while candidate and tok.count(trial) > budget:
                candidate = candidate[:-1]
                trial = assemble(system_prompt, [documents[j] for j in included] + [candidate])
            truncated_text = candidate
        break

    parts = [documents[i] for i in included]
    if truncated_text:
        parts.append(truncated_text)
    prompt = assemble(system_prompt, parts)
    return {
        "included": included,
        "truncated_text": truncated_text,
        "prompt": prompt,
        "total_tokens": tok.count(prompt),
    }
