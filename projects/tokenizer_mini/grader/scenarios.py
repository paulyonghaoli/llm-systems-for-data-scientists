"""Seeded scenario generation.

Documents are drawn from the same held-out samples the lesson measures, so a
packer that works only on English prose scores badly - which is the point.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

from llmlab.tokenizer import BPETokenizer

REPO = Path(__file__).resolve().parents[3]
CORPUS = REPO / "data" / "corpus.txt"
SAMPLES = REPO / "data" / "samples"

VOCAB_SIZE = 512


@lru_cache(maxsize=1)
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(CORPUS.read_text(encoding="utf-8"), vocab_size=VOCAB_SIZE)


@lru_cache(maxsize=1)
def _pool() -> list[str]:
    chunks: list[str] = []
    for path in sorted(SAMPLES.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) >= 40:
                chunks.append(para)
    for para in CORPUS.read_text(encoding="utf-8").split("\n\n"):
        para = para.strip()
        if len(para) >= 40:
            chunks.append(para)
    return chunks


def scenario(seed: int) -> dict:
    rng = random.Random(seed)
    pool = _pool()
    n_docs = rng.randint(3, 9)
    documents = [rng.choice(pool) for _ in range(n_docs)]
    system_prompt = rng.choice([
        "You are a careful assistant. Answer only from the documents provided.",
        "Summarise the following records for an internal report.",
        "Answer the question using the context below. If it is not there, say so.",
    ])
    tok = tokenizer()
    full_cost = tok.count("\n\n".join([system_prompt, *documents]))
    # Limits that bite: usually somewhere between "one document fits" and
    # "everything fits", so truncation is exercised most of the time.
    context_limit = rng.randint(int(full_cost * 0.25), int(full_cost * 1.1)) + 40
    reserved_output = rng.choice([0, 16, 64, 128])
    return {
        "seed": seed,
        "system_prompt": system_prompt,
        "documents": documents,
        "context_limit": context_limit,
        "reserved_output": reserved_output,
    }


def degenerate_scenarios() -> list[dict]:
    """The three edge cases that a packer written against the happy path gets
    wrong: nothing to pack, no room to pack it, and one document that cannot
    fit however much you trim the others."""
    tok = tokenizer()
    long_doc = CORPUS.read_text(encoding="utf-8")
    system = "You are a careful assistant."
    return [
        {
            "name": "no documents",
            "expect": "ok",
            "system_prompt": system,
            "documents": [],
            "context_limit": 500,
            "reserved_output": 0,
        },
        {
            # The system prompt is not negotiable, so the only correct answer
            # is to refuse. A packer that trims it instead returns a valid
            # request with the instructions filed off.
            "name": "budget smaller than the system prompt",
            "expect": "raises",
            "system_prompt": system,
            "documents": ["some text that will never fit"],
            "context_limit": max(1, tok.count(system) // 2),
            "reserved_output": 0,
        },
        {
            "name": "one document far larger than the window",
            "expect": "ok",
            "system_prompt": system,
            "documents": [long_doc],
            "context_limit": tok.count(system) + 120,
            "reserved_output": 16,
        },
    ]
