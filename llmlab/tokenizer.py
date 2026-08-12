"""Byte-level BPE, written to be read.

This is the reference implementation that lesson 1.1 builds toward. It is
deliberately small and deliberately explicit about the three decisions that
are usually left implicit in tutorials, because each of them changes the
resulting vocabulary and none of them is checked by a round-trip test:

1. **Pre-tokenization.** Merges never cross the boundaries produced by
   `SPLIT_PATTERN`. Without this, BPE happily learns a single token for
   ``". The "`` and the model's alphabet fills up with punctuation glue.
2. **Tie-breaking.** When two pairs are equally frequent, we take the
   lexicographically smallest pair. Any rule works; *having* a rule is what
   makes training reproducible. Two implementations that disagree here
   produce different vocabularies and are not interchangeable.
3. **`vocab_size` is a maximum, not a promise.** Training stops early when no
   pair occurs twice, because a merge that fires once trades a vocabulary
   slot for nothing.

Byte-level, so there is no such thing as an out-of-vocabulary character: every
string decodes to bytes and every byte is already a token.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

#: Pre-tokenizer. GPT-2's published pattern uses Unicode property classes
#: (``\p{L}``, ``\p{N}``) which need the third-party ``regex`` module; this is
#: the closest stdlib equivalent, so it is what runs in the browser. The final
#: ``[\s\S]`` branch is a catch-all that guarantees no character is ever
#: dropped — see ``test_pretokenize_is_lossless``.
SPLIT_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)"  # common English contractions
    r"| ?[^\W\d_]+"  # a run of letters, optional leading space
    r"| ?\d+"  # a run of digits, optional leading space
    r"| ?[^\s\w]+"  # a run of punctuation or symbols
    r"|\s+(?!\S)"  # trailing whitespace at end of input
    r"|\s+"  # any other whitespace run
    r"|[\s\S]",  # catch-all: nothing may be dropped
    re.UNICODE,
)

Pair = tuple[int, int]


def pretokenize(text: str) -> list[str]:
    """Split text into chunks that merges are not allowed to cross.

    Lossless: ``"".join(pretokenize(t)) == t`` for every string.
    """
    return SPLIT_PATTERN.findall(text)


def pair_counts(chunks: list[list[int]]) -> Counter[Pair]:
    """Count every adjacent pair across all chunks."""
    counts: Counter[Pair] = Counter()
    for ids in chunks:
        for pair in zip(ids, ids[1:], strict=False):
            counts[pair] += 1
    return counts


def merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@dataclass
class BPETokenizer:
    """A trained byte-level BPE tokenizer.

    `merges` is ordered: insertion order is learning order, and encoding
    applies the *earliest-learned* applicable merge first. That ordering is
    the tokenizer — losing it makes the vocabulary useless.
    """

    merges: dict[Pair, int] = field(default_factory=dict)
    vocab: dict[int, bytes] = field(default_factory=lambda: {i: bytes([i]) for i in range(256)})

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (the byte alphabet), got {vocab_size}")
        chunks = [list(chunk.encode("utf-8")) for chunk in pretokenize(text)]
        merges: dict[Pair, int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for new_id in range(256, vocab_size):
            stats = pair_counts(chunks)
            if not stats:
                break
            # Most frequent pair; ties broken by the lexicographically
            # smallest pair so training is reproducible.
            best = min(stats, key=lambda p: (-stats[p], p))
            if stats[best] < 2:
                break
            chunks = [merge(ids, best, new_id) for ids in chunks]
            merges[best] = new_id
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]

        return cls(merges=merges, vocab=vocab)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text):
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                present = {p for p in zip(ids, ids[1:], strict=False) if p in self.merges}
                if not present:
                    break
                # Earliest-learned merge wins. Applying them out of order
                # produces a different — and wrong — token sequence.
                best = min(present, key=lambda p: self.merges[p])
                ids = merge(ids, best, self.merges[best])
            out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
