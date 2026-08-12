"""Tests for the reference BPE.

The interesting ones are the properties that a naive implementation passes
and a subtly wrong one does not: losslessness of pre-tokenization, merge
*ordering* during encode, and reproducibility of tie-breaking.
"""

from __future__ import annotations

import random

import pytest

from llmlab.tokenizer import BPETokenizer, merge, pair_counts, pretokenize

CORPUS = (
    "the cat sat on the mat. the cat sat on the hat. "
    "a cat and a hat and a mat and a bat. " * 8
)

MESSY = "Hello, world! naïve café 東京 🙂 x_y_z 3.14159 <div class='a'>&amp;</div>\n\ttab\n"


# --- pre-tokenization -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", " ", "a", "_", "a_b", MESSY, "trailing   ", "\n\n\n", "1234", " 1234", "don't"],
)
def test_pretokenize_is_lossless(text: str) -> None:
    """No branch of SPLIT_PATTERN may drop a character.

    This is why the pattern ends in a `[\\s\\S]` catch-all: an earlier draft
    had no underscore branch, and `re.findall` silently discarded every `_`.
    A round-trip test on ASCII prose would never have noticed.
    """
    assert "".join(pretokenize(text)) == text


def test_pretokenize_is_lossless_on_random_unicode() -> None:
    rng = random.Random(0)
    alphabet = "abz09 _-.,!'\n\té東\U0001f642"
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert "".join(pretokenize(text)) == text


def test_pretokenize_keeps_the_leading_space_with_the_word() -> None:
    # " the" and "the" are different chunks, and therefore different tokens.
    assert pretokenize("the the") == ["the", " the"]


def test_pretokenize_splits_letters_from_digits() -> None:
    assert pretokenize("abc123") == ["abc", "123"]


# --- merge mechanics --------------------------------------------------------


def test_pair_counts_counts_adjacent_pairs_only() -> None:
    counts = pair_counts([[1, 2, 1, 2, 3]])
    assert counts[(1, 2)] == 2
    assert counts[(2, 1)] == 1
    assert counts[(1, 3)] == 0


def test_merge_is_non_overlapping() -> None:
    # "aaa" with pair (a,a): the classic off-by-one. Two greedy left-to-right
    # merges are impossible in a 3-run; the correct answer leaves one behind.
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge([1, 1, 1, 1], (1, 1), 9) == [9, 9]


def test_merge_leaves_other_ids_untouched() -> None:
    assert merge([5, 1, 2, 5], (1, 2), 9) == [5, 9, 5]


# --- training ---------------------------------------------------------------


def test_train_respects_vocab_size_as_a_maximum() -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    assert len(tok.merges) <= 300 - 256
    assert tok.vocab_size <= 300


def test_train_stops_early_when_nothing_repeats() -> None:
    tok = BPETokenizer.train("abcdef", vocab_size=1000)
    assert tok.merges == {}
    assert tok.vocab_size == 256


def test_train_is_reproducible() -> None:
    a = BPETokenizer.train(CORPUS, vocab_size=320)
    b = BPETokenizer.train(CORPUS, vocab_size=320)
    assert a.merges == b.merges
    assert a.vocab == b.vocab


def test_training_reduces_token_count() -> None:
    base = len(CORPUS.encode("utf-8"))
    tok = BPETokenizer.train(CORPUS, vocab_size=350)
    assert tok.count(CORPUS) < base


# --- encode / decode --------------------------------------------------------


@pytest.mark.parametrize("text", ["", "the cat", MESSY, "unseen ünicode \U0001f680", "1234567890"])
def test_round_trip(text: str) -> None:
    tok = BPETokenizer.train(CORPUS, vocab_size=350)
    assert tok.decode(tok.encode(text)) == text


def test_encode_applies_the_earliest_merge_first() -> None:
    """Encoding order is not an implementation detail.

    Hand-built tokenizer: 'a'+'b' was learned first, then 'b'+'c'. On "abc"
    the correct encode fires (a,b) and leaves 'c'; firing the later-learned
    (b,c) first gives a different, wrong sequence.
    """
    a, b, c = ord("a"), ord("b"), ord("c")
    tok = BPETokenizer(
        merges={(a, b): 256, (b, c): 257},
        vocab={**{i: bytes([i]) for i in range(256)}, 256: b"ab", 257: b"bc"},
    )
    assert tok.encode("abc") == [256, c]


def test_encode_never_merges_across_a_pretoken_boundary() -> None:
    tok = BPETokenizer.train("ab ab ab ab ab ab", vocab_size=300)
    # 'b' and ' a' are adjacent in the text but sit in different chunks,
    # so no token may ever span them.
    for pair in tok.merges:
        assert tok.vocab[pair[0]] + tok.vocab[pair[1]] == tok.vocab[tok.merges[pair]]
    for token in tok.vocab.values():
        assert token not in (b"b a", b"b ab")


def test_unknown_bytes_still_encode() -> None:
    """Byte-level BPE has no OOV. A tokenizer trained only on English still
    round-trips Japanese — just expensively."""
    tok = BPETokenizer.train(CORPUS, vocab_size=350)
    text = "東京は雨です"
    assert tok.decode(tok.encode(text)) == text
    assert tok.count(text) == len(text.encode("utf-8"))
