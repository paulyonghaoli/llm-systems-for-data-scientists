---
status: Verified
last_verified: 2026-08-09
volatility: high
pyodide: true
---

# 1.1 · Tokens are not words: BPE from scratch

Every number in this lesson is produced by
[`experiments/bpe_compression.py`](https://github.com/paulyonghaoli/llm-systems-for-data-scientists/blob/main/experiments/bpe_compression.py)
and re-checked against the prose by CI. Comparisons against real production
tokenizers come from a fixture recorded on **2026-08-09**; see
[the living doc](../../living/models.md) for what that pin means and when it
expires.

## A · Why this matters

Three things you are about to depend on are denominated in tokens: the bill,
the context limit, and the truncation policy that decides which half of a
document the model actually sees. None of them is denominated in words or
characters, and the conversion factor is not a constant. It is a function
somebody else fitted, on a corpus you cannot inspect, before the model was
trained.

Measured on the held-out samples in this repository, under one widely
deployed vocabulary (`cl100k_base`):

| Text | Tokens per character |
|---|---|
| English prose | 0.206 <!-- computed: bpe_compression.tpc_english_cl100k --> |
| Python source | 0.259 <!-- computed: bpe_compression.tpc_python_cl100k --> |
| JSON | 0.356 <!-- computed: bpe_compression.tpc_json_cl100k --> |
| A numeric table | 0.488 <!-- computed: bpe_compression.tpc_numbers_cl100k --> |
| Japanese prose | 1.062 <!-- computed: bpe_compression.tpc_japanese_cl100k --> |

The spread between the top and bottom rows is a factor of
5.16 <!-- computed: bpe_compression.english_over_japanese_cl100k -->. A
"2,000-character chunk" is about 412 tokens of English and about 2,124 tokens
of Japanese. If your chunker was sized against English and your users write
Japanese, you are not sending the chunk you think you are sending, and nothing
in your stack will tell you.

??? question "You budget 4 characters per token and send a 40,000-character document into a 10,000-token window. What happens, and for which inputs?"
    Your estimate says exactly 10,000 tokens — right at the edge. For English
    it is roughly 8,200, so you are fine and never learn anything. For the
    numeric table it is roughly 19,500, so the request is rejected or silently
    truncated. The rule of thumb is not wrong on average; it is wrong on the
    inputs that matter, and being right on average is what stops you noticing.

## B · Mental model

**A tokenizer is a compression scheme frozen into a contract.**

It is fitted once, by counting, on a corpus chosen by someone else. Then it
never changes for the life of the model. Everything the model reads passes
through it and everything the model writes comes back out through it, so its
vocabulary is the model's entire alphabet. If a kind of text is represented
badly, the model handles that text badly, and no amount of prompting repairs
it — the damage is upstream of anything you can say.

Two consequences worth holding onto:

- **It is not preprocessing you own.** You cannot refit it, and in most cases
  you cannot see what it was fitted on.
- **It degrades ungracefully.** Text far from the training distribution does
  not get slightly worse representation; it falls all the way back to
  individual bytes.

## C · Mechanism

Byte-pair encoding begins from the raw bytes, which is the decision that makes
everything afterwards simple: all 256 byte values are symbols before training
starts, so no character can ever be out of vocabulary and no special "unknown"
token is required. From there the algorithm counts how often each **adjacent
pair** of symbols occurs across the corpus, takes the most frequent pair, mints
a new symbol to stand for it, and replaces every occurrence of that pair with
the new symbol — after which it counts again and repeats, until the vocabulary
reaches the size it was asked for.

That really is the entire algorithm, and it is short enough to write in an
afternoon. What separates one implementation from another is therefore not the
loop but three decisions taken around it, each of which changes the resulting
vocabulary, and none of which shows up in the round-trip test that most people
reach for as a correctness check.

!!! info "Terms used in this lesson"
    **Merge** — one learned rule replacing a specific adjacent pair of symbols
    with a single new symbol. A trained tokenizer is an *ordered* list of
    these, and the order is part of the tokenizer.

    **Pre-tokenization** — cutting the text into chunks before training, so
    that no merge is ever allowed to span a chunk boundary.

    **Vocabulary** — the full symbol table: the 256 byte values plus one entry
    per learned merge. Its size is the `vocab_size` you requested, or less.

**Decision 1 — where merges may not happen.** If you run the counting step on
raw English, the first pairs it learns are sequences like `e␣` and `.␣`,
because whitespace is the most frequent neighbour of very nearly everything.
Left to itself the procedure will therefore learn a single symbol meaning "full
stop, space, the word *the*", which is genuinely frequent and completely
useless, since it welds together three things that have nothing to do with one
another and that will never recur as a unit in a context where the model needs
them together. The remedy is to cut the text into chunks first, along the
boundaries between letters, digits, punctuation and whitespace, and to confine
every merge within a single chunk.

**Decision 2 — tie-breaking.** Two pairs will frequently occur exactly as often
as each other, especially in the early rounds and on small corpora, so the
procedure needs a rule for choosing between them. Any rule works, and what
matters is *having* one: without a rule the outcome depends on the order a
dictionary happened to be built in, which means two runs of the same program
over the same text produce different tokenizers and neither can be regenerated
on demand. Since a vocabulary that cannot be regenerated also cannot be
audited, this implementation takes the lexicographically smallest of the tied
pairs, which is arbitrary but stated.

**Decision 3 — when to stop.** The `vocab_size` you pass is a maximum rather
than a promise, because once no remaining pair occurs more than once, each
further merge would spend a vocabulary slot in order to shorten exactly one
position in the corpus — a trade that is never worth making. Training therefore
stops at that point, and a small corpus yields a small vocabulary no matter
what size was requested.

One further decision belongs to *encoding* rather than to training, and it
causes more confusion than the other three combined. When several learned
merges are applicable to the same sequence at the same moment, the one that was
learned **earliest** must fire, rather than the one that appears leftmost in
the text or the one that would produce the longest token, because encoding is
replaying the order in which training built the vocabulary up.

??? question "Two colleagues train BPE on the same corpus with the same `vocab_size` and get different vocabularies. Nothing is random. What differs?"
    Their tie-breaking. Equal-frequency pairs are common, especially early,
    and if the rule is "whatever `max()` returns first" then the answer
    depends on dictionary insertion order — which depends on the order the
    corpus was read. Both vocabularies are valid BPE; neither is
    reproducible, and a vocabulary you cannot regenerate is one you cannot
    audit.

## D · From data science to LLM systems

The closest thing you already know is `CountVectorizer`, and the analogy is
good for about thirty seconds.

| `CountVectorizer` | A tokenizer |
|---|---|
| Vocabulary learned from your corpus by counting | Same — BPE is a counting procedure, not a trained model |
| You choose `max_features` | You choose nothing; `vocab_size` was chosen years ago |
| Out-of-vocabulary terms are dropped | There is no out-of-vocabulary; everything backs off to bytes |
| Order is discarded | Order is the entire point |
| Refit it when the data shifts | It cannot be refit. It is an input contract |

Two habits transfer badly enough to be worth naming.

**The leakage habit.** You are trained to split before you fit, because
fitting a vectorizer on the test set leaks. Here the vectorizer was fitted on a
corpus that plausibly contains your evaluation set, and you cannot check.
That is not a tokenizer problem you can solve; it is a contamination problem
you will meet again in Course IV, and the honest response is to design
evaluations that assume it.

**The "preprocessing is mine" habit.** In a normal pipeline, noticing that
your vectorizer learned three spellings of the same word is the beginning of a
fix. Here it is the beginning of a workaround. The useful move is not to
normalise the text into something the tokenizer likes — that pushes the input
further from what the model was trained on — but to *measure* the cost and
budget for it.

??? question "Why is `fit_transform` on the full dataset a bug for a vectorizer but unavoidable for a tokenizer?"
    Because the tokenizer's fit already happened, once, globally, before you
    existed. There is no split you can perform that undoes it. The correct
    response is not to fix the fit but to stop assuming your evaluation data
    is unseen — which is a real change to how you design evaluations, not a
    pipeline tweak.

## E · Minimal implementation

The whole trainer, with the three decisions made explicit:

```python
from collections import Counter

def train_bpe(text, vocab_size):
    chunks = [list(c.encode("utf-8")) for c in pretokenize(text)]  # decision 1
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}

    for new_id in range(256, vocab_size):
        stats = pair_counts(chunks)
        if not stats:
            break
        best = min(stats, key=lambda p: (-stats[p], p))   # decision 2
        if stats[best] < 2:                               # decision 3
            break
        chunks = [merge(ids, best, new_id) for ids in chunks]
        merges[best] = new_id
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]

    return merges, vocab
```

And encoding, which is the same loop with the order fixed rather than
discovered:

```python
def encode(text, merges):
    out = []
    for chunk in pretokenize(text):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            applicable = {p for p in zip(ids, ids[1:]) if p in merges}
            if not applicable:
                break
            earliest = min(applicable, key=lambda p: merges[p])  # not leftmost
            ids = merge(ids, earliest, merges[earliest])
        out.extend(ids)
    return out
```

You will write `pair_counts`, `merge` and `encode` yourself in §I. The full
reference is `llmlab/tokenizer.py`.

## F · Production practice

Three implementations you will actually meet. Versions pinned as of
**2026-08-09**; treat everything in this section as dated.

- **`tiktoken`** (0.13.0) — byte-level BPE, the encodings named
  `cl100k_base` and `o200k_base`. Fast, Rust-backed, and **not available in
  the browser**, which is why this curriculum ships recorded counts from it
  rather than calling it. Its pre-tokenizer uses Unicode property classes
  (`\p{L}`, `\p{N}`) via the `regex` package.
- **SentencePiece** — trains directly on raw text with no pre-tokenization
  step, representing whitespace explicitly as `▁`. Supports a unigram
  language-model vocabulary as well as BPE. Common in multilingual and
  open-weight models.
- **`tokenizers`** (Hugging Face) — a pipeline of normalizer →
  pre-tokenizer → model → decoder, which is a useful mental frame even if you
  use something else: those are the four places a tokenizer can surprise you.

The stdlib `re` module cannot express `\p{L}`, so the implementation here uses
the closest equivalent (`[^\W\d_]`) plus a catch-all branch that guarantees no
character is ever dropped. That catch-all is not decoration: an earlier draft
had no branch matching `_`, and `re.findall` silently discarded every
underscore in the corpus. A round-trip test on ASCII prose would not have
caught it, because the underscores were gone before the round trip started.
There is a test for it now.

## G · Experiment

Train on this repository's frozen 6.7 <!-- computed: bpe_compression.corpus_kb -->
KB corpus and measure tokens per character on five held-out samples:

```bash
python experiments/bpe_compression.py
```

| Sample | Bytes/char | Ours (vocab 512) | `cl100k_base` | `o200k_base` |
|---|---|---|---|---|
| English | 1.000 | 0.472 <!-- computed: bpe_compression.tpc_english_ours --> | 0.206 | 0.206 |
| Python | 1.000 | 0.660 <!-- computed: bpe_compression.tpc_python_ours --> | 0.259 | 0.259 |
| JSON | 1.000 | 0.868 <!-- computed: bpe_compression.tpc_json_ours --> | 0.356 | 0.355 |
| Numeric table | 1.000 | 0.956 <!-- computed: bpe_compression.tpc_numbers_ours --> | 0.488 | 0.488 |
| Japanese | 2.977 | 2.977 <!-- computed: bpe_compression.tpc_japanese_ours --> | 1.062 | 0.789 <!-- computed: bpe_compression.tpc_japanese_o200k --> |

Three things in that table are worth more than the table.

**Our tokenizer is 2.29× <!-- computed: bpe_compression.ratio_english_ours_over_cl100k -->
worse than a production one on English**, using an identical algorithm. The
entire difference is training data: 6.7 KB of one person's prose against a
large fraction of the written internet. Scale is not a detail of tokenizer
quality; it is essentially all of it.

**On Japanese, our tokenizer achieves nothing at all.** Its tokens-per-character
is 2.977, which is exactly the *bytes*-per-character of the same text — every
single character has fallen back to raw bytes, and not one learned merge
applies. This is what "degrades ungracefully" looks like when you measure it.

??? question "Our tokenizer's Japanese figure equals the *bytes* per character exactly, to three decimal places. Why is that not a coincidence?"
    Because it means no merge fired even once. Byte-level BPE starts with one
    token per byte and only gets cheaper by applying learned merges; if the
    output is exactly one token per byte, the learned vocabulary contributed
    nothing at all. The corpus was English, so not one of its 256 merges
    describes a sequence that occurs in Japanese text.

**Between the two production vocabularies, English is unchanged and Japanese
is 25.7% <!-- computed: bpe_compression.japanese_saving_o200k_pct --> cheaper.**
0.206 versus 0.206 on English; 0.259 versus 0.259 on Python. A newer, larger
vocabulary bought nothing whatsoever on Latin-script text and a quarter off
Japanese. If you serve only English, a tokenizer generation change is not an
upgrade you will ever notice on the bill.

## H · Failure modes and cost traps

**Budgeting in characters.** The one that actually costs money. Four
characters per token is a fine estimate of English prose and a poor estimate of
everything else, and estimates do not compose: an under-estimate on a long
document becomes a rejected request or a silent truncation.

**Trusting a round-trip test.** `['the', 're']` and `['ther', 'e']` both decode
to `"there"`. An encoder that takes the leftmost applicable merge instead of
the earliest-learned one round-trips perfectly on every input and disagrees
with the model's vocabulary constantly. Round-tripping tests your decoder
against your encoder; it says nothing about whether either matches the
tokenizer the model was trained with.

??? question "You have written an encoder and want one test that would actually catch a wrong merge order. What is it?"
    Encode a string whose correct tokenization uses two early merges that a
    leftmost-first implementation would skip over, and assert on the *pieces*
    rather than the ids: `[decode([i]) for i in encode("there")]` must be
    `["the", "re"]`, not `["ther", "e"]`. Asserting on pieces rather than ids
    also survives a change to the vocabulary, and it fails with a message a
    human can read.

**Truncating on characters.** `text[:2000]` can cut mid-token, handing the
model a final token it never saw in training, and the number of tokens you
actually send varies several-fold by language. Encoding, slicing the ids and
decoding costs nothing and cannot do this.

**Stripping whitespace to save room.** The leading space belongs to the word
after it. Removing spaces replaces common single tokens with rarer multi-token
sequences: the count goes *up*, and the input gets stranger.

**Comparing serialization formats per character.** This one is mine. The first
version of the experiment above compared tokens-per-character between the JSON
sample and the numeric table and concluded JSON was 27% cheaper. Both files are
in the repository, both numbers were correct, and the conclusion was
meaningless — they carry different information. Measured properly, on the
*same* seven records:

| Format | Tokens |
|---|---|
| Tab-separated table | 201 <!-- computed: bpe_compression.tok_numbers_cl100k --> |
| Pretty-printed JSON | 534 <!-- computed: bpe_compression.tok_numbers_json_cl100k --> |

That is 2.66× <!-- computed: bpe_compression.json_vs_tsv_same_data_cl100k --> —
333 <!-- computed: bpe_compression.json_vs_tsv_extra_tokens --> extra tokens for
seven rows — to say exactly the same thing. Per character JSON *does* look
cheaper, because repeated English-like keys tokenize beautifully. Tokens per
character is the wrong unit for this question; tokens per unit of information
is the right one, and the two point in opposite directions.

**Assuming `vocab_size` is what you got.** Ask for 1024 on a small corpus and
you get 563 <!-- computed: bpe_compression.merges_v1024 --> merges, because the
corpus ran out of repeated pairs. Code that indexes an embedding table by
assumed vocabulary size will fail somewhere much less obvious.

## I · Graded practice

Three exercises, in order. Each starter contains a real bug rather than a
blank; run it first and read what it prints.

<code-exercise src="tok-l1-pairs"></code-exercise>

<code-exercise src="tok-l1-merge"></code-exercise>

<code-exercise src="tok-l1-encode"></code-exercise>

<quiz-bank src="tok-l1"></quiz-bank>

Then the graded artifact for this module:
[**Mini-project 1 · The context packer**](project-tokenizer.md). It asks for
one function and scores it against a published rubric on seeded document sets.
The starter — four characters per token — scores 26.6 out of 100.

## J · Annotated references

- **Sennrich, Haddow & Birch (2015), *Neural Machine Translation of Rare Words
  with Subword Units*.** The paper that brought BPE from data compression to
  NLP. Short, and the motivation section is still the clearest statement of
  the problem.
- **Gage (1994), *A New Algorithm for Data Compression*.** The original. Worth
  two minutes to see that nothing about the method is specific to language.
- **Karpathy, `minbpe`.** A readable reference implementation with the
  training loop and the regex pre-tokenizer separated out. The closest thing
  to this lesson's §E in a maintained repository.
- **The `tiktoken` source.** Read `_educational.py` rather than the Rust core;
  it exists precisely to show the algorithm without the performance work.
- **Kudo & Richardson (2018), *SentencePiece*.** For the alternative design:
  no pre-tokenization, whitespace as a first-class symbol, unigram vocabularies.

*Links deliberately omitted; all five are findable by title, and a dead link
in a lesson is worse than no link.*

## K · Extension

**Measure your own documents.** Twenty minutes, no account, no API key.

Take a real sample of the text your system will actually handle — not sample
data, the real thing — and produce the table from §G for it. Split it by
language, by document type, and by whichever formats you serialize into. You
now have a conversion factor per class of input, which will predict your costs
and your truncation failures better than any general advice.

Two questions to answer while you are there: what is the worst
tokens-per-character you can find in your own corpus, and what would the
context budget have to be for that document to fit?

**If you have an API key** (optional, costs pennies): send the same document
through a real endpoint and compare the token count the provider reports
against the count you computed locally. They should match exactly. When they
do not, the interesting part is *why* — usually a chat template adding
role markers and separators you did not count. That gap is the subject of
lesson 1.3.
