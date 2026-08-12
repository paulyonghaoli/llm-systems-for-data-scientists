"""How much does a tokenizer trained on 11 KB of one person's prose actually buy?

    python experiments/bpe_compression.py           # human-readable table
    python experiments/bpe_compression.py --json    # machine-readable, for gate 18

Trains the reference BPE on data/corpus.txt and measures tokens-per-character
on five held-out samples, alongside the pinned real-tokenizer counts recorded
by experiments/record_tiktoken.py.

Every number quoted in lesson 1.1 comes from here. `tools/gate_numbers.py`
re-runs this and checks the lesson still agrees with it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmlab.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.txt"
SAMPLES = ROOT / "data" / "samples"
FIXTURE = ROOT / "data" / "fixtures" / "tiktoken_counts.json"

VOCAB_SIZES = [512, 1024]
MAIN_VOCAB = 512


def compute() -> dict[str, float]:
    corpus = CORPUS.read_text(encoding="utf-8")
    samples = {p.stem: p.read_text(encoding="utf-8") for p in sorted(SAMPLES.glob("*.txt"))}
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    out: dict[str, float] = {
        "corpus_chars": len(corpus),
        "corpus_bytes": len(corpus.encode("utf-8")),
        "corpus_kb": round(len(corpus.encode("utf-8")) / 1024, 1),
    }

    tokenizers: dict[int, BPETokenizer] = {}
    for size in VOCAB_SIZES:
        tok = BPETokenizer.train(corpus, vocab_size=size)
        tokenizers[size] = tok
        out[f"merges_v{size}"] = len(tok.merges)
        out[f"corpus_tpc_v{size}"] = round(tok.count(corpus) / len(corpus), 3)

    tok = tokenizers[MAIN_VOCAB]

    for stem, text in samples.items():
        chars = len(text)
        out[f"chars_{stem}"] = chars
        out[f"bytes_{stem}"] = len(text.encode("utf-8"))
        out[f"bpc_{stem}"] = round(len(text.encode("utf-8")) / chars, 3)

        ours = tok.count(text)
        out[f"tok_{stem}_ours"] = ours
        out[f"tpc_{stem}_ours"] = round(ours / chars, 3)

        for enc in ("cl100k_base", "o200k_base"):
            n = fixture["encodings"][enc]["counts"][stem]
            short = enc.split("_")[0]
            out[f"tok_{stem}_{short}"] = n
            out[f"tpc_{stem}_{short}"] = round(n / chars, 3)

        out[f"ratio_{stem}_ours_over_cl100k"] = round(ours / fixture["encodings"]
                                                      ["cl100k_base"]["counts"][stem], 2)

    # The headline comparisons the lesson leans on.
    out["english_over_japanese_cl100k"] = round(
        out["tpc_japanese_cl100k"] / out["tpc_english_cl100k"], 2)
    out["japanese_saving_o200k_pct"] = round(
        100 * (1 - out["tok_japanese_o200k"] / out["tok_japanese_cl100k"]), 1)
    # Serialization cost, measured the only way that means anything: the SAME
    # seven records, as a tab-separated table and as pretty-printed JSON.
    #
    # The first version of this experiment compared tokens-per-character
    # between json.txt and numbers.txt and "showed" JSON was 27% cheaper.
    # Those two files carry different information, so the comparison was
    # meaningless. Per character JSON is cheap (repetitive English-ish keys);
    # per unit of information it is not. Lesson 1.1 §H keeps the mistake.
    out["json_vs_tsv_same_data_cl100k"] = round(
        out["tok_numbers_json_cl100k"] / out["tok_numbers_cl100k"], 2)
    out["json_vs_tsv_extra_tokens"] = out["tok_numbers_json_cl100k"] - out["tok_numbers_cl100k"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    values = compute()
    if args.json:
        print(json.dumps(values, indent=1))
        return 0

    print(f"corpus: {values['corpus_chars']} chars / {values['corpus_bytes']} bytes "
          f"({values['corpus_kb']} KB)")
    for size in VOCAB_SIZES:
        print(f"  vocab {size}: {values[f'merges_v{size}']:.0f} merges learned, "
              f"{values[f'corpus_tpc_v{size}']} tokens/char on the training corpus")
    print()
    head = f"{'sample':<13} {'chars':>6} {'bytes/ch':>9} {'ours':>7} {'cl100k':>8} {'o200k':>7}"
    print(head)
    print("-" * len(head))
    for stem in ("english", "python", "json", "numbers", "numbers_json", "japanese"):
        print(f"{stem:<13} {values[f'chars_{stem}']:>6.0f} {values[f'bpc_{stem}']:>9.3f} "
              f"{values[f'tpc_{stem}_ours']:>7.3f} {values[f'tpc_{stem}_cl100k']:>8.3f} "
              f"{values[f'tpc_{stem}_o200k']:>7.3f}")
    print("\n(figures after 'bytes/ch' are tokens per character; lower is cheaper)")
    print(f"same 7 records as JSON vs TSV, cl100k: "
          f"{values['tok_numbers_json_cl100k']:.0f} vs {values['tok_numbers_cl100k']:.0f} tokens "
          f"= {values['json_vs_tsv_same_data_cl100k']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
