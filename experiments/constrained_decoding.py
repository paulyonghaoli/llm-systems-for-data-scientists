"""What a logit mask actually does, position by position.

    python experiments/constrained_decoding.py
    python experiments/constrained_decoding.py --json

Constrained decoding replaces "ask for JSON and repair what comes back" with
"make invalid JSON unrepresentable". At each step the grammar says which tokens
could legally come next, everything else is set to negative infinity before the
softmax, and the sampler picks from what remains.

The interesting question is not whether it works — validity is 100% by
construction, which is not a measurement — but **how much of the output the
grammar is choosing rather than the model.** That is structural, computable
without a model, and rarely stated.

## Two simplifications, stated

**Character level, not token level.** Real decoders mask over a BPE vocabulary
where one token may span several characters, which makes the bookkeeping harder
and the conclusions identical. Working per character keeps the automaton
readable.

**Fixed key order.** The grammar below accepts exactly one ordering of the
object's keys. Real grammar-constrained decoders accept any ordering, at the
cost of a genuine parser rather than the positional matcher here. Both
enforce the same language membership; this one is simply narrower.

Neither simplification touches the measured quantity, which is the fraction of
positions where the grammar leaves the model no choice at all.
"""

from __future__ import annotations

import argparse
import json
import string

# The printable vocabulary a character-level decoder would choose among.
VOCAB = sorted(set(string.printable) - set("\x0b\x0c"))

HEX = set(string.hexdigits.lower())
DIGITS = set(string.digits)
STATUSES = ("cancelled", "fulfilled", "pending")

#: The document shape, as an alternating sequence of literals and fields.
#: A field is (name, alphabet, min_len, max_len) or (name, alternatives).
GRAMMAR: list[object] = [
    '{"record_id": "',
    ("record_id", HEX, 6, 6),
    '", "status": "',
    ("status", STATUSES),
    '", "quantity": ',
    ("quantity", DIGITS, 1, 3),
    "}",
]


def legal_next(prefix: str) -> set[str]:
    """Every character that could legally follow `prefix`.

    Walks the grammar, consuming the prefix, and returns the alphabet
    available at the position it lands on. An empty set means the document is
    complete and nothing may follow.
    """
    i = 0  # position within prefix
    for part in GRAMMAR:
        if isinstance(part, str):
            for ch in part:
                if i == len(prefix):
                    return {ch}
                if prefix[i] != ch:
                    return set()  # prefix already left the language
                i += 1
            continue

        if len(part) == 2:  # an enumeration
            _, alternatives = part
            consumed = prefix[i:]
            live = [a for a in alternatives
                    if a.startswith(consumed) or consumed.startswith(a)]
            done = [a for a in alternatives if consumed.startswith(a)]
            if done:
                i += len(max(done, key=len))
                continue
            options = {a[len(consumed)] for a in live if len(a) > len(consumed)}
            return options

        name, alphabet, lo, hi = part  # noqa: F841
        run = 0
        while i + run < len(prefix) and prefix[i + run] in alphabet and run < hi:
            run += 1
        if i + run < len(prefix):  # the field ended inside the prefix
            if run < lo:
                return set()
            i += run
            continue
        if run < lo:
            return set(alphabet)
        if run < hi:
            # Either another field character, or whatever follows the field.
            nxt = GRAMMAR[GRAMMAR.index(part) + 1]
            follow = {nxt[0]} if isinstance(nxt, str) else set()
            return set(alphabet) | follow
        i += run
    return set()


def compute() -> dict[str, float]:
    document = '{"record_id": "a1f3c9", "status": "fulfilled", "quantity": 17}'

    forced = free = 0
    sizes: list[int] = []
    for cut in range(len(document)):
        options = legal_next(document[:cut])
        sizes.append(len(options))
        if len(options) == 1:
            forced += 1
        else:
            free += 1

    out: dict[str, float] = {
        "vocab_size": len(VOCAB),
        "document_chars": len(document),
        "forced_positions": forced,
        "free_positions": free,
        "forced_pct": round(100 * forced / len(document), 1),
        "mean_options": round(sum(sizes) / len(sizes), 1),
        "mean_allowed_pct": round(100 * sum(sizes) / len(sizes) / len(VOCAB), 1),
        "validity_pct": 100.0,
    }

    # The enum is the case schema validation can only catch afterwards.
    after_status_quote = document.index('"status": "') + len('"status": "')
    out["status_first_char_options"] = len(legal_next(document[:after_status_quote]))

    # Where the model does get a say, how wide is the choice?
    free_sizes = [s for s in sizes if s > 1]
    out["mean_options_when_free"] = round(sum(free_sizes) / len(free_sizes), 1)
    out["max_options"] = max(sizes)
    out["max_options_pct"] = round(100 * max(sizes) / len(VOCAB), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"vocabulary: {v['vocab_size']:.0f} printable characters")
    print(f"document:   {v['document_chars']:.0f} characters\n")
    print(f"positions where the grammar allows exactly one character: "
          f"{v['forced_positions']:.0f} of {v['document_chars']:.0f} "
          f"({v['forced_pct']}%)")
    print(f"positions where the model has a genuine choice:          "
          f"{v['free_positions']:.0f}")
    print(f"\nmean legal characters per position: {v['mean_options']} "
          f"({v['mean_allowed_pct']}% of the vocabulary)")
    print(f"  when the model does have a choice: {v['mean_options_when_free']}")
    print(f"  widest choice anywhere:            {v['max_options']:.0f} "
          f"({v['max_options_pct']}%)")
    print(f"\nafter '\"status\": \"' the grammar allows "
          f"{v['status_first_char_options']:.0f} characters — one per permitted "
          f"value.\nAn enum is enforced during generation rather than detected "
          f"after it.")
    print(f"\nvalidity: {v['validity_pct']:.0f}% by construction, not by measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
