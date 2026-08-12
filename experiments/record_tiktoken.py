"""Record token counts from a real production tokenizer. RUN BY HAND ONLY.

    pip install -e ".[record]"
    python experiments/record_tiktoken.py

Writes data/fixtures/tiktoken_counts.json — a dated, version-pinned fixture.
Everything downstream reads the fixture, never tiktoken, so the whole test
suite runs offline (gate 13) and the numbers in the lessons cannot silently
change under us.

This is the same pattern as the cassettes described in PLAN.md §5a: capture
real behaviour once, with a date and a version stamped on it, and replay it
forever. Re-recording is a content event, not maintenance — if a number moves,
the diff belongs in the lesson.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "data" / "fixtures" / "tiktoken_counts.json"

ENCODINGS = ["cl100k_base", "o200k_base"]

#: A ChatML-shaped wrapper. Providers differ in the exact markers; what is
#: stable is that every message is wrapped in *something*, and that the wrapper
#: is billed. Recorded as ordinary text, because these markers are not special
#: tokens in either encoding below — see lesson 1.3 §F for what changes when
#: they are.
TURN_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"
GENERATION_PROMPT = "<|im_start|>assistant\n"

CONVERSATION = [
    ("system", "You are a support assistant for a logistics company. Answer only "
               "from the documents provided, cite the record id, and say when you "
               "do not know."),
    ("user", "My shipment TL-4471 was due on Tuesday and has not arrived. Can you "
             "tell me where it is?"),
    ("assistant", "Record TL-4471 shows the consignment left the north-east depot "
                  "on Monday evening and is currently marked in transit. The "
                  "delivery estimate has moved to Thursday."),
    ("user", "Why did it move? Nobody told me."),
    ("assistant", "The record notes an address verification hold placed on Tuesday "
                  "morning, which paused the consignment for one working day. "
                  "Notifications are only sent when a hold exceeds 48 hours."),
    ("user", "Can you release the hold and get it out today?"),
    ("assistant", "I cannot release a hold from this system. The depot team owns "
                  "that action, and the record shows the verification was already "
                  "completed at 09:14 on Wednesday."),
    ("user", "Fine. What is the current estimate, and can I get a refund on the "
             "shipping fee?"),
]


def _render(messages: list[tuple[str, str]]) -> str:
    body = "".join(TURN_TEMPLATE.format(role=r, content=c) for r, c in messages)
    return body + GENERATION_PROMPT


def _chat_overhead(enc) -> dict:  # noqa: ANN001
    """Token counts for a growing conversation, wrapped and unwrapped."""
    wrapped, bare = [], []
    for k in range(1, len(CONVERSATION) + 1):
        prefix = CONVERSATION[:k]
        wrapped.append(len(enc.encode(_render(prefix))))
        bare.append(len(enc.encode("".join(c for _, c in prefix))))
    return {
        "turns": len(CONVERSATION),
        "roles": [r for r, _ in CONVERSATION],
        "wrapped_tokens": wrapped,
        "bare_tokens": bare,
    }


def main() -> int:
    try:
        import tiktoken
    except ImportError:
        print('tiktoken is not installed. `pip install -e ".[record]"` first.', file=sys.stderr)
        return 2

    payload: dict[str, object] = {
        "recorded_on": date.today().isoformat(),
        "tiktoken_version": tiktoken.__version__,
        "note": (
            "Token counts for the held-out samples in data/samples/, from real "
            "production tokenizers. Regenerate with experiments/record_tiktoken.py; "
            "never called at test time."
        ),
        "encodings": {},
    }

    samples = {p.stem: p.read_text(encoding="utf-8") for p in sorted(SAMPLES.glob("*.txt"))}

    for name in ENCODINGS:
        enc = tiktoken.get_encoding(name)
        payload["encodings"][name] = {  # type: ignore[index]
            "n_vocab": enc.n_vocab,
            "counts": {stem: len(enc.encode(text)) for stem, text in samples.items()},
            "chat": _chat_overhead(enc),
        }
        print(f"{name}: n_vocab={enc.n_vocab}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
