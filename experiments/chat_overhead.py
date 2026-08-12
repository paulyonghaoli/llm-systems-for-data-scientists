"""What a conversation costs, and why it is not what you sent.

    python experiments/chat_overhead.py
    python experiments/chat_overhead.py --json

Two things nobody counts:

1. **The wrapper.** Messages are not sent as text; they are sent inside a chat
   template with role markers around every turn. Those markers are tokens and
   they are billed.
2. **The resend.** A stateless API has no memory, so turn N re-sends turns
   1..N-1. Total spend over a conversation is therefore **quadratic** in its
   length, not linear, and the last request is the cheapest way to
   under-estimate it.

Counts come from `data/fixtures/tiktoken_counts.json`, recorded from a real
tokenizer by `experiments/record_tiktoken.py` on a stated date. The
extrapolation to longer conversations is fitted to those measurements, and the
fit is reported so it can be judged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "data" / "fixtures" / "tiktoken_counts.json"

ENCODING = "cl100k_base"

#: Overhead per message when the role markers really are single special
#: tokens, as they are in most production templates. Used for the contrast in
#: lesson 1.3 §G; the measured figure treats them as ordinary text.
SPECIAL_TOKEN_OVERHEAD = 4

PROJECT_TURNS = 20


def compute() -> dict[str, float]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chat = fixture["encodings"][ENCODING]["chat"]
    roles = chat["roles"]
    wrapped = chat["wrapped_tokens"]
    bare = chat["bare_tokens"]
    n = len(roles)

    overhead = [w - b for w, b in zip(wrapped, bare, strict=True)]
    marginal = [overhead[i] - overhead[i - 1] for i in range(1, n)]

    out: dict[str, float] = {
        "messages": n,
        "final_prompt_tokens": wrapped[-1],
        "final_content_tokens": bare[-1],
        "final_overhead_tokens": overhead[-1],
        "overhead_share_pct": round(100 * overhead[-1] / wrapped[-1], 1),
        "overhead_per_message": round(sum(marginal) / len(marginal), 1),
        "special_token_overhead_per_message": SPECIAL_TOKEN_OVERHEAD,
        "special_token_overhead_share_pct": round(
            100 * SPECIAL_TOKEN_OVERHEAD * n / (bare[-1] + SPECIAL_TOKEN_OVERHEAD * n), 1),
    }

    # A request is sent after every user message.
    request_sizes = [wrapped[i] for i, r in enumerate(roles) if r == "user"]
    out["requests"] = len(request_sizes)
    out["conversation_total_tokens"] = sum(request_sizes)
    out["amplification_vs_final"] = round(sum(request_sizes) / wrapped[-1], 2)

    # Fit prompt_size(i) = first + (i-1) * per_exchange to the measured
    # request sizes, then project. Linear fit, so the projected total is
    # quadratic in the number of turns.
    deltas = [request_sizes[i] - request_sizes[i - 1] for i in range(1, len(request_sizes))]
    per_exchange = sum(deltas) / len(deltas)
    first = request_sizes[0]
    out["per_exchange_tokens"] = round(per_exchange, 1)

    fitted = [first + i * per_exchange for i in range(len(request_sizes))]
    out["fit_max_error_pct"] = round(
        100 * max(abs(f - a) / a for f, a in zip(fitted, request_sizes, strict=True)), 1)

    n_proj = PROJECT_TURNS
    projected_total = sum(first + i * per_exchange for i in range(n_proj))
    stateless_total = first * n_proj
    out["projected_turns"] = n_proj
    out["projected_final_prompt"] = round(first + (n_proj - 1) * per_exchange)
    out["projected_total_tokens"] = round(projected_total)
    out["projected_stateless_total"] = round(stateless_total)
    out["projected_amplification"] = round(projected_total / stateless_total, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"An {v['messages']:.0f}-message conversation, {ENCODING}:")
    print(f"  content the user and model actually wrote : "
          f"{v['final_content_tokens']:.0f} tokens")
    print(f"  chat-template wrapper                     : "
          f"{v['final_overhead_tokens']:.0f} tokens "
          f"({v['overhead_share_pct']}% of the final prompt)")
    print(f"  marginal wrapper cost per message         : "
          f"{v['overhead_per_message']} tokens as literal text, "
          f"{v['special_token_overhead_per_message']:.0f} as special tokens")

    print(f"\nBilling over the conversation ({v['requests']:.0f} requests):")
    print(f"  final prompt                : {v['final_prompt_tokens']:.0f} tokens")
    print(f"  total input actually billed : {v['conversation_total_tokens']:.0f} tokens "
          f"({v['amplification_vs_final']}x the final prompt)")

    print(f"\nProjected to {v['projected_turns']:.0f} turns "
          f"(+{v['per_exchange_tokens']} tokens per exchange, "
          f"fit error <= {v['fit_max_error_pct']}%):")
    print(f"  final prompt                : {v['projected_final_prompt']:.0f} tokens")
    print(f"  total input billed          : {v['projected_total_tokens']:.0f} tokens")
    print(f"  if each turn were stateless : {v['projected_stateless_total']:.0f} tokens "
          f"({v['projected_amplification']}x less)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
