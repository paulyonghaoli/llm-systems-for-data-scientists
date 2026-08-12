"""What temperature, top-k and top-p actually do to a next-token distribution.

    python experiments/sampling_shape.py
    python experiments/sampling_shape.py --json

## The distribution these numbers are computed on

There is no model here, so the next-token distribution is synthetic — and the
choice of synthetic distribution is the claim, so it is stated rather than
buried. Next-token probabilities over a large vocabulary are approximately
Zipfian, `p_i ∝ i^-alpha`, and `alpha` is what changes between contexts:

| alpha | The context feels | Top-1 probability |
|---|---|---|
| 0.8 | wide open — many continuations plausible | low |
| 1.2 | ordinary prose | moderate |
| 2.0 | nearly determined — a closing bracket, a common idiom | high |

Every result below is reported across all three rather than at one setting,
because the interesting behaviour is precisely that these knobs do different
things depending on how confident the model already was. A single number here
would be a number about `alpha`, not about sampling.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

SEED = 20260809
VOCAB = 50257
ALPHAS = (0.8, 1.2, 2.0)
DRAWS = 20000


def zipf_probs(alpha: float, vocab: int = VOCAB) -> np.ndarray:
    ranks = np.arange(1, vocab + 1, dtype=np.float64)
    w = ranks ** (-alpha)
    return w / w.sum()


def logits_from(probs: np.ndarray) -> np.ndarray:
    return np.log(probs)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def nucleus_size(probs: np.ndarray, p: float) -> int:
    """How many tokens top-p actually keeps."""
    s = np.sort(probs)[::-1]
    return int(np.searchsorted(np.cumsum(s), p) + 1)


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    keep = order[: int(np.searchsorted(cum, p) + 1)]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def entropy_bits(probs: np.ndarray) -> float:
    q = probs[probs > 0]
    return float(-(q * np.log2(q)).sum())


def repetition_penalty(logits: np.ndarray, token: int, penalty: float,
                       sign_aware: bool = True) -> np.ndarray:
    """Penalise an already-seen token.

    The sign-aware branch is the correct one. Dividing unconditionally is the
    well-known bug: for a token whose logit is negative — which is almost all
    of them — dividing by a penalty greater than 1 moves the logit *towards
    zero*, making the token more likely rather than less.
    """
    out = logits.copy()
    score = out[token]
    if sign_aware:
        out[token] = score / penalty if score > 0 else score * penalty
    else:
        out[token] = score / penalty
    return out


def compute() -> dict[str, float]:
    out: dict[str, float] = {}
    rng = np.random.default_rng(SEED)

    for alpha in ALPHAS:
        tag = str(alpha).replace(".", "")
        probs = zipf_probs(alpha)
        logits = logits_from(probs)

        out[f"top1_a{tag}_pct"] = round(100 * float(probs.max()), 2)
        out[f"entropy_a{tag}_bits"] = round(entropy_bits(probs), 2)

        # 1. How many tokens does top-p keep? Not a fixed number.
        for p in (0.90, 0.95):
            ptag = str(int(p * 100))
            out[f"nucleus_p{ptag}_a{tag}"] = nucleus_size(probs, p)

        # 2. Temperature changes the nucleus size dramatically.
        for t in (0.5, 1.0, 1.5):
            ttag = str(t).replace(".", "")
            out[f"nucleus_p90_t{ttag}_a{tag}"] = nucleus_size(softmax(logits, t), 0.90)

        # 3. Order of operations. Temperature-then-top-p is not the same
        #    distribution as top-p-then-temperature.
        t = 0.7
        a = top_p_filter(softmax(logits, t), 0.90)
        pre = top_p_filter(probs, 0.90)
        b = softmax(np.where(pre > 0, logits, -np.inf), t)
        out[f"order_tv_distance_a{tag}"] = round(0.5 * float(np.abs(a - b).sum()), 3)
        out[f"order_nucleus_temp_first_a{tag}"] = int((a > 0).sum())
        out[f"order_nucleus_topp_first_a{tag}"] = int((b > 0).sum())

    # 4. The repetition-penalty sign bug, on the ordinary-prose distribution.
    probs = zipf_probs(1.2)
    logits = logits_from(probs)
    token = 500  # an unremarkable token with a negative logit
    base = float(softmax(logits)[token])
    correct = float(softmax(repetition_penalty(logits, token, 1.2, True))[token])
    buggy = float(softmax(repetition_penalty(logits, token, 1.2, False))[token])
    out["penalty_logit_value"] = round(float(logits[token]), 2)
    out["penalty_base_prob_ppm"] = round(1e6 * base, 1)
    out["penalty_correct_prob_ppm"] = round(1e6 * correct, 1)
    out["penalty_buggy_prob_ppm"] = round(1e6 * buggy, 1)
    out["penalty_buggy_ratio"] = round(buggy / base, 2)

    # 5. Greedy is not "temperature 0 sampling": at low but non-zero
    #    temperature the argmax is still usually chosen, but not always.
    probs = zipf_probs(1.2)
    logits = logits_from(probs)
    argmax = int(np.argmax(probs))
    for t in (0.1, 0.3, 0.7, 1.0):
        ttag = str(t).replace(".", "")
        q = softmax(logits, t)
        draws = rng.choice(VOCAB, size=DRAWS, p=q)
        out[f"argmax_rate_t{ttag}_pct"] = round(100 * float((draws == argmax).mean()), 1)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    v = compute()
    if args.json:
        print(json.dumps(v, indent=1))
        return 0

    print(f"{'alpha':>6} {'top-1':>8} {'entropy':>9} {'nucleus@.90':>12} {'@.95':>7} "
          f"{'T=0.5':>7} {'T=1.5':>7}")
    for alpha in ALPHAS:
        t = str(alpha).replace(".", "")
        print(f"{alpha:>6} {v[f'top1_a{t}_pct']:>7.2f}% {v[f'entropy_a{t}_bits']:>8.2f}b "
              f"{v[f'nucleus_p90_a{t}']:>12.0f} {v[f'nucleus_p95_a{t}']:>7.0f} "
              f"{v[f'nucleus_p90_t05_a{t}']:>7.0f} {v[f'nucleus_p90_t15_a{t}']:>7.0f}")

    print("\nOrder of operations at T=0.7, top-p=0.90:")
    for alpha in ALPHAS:
        t = str(alpha).replace(".", "")
        print(f"  alpha={alpha}: temperature first keeps "
              f"{v[f'order_nucleus_temp_first_a{t}']:.0f} tokens, top-p first keeps "
              f"{v[f'order_nucleus_topp_first_a{t}']:.0f}; "
              f"total-variation distance {v[f'order_tv_distance_a{t}']}")

    print(f"\nRepetition penalty 1.2 on a token with logit "
          f"{v['penalty_logit_value']}:")
    print(f"  before          {v['penalty_base_prob_ppm']:>8.1f} ppm")
    print(f"  sign-aware      {v['penalty_correct_prob_ppm']:>8.1f} ppm  (down, as intended)")
    print(f"  divide-always   {v['penalty_buggy_prob_ppm']:>8.1f} ppm  "
          f"({v['penalty_buggy_ratio']}x - the penalty made it MORE likely)")

    print("\nHow often the argmax is drawn, by temperature:")
    for t in (0.1, 0.3, 0.7, 1.0):
        tag = str(t).replace(".", "")
        print(f"  T={t}: {v[f'argmax_rate_t{tag}_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
