"""Render every figure the lessons embed, from the same code the prose quotes.

    python tools/figures.py            # write docs/assets/generated/figures/
    python tools/figures.py --check    # gate 22: re-render and diff, write nothing

Figures are generated rather than drawn, for the same reason gate 18 exists for
numbers: a hand-drawn figure showing a power curve is a claim nobody re-checks,
while a rendered one is the output of the experiment it illustrates and cannot
disagree with the table beside it. Each figure renders twice, light and dark,
and the lesson embeds both with `.fig-light` / `.fig-dark` classes so the
site's theme toggle picks the right one.

Determinism matters because gate 22 compares bytes: `svg.hashsalt` pins the
ids matplotlib would otherwise randomise per process, and `metadata` strips
the creation date it would otherwise stamp into every file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Deterministic SVG ids: without a fixed hashsalt matplotlib randomises the
# clip-path ids on every run, so identical figures produce different bytes and
# both gate 22 and git diffs churn forever.
matplotlib.rcParams["svg.hashsalt"] = "llmds"
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "assets" / "generated" / "figures"

THEMES = {
    "light": {"fg": "#1a1a1a", "bg": "none", "grid": "#d0d0d0",
              "paired": "#00796b", "unpaired": "#c62828", "accent": "#5e35b1"},
    "dark": {"fg": "#e0e0e0", "bg": "none", "grid": "#454545",
             "paired": "#4db6ac", "unpaired": "#ef9a9a", "accent": "#b39ddb"},
}


def _style(ax, theme: dict) -> None:
    ax.set_facecolor(theme["bg"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(theme["fg"])
    ax.tick_params(colors=theme["fg"], labelsize=9)
    ax.xaxis.label.set_color(theme["fg"])
    ax.yaxis.label.set_color(theme["fg"])
    ax.title.set_color(theme["fg"])
    ax.grid(True, color=theme["grid"], linewidth=0.5, alpha=0.6)


def _save(fig, name: str, variant: str, outdir: Path) -> None:
    fig.savefig(
        outdir / f"{name}-{variant}.svg",
        format="svg",
        bbox_inches="tight",
        transparent=True,
        # The default metadata includes a creation date, which would make
        # byte-identical re-rendering impossible.
        metadata={"Date": None, "Creator": "tools/figures.py"},
    )
    plt.close(fig)


# --- figures ----------------------------------------------------------------


def fig_eval_power(outdir: Path) -> None:
    """Power vs n for the paired and unpaired tests, at two values of rho.

    The lesson's table gives four points of this picture; the figure shows the
    whole shape, and in particular that the paired curve pulls away *faster*
    as the two systems being compared become more alike — which is the
    counterintuitive half of lesson 0.3's argument.
    """
    from experiments.eval_power import power

    ns = [50, 100, 200, 300, 500, 750, 1000]
    rhos = [(0.5, "-"), (0.9, "--")]
    curves: dict[tuple[float, str], list[float]] = {}
    for rho, _ in rhos:
        for test in ("paired", "unpaired"):
            curves[(rho, test)] = [
                100 * power(n, 0.80, 0.85, trials=3000, rho=rho)[test] for n in ns
            ]

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        for rho, dash in rhos:
            ax.plot(ns, curves[(rho, "paired")], dash, color=theme["paired"],
                    linewidth=2, label=f"paired, rho={rho}")
            ax.plot(ns, curves[(rho, "unpaired")], dash, color=theme["unpaired"],
                    linewidth=2, label=f"unpaired, rho={rho}")
        ax.axhline(80, color=theme["accent"], linewidth=1, alpha=0.8)
        ax.annotate("80% power", (ns[0], 81), color=theme["accent"], fontsize=8)
        ax.set_xlabel("items in the evaluation set (n)")
        ax.set_ylabel("chance of detecting a real 80% → 85% improvement")
        ax.set_title("The paired test pulls further ahead the more alike the systems are")
        ax.set_ylim(0, 102)
        ax.legend(loc="lower right", fontsize=8, frameon=False,
                  labelcolor=theme["fg"])
        _save(fig, "eval-power", variant, outdir)


def fig_aggregate_masking(outdir: Path) -> None:
    """Aggregate success rate against the share of traffic that is broken.

    The table in lesson 0.5 gives four rows; the figure shows the whole curve
    and, more usefully, shades the region where a completely dead subgroup
    still cannot pull the aggregate under the dashboard floor. That region is
    the reason the incident lasts for weeks.
    """
    from experiments.aggregate_masking import FLOOR, HEALTHY

    shares = [i / 200 for i in range(1, 81)]  # 0.5% .. 40%
    dead = [100 * (1 - s) * HEALTHY for s in shares]
    half = [100 * ((1 - s) * HEALTHY + s * 0.45) for s in shares]
    cutoff = 100 * (HEALTHY - FLOOR) / HEALTHY

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        xs = [100 * s for s in shares]
        ax.plot(xs, dead, color=theme["unpaired"], linewidth=2,
                label="the slice fails every request")
        ax.plot(xs, half, "--", color=theme["paired"], linewidth=2,
                label="the slice succeeds 45% of the time")
        ax.axhline(100 * FLOOR, color=theme["accent"], linewidth=1.2)
        ax.annotate(f"dashboard floor, {100 * FLOOR:.0f}%", (26, 100 * FLOOR + 0.4),
                    color=theme["accent"], fontsize=8)
        ax.axvspan(0, cutoff, color=theme["accent"], alpha=0.12)
        ax.annotate("no alert\npossible", (cutoff / 2, 78), color=theme["fg"],
                    fontsize=8, ha="center")
        ax.set_xlabel("share of traffic in the broken slice (%)")
        ax.set_ylabel("aggregate success rate (%)")
        ax.set_title("Below 5.3% of traffic, a totally broken slice cannot breach the floor")
        ax.set_xlim(0, 40)
        ax.set_ylim(60, 97)
        ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=theme["fg"])
        _save(fig, "aggregate-masking", variant, outdir)


def fig_nucleus_temperature(outdir: Path) -> None:
    """How many tokens top-p keeps, against temperature, for three contexts.

    A log axis is doing real work here: the count spans four orders of
    magnitude, which is precisely the point the lesson makes and precisely
    what a table of four numbers cannot show.
    """
    from experiments.sampling_shape import ALPHAS, logits_from, nucleus_size, softmax, zipf_probs

    temps = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
    labels = {0.8: "wide open", 1.2: "ordinary prose", 2.0: "nearly determined"}
    curves = {}
    for alpha in ALPHAS:
        logits = logits_from(zipf_probs(alpha))
        curves[alpha] = [nucleus_size(softmax(logits, t), 0.90) for t in temps]

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        colours = [theme["unpaired"], theme["accent"], theme["paired"]]
        for alpha, colour in zip(ALPHAS, colours, strict=True):
            ax.plot(temps, curves[alpha], "o-", color=colour, linewidth=2,
                    markersize=4, label=labels[alpha])
        ax.set_yscale("log")
        ax.set_xlabel("temperature")
        ax.set_ylabel("tokens kept by top-p = 0.90  (log scale)")
        ax.set_title("'top_p = 0.9' is a policy, not a budget")
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=theme["fg"])
        _save(fig, "nucleus-temperature", variant, outdir)


def fig_conversation_cost(outdir: Path) -> None:
    """Cumulative input tokens billed over a conversation, against turn count."""
    from experiments.chat_overhead import compute

    v = compute()
    first, per = v["conversation_total_tokens"] / v["requests"], v["per_exchange_tokens"]
    first = v["final_prompt_tokens"] - (v["requests"] - 1) * per
    turns = list(range(1, 21))
    cumulative = []
    running = 0.0
    for i in turns:
        running += first + (i - 1) * per
        cumulative.append(running)
    stateless = [first * i for i in turns]

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        ax.plot(turns, cumulative, color=theme["unpaired"], linewidth=2,
                label="stateless API: every turn re-sends the history")
        ax.plot(turns, stateless, "--", color=theme["paired"], linewidth=2,
                label="if the model remembered")
        ax.fill_between(turns, stateless, cumulative, color=theme["unpaired"], alpha=0.12)
        ax.annotate("what the resend costs you", (11, 6200), color=theme["fg"], fontsize=8)
        ax.set_xlabel("turn")
        ax.set_ylabel("cumulative input tokens billed")
        ax.set_title("Conversation cost is quadratic, and the last request hides it")
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=theme["fg"])
        _save(fig, "conversation-cost", variant, outdir)


def fig_design_costs(outdir: Path) -> None:
    """Cost per query against corpus size, for the three architectures."""
    from experiments.design_costs import (
        WINDOW_TOKENS,
        long_context,
        map_reduce,
        retrieval,
    )

    ns = [10, 25, 50, 100, 250, 500, 1000, 2500]
    series = {"long context": long_context, "retrieval": retrieval, "map-reduce": map_reduce}
    ceiling = max(n for n in range(1, 5000) if long_context(n)["fits"])

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        colours = [theme["accent"], theme["paired"], theme["unpaired"]]
        for (label, fn), colour in zip(series.items(), colours, strict=True):
            ax.plot(ns, [fn(n)["cost"] for n in ns], "o-", color=colour,
                    linewidth=2, markersize=4, label=label)
        ax.axvline(ceiling, color=theme["fg"], linewidth=1, alpha=0.5)
        ax.annotate(f"long context stops\nfitting the window\n(N = {ceiling})",
                    (ceiling * 1.15, 4000), color=theme["fg"], fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("corpus size, N chunks  (log scale)")
        ax.set_ylabel(f"cost per query, input-token units  (log scale)\n"
                      f"window = {WINDOW_TOKENS:,} tokens")
        ax.set_title("Only retrieval is flat in the size of the corpus")
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=theme["fg"])
        _save(fig, "design-costs", variant, outdir)


def fig_fewshot_selection(outdir: Path) -> None:
    """How often the query's own label appears among the chosen examples, vs k.

    The crossing is the finding: below k = number of labels, forcing balance
    actively hurts the metric that matters most, because round-robin spends
    scarce slots on labels the query has nothing to do with.
    """
    import json as _json
    import random as _random

    import numpy as _np

    from experiments.fewshot_selection import POOL, SEED, STRATEGIES, tfidf

    rows = [_json.loads(line) for line in POOL.read_text(encoding="utf-8").splitlines()]
    labels = [r["label"] for r in rows]
    vectors, _ = tfidf([r["text"] for r in rows])
    pairwise = vectors @ vectors.T
    n_labels = len(set(labels))
    ks = [1, 2, 3, 4, 5, 6, 8, 10]

    curves: dict[str, list[float]] = {}
    for name, fn in STRATEGIES.items():
        series = []
        for k in ks:
            rng = _random.Random(SEED)
            hit = []
            for q in range(len(rows)):
                mask = [i for i in range(len(rows)) if i != q]
                kwargs = ({"pairwise": pairwise[_np.ix_(mask, mask)]}
                          if name == "mmr" else {})
                picked = fn(pairwise[q, mask], [labels[i] for i in mask], k, rng, **kwargs)
                hit.append(labels[q] in {labels[mask[i]] for i in picked})
            series.append(100 * float(_np.mean(hit)))
        curves[name] = series

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        styles = {"random": ("-", theme["fg"]), "top_k": ("-", theme["unpaired"]),
                  "mmr": ("-", theme["accent"]), "balanced": ("--", theme["paired"])}
        for name, series in curves.items():
            dash, colour = styles[name]
            ax.plot(ks, series, dash, color=colour, linewidth=2, marker="o",
                    markersize=4, label=name)
        ax.axvline(n_labels, color=theme["fg"], linewidth=1, alpha=0.5)
        ax.annotate(f"k = number of labels ({n_labels})", (n_labels + 0.15, 22),
                    color=theme["fg"], fontsize=8)
        ax.set_xlabel("examples selected (k)")
        ax.set_ylabel("prompts containing an example of the query's own label (%)")
        ax.set_title("Forcing label balance backfires below k = number of labels")
        ax.set_ylim(0, 105)
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=theme["fg"])
        _save(fig, "fewshot-selection", variant, outdir)


def fig_self_consistency(outdir: Path) -> None:
    """Majority-vote accuracy against k, for several per-sample accuracies.

    The 50% line is the point of the figure: above it the curves climb, below
    it they fall, and the curve that starts below never recovers however many
    samples you buy.
    """
    from experiments.self_consistency import majority_correct

    ks = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    ps = [(0.40, "40% per sample"), (0.55, "55%"), (0.70, "70%"), (0.85, "85%")]

    for variant, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        _style(ax, theme)
        colours = [theme["unpaired"], theme["fg"], theme["accent"], theme["paired"]]
        for (p, label), colour in zip(ps, colours, strict=True):
            ys = [100 * majority_correct(p, k) for k in ks]
            ax.plot(ks, ys, "o-", color=colour, linewidth=2, markersize=4, label=label)
        ax.axhline(50, color=theme["fg"], linewidth=1, linestyle=":", alpha=0.7)
        ax.annotate("50%: below this line, voting makes it worse",
                    (2, 52), color=theme["fg"], fontsize=8)
        ax.set_xlabel("samples voted over (k)")
        ax.set_ylabel("accuracy of the majority answer (%)")
        ax.set_title("Self-consistency amplifies whatever the model already was")
        ax.set_ylim(0, 104)
        ax.legend(loc="center right", fontsize=8, frameon=False, labelcolor=theme["fg"])
        _save(fig, "self-consistency", variant, outdir)


FIGURES = [
    fig_eval_power,
    fig_fewshot_selection,
    fig_self_consistency,
    fig_aggregate_masking,
    fig_nucleus_temperature,
    fig_conversation_cost,
    fig_design_costs,
]


def render_all(outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(outdir)
    return sorted(p.name for p in outdir.glob("*.svg"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="re-render into a scratch dir and diff against the "
                         "committed SVGs; write nothing")
    args = ap.parse_args()

    if not args.check:
        names = render_all(OUT)
        print(f"rendered {len(names)} SVG(s) into {OUT.relative_to(ROOT)}")
        return 0

    # Gate 22. A figure that drifts from the code that produced it is the
    # pictorial version of the stale number gate 18 catches.
    import tempfile

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        fresh = render_all(scratch)
        committed = sorted(p.name for p in OUT.glob("*.svg")) if OUT.exists() else []
        for name in fresh:
            if name not in committed:
                problems.append(f"{name}: rendered by tools/figures.py but not committed")
                continue
            if (scratch / name).read_bytes() != (OUT / name).read_bytes():
                problems.append(
                    f"{name}: committed SVG differs from a fresh render - the figure "
                    f"has drifted from the code that produces it; run tools/figures.py")
        for name in committed:
            if name not in fresh:
                problems.append(f"{name}: committed but no longer rendered - delete it "
                                f"or restore its figure function")

    print(f"figures checked: {len(fresh)}")
    if problems:
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
