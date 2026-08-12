"""Flip the site from soft launch (unindexed) to public, or back.

    python tools/launch.py --status
    python tools/launch.py --go        # allow indexing
    python tools/launch.py --unlaunch  # restore the noindex guards

Two guards keep the site out of search results during soft launch:
`docs/robots.txt` (disallow all) and a `noindex, nofollow` meta tag injected by
`overrides/main.html`. This script toggles both together so they cannot drift
out of sync, which is the failure the sibling robotics repo hit first.

It does NOT change repository visibility anywhere — that stays a deliberate
manual decision, and the project's framing constraint is that this is
self-study shared with peers rather than anything to be promoted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROBOTS = ROOT / "docs" / "robots.txt"
OVERRIDE = ROOT / "overrides" / "main.html"

SITE = "https://llm-systems-for-data-scientists.paullimale.workers.dev"

ROBOTS_BLOCKED = "User-agent: *\nDisallow: /\n"
ROBOTS_OPEN = f"User-agent: *\nAllow: /\n\nSitemap: {SITE}/sitemap.xml\n"

META_LINE = '  <meta name="robots" content="noindex, nofollow">'
OVERRIDE_TEMPLATE = """{{% extends "base.html" %}}

{{#
  Soft-launch guard. `python tools/launch.py --go` removes the noindex tag;
  `--unlaunch` restores it. Do not edit by hand — the script keeps this in
  sync with docs/robots.txt.
#}}
{{% block extrahead %}}
{meta}
{{% endblock %}}
"""


def is_blocked() -> bool:
    # Match the actual meta tag rather than the word "noindex", because the
    # template's own explanatory comment mentions it — a substring check on
    # the bare word always reports "blocked".
    return 'content="noindex' in OVERRIDE.read_text(encoding="utf-8")


def apply(blocked: bool) -> None:
    ROBOTS.write_text(ROBOTS_BLOCKED if blocked else ROBOTS_OPEN, encoding="utf-8")
    meta = META_LINE if blocked else "  {# indexing allowed #}"
    OVERRIDE.write_text(OVERRIDE_TEMPLATE.format(meta=meta), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--go", action="store_true", help="allow search indexing")
    g.add_argument("--unlaunch", action="store_true", help="restore noindex")
    args = ap.parse_args()

    if args.status:
        state = "SOFT LAUNCH (not indexed)" if is_blocked() else "PUBLIC (indexed)"
        print(f"site indexing: {state}")
        print(f"  {ROBOTS.relative_to(ROOT)}: "
              f"{ROBOTS.read_text(encoding='utf-8').strip().splitlines()[1]}")
        return 0

    apply(blocked=bool(args.unlaunch))
    print("site indexing:",
          "restored to SOFT LAUNCH" if args.unlaunch else "OPENED to search engines")
    print("\nnext:")
    print("  python tools/verify.py && .venv/Scripts/mkdocs build && npx wrangler deploy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
