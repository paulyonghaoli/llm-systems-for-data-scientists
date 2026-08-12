"""MkDocs build hook: convert curriculum YAML -> docs/assets/generated JSON.

Registered via `hooks:` in mkdocs.yml. Fails the build on invalid content, so
a broken quiz or exercise can never ship.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.content_lib import export_json, load_all  # noqa: E402

DOCS = ROOT / "docs"


def _h1_of(path: Path) -> str | None:
    if not path.exists():
        return None
    m = re.search(r"^#\s+(.+)$", path.read_text(encoding="utf-8"), re.M)
    return m.group(1).strip() if m else None


def _write_lesson_manifest() -> int:
    """List every study page, for the progress summary's denominator.

    Derived from disk rather than from the nav so it runs in `on_pre_build`,
    before MkDocs has built one. Titles come from the H1 so the summary shows
    the same name the page does.
    """
    modules_dir = DOCS / "modules"
    if not modules_dir.exists():
        return 0
    module_names = {
        d.name: (_h1_of(d / "index.md") or d.name)
        for d in sorted(modules_dir.iterdir())
        if d.is_dir()
    }

    out = []
    for md in sorted(modules_dir.rglob("*.md")):
        rel = md.relative_to(DOCS).as_posix()
        # Relative to the site root, not the domain root: progress.js
        # prefixes whatever base it is actually served from.
        url = rel[: -len("index.md")] if rel.endswith("index.md") else rel[: -len(".md")] + "/"
        out.append({
            "url": url,
            "title": _h1_of(md) or md.stem,
            "module": module_names.get(md.parent.name, md.parent.name),
            "slug": md.parent.name,
        })
    target = DOCS / "assets" / "generated" / "lessons.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=1)
    if not target.exists() or target.read_text(encoding="utf-8") != payload:
        target.write_text(payload, encoding="utf-8")
    return len(out)


def on_pre_build(config, **kwargs):  # noqa: ANN001, ARG001
    cs = load_all()
    if cs.errors:
        details = "\n".join(f"  - {e}" for e in cs.errors)
        raise SystemExit(f"interactive content invalid:\n{details}")
    export_json(cs)
    _write_lesson_manifest()
