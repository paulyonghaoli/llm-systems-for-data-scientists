"""Gate 19 — the joins nobody else checks.

`validate_content.py` proves every exercise's solution passes and every
starter fails. That is necessary and not sufficient. This checks the seams:

  - every <code-exercise>/<quiz-bank> in a page resolves to real content
  - every exercise and bank is actually embedded by some page (no orphans)
  - every content file's `lesson:` field names a real page
  - every page is reachable from the mkdocs nav (gate 7)
  - a module index that calls a lesson "planned" does not link to it, and one
    that calls it "available" does
  - every page has an H1
  - the curriculum map's status markers match which files exist
  - every lesson declares the schema sections CONTRIBUTING.md requires

Every check here started as something checked by hand once.

    python tools/audit.py
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.content_lib import load_all  # noqa: E402

DOCS = ROOT / "docs"
EX_RE = re.compile(r'<code-exercise\s+src="([^"]+)"')
QZ_RE = re.compile(r'<quiz-bank\s+src="([^"]+)"')

#: The uniform lesson schema from PLAN.md §2. Drift is only detectable
#: because every lesson uses the same headings in the same order.
SCHEMA_SECTIONS = [
    "A · Why this matters",
    "B · Mental model",
    "C · Mechanism",
    "D · From data science to LLM systems",
    "E · Minimal implementation",
    "F · Production practice",
    "G · Experiment",
    "H · Failure modes and cost traps",
    "I · Graded practice",
    "J · Annotated references",
    "K · Extension",
]


#: Gate 21. PLAN.md §4 declares lessons of 2,500-4,000 words, and the sibling
#: robotics curriculum measured what happens without a check: a stated target
#: of 2,500-4,000 against an actual median of 1,079. Prose style is measured
#: too, because the failure there was not only length — a stream of short
#: sentences reads as notes rather than as a textbook, and nothing else in the
#: toolchain can see it.
MIN_WORDS = 2500
#: PLAN.md §4 declares a range, and a gate that enforces only its floor lets a
#: lesson sprawl. Discovered when a stripper bug undercounted 2.4 by 1,666
#: words and the padding added to "fix" it went unnoticed.
MAX_WORDS = 4000
MIN_MEAN_SENTENCE = 22
MAX_SHORT_SENTENCE_PCT = 18
SHORT_SENTENCE_WORDS = 10

#: Lessons written before gate 21 existed, awaiting the deepening pass. This
#: list may only shrink: a lesson on it that now *passes* is also an error, so
#: the entry has to be deleted rather than left to rot into a permanent
#: exemption. Empty means the backlog is cleared.
DEEPENING_BACKLOG: set[str] = set()  # cleared 2026-08-11; every lesson meets gate 21


def prose_of(md: str) -> str:
    """Lesson text with everything that is not prose removed."""
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)  # front matter
    # Fenced code, anchored to the start of a line. An unanchored `\x60{3}`
    # also matches inline code spans — a table cell describing a ```json
    # fence, for instance — and those stray markers mis-pair with the real
    # fences, silently swallowing everything in between. That undercounted
    # lesson 2.4 by several hundred words and is the same class of bug as the
    # bare `<` below: a stripper that is too eager deletes the prose it was
    # meant to measure.
    md = re.sub(r"^[ \t]*```.*?^[ \t]*```", " ", md, flags=re.S | re.M)
    md = re.sub(r"<!--.*?-->", " ", md, flags=re.S)  # computed markers
    md = re.sub(r"^\|.*$", " ", md, flags=re.M)  # tables
    md = re.sub(r"\$\$.*?\$\$", " ", md, flags=re.S)  # display maths
    # Real tags only. `<[^>]+>` would treat a bare "<" in prose — "whenever
    # s < (p_h - F) / p_h" — as the start of a tag and delete everything up to
    # the next ">", which silently removed 1,615 words from one lesson while
    # this check was still a throwaway script.
    md = re.sub(r"</?[a-zA-Z][^>\n]*>", " ", md)
    return md


def prose_stats(md: str) -> tuple[int, float, float]:
    """(words, mean sentence length, percentage of short sentences)."""
    body = prose_of(md)
    flat = re.sub(r"\s+", " ", body)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.split()) >= 3]
    if not sentences:
        return len(body.split()), 0.0, 100.0
    lengths = [len(s.split()) for s in sentences]
    short = sum(1 for n in lengths if n <= SHORT_SENTENCE_WORDS)
    return len(body.split()), statistics.fmean(lengths), 100 * short / len(lengths)


def all_docs() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md"))


def nav_targets() -> set[str]:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    return set(re.findall(r"([\w\-/]+\.md)", text))


def is_lesson(path: Path) -> bool:
    """A numbered lesson page, as opposed to an index, project or exam page."""
    return bool(re.match(r"^\d+-", path.name)) and path.parent.name != "modules"


def main() -> int:
    cs = load_all()
    problems: list[str] = []

    docs = all_docs()
    doc_ids = {p.relative_to(DOCS).as_posix()[:-3] for p in docs}
    nav = nav_targets()

    referenced_ex: set[str] = set()
    referenced_qz: set[str] = set()

    for p in docs:
        text = p.read_text(encoding="utf-8")
        rel = p.relative_to(ROOT).as_posix()

        # 0. No stray control characters.
        #
        # Gate 28. Writing a lesson through a Python string literal turns an
        # un-escaped `\text` into a TAB and `\frac` into TAB + "rac", because
        # `\t` and `\f` are escape sequences. The LaTeX in a maths block then
        # renders as garbage while every other gate passes: the word count is
        # unchanged, the sections are all present, the components resolve, and
        # nothing raises. Only opening the page shows it.
        #
        # Tabs are included deliberately. No file in docs/ or curriculum/ uses
        # one for indentation, so a tab here is always the corruption rather
        # than a formatting choice.
        for m in re.finditer(r"[\x00-\x09\x0b-\x1f]", text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(
                f"{rel}:{line}: control character {m.group().encode().hex()} in prose — "
                f"almost certainly an un-escaped LaTeX backslash (\\t, \\f, \\a) "
                f"that a Python string literal ate"
            )
            break  # one report per file is enough to send someone to look

        # 1. Every embedded component resolves.
        for eid in EX_RE.findall(text):
            referenced_ex.add(eid)
            if eid not in cs.exercises:
                problems.append(f"{rel}: <code-exercise src={eid!r}> does not exist")
        for qid in QZ_RE.findall(text):
            referenced_qz.add(qid)
            if qid not in cs.quizzes:
                problems.append(f"{rel}: <quiz-bank src={qid!r}> does not exist")

        # 6. Every page has an H1.
        if not re.search(r"^#\s+\S", text, re.M):
            problems.append(f"{rel}: no H1")

        # 9. Tier-1 self-checks: PLAN.md §3 targets 4-8 per lesson. Caught by
        #    hand once (a lesson shipped with two), so it is checked here now.
        if is_lesson(p):
            checks = len(re.findall(r'^\?\?\?\+? question ', text, re.M))
            if not 4 <= checks <= 8:
                problems.append(
                    f"{rel}: {checks} inline self-check(s); the target is 4-8 (PLAN.md §3)"
                )

        # 10. Gate 21 — lesson depth and prose style.
        if is_lesson(p):
            words, mean_len, short_pct = prose_stats(text)
            key = p.relative_to(DOCS).as_posix()
            deep = (words >= MIN_WORDS and mean_len >= MIN_MEAN_SENTENCE
                    and short_pct <= MAX_SHORT_SENTENCE_PCT)
            if key in DEEPENING_BACKLOG:
                if deep:
                    problems.append(
                        f"{rel}: now meets gate 21 ({words} words, mean {mean_len:.1f}, "
                        f"{short_pct:.1f}% short) — remove it from DEEPENING_BACKLOG in "
                        f"tools/audit.py so the exemption cannot outlive its reason")
                continue
            if words < MIN_WORDS:
                problems.append(
                    f"{rel}: {words} words of prose; PLAN.md §4 declares 2,500-4,000. "
                    f"Add substance — definitions, worked arithmetic, an answered "
                    f"objection — rather than padding")
            if words > MAX_WORDS:
                problems.append(
                    f"{rel}: {words} words of prose; PLAN.md §4 declares 2,500-4,000. "
                    f"Cut, or split the lesson — a chapter nobody finishes teaches "
                    f"nothing")
            if mean_len < MIN_MEAN_SENTENCE:
                problems.append(
                    f"{rel}: mean sentence {mean_len:.1f} words (target 25-30, floor "
                    f"{MIN_MEAN_SENTENCE}). Join clauses with because/since/so that, so "
                    f"the reasoning sits inside sentences rather than between them")
            if short_pct > MAX_SHORT_SENTENCE_PCT:
                problems.append(
                    f"{rel}: {short_pct:.1f}% of sentences are <= {SHORT_SENTENCE_WORDS} "
                    f"words (target <15%, ceiling {MAX_SHORT_SENTENCE_PCT}%). One-line "
                    f"reveals and sentence fragments are essay mannerisms, not textbook prose")

        # 8. Lessons follow the uniform schema.
        if is_lesson(p):
            headings = re.findall(r"^##\s+(.+?)\s*$", text, re.M)
            present = [h for h in headings if h in SCHEMA_SECTIONS]
            missing = [s for s in SCHEMA_SECTIONS if s not in headings]
            if missing:
                problems.append(f"{rel}: missing schema section(s): {', '.join(missing)}")
            elif present != SCHEMA_SECTIONS:
                problems.append(f"{rel}: schema sections are out of order")

    # 2. Orphans: content nothing embeds.
    for eid in sorted(set(cs.exercises) - referenced_ex):
        problems.append(f"exercise {eid}: no page embeds it")
    for qid in sorted(set(cs.quizzes) - referenced_qz):
        problems.append(f"quiz bank {qid}: no page embeds it")

    # 3. Every content file points at a real page.
    for kind, coll in (("exercise", cs.exercises), ("quiz bank", cs.quizzes)):
        for cid, spec in sorted(coll.items()):
            lesson = spec.get("lesson")
            if lesson and lesson not in doc_ids:
                problems.append(f"{kind} {cid}: lesson {lesson!r} is not a page")

    # 4. Every page is in the nav (gate 7).
    for p in docs:
        rel = p.relative_to(DOCS).as_posix()
        if rel not in nav:
            problems.append(f"{rel}: not reachable from the mkdocs nav")

    # 5. Module index claims match reality.
    for idx in sorted((DOCS / "modules").glob("*/index.md")):
        text = idx.read_text(encoding="utf-8")
        rel = idx.relative_to(ROOT).as_posix()
        for line in text.splitlines():
            m = re.match(r"\s*(?:\d+\.|[-*])\s+(.*)", line)
            if not m:
                continue
            body = m.group(1)
            linked = re.search(r"\]\(([\w\-.]+\.md)\)", body)
            if "*planned*" in body and linked:
                problems.append(f"{rel}: calls {linked.group(1)} planned and links to it")
            if "**available**" in body and not linked:
                problems.append(f"{rel}: calls a lesson available with no link: {body[:60]}")
            if linked and not (idx.parent / linked.group(1)).exists():
                problems.append(f"{rel}: links to missing {linked.group(1)}")

    # 7. The curriculum map must not claim a lesson is built when it is not.
    cmap = DOCS / "curriculum.md"
    if cmap.exists():
        for m in re.finditer(r"\]\(([\w\-./]+\.md)\)", cmap.read_text(encoding="utf-8")):
            target = (DOCS / m.group(1)).resolve()
            if not target.exists():
                problems.append(f"docs/curriculum.md: links to missing {m.group(1)}")

    lessons = [p for p in docs if is_lesson(p)]
    print(f"pages {len(docs)}  lessons {len(lessons)}  "
          f"exercises {len(cs.exercises)}  banks {len(cs.quizzes)}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("audit OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
