"""Check that every Markdown link in the repo points at a file that exists.

Why this exists: docs/ was reorganised into subdirectories, and the links
into it live in four different places -- other Markdown files, absolute
`blob/main/` GitHub URLs (which a relative link cannot express from inside
a published Dart package), dartdoc comments, and script docstrings. A moved
file breaks all of them silently. This walks the tree and fails instead.

Two kinds of link are resolved:

  * relative links -- `../results/benchmarks.md`, `guide/tuning.md` --
    resolved against the directory of the file that contains them;
  * self-links -- `https://github.com/oboroge0/hayamimi/blob/main/docs/...`
    -- resolved against the repository root, so a `blob/main/` URL to a file
    this repo no longer has is a failure too.

Everything else (other http(s) URLs, mailto:, bare `#anchor` fragments) is
ignored: this checker is about paths inside the tree, not reachability of
the internet. Anchors on an in-tree link are stripped and not verified --
GitHub's anchor slugs for Japanese headings are not worth reimplementing.

Standard library only, and it reads no models, so it is safe to run in CI
and from `tests/test_doc_links.py`.

    python scripts/check_doc_links.py            # whole repo
    python scripts/check_doc_links.py docs       # one subtree
"""

from __future__ import annotations

import os
import re
import sys
from urllib.parse import unquote

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that hold generated or vendored Markdown nobody edits.
SKIP_DIRS = {".dart_tool", "build", "node_modules", ".git", ".claude"}

SELF_URL_PREFIX = "https://github.com/oboroge0/hayamimi/blob/main/"

# [text](target) -- target stops at whitespace (a "title") or the closing
# paren. Angle-bracket targets (<...>) are unwrapped by _clean().
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")


def _clean(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return target


def _resolve(md_path: str, target: str) -> str | None:
    """Return the absolute path a link points at, or None if not checkable."""
    target = _clean(target)
    if not target or target.startswith("#"):
        return None
    if target.startswith(SELF_URL_PREFIX):
        rest = target[len(SELF_URL_PREFIX):]
        base = REPO_ROOT
    elif re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return None  # http(s), mailto, tel, ...
    elif target.startswith("//"):
        return None  # protocol-relative
    elif target.startswith("/"):
        rest = target.lstrip("/")
        base = REPO_ROOT
    else:
        rest = target
        base = os.path.dirname(md_path)

    rest = rest.split("#", 1)[0].split("?", 1)[0]
    if not rest:
        return None
    return os.path.normpath(os.path.join(base, unquote(rest)))


def iter_markdown(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.lower().endswith(".md"):
                yield os.path.join(dirpath, name)


def check(root: str) -> list[tuple[str, int, str]]:
    broken: list[tuple[str, int, str]] = []
    for md_path in iter_markdown(root):
        with open(md_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                for target in LINK_RE.findall(line):
                    resolved = _resolve(md_path, target)
                    if resolved is None:
                        continue
                    if not os.path.exists(resolved):
                        rel = os.path.relpath(md_path, REPO_ROOT)
                        broken.append((rel.replace(os.sep, "/"), lineno, _clean(target)))
    return broken


def main(argv: list[str]) -> int:
    root = os.path.abspath(argv[1]) if len(argv) > 1 else REPO_ROOT
    broken = check(root)
    for path, lineno, target in broken:
        print(f"{path}:{lineno}: broken link -> {target}")
    if broken:
        print(f"\n{len(broken)} broken link(s).")
        return 1
    print("All Markdown links resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
