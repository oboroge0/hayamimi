"""Every Markdown link in the repo must point at a file that exists.

docs/ was reorganised into subdirectories on 2026-09-03 and the links into
it live in several places at once -- other Markdown, absolute `blob/main/`
GitHub URLs in the published Dart package's README, and script docstrings.
A move breaks them all silently, so this makes it a test failure instead.

Standard library only and no models are touched: `scripts/check_doc_links.py`
just walks the tree and stats paths, so this runs anywhere pytest does.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_doc_links  # noqa: E402


def test_no_broken_markdown_links():
    broken = check_doc_links.check(ROOT)
    assert not broken, "broken Markdown links:\n" + "\n".join(
        f"  {path}:{lineno} -> {target}" for path, lineno, target in broken
    )
