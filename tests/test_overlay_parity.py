"""Guards against scripts/subtitle_server.py's OVERLAY_HTML and
mobile/hayamimi_core/lib/server/overlay_html.dart drifting apart.

The Dart file's own docstring says it's "ported from OVERLAY_HTML ... so
both the desktop and mobile subtitle servers look identical to OBS" -- but
nothing enforced that beyond the comment. This compares the two structurally
(comments stripped, whitespace normalized) so the actual markup/CSS/JS is
required to match while unrelated comment-wording differences between the
two files don't false-positive.

This test only reads the mobile Dart file; it never modifies it.
"""
import difflib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_PATH = os.path.join(ROOT, "scripts", "subtitle_server.py")
DART_PATH = os.path.join(ROOT, "mobile", "hayamimi_core", "lib", "server", "overlay_html.dart")


def _extract_py_overlay_html() -> str:
    src = open(PY_PATH, encoding="utf-8").read()
    m = re.search(r'OVERLAY_HTML = """(.*?)"""', src, re.S)
    assert m, f"could not find OVERLAY_HTML triple-quoted string in {PY_PATH}"
    return m.group(1)


def _extract_dart_overlay_html() -> str:
    src = open(DART_PATH, encoding="utf-8").read()
    m = re.search(r"const String overlayHtml = '''\n(.*?)''';", src, re.S)
    assert m, f"could not find overlayHtml raw string in {DART_PATH}"
    return m.group(1)


def _normalize(html: str) -> list:
    """Drop blank lines, leading/trailing whitespace, and full-line `//`
    comments -- the two files' inline explanatory comments are allowed to
    differ in wording, but the actual markup/CSS/JS must not drift."""
    lines = []
    for line in html.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if stripped:
            lines.append(stripped)
    return lines


def test_overlay_html_matches_mobile_port():
    py_lines = _normalize(_extract_py_overlay_html())
    dart_lines = _normalize(_extract_dart_overlay_html())
    if py_lines != dart_lines:
        diff = "\n".join(difflib.unified_diff(
            py_lines, dart_lines,
            fromfile="scripts/subtitle_server.py (OVERLAY_HTML)",
            tofile="mobile/hayamimi_core/lib/server/overlay_html.dart (overlayHtml)",
            lineterm="",
        ))
        raise AssertionError(
            "OVERLAY_HTML (scripts/subtitle_server.py) and overlayHtml "
            "(mobile/hayamimi_core/lib/server/overlay_html.dart) have "
            "diverged. The mobile overlay is a manual port of the desktop "
            "one -- when you change one, change BOTH so OBS sees the same "
            "page regardless of which server is running.\n\n" + diff
        )
