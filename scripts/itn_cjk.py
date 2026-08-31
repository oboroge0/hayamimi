"""Conservative inverse text normalization (ITN) for Japanese/Chinese/Cantonese
kanji numerals -> arabic digits.

Pure Python, zero dependencies, deterministic (no catastrophic-backtracking
regex: every quantifier below is bounded).

Postprocessing order (fixed, documented here as the source of truth and
implemented in asr_engine.RoutedASR.transcribe()):

    ITN (this module) -> punctuation restore (ja only, punct_ja.py) ->
    user find/replace (RoutedASR._replace / --replace / set_replacements)

ITN runs first so punctuation restoration sees normalized digits (matters
for e.g. decimal points), and the user's own --replace dictionary always
gets the last word -- it can undo or override anything ITN or punctuation
produced. See RoutedASR.set_itn_overrides() for the user-level exclude/force
knobs, which take precedence over the built-in rules below but are still
applied before punctuation and before --replace.

What this module converts (conservative by design -- when a candidate is
ambiguous, it is left alone):

  1. Magnitude-form ("kurai-dori") numbers: digits combined with the unit
     characters 十/百/千 are fully computed to an integer (三千二百 -> 3200,
     一百五十 -> 150). When 万/萬 or 億 are present they are KEPT as literal
     units and only the digit portion before each is converted
     (四万人弱 -> 4万人弱), matching how these are still read/written in
     practice.
  2. Decimals with 点: both sides are read digit-by-digit unless the integer
     side itself contains a magnitude unit (二点四 -> 2.4, 八零二点一 -> 802.1).
  3. Bare digit-string runs of two or more kanji digits with NO magnitude
     unit are read digit-by-digit, matching how years/codes are spoken
     (一九四五年 -> 1945年, 二〇二〇 -> 2020).

A single bare numeral character (e.g. lone 一, 十) is never converted by
default -- this is what keeps idioms and proper nouns like 一番/一緒/一部/
四国/九州/十分/一石二鳥/二人三脚 untouched, since each of those breaks into
single-character runs once the non-digit characters between them stop the
scan. A small built-in exclusion list adds an explicit backstop for phrases
that might otherwise form a longer run.

Traditional forms are handled alongside simplified ones (萬/万, 兩/两/二,
零/〇), since yue output from SenseVoice comes out in traditional script.
"""
from __future__ import annotations

import re
from typing import NamedTuple

APPLICABLE_LANGS = frozenset({"ja", "zh", "yue"})

# kanji digit -> value. Simplified and traditional variants share a slot;
# 兩/两 (liang, "two" used before a classifier) maps to the same value as 二.
DIGIT = {
    "〇": 0, "零": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
    "兩": 2, "两": 2,
}

# multiplier units usable inside a single <10000 magnitude-form segment
SMALL_UNIT = {"十": 10, "百": 100, "千": 1000}

# big units kept literal in the output; only their preceding segment is
# converted to arabic (四万人弱 -> 4万人弱, never "40000人弱")
BIG_UNITS = ("万", "萬")

_ALL_NUMERAL_CHARS = "".join(sorted(set(DIGIT) | set(SMALL_UNIT) | set(BIG_UNITS) | {"億"}))

# bounded repetition throughout: {1,12} / {1,10} caps the scan, so this can
# never backtrack catastrophically regardless of input length
_TOKEN_RE = re.compile(
    rf"[{_ALL_NUMERAL_CHARS}]{{1,12}}(?:点[{''.join(sorted(DIGIT))}]{{1,10}})?"
)

DEFAULT_EXCLUSIONS = frozenset({
    "一番", "一緒", "一部", "四国", "九州", "十分", "一石二鳥", "二人三脚",
    "一体", "一旦", "一気", "七夕", "三味線", "二重", "十字架", "一杯",
})


class ITNOverrides(NamedTuple):
    """User-level ITN overrides. exclude ADDS to the built-in exclusion
    list (never replaces it); force mappings win over rule-based output
    for any matching literal substring."""
    exclude: frozenset[str]
    force: dict


EMPTY_OVERRIDES = ITNOverrides(exclude=frozenset(), force={})


def _has_small_unit(s: str) -> bool:
    return any(c in SMALL_UNIT for c in s)


def _digit_string(s: str) -> str | None:
    """Digit-by-digit reading: each character independently mapped, no
    positional arithmetic. Returns None if any character isn't a plain
    digit (defensive; the caller only feeds pre-validated runs)."""
    out = []
    for ch in s:
        if ch not in DIGIT:
            return None
        out.append(str(DIGIT[ch]))
    return "".join(out)


def _parse_small(s: str) -> int | None:
    """Arithmetic reading of a magnitude-form segment bounded to <10000
    (i.e. containing only digits and 十/百/千, no 万/萬/億). Returns None on
    an unparseable character."""
    if not s:
        return 0
    total = 0
    num = 0
    for ch in s:
        if ch in DIGIT:
            num = DIGIT[ch]
        elif ch in SMALL_UNIT:
            total += (num if num != 0 else 1) * SMALL_UNIT[ch]
            num = 0
        else:
            return None
    return total + num


def _convert_big(tok: str) -> str | None:
    """Magnitude-form token containing 億 and/or 万/萬: each big unit is kept
    literal, with only its preceding segment converted to arabic."""
    parts = []
    remaining = tok
    if "億" in remaining:
        idx = remaining.index("億")
        seg = remaining[:idx]
        val = _parse_small(seg) if seg else 1
        if val is None:
            return None
        parts.append(f"{val}億")
        remaining = remaining[idx + 1:]
    for wan in BIG_UNITS:
        if wan in remaining:
            idx = remaining.index(wan)
            seg = remaining[:idx]
            val = _parse_small(seg) if seg else 1
            if val is None:
                return None
            parts.append(f"{val}{wan}")
            remaining = remaining[idx + 1:]
            break
    if remaining:
        if _has_small_unit(remaining):
            val = _parse_small(remaining)
        else:
            ds = _digit_string(remaining)
            val = int(ds) if ds is not None else None
        if val is None:
            return None
        parts.append(str(val))
    return "".join(parts)


def _convert_token(tok: str) -> str:
    if "点" in tok:
        integer_part, _, decimal_part = tok.partition("点")
        if not integer_part or not decimal_part:
            return tok  # "点" with nothing on one side: not a real decimal
        if _has_small_unit(integer_part):
            int_val = _parse_small(integer_part)
            int_str = str(int_val) if int_val is not None else None
        else:
            int_str = _digit_string(integer_part)
        dec_str = _digit_string(decimal_part)
        if int_str is None or dec_str is None:
            return tok
        return f"{int_str}.{dec_str}"

    if len(tok) == 1:
        return tok  # single bare numeral: never converted by default

    if "億" in tok or any(u in tok for u in BIG_UNITS):
        converted = _convert_big(tok)
        return converted if converted is not None else tok

    if _has_small_unit(tok):
        val = _parse_small(tok)
        return str(val) if val is not None else tok

    # pure digit-string run (len >= 2, no units): read digit-by-digit
    converted = _digit_string(tok)
    return converted if converted is not None else tok


def _protected_spans(text: str, exclusions: frozenset) -> list[tuple[int, int]]:
    spans = []
    for word in exclusions:
        if not word:
            continue
        start = 0
        while True:
            idx = text.find(word, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(word)))
            start = idx + 1
    return spans


def convert(text: str, lang: str, exclude: frozenset | set | None = None,
            force: dict | None = None) -> str:
    """Apply conservative CJK ITN to `text` for lang in {ja, zh, yue}; a no-op
    for any other language.

    `force` is applied FIRST as a plain literal find/replace -- its output
    (typically already arabic) is immune to the rule-based pass below, since
    that pass only matches kanji-numeral characters. This is what makes user
    force-mappings win over the built-in rules.

    `exclude` ADDS to DEFAULT_EXCLUSIONS; any rule-based candidate whose span
    overlaps an excluded phrase's occurrence is left untouched.
    """
    if lang not in APPLICABLE_LANGS or not text:
        return text

    if force:
        for wrong, right in force.items():
            if wrong:
                text = text.replace(wrong, right)

    exclusions = DEFAULT_EXCLUSIONS if not exclude else (DEFAULT_EXCLUSIONS | frozenset(exclude))
    protected = _protected_spans(text, exclusions)

    def _sub(m: re.Match) -> str:
        s, e = m.span()
        if any(s < pe and ps < e for ps, pe in protected):
            return m.group(0)
        return _convert_token(m.group(0))

    return _TOKEN_RE.sub(_sub, text)
