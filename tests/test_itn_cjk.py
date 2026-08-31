"""Pure-logic tests for the conservative CJK ITN module (scripts/itn_cjk.py).

No models needed -- these always run. Convert-cases and must-not-convert
cases are kept in roughly equal measure, since the module's whole design
goal is staying conservative (leave ambiguous/idiomatic text alone).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import itn_cjk  # noqa: E402


# ---- languages the module applies to ---------------------------------------

def test_only_applies_to_ja_zh_yue():
    for lang in ("ja", "zh", "yue"):
        assert itn_cjk.convert("三千二百", lang) == "3200"
    for lang in ("en", "ko", "de", ""):
        assert itn_cjk.convert("三千二百", lang) == "三千二百"


def test_empty_text_is_a_noop():
    assert itn_cjk.convert("", "ja") == ""


# ---- magnitude-form (位取り) conversion -------------------------------------

def test_magnitude_form_full_arithmetic_without_big_units():
    assert itn_cjk.convert("三千二百", "ja") == "3200"
    assert itn_cjk.convert("一百五十", "zh") == "150"
    assert itn_cjk.convert("四十九", "ja") == "49"


def test_magnitude_form_keeps_man_and_oku_as_literal_units():
    # the FLEURS real example: 万/億 stay as units, only the digit part converts
    assert itn_cjk.convert("四万人弱", "ja") == "4万人弱"
    assert itn_cjk.convert("一億円", "ja") == "1億円"
    assert itn_cjk.convert("三千万円", "ja") == "3000万円"


def test_traditional_wan_and_liang_forms():
    assert itn_cjk.convert("四萬人", "yue") == "4萬人"
    assert itn_cjk.convert("兩百", "yue") == "200"
    assert itn_cjk.convert("两千", "zh") == "2000"


# ---- decimals with 点 --------------------------------------------------------

def test_decimal_digit_by_digit_reading():
    assert itn_cjk.convert("二点四", "ja") == "2.4"
    # the FLEURS real example: leading zero, digit-string integer part
    assert itn_cjk.convert("八零二点一", "zh") == "802.1"


def test_decimal_with_magnitude_integer_part():
    assert itn_cjk.convert("三十点五", "ja") == "30.5"


# ---- digit-string years/codes -----------------------------------------------

def test_bare_digit_run_reads_digit_by_digit():
    assert itn_cjk.convert("一九四五年", "ja") == "1945年"
    assert itn_cjk.convert("二〇二〇", "zh") == "2020"
    assert itn_cjk.convert("零一二三", "ja") == "0123"


# ---- must NOT convert: single bare numerals ---------------------------------

def test_single_bare_numeral_untouched():
    assert itn_cjk.convert("一つください", "ja") == "一つください"
    assert itn_cjk.convert("十本", "ja") == "十本"
    assert itn_cjk.convert("九月", "ja") == "九月"


# ---- must NOT convert: idioms / proper nouns --------------------------------

IDIOM_SENTENCES = [
    "一番好きです",
    "一緒に行きます",
    "一部だけです",
    "四国に住んでいます",
    "九州出身です",
    "十分に休みました",
    "一石二鳥の作戦です",
    "二人三脚で頑張ります",
]


@pytest.mark.parametrize("sentence", IDIOM_SENTENCES)
def test_idioms_and_proper_nouns_stay_untouched(sentence):
    assert itn_cjk.convert(sentence, "ja") == sentence


def test_default_exclusion_list_blocks_a_longer_run_directly():
    # exercise the exclusion-span mechanism itself (not just the
    # single-char-run coincidence the idioms above rely on)
    assert itn_cjk.convert("十分だと思う", "ja") == "十分だと思う"
    assert itn_cjk.convert("一石二鳥だ", "ja") == "一石二鳥だ"


# ---- user overrides: exclude / force ----------------------------------------

def test_user_exclude_adds_to_builtin_list():
    # without an exclude, this two-digit run converts
    assert itn_cjk.convert("二十歳", "ja") == "20歳"
    assert itn_cjk.convert("二十歳", "ja", exclude={"二十歳"}) == "二十歳"
    # built-ins still apply alongside a user exclude
    assert itn_cjk.convert("一番だ", "ja", exclude={"二十歳"}) == "一番だ"


def test_user_force_wins_over_rule_output():
    # force is a literal substitution applied before the rule pass, so its
    # output (already arabic here) is immune to re-conversion
    assert itn_cjk.convert("三千二百人", "ja", force={"三千二百": "3,200"}) == "3,200人"


def test_user_force_can_apply_to_text_the_rules_would_leave_alone():
    assert itn_cjk.convert("一番だ", "ja", force={"一番": "No.1"}) == "No.1だ"
