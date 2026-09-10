"""Unit tests for the opt-in 4-class (+ none) ja punctuation restorer,
scripts/punct_ja.py::PunctuatorJa4Class (improvement track C -- see
docs/eval/punct_retrain.md for data/training/eval details).

Two tiers, matching the repo's "model-free tests stay model-free" split
(tests/test_units.py's own docstring):

  - test_reconstruct_* : pure, model-free tests of
    reconstruct_punct4_text() (offsets/label-ids -> punctuated string).
    These run unconditionally, no onnxruntime/tokenizers/model files
    needed -- this is the logic actually worth regression-testing.

  - test_restore_smoke_* : end-to-end smoke tests that load the real
    trained ONNX model via PunctuatorJa4Class. Skipped automatically if
    `onnxruntime`/`tokenizers` aren't installed or the model directory
    (models/punct-ja-4class-permissive/, a large git-ignored artifact, not
    checked in) isn't present -- e.g. CI (.github/workflows/test.yml only
    installs sherpa-onnx numpy scipy soundfile pytest, no
    onnxruntime/tokenizers, and never downloads this model).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from punct_ja import (PUNCT4_DEFAULT_MODEL_DIR, PUNCT4_LABELS, PunctuatorJa4Class,
                       reconstruct_punct4_text)

# label ids, matching PUNCT4_LABELS = ("O", "、", "。", "？", "！")
O, TOUTEN, KUTEN, Q, EXCLAIM = range(5)


# ---- reconstruct_punct4_text: pure, model-free -----------------------------

def test_reconstruct_inserts_period_and_comma_after_token_end():
    text = "きょうはいいてんきです"
    # 3 "tokens": きょうは(0,4) いいてんきです(4,11) [SEP](0,0)
    offsets = [(0, 4), (4, 11), (0, 0)]
    labels = [TOUTEN, KUTEN, O]
    assert reconstruct_punct4_text(text, offsets, labels) == "きょうは、いいてんきです。"


def test_reconstruct_inserts_question_mark():
    text = "だいじょうぶですか"
    offsets = [(0, 9)]
    labels = [Q]
    assert reconstruct_punct4_text(text, offsets, labels) == "だいじょうぶですか？"


def test_reconstruct_inserts_exclamation_mark():
    text = "すごい"
    offsets = [(0, 3)]
    labels = [EXCLAIM]
    assert reconstruct_punct4_text(text, offsets, labels) == "すごい！"


def test_reconstruct_no_mark_is_noop():
    text = "なにもおきません"
    offsets = [(0, len(text))]
    labels = [O]
    assert reconstruct_punct4_text(text, offsets, labels) == text


def test_reconstruct_ignores_special_token_offsets():
    # CLS/SEP report offset (0, 0) -- must not be mistaken for "insert a
    # mark before the first character".
    text = "abc"
    offsets = [(0, 0), (0, 3), (0, 0)]
    labels = [KUTEN, O, KUTEN]  # first/last are the special tokens, ignored
    assert reconstruct_punct4_text(text, offsets, labels) == "abc"


def test_reconstruct_only_first_mark_per_position_wins():
    # a defensive case: if two tokens somehow both end at the same char
    # index (shouldn't happen with a real fast tokenizer, but the
    # reconstruction must not silently double-insert), the earlier one is
    # already committed to that slot before a later same-position token
    # would try to overwrite it -- verified by construction (mark_at[i] is
    # only ever set once per zip iteration reaching that i).
    text = "ab"
    offsets = [(0, 2), (0, 2)]
    labels = [KUTEN, Q]
    out = reconstruct_punct4_text(text, offsets, labels)
    # exactly one mark was inserted at that position, and it's a valid one
    assert out in ("ab。", "ab？")
    assert len(out) == 3


def test_reconstruct_multi_sentence_marks_mid_and_end():
    text = "きょうはあめですあしたははれるでしょうか"
    offsets = [(0, 8), (8, len(text))]
    labels = [KUTEN, Q]
    assert reconstruct_punct4_text(text, offsets, labels) == (
        "きょうはあめです。あしたははれるでしょうか？"
    )


def test_punct4_labels_order():
    # reconstruct_punct4_text indexes into PUNCT4_LABELS by predicted class
    # id -- pin the label order since scripts/train_punct_ja.py's LABELS
    # list and the trained model's classifier head both depend on it
    # matching exactly (changing the order would silently relabel every
    # prediction without any error).
    assert PUNCT4_LABELS == ("O", "、", "。", "？", "！")


# ---- end-to-end smoke test: real model, skipped if unavailable ------------

def _punct4_model_available():
    if not PUNCT4_DEFAULT_MODEL_DIR.exists():
        return False
    onnx_path = PUNCT4_DEFAULT_MODEL_DIR / "punct_4class.onnx"
    tokenizer_path = PUNCT4_DEFAULT_MODEL_DIR / "hf" / "tokenizer.json"
    return onnx_path.exists() and tokenizer_path.exists()


pytest.importorskip("onnxruntime", reason="opt-in PunctuatorJa4Class needs onnxruntime")
pytest.importorskip("tokenizers", reason="opt-in PunctuatorJa4Class needs the tokenizers package")


@pytest.mark.skipif(not _punct4_model_available(),
                     reason="models/punct-ja-4class-permissive/ not present (git-ignored, opt-in model)")
def test_restore_smoke_question():
    p = PunctuatorJa4Class()
    out = p.restore("これって本当に大丈夫なんですか")
    assert out.endswith("？")


@pytest.mark.skipif(not _punct4_model_available(),
                     reason="models/punct-ja-4class-permissive/ not present (git-ignored, opt-in model)")
def test_restore_smoke_exclamation():
    # The permissive (round-2) model learned ！ from real web text rather
    # than the ~600 self-authored templates round 1 used; a plain
    # exclamation is the cheapest check that the class is live at all.
    p = PunctuatorJa4Class()
    assert p.restore("本当にありがとうございました").endswith("！")


@pytest.mark.skipif(not _punct4_model_available(),
                     reason="models/punct-ja-4class-permissive/ not present (git-ignored, opt-in model)")
def test_restore_smoke_only_inserts_marks():
    """restore() must be purely additive: deleting the marks it inserted
    has to give back the input, character for character.

    This is the invariant that breaks first if the tokenizer and the model
    ever drift apart -- character offsets that are off by one, or a
    tokenizer that normalizes its input, would corrupt the text rather
    than merely mispunctuate it. It caught nothing when written, which is
    the point: it pins the property that made it safe to swap the base
    model (and its tokenizer) under this class in the first place.
    """
    p = PunctuatorJa4Class()
    for text in (
        "明日の会議は午後三時から始まりますので資料の準備をお願いします",
        "きょうはとてもいい天気ですね",
        "ローマ字のABCと数字の123が混ざった文です",
    ):
        out = p.restore(text)
        assert "".join(c for c in out if c not in "、。？！") == text


@pytest.mark.skipif(not _punct4_model_available(),
                     reason="models/punct-ja-4class-permissive/ not present (git-ignored, opt-in model)")
def test_restore_smoke_empty_string():
    p = PunctuatorJa4Class()
    assert p.restore("") == ""
    assert p.restore("   ") == ""


def test_missing_model_dir_raises_file_not_found(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    with pytest.raises(FileNotFoundError):
        PunctuatorJa4Class(model_dir=tmp_path)
