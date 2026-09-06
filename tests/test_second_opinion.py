"""Unit tests for the ja second-opinion agreement gate (asr_engine)."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import asr_engine
from asr_engine import (ModelUnavailable, choose_second_opinion,
                        hyp_agreement_cer)


def test_agreement_cer_ignores_spaces_and_punct():
    assert hyp_agreement_cer("高校生200人に聞きました。", "高校生 200人に聞きました") == 0.0


def test_agreement_cer_scales_with_divergence():
    # extra trailing content (the background-speech failure mode) drives the
    # rate well past any sane threshold
    a = "高校生200人に聞きました"
    b = "高校生200人に聞きましたネットの見過ぎを感じ使うのをやめたというもいます"
    assert hyp_agreement_cer(a, b) > 0.5


def test_choose_adopts_second_when_close():
    text, used = choose_second_opinion("きょうのニュースをお伝えします",
                                       "今日のニュースをお伝えします", 0.25)
    assert used and text == "今日のニュースをお伝えします"


def test_choose_keeps_primary_when_divergent():
    text, used = choose_second_opinion(
        "今100mのターンを迎える", "ただ本多選手はこの得意のドルフィンキックで", 0.25)
    assert not used and text == "今100mのターンを迎える"


def test_choose_keeps_primary_on_empty_second():
    text, used = choose_second_opinion("なにか", "   ", 0.25)
    assert not used and text == "なにか"


class _FakeASR:
    """Just enough of RoutedASR to drive _maybe_second_opinion unbound."""

    def __init__(self, get=None, decode=None):
        self._pja_warned = False
        self._ja_second_opinion = True
        self._agree_threshold = 0.25
        self._get_impl = get
        self._decode_impl = decode

    def _get(self, name):
        assert name == "pja"
        if self._get_impl is not None:
            return self._get_impl(name)
        return object()

    def _decode(self, rec, samples, sr):
        return self._decode_impl(rec, samples, sr)

    def _emit(self, event):
        pass  # no on_event sink wired for this stub; just enough to not AttributeError


def test_missing_model_warns_once_and_disables(capsys):
    def raise_unavailable(name):
        raise ModelUnavailable(name)

    fake = _FakeASR(get=raise_unavailable)
    out = asr_engine.RoutedASR._maybe_second_opinion(fake, "元テキスト", None, 16000)
    assert out == "元テキスト"
    assert fake._pja_warned is True
    assert fake._ja_second_opinion is False
    assert "second opinion" in capsys.readouterr().out


def test_second_decode_failure_keeps_primary():
    def boom(rec, s, sr):
        raise RuntimeError("decode failed")

    fake = _FakeASR(decode=boom)
    out = asr_engine.RoutedASR._maybe_second_opinion(fake, "元テキスト", None, 16000)
    assert out == "元テキスト"


def test_agreeing_second_is_adopted():
    fake = _FakeASR(decode=lambda rec, s, sr: "元テキストです")
    out = asr_engine.RoutedASR._maybe_second_opinion(fake, "元テキストてす", None, 16000)
    assert out == "元テキストです"
