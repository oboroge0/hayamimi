"""Fast regression tests for the model-free logic.

Run with: .venv/Scripts/python -m pytest tests -q
No ASR/MT models are loaded; these guard the plumbing that past iterations
broke or nearly broke (preroll bleed, replacement parsing, digit guard,
routing table consistency).
"""
import json
import os
import sys
import types
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from realtime_transcribe import (AudioHistory, PREROLL_S, digits_consistent,
                                 build_arg_parser, build_translators,
                                 mix_input_blocks, parse_audio_device,
                                 parse_translation_targets, translate_by_sentence,
                                 Refiner, TranslationWorker, translators_for_source,
                                 WavRecorder)
from subtitle_server import DASHBOARD_HTML, SubtitleServer
from translate_m2m import TranslatorM2M
import asr_engine


class FakeTranslator:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return self.mapping.get(text, text)


# ---- local audio input mixing --------------------------------------------

def test_parse_audio_device_accepts_index_or_name():
    assert parse_audio_device("0") == 0
    assert parse_audio_device(" BlackHole 2ch ") == "BlackHole 2ch"
    assert parse_audio_device(None) is None


def test_mix_input_blocks_preserves_single_input_and_clips_two_inputs():
    meeting = np.array([0.25, 0.8, -0.8], dtype=np.float32)
    microphone = np.array([0.5, 0.7, -0.7], dtype=np.float32)
    assert mix_input_blocks([meeting]) is meeting
    np.testing.assert_allclose(
        mix_input_blocks([meeting, microphone]),
        np.array([0.75, 1.0, -1.0], dtype=np.float32),
    )


def test_dual_input_cli_options_are_accepted():
    args = build_arg_parser().parse_args([
        "--device", "BlackHole 2ch", "--mic-device", "2", "--translate", "ja",
    ])
    assert args.device == "BlackHole 2ch"
    assert args.mic_device == "2"


def test_wav_recorder_writes_exact_mixed_stream_and_refuses_overwrite(tmp_path):
    path = tmp_path / "meeting.wav"
    recorder = WavRecorder(str(path), sample_rate=16000)
    recorder.write(np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32))
    recorder.close()

    with wave.open(str(path), "rb") as saved:
        assert saved.getnchannels() == 1
        assert saved.getsampwidth() == 2
        assert saved.getframerate() == 16000
        assert saved.getnframes() == 5
    assert recorder.duration_s == 5 / 16000
    with pytest.raises(FileExistsError):
        WavRecorder(str(path), sample_rate=16000)


# ---- digit guard -----------------------------------------------------------

def test_digits_pass_when_source_has_no_ascii_digits():
    assert digits_consistent("午後三時から始まります", "It starts at 3 pm")


def test_digits_pass_on_exact_and_superset():
    assert digits_consistent("3時です", "It is 3:00")
    assert digits_consistent("500円です", "It costs 500 yen")


def test_digits_fail_on_mangled_number():
    assert not digits_consistent("500万円です", "It costs 5 pounds")


# ---- sentence-split translation -------------------------------------------

def test_translate_by_sentence_splits_on_terminators():
    tr = FakeTranslator({"こんにちは。": "Hello.", "元気ですか？": "How are you?"})
    assert translate_by_sentence(tr, "こんにちは。元気ですか？") == "Hello. How are you?"
    assert tr.calls == ["こんにちは。", "元気ですか？"]


def test_translate_by_sentence_drops_untranslated():
    tr = FakeTranslator({})
    assert translate_by_sentence(tr, "こんにちは。") == ""


def test_translate_by_sentence_number_guard_falls_back():
    tr = FakeTranslator({"500万円です。": "It costs 5 pounds."})
    # mangled number -> per-sentence fallback -> treated as untranslated
    assert translate_by_sentence(tr, "500万円です。") == ""


def test_translate_by_sentence_splits_english_periods():
    tr = FakeTranslator({"Hello world.": "こんにちは世界。", "How are you?": "お元気ですか？"})
    assert translate_by_sentence(tr, "Hello world. How are you?") == "こんにちは世界。 お元気ですか？"
    assert tr.calls == ["Hello world.", "How are you?"]


# ---- translation routing / CLI --------------------------------------------

def test_translate_ja_cli_is_accepted_without_changing_default():
    parser = build_arg_parser()
    assert parser.parse_args([]).translate is None
    assert parser.parse_args(["--translate", "ja"]).translate == "ja"
    assert parser.parse_args(["--translate"]).translate == "en"


def test_parse_translation_targets_keeps_existing_routes_and_rejects_unknown():
    assert parse_translation_targets("en,zh,ko,ja,en") == ["en", "zh", "ko", "ja"]
    with pytest.raises(ValueError, match="unsupported translation target"):
        parse_translation_targets("fr")


def test_build_translators_routes_english_to_japanese_through_m2m(monkeypatch):
    class FakeM2M:
        def __init__(self, target_lang, source_lang="ja"):
            self.target_lang = target_lang
            self.source_lang = source_lang

    class FakeJaEn:
        source_lang = "ja"
        target_lang = "en"

    monkeypatch.setitem(sys.modules, "translate_m2m", types.SimpleNamespace(TranslatorM2M=FakeM2M))
    monkeypatch.setitem(sys.modules, "translate_ja_en", types.SimpleNamespace(TranslatorJaEn=FakeJaEn))
    translators = build_translators("ja,en,zh,ko")
    assert (translators["ja"].source_lang, translators["ja"].target_lang) == ("en", "ja")
    assert (translators["en"].source_lang, translators["en"].target_lang) == ("ja", "en")
    assert [lang for lang, _ in translators_for_source(translators, "ja")] == ["en", "zh", "ko"]
    assert [lang for lang, _ in translators_for_source(translators, "en")] == ["ja"]


def test_m2m_english_to_japanese_uses_language_tokens():
    calls = []

    class FakeSentencePiece:
        def encode(self, text, out_type):
            assert (text, out_type) == ("Hello world.", str)
            return ["▁Hello", "▁world", "."]

        def decode(self, tokens):
            assert tokens == ["▁こんにちは", "▁世界"]
            return "こんにちは世界。"

    class FakeResult:
        hypotheses = [["__ja__", "▁こんにちは", "▁世界", "</s>"]]

    class FakeCTranslate2:
        def translate_batch(self, source, **kwargs):
            calls.append((source, kwargs))
            return [FakeResult()]

    tr = TranslatorM2M.__new__(TranslatorM2M)
    tr.source_lang = "en"
    tr.target_lang = "ja"
    tr._source_token = "__en__"
    tr._target_token = "__ja__"
    tr._beam_size = 4
    tr._sp = FakeSentencePiece()
    tr._translator = FakeCTranslate2()

    assert tr.translate("Hello world.") == "こんにちは世界。"
    assert calls[0][0] == [["__en__", "▁Hello", "▁world", ".", "</s>"]]
    assert calls[0][1]["target_prefix"] == [["__ja__"]]


def test_dashboard_and_final_events_bind_translation_by_line_id():
    assert "cardsById.set(ev.line_id, card)" in DASHBOARD_HTML
    assert "cardsById.get(ev.line_id)" in DASHBOARD_HTML
    assert "(ev.translations || []).forEach" in DASHBOARD_HTML
    assert 'tr.className = "refine-trans"' in DASHBOARD_HTML
    server = SubtitleServer()
    q = server.subscribe()
    server.final("Hello", lang="en", line_id="final-7")
    event = json.loads(q.get_nowait())
    assert event["line_id"] == "final-7"


def test_translation_worker_filters_source_and_preserves_line_id():
    class RecordingServer:
        def __init__(self):
            self.events = []

        def publish(self, event):
            self.events.append(event)

    en_ja = FakeTranslator({"Hello": "こんにちは"})
    en_ja.source_lang = "en"
    ja_en = FakeTranslator({"Hello": "wrong route"})
    ja_en.source_lang = "ja"
    server = RecordingServer()
    worker = TranslationWorker({"ja": en_ja, "en": ja_en}, server=server)
    worker.submit("Hello", "en", line_id="final-9")
    worker.wait()
    assert en_ja.calls == ["Hello"]
    assert ja_en.calls == []
    assert server.events == [{
        "type": "translation", "lang": "ja", "text": "こんにちは",
        "source_lang": "en", "line_id": "final-9",
    }]


def test_refiner_publishes_source_and_local_translation_together():
    class FakeASR:
        def transcribe(self, samples, sample_rate, known_lang=None, live=False):
            return {"text": "Hello world."}

    class RecordingServer:
        def __init__(self):
            self.events = []

        def publish(self, event):
            self.events.append(event)

    server = RecordingServer()
    printer = types.SimpleNamespace(server=server)
    history = types.SimpleNamespace(buf=np.zeros(20, dtype=np.float32), offset=0)
    translator = FakeTranslator({"Hello world.": "こんにちは世界。"})
    translator.source_lang = "en"
    refiner = Refiner(FakeASR(), history, sample_rate=10, printer=printer,
                      translators={"ja": translator})
    refiner.spans = [(0, 10, "en", "Hello world.", "S1")]

    refiner.maybe_refine(now_sample=10, force=True)

    assert server.events == [{
        "type": "refine", "text": "Hello world.", "lang": "en", "speaker": "S1",
        "translations": [{"lang": "ja", "text": "こんにちは世界。"}],
    }]


# ---- AudioHistory / preroll -----------------------------------------------

def test_preroll_prepends_context():
    h = AudioHistory(sample_rate=100)  # tiny sr keeps the math readable
    h.push(np.arange(1000, dtype=np.float32))
    seg = np.zeros(10, dtype=np.float32)
    out = h.with_preroll(seg_start=500, seg_samples=seg)
    assert len(out) == int(PREROLL_S * 100) + 10
    assert out[0] == 500 - int(PREROLL_S * 100)


def test_preroll_never_bleeds_into_previous_segment():
    h = AudioHistory(sample_rate=100)
    h.push(np.arange(1000, dtype=np.float32))
    h.with_preroll(seg_start=300, seg_samples=np.zeros(100, dtype=np.float32))  # ends at 400
    out = h.with_preroll(seg_start=450, seg_samples=np.zeros(10, dtype=np.float32))
    assert out[0] == 400  # clamped to the previous segment's end, not 450-preroll


def test_preroll_respects_rolling_buffer_offset():
    h = AudioHistory(sample_rate=100, keep_s=5.0)  # keeps 500 samples
    h.push(np.arange(2000, dtype=np.float32))
    assert h.offset == 1500
    out = h.with_preroll(seg_start=1600, seg_samples=np.zeros(10, dtype=np.float32))
    assert out[0] == 1600 - int(PREROLL_S * 100)


# ---- replacement dictionary -----------------------------------------------

def test_load_replacements_formats(tmp_path):
    f = tmp_path / "dict.txt"
    f.write_text("# comment\nA=B\nC\tD\nE→F\n\n", encoding="utf-8")
    pairs = asr_engine._load_replacements(str(f))
    assert pairs == [("A", "B"), ("C", "D"), ("E", "F")]


def test_load_replacements_empty_path():
    assert asr_engine._load_replacements("") == []


# ---- hotword encodability check (issue #1) ---------------------------------

def test_cjkchar_units_splits_cjk_chars_and_keeps_ascii_words():
    assert asr_engine._cjkchar_units("屈足湖") == ["屈", "足", "湖"]
    assert asr_engine._cjkchar_units("GANKE FES") == ["GANKE", "FES"]
    assert asr_engine._cjkchar_units("屈足 FES") == ["屈", "足", "FES"]


def test_check_hotwords_encodable_all_missing(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blk> 0\n<unk> 1\n▁ƊĢĥ 2\n", encoding="utf-8")
    hw = tmp_path / "hotwords.txt"
    hw.write_text("屈足湖\nGANKE FES\n# comment\n\n", encoding="utf-8")
    total, bad = asr_engine.check_hotwords_encodable(str(hw), str(tokens))
    assert (total, bad) == (2, 2)


def test_check_hotwords_encodable_some_found(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<blk> 0\n屈 1\n足 2\n", encoding="utf-8")
    hw = tmp_path / "hotwords.txt"
    hw.write_text("屈足\n屈足湖\n", encoding="utf-8")
    total, bad = asr_engine.check_hotwords_encodable(str(hw), str(tokens))
    assert (total, bad) == (2, 1)  # "屈足" fully found, "屈足湖" missing 湖


def test_check_hotwords_encodable_empty_path():
    assert asr_engine.check_hotwords_encodable("", "tokens.txt") == (0, 0)


# ---- routing table consistency --------------------------------------------

def test_routing_sets_are_disjoint():
    sets = [asr_engine.RZ_LANGS, asr_engine.PARA_LANGS, asr_engine.SV_LANGS, asr_engine.V3_LANGS]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"overlap between routing sets {i} and {j}"


def test_expected_language_homes():
    assert "ja" in asr_engine.RZ_LANGS
    assert "en" in asr_engine.V3_LANGS       # rz English is ALL-CAPS; must stay on v3
    assert "yue" in asr_engine.SV_LANGS      # LID can't detect yue; sv arbitrates
    assert "zh" in asr_engine.PARA_LANGS


# ---- script correction matrix ----------------------------------------------

def test_script_correction_matrix():
    f = asr_engine.script_corrected_lang
    assert f("en", "選手紹介のときのブルーカーペット") == "ja"   # CJK under latin tag
    assert f("ja", "NICETOMEETYOUEVERYONE") == "en"             # ASCII caps under ja tag
    assert f("vi", "안녕하세요 반갑습니다 여러분") == "ko"        # hangul under wrong tag
    assert f("yue", "值得在这座迷人村庄") == "yue"               # CJK under CJK tag: keep
    assert f("ja", "はい") == "ja"                               # short: keep
    assert f("en", "Nice to meet you.") == "en"


def test_has_kana():
    assert asr_engine._has_kana("漢字とかなの文")
    assert not asr_engine._has_kana("值得在这座迷人村庄")


# ---- sticky LID hysteresis --------------------------------------------------

def test_sticky_first_utterance_has_no_last_lang_yet():
    # last_lang=None (session bootstrap): accept immediately, no hold.
    lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
        "en", None, 3.0, 2.0, 2, None, 0)
    assert (lang, suppress, pend, cnt) == ("en", False, None, 0)


def test_sticky_same_lang_resets_pending():
    lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
        "ja", "ja", 3.0, 2.0, 2, "en", 1)
    assert (lang, suppress, pend, cnt) == ("ja", False, None, 0)


def test_sticky_single_misfire_is_held_not_switched():
    # A lone babble-noise misfire to "en" while the session is "ja" must be
    # held at "ja" for a long (>= min_switch_s) segment, and the omni
    # fallback must NOT be suppressed (it's presumed real speech, just
    # decoded under the wrong tier's model).
    lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 2, None, 0)
    assert lang == "ja"
    assert suppress is False
    assert (pend, cnt) == ("en", 1)


def test_sticky_short_misfire_suppresses_fallback():
    # Same, but the segment is shorter than min_switch_s (jingle/SFX case):
    # held language's empty decode must not be resurrected by omni.
    lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
        "en", "ja", 1.0, 2.0, 2, None, 0)
    assert lang == "ja"
    assert suppress is True


def test_sticky_confirmed_switch_after_two_consecutive_detections():
    # First "en" sighting: held. Second consecutive "en" sighting: switch.
    lang1, _, pend1, cnt1 = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 2, None, 0)
    assert lang1 == "ja" and (pend1, cnt1) == ("en", 1)

    lang2, suppress2, pend2, cnt2 = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 2, pend1, cnt1)
    assert lang2 == "en"
    assert suppress2 is False
    assert (pend2, cnt2) == (None, 0)


def test_missing_omni_fallback_does_not_terminate_session():
    routed = asr_engine.RoutedASR.__new__(asr_engine.RoutedASR)
    routed.last_lang = None
    routed._pending_lang = None
    routed._pending_count = 0
    routed.min_switch_s = 2.0
    routed.lid_switch_confirm = 2
    routed._unavailable = {"omni"}
    routed._route = lambda lang: (object(), "v3")
    routed._decode = lambda rec, samples, sample_rate: ""
    routed._get = lambda name: (_ for _ in ()).throw(asr_engine.ModelUnavailable(name))
    routed._replace = lambda text: text

    result = routed.transcribe(
        np.zeros(16000, dtype=np.float32), 16000,
        known_lang="en", speech_s=1.0,
    )

    assert result["text"] == ""
    assert result["tier"] == "v3"


def test_sticky_alternating_misfires_never_accumulate():
    # Two DIFFERENT wrong-language misfires in a row must not add up to a
    # switch -- each one resets the pending candidate.
    lang1, _, pend1, cnt1 = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 2, None, 0)
    assert (pend1, cnt1) == ("en", 1)

    lang2, _, pend2, cnt2 = asr_engine.resolve_sticky_lang(
        "zh", "ja", 3.0, 2.0, 2, pend1, cnt1)
    assert lang2 == "ja"          # still held at the session language
    assert (pend2, cnt2) == ("zh", 1)  # candidate reset to the new guess


def test_reset_session_clears_sticky_state():
    # reset_session() only touches plain attributes (no model calls), so it
    # can be exercised against a bare stand-in without loading any models.
    class _Stub:
        pass

    stub = _Stub()
    stub.last_lang = "ja"
    stub._pending_lang = "en"
    stub._pending_count = 1
    asr_engine.RoutedASR.reset_session(stub)
    assert stub.last_lang is None
    assert stub._pending_lang is None
    assert stub._pending_count == 0


def test_sticky_switch_confirm_one_disables_hysteresis():
    # switch_confirm=1 reproduces "switch immediately" (pre-hysteresis
    # behaviour), useful as a regression anchor for responsiveness.
    lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 1, None, 0)
    assert lang == "en"
    assert (pend, cnt) == (None, 0)


# ---- --lang-switch-guard actually gates switching (issue #2) ---------------

def test_sticky_short_detection_never_advances_pending_count():
    # A detection shorter than min_switch_s must not count toward
    # switch_confirm at all -- otherwise raising --lang-switch-guard would
    # have no effect on whether the session switches (issue #2).
    lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
        "zh", "ja", 1.9, 10.0, 2, None, 0)
    assert lang == "ja"
    assert suppress is True
    assert (pend, cnt) == (None, 0)  # no candidate accumulated


def test_sticky_short_detections_never_confirm_a_switch():
    # Reporter's scenario (issue #2): a ja-only session hit by repeated
    # short zh misfires under a large guard must stay on ja no matter how
    # many short misfires land in a row, because none of them ever advance
    # the confirmation counter.
    lang, pend, cnt = "ja", None, 0
    for speech_s in (9.1, 1.9, 1.5, 1.5):
        lang, suppress, pend, cnt = asr_engine.resolve_sticky_lang(
            "zh", "ja", speech_s, 10.0, 2, pend, cnt)
        assert lang == "ja"
        assert suppress is True
        assert (pend, cnt) == (None, 0)


def test_sticky_short_detection_does_not_reset_a_real_candidate():
    # A long (real) candidate detection starts accumulating...
    lang1, _, pend1, cnt1 = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 2, None, 0)
    assert (pend1, cnt1) == ("en", 1)

    # ...and a short, unrelated blip must not wipe it out.
    lang2, suppress2, pend2, cnt2 = asr_engine.resolve_sticky_lang(
        "zh", "ja", 0.5, 2.0, 2, pend1, cnt1)
    assert lang2 == "ja"
    assert suppress2 is True
    assert (pend2, cnt2) == ("en", 1)  # unchanged

    # The original candidate can still confirm on its next long detection.
    lang3, suppress3, pend3, cnt3 = asr_engine.resolve_sticky_lang(
        "en", "ja", 3.0, 2.0, 2, pend2, cnt2)
    assert lang3 == "en"
    assert (pend3, cnt3) == (None, 0)


def test_sticky_long_detections_still_confirm_a_switch_under_large_guard():
    # Detections at or above the guard length behave exactly as before:
    # switch_confirm consecutive ones confirm a genuine switch.
    lang1, _, pend1, cnt1 = asr_engine.resolve_sticky_lang(
        "zh", "ja", 10.0, 10.0, 2, None, 0)
    assert (pend1, cnt1) == ("zh", 1)

    lang2, suppress2, pend2, cnt2 = asr_engine.resolve_sticky_lang(
        "zh", "ja", 10.0, 10.0, 2, pend1, cnt1)
    assert lang2 == "zh"
    assert suppress2 is False
    assert (pend2, cnt2) == (None, 0)
