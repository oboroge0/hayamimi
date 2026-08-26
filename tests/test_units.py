"""Fast regression tests for the model-free logic.

Run with: .venv/Scripts/python -m pytest tests -q
No ASR/MT models are loaded; these guard the plumbing that past iterations
broke or nearly broke (preroll bleed, replacement parsing, digit guard,
routing table consistency).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from realtime_transcribe import (AudioHistory, PREROLL_S, digits_consistent,
                                 translate_by_sentence)
import asr_engine
import translate_m2m


class FakeTranslator:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def translate(self, text):
        self.calls.append(text)
        return self.mapping.get(text, text)


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


# ---- M2M-100 target acceptance / validation tiers ---------------------------

def test_is_supported_target_accepts_known_m2m100_codes():
    # __zh__ / __ko__ / __es__ / __fr__ all exist in the model's own vocabulary.
    assert translate_m2m.is_supported_target("zh")
    assert translate_m2m.is_supported_target("ko")
    assert translate_m2m.is_supported_target("es")
    assert translate_m2m.is_supported_target("fr")


def test_is_supported_target_rejects_unknown_code():
    assert not translate_m2m.is_supported_target("xx")
    assert not translate_m2m.is_supported_target("not-a-lang-code")


def test_is_supported_target_missing_model_dir_returns_false():
    # A bad model_dir must fail closed (no vocab file to check against), not raise.
    assert not translate_m2m.is_supported_target("zh", model_dir="/no/such/dir")


def test_validated_targets_are_a_subset_of_supported():
    for lang in translate_m2m.VALIDATED_TARGETS:
        assert translate_m2m.is_supported_target(lang), f"{lang} is VALIDATED but not accepted by the model"


def test_validated_targets_tier_matches_measured_set():
    # zh/ko/es have measured chrF (docs/TRANSLATE_M2M.md); an arbitrary M2M-100
    # target with no measurement (e.g. fr) must stay out of the validated tier.
    assert {"zh", "ko", "es"} <= translate_m2m.VALIDATED_TARGETS
    assert "fr" not in translate_m2m.VALIDATED_TARGETS


def test_translator_rejects_unsupported_target_before_loading_model():
    with pytest.raises(ValueError):
        translate_m2m.TranslatorM2M("xx")


def test_default_beam_size_used_for_unvalidated_targets():
    # es/fr etc. have no dedicated entry in BEAM_SIZE_BY_TARGET; they must fall
    # back to DEFAULT_BEAM_SIZE rather than KeyError.
    assert translate_m2m.BEAM_SIZE_BY_TARGET.get("fr", translate_m2m.DEFAULT_BEAM_SIZE) == translate_m2m.DEFAULT_BEAM_SIZE
    assert "es" not in translate_m2m.BEAM_SIZE_BY_TARGET  # measured via the fallback, not a dedicated tuning


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


def test_switch_confirm_one_and_guard_zero_fully_disable_the_lock():
    # --lid-switch-confirm 1 --lang-switch-guard 0: the combination that
    # should turn the sticky-LID lock fully off. A single misfire to "zh"
    # (even a very short one -- guard=0 means min_switch_s=0.0, so no
    # detection is ever "too short" to count) switches immediately, and the
    # very next detection back to "ja" switches back immediately too. This
    # is the escape hatch for validating whether the lock -- not just its
    # tuning -- is the right call for a given setup.
    lang1, suppress1, pend1, cnt1 = asr_engine.resolve_sticky_lang(
        "zh", "ja", 0.3, 0.0, 1, None, 0)
    assert lang1 == "zh"
    assert suppress1 is False
    assert (pend1, cnt1) == (None, 0)

    lang2, suppress2, pend2, cnt2 = asr_engine.resolve_sticky_lang(
        "ja", "zh", 0.3, 0.0, 1, pend1, cnt1)
    assert lang2 == "ja"
    assert suppress2 is False
    assert (pend2, cnt2) == (None, 0)


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


# ---- dual-LID switch confirmation (docs/LID.md) ----------------------------

def test_sv_lid_tag_normalizes_bracketed_tag():
    assert asr_engine.sv_lid_tag("<|ja|>") == "ja"
    assert asr_engine.sv_lid_tag("<|yue|>") == "yue"
    assert asr_engine.sv_lid_tag("<|unk|>") == ""


def test_dual_confirm_same_lang_is_a_noop():
    lang, switched = asr_engine.resolve_dual_confirm("ja", "ja", 3.0, "ja")
    assert (lang, switched) == ("ja", False)


def test_dual_confirm_bootstrap_trusts_probe_over_whisper_misfire():
    # LID.md real-mic accident scenario: session start, whisper-tiny says
    # "zh" (wrong), SenseVoice's own probe says "ja" (right) -- the very
    # first segment must decode as "ja", not "zh".
    lang, switched = asr_engine.resolve_dual_confirm("zh", None, 3.0, "ja")
    assert lang == "ja"
    assert switched is False  # the two LIDs disagreed; not a confirmed agreement


def test_dual_confirm_bootstrap_agreement_marks_switched():
    lang, switched = asr_engine.resolve_dual_confirm("en", None, 3.0, "en")
    assert (lang, switched) == ("en", True)


def test_dual_confirm_holds_current_lang_on_disagreement():
    # LID.md scenario: a ja-only session hit by a whisper-tiny "en" misfire;
    # SenseVoice's probe still says "ja" -> stays on "ja", no switch.
    lang, switched = asr_engine.resolve_dual_confirm("en", "ja", 1.5, "ja")
    assert lang == "ja"
    assert switched is False


def test_dual_confirm_switches_immediately_on_agreement():
    # LID.md scenario: whisper-tiny and SenseVoice both say "en" -> switch
    # immediately, no length or repeat-count gate needed (both LIDs agreeing
    # measured 85-98% accurate at every length in docs/LID.md table 3).
    lang, switched = asr_engine.resolve_dual_confirm("en", "ja", 1.0, "en")
    assert lang == "en"
    assert switched is True


def test_dual_confirm_ignores_sub_probe_length_even_on_agreement():
    lang, switched = asr_engine.resolve_dual_confirm("en", "ja", 0.3, "en")
    assert lang == "ja"
    assert switched is False


def test_dual_confirm_mismatch_at_bootstrap_falls_back_to_whisper_guess():
    # if the caller couldn't get any SenseVoice tag at all (e.g. --minimal
    # install without the sv model), sv_lang is "" -- bootstrap must still
    # resolve to something rather than staying silent forever.
    lang, switched = asr_engine.resolve_dual_confirm("zh", None, 3.0, "")
    assert lang == "zh"
    assert switched is False


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
