"""Routed multilingual ASR engine on top of sherpa-onnx.

Piper-style tiered catalog: a whisper-tiny spoken-language identifier routes
each audio segment to the best model for that language.

  tier 0  ja                   -> ReazonSpeech k2 zipformer (best real-speech ja, fastest)
  tier 1  zh                   -> Paraformer-zh (best real-speech zh)
  tier 1  ko/yue               -> SenseVoice small
  tier 2  en + 24 EU langs     -> Parakeet TDT v3 (casing + punctuation)
  tier 3  everything else      -> Omnilingual ASR 300M CTC (1600+ languages)

Models are loaded lazily on first use and, when `max_resident` is set, the
least-recently-used ones are unloaded so memory stays bounded no matter how
many languages a session wanders through.
"""
import difflib
import glob
import os
import re
import sys
import threading
import time
from typing import Callable

import numpy as np
import sherpa_onnx

import itn_cjk

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
V3_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
SV_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
OMNI_MODEL_DIR = os.path.join(MODELS_DIR, "omnilingual-300m-ctc-int8")
WHISPER_TINY_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-whisper-tiny")
RZ_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17")
PARA_ZH_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-paraformer-zh-int8-2025-10-07")
PJA_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8")
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

# ReazonSpeech ja-en zipformer: on real broadcast Japanese it beats even
# whisper-turbo (CER 8.6% vs 13.8%) at RTF 0.02. See docs/eval/eval_real.md.
# English goes to v3 instead: rz outputs unpunctuated ALL-CAPS English
# (WER 1.6% vs v3's 2.5%, but v3's casing/punctuation reads far better).
RZ_LANGS = {"ja"}

# The ja tier's sherpa-onnx OfflineRecognizerConfig values, hoisted out of
# _build_reazon() so scripts/dump_ja_config.py can report the configuration a
# ja-only re-implementation has to match WITHOUT loading a model (see
# docs/spec/ja_pipeline.ja.md). _build_reazon() below is the only other reader; these
# are the same literals it used before, moved rather than changed.
RZ_MODEL_TYPE = "zipformer"
RZ_DECODING_METHOD = "modified_beam_search"
RZ_MODELING_UNIT = "cjkchar"
RZ_HOTWORDS_SCORE = 2.0
# The four files the recognizer is built from, as the glob patterns _find()
# resolves them with, in the order from_transducer() takes them.
RZ_MODEL_FILES = {
    "encoder": "encoder-*.int8.onnx",
    "decoder": "decoder-*.int8.onnx",
    "joiner": "joiner-*.int8.onnx",
    "tokens": "tokens.txt",
}

# Paraformer-zh beats SenseVoice on real Chinese (CER 5.6% vs 7.5%); the
# dedicated Korean zipformer is worse (30%), so ko stays on SenseVoice.
# See docs/eval/eval_real_zhko.md.
PARA_LANGS = {"zh"}

# SenseVoice small coverage (built-in ITN and punctuation).
SV_LANGS = {"ko", "yue"}

# Languages covered by the Parakeet-TDT-0.6B-v3 multilingual model.
V3_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}

# The language codes RoutedASR routes to a SPECIFIC tier (as opposed to
# falling through to the 1600-language Omnilingual generalist, which has no
# fixed code list of its own). set_forced_lang() (GitHub issue #29's runtime
# API) validates against this set: forcing the session onto a code outside
# it would silently ride the omni fallback for every segment, which is very
# unlikely to be what a caller setting --lang/--mode single intended.
ROUTABLE_LANGS = RZ_LANGS | PARA_LANGS | SV_LANGS | V3_LANGS

LID_MAX_SECONDS = 4.0  # only feed the first N seconds of a segment to the LID model

# --- head-dropout retry ----------------------------------------------------
#
# The catalog's offline recognizers are trained on single caption-length
# utterances and can collapse a buffer that holds SEVERAL of them into one:
# given a multi-sentence clip in a single create_stream/decode_stream call,
# the leading sentences sometimes never come out. Measured on FLEURS ja test
# with ReazonSpeech (int8, fp16 and fp32 all identical; greedy and
# modified_beam_search all identical): clip 15 (18.3s, 4 sentences) came back
# as just "植物がなければ動物は生きていけません", CER 0.67.
#
# Splitting the buffer at its internal silences and decoding one piece per
# call recovers clip 15 (CER 0.67 -> 0.11). But doing that UNCONDITIONALLY is
# a net loss: an external FLEURS 5x100 A/B measured ja 8.6% -> 9.9%,
# en 9.4% -> 10.2%, ko 8.1% -> 9.1% -- on ja, 16 clips improved and 26 got
# worse, because words that straddle a piece boundary get lost and the
# per-piece decodes are shorter and therefore worse-conditioned. Most buffers
# do not have the defect and must not be touched.
#
# So the split is a RETRY, not a decode path: the whole buffer is decoded
# exactly as it always was, and only if the result looks like it dropped its
# leading content (see _looks_truncated) is the split attempted, and only if
# the split result is plausibly better (see _retry_is_better) is it kept.
# When in doubt the whole-buffer decode wins.
#
# The live path is untouched twice over: it decodes one VAD segment per call
# (a speech run with no internal silence >= min_silence, so there is nothing
# to split), and a single VAD segment is short and dense enough never to trip
# the suspicion gate in the first place.
#
# SEGMENT_MIN_SILENCE_S is deliberately equal to the live VAD's min_silence
# (0.35s in realtime_transcribe.build_vad), so a buffer the live VAD produced
# splits into exactly one piece and the retry becomes a no-op.
SEGMENT_MIN_S = 4.0             # below this a buffer can't hold two real utterances
SEGMENT_MIN_SILENCE_S = 0.35    # only cut at silences at least this long
SEGMENT_MIN_SPEECH_S = 0.25     # ignore shorter blips (same as the live VAD)
SEGMENT_PAD_S = 0.35            # real context kept on both sides of a piece
SEGMENT_FALLBACK_CHUNK_S = 8.0  # no silero_vad.onnx: coarse fixed-length chunks
SEGMENT_FALLBACK_OVERLAP_S = 0.25
SEGMENT_SAMPLE_RATE = 16000     # silero_vad.onnx is a 16k model

# Output density (alphanumeric characters per second of SPEECH) below which a
# decode is suspected of having dropped its leading content. Calibrated on
# FLEURS ja test through the production path with the ReazonSpeech tier: 60
# random clips scored between 3.46 and 14.22 chars/s (median 6.54), while the
# known head-dropout clip 15 scored 1.70. 2.4 is the log-midpoint of that
# gap: 41% above the defect, 31% below the lowest healthy clip seen.
#
# The denominator is SPEECH seconds, not buffer seconds, because how much
# silence a buffer carries is an accident of how it was cut -- FLEURS clips
# carry seconds of it, a live VAD segment carries almost none, and the same
# defect has to look the same in both. Since speech <= buffer, the buffer
# figure is always the smaller of the two, so it is safe to use as a free
# pre-gate that decides whether running the VAD at all is worth it: it can
# raise a false alarm but it can never miss one the speech figure would
# catch.
#
# Latin-script output packs fewer phonemes per character, so the same speech
# yields roughly 2.5x more characters; DENSITY_FLOOR_LATIN is that ratio
# applied to the measured CJK floor. Validated against the FLEURS en test
# distribution (100 clips, full-model run): healthy clips sit at 6.9-15.6
# chars/s even on the pessimistic buffer-seconds basis (median 10.7), so
# 6.0 does not intrude on the healthy range. The only two clips below it
# were short single sentences inside long mostly-silent buffers -- exactly
# the case the speech-seconds denominator re-normalizes -- and the retry's
# tail-match acceptance is a second guard behind that.
#
# Density catches only the severe cases. FLEURS ja clip 324 also drops a
# sentence but scores well inside the healthy range; it is out of reach here
# (and out of reach of splitting anyway -- it says three sentences in one
# 5.1s breath with no pause Silero can see).
DENSITY_FLOOR_CJK = 2.4
DENSITY_FLOOR_LATIN = 6.0

# How much of the whole-buffer text must reappear in the retry text, and over
# how long a tail, for the retry to count as "the same utterance plus more"
# rather than a different reading of the audio.
RETRY_TAIL_CHARS = 12
RETRY_TAIL_MATCH = 0.6

# Languages written without spaces between words: their pieces are joined
# with nothing at all, everything else gets a single space.
NO_SPACE_LANGS = {"ja", "zh", "yue", "ko"}

# --- ja second-opinion agreement gate ---------------------------------------
#
# parakeet-tdt_ctc-0.6b-ja is more accurate than the ReazonSpeech tier on
# clean single-speaker audio (FLEURS ja CER 6.6% vs 7.4%) but transcribes
# off-caption speech (background commentary, adjacent utterances) on real
# broadcast audio, where the caption-trained ReazonSpeech stays on the main
# utterance (realset 31.2% vs 7.2%). When the two DISAGREE, that off-caption
# pickup is almost always why -- so agreement between them is a clean-audio
# signal. Simulated on both eval sets (hayamimi-paper candidate results):
# adopting parakeet only when the hypotheses' mutual CER <= 0.25 scores
# FLEURS 5.9% / realset 5.8%, better than either model alone on both.
SECOND_OPINION_THRESHOLD = 0.25

_AGREE_STRIP_RE = re.compile(r"[\s。、．，,\.!?！？]")


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def hyp_agreement_cer(a: str, b: str) -> float:
    """Character error rate between two hypotheses (whitespace/punct-blind).

    Symmetric enough for gating: the denominator is the first argument's
    length, and callers pass the primary (ReazonSpeech) text first.
    """
    a = _AGREE_STRIP_RE.sub("", a)
    b = _AGREE_STRIP_RE.sub("", b)
    if not a:
        return 0.0 if not b else 1.0
    return _levenshtein(a, b) / len(a)


def choose_second_opinion(primary: str, second: str,
                          threshold: float = SECOND_OPINION_THRESHOLD) -> tuple[str, bool]:
    """Return (chosen_text, used_second).

    Adopt the second opinion only when it broadly agrees with the primary
    (mutual CER <= threshold). An empty or wildly different second opinion --
    the background-speech failure mode -- leaves the primary standing.
    """
    if not second.strip() or not primary.strip():
        return primary, False
    if hyp_agreement_cer(primary, second) <= threshold:
        return second, True
    return primary, False


def _find(model_dir: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


def _build_reazon(threads: int, hotwords_file: str = "",
                  hotwords_score: float = RZ_HOTWORDS_SCORE):
    # modified_beam_search: CER 8.6% -> 5.8% on real broadcast ja for +25%
    # decode time (still 37x realtime). v3/en showed no gain and stays greedy.
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=_find(RZ_MODEL_DIR, RZ_MODEL_FILES["encoder"]),
        decoder=_find(RZ_MODEL_DIR, RZ_MODEL_FILES["decoder"]),
        joiner=_find(RZ_MODEL_DIR, RZ_MODEL_FILES["joiner"]),
        tokens=os.path.join(RZ_MODEL_DIR, RZ_MODEL_FILES["tokens"]),
        num_threads=threads,
        model_type=RZ_MODEL_TYPE,
        decoding_method=RZ_DECODING_METHOD,
        hotwords_file=hotwords_file,
        hotwords_score=hotwords_score,
        modeling_unit=RZ_MODELING_UNIT,
    )


def _build_paraformer_zh(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=_find(PARA_ZH_DIR, "model*.onnx"),
        tokens=os.path.join(PARA_ZH_DIR, "tokens.txt"),
        num_threads=threads,
    )


def _build_sense_voice(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=_find(SV_MODEL_DIR, "model*.onnx"),
        tokens=os.path.join(SV_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
        use_itn=True,
        language="",  # auto: SenseVoice has its own internal LID for its 5 langs
    )


def _build_v3_recognizer(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=_find(V3_MODEL_DIR, "encoder*.onnx"),
        decoder=_find(V3_MODEL_DIR, "decoder*.onnx"),
        joiner=_find(V3_MODEL_DIR, "joiner*.onnx"),
        tokens=os.path.join(V3_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
        model_type="nemo_transducer",
    )


def _build_parakeet_ja(threads: int):
    # Second-opinion model for the ja refine gate (docs: fix_plan.md /
    # hayamimi-paper candidate results). Better than ReazonSpeech on clean
    # read speech but transcribes off-caption background speech on broadcast
    # audio, so it is only ever consulted through choose_second_opinion().
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=_find(PJA_MODEL_DIR, "model*.onnx"),
        tokens=os.path.join(PJA_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
    )


def _build_omnilingual(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=_find(OMNI_MODEL_DIR, "model*.onnx"),
        tokens=os.path.join(OMNI_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
    )


def _build_lid(threads: int):
    whisper_cfg = sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
        encoder=_find(WHISPER_TINY_DIR, "tiny-encoder.int8.onnx"),
        decoder=_find(WHISPER_TINY_DIR, "tiny-decoder.int8.onnx"),
    )
    cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(whisper=whisper_cfg, num_threads=threads)
    return sherpa_onnx.SpokenLanguageIdentification(cfg)


def _has_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def script_corrected_lang(tagged: str, text: str) -> str:
    """Correct an LID tag that contradicts the script of the decoded text."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return tagged
    cjk = sum(1 for c in letters if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    hangul = sum(1 for c in letters if "가" <= c <= "힯")
    if hangul / len(letters) > 0.3 and tagged != "ko":
        return "ko"
    if tagged == "ko" and hangul == 0 and len(letters) >= 4:
        # SenseVoice Korean output is hangul; hangul-free "ko" is a mislabel
        return "zh" if cjk / len(letters) > 0.3 else "en"
    frac = cjk / len(letters)
    if frac > 0.3 and tagged not in ("ja", "zh", "yue", "ko"):
        return "ja"
    if frac < 0.05 and tagged == "ja" and len(letters) >= 8:
        return "en"
    return tagged



# The 5 languages SenseVoice's own internal LID can arbitrate (its model
# directory name: zh-en-ja-ko-yue). whisper-tiny can never actually emit
# "yue" as a candidate (see the zh/yue arbitration in transcribe() below),
# but it's listed here for documentation symmetry with docs/eval/lid.md.
DUAL_CONFIRM_LANGS = {"ja", "en", "zh", "ko", "yue"}

# A whisper-tiny candidate shorter than this is presumed non-speech noise
# (jingle/SFX/misfire) and never confirms a switch even if SenseVoice agrees.
MIN_PROBE_S = 0.5

_SV_LID_CODES = ("ja", "en", "zh", "ko", "yue")


def sv_lid_tag(sv_tag: str) -> str:
    """Normalize SenseVoice's raw '<|xx|>'-style language tag to a bare code
    ("ja"/"en"/"zh"/"ko"/"yue"), or "" if none of the 5 codes appear."""
    for code in _SV_LID_CODES:
        if code in sv_tag:
            return code
    return ""


def resolve_dual_confirm(
    lang: str, last_lang: str | None, speech_s: float | None, sv_lang: str,
) -> tuple[str, bool]:
    """Dual-LID switch confirmation for the 5 SenseVoice-covered languages.

    docs/eval/lid.md measured whisper-tiny alone at only 59-65% LID accuracy at
    2 seconds (far worse under babble noise -- 59%), but whisper-tiny AND
    SenseVoice's own internal LID AGREEING on the same language hits
    85-98% accuracy at the same length ("一致時正解率" in the LID.md
    tables, which beats "単独LID全体正解率" at every measured length in
    both clean and babble_snr10). So instead of gating a switch on segment
    length or repeat-count (the old resolve_sticky_lang hysteresis), gate
    it on the two independent LID signals agreeing: length and repeat-count
    add nothing once both models agree, and agreement is available from the
    very first segment.

    `sv_lang` is the caller's already-computed SenseVoice LID tag for this
    exact audio (via sv_lid_tag() on its decode's .lang field) -- this
    function is pure and makes no model calls itself.

    Session bootstrap (last_lang is None) has no current language to hold
    at while waiting for agreement, so it resolves directly to `sv_lang`:
    SenseVoice's own LID is already more accurate alone than whisper-tiny
    alone (docs/eval/lid.md table 2 vs table 1), and this is confirmed by
    agreement whenever sv_lang == lang too (this is the case tests exercise
    where whisper-tiny misfires "zh" but SenseVoice correctly says "ja" --
    the first segment must decode as "ja", not "zh", not silence).

    A candidate shorter than MIN_PROBE_S is presumed non-speech noise and
    never confirms a switch, even on agreement. This still applies at
    bootstrap: this function still returns a best-effort `resolved` for the
    caller to decode THIS segment with, but callers must not treat a
    too-short bootstrap resolution as the session's confirmed language --
    RoutedASR.transcribe() checks `speech_s < MIN_PROBE_S` on the same
    bootstrap call and skips seeding self.last_lang from it, so a lone
    jingle/SFX misfire on segment 1 can't lock the whole session onto
    whatever it happened to guess.

    Returns (resolved_lang, switched) -- `switched` is True only when this
    call is the reason the session's language changed (both LIDs agreed on
    something new), so callers can clear any stale hysteresis state or
    count corrections.
    """
    if lang == last_lang:
        return lang, False
    too_short = speech_s is not None and speech_s < MIN_PROBE_S
    if last_lang is None:
        # no current language to hold at: trust the probe's own judgment
        resolved = sv_lang or lang
        return resolved, (not too_short and sv_lang == lang)
    if too_short:
        return last_lang, False
    if sv_lang == lang:
        return lang, True
    if lang == "zh" and sv_lang == "yue":
        # whisper-tiny cannot emit "yue" as a candidate -- it folds Cantonese
        # into "zh" (see DUAL_CONFIRM_LANGS docstring above). When SenseVoice's
        # own LID says "yue" on the same audio, that's whisper's only possible
        # spelling of agreement, so treat it as a confirmed switch to "yue"
        # rather than "zh".
        return "yue", True
    return last_lang, False


# The refine pass re-runs whisper-tiny LID on a merged utterance group,
# trusting it more than the fast path's per-segment vote because the group
# is (usually) longer. But a Refiner "group" can be a single short segment
# sitting alone between silence gaps -- not a real multi-segment utterance
# -- so that assumption doesn't automatically hold. docs/eval/lid.md's table 1
# shows whisper-tiny-ALONE accuracy hasn't clearly separated from chance for
# several languages below ~2.5s (babble_snr10 overall: 44% at 1.5s, 59% at
# 2.0s, 65% at 2.5s); below that, a lone re-judgment is a coin flip that can
# undo a live decision the dual-LID-confirmed bootstrap path already got
# right. Real-mic incident: a 1.9s segment correctly resolved live as "ko"
# (bootstrap dual-confirm) sat alone in its own refine group and got
# whisper-tiny-alone re-judged back to "ru", reproducing the exact garbled
# collapse the bootstrap fix exists to prevent.
REFINE_MIN_REGROUP_S = 2.5


def resolve_refine_lang(
    current_lang: str, whisper_lang: str, sv_lang: str, group_duration_s: float,
) -> tuple[str, bool]:
    """Decide whether the refine pass's LID re-judgment should override the
    fast path's per-segment language for this utterance group.

    Applies the same dual-LID confirmation as the live path
    (resolve_dual_confirm): a whisper-tiny re-judgment that disagrees with
    the group's current language is only accepted when SenseVoice's own
    probe on the SAME merged audio agrees with whisper-tiny. Below
    REFINE_MIN_REGROUP_S total group duration, the re-judgment is skipped
    outright regardless of agreement -- callers should not even bother
    running the SenseVoice probe in that case.

    `sv_lang` is the caller's SenseVoice LID tag for the merged group audio
    (via sv_lid_tag() on its decode's .lang field); this function is pure
    and makes no model calls itself.

    Returns (resolved_lang, changed).
    """
    if whisper_lang == current_lang:
        return current_lang, False
    if group_duration_s < REFINE_MIN_REGROUP_S:
        return current_lang, False
    if sv_lang == whisper_lang:
        return whisper_lang, True
    return current_lang, False


def resolve_sticky_lang(
    lang: str, last_lang: str | None, speech_s: float | None,
    min_switch_s: float, switch_confirm: int,
    pending_lang: str | None, pending_count: int,
    bootstrap_probe_lang: str | None = None,
) -> tuple[str, bool, str | None, int]:
    """Sticky-LID hysteresis: decide whether to accept a new LID detection
    as a real language switch, or hold the session's current language.

    A single new-language detection can be a babble-noise misfire
    (docs/eval/noise.md -- whisper-tiny LID exposes no confidence score to
    threshold on) or a jingle/SFX blip (docs/eval/video_test.md) rather than a
    genuine switch. A real switch -- the speaker changing language, or a new
    speaker -- repeats the SAME new language on the next detection, while a
    misfire lands on a random one. `switch_confirm` CONSECUTIVE detections
    of one new language are required before switching; staying on the
    current language needs no confirmation (asymmetric, so noise can't lock
    the session onto a wrong language). This costs a genuine switch at most
    `switch_confirm - 1` segments of latency.

    `min_switch_s` (--lang-switch-guard) is the noise filter on each
    individual candidate detection: a new-language segment shorter than this
    is presumed non-speech (jingle/SFX/misfire) and does NOT advance the
    switch_confirm counter at all -- it neither starts nor extends a
    pending candidate, and it doesn't reset one either, since a real
    candidate already accumulating shouldn't be wiped out by an unrelated
    short blip. This is what makes --lang-switch-guard actually control
    switch stickiness (GitHub issue #2): only detections at or above the
    guard length can ever confirm a switch. It also suppresses the
    omnilingual fallback for that segment, so a held language's empty
    decode isn't resurrected by it.

    Session bootstrap (last_lang is None) used to instant-accept the very
    first detection unconditionally. That let a single whisper-tiny
    misfire to a language SenseVoice can't arbitrate (e.g. "ru") seed the
    whole session with a collapsed decode, bypassing the dual-LID
    confirmation used for the 5 SenseVoice-covered languages entirely (a
    real-mic incident: whisper-tiny said "ru" on the first segment of a
    Japanese session and the session never recovered). Bootstrap now goes
    through the SAME switch_confirm accumulation as any other switch --
    `lang` must repeat `switch_confirm` times (each >= min_switch_s) before
    it becomes the session's language. `bootstrap_probe_lang` is the
    caller's SenseVoice probe result for THIS exact audio (only meaningful
    at bootstrap, since a probe is always cheap enough to run on the very
    first segment): while no candidate has accumulated switch_confirm
    detections, segments decode using bootstrap_probe_lang instead of
    blindly trusting whisper-tiny's possibly-wrong candidate, since
    SenseVoice alone already measures more accurate than whisper-tiny alone
    (docs/eval/lid.md table 2 vs table 1). If no probe was available
    (bootstrap_probe_lang falsy, e.g. --minimal install), this falls back
    to whisper-tiny's own candidate, same as before.

    Returns (resolved_lang, suppress_fallback, new_pending_lang, new_pending_count).
    """
    if last_lang is not None and lang == last_lang:
        return lang, False, None, 0

    # while nothing has been confirmed yet, decode with the best available
    # guess: the established session language, or (at bootstrap) the
    # SenseVoice probe's own judgment if the caller has one
    fallback = last_lang if last_lang is not None else (bootstrap_probe_lang or lang)

    is_short = speech_s is not None and speech_s < min_switch_s
    if is_short:
        return fallback, True, pending_lang, pending_count

    if lang == pending_lang:
        pending_count += 1
    else:
        pending_lang, pending_count = lang, 1

    if pending_count < switch_confirm:
        # Hold the session language for this segment; it's a genuine-speech
        # candidate (>= min_switch_s) merely decoded under the wrong tier's
        # model, so let the omni fallback have a shot if that specialist
        # draws a blank.
        return fallback, False, pending_lang, pending_count

    return lang, False, None, 0


def _cjkchar_units(phrase: str) -> list[str]:
    """Split a hotword phrase the way sherpa-onnx's cjkchar modeling_unit
    encodes it: each CJK character becomes its own lookup unit, and runs of
    non-CJK, non-whitespace characters are grouped into whole-word units
    (matches the "屈 足 湖" / "GANKE FES" splits seen in sherpa-onnx's own
    "Cannot find ID for token" warnings)."""
    units: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            units.append("".join(buf))
            buf.clear()

    for ch in phrase:
        if ch.isspace():
            flush()
            continue
        if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯":
            flush()
            units.append(ch)
        else:
            buf.append(ch)
    flush()
    return units


def _load_token_vocab(tokens_path: str) -> set[str]:
    vocab: set[str] = set()
    if not os.path.isfile(tokens_path):
        return vocab
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # "<token> <id>"; rsplit so a token that itself contains a
            # literal space (rare) still separates cleanly from the id.
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                vocab.add(parts[0])
    return vocab


def check_hotwords_encodable(hotwords_path: str, tokens_path: str) -> tuple[int, int]:
    """Return (total_hotwords, num_unencodable) for the cjkchar modeling_unit
    used by the ja (ReazonSpeech) tier.

    sherpa-onnx encodes each hotword by looking up its cjkchar units
    (see `_cjkchar_units`) directly against tokens.txt. ReazonSpeech's
    tokens.txt is byte-level BPE, not a cjkchar vocabulary, so every lookup
    normally misses -- sherpa-onnx only reports this as stderr warnings and
    still exits 0 (GitHub issue #1). This lets callers surface that loudly
    instead of leaving it buried in stderr.
    """
    if not hotwords_path or not os.path.isfile(hotwords_path):
        return 0, 0
    vocab = _load_token_vocab(tokens_path)
    total = 0
    bad = 0
    with open(hotwords_path, encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase or phrase.startswith("#"):
                continue
            total += 1
            units = _cjkchar_units(phrase)
            if not units or any(u not in vocab for u in units):
                bad += 1
    return total, bad


def _load_replacements(path: str) -> list[tuple[str, str]]:
    """User dictionary: one "wrong=right" (or tab/arrow-separated) pair per line."""
    if not path:
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("=", "	", "→"):
                if sep in line:
                    wrong, right = line.split(sep, 1)
                    pairs.append((wrong.strip(), right.strip()))
                    break
    return pairs


# a key file per model: checked BEFORE building, because sherpa-onnx's C++
# layer exits the process (not a catchable exception) on an empty model path
_KEY_FILES = {
    "rz": (RZ_MODEL_DIR, RZ_MODEL_FILES["encoder"]),
    "pz": (PARA_ZH_DIR, "model*.onnx"),
    "sv": (SV_MODEL_DIR, "model*.onnx"),
    "v3": (V3_MODEL_DIR, "encoder*.onnx"),
    "omni": (OMNI_MODEL_DIR, "model*.onnx"),
    "pja": (PJA_MODEL_DIR, "model*.onnx"),
    "lid": (WHISPER_TINY_DIR, "tiny-encoder.int8.onnx"),
}


def _model_present(name: str) -> bool:
    d, pat = _KEY_FILES[name]
    return bool(_find(d, pat))


def _build_lid_guarded(threads: int):
    """The single guarded entry point for the whisper-tiny LID model.

    Every native sherpa-onnx construction in this module must be preceded by
    a _model_present() check: sherpa-onnx's C++ layer calls the process's
    exit() -- not a catchable Python exception -- when handed an empty or
    invalid model path (see _KEY_FILES' comment and ModelUnavailable below).
    _get() already funnels the six _BUILDERS recognizers through exactly
    that check; the LID model is built eagerly in RoutedASR.__init__
    (outside _get()'s lazy-load path), so it needs its own guard here rather
    than silently reaching sherpa_onnx.SpokenLanguageIdentification(...)
    with a possibly-missing encoder/decoder path.
    """
    if not _model_present("lid"):
        raise ModelUnavailable("lid")
    return _build_lid(threads)


_BUILDERS = {
    "rz": _build_reazon,
    "pz": _build_paraformer_zh,
    "sv": _build_sense_voice,
    "v3": _build_v3_recognizer,
    "omni": _build_omnilingual,
    "pja": _build_parakeet_ja,
}

# preload priority when a residency cap is in effect
_PRELOAD_ORDER = ("pz", "sv", "v3", "omni")


class ModelUnavailable(RuntimeError):
    """Raised when a model tier is not present on disk (--minimal install)."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class RoutedASR:
    """Lazily loads catalog models and routes each segment by detected language.

    max_resident bounds how many recognizers besides tier-0 ("rz", always kept:
    it is the ja/en primary and the default draft model) stay in memory; the
    least recently used one is dropped when the cap would be exceeded.
    """

    def __init__(self, threads: int = 4, warmup: bool = True, preload: bool = True,
                 max_resident: int | None = None, punctuate: bool = True,
                 hotwords_file: str = "", replace_file: str = "",
                 lid_switch_confirm: int = 2, dual_confirm: bool = True,
                 forced_lang: str | None = None,
                 ja_second_opinion: bool = False,
                 agree_threshold: float = SECOND_OPINION_THRESHOLD,
                 on_event: "Callable[[dict], None] | None" = None):
        # on_event (GitHub issue #29): the engine's structured-event sink.
        # Set before ANY other construction step below so every model build
        # in this constructor (the lid build a few lines down included) is
        # observable, not just the ones reached later via _get(). Optional
        # so every existing caller (eval scripts, tests) that constructs a
        # RoutedASR without it keeps working unchanged -- _emit() below is a
        # no-op when this is None. realtime_transcribe.main() passes
        # hub.publish so model_load/model_fallback/warning events reach the
        # same EventHub every other structured event goes through.
        self._on_event = on_event
        self._threads = threads
        self.dual_confirm = dual_confirm  # --mode balanced (default); False = --mode fast
        self.forced_lang = forced_lang    # --mode single: skip all LID/switch logic
        self._models: dict[str, object] = {}
        self._last_used: dict[str, float] = {}
        self._max_resident = max_resident
        self._punctuate = punctuate
        self._punct = None
        self._ko_spacer = None
        self._ko_spacer_ok = True
        self._hotwords_file = hotwords_file
        self._warn_hotwords_encodability(hotwords_file)
        self._replacements = _load_replacements(replace_file)
        self._itn_overrides = itn_cjk.EMPTY_OVERRIDES
        # refine-time ja second opinion (parakeet-ja agreement gate); live
        # finals are never affected -- see transcribe()'s second_opinion arg.
        self._ja_second_opinion = ja_second_opinion
        self._agree_threshold = agree_threshold
        self._pja_warned = False
        self._load_lock = threading.Lock()  # punct + registry bookkeeping
        self._seg_lock = threading.Lock()  # the offline-split VAD is stateful
        self._seg_vad = None          # lazily built VAD for offline splitting
        self._seg_vad_capacity = 0.0  # its buffer_size_in_seconds
        self._seg_vad_ok = True       # False once silero_vad.onnx proves missing
        self._model_locks = {name: threading.Lock() for name in _BUILDERS}
        self.last_lang = None  # sticky language from the most recent final
        self._unavailable: set[str] = set()  # models missing on disk (--minimal installs)
        self._fallback_warned: set[str] = set()  # requested names already model_fallback-reported
        self._pending_lang = None   # candidate new language awaiting confirmation
        self._pending_count = 0
        self.lid_switch_confirm = lid_switch_confirm  # consecutive detections to accept a switch
        t0 = time.perf_counter()
        self._emit({"type": "model_load", "model": "lid", "phase": "start", "ms": None})
        self.lid = _build_lid_guarded(threads)
        self._emit({"type": "model_load", "model": "lid", "phase": "done",
                    "ms": (time.perf_counter() - t0) * 1000})
        if warmup:
            # LID + tier-0 pay their one-time kernel/allocation costs here
            # so the first real segment isn't penalized.
            silence = np.zeros(16000, dtype=np.float32)
            self._identify_lang(silence, 16000)
            self._decode(self._get("rz"), silence, 16000)
        if preload:
            # pull the other tiers in on a daemon thread so the first
            # non-tier-0 utterance doesn't pay the ~2s model-load cost.
            threading.Thread(target=self._preload_rest, daemon=True).start()

    def _emit(self, event: dict) -> None:
        """Forward a structured event (model_load/model_fallback/warning) to
        the caller's on_event sink, if one was given. An event-hub failure
        (a broken listener, a full queue, whatever) must never break
        transcription, so any exception from the callback is swallowed
        here -- EventHub.publish() already does its own once-per-listener
        error reporting, so there is nothing useful left to do with it at
        this call site besides not propagating it.
        """
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    def _warn_hotwords_encodability(self, hotwords_file: str):
        """Print a loud, hard-to-miss warning if --hotwords entries can't be
        encoded against the ja (ReazonSpeech) tier's tokens.txt.

        sherpa-onnx only reports failed encodes as stderr warnings and still
        exits 0 with a normal-looking transcript, so this was silently doing
        nothing for every ReazonSpeech user until they happened to read
        stderr closely (GitHub issue #1). ReazonSpeech ships a byte-level
        BPE tokens.txt with no bpe.model, so there is currently no
        modeling_unit that encodes cjkchar-style hotwords against it; until
        that's fixed upstream (or hayamimi ships its own bpe.model), the
        best we can do is make the failure visible and point at --replace.

        GitHub issue #29: also emits a `warning` event (code
        "hotwords_unencodable") alongside the print, so an embedding app
        without a terminal to read stderr from still finds out.
        """
        if not hotwords_file:
            return
        tokens_path = os.path.join(RZ_MODEL_DIR, "tokens.txt")
        total, bad = check_hotwords_encodable(hotwords_file, tokens_path)
        if bad == 0:
            return
        if bad == total:
            message = (f"0/{total} hotwords could be encoded for the ja tier "
                       f"(ReazonSpeech uses byte-level BPE tokens, incompatible with "
                       f"modeling_unit=cjkchar) -- --hotwords will have NO EFFECT on ja "
                       f"output. Use --replace for post-hoc find/replace instead.")
            print(f"[hayamimi] WARNING: {message}")
        else:
            message = (f"{bad}/{total} hotwords cannot be encoded for the ja tier "
                       f"(ReazonSpeech uses byte-level BPE tokens); --hotwords will "
                       f"have no effect for these. Consider --replace instead.")
            print(f"[hayamimi] warning: {message}")
        self._emit({"type": "warning", "code": "hotwords_unencodable", "message": message})

    def _preload_rest(self):
        if self._punctuate:
            self.punct  # first: ja finals need this almost immediately
        self.ko_spacer  # cheap (~1s); fixes SenseVoice's over-split Korean
        silence = np.zeros(16000, dtype=np.float32)
        budget = None if self._max_resident is None else self._max_resident
        for name in _PRELOAD_ORDER:
            if budget is not None and budget <= 0:
                break
            try:
                self._decode(self._get(name), silence, 16000)
            except ModelUnavailable:
                continue  # --minimal install: this tier simply isn't there
            if budget is not None:
                budget -= 1

    @property
    def punct(self):
        """Japanese punctuation restorer (BERT ONNX); None if unavailable."""
        if self._punct is None and self._punctuate:
            with self._load_lock:
                if self._punct is None:
                    t0 = time.perf_counter()
                    self._emit({"type": "model_load", "model": "punct", "phase": "start",
                               "ms": None})
                    try:
                        from punct_ja import PunctuatorJa

                        self._punct = PunctuatorJa()
                    except Exception:
                        self._punctuate = False  # missing model/deps: degrade quietly
                    self._emit({"type": "model_load", "model": "punct", "phase": "done",
                               "ms": (time.perf_counter() - t0) * 1000})
        return self._punct

    @property
    def ko_spacer(self):
        """Kiwi morphological analyzer used to re-space Korean output."""
        if self._ko_spacer is None and self._ko_spacer_ok:
            with self._load_lock:
                if self._ko_spacer is None and self._ko_spacer_ok:
                    try:
                        from kiwipiepy import Kiwi

                        spacer = Kiwi()
                        spacer.space("한 국", reset_whitespace=True)  # warmup
                        self._ko_spacer = spacer
                    except Exception:
                        self._ko_spacer_ok = False  # missing dep: degrade quietly
        return self._ko_spacer

    def _get(self, name: str):
        if name in self._unavailable:
            raise ModelUnavailable(name)
        if name not in self._models and not _model_present(name):
            self._unavailable.add(name)
            print(f"[hayamimi] model '{name}' not found under models/ "
                  f"(minimal install?): routing falls back", file=sys.stderr)
            raise ModelUnavailable(name)
        rec = self._models.get(name)
        if rec is None:
            # per-model lock: loading v3 must not block a prefetch of omni
            with self._model_locks[name]:
                rec = self._models.get(name)
                if rec is None:
                    t0 = time.perf_counter()
                    self._emit({"type": "model_load", "model": name, "phase": "start",
                               "ms": None})
                    try:
                        if name == "rz":
                            rec = _build_reazon(self._threads, self._hotwords_file)
                        else:
                            rec = _BUILDERS[name](self._threads)
                    except Exception as exc:
                        # a --minimal install ships only some models: degrade
                        self._unavailable.add(name)
                        print(f"[hayamimi] model '{name}' unavailable "
                              f"(minimal install?): routing falls back",
                              file=sys.stderr)
                        raise ModelUnavailable(name) from exc
                    self._emit({"type": "model_load", "model": name, "phase": "done",
                               "ms": (time.perf_counter() - t0) * 1000})
                    with self._load_lock:
                        self._evict_if_needed(incoming=name)
                        self._models[name] = rec
        self._last_used[name] = time.monotonic()
        return rec

    def _get_with_fallback(self, name: str) -> tuple[object, str]:
        for cand in (name, "rz", "sv", "v3", "pz", "omni"):
            try:
                rec = self._get(cand)
            except ModelUnavailable:
                continue
            if cand != name and name not in self._fallback_warned:
                # Only reported once per distinct requested name per
                # session (matching _get()'s own once-only stderr print,
                # via self._unavailable) -- a whole missing tier would
                # otherwise emit this event on every single segment for
                # the rest of the session.
                self._fallback_warned.add(name)
                self._emit({"type": "model_fallback", "requested": name, "used": cand,
                           "reason": f"'{name}' unavailable (minimal install?)"})
            return rec, cand
        raise RuntimeError(
            "no ASR models found under models/ -- run scripts/download_models.py")

    def _evict_if_needed(self, incoming: str):
        if self._max_resident is None or incoming == "rz":
            return
        resident = [n for n in self._models if n != "rz"]
        if len(resident) < self._max_resident:
            return
        victim = min(resident, key=lambda n: self._last_used.get(n, 0.0))
        del self._models[victim]

    @property
    def resident_models(self) -> list[str]:
        return sorted(self._models)

    def _identify_lang(self, samples: np.ndarray, sample_rate: int) -> str:
        clip = samples
        # skip the leading quiet (preroll padding): it eats into the 4s LID
        # window and cost the demo capture its first-utterance language
        loud = np.flatnonzero(np.abs(clip) > 0.015)
        if len(loud) and loud[0] > sample_rate // 10:
            clip = clip[max(loud[0] - sample_rate // 20, 0):]
        max_len = int(LID_MAX_SECONDS * sample_rate)
        if len(clip) > max_len:
            clip = clip[:max_len]
        stream = self.lid.create_stream()
        stream.accept_waveform(sample_rate, clip)
        return self.lid.compute(stream)

    @staticmethod
    def _decode_full(rec, samples: np.ndarray, sample_rate: int) -> tuple[str, str]:
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        rec.decode_stream(stream)
        text = stream.result.text
        # ReazonSpeech models emit TV-subtitle annotation brackets around
        # boundary words; they carry no speech content.
        for junk in ("［", "］", "〈", "〉"):
            text = text.replace(junk, "")
        return text, getattr(stream.result, "lang", "") or ""

    @classmethod
    def _decode(cls, rec, samples: np.ndarray, sample_rate: int) -> str:
        return cls._decode_full(rec, samples, sample_rate)[0]

    def _segment_vad(self, seconds: float):
        """Lazily built Silero VAD used only to split offline buffers.

        Separate from the live capture VAD in realtime_transcribe.py: this one
        never force-splits on length (max_speech_duration is effectively
        disabled) because its job is to find the real silences in a buffer,
        not to endpoint a stream. Rebuilt only when a longer buffer than any
        seen so far arrives, since buffer_size_in_seconds is fixed at
        construction and a buffer that overflows it loses segments.

        Callers must hold self._seg_lock: the detector carries per-stream
        state, and the refine worker thread transcribes concurrently with the
        main thread's finals (realtime_transcribe.Refiner runs its own FIFO
        worker), so two buffers must never be pushed through it at once.
        """
        need = max(60.0, seconds + 2.0)
        if self._seg_vad is not None and self._seg_vad_capacity >= need:
            self._seg_vad.reset()
            return self._seg_vad
        cfg = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=VAD_MODEL_PATH,
                threshold=0.5,
                min_silence_duration=SEGMENT_MIN_SILENCE_S,
                min_speech_duration=SEGMENT_MIN_SPEECH_S,
                window_size=512,
                # a large finite value: this VAD must never force-split a long
                # pause-free run, that is the recognizer's business
                max_speech_duration=need,
            ),
            sample_rate=SEGMENT_SAMPLE_RATE,
            num_threads=1,
        )
        self._seg_vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=need)
        self._seg_vad_capacity = need
        return self._seg_vad

    def _speech_pieces(self, samples: np.ndarray, sample_rate: int):
        """Split a long buffer into one piece per utterance, for the retry.

        Returns (pieces, speech_seconds). `pieces` is None when there is
        nothing to split: too short to hold two utterances, an unsupported
        sample rate, or a buffer with no internal silence -- in which case the
        retry gives up and the whole-buffer text stands. `speech_seconds` is
        how much of the buffer the VAD called speech, or the whole buffer's
        length when no VAD ran (the pessimistic reading: it makes the density
        measure smaller, never larger, so it cannot manufacture suspicion).
        """
        seconds = len(samples) / float(sample_rate)
        if seconds <= SEGMENT_MIN_S or sample_rate != SEGMENT_SAMPLE_RATE:
            return None, seconds
        if self._seg_vad_ok and not os.path.exists(VAD_MODEL_PATH):
            self._seg_vad_ok = False
            message = (f"{os.path.basename(VAD_MODEL_PATH)} not found under models/: "
                       f"long offline buffers fall back to fixed "
                       f"{SEGMENT_FALLBACK_CHUNK_S:.0f}s chunking")
            print(f"[hayamimi] {message}", file=sys.stderr)
            self._emit({"type": "warning", "code": "segmentation_vad_unavailable",
                       "message": message})
        if not self._seg_vad_ok:
            return self._fixed_chunks(samples, sample_rate), seconds

        try:
            with self._seg_lock:
                vad = self._segment_vad(seconds)
                spans, pos = [], 0
                audio = np.ascontiguousarray(samples, dtype=np.float32)
                while pos < len(audio):
                    chunk = audio[pos:pos + 512]
                    if len(chunk) < 512:
                        chunk = np.pad(chunk, (0, 512 - len(chunk)))
                    vad.accept_waveform(chunk)
                    pos += 512
                    while not vad.empty():
                        spans.append((vad.front.start, len(vad.front.samples)))
                        vad.pop()
                vad.flush()
                while not vad.empty():
                    spans.append((vad.front.start, len(vad.front.samples)))
                    vad.pop()
        except Exception as exc:  # a VAD failure must never lose the audio
            self._seg_vad, self._seg_vad_ok = None, False
            message = f"offline segmentation VAD failed ({exc}): long buffers fall back to fixed chunking"
            print(f"[hayamimi] {message}", file=sys.stderr)
            self._emit({"type": "warning", "code": "segmentation_vad_unavailable",
                       "message": message})
            return self._fixed_chunks(samples, sample_rate), seconds

        speech_s = sum(count for _start, count in spans) / float(sample_rate)
        if len(spans) < 2:
            # one utterance (or none): nothing to split, so the whole-buffer
            # decode stands however sparse it looked
            return None, (speech_s or seconds)
        pad = int(SEGMENT_PAD_S * sample_rate)
        pieces = []
        for start, count in spans:
            a = max(start - pad, 0)
            b = min(start + count + pad, len(samples))
            if b <= a:
                continue
            # Real audio on both sides for context, then a margin of actual
            # silence around it. A piece that opens on speech loses its first
            # word -- the ja model dropped "東京の" off "東京の天気は晴れです"
            # when the piece started on the onset -- and the VAD's own onset
            # can land a little late, so the leading silence has to be
            # synthesized rather than borrowed from the buffer.
            silence = np.zeros(pad, dtype=np.float32)
            pieces.append(np.concatenate(
                [silence, np.asarray(samples[a:b], dtype=np.float32), silence]))
        return (pieces or None), (speech_s or seconds)

    @staticmethod
    def _fixed_chunks(samples: np.ndarray, sample_rate: int):
        """VAD-less fallback for a --minimal install missing silero_vad.onnx.

        Blind fixed-length chunks with a small overlap so a word straddling a
        seam is at least whole in one of the two chunks. The overlap can make
        a word at the seam appear twice; that is the documented cost of having
        no VAD, and it is still far better than losing whole sentences.
        """
        step = int((SEGMENT_FALLBACK_CHUNK_S - SEGMENT_FALLBACK_OVERLAP_S) * sample_rate)
        size = int(SEGMENT_FALLBACK_CHUNK_S * sample_rate)
        if step <= 0 or len(samples) <= size:
            return None
        pieces = [samples[i:i + size] for i in range(0, len(samples), step)]
        return [p for p in pieces if len(p) > int(0.2 * sample_rate)] or None

    @staticmethod
    def _join_pieces(parts: list[str], lang: str) -> str:
        sep = "" if lang in NO_SPACE_LANGS else " "
        return sep.join(p.strip() for p in parts if p.strip())

    @staticmethod
    def _text_density(text: str, seconds: float) -> float:
        """Alphanumeric characters per second.

        Punctuation is excluded so the measure does not move when the ja
        punctuation restorer or the CJK ITN pass rewrites the text.
        """
        if seconds <= 0:
            return float("inf")
        return sum(1 for c in text if c.isalnum()) / seconds

    @staticmethod
    def _looks_truncated(text: str, seconds: float, lang: str,
                         buffer_s: float | None = None) -> bool:
        """Does this whole-buffer decode look like it lost its leading content?

        `seconds` is speech seconds; `buffer_s` defaults to it and only exists
        so the cheap pre-gate can pass the buffer length before any VAD has
        run. Empty text is deliberately NOT suspicious: that is the omni
        fallback's business, and re-decoding silence in pieces would only
        invent words.
        """
        if (buffer_s if buffer_s is not None else seconds) <= SEGMENT_MIN_S:
            return False
        if not text.strip():
            return False
        floor = DENSITY_FLOOR_CJK if lang in NO_SPACE_LANGS else DENSITY_FLOOR_LATIN
        return RoutedASR._text_density(text, seconds) < floor

    @staticmethod
    def _retry_is_better(whole: str, retry: str, seconds: float, lang: str) -> bool:
        """Is the split retry plausibly a recovery rather than a different read?

        Two conditions, both required, and both biased towards keeping the
        whole-buffer text when the answer is unclear:

          * the retry is longer AND its density is back inside the healthy
            range -- a retry that is merely a bit longer but still sparse has
            not recovered anything;
          * most of the whole decode's tail reappears in the retry. The whole
            decode is the SURVIVING end of the utterance, so a genuine
            recovery keeps it and prepends what was lost. Matching is fuzzy
            (longest common substring) because the piece boundaries change
            the acoustic context and the model re-renders a few characters.
        """
        whole, retry = whole.strip(), retry.strip()
        if not retry or len(retry) <= len(whole):
            return False
        floor = DENSITY_FLOOR_CJK if lang in NO_SPACE_LANGS else DENSITY_FLOOR_LATIN
        if RoutedASR._text_density(retry, seconds) < floor:
            return False
        tail = whole[-RETRY_TAIL_CHARS:]
        if not tail:
            return False
        match = difflib.SequenceMatcher(None, tail, retry, autojunk=False)
        longest = match.find_longest_match(0, len(tail), 0, len(retry))
        return longest.size / len(tail) >= RETRY_TAIL_MATCH

    def _split_retry(self, text: str, samples: np.ndarray, sample_rate: int,
                     lang: str, model: str) -> str:
        """Re-decode a suspicious buffer one utterance at a time.

        Hard-pinned to `model`, the recognizer that produced `text`: no
        per-piece LID, no per-piece script correction, no per-piece omni
        fallback (an empty piece stays empty and simply drops out of the
        join). The language decision was made once, at buffer level, and a
        retry must not be able to move it -- fragments are short and a short
        fragment is exactly what makes a language model guess wrong.

        Returns `text` unchanged unless the retry clears _retry_is_better.
        """
        pieces, speech_s = self._speech_pieces(samples, sample_rate)
        # second half of the two-stage gate: the caller's cheap check used the
        # whole buffer's length, this one uses the speech the VAD actually
        # found, which is what the floor was calibrated against
        if not pieces or not self._looks_truncated(text, speech_s, lang):
            return text
        try:
            rec = self._get(model)
        except (ModelUnavailable, KeyError):
            return text
        retry = self._join_pieces(
            [self._decode(rec, piece, sample_rate) for piece in pieces], lang)
        return retry if self._retry_is_better(text, retry, speech_s, lang) else text

    def _sv_probe(self, cached, samples: np.ndarray, sample_rate: int):
        """Run SenseVoice's confirmation-probe decode on `samples`, memoized
        across the (up to three) call sites in transcribe() that may need it
        for the same segment: session bootstrap, dual-LID switch
        confirmation, and zh/yue arbitration.

        `cached` is whatever a previous call for this exact segment already
        returned (or None if none has run yet / the previous attempt failed).
        A non-None `cached` is reused as-is with zero elapsed time -- this is
        the memoization. A None `cached` always triggers a fresh decode
        attempt, so a failed probe (ModelUnavailable) is retried on every
        call site rather than being cached as a permanent failure, matching
        the pre-refactor behavior.

        Returns (probe_result, elapsed_ms) where probe_result is the
        (text, tag) pair from _decode_full, or None if the probe hasn't run
        (cached was None and unavailable) or ran but the sv model is
        unavailable (minimal install).
        """
        if cached is not None:
            return cached, 0.0
        t0 = time.perf_counter()
        result = None
        try:
            result = self._decode_full(self._get("sv"), samples, sample_rate)
        except ModelUnavailable:
            pass  # minimal install: no probe possible
        return result, (time.perf_counter() - t0) * 1000

    def _route(self, lang: str) -> tuple[object, str]:
        if lang in RZ_LANGS:
            return self._get_with_fallback("rz")
        if lang in PARA_LANGS:
            return self._get_with_fallback("pz")
        if lang in SV_LANGS:
            return self._get_with_fallback("sv")
        if lang in V3_LANGS:
            return self._get_with_fallback("v3")
        return self._get_with_fallback("omni")

    def partial(self, samples: np.ndarray, sample_rate: int,
                lang_hint: str | None = None) -> str:
        """Fast draft transcription of an in-progress utterance.

        Prefers the caller's early-LID hint for THIS utterance (so drafts
        switch language as soon as the mid-utterance LID fires, ~2s in);
        otherwise falls back to the session's sticky language. Without
        either, the tier-0 ja/en model drafts. Fixes Korean drafts staying
        blank after an English section (the sticky en model returns nothing
        for Korean speech).

        --mode single (self.forced_lang set) skips all of the above: the
        draft always routes straight to the forced language, same as
        transcribe(), with no LID/SenseVoice probing at all.
        """
        if self.forced_lang is not None:
            rec, _ = self._route(self.forced_lang)
            return self._replace(self._decode(rec, samples, sample_rate))

        if lang_hint is not None:
            # this utterance's language is confirmed: use its specialist
            rec, _ = self._route(lang_hint)
            return self._replace(self._decode(rec, samples, sample_rate))

        sticky = self.last_lang
        if sticky in V3_LANGS and sticky != "en":
            # EU languages: SenseVoice can't probe these; trust the session
            rec, _ = self._route(sticky)
            return self._replace(self._decode(rec, samples, sample_rate))

        # Before the early LID lands, never show the previous language's
        # guesswork (an English model romanizing Korean reads as garbage --
        # user feedback). SenseVoice runs its own per-utterance LID over
        # ja/zh/ko/yue/en, so probe with it and follow its tag: the very
        # first draft comes out in the right language.
        try:
            sv_text, sv_tag = self._decode_full(self._get("sv"), samples, sample_rate)
        except ModelUnavailable:
            rec = self._get_with_fallback("rz")[0]
            return self._replace(self._decode(rec, samples, sample_rate))
        if "ja" in sv_tag:
            try:
                rz_text = self._decode(self._get("rz"), samples, sample_rate)
                if rz_text.strip():
                    return self._replace(rz_text)
            except ModelUnavailable:
                pass
        return self._replace(sv_text)

    def _replace(self, text: str) -> str:
        for wrong, right in self._replacements:
            text = text.replace(wrong, right)
        return text

    def set_replacements(self, mapping: dict) -> None:
        """Replace the postprocessing find/replace dictionary at runtime.

        Thread-safe via a single atomic reference swap: the incoming dict is
        converted to an immutable tuple of pairs and assigned wholesale, so
        any in-flight _replace() call sees either the old list or the new
        one, never a partially-updated one (plain attribute assignment is a
        single pointer store under the GIL -- no lock needed). Callers pass
        the FULL desired mapping; this is a replace, not a merge, matching
        how the --replace file is loaded once at __init__.

        User replacements are always applied LAST in the postprocessing
        order -- see the docstring in itn_cjk.py and the ordering comment
        in transcribe() -- so this can override anything ITN or punctuation
        restoration produced.
        """
        self._replacements = tuple(mapping.items())

    def set_itn_overrides(self, exclude: set | None = None,
                           force: dict | None = None) -> None:
        """Replace the user-level CJK ITN overrides at runtime.

        Same atomic-swap pattern as set_replacements: a fresh immutable
        itn_cjk.ITNOverrides snapshot is built and assigned in one step.
        `exclude` is ADDED to itn_cjk's built-in idiom/proper-noun
        exclusions (never replaces them); `force` mappings win over
        rule-based ITN output for any matching literal text. Precedence
        overall: user force/exclude > built-in ITN rules > punctuation >
        user --replace/set_replacements (applied last of all).
        """
        self._itn_overrides = itn_cjk.ITNOverrides(
            exclude=frozenset(exclude or ()),
            force=dict(force or {}),
        )

    def identify(self, samples: np.ndarray, sample_rate: int) -> str:
        """Public LID hook so callers can identify the language mid-utterance.

        Also kicks off a background prefetch of that language's model, so by
        the time the utterance finalizes the recognizer is already resident.
        """
        lang = self._identify_lang(samples, sample_rate)
        threading.Thread(target=self._route, args=(lang,), daemon=True).start()
        return lang

    min_switch_s = 2.0  # a shorter utterance can't establish a new language

    # --- runtime setters (GitHub issue #29) -------------------------------
    #
    # These are all read from the decode thread (transcribe()/partial(),
    # both called from realtime_transcribe.run_stream()'s single-threaded
    # ingestion loop) and written from wherever an embedding app or the
    # SubtitleServer's POST /config handler lives -- typically a different
    # thread (an HTTP handler thread, or a GUI's own thread). Every setter
    # below only does plain attribute assignment (or, for set_punctuate,
    # one attribute assignment plus clearing a cached None), which is a
    # single pointer store under the GIL: an in-flight transcribe() call
    # sees either the old value or the new one for the remainder of that
    # call, never a torn/partial update. No lock is taken for the same
    # reason set_replacements()/set_itn_overrides() above don't take one.

    def set_forced_lang(self, lang: str | None) -> None:
        """--mode single / --lang runtime setter. `None` reverts to normal
        auto-routing (LID-driven); any other value must be one of
        ROUTABLE_LANGS (the codes this engine actually has a dedicated
        tier for -- see that constant's docstring) or ValueError is raised.

        Resets the sticky/pending LID state (reset_session()) either way:
        turning single-language mode ON must not let a stale in-flight
        switch-confirmation candidate from auto mode leak into the forced
        regime, and turning it OFF must not resume auto-routing already
        "sticky" on whatever language was forced a moment ago.
        """
        if lang is not None and lang not in ROUTABLE_LANGS:
            raise ValueError(f"unsupported language code: {lang!r} "
                             f"(must be one of {sorted(ROUTABLE_LANGS)} or None)")
        self.forced_lang = lang
        self.reset_session()

    def set_dual_confirm(self, enabled: bool) -> None:
        """--mode balanced/fast runtime setter: whether a language switch
        for the 5 SenseVoice-covered languages (DUAL_CONFIRM_LANGS) needs
        SenseVoice's own LID to agree before it's accepted (see
        resolve_dual_confirm's docstring). No session-state reset needed --
        unlike set_forced_lang, this only changes which resolution function
        the NEXT switch decision uses, it doesn't invalidate the language
        already established."""
        self.dual_confirm = bool(enabled)

    def set_punctuate(self, enabled: bool) -> None:
        """Turn Japanese punctuation restoration on or off at runtime.

        Turning it OFF just flips the flag the `punct` property checks;
        the loaded PunctuatorJa instance (if any) is left in memory so
        turning it back on later is free. Turning it ON while it was
        already off -- including the failure-degrade case, where the
        `punct` property itself set self._punctuate = False after a failed
        load -- clears self._punct so the next `.punct` access retries the
        load instead of treating this as still-degraded. A call that
        merely confirms an already-on state (self._punctuate already True)
        is a no-op and does NOT discard an already-loaded model.
        """
        enabled = bool(enabled)
        if enabled and not self._punctuate:
            self._punct = None  # retry the load on next `.punct` access
        self._punctuate = enabled

    def set_lid_switch_confirm(self, n: int) -> None:
        """--lid-switch-confirm runtime setter: consecutive same-language
        detections required before the fallback (non-dual-confirm) switch
        hysteresis (resolve_sticky_lang) accepts a new language. Must be
        >= 1 (0 would mean "never confirm anything")."""
        n = int(n)
        if n < 1:
            raise ValueError(f"lid_switch_confirm must be >= 1, got {n}")
        self.lid_switch_confirm = n

    def set_min_switch_s(self, seconds: float) -> None:
        """--lang-switch-guard runtime setter: a new-language candidate
        shorter than this many seconds never counts toward confirming a
        switch (see resolve_sticky_lang's docstring). Must be >= 0 (0
        disables the guard, matching --mode fast's default)."""
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError(f"min_switch_s must be >= 0, got {seconds}")
        self.min_switch_s = seconds

    def reset_session(self):
        """Clear the sticky/pending language state.

        Call this between unrelated audio streams (a new recording, a new
        speaker with no continuity from the last one, an eval harness moving
        to the next independent clip set). Without it, the sticky-LID
        hysteresis in transcribe() would treat the first utterance of the
        new stream as a language SWITCH away from whatever the previous,
        unrelated stream last said -- costing it an extra confirmation
        segment for no reason, since there was never a real session to
        switch away from.
        """
        self.last_lang = None
        self._pending_lang = None
        self._pending_count = 0

    def _maybe_second_opinion(self, text: str, samples: np.ndarray,
                              sample_rate: int) -> str:
        """Decode `samples` with parakeet-ja and adopt its text only when it
        agrees with the standing ReazonSpeech text (choose_second_opinion).

        Model missing on disk: warn once, permanently disable the gate for
        this session, and keep the standing text -- never fail a refine over
        an optional model.
        """
        try:
            rec = self._get("pja")
        except ModelUnavailable:
            if not self._pja_warned:
                message = ("ja second opinion requested but "
                          f"{os.path.basename(PJA_MODEL_DIR)} is not installed; "
                          "gate disabled (run download_models.py --eval-baselines).")
                print(f"[hayamimi] warning: {message}")
                self._emit({"type": "warning", "code": "second_opinion_unavailable",
                           "message": message})
                self._pja_warned = True
            self._ja_second_opinion = False
            return text
        try:
            second = self._decode(rec, samples, sample_rate)
        except Exception:
            return text  # a second-opinion failure must never lose the primary
        chosen, _used = choose_second_opinion(text, second, self._agree_threshold)
        return chosen

    def transcribe(self, samples: np.ndarray, sample_rate: int,
                   known_lang: str | None = None, speech_s: float | None = None,
                   live: bool = True, second_opinion: bool | None = None) -> dict:
        """live=False (e.g. the refine pass re-decoding past audio) must not
        touch the sticky/pending language state of the live stream.

        second_opinion: apply the ja parakeet agreement gate to this decode.
        None (default) resolves to "enabled by the constructor AND this is a
        refine decode (live=False)" -- live finals never pay the second
        decode. Pass True/False to override (evaluation harnesses)."""
        if self.forced_lang is not None:
            # --mode single: no LID, no switch logic, ever.
            lang, lid_ms = self.forced_lang, 0.0
        elif known_lang is not None and (speech_s is None or speech_s < 4.0):
            # trust the mid-utterance early LID only for short utterances; a
            # long one gives the full 4s window a chance to overrule the
            # guess made from its first 2 seconds (first-clip-per-language
            # errors in the demo capture came from exactly this)
            lang, lid_ms = known_lang, 0.0
        else:
            t0 = time.perf_counter()
            lang = self._identify_lang(samples, sample_rate)
            lid_ms = (time.perf_counter() - t0) * 1000

        suppress_fallback = False
        probe_ms = 0.0
        sv_probe = None  # (text, raw_tag) if a SenseVoice confirmation probe already ran
        bootstrap_probe_lang = None  # SenseVoice's tag for THIS audio, bootstrap only
        # True only for a too-short (<MIN_PROBE_S) segment that opens a brand
        # new session: this decode is a best-effort guess for THIS segment
        # only, not a confirmed session language, so it must not seed
        # self.last_lang below (see the docstring note on resolve_dual_confirm's
        # too_short handling -- a jingle/SFX misfire on segment 1 must not
        # lock the whole session onto whatever it happened to guess).
        suppress_bootstrap_seed = False
        was_bootstrap = self.last_lang is None
        if self.forced_lang is not None or not live:
            pass  # out-of-band decode / forced single-language: no switch resolution
        else:
            if self.last_lang is None and self.dual_confirm:
                # session bootstrap: whisper-tiny alone is unreliable at short
                # lengths (docs/eval/lid.md) and can guess a language SenseVoice
                # can't even arbitrate (a real-mic incident: whisper-tiny said
                # "ru" on segment 1 of a Japanese session and it collapsed
                # from there). Always confirm the very first segment against
                # SenseVoice's own LID, whatever whisper-tiny said.
                sv_probe, elapsed_ms = self._sv_probe(sv_probe, samples, sample_rate)
                if sv_probe is not None:
                    bootstrap_probe_lang = sv_lid_tag(sv_probe[1])
                probe_ms = elapsed_ms

            if self.dual_confirm and lang in DUAL_CONFIRM_LANGS:
                if lang != self.last_lang:
                    # docs/eval/lid.md: for the 5 SenseVoice-covered languages,
                    # confirm a candidate switch with SenseVoice's own LID
                    # instead of the old length/repeat-count hysteresis (see
                    # resolve_dual_confirm). Reuse the bootstrap probe above
                    # if it already decoded this exact audio.
                    sv_lang = bootstrap_probe_lang
                    if sv_probe is None:
                        sv_lang = ""
                        sv_probe, elapsed_ms = self._sv_probe(sv_probe, samples, sample_rate)
                        if sv_probe is not None:
                            sv_lang = sv_lid_tag(sv_probe[1])
                        probe_ms += elapsed_ms
                    lang, switched = resolve_dual_confirm(lang, self.last_lang, speech_s, sv_lang)
                    suppress_fallback = speech_s is not None and speech_s < MIN_PROBE_S
                    if was_bootstrap and suppress_fallback:
                        suppress_bootstrap_seed = True
                    if switched:
                        # a confirmed dual-LID switch supersedes any hysteresis
                        # candidate the fallback (non-SV) path was accumulating
                        self._pending_lang, self._pending_count = None, 0
                # lang == last_lang: already the session language, nothing to resolve
            else:
                # European/other languages SenseVoice can't arbitrate: fall
                # back to the length + consecutive-detection hysteresis. At
                # bootstrap, decode via the SenseVoice probe above until
                # whisper-tiny's own candidate repeats lid_switch_confirm
                # times (see resolve_sticky_lang's bootstrap_probe_lang).
                lang, suppress_fallback, self._pending_lang, self._pending_count = resolve_sticky_lang(
                    lang, self.last_lang, speech_s, self.min_switch_s, self.lid_switch_confirm,
                    self._pending_lang, self._pending_count,
                    bootstrap_probe_lang=bootstrap_probe_lang,
                )

        t0 = time.perf_counter()
        sv_text, sv_lang2 = (sv_probe if sv_probe is not None else (None, None))
        # `text_model` tracks WHICH recognizer produced the text that is
        # currently standing, so the head-dropout retry at the bottom can pin
        # itself to that same one. It is not always `tier`: the zh branch
        # below can report tier "pz" while keeping SenseVoice's transcript.
        text_model = "omni"
        if self.forced_lang is not None:
            # --mode single: route straight to the forced language, no
            # zh/yue arbitration and no script-based re-decode below.
            rec, tier = self._route(lang)
            text, text_model = self._decode(rec, samples, sample_rate), tier
        elif lang == "zh":
            # whisper-tiny LID labels Cantonese as "zh" (measured 0/12 correct
            # on FLEURS yue), so let SenseVoice's internal LID arbitrate: keep
            # its transcript for yue, re-decode with Paraformer for true zh.
            # Reuse the switch-confirmation probe above if it already decoded
            # this exact audio through SenseVoice instead of a second pass.
            if sv_text is None:
                sv_probe, _elapsed_ms = self._sv_probe(sv_probe, samples, sample_rate)
                sv_text, sv_lang2 = sv_probe if sv_probe is not None else (None, None)
            if sv_text is not None:
                text, text_model = sv_text, "sv"
                if "yue" in sv_lang2:
                    lang, tier = "yue", "sv"
                else:
                    rec, tier = self._get_with_fallback("pz")
                    text2 = self._decode(rec, samples, sample_rate)
                    if text2.strip():
                        text, text_model = text2, tier
            else:
                rec, tier = self._route(lang)
                text, text_model = self._decode(rec, samples, sample_rate), tier
        elif lang in ("ko", "yue") and sv_text is not None and sv_lid_tag(sv_lang2) == lang:
            # the switch-confirmation probe already decoded this exact audio
            # through the tier "ko"/"yue" routes to anyway; reuse it instead
            # of a second SenseVoice pass over the same samples.
            text, tier, text_model = sv_text, "sv", "sv"
        else:
            rec, tier = self._route(lang)
            text, text_model = self._decode(rec, samples, sample_rate), tier
        if not text.strip() and tier != "omni" and not suppress_fallback:
            # safety net: the specialist came back empty (likely LID mistake);
            # the 1600-language generalist gets the last word.
            text = self._decode(self._get("omni"), samples, sample_rate)
            tier, text_model = "omni", "omni"
        corrected = script_corrected_lang(lang, text)
        if self.forced_lang is None and live and text.strip() and corrected != lang:
            # the decoded script contradicts the LID tag (romaji-mangled
            # English under a ja tag, CJK under a non-CJK tag): re-decode
            # with the right model before anyone sees the final.
            if corrected == "ja" and not _has_kana(text) and "sv" not in self._unavailable:
                # han-only text is just as likely zh/yue as ja; let
                # SenseVoice's internal LID arbitrate instead of assuming
                # (assuming ja here cost yue 7.4% -> 24% CER, iteration 27)
                try:
                    sv2 = self._get("sv")
                except ModelUnavailable:
                    sv2 = None
                text2, sv_lang = (self._decode_full(sv2, samples, sample_rate)
                                  if sv2 is not None else ("", ""))
                if text2.strip():
                    if "yue" in sv_lang:
                        lang, tier, text, text_model = "yue", "sv", text2, "sv"
                    elif "zh" in sv_lang:
                        text3 = self._decode(self._get_with_fallback("pz")[0], samples, sample_rate)
                        lang, tier = "zh", "pz"
                        text, text_model = ((text3, "pz") if text3.strip()
                                            else (text2, "sv"))
                    elif "ja" in sv_lang:
                        text3 = self._decode(self._get_with_fallback("rz")[0], samples, sample_rate)
                        lang, tier = "ja", "rz"
                        text, text_model = ((text3, "rz") if text3.strip()
                                            else (text2, "sv"))
                    elif "ko" in sv_lang:
                        lang, tier, text, text_model = "ko", "sv", text2, "sv"
            else:
                rec2, tier2 = self._route(corrected)
                text2 = self._decode(rec2, samples, sample_rate)
                if text2.strip():
                    lang, tier, text, text_model = corrected, tier2, text2, tier2

        # Everything above is byte-identical to the pre-retry engine, and the
        # language decision is now FINAL. Only if that whole-buffer decode
        # looks like it dropped its leading content (docs/results/benchmarks.md, the
        # FLEURS ja head-dropout) is the buffer re-decoded one utterance at a
        # time, through the very recognizer that produced `text` -- and even
        # then the result is kept only if it looks like a recovery.
        buffer_s = len(samples) / float(sample_rate)
        if self._looks_truncated(text, buffer_s, lang):
            text = self._split_retry(text, samples, sample_rate, lang, text_model)

        # ja second-opinion agreement gate (refine decodes only by default):
        # runs AFTER the language decision and head-dropout retry are final,
        # BEFORE ITN/punctuation so the adopted text flows through the same
        # postprocessing as any other ja text.
        if second_opinion is None:
            # getattr: transcribe() is also exercised against minimal stubs in
            # tests that predate this attribute
            second_opinion = getattr(self, "_ja_second_opinion", False) and not live
        if second_opinion and lang == "ja" and text.strip():
            text = self._maybe_second_opinion(text, samples, sample_rate)

        # Fixed postprocessing order (see itn_cjk.py's module docstring):
        #   ITN -> punctuation restore -> user --replace, applied LAST so it
        # can always override anything the earlier stages produced.
        if lang in itn_cjk.APPLICABLE_LANGS and text.strip():
            overrides = self._itn_overrides
            text = itn_cjk.convert(text, lang, exclude=overrides.exclude,
                                   force=overrides.force)
        if lang == "ko" and text.strip() and self.ko_spacer is not None:
            try:
                # SenseVoice emits a space between every token; Kiwi restores
                # real Korean word spacing (docs/results/benchmarks.md iteration 20)
                text = self.ko_spacer.space(text, reset_whitespace=True)
            except Exception:
                pass
        if lang == "ja" and text.strip() and self.punct is not None:
            try:
                text = self.punct.restore(text)
            except Exception:
                pass  # a punctuation failure must never lose the transcription
        text = self._replace(text)
        decode_ms = (time.perf_counter() - t0) * 1000

        if live and text.strip() and not suppress_bootstrap_seed:
            # empty results must not poison the sticky language; neither
            # must a too-short bootstrap noise blip (suppress_bootstrap_seed)
            self.last_lang = lang
        return {"text": text, "lang": lang, "tier": tier, "lid_ms": lid_ms,
                "decode_ms": decode_ms, "probe_ms": probe_ms}
