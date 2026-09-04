"""Guards the head-dropout split RETRY in RoutedASR.transcribe().

Background (docs/results/benchmarks.md / the FLEURS ja regression): the catalog's
offline recognizers can collapse a buffer holding several utterances into a
single one -- fed a whole multi-sentence clip in one decode_stream() call,
the ReazonSpeech zipformer returned only the LAST sentence and silently
dropped everything before it (FLEURS ja clip 15, 18.3s: CER 0.67).

Splitting the buffer at its internal silences fixes those clips, but an
external FLEURS 5x100 A/B showed that splitting UNCONDITIONALLY is a net loss
(ja 8.6% -> 9.9%, en 9.4% -> 10.2%, ko 8.1% -> 9.1%). So the split is now a
retry, gated on the whole-buffer decode looking suspiciously sparse, and kept
only when it looks like a recovery. The two things these tests must pin down
are therefore:

  * a normal buffer is byte-identical to the plain whole-buffer decode;
  * a buffer that DID drop its leading sentences gets them back.

Three layers here: the pure helpers (always run), the suspicion/acceptance
logic against a stubbed recognizer (always runs), and an end-to-end decode of
a synthesized three-sentence clip, which needs the ja model + Silero VAD on
disk and is skipped otherwise (same convention as tests/test_diarize.py).
"""
import asyncio
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import asr_engine  # noqa: E402
from asr_engine import RoutedASR  # noqa: E402

FIXTURE_WAV = os.path.join(ROOT, "testdata", "multi_sentence_ja.wav")
SENTENCES = [
    ("東京", "東京の天気は晴れです。"),
    ("会議", "明日の会議は十時からです。"),
    ("資料", "資料は昨日送りました。"),
]
GAP_S = 0.5
SR = 16000


# --- pure helpers ----------------------------------------------------------

def test_join_pieces_concatenates_cjk_without_spaces():
    parts = ["東京の天気は晴れです", "明日の会議は十時からです"]
    assert RoutedASR._join_pieces(parts, "ja") == "東京の天気は晴れです明日の会議は十時からです"
    for lang in ("zh", "yue", "ko"):
        assert " " not in RoutedASR._join_pieces(parts, lang)


def test_join_pieces_space_joins_word_delimited_languages():
    assert RoutedASR._join_pieces(["hello there", "how are you"], "en") == \
        "hello there how are you"
    assert RoutedASR._join_pieces(["guten tag", "wie geht"], "de") == "guten tag wie geht"


def test_join_pieces_drops_empty_and_whitespace_only_pieces():
    assert RoutedASR._join_pieces(["  ", "hi", "", "  there "], "en") == "hi there"
    assert RoutedASR._join_pieces([], "ja") == ""


def test_fixed_chunks_returns_none_for_short_audio():
    short = np.zeros(int(2.0 * SR), dtype=np.float32)
    assert RoutedASR._fixed_chunks(short, SR) is None


def test_fixed_chunks_cover_the_whole_buffer_with_overlap():
    seconds = 30.0
    audio = np.arange(int(seconds * SR), dtype=np.float32)
    chunks = RoutedASR._fixed_chunks(audio, SR)
    assert chunks is not None and len(chunks) > 1
    # every chunk is at most one chunk-length, and together they reach the end
    assert all(len(c) <= int(asr_engine.SEGMENT_FALLBACK_CHUNK_S * SR) for c in chunks)
    assert chunks[-1][-1] == audio[-1]
    # consecutive chunks overlap rather than butt up against each other, so a
    # word straddling a seam survives whole in one of them
    step = int((asr_engine.SEGMENT_FALLBACK_CHUNK_S
                - asr_engine.SEGMENT_FALLBACK_OVERLAP_S) * SR)
    assert chunks[1][0] == audio[step]
    assert step < int(asr_engine.SEGMENT_FALLBACK_CHUNK_S * SR)


def test_text_density_counts_only_alphanumerics():
    # punctuation must not move the measure: the ja punctuation restorer and
    # the CJK ITN pass both run after the retry decision
    assert RoutedASR._text_density("あいう", 1.0) == 3.0
    assert RoutedASR._text_density("あ、い。う！", 1.0) == 3.0
    assert RoutedASR._text_density("anything", 0.0) == float("inf")


# --- the suspicion gate ----------------------------------------------------

def test_short_buffers_are_never_suspicious():
    # a live VAD segment is short by construction; it must never be retried
    assert not RoutedASR._looks_truncated("あ", 1.0, "ja",
                                          buffer_s=asr_engine.SEGMENT_MIN_S)


def test_empty_text_is_not_suspicious():
    # an empty decode is the omni fallback's business; re-decoding it in
    # pieces would only invent words
    assert not RoutedASR._looks_truncated("", 10.0, "ja", buffer_s=20.0)
    assert not RoutedASR._looks_truncated("   ", 10.0, "ja", buffer_s=20.0)


def test_sparse_cjk_output_over_a_long_buffer_is_suspicious():
    # FLEURS ja clip 15: 18 characters over 10.6s of speech = 1.7 chars/s,
    # against a healthy population measured at 3.46-14.22
    text = "植物がなければ動物は生きていけません"
    assert RoutedASR._looks_truncated(text, 10.6, "ja", buffer_s=18.3)


def test_normal_density_cjk_output_is_not_suspicious():
    text = "私たちは植物で家を作り植物から衣類を作ります日々食べる食材の植物です" * 2
    assert not RoutedASR._looks_truncated(text, 10.6, "ja", buffer_s=18.3)


def test_latin_languages_use_their_own_higher_floor():
    # the same character count is sparse for latin and not for CJK
    text = "a" * 30  # 3 chars/s over 10s
    assert RoutedASR._looks_truncated(text, 10.0, "en", buffer_s=10.5)
    assert not RoutedASR._looks_truncated(text, 10.0, "ja", buffer_s=10.5)


# --- the acceptance gate ---------------------------------------------------

WHOLE = "植物がなければ動物は生きていけません"
RECOVERED = ("私たちは植物で家を作り植物から衣類を作ります日々食べる食材の植物です"
             "植物がなければ動物は生きていけません")


def test_retry_accepted_when_it_prepends_to_the_surviving_tail():
    assert RoutedASR._retry_is_better(WHOLE, RECOVERED, 10.6, "ja")


def test_retry_rejected_when_not_longer():
    assert not RoutedASR._retry_is_better(WHOLE, WHOLE, 10.6, "ja")
    assert not RoutedASR._retry_is_better(WHOLE, "植物がなければ", 10.6, "ja")
    assert not RoutedASR._retry_is_better(WHOLE, "", 10.6, "ja")


def test_retry_rejected_when_it_loses_the_whole_decodes_tail():
    # longer, dense, but a different reading of the audio: the surviving tail
    # is gone, so this is not a recovery and the conservative bias applies
    other = "まったくちがうぶんしょうがここにながながとつづいていくばかりです" * 2
    assert not RoutedASR._retry_is_better(WHOLE, other, 10.6, "ja")


def test_retry_rejected_when_it_is_still_sparse():
    # longer and tail-preserving, but density never recovers -- nothing was
    # really regained
    assert not RoutedASR._retry_is_better(WHOLE, "あ" + WHOLE, 10.6, "ja")


def test_split_retry_pins_the_recognizer_and_never_reroutes():
    """Every piece must go through the model that produced the whole-buffer
    text: no per-piece LID, no per-piece script correction, no per-piece omni
    fallback. A piece that comes back empty stays empty."""
    asr = RoutedASR.__new__(RoutedASR)
    pieces = [np.zeros(SR, dtype=np.float32) for _ in range(3)]
    asr._speech_pieces = lambda samples, sr: (pieces, 10.6)
    asked = []

    def fake_get(name):
        asked.append(name)
        return f"<{name}>"

    head = "私たちは植物で家を作り植物から衣類を作ります日々食べる食材の植物です"
    outs = iter([head, "", WHOLE])
    asr._get = fake_get
    asr._decode = lambda rec, samples, sr: (
        next(outs) if rec == "<rz>" else pytest.fail(f"wrong recognizer {rec}"))

    got = asr._split_retry(WHOLE, np.zeros(int(18.3 * SR), dtype=np.float32),
                           SR, "ja", "rz")
    assert asked == ["rz"], "the retry must not consult any other model"
    assert got == head + WHOLE


def test_split_retry_keeps_the_whole_text_when_there_is_nothing_to_split():
    asr = RoutedASR.__new__(RoutedASR)
    asr._speech_pieces = lambda samples, sr: (None, 10.6)
    asr._get = lambda name: pytest.fail("no decode should happen")
    assert asr._split_retry(WHOLE, np.zeros(SR, dtype=np.float32), SR, "ja", "rz") == WHOLE


def test_split_retry_stands_down_when_the_speech_measure_clears_the_buffer_alarm():
    """The cheap pre-gate divides by the whole buffer, which a silence-heavy
    clip makes look sparse. Once the VAD says how much of it was actually
    speech, that alarm has to be droppable without any decode running."""
    asr = RoutedASR.__new__(RoutedASR)
    asr._speech_pieces = lambda samples, sr: ([np.zeros(SR, dtype=np.float32)] * 2, 3.0)
    asr._get = lambda name: pytest.fail("no decode should happen")
    assert asr._split_retry(WHOLE, np.zeros(int(30.0 * SR), dtype=np.float32),
                            SR, "ja", "rz") == WHOLE


# --- end-to-end ------------------------------------------------------------

def _models_present() -> bool:
    return (asr_engine._model_present("rz")
            and os.path.exists(asr_engine.VAD_MODEL_PATH))


def _build_fixture() -> bool:
    """Synthesize the three-sentence clip with edge-tts (the repo's test-audio
    convention, scripts/make_testset.py) and cache it under testdata/.

    Deliberately starts on speech with NO leading silence, so this exercises
    the multi-utterance collapse itself rather than any head-trimming effect.
    """
    if os.path.exists(FIXTURE_WAV):
        return True
    try:
        import edge_tts
        import soundfile as sf
    except Exception:
        return False
    import tempfile

    pieces = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_keyword, text) in enumerate(SENTENCES):
            mp3 = os.path.join(tmp, f"{i}.mp3")
            wav = os.path.join(tmp, f"{i}.wav")
            try:
                asyncio.run(edge_tts.Communicate(text, "ja-JP-NanamiNeural").save(mp3))
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                                "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16", wav],
                               check=True)
            except Exception:
                return False  # no network / no ffmpeg: the test skips
            y, _ = sf.read(wav, dtype="float32")
            # trim edge-tts's own leading/trailing padding so the gaps between
            # sentences are exactly GAP_S and the clip opens on speech
            loud = np.flatnonzero(np.abs(y) > 0.01)
            if len(loud):
                y = y[loud[0]:loud[-1] + 1]
            pieces.append(y)

    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    joined = pieces[0]
    for p in pieces[1:]:
        joined = np.concatenate([joined, gap, p])
    os.makedirs(os.path.dirname(FIXTURE_WAV), exist_ok=True)
    sf.write(FIXTURE_WAV, joined, SR, subtype="PCM_16")
    return True


needs_models = pytest.mark.skipif(
    not _models_present(),
    reason="ja model / silero_vad.onnx not present under models/",
)


def _fixture_samples():
    import soundfile as sf

    if not _build_fixture():
        pytest.skip("could not synthesize testdata/multi_sentence_ja.wav "
                    "(edge-tts / ffmpeg / network unavailable)")
    samples, sr = sf.read(FIXTURE_WAV, dtype="float32")
    assert len(samples) / sr > asr_engine.SEGMENT_MIN_S, \
        "fixture must be long enough to engage the retry"
    return samples, sr


def _engine():
    return RoutedASR(threads=2, preload=False, warmup=False, punctuate=False,
                     forced_lang="ja")


@needs_models
def test_a_dropped_multi_utterance_buffer_is_recovered_by_the_retry():
    samples, sr = _fixture_samples()
    asr = _engine()
    # the whole-buffer decode loses everything but the last sentence, and
    # that is exactly what the suspicion gate is looking for
    whole = asr._decode(asr._get("rz"), samples, sr)
    speech_s = asr._speech_pieces(samples, sr)[1]
    assert asr._looks_truncated(whole, speech_s, "ja", buffer_s=len(samples) / sr)

    text = asr.transcribe(samples, sr)["text"]
    missing = [kw for kw, _ in SENTENCES if kw not in text]
    assert not missing, f"lost sentence(s) {missing} from a multi-utterance buffer: {text!r}"


@needs_models
def test_a_non_suspicious_buffer_is_byte_identical_to_the_whole_decode(monkeypatch):
    """The hard requirement the unconditional split violated: when the gate
    does not fire, transcribe() must return exactly what a single
    whole-buffer decode returns, character for character."""
    samples, sr = _fixture_samples()
    asr = _engine()
    whole = asr._decode(asr._get("rz"), samples, sr)
    # floor 0 = nothing is ever sparse enough to be suspicious
    monkeypatch.setattr(asr_engine, "DENSITY_FLOOR_CJK", 0.0)
    monkeypatch.setattr(asr_engine, "DENSITY_FLOOR_LATIN", 0.0)
    monkeypatch.setattr(RoutedASR, "_speech_pieces",
                        lambda *a, **k: pytest.fail("the retry must not run"))
    assert asr.transcribe(samples, sr)["text"] == whole


@needs_models
def test_pause_free_buffer_has_nothing_to_split():
    """The live path only ever hands transcribe() one VAD segment, i.e. a
    speech run with no internal silence >= the split threshold. Such a buffer
    has no pieces, so even a suspicious one is left exactly as decoded."""
    asr = _engine()
    rng = np.random.default_rng(0)
    # continuous band-limited noise: no silence anywhere, well over the
    # SEGMENT_MIN_S threshold
    noisy = np.cumsum(rng.normal(0, 0.05, int(10.0 * SR))).astype(np.float32)
    noisy = np.clip(noisy - np.mean(noisy), -1.0, 1.0)
    assert asr._speech_pieces(noisy, SR)[0] is None

    short = np.zeros(int(2.0 * SR), dtype=np.float32)
    assert asr._speech_pieces(short, SR)[0] is None
    # unsupported sample rate: the VAD model is 16k only, decode as before
    assert asr._speech_pieces(np.zeros(int(10.0 * 44100), dtype=np.float32), 44100)[0] is None


# --- live path: utterance-initial words survive the VAD's late onset -------

# utterance-INITIAL word of each fixture sentence, with the kana spellings
# the ja models legitimately emit (a kana head is not a dropped head)
HEAD_VARIANTS = [("東京", "とうきょう"), ("明日", "あした"), ("資料", "しりょう")]


@needs_models
def test_live_path_preroll_keeps_utterance_initial_words(capsys):
    """Silero's speech-start detection lags the true onset (measured 198ms
    behind on this fixture's fast-onset first sentence), so decoding
    vad.front.samples as-is can lose or garble the utterance's first word --
    with the preroll forced to 0, this very clip decoded 資料は→昨日は
    (docs/results/benchmarks.md 2026-08-31, live-path verification). The live path
    guards against that by prepending up to PREROLL_S of real context from
    AudioHistory in drain_segments(); this test pins that end-to-end: every
    sentence's initial word must survive the full VAD -> preroll -> decode
    chain."""
    import realtime_transcribe as rt

    samples, sr = _fixture_samples()
    asr = _engine()
    vad = rt.build_vad()
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=False)
    history = rt.AudioHistory(sr)
    rt.run_stream(rt.wav_chunks(samples, sr, realtime=False), vad, sr, asr,
                  stats, printer, refiner=None, history=history)
    # same finalization the CLI's finish() does for whatever the VAD still holds
    vad.flush()
    rt.drain_segments(vad, sr, asr, stats, printer, history)

    out = capsys.readouterr().out
    assert stats.segments == len(SENTENCES), \
        f"expected one final per sentence, got {stats.segments}: {out!r}"
    missing = [v[0] for v in HEAD_VARIANTS if not any(k in out for k in v)]
    assert not missing, \
        f"utterance-initial word(s) {missing} dropped on the live path: {out!r}"
