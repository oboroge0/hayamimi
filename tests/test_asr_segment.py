"""Guards the offline multi-utterance split added to RoutedASR.transcribe().

Background (docs/BENCHMARKS.md / the FLEURS ja regression): the catalog's
offline recognizers collapse a buffer holding several utterances into a
single one -- fed a whole multi-sentence clip in one decode_stream() call,
the ReazonSpeech zipformer returned only the LAST sentence and silently
dropped everything before it (FLEURS ja clip 15, 18.3s: CER 0.67). The engine
now splits such a buffer at its internal silences and decodes one utterance
per call.

Two layers of test here:
  * the joining/chunking helpers, which are pure and always run;
  * an end-to-end decode of a synthesized three-sentence clip, which needs
    the ja model + Silero VAD on disk and is skipped otherwise (same
    convention as tests/test_diarize.py).
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


@needs_models
def test_all_sentences_survive_a_multi_utterance_buffer():
    import soundfile as sf

    if not _build_fixture():
        pytest.skip("could not synthesize testdata/multi_sentence_ja.wav "
                    "(edge-tts / ffmpeg / network unavailable)")
    samples, sr = sf.read(FIXTURE_WAV, dtype="float32")
    assert len(samples) / sr > asr_engine.SEGMENT_MIN_S, \
        "fixture must be long enough to engage the split"

    asr = RoutedASR(threads=2, preload=False, warmup=False, punctuate=False,
                    forced_lang="ja")
    text = asr.transcribe(samples, sr)["text"]
    missing = [kw for kw, _ in SENTENCES if kw not in text]
    assert not missing, f"lost sentence(s) {missing} from a multi-utterance buffer: {text!r}"


@needs_models
def test_pause_free_buffer_is_decoded_verbatim():
    """The live path only ever hands transcribe() one VAD segment, i.e. a
    speech run with no internal silence >= the split threshold. Such a buffer
    must come back from the split machinery untouched (None), so the live
    path's output is bit-for-bit what it was before segmentation existed."""
    asr = RoutedASR(threads=2, preload=False, warmup=False, punctuate=False,
                    forced_lang="ja")
    rng = np.random.default_rng(0)
    # continuous band-limited noise: no silence anywhere, well over the
    # SEGMENT_MIN_S threshold
    noisy = np.cumsum(rng.normal(0, 0.05, int(10.0 * SR))).astype(np.float32)
    noisy = np.clip(noisy - np.mean(noisy), -1.0, 1.0)
    assert asr._speech_pieces(noisy, SR) is None

    short = np.zeros(int(2.0 * SR), dtype=np.float32)
    assert asr._speech_pieces(short, SR) is None
    # unsupported sample rate: the VAD model is 16k only, decode as before
    assert asr._speech_pieces(np.zeros(int(10.0 * 44100), dtype=np.float32), 44100) is None
