"""Process-safety regression tests (GitHub issue #30).

Two independent landmines an app embedding scripts/asr_engine.py or
scripts/realtime_transcribe.py could hit:

  1. sherpa-onnx's C++ layer calls the process's exit() -- not a catchable
     Python exception -- when handed an empty/invalid model path. Every
     native construction in this codebase must be preceded by a
     file-presence check that raises asr_engine.ModelUnavailable instead.
     These tests prove the guard fires (and the native builder is never
     reached) rather than actually letting a bad path hit sherpa-onnx,
     which would crash the test process itself if the guard regressed.

  2. The live capture loop (realtime_transcribe.run_stream / mic_chunks /
     wav_chunks) had no cancellation mechanism besides KeyboardInterrupt, so
     an app embedding it on its own thread had no way to stop it cleanly.
     These tests exercise the threading.Event stop token end to end.

No ASR/VAD models are loaded here; everything is guard functions and stubs,
so this file is cheap to run.
"""
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import asr_engine  # noqa: E402
from asr_engine import ModelUnavailable  # noqa: E402


# --- the exit() landmine: RoutedASR construction ----------------------------

def test_routed_asr_init_raises_not_exits_when_lid_missing(tmp_path, monkeypatch):
    """A missing whisper-tiny LID model must raise ModelUnavailable out of
    RoutedASR.__init__, not crash the process.

    self.lid = _build_lid_guarded(threads) is the very first native
    construction __init__ does (docs comment in asr_engine._KEY_FILES);
    everything before it in __init__ is plain attribute/lock setup, so this
    exercises the real construction path end to end, not just the guard
    function in isolation.
    """
    empty_dir = tmp_path / "empty_whisper_tiny"
    empty_dir.mkdir()
    monkeypatch.setitem(asr_engine._KEY_FILES, "lid",
                        (str(empty_dir), "tiny-encoder.int8.onnx"))
    with pytest.raises(ModelUnavailable) as exc_info:
        asr_engine.RoutedASR(threads=1, warmup=False, preload=False)
    assert exc_info.value.name == "lid"


def test_build_lid_guarded_never_reaches_sherpa_onnx_when_missing(tmp_path, monkeypatch):
    """Guard function in isolation: when the key file is absent, the real
    sherpa-onnx builder must never be called at all.

    Calling the real (unguarded) builder with an empty path is exactly the
    landmine this issue is about -- it would call the process's exit(), not
    raise -- so this test cannot exercise that path directly without risking
    killing the test runner. Instead it proves the guard short-circuits
    BEFORE _build_lid() would ever run.
    """
    empty_dir = tmp_path / "empty_whisper_tiny"
    empty_dir.mkdir()
    monkeypatch.setitem(asr_engine._KEY_FILES, "lid",
                        (str(empty_dir), "tiny-encoder.int8.onnx"))

    def boom(threads):
        pytest.fail("_build_lid() must not be called when the model is "
                     "missing on disk -- that is the exit() landmine")

    monkeypatch.setattr(asr_engine, "_build_lid", boom)
    with pytest.raises(ModelUnavailable) as exc_info:
        asr_engine._build_lid_guarded(threads=1)
    assert exc_info.value.name == "lid"


def test_build_lid_guarded_builds_normally_when_present(monkeypatch):
    """Sanity check the other direction: presence still reaches the builder
    and returns whatever it returns (guard is a pure pass-through)."""
    monkeypatch.setattr(asr_engine, "_model_present", lambda name: True)
    sentinel = object()
    monkeypatch.setattr(asr_engine, "_build_lid", lambda threads: sentinel)
    assert asr_engine._build_lid_guarded(threads=1) is sentinel


# --- the exit() landmine: the live capture VAD ------------------------------

def test_build_vad_raises_not_exits_when_model_missing(tmp_path, monkeypatch):
    """realtime_transcribe.build_vad() must guard the same way as the ASR
    recognizers: a missing silero_vad.onnx must raise ModelUnavailable
    before sherpa_onnx.VoiceActivityDetector ever sees the bad path."""
    import realtime_transcribe as rt

    missing = tmp_path / "no_such_silero_vad.onnx"
    monkeypatch.setattr(rt, "VAD_MODEL", str(missing))

    def boom(cfg, buffer_size_in_seconds):
        pytest.fail("sherpa_onnx.VoiceActivityDetector must not be "
                     "constructed when the VAD model file is missing")

    monkeypatch.setattr(rt.sherpa_onnx, "VoiceActivityDetector", boom)
    with pytest.raises(ModelUnavailable) as exc_info:
        rt.build_vad()
    assert exc_info.value.name == "vad"


# --- stop token: run_stream honors a threading.Event ------------------------

class _StubVad:
    """Just enough of sherpa_onnx.VoiceActivityDetector for run_stream to
    drive without ever producing a segment -- this test is only about
    whether the chunk loop notices stop_event, not about VAD/ASR behavior."""

    def accept_waveform(self, chunk):
        pass

    def empty(self):
        return True

    def is_speech_detected(self):
        return False

    def flush(self):
        pass


def test_run_stream_stops_mid_wav_when_stop_event_is_set():
    """The wav path: stop_event set from another thread must interrupt
    run_stream before the whole (paced) file has been consumed."""
    import realtime_transcribe as rt

    sr = 16000
    samples = np.zeros(int(5.0 * sr), dtype=np.float32)  # 5s, paced playback
    vad = _StubVad()
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=False)
    stop_event = threading.Event()

    def run():
        rt.run_stream(rt.wav_chunks(samples, sr, realtime=True), vad, sr, None,
                      stats, printer, stop_event=stop_event)

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.15)
    stop_event.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive(), "run_stream did not return after stop_event was set"
    # stopped partway through 5s of paced audio -- nowhere near the full file
    assert 0.0 < stats.total_audio_s < 5.0


def test_run_stream_without_stop_event_runs_to_completion():
    """Regression guard: omitting stop_event (the CLI's own KeyboardInterrupt
    path, and every pre-existing caller) must behave exactly as before --
    the whole chunk source is consumed."""
    import realtime_transcribe as rt

    sr = 16000
    samples = np.zeros(int(0.2 * sr), dtype=np.float32)
    vad = _StubVad()
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=False)

    rt.run_stream(rt.wav_chunks(samples, sr, realtime=False), vad, sr, None, stats, printer)

    # wav_chunks pads its final chunk up to a full WINDOW_SIZE window, so the
    # consumed total can exceed the raw sample count by less than one window
    assert stats.total_audio_s >= 0.2
    assert stats.total_audio_s < 0.2 + rt.WINDOW_SIZE / sr


# --- stop token: mic_chunks polls instead of blocking forever ---------------

class _FakeInputStream:
    """Stand-in for sounddevice.InputStream: never actually calls back with
    audio, just tracks that it was opened and closed like a context manager
    should. mic_chunks() must still notice stop_event via its bounded queue
    timeout instead of blocking on q.get() forever."""

    def __init__(self, *args, callback=None, **kwargs):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_mic_chunks_notices_stop_event_without_audio(monkeypatch):
    import realtime_transcribe as rt

    fake_sd = types.SimpleNamespace(InputStream=_FakeInputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stop_event = threading.Event()
    gen = rt.mic_chunks(stop_event=stop_event)

    def stopper():
        time.sleep(0.05)
        stop_event.set()

    threading.Thread(target=stopper, daemon=True).start()

    start = time.perf_counter()
    with pytest.raises(StopIteration):
        while True:
            next(gen)
    elapsed = time.perf_counter() - start
    # bounded by MIC_QUEUE_TIMEOUT_S polling, not an indefinite block
    assert elapsed < 1.0
