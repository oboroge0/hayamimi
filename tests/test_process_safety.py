"""Process-safety regression tests (GitHub issue #30).

sherpa-onnx's C++ layer calls the process's exit() -- not a catchable Python
exception -- when handed an empty/invalid model path. Every native
construction in this codebase must be preceded by a file-presence check
that raises asr_engine.ModelUnavailable instead. These tests prove the
guard fires (and the native builder is never reached) rather than actually
letting a bad path hit sherpa-onnx, which would crash the test process
itself if the guard regressed.

No ASR/VAD models are loaded here; everything is guard functions and
monkeypatched builders, so this file is cheap to run.
"""
import os
import sys

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
