"""Smoke tests for scripts/diarize.py's GroupDiarizer.

Needs the pyannote segmentation-3.0 model (see docs/design/diarization.md
section 2) and the CAM++ embedding model already used by --speakers; both
skip cleanly if missing so this doesn't break a minimal checkout.
"""
import os
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from diarize import SEGMENTATION_MODEL, GroupDiarizer

pytestmark = pytest.mark.skipif(
    not os.path.exists(SEGMENTATION_MODEL),
    reason="pyannote segmentation-3.0 model not downloaded (see docs/design/diarization.md section 2)",
)

TWO_SPEAKERS_WAV = os.path.join(os.path.dirname(SCRIPTS_DIR), "testdata", "two_speakers.wav")


def test_group_diarizer_missing_model_raises_filenotfound(monkeypatch):
    monkeypatch.setattr("diarize.SEGMENTATION_MODEL", "/nonexistent/model.onnx")
    with pytest.raises(FileNotFoundError):
        GroupDiarizer()


def test_group_diarizer_short_buffer_returns_empty():
    diarizer = GroupDiarizer()
    samples = np.zeros(int(0.1 * diarizer.sample_rate), dtype=np.float32)
    assert diarizer.process(samples, diarizer.sample_rate) == []


@pytest.mark.skipif(not os.path.exists(TWO_SPEAKERS_WAV), reason="testdata/two_speakers.wav not present")
def test_group_diarizer_finds_multiple_speakers_on_two_speaker_audio():
    import wave

    with wave.open(TWO_SPEAKERS_WAV, "rb") as f:
        sr = f.getframerate()
        data = f.readframes(f.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    diarizer = GroupDiarizer()
    segments = diarizer.process(samples, sr)
    assert len(segments) >= 2
    speakers = {spk for spk, _, _ in segments}
    assert len(speakers) >= 2  # this fixture is specifically two distinct speakers
    # segments must be sorted by start time and each have start < end
    starts = [s for _, s, _ in segments]
    assert starts == sorted(starts)
    for _, start, end in segments:
        assert end > start
