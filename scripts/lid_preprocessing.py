"""Shared, dependency-free LID input preprocessing.

Deliberately has NO sherpa_onnx/torch/speechbrain import: asr_engine.py
(sherpa_onnx, main venv) and scripts/eval_lid_voxlingua.py (torch,
.venv-train -- see that script's docstring for why it can't import
asr_engine) both need the exact same preprocessing applied ahead of ANY LID
backend's forward pass, so this lives somewhere both sides of the venv split
can import without dragging in the other side's heavy dependency.
"""
import numpy as np

LID_MAX_SECONDS = 4.0  # only feed the first N seconds of a segment to the LID model


def trim_lid_clip(samples: np.ndarray, sample_rate: int,
                   max_seconds: float = LID_MAX_SECONDS) -> np.ndarray:
    """Preprocessing shared by every LID backend (production whisper-tiny via
    asr_engine.RoutedASR._identify_lang, and every candidate backend in
    scripts/eval_lid_candidates.py and scripts/eval_lid_voxlingua.py): skip
    the leading quiet (preroll padding, which otherwise eats into the
    fixed-length LID window -- this cost the demo capture its first-utterance
    language), then cap to max_seconds so longer buffers don't change the
    cost/accuracy tradeoff the length-vs-accuracy curve (docs/eval/lid.md)
    was measured against.

    Every LID candidate must see identical input framing to the production
    detector and to each other -- comparing candidates fairly means
    comparing them on the same input, not giving one challenger raw audio
    while another gets the trimmed clip.
    """
    clip = samples
    loud = np.flatnonzero(np.abs(clip) > 0.015)
    if len(loud) and loud[0] > sample_rate // 10:
        clip = clip[max(loud[0] - sample_rate // 20, 0):]
    max_len = int(max_seconds * sample_rate)
    if len(clip) > max_len:
        clip = clip[:max_len]
    return clip
