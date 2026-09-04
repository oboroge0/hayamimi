"""Unit tests for the pure logic in scripts/eval_diar_overlap.py (Round 6).

No models and no audio -- these guard the four pieces that decide whether the
overlap-aware prototype's numbers mean anything: the powerset decode (does a
pair class really produce two active speakers?), the frame->segment
post-processing, the overlap-stripping used to isolate the overlap gain from
the clustering gain, and the reference overlap-floor computation the results
are compared against.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from eval_diar_overlap import (  # noqa: E402
    POWERSET,
    activity_to_segments,
    assign_by_centroid,
    cluster_local_speakers,
    overlap_fraction,
    powerset_decode,
    strip_overlap,
)
from global_recluster import pool_audio_for_group  # noqa: E402


# ---- powerset_decode --------------------------------------------------------

def test_powerset_class_order_matches_pyannote():
    # empty set, then singletons, then pairs -- the order the model was
    # trained with. Getting this wrong silently relabels every speaker.
    assert POWERSET == [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


def test_powerset_decode_singleton_and_pair_and_silence():
    logits = np.full((3, 7), -10.0, dtype=np.float32)
    logits[0, 0] = 0.0   # silence
    logits[1, 2] = 0.0   # speaker 1 alone
    logits[2, 5] = 0.0   # speakers 0 and 2 together
    out = powerset_decode(logits)
    assert not out[0].any()
    assert out[1].tolist() == [False, True, False]
    # the whole point of the prototype: a pair class yields TWO active speakers
    assert out[2].tolist() == [True, False, True]


# ---- activity_to_segments ---------------------------------------------------

def test_activity_to_segments_bridges_then_drops():
    # frames of 0.1s: 5 on, 1 off, 5 on, 10 off, 1 on
    active = np.array([1] * 5 + [0] + [1] * 5 + [0] * 10 + [1], dtype=bool)
    segs = activity_to_segments(active, 0.1, min_on=0.3, min_off=0.25)
    # the 0.1s gap is bridged into one 1.1s turn; the lone trailing 0.1s
    # frame is below min_on and is dropped
    assert len(segs) == 1
    assert segs[0][0] == 0.0
    assert abs(segs[0][1] - 1.1) < 1e-9


def test_activity_to_segments_empty():
    assert activity_to_segments(np.zeros(10, dtype=bool), 0.1, 0.3, 0.3) == []


# ---- strip_overlap ----------------------------------------------------------

def test_strip_overlap_keeps_locally_longer_turn():
    hyp = [("A", 0.0, 10.0), ("B", 4.0, 5.0)]
    out = strip_overlap(hyp)
    # B's 1s interjection loses to A's 10s turn, so A survives unbroken
    assert out == [("A", 0.0, 10.0)]


def test_strip_overlap_removes_all_simultaneity():
    hyp = [("A", 0.0, 6.0), ("B", 4.0, 10.0)]
    out = strip_overlap(hyp)
    total = sum(e - s for _, s, e in out)
    assert abs(total - 10.0) < 1e-9          # union preserved, no gaps invented
    for i, (_, s, e) in enumerate(out):      # and nothing overlaps any more
        for _, s2, e2 in out[i + 1:]:
            assert e <= s2 or e2 <= s


# ---- overlap_fraction -------------------------------------------------------

def test_overlap_fraction_counts_unavoidable_miss():
    # 10s of A, 10s of B, 5s of them together -> total turn time 20s+...
    ref = [("A", 0.0, 10.0), ("B", 5.0, 15.0)]
    # total = 20s, overlapped region 5s with n=2 -> (n-1)*5 = 5s guaranteed miss
    assert abs(overlap_fraction(ref) - 0.25) < 1e-9


def test_overlap_fraction_zero_without_overlap():
    assert overlap_fraction([("A", 0.0, 5.0), ("B", 5.0, 10.0)]) == 0.0


# ---- clustering -------------------------------------------------------------

def test_cluster_respects_cannot_link_within_a_window():
    # two near-identical embeddings that would certainly merge -- except they
    # were found in the same window, so they must be different people
    emb = np.array([[1.0, 0.0], [1.0, 0.001]], dtype=np.float32)
    same = cluster_local_speakers(emb, [7, 7], threshold=0.5, num_clusters=None)
    assert len(set(same.tolist())) == 2
    other = cluster_local_speakers(emb, [7, 8], threshold=0.5, num_clusters=None)
    assert len(set(other.tolist())) == 1


def test_assign_by_centroid_pulls_unreliable_points_to_nearest_cluster():
    emb = np.array([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    reliable = np.array([True, True, False])
    labels = assign_by_centroid(emb, reliable, np.array([0, 1]))
    assert labels[0] == 0 and labels[1] == 1
    assert labels[2] == 0  # the unreliable point lands with the point it resembles


# ---- pool_audio_for_group (Round 8, docs/design/diarization.md section 18 T1) -

def test_pool_audio_for_group_uses_diarizer_turns_when_present():
    # a single-speaker group where the diarizer still found real (filtered)
    # speech turns -- the pool entry should be built from exactly that
    # speech, concatenated, not the raw group buffer (which may include
    # non-speech samples the diarizer didn't call speech).
    buf = np.arange(100, dtype=np.float32)
    turns = [(0, 10, 20), (0, 50, 65)]  # (local_id, start_sample, end_sample)
    audio, dur_s = pool_audio_for_group(buf, sr=10, turns=turns)
    assert audio.tolist() == list(range(10, 20)) + list(range(50, 65))
    # 10 samples + 15 samples at sr=10 -> 1.0s + 1.5s
    assert abs(dur_s - 2.5) < 1e-9


def test_pool_audio_for_group_falls_back_to_full_buffer_when_declined():
    # the diarizer produced no usable turns at all (exception, or every
    # candidate turn below the duration floor) -- something is better than
    # nothing, so the pool entry falls back to the whole group buffer.
    buf = np.arange(20, dtype=np.float32)
    audio, dur_s = pool_audio_for_group(buf, sr=10, turns=[])
    assert audio.tolist() == list(range(20))
    assert abs(dur_s - 2.0) < 1e-9


def test_pool_audio_for_group_declined_duration_prefers_group_span_over_buffer_length():
    # eval_diar.py's convention: g_start/g_end are sample indices into the
    # ORIGINAL (whole-file) audio, which can differ from len(buf) if the
    # caller ever passes a buffer that isn't exactly [g_start, g_end) --
    # when given, they take precedence over len(buf) for the duration.
    buf = np.arange(20, dtype=np.float32)
    audio, dur_s = pool_audio_for_group(buf, sr=10, turns=[], g_start=100, g_end=135)
    assert abs(dur_s - 3.5) < 1e-9  # (135 - 100) / 10, not len(buf) / 10 == 2.0


def test_pool_audio_for_group_empty_group_yields_empty_audio():
    # an entirely empty group (no turns, zero-length buffer) must not crash
    # and must signal "nothing to embed" via a zero-length array -- callers
    # skip adding a pool entry in that case, same as they already do for an
    # empty real local-cluster embedding.
    audio, dur_s = pool_audio_for_group(np.zeros(0, dtype=np.float32), sr=10, turns=[])
    assert len(audio) == 0
