"""Unit tests for scripts/eval_diar_osd.py's pure logic (Track F,
docs/design/diarization.md section 21).

No models and no audio -- these guard the three gates pseudo_osd_decode()
adds on top of section 16's plain argmax / section 20's single-frame
pair_gate: the joint per-speaker marginal floor, the hysteresis (Schmitt
trigger) across the frame sequence, and the minimum accepted-run length.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from eval_diar_overlap import POWERSET, powerset_decode  # noqa: E402
from eval_diar_osd import pseudo_osd_decode  # noqa: E402


def _logits(probs_per_frame):
    """List of 7-length probability lists (each summing to ~1) -> (frames, 7)
    log-space array, so pseudo_osd_decode's internal np.exp(logits) round-trips
    back to exactly these probabilities."""
    return np.log(np.asarray(probs_per_frame, dtype=np.float64)).astype(np.float32)


# ---- degenerate case: reduces to plain unconditional argmax -----------------

def test_pseudo_osd_degenerates_to_plain_argmax():
    # hi_thresh=lo_thresh=0.0 (Schmitt trigger always armed for any pair
    # frame), joint_floor=0.0 (marginals are always >=0, so never binds),
    # min_run_frames=1 (no run ever rejected) together must reproduce
    # powerset_decode()'s unconditional per-frame argmax exactly.
    probs = [
        [0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],   # silence
        [0.02, 0.90, 0.02, 0.02, 0.02, 0.01, 0.01],   # speaker 0 alone
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.88, 0.02],   # speakers 0 and 2
        [0.10, 0.05, 0.05, 0.05, 0.45, 0.20, 0.10],   # low-confidence pair
    ]
    logits = _logits(probs)
    out_pseudo = pseudo_osd_decode(logits, hi_thresh=0.0, lo_thresh=0.0,
                                   joint_floor=0.0, min_run_frames=1)
    out_plain = powerset_decode(logits)
    assert out_pseudo.tolist() == out_plain.tolist()


# ---- joint marginal floor ----------------------------------------------------

def test_joint_floor_rejects_lopsided_pair_confidence():
    # pair class (0,1) is the argmax at 0.40, but speaker 1's total marginal
    # (0.05 singleton + 0.40 pair(0,1) + 0.05 pair(1,2) = 0.50) is much
    # weaker than speaker 0's (0.35 + 0.40 + 0.10 = 0.85): the model is
    # confident about SOME simultaneous class, lopsidedly. Section 20's
    # single-number pair-class threshold could not see this asymmetry --
    # only the joint per-speaker marginal can.
    probs = [[0.02, 0.35, 0.05, 0.03, 0.40, 0.10, 0.05]]
    logits = _logits(probs)
    assert abs(sum(probs[0]) - 1.0) < 1e-9

    lenient = pseudo_osd_decode(logits, hi_thresh=0.0, lo_thresh=0.0,
                                joint_floor=0.4, min_run_frames=1)
    assert lenient[0].tolist() == [True, True, False]   # both marginals (0.85, 0.50) clear 0.4

    strict = pseudo_osd_decode(logits, hi_thresh=0.0, lo_thresh=0.0,
                               joint_floor=0.6, min_run_frames=1)
    # speaker 1's marginal (0.50) fails the 0.6 floor -> downgraded to the
    # best of the empty/singleton classes, which is speaker 0 alone (0.35)
    assert strict[0].tolist() == [True, False, False]


# ---- hysteresis (Schmitt trigger) -------------------------------------------

def test_hysteresis_bridges_a_dip_between_thresholds_then_exits():
    # frame 0: pair-class posterior 0.75 >= hi(0.7) -> enters armed state
    # frame 1: 0.55, between lo(0.5) and hi(0.7) -> a single-frame threshold
    #          at 0.7 (section 20 style) would already reject this frame, but
    #          hysteresis keeps it armed because it hasn't dropped below lo
    # frame 2: 0.45 < lo(0.5) -> armed state is left, frame is downgraded
    probs = [
        [0.02, 0.05, 0.03, 0.02, 0.75, 0.08, 0.05],
        [0.05, 0.15, 0.05, 0.05, 0.55, 0.10, 0.05],
        [0.05, 0.40, 0.03, 0.02, 0.45, 0.03, 0.02],
    ]
    for row in probs:
        assert abs(sum(row) - 1.0) < 1e-9
    logits = _logits(probs)
    out = pseudo_osd_decode(logits, hi_thresh=0.7, lo_thresh=0.5,
                            joint_floor=0.0, min_run_frames=1)
    assert out[0].tolist() == [True, True, False]   # entered
    assert out[1].tolist() == [True, True, False]   # bridged by hysteresis
    assert out[2].tolist() == [True, False, False]  # exited -> speaker 0 alone


def test_hysteresis_never_arms_below_hi_thresh():
    # a pair-class posterior that never reaches hi_thresh should never be
    # accepted, no matter how far above lo_thresh it sits.
    probs = [[0.05, 0.40, 0.03, 0.02, 0.45, 0.03, 0.02]] * 3
    logits = _logits(probs)
    out = pseudo_osd_decode(logits, hi_thresh=0.7, lo_thresh=0.3,
                            joint_floor=0.0, min_run_frames=1)
    assert all(row.tolist() == [True, False, False] for row in out)


# ---- minimum run-length ------------------------------------------------------

def test_min_run_frames_rejects_short_accepted_runs():
    # two consecutive frames both clear hi_thresh on their own, so with
    # min_run_frames=1 they are accepted as overlap...
    probs = [[0.03, 0.05, 0.03, 0.02, 0.80, 0.05, 0.02]] * 2
    logits = _logits(probs)
    lenient = pseudo_osd_decode(logits, hi_thresh=0.7, lo_thresh=0.5,
                                joint_floor=0.0, min_run_frames=1)
    assert all(row.tolist() == [True, True, False] for row in lenient)

    # ...but a run of 2 is shorter than min_run_frames=3, so the whole run is
    # rejected back to the dominant single speaker (index 1: speaker 0 alone).
    strict = pseudo_osd_decode(logits, hi_thresh=0.7, lo_thresh=0.5,
                               joint_floor=0.0, min_run_frames=3)
    assert all(row.tolist() == [True, False, False] for row in strict)


def test_min_run_frames_keeps_runs_at_least_as_long_as_the_floor():
    probs = [[0.03, 0.05, 0.03, 0.02, 0.80, 0.05, 0.02]] * 3
    logits = _logits(probs)
    out = pseudo_osd_decode(logits, hi_thresh=0.7, lo_thresh=0.5,
                            joint_floor=0.0, min_run_frames=3)
    assert all(row.tolist() == [True, True, False] for row in out)


# ---- input validation ---------------------------------------------------------

def test_lo_thresh_above_hi_thresh_is_rejected():
    logits = _logits([[0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
    try:
        pseudo_osd_decode(logits, hi_thresh=0.3, lo_thresh=0.5,
                          joint_floor=0.0, min_run_frames=1)
        assert False, "expected an AssertionError"
    except AssertionError:
        pass


def test_powerset_class_order_still_matches_pyannote():
    # pseudo_osd_decode relies on the same class ordering powerset_decode
    # does (POWERSET, imported from eval_diar_overlap) -- guard it here too
    # since this module's _MEMBERSHIP matrix is built from it.
    assert POWERSET == [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
