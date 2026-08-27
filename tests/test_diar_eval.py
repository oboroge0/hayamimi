"""Unit tests for the DER-scoring plumbing in scripts/eval_diar.py.

No models, no downloaded audio -- these guard the pure logic (RTTM parsing,
segment-type coercion) plus a DER sanity check against `simpleder` directly,
so a regression in either the parser or how we feed simpleder shows up here
before a full AMI sweep would surface it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from eval_diar import der_breakdown, group_segments, parse_rttm, segments_to_der_tuples
from realtime_transcribe import AudioHistory, PartialPrinter, Refiner, SessionStats
import numpy as np


# ---- parse_rttm -------------------------------------------------------------

def test_parse_rttm_basic(tmp_path):
    rttm = tmp_path / "x.rttm"
    rttm.write_text(
        "SPEAKER meet1 1 34.27 10.12 <NA> <NA> FEE041 <NA> <NA>\n"
        "SPEAKER meet1 1 46.43 10.42 <NA> <NA> FEE044 <NA> <NA>\n",
        encoding="utf-8",
    )
    turns = parse_rttm(str(rttm))
    assert turns == [
        ("FEE041", 34.27, 34.27 + 10.12),
        ("FEE044", 46.43, 46.43 + 10.42),
    ]


def test_parse_rttm_ignores_non_speaker_lines_and_blanks(tmp_path):
    rttm = tmp_path / "x.rttm"
    rttm.write_text(
        "\n"
        "SPKR-INFO meet1 1 <NA> <NA> <NA> unknown FEE041 <NA> <NA>\n"
        "SPEAKER meet1 1 1.0 2.0 <NA> <NA> S1 <NA> <NA>\n"
        "\n",
        encoding="utf-8",
    )
    turns = parse_rttm(str(rttm))
    assert turns == [("S1", 1.0, 3.0)]


def test_parse_rttm_empty_file(tmp_path):
    rttm = tmp_path / "x.rttm"
    rttm.write_text("", encoding="utf-8")
    assert parse_rttm(str(rttm)) == []


def test_parse_rttm_rejects_malformed_line(tmp_path):
    rttm = tmp_path / "x.rttm"
    rttm.write_text("SPEAKER meet1 1 1.0 2.0\n", encoding="utf-8")  # missing speaker field
    with pytest.raises(ValueError):
        parse_rttm(str(rttm))


def test_parse_rttm_rejects_negative_duration(tmp_path):
    rttm = tmp_path / "x.rttm"
    rttm.write_text("SPEAKER meet1 1 1.0 -2.0 <NA> <NA> S1 <NA> <NA>\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_rttm(str(rttm))


def test_parse_rttm_round_trips_zero_duration(tmp_path):
    # 0-length turns are rare but legal RTTM; parse_rttm itself keeps them
    # (segments_to_der_tuples is what filters them for scoring).
    rttm = tmp_path / "x.rttm"
    rttm.write_text("SPEAKER meet1 1 5.0 0.0 <NA> <NA> S1 <NA> <NA>\n", encoding="utf-8")
    assert parse_rttm(str(rttm)) == [("S1", 5.0, 5.0)]


# ---- segments_to_der_tuples --------------------------------------------------

def test_segments_to_der_tuples_coerces_types():
    import numpy as np

    segs = [("S1", np.float32(1.0), np.float32(2.5))]
    out = segments_to_der_tuples(segs)
    assert out == [("S1", 1.0, 2.5)]
    speaker, start, end = out[0]
    assert isinstance(speaker, str)
    assert isinstance(start, float)
    assert isinstance(end, float)


def test_segments_to_der_tuples_drops_zero_and_negative_length():
    segs = [("S1", 1.0, 1.0), ("S1", 2.0, 1.5), ("S1", 1.0, 2.0)]
    out = segments_to_der_tuples(segs)
    assert out == [("S1", 1.0, 2.0)]


def test_segments_to_der_tuples_empty():
    assert segments_to_der_tuples([]) == []


# ---- DER sanity checks against simpleder directly ----------------------------

def test_der_zero_for_identical_ref_and_hyp():
    simpleder = pytest.importorskip("simpleder")
    ref = [("A", 0.0, 5.0), ("B", 5.0, 10.0)]
    der = simpleder.DER(ref, ref, collar=0.0)
    assert der == pytest.approx(0.0)


def test_der_nonzero_for_swapped_speaker_labels_within_overlap():
    simpleder = pytest.importorskip("simpleder")
    # Optimal speaker-label assignment (Hungarian matching) means a pure
    # global relabeling (A<->B swapped everywhere) still scores DER=0 --
    # DER cares about the partition into speakers, not the label strings.
    ref = [("A", 0.0, 5.0), ("B", 5.0, 10.0)]
    hyp_swapped = [("B", 0.0, 5.0), ("A", 5.0, 10.0)]
    assert simpleder.DER(ref, hyp_swapped, collar=0.0) == pytest.approx(0.0)

    # But a genuine confusion (wrong speaker assigned to a sub-interval) is
    # NOT free -- this is what "speaker label flips mid-session" would cost.
    ref2 = [("A", 0.0, 10.0)]
    hyp_confused = [("A", 0.0, 5.0), ("B", 5.0, 10.0)]
    der = simpleder.DER(ref2, hyp_confused, collar=0.0)
    assert der > 0.0


def test_der_end_to_end_from_parsed_rttm(tmp_path):
    simpleder = pytest.importorskip("simpleder")
    rttm = tmp_path / "ref.rttm"
    rttm.write_text(
        "SPEAKER meet 1 0.0 5.0 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER meet 1 5.0 5.0 <NA> <NA> B <NA> <NA>\n",
        encoding="utf-8",
    )
    ref = segments_to_der_tuples(parse_rttm(str(rttm)))
    assert simpleder.DER(ref, ref, collar=0.25) == pytest.approx(0.0)


# ---- der_breakdown (iteration 5: Miss/FA/Confusion via pyannote.metrics) ----

def test_der_breakdown_zero_for_identical_ref_and_hyp():
    pytest.importorskip("pyannote.metrics")
    ref = [("A", 0.0, 5.0), ("B", 5.0, 10.0)]
    result = der_breakdown(ref, ref, collar=0.0)
    assert result["miss"] == pytest.approx(0.0)
    assert result["false_alarm"] == pytest.approx(0.0)
    assert result["confusion"] == pytest.approx(0.0)
    assert result["der_breakdown"] == pytest.approx(0.0)


def test_der_breakdown_isolates_confusion_from_miss():
    pytest.importorskip("pyannote.metrics")
    # Pure confusion: same coverage, wrong speaker for the second half.
    ref = [("A", 0.0, 10.0)]
    hyp_confused = [("A", 0.0, 5.0), ("B", 5.0, 10.0)]
    confused = der_breakdown(ref, hyp_confused, collar=0.0)
    assert confused["confusion"] > 0.0
    assert confused["miss"] == pytest.approx(0.0)
    assert confused["false_alarm"] == pytest.approx(0.0)

    # Pure miss: hypothesis simply doesn't cover the second half at all.
    hyp_missed = [("A", 0.0, 5.0)]
    missed = der_breakdown(ref, hyp_missed, collar=0.0)
    assert missed["miss"] > 0.0
    assert missed["confusion"] == pytest.approx(0.0)


# ---- group_segments (iteration 3-4 Refiner-style grouping) ------------------

SR = 16000  # sample rate used to convert the group_gap_s/group_max_s knobs


def test_group_segments_single_segment():
    segs = [(0, SR, "S1")]
    groups = group_segments(segs, SR, group_gap_s=2.0, group_max_s=25.0)
    assert groups == [[(0, SR, "S1")]]


def test_group_segments_merges_close_segments():
    # two 1s segments 0.5s apart: gap (0.5s) < group_gap_s (2.0s) -> one group
    segs = [(0, SR, "S1"), (int(1.5 * SR), int(2.5 * SR), "S1")]
    groups = group_segments(segs, SR, group_gap_s=2.0, group_max_s=25.0)
    assert len(groups) == 1
    assert groups[0] == segs


def test_group_segments_splits_on_silence_gap():
    # 3s gap between segments >= group_gap_s (2.0s) -> two groups
    segs = [(0, SR, "S1"), (int(4 * SR), int(5 * SR), "S1")]
    groups = group_segments(segs, SR, group_gap_s=2.0, group_max_s=25.0)
    assert groups == [[(0, SR, "S1")], [(int(4 * SR), int(5 * SR), "S1")]]


def test_group_segments_splits_on_max_length():
    # a group already at/over group_max_s must be closed before folding in
    # the next segment, even with no silence gap between them (mirrors
    # Refiner.maybe_refine()'s "due" check running on the already-
    # accumulated spans).
    segs = [
        (0, 26 * SR, "S1"),           # single 26s span, already >= 25s
        (26 * SR, 27 * SR, "S1"),     # back-to-back, no gap
    ]
    groups = group_segments(segs, SR, group_gap_s=2.0, group_max_s=25.0)
    assert groups == [segs[:1], segs[1:]]


def test_group_segments_empty_input():
    assert group_segments([], SR, group_gap_s=2.0, group_max_s=25.0) == []


# ---- eval-vs-production grouping divergence (docs/DIARIZATION_PLAN.md ------
# section 10.6/10.8 root-cause investigation) --------------------------------

class _FakeAsrForGrouping:
    """Just enough of RoutedASR's interface for Refiner.maybe_refine()'s
    work() to run to completion without a real model: _identify_lang always
    agrees with the span's language (so resolve_refine_lang() never tries a
    SenseVoice re-judgment) and transcribe() returns fixed non-empty text
    (so the "never lose content" guard doesn't discard the group)."""

    def _identify_lang(self, buf, sr):
        return "en"

    def transcribe(self, buf, sr, known_lang=None, live=False, speech_s=None):
        return {"text": "ok"}


def test_production_over_splits_relative_to_eval_replica_on_a_skipped_vad_segment():
    """docs/DIARIZATION_PLAN.md section 10.6 found the full production
    pipeline (--speakers) reaching far more global speaker labels (S1..S13
    on ES2011a) than eval_diar.py's --method refine_diarize DER hypothesis
    for the identical audio (4-5). Section 10.8 traces one concrete,
    reproducible contributor (a second exists too -- see
    speaker_id.SpeakerLabeler.centroid_summary()'s docstring): eval's
    generate_diarize_hypothesis() calls SpeakerLabeler.label() and appends
    to group_segments()'s input for EVERY VAD segment, speech or not (no
    ASR in that path to tell the difference). Production's
    realtime_transcribe.drain_segments() only calls Refiner.add_span() when
    ASR actually returned non-empty text -- a VAD misfire (jingle, breath,
    cross-talk noise) that ASR correctly declines to transcribe never
    becomes a span. But Refiner.maybe_refine()'s "due" (silence-gap) check
    is evaluated against the real current audio position (run_stream passes
    it real elapsed samples, not "time of the last kept span" -- see
    run_stream()'s refiner.maybe_refine(int(audio_pos * sample_rate)) call),
    so skipping that segment's own span doesn't skip the real time it took:
    a gap that stays comfortably under GROUP_GAP_S when the skipped
    segment's boundaries are counted (eval) can cross it once that
    segment's own start/end stop being available as intermediate points
    (production) -- closing an extra group, and therefore giving the
    refine-path remap one extra, spurious chance to open a new global
    centroid at a segment boundary eval's replica would never have split
    on.
    """
    seg1 = (0, 16000, "x")        # 1.0s of real speech
    noise = (17000, 48000, "x")   # VAD-flagged noise/jingle; ASR returns "" for it in production
    seg3 = (49000, 65000, "x")    # 1.0s of real speech

    # eval_diar.py's replica: every VAD segment (including the noise one)
    # is a boundary point. Both gaps (1000 samples each) are tiny relative
    # to GROUP_GAP_S*SR (32000 samples) -- one group.
    eval_groups = group_segments([seg1, noise, seg3], SR, group_gap_s=2.0, group_max_s=25.0)
    assert len(eval_groups) == 1

    # Production: the noise segment's ASR text is empty, so drain_segments()
    # never hands it to Refiner.add_span() -- only seg1 and seg3 become
    # spans. maybe_refine() still gets called with the real (post-noise)
    # audio position, exactly as run_stream() does after every drain.
    stats = SessionStats()
    history = AudioHistory(SR, keep_s=30.0)
    history.push(np.zeros(70000, dtype=np.float32))
    printer = PartialPrinter(enabled=False)
    refiner = Refiner(_FakeAsrForGrouping(), history, SR, printer, stats=stats)

    refiner.add_span(seg1[0], seg1[1], "en", "hello there", "")
    refiner.maybe_refine(17000)  # mirrors run_stream() right after the noise VAD segment starts
    # noise segment: ASR text empty -> drain_segments skips add_span(), but
    # real audio time still advances past its end (48000) before the next
    # drain reaches maybe_refine, exactly as run_stream() would call it.
    refiner.maybe_refine(49000, force_sync=True)  # gap vs. seg1's end (16000) is 33000 >= 32000: due
    refiner.add_span(seg3[0], seg3[1], "en", "second utterance", "")
    refiner.maybe_refine(66000, force=True, force_sync=True)  # flush the trailing group

    assert stats.refine_groups_closed == 2, (
        "production should have closed 2 groups (seg1 alone, then seg3 alone) where "
        f"eval's replica sees only 1 -- got {stats.refine_groups_closed}")
