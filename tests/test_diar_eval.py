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

from eval_diar import parse_rttm, segments_to_der_tuples


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
