"""Unit tests for scripts/speaker_id.py's global-centroid matching logic.

No model loaded: SpeakerLabeler.__init__ pulls in the real CAM++ ONNX
extractor, which these tests don't need -- match_embedding() only touches
_centroids/_counts/_threshold, so tests build a bare instance with
object.__new__() and set just those attributes directly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from speaker_id import SpeakerLabeler


def make_labeler(threshold: float = 0.45) -> SpeakerLabeler:
    labeler = object.__new__(SpeakerLabeler)
    labeler._threshold = threshold
    labeler._centroids = []
    labeler._counts = []
    return labeler


def unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_match_embedding_opens_first_speaker():
    labeler = make_labeler()
    label = labeler.match_embedding(unit([1.0, 0.0]))
    assert label == "S1"
    assert len(labeler._centroids) == 1
    assert labeler._counts == [1]


def test_match_embedding_joins_existing_within_threshold():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))
    label = labeler.match_embedding(unit([0.99, 0.01]))  # cos sim ~1, well above 0.9
    assert label == "S1"
    assert len(labeler._centroids) == 1  # no new speaker opened
    assert labeler._counts == [2]  # folded into the running mean


def test_match_embedding_opens_new_speaker_below_threshold():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))
    label = labeler.match_embedding(unit([0.0, 1.0]))  # orthogonal: cos sim 0
    assert label == "S2"
    assert len(labeler._centroids) == 2


def test_match_embedding_update_false_does_not_mutate_on_hit():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))
    label = labeler.match_embedding(unit([0.99, 0.01]), update=False)
    assert label == "S1"
    assert labeler._counts == [1]  # NOT folded in -- read-only lookup


def test_match_embedding_update_false_returns_empty_on_miss():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))
    label = labeler.match_embedding(unit([0.0, 1.0]), update=False)
    assert label == ""
    assert len(labeler._centroids) == 1  # no new speaker opened


def test_match_embedding_picks_nearest_of_several_centroids():
    labeler = make_labeler(threshold=0.5)
    labeler.match_embedding(unit([1.0, 0.0, 0.0]))       # S1
    labeler.match_embedding(unit([0.0, 1.0, 0.0]))       # S2
    label = labeler.match_embedding(unit([0.05, 0.95, 0.0]))  # closest to S2
    assert label == "S2"
