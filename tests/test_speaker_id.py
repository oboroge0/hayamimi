"""Unit tests for scripts/speaker_id.py's global-centroid matching logic.

No model loaded: SpeakerLabeler.__init__ pulls in the real CAM++ ONNX
extractor, which these tests don't need -- match_embedding() only touches
_centroids/_counts/_threshold, so tests build a bare instance with
object.__new__() and set just those attributes directly.

The two real-audio tests at the bottom of this file (fast-path regression
check + the hysteresis two_speakers.wav failure mode, docs/
DIARIZATION_PLAN.md section 9 / iteration 6) DO load the real CAM++ model
and, for the second one, scripts/diarize.py's GroupDiarizer (which also
needs the pyannote segmentation-3.0 model) -- both skip cleanly if their
model/fixture is missing, same pattern as tests/test_diarize.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from speaker_id import EMBED_MODEL, PROVISIONAL_CONFIRM_HITS, SpeakerLabeler


def make_labeler(threshold: float = 0.45, merge_enabled: bool = False,
                 merge_threshold: float = 0.8, hysteresis_enabled: bool = False,
                 hysteresis_min_hits: int = 2) -> SpeakerLabeler:
    labeler = object.__new__(SpeakerLabeler)
    labeler._threshold = threshold
    labeler._centroids = []
    labeler._counts = []
    labeler._merge_enabled = merge_enabled
    labeler._merge_threshold = merge_threshold
    labeler._alias = {}
    labeler._merge_history = {}
    labeler._hysteresis_enabled = hysteresis_enabled
    labeler._hysteresis_min_hits = hysteresis_min_hits
    labeler._confirmed = []
    labeler._open_log = []
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


# ---------------------------------------------------------------------------
# Round 5 (docs/DIARIZATION_PLAN.md section 15) T3: exclude_provisional
# ---------------------------------------------------------------------------

def test_exclude_provisional_skips_a_one_off_centroid_as_a_match_target():
    labeler = make_labeler(threshold=0.3)
    labeler.match_embedding(unit([1.0, 0.0]))  # S1: confirmed (2 hits)
    labeler.match_embedding(unit([0.99, 0.01]))
    labeler.match_embedding(unit([0.0, 1.0]))  # S2: still provisional (1 hit)
    assert PROVISIONAL_CONFIRM_HITS == 2
    # would ordinarily match S2 (closest), but S2 is still provisional
    probe = unit([0.05, 0.95])
    label = labeler.match_embedding(probe, update=False, exclude_provisional=True)
    assert label == ""  # no eligible confirmed centroid close enough either


def test_exclude_provisional_still_matches_a_confirmed_centroid():
    labeler = make_labeler(threshold=0.3)
    labeler.match_embedding(unit([1.0, 0.0]))  # S1: confirmed
    labeler.match_embedding(unit([0.99, 0.01]))
    label = labeler.match_embedding(unit([0.98, 0.02]), update=False, exclude_provisional=True)
    assert label == "S1"


def test_exclude_provisional_false_is_a_no_op_default():
    labeler = make_labeler(threshold=0.3)
    labeler.match_embedding(unit([0.0, 1.0]))  # S1: provisional (1 hit)
    label = labeler.match_embedding(unit([0.05, 0.95]), update=False)
    assert label == "S1"


# ---------------------------------------------------------------------------
# Round 5 (docs/DIARIZATION_PLAN.md section 15) T1: constrained joint remap
# ---------------------------------------------------------------------------

def test_joint_remap_single_cluster_matches_independent_match_embedding():
    # single-local-cluster groups must take the exact old path -- no joint
    # machinery, byte-for-byte the same as calling match_embedding().
    labeler = make_labeler(threshold=0.3)
    labeler.match_embedding(unit([1.0, 0.0]))  # S1
    emb = unit([0.99, 0.01])
    labels = labeler.match_embeddings_joint([emb], update=True, threshold=0.3)
    assert labels == ["S1"]


def test_joint_remap_no_live_centroids_opens_distinct_new_speakers():
    labeler = make_labeler(threshold=0.3)
    labels = labeler.match_embeddings_joint(
        [unit([1.0, 0.0]), unit([0.0, 1.0])], update=True, threshold=0.3, source="remap")
    assert labels == ["S1", "S2"]
    assert labeler.centroid_open_counts() == {"remap": 2}


def test_joint_remap_keeps_two_clusters_on_distinct_speakers_when_both_eligible():
    # Without the joint constraint, two local clusters that are BOTH
    # individually closest to the same global centroid but still eligible
    # (>= threshold) for the other one would greedily collapse onto one
    # speaker. The Hungarian assignment instead picks the higher-total
    # pairing that keeps them apart.
    labeler = make_labeler(threshold=0.02)
    labeler.match_embedding(unit([1.0, 0.0]))  # S1
    labeler.match_embedding(unit([0.0, 1.0]))  # S2
    a = unit([0.95, 0.05])  # closer to S1, but still eligible for S2
    b = unit([0.9, 0.1])    # closer to S1 too, but still eligible for S2
    labels = labeler.match_embeddings_joint([a, b], update=True, threshold=0.02, source="remap")
    assert labels == ["S1", "S2"]


def test_joint_remap_falls_back_to_independent_match_when_below_threshold():
    # b's jointly-optimal partner (S2) falls below the eligibility floor,
    # so it has no eligible candidate and falls back to an independent
    # match_embedding() call -- which, unconstrained, matches its own
    # nearest centroid (S1, already claimed by a). This is the documented
    # fallback trade-off, not a bug: eligibility floor first, constraint
    # second.
    labeler = make_labeler(threshold=0.3)
    labeler.match_embedding(unit([1.0, 0.0]))  # S1
    labeler.match_embedding(unit([0.0, 1.0]))  # S2
    a = unit([0.95, 0.05])
    b = unit([0.9, 0.1])
    labels = labeler.match_embeddings_joint([a, b], update=True, threshold=0.3, source="remap")
    assert labels == ["S1", "S1"]


def test_joint_remap_update_false_does_not_mutate_centroids():
    labeler = make_labeler(threshold=0.02)
    labeler.match_embedding(unit([1.0, 0.0]))  # S1
    labeler.match_embedding(unit([0.0, 1.0]))  # S2
    before = [c.copy() for c in labeler._centroids]
    counts_before = list(labeler._counts)
    a = unit([0.95, 0.05])
    b = unit([0.1, 0.99])
    labels = labeler.match_embeddings_joint([a, b], update=False, threshold=0.02, source="remap")
    assert labels == ["S1", "S2"]
    assert counts_before == labeler._counts
    for old, new in zip(before, labeler._centroids):
        assert np.allclose(old, new)


def test_joint_remap_empty_input_returns_empty_list():
    labeler = make_labeler(threshold=0.3)
    assert labeler.match_embeddings_joint([]) == []


# ---------------------------------------------------------------------------
# docs/DIARIZATION_PLAN.md section 10.6/10.8: centroid-open diagnostics
# ---------------------------------------------------------------------------

def test_match_embedding_default_source_is_empty_and_only_logged_on_open():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))  # opens S1, no source given
    # a hit on an existing centroid must not append to the open log at all
    labeler.match_embedding(unit([0.99, 0.01]))
    assert labeler._open_log == [""]


def test_centroid_open_counts_tags_fast_and_remap_sources_separately():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]), source="fast")     # opens S1
    labeler.match_embedding(unit([0.0, 1.0]), source="fast")     # opens S2 (miss)
    labeler.match_embedding(unit([-1.0, 0.0]), source="remap")   # opens S3 (miss)
    assert labeler.centroid_open_counts() == {"fast": 2, "remap": 1}


def test_label_forwards_source_to_match_embedding():
    labeler = make_labeler(threshold=0.9)
    labeler._extractor = None  # embed() isn't reached: patch label()'s callee instead
    labeler.embed = lambda samples, sr: unit([1.0, 0.0])
    labeler.label(np.zeros(10), 16000)  # default source="fast"
    labeler.embed = lambda samples, sr: unit([0.0, 1.0])
    labeler.label(np.zeros(10), 16000, source="probe")
    assert labeler._open_log == ["fast", "probe"]


def test_centroid_summary_flags_one_off_opens_with_a_final_match_count_of_one():
    # docs/DIARIZATION_PLAN.md section 10.8's root-cause finding: production
    # prints every fast-path S{n} the instant it opens, including a
    # centroid that (like this one) never gets matched again -- exactly
    # what centroid_summary()'s final_match_count is meant to surface, since
    # eval_diar.py's DER hypothesis construction (per-group majority vote /
    # remap-selected labels) structurally never surfaces such a label
    # unless it happens to also win a group.
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]), source="fast")   # S1: matched again below
    labeler.match_embedding(unit([0.99, 0.01]), source="fast")  # folds into S1
    labeler.match_embedding(unit([0.0, 1.0]), source="fast")   # S2: a one-off outlier
    assert labeler.centroid_summary() == [
        ("S1", "fast", 2),
        ("S2", "fast", 1),
    ]


# ---------------------------------------------------------------------------
# issue #11 (docs/DIARIZATION_PLAN.md section 10.8, option B): display-only
# provisional labeling. Unlike section 9's hysteresis, this must NEVER
# change what match_embedding()/label() return (assignment) -- only what a
# caller shows via display_label() at print time.
# ---------------------------------------------------------------------------

def test_new_centroid_is_provisional_until_second_hit():
    labeler = make_labeler(threshold=0.9)
    label = labeler.match_embedding(unit([1.0, 0.0]))  # opens S1, 1st hit
    assert label == "S1"  # assignment: plain canonical label, no "?"
    assert labeler.is_provisional("S1") is True
    assert labeler.display_label("S1") == "S1?"

    label2 = labeler.match_embedding(unit([0.99, 0.01]))  # 2nd hit on S1
    assert label2 == "S1"  # assignment unchanged by provisional status
    assert labeler.is_provisional("S1") is False
    assert labeler.display_label("S1") == "S1"


def test_confirmed_centroid_stays_confirmed_after_more_hits():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))
    labeler.match_embedding(unit([0.99, 0.01]))  # confirms S1 (count=2)
    labeler.match_embedding(unit([0.98, 0.02]))  # a 3rd hit
    assert labeler.is_provisional("S1") is False
    assert labeler.display_label("S1") == "S1"


def test_display_label_does_not_affect_which_centroid_is_assigned():
    """The core invariant: display_label() is purely a read of _counts, so
    calling it (or is_provisional()) any number of times, in any order,
    must never change _centroids/_counts/_alias -- the assignment a later
    match_embedding() call makes is unaffected."""
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))
    before_centroids = [c.copy() for c in labeler._centroids]
    before_counts = list(labeler._counts)

    labeler.display_label("S1")
    labeler.is_provisional("S1")
    labeler.display_label("S1")

    assert [c.tolist() for c in labeler._centroids] == [c.tolist() for c in before_centroids]
    assert labeler._counts == before_counts

    # and a real assignment call afterwards behaves exactly as it would have
    # with no display_label()/is_provisional() calls in between.
    label = labeler.match_embedding(unit([0.0, 1.0]))
    assert label == "S2"


def test_display_label_empty_string_passthrough():
    labeler = make_labeler(threshold=0.9)
    assert labeler.is_provisional("") is False
    assert labeler.display_label("") == ""


def test_display_label_out_of_range_label_passthrough():
    """A label for a centroid this labeler has never opened (e.g. stale
    state from a caller bug) must not raise -- treated as not provisional."""
    labeler = make_labeler(threshold=0.9)
    assert labeler.is_provisional("S9") is False
    assert labeler.display_label("S9") == "S9"


def test_provisional_confirm_hits_constant_is_two():
    # Guards the "confirmed on the 2nd occurrence" spec in issue #11 against
    # an accidental constant change silently altering behavior.
    assert PROVISIONAL_CONFIRM_HITS == 2


def test_provisional_label_count_counts_only_still_provisional_centroids():
    labeler = make_labeler(threshold=0.9)
    labeler.match_embedding(unit([1.0, 0.0]))   # S1: 1 hit, provisional
    labeler.match_embedding(unit([0.99, 0.01]))  # confirms S1
    labeler.match_embedding(unit([0.0, 1.0]))   # S2: 1 hit, provisional
    labeler.match_embedding(unit([-1.0, 0.0]))  # S3: 1 hit, provisional
    assert labeler.provisional_label_count() == 2  # S2, S3


def test_provisional_label_count_skips_merged_away_centroids():
    labeler = make_labeler(merge_enabled=True, merge_threshold=0.9)
    labeler._centroids = [unit([1.0, 0.0]), unit([0.99, 0.1411])]  # both 1-hit, provisional
    labeler._counts = [1, 1]
    labeler._confirmed = [True, True]
    labeler.merge_centroids()  # folds S2 into S1; S2's slot is now aliased/dead
    # _merge_into() sums both centroids' counts (1+1=2), so the survivor is
    # confirmed by the merge itself; the aliased S2 slot must not be
    # separately counted as still-provisional.
    assert labeler.provisional_label_count() == 0


# ---------------------------------------------------------------------------
# iteration 6 (docs/DIARIZATION_PLAN.md section 9): new-speaker hysteresis
# ---------------------------------------------------------------------------

def test_hysteresis_first_speaker_confirmed_immediately():
    labeler = make_labeler(threshold=0.9, hysteresis_enabled=True)
    label = labeler.match_embedding(unit([1.0, 0.0]))
    assert label == "S1"
    assert labeler._confirmed == [True]  # nothing to fall back to yet


def test_hysteresis_provisional_speaker_falls_back_to_nearest_confirmed():
    labeler = make_labeler(threshold=0.9, hysteresis_enabled=True, hysteresis_min_hits=2)
    labeler.match_embedding(unit([1.0, 0.0]))          # S1, confirmed
    label = labeler.match_embedding(unit([0.0, 1.0]))  # miss -> opens provisional 2nd centroid
    assert label == "S1"  # displayed under nearest confirmed speaker, not a new "S2"
    assert len(labeler._centroids) == 2  # still opened its own centroid internally
    assert labeler._confirmed == [True, False]


def test_hysteresis_confirms_after_min_hits_and_gets_own_label():
    labeler = make_labeler(threshold=0.9, hysteresis_enabled=True, hysteresis_min_hits=2)
    labeler.match_embedding(unit([1.0, 0.0]))              # S1
    labeler.match_embedding(unit([0.0, 1.0]))              # opens provisional, shown as S1
    label = labeler.match_embedding(unit([0.01, 0.9999]))  # 2nd hit on the provisional centroid
    assert label == "S2"  # now confirmed
    assert labeler._confirmed == [True, True]


def test_hysteresis_disabled_opens_new_speaker_immediately():
    labeler = make_labeler(threshold=0.9, hysteresis_enabled=False)
    labeler.match_embedding(unit([1.0, 0.0]))
    label = labeler.match_embedding(unit([0.0, 1.0]))
    assert label == "S2"  # unchanged behavior when the flag is off


# ---------------------------------------------------------------------------
# iteration 6 (docs/DIARIZATION_PLAN.md section 9): periodic centroid merging
# ---------------------------------------------------------------------------

def test_merge_centroids_folds_similar_speakers():
    labeler = make_labeler(merge_enabled=True, merge_threshold=0.9)
    labeler._centroids = [unit([1.0, 0.0]), unit([0.99, 0.1411])]  # cos sim ~0.99
    labeler._counts = [1, 1]
    labeler._confirmed = [True, True]

    merged = labeler.merge_centroids()

    assert merged == {"S2": "S1"}
    assert labeler._alias == {1: 0}
    assert labeler.merge_history() == {"S2": "S1"}


def test_merge_centroids_below_threshold_does_nothing():
    labeler = make_labeler(merge_threshold=0.99)
    labeler._centroids = [unit([1.0, 0.0]), unit([0.0, 1.0])]  # orthogonal
    labeler._counts = [1, 1]
    labeler._confirmed = [True, True]

    merged = labeler.merge_centroids()

    assert merged == {}
    assert labeler._alias == {}
    assert labeler.merge_history() == {}


def test_maybe_merge_centroids_is_noop_when_disabled():
    labeler = make_labeler(merge_enabled=False, merge_threshold=0.9)
    labeler._centroids = [unit([1.0, 0.0]), unit([0.99, 0.1411])]
    labeler._counts = [1, 1]
    labeler._confirmed = [True, True]

    merged = labeler.maybe_merge_centroids()

    assert merged == {}
    assert labeler._alias == {}  # merge_enabled=False never touches state


def test_merged_centroid_absorbs_future_matches():
    labeler = make_labeler(threshold=0.5, merge_enabled=True, merge_threshold=0.9)
    labeler._centroids = [unit([1.0, 0.0]), unit([0.99, 0.1411])]
    labeler._counts = [1, 1]
    labeler._confirmed = [True, True]
    labeler.merge_centroids()  # folds S2 into S1

    label = labeler.match_embedding(unit([0.0, 1.0]))  # would have been nearest to old S2

    assert label != "S2"  # S2's slot is dead; nothing should ever be labeled S2 again
    assert len(labeler._centroids) == 3  # opened a brand-new (3rd) centroid instead
    assert label == "S3"


def test_merge_history_flattens_across_calls():
    """A speaker merged away in one call, whose survivor is itself merged
    away in a later call, should map straight to the final survivor -- not
    leave callers to walk a merge_history() chain themselves."""
    labeler = make_labeler(merge_threshold=0.9)
    labeler._centroids = [unit([1.0, 0.0]), unit([0.5, 0.8660]), unit([0.49, 0.8717])]
    labeler._counts = [1, 1, 1]
    labeler._confirmed = [True, True, True]

    first = labeler.merge_centroids()  # S2, S3 are near-identical -> S3 merges into S2
    assert first == {"S3": "S2"}

    # Simulate the merged S2 centroid later drifting close to S1 (e.g. more
    # audio folded in over the session).
    labeler._centroids[1] = unit([0.95, 0.3122])
    second = labeler.merge_centroids()  # now S2 merges into S1
    assert second == {"S2": "S1"}

    assert labeler.merge_history() == {"S3": "S1", "S2": "S1"}


# ---------------------------------------------------------------------------
# real-audio regression: fast path on a genuine short two-speaker recording
# (docs/DIARIZATION_PLAN.md section 9 / iteration 6's speaker-count work
# must not regress the module's stated use case -- "turn-taking
# conversations", commonly just 1-2 speakers, per the module docstring)
# ---------------------------------------------------------------------------

TESTDATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testdata")
TWO_SPEAKERS_WAV = os.path.join(TESTDATA_DIR, "two_speakers.wav")

_needs_embed_model = pytest.mark.skipif(
    not os.path.exists(EMBED_MODEL), reason="campplus_sv.onnx not downloaded"
)
_needs_two_speakers_wav = pytest.mark.skipif(
    not os.path.exists(TWO_SPEAKERS_WAV), reason="testdata/two_speakers.wav not present"
)


def _read_wav_float32(path: str):
    import wave

    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        data = f.readframes(f.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


@_needs_embed_model
@_needs_two_speakers_wav
def test_fast_path_default_finds_both_real_speakers():
    """Default SpeakerLabeler() (hysteresis_enabled=False, section 9's
    conclusion) must still tell testdata/two_speakers.wav's two real
    speakers apart when fed their utterances in chronological order --
    the ground truth this module exists to serve. Segment boundaries come
    from GroupDiarizer's own offline diarization (already asserted correct
    for this fixture in tests/test_diarize.py) so this test is purely
    about SpeakerLabeler.label(), not diarization quality.
    """
    from diarize import SEGMENTATION_MODEL, GroupDiarizer

    if not os.path.exists(SEGMENTATION_MODEL):
        pytest.skip("pyannote segmentation-3.0 model not downloaded")

    samples, sr = _read_wav_float32(TWO_SPEAKERS_WAV)
    segments = sorted(GroupDiarizer().process(samples, sr), key=lambda t: t[1])

    labeler = SpeakerLabeler()
    labels = []
    for _local_spk, start_s, end_s in segments:
        start, end = int(start_s * sr), int(end_s * sr)
        if end - start < int(0.3 * sr):
            continue
        labels.append(labeler.label(samples[start:end], sr))

    assert len(set(labels)) >= 2, (
        f"expected >=2 distinct global labels for a real 2-speaker recording, got {labels}"
    )


@_needs_embed_model
@_needs_two_speakers_wav
def test_provisional_display_does_not_reproduce_section_9s_swallowing_bug():
    """Issue #11's display-only mitigation must NOT reproduce section 9's
    hysteresis failure mode (test_hysteresis_can_swallow_a_rare_real_speaker
    above): since display_label() never changes which centroid an embedding
    is assigned to, the rare second speaker (one ~3s segment among four)
    still gets their own distinct canonical label -- it just displays with
    a "?" suffix (never confirmed by a 2nd hit) instead of a bare "S2".
    Structurally this can't regress to the section-9 bug because assignment
    is untouched, but this test fixes that as an explicit guard.
    """
    from diarize import SEGMENTATION_MODEL, GroupDiarizer

    if not os.path.exists(SEGMENTATION_MODEL):
        pytest.skip("pyannote segmentation-3.0 model not downloaded")

    samples, sr = _read_wav_float32(TWO_SPEAKERS_WAV)
    segments = sorted(GroupDiarizer().process(samples, sr), key=lambda t: t[1])

    labeler = SpeakerLabeler()  # defaults: hysteresis/merge both off
    canonical_labels = []
    display_labels = []
    for _local_spk, start_s, end_s in segments:
        start, end = int(start_s * sr), int(end_s * sr)
        if end - start < int(0.3 * sr):
            continue
        label = labeler.label(samples[start:end], sr)
        canonical_labels.append(label)
        display_labels.append(labeler.display_label(label))

    # assignment: unchanged from the pre-existing fast-path regression test
    assert len(set(canonical_labels)) >= 2, (
        f"expected >=2 distinct canonical labels, got {canonical_labels}"
    )
    # display: still >=2 distinct strings shown to the user -- the rare
    # speaker isn't silently absorbed into the other one's label, unlike
    # section 9's rejected hysteresis approach.
    assert len(set(display_labels)) >= 2, (
        f"expected >=2 distinct displayed labels, got {display_labels}"
    )


@_needs_embed_model
@_needs_two_speakers_wav
def test_hysteresis_can_swallow_a_rare_real_speaker():
    """Documents why hysteresis_enabled defaults to False (see speaker_id.py's
    HYSTERESIS_MIN_HITS comment): testdata/two_speakers.wav's second
    speaker only speaks once (one ~3s segment among four), so with
    hysteresis on they never accumulate hysteresis_min_hits and are
    permanently displayed under the first speaker's label instead of
    their own -- collapsing a genuine 2-speaker recording down to 1
    reported speaker. This is a reproduction/regression guard, not an
    endorsement -- if this ever starts passing with >=2 labels, the
    trade-off that kept hysteresis off by default may be worth revisiting.
    """
    from diarize import SEGMENTATION_MODEL, GroupDiarizer

    if not os.path.exists(SEGMENTATION_MODEL):
        pytest.skip("pyannote segmentation-3.0 model not downloaded")

    samples, sr = _read_wav_float32(TWO_SPEAKERS_WAV)
    segments = sorted(GroupDiarizer().process(samples, sr), key=lambda t: t[1])

    labeler = SpeakerLabeler(hysteresis_enabled=True)  # opt-in, off by default
    labels = []
    for _local_spk, start_s, end_s in segments:
        start, end = int(start_s * sr), int(end_s * sr)
        if end - start < int(0.3 * sr):
            continue
        labels.append(labeler.label(samples[start:end], sr))

    assert len(set(labels)) == 1, (
        f"expected the known under-count failure mode (all one label), got {labels} -- "
        "if this now finds >=2 speakers, revisit hysteresis_enabled's default"
    )
