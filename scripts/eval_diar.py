"""Diarization Error Rate (DER) scoring for hayamimi's speaker labeling.

docs/DIARIZATION_PLAN.md iteration (2): score the current --speakers
(scripts/speaker_id.py's SpeakerLabeler, a naive online-nearest-centroid
turn-taking assignment) against the AMI reference subset built by
scripts/make_diarset.py, using DER as the metric.

Two independent pieces:
  - RTTM parsing + DER computation (parse_rttm, DER via the `simpleder`
    package) -- pure functions, unit-tested in tests/test_diar_eval.py.
  - Hypothesis generation: runs the *production* pipeline pieces (the same
    VAD builder and the same SpeakerLabeler class realtime_transcribe.py
    uses for --speakers) over a wav file to get (speaker, start, end)
    triples, without going through ASR decoding (not needed for a speaker
    label + timing hypothesis, and skipping it makes the sweep much faster).

Usage:
    python scripts/eval_diar.py                       # score all meetings in the manifest
    python scripts/eval_diar.py --meeting ES2011a      # just one
    python scripts/eval_diar.py --collar 0.0           # no collar (stricter)
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

EVAL_DIAR_DIR = os.path.join(ROOT, "testdata", "eval_diar")
MANIFEST_PATH = os.path.join(EVAL_DIAR_DIR, "manifest.json")

DEFAULT_COLLAR = 0.25  # NIST RT convention: forgive this many seconds around
                        # each reference turn boundary in both directions


# ---------------------------------------------------------------------------
# RTTM parsing (pure, unit-tested)
# ---------------------------------------------------------------------------

def parse_rttm(path: str) -> list[tuple[str, float, float]]:
    """Parse a NIST RTTM file's SPEAKER lines into (speaker, start, end) triples.

    SPEAKER line format: `SPEAKER <file> <chnl> <start> <dur> <NA> <NA>
    <speaker> <NA> <NA>`. Non-SPEAKER lines and blank lines are ignored.
    Raises ValueError on a malformed SPEAKER line (too few fields, or start
    later than start+dur i.e. a negative duration) rather than silently
    dropping bad data.
    """
    turns = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            if len(parts) < 8:
                raise ValueError(f"{path}:{lineno}: malformed SPEAKER line: {line!r}")
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            if dur < 0:
                raise ValueError(f"{path}:{lineno}: negative duration: {line!r}")
            turns.append((speaker, start, start + dur))
    return turns


def segments_to_der_tuples(segments: list[tuple[str, float, float]]
                           ) -> list[tuple[str, float, float]]:
    """Coerce a (speaker, start, end) list to the exact types simpleder wants:
    speaker must be str, start/end must be Python float (not numpy float32,
    which simpleder's isinstance checks reject) and start <= end.

    Zero/negative-length segments are dropped -- they carry no scorable
    speech time and simpleder's `check_input` rejects start > end.
    """
    out = []
    for speaker, start, end in segments:
        s, e = float(start), float(end)
        if e > s:
            out.append((str(speaker), s, e))
    return out


# ---------------------------------------------------------------------------
# Hypothesis generation: production VAD + SpeakerLabeler, no ASR
# ---------------------------------------------------------------------------

def generate_speaker_hypothesis(wav_path: str, min_silence: float = 0.35,
                                max_speech: float = 12.0,
                                hysteresis_enabled: bool | None = None,
                                hysteresis_min_hits: int = None,
                                vad_threshold: float = 0.5) -> list[tuple[str, float, float]]:
    """Run hayamimi's real --speakers pipeline pieces over a wav file and
    return (speaker, start_s, end_s) for each VAD-finalized segment.

    Deliberately reuses the exact production building blocks so this is a
    faithful baseline measurement, not a reimplementation:
      - realtime_transcribe.build_vad(): the same Silero VAD config
        (min_speech_duration=0.25s, the configured min_silence_duration,
        max_speech_duration) used by --speakers.
      - realtime_transcribe.AudioHistory.with_preroll(): the same
        pre-onset audio padding drain_segments() feeds into the speaker
        embedder (not just the raw VAD segment) before labeling.
      - speaker_id.SpeakerLabeler: unmodified, same nearest-centroid
        assignment logic as production.

    ASR decoding (RoutedASR) is intentionally skipped: it has no bearing on
    the speaker-label + turn-timing hypothesis DER scores, and skipping it
    makes the AMI sweep an order of magnitude faster.
    """
    import numpy as np
    from realtime_transcribe import SAMPLE_RATE, AudioHistory, build_vad, read_wave, wav_chunks
    from speaker_id import SpeakerLabeler

    samples, sr = read_wave(wav_path, target_rate=SAMPLE_RATE)
    vad = build_vad(min_silence, max_speech, vad_threshold)
    history = AudioHistory(sr)
    # merge_enabled is deliberately not threaded through here: the baseline
    # method has no group/"clean copy" boundary to hang maybe_merge_centroids()
    # off of (see generate_diarize_hypothesis's docstring) -- only
    # hysteresis, which is purely a fast-path (label()) behavior, applies.
    # hysteresis_enabled=None (the default, distinct from False) means "use
    # speaker_id.SpeakerLabeler's own default" (False -- section 9 tried
    # and rejected it as the module default, see speaker_id.py's
    # HYSTERESIS_MIN_HITS comment), matching how remap_threshold=None
    # already worked in this module. Pass --hysteresis-enabled explicitly
    # to measure the opt-in mitigation anyway.
    labeler_kwargs = {}
    if hysteresis_enabled is not None:
        labeler_kwargs["hysteresis_enabled"] = hysteresis_enabled
    if hysteresis_min_hits is not None:
        labeler_kwargs["hysteresis_min_hits"] = hysteresis_min_hits
    labeler = SpeakerLabeler(**labeler_kwargs)

    hyp: list[tuple[str, float, float]] = []

    def drain():
        while not vad.empty():
            segment = vad.front
            seg_samples = np.asarray(segment.samples, dtype=np.float32)
            seg_start, seg_end = segment.start, segment.start + len(seg_samples)
            labeled_samples = history.with_preroll(seg_start, seg_samples)
            vad.pop()
            label = labeler.label(labeled_samples, sr)
            hyp.append((label, seg_start / sr, seg_end / sr))

    for chunk in wav_chunks(samples, sr, realtime=False):
        vad.accept_waveform(chunk)
        history.push(chunk)
        drain()

    vad.flush()
    drain()
    return hyp


# ---------------------------------------------------------------------------
# New (iteration 3-4) hypothesis: VAD -> fast label -> Refiner-style groups
# -> per-group offline diarization -> local-to-global remap
# ---------------------------------------------------------------------------

def group_segments(fast_segments: list[tuple[int, int, str]], sample_rate: int,
                   group_gap_s: float, group_max_s: float) -> list[list[tuple[int, int, str]]]:
    """Pure grouping logic, factored out of generate_diarize_hypothesis() for
    unit testing: replicates realtime_transcribe.Refiner.maybe_refine()'s
    "due" condition (silence gap >= group_gap_s OR accumulated length >=
    group_max_s closes a group) over a flat list of (start_sample,
    end_sample, label) VAD segments, minus the language-boundary split
    (there's no decoded text here to split on).
    """
    groups: list[list[tuple[int, int, str]]] = []
    cur: list[tuple[int, int, str]] = []
    for seg in fast_segments:
        if cur:
            first_start, last_end = cur[0][0], cur[-1][1]
            gap = seg[0] - last_end
            length = last_end - first_start
            if gap >= int(group_gap_s * sample_rate) or length >= int(group_max_s * sample_rate):
                groups.append(cur)
                cur = []
        cur.append(seg)
    if cur:
        groups.append(cur)
    return groups


def generate_diarize_hypothesis(wav_path: str, min_silence: float = 0.35,
                                max_speech: float = 12.0,
                                diar_threshold: float = None,
                                sim_threshold: float = None,
                                remap_threshold: float = None,
                                merge_enabled: bool | None = None,
                                merge_threshold: float = None,
                                hysteresis_enabled: bool | None = None,
                                hysteresis_min_hits: int = None,
                                min_duration_on: float = None,
                                min_duration_off: float = None,
                                vad_threshold: float = 0.5,
                                min_remap_update_s: float = 0.0,
                                joint_remap: bool = False,
                                ) -> tuple[list[tuple[str, float, float]], dict]:
    """Score the new method: same VAD + fast SpeakerLabeler as the baseline,
    but grouped exactly the way production's Refiner groups utterances
    (silence gap >= GROUP_GAP_S or accumulated length >= GROUP_MAX_S closes
    a group -- see realtime_transcribe.Refiner.add_span/maybe_refine), then
    each closed group's audio is re-diarized with
    scripts/diarize.GroupDiarizer (pyannote segmentation + CAM++ + FastClustering)
    exactly as realtime_transcribe.Refiner._emit_turns() does, remapping
    local clusters onto a session-global SpeakerLabeler via match_embedding().

    This reuses the *real* Refiner grouping/remap policy (same constants,
    same classes) but skips ASR entirely -- the hypothesis only needs
    speaker labels and turn timing, and running ASR would just slow down
    the AMI sweep for no scoring benefit.

    Returns (hyp, stats) where stats carries diarization wall-clock time
    and call count, for the RTF measurement docs/DIARIZATION_PLAN.md
    iteration 3 asks for.

    sim_threshold/remap_threshold plumb straight through to
    speaker_id.SpeakerLabeler's two independently-tunable thresholds (see
    its __init__ docstring): sim_threshold governs the fast-path label()
    calls that build fast_segments below, remap_threshold governs the
    match_embedding() calls that remap each group's local diarization
    clusters onto the global centroid set. sim_threshold defaults to
    speaker_id.SIM_THRESHOLD when None; remap_threshold defaults to
    speaker_id.REMAP_THRESHOLD when None (SpeakerLabeler's own default --
    *not* falling back to sim_threshold, so this function's default call
    matches production's default, not the old single-threshold behavior).
    docs/DIARIZATION_PLAN.md section 8 (iteration 5) swept these
    independently of diar_threshold (diarize.DEFAULT_THRESHOLD, which only
    affects the *local* clustering within one group) and landed on
    REMAP_THRESHOLD=0.35 with SIM_THRESHOLD left at 0.45.

    merge_enabled/merge_threshold and hysteresis_enabled/hysteresis_min_hits
    plumb through to speaker_id.SpeakerLabeler's iteration-6 (section 9)
    speaker-count mitigations. merge_centroids() is called once per closed
    group (this function's natural "clean copy" boundary), matching how
    realtime_transcribe.Refiner._emit_turns() would call it.

    min_duration_on/min_duration_off plumb straight through to
    diarize.GroupDiarizer's sherpa-onnx OfflineSpeakerDiarizationConfig
    (default None -- falls back to diarize.DEFAULT_MIN_DURATION_ON/OFF).
    Round 2 (docs/DIARIZATION_PLAN.md section 12) sweeps these.

    min_remap_update_s (Round 4, docs/DIARIZATION_PLAN.md section 14 T2):
    0.0 (default) is a no-op -- every local cluster remaps exactly as
    before (match_embedding(update=True), which can fold the cluster's
    embedding into an existing global centroid's running mean, or open a
    brand-new centroid, on every call). When > 0, a local cluster whose
    total speech duration is below this many seconds is remapped
    READ-ONLY instead (match_embedding(update=False)): it can still match
    an existing centroid (so it gets labeled), but it can never fold its
    (likely noisier, since it's short) embedding into that centroid's mean
    and can never open a brand-new global centroid on a miss -- it falls
    back to the group's fast-path majority label instead. Round 4's T1
    attribution found REMAP error (a local cluster whose own diarization
    was correct but the global relabel step picked the wrong S{n})
    dominates confusion on both IS1009a (61%) and ES2011a (49%); this
    flag is the "ignore clusters <1s for centroid updates" experiment T2
    describes as one candidate fix for that.

    joint_remap (Round 5, docs/DIARIZATION_PLAN.md section 15 T1): False
    (default) is a no-op -- each group's local clusters (other than any
    excluded by min_remap_update_s above) still remap independently via
    match_embedding(), exactly the pre-existing behavior. When True,
    those clusters are remapped together via
    speaker_id.SpeakerLabeler.match_embeddings_joint() (Hungarian
    assignment maximizing total similarity, constrained so distinct local
    clusters can't land on the same global speaker) -- see that method's
    docstring. Mirrors realtime_transcribe.Refiner._emit_turns() exactly,
    same as every other flag here.
    """
    import numpy as np
    from realtime_transcribe import GROUP_GAP_S, GROUP_MAX_S, SAMPLE_RATE, AudioHistory, build_vad, read_wave, wav_chunks
    from speaker_id import SIM_THRESHOLD, SpeakerLabeler
    from diarize import DEFAULT_MIN_DURATION_OFF, DEFAULT_MIN_DURATION_ON, DEFAULT_THRESHOLD, GroupDiarizer

    samples, sr = read_wave(wav_path, target_rate=SAMPLE_RATE)
    vad = build_vad(min_silence, max_speech, vad_threshold)
    history = AudioHistory(sr)
    # the single session-global centroid set, shared by fast path + remap.
    # Only pass remap_threshold/merge_threshold/hysteresis_min_hits through
    # when explicitly given -- passing None explicitly (vs. omitting the
    # kwarg) makes SpeakerLabeler fall back to its own module-level default
    # instead of whatever this function's caller left unspecified.
    # merge_enabled/hysteresis_enabled=None means "use SpeakerLabeler's own
    # default" -- both default False (section 9 tried and rejected both as
    # the module default, see speaker_id.py's HYSTERESIS_MIN_HITS comment),
    # same tri-state idiom as remap_threshold above.
    labeler_kwargs = {}
    if merge_enabled is not None:
        labeler_kwargs["merge_enabled"] = merge_enabled
    if hysteresis_enabled is not None:
        labeler_kwargs["hysteresis_enabled"] = hysteresis_enabled
    if remap_threshold is not None:
        labeler_kwargs["remap_threshold"] = remap_threshold
    if merge_threshold is not None:
        labeler_kwargs["merge_threshold"] = merge_threshold
    if hysteresis_min_hits is not None:
        labeler_kwargs["hysteresis_min_hits"] = hysteresis_min_hits
    labeler = SpeakerLabeler(
        threshold=SIM_THRESHOLD if sim_threshold is None else sim_threshold,
        **labeler_kwargs,
    )
    diarizer = GroupDiarizer(
        threshold=DEFAULT_THRESHOLD if diar_threshold is None else diar_threshold,
        min_duration_on=DEFAULT_MIN_DURATION_ON if min_duration_on is None else min_duration_on,
        min_duration_off=DEFAULT_MIN_DURATION_OFF if min_duration_off is None else min_duration_off,
    )

    fast_segments: list[tuple[int, int, str]] = []  # (start_sample, end_sample, fast S{n})

    def drain():
        while not vad.empty():
            segment = vad.front
            seg_samples = np.asarray(segment.samples, dtype=np.float32)
            seg_start, seg_end = segment.start, segment.start + len(seg_samples)
            labeled_samples = history.with_preroll(seg_start, seg_samples)
            vad.pop()
            label = labeler.label(labeled_samples, sr)
            fast_segments.append((seg_start, seg_end, label))

    for chunk in wav_chunks(samples, sr, realtime=False):
        vad.accept_waveform(chunk)
        history.push(chunk)
        drain()
    vad.flush()
    drain()

    groups = group_segments(fast_segments, sr, GROUP_GAP_S, GROUP_MAX_S)

    hyp: list[tuple[str, float, float]] = []
    diar_time = 0.0
    for group in groups:
        g_start, g_end = group[0][0], group[-1][1]
        buf = samples[g_start:g_end]
        fast_labels = [lbl for _, _, lbl in group]
        majority = max(set(fast_labels), key=fast_labels.count) if fast_labels else ""

        t0 = time.time()
        try:
            raw = diarizer.process(buf, sr)
        except Exception:
            raw = []
        diar_time += time.time() - t0

        turns = [(lid, s, e) for lid, s, e in raw if e - s >= 0.3]
        if len({lid for lid, _, _ in turns}) < 2:
            # single speaker (or diarizer declined): same fallback
            # Refiner._emit_turns() takes -- one majority-vote span.
            hyp.append((majority, g_start / sr, g_end / sr))
            continue

        local_ids = sorted({t[0] for t in turns})
        global_label: dict[int, str] = {}
        cluster_embs: dict[int, np.ndarray] = {}
        empty_ids = []
        for local_id in local_ids:
            pieces = [buf[int(round(s * sr)):int(round(e * sr))]
                      for lid, s, e in turns if lid == local_id]
            cluster_audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            if len(cluster_audio) == 0:
                global_label[local_id] = majority
                empty_ids.append(local_id)
                continue
            cluster_embs[local_id] = labeler.embed(cluster_audio, sr)

        # min_remap_update_s (Round 4 T2) still gates short clusters to a
        # read-only probe, independent of joint_remap below -- see
        # generate_diarize_hypothesis()'s docstring. Only the remaining
        # clusters are eligible for the Round 5 T1 joint assignment.
        short_ids = []
        for local_id in cluster_embs:
            cluster_dur_s = sum(e - s for lid, s, e in turns if lid == local_id)
            if min_remap_update_s > 0 and cluster_dur_s < min_remap_update_s:
                short_ids.append(local_id)
                probe = labeler.match_embedding(
                    cluster_embs[local_id], update=False, threshold=labeler.remap_threshold,
                    source="remap")
                global_label[local_id] = probe if probe else majority

        joint_ids = [lid for lid in local_ids if lid not in short_ids and lid not in empty_ids]
        if joint_ids:
            if joint_remap:
                labels = labeler.match_embeddings_joint(
                    [cluster_embs[lid] for lid in joint_ids], update=True,
                    threshold=labeler.remap_threshold, source="remap")
                for local_id, label in zip(joint_ids, labels):
                    global_label[local_id] = label
            else:
                for local_id in joint_ids:
                    global_label[local_id] = labeler.match_embedding(
                        cluster_embs[local_id], update=True, threshold=labeler.remap_threshold,
                        source="remap")

        # iteration 6 (section 9): this group's remap is exactly the "clean
        # copy" boundary merge_centroids() is meant for -- give it a chance
        # to fold any two global centroids that have drifted together
        # before the next group opens new ones against them.
        labeler.maybe_merge_centroids()

        for local_id, s, e in turns:
            hyp.append((global_label[local_id], (g_start + s * sr) / sr, (g_start + e * sr) / sr))

    stats = {
        "diar_time_s": diar_time,
        "n_groups": len(groups),
        "audio_s": len(samples) / sr,
        "merge_history": labeler.merge_history(),
        # docs/DIARIZATION_PLAN.md section 10.6 diagnostics: compare against
        # realtime_transcribe.py's "session summary" refine_groups_closed /
        # centroid_open_counts lines for the same file -- this eval path has
        # no decoded text to force a language-boundary group split (see
        # group_segments()'s docstring), so a materially higher n_groups /
        # "remap"-sourced centroid count in production than here is evidence
        # that language-boundary over-splitting (not just this dual
        # fast+remap architecture, which both paths share) drives the extra
        # global speakers observed in the full pipeline.
        "centroid_open_counts": labeler.centroid_open_counts(),
        "centroid_summary": labeler.centroid_summary(),
    }
    return hyp, stats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def der_breakdown(ref: list[tuple[str, float, float]], hyp: list[tuple[str, float, float]],
                  collar: float, skip_overlap: bool = False) -> dict:
    """Miss/False-Alarm/Confusion breakdown via pyannote.metrics, as a
    fraction of total reference speech time (same units simpleder.DER's
    return value is in). Kept separate from score_meeting()'s simpleder
    call (docs/DIARIZATION_PLAN.md section 4's two-tier plan: simpleder for
    the fast day-to-day sweep loop, pyannote.metrics only when the error
    breakdown itself is needed -- iteration 5, to tell whether the
    remaining over-splitting after the section-8 threshold tuning is
    boundary noise (Miss/FA) or genuine speaker confusion).

    skip_overlap (T2, docs/DIARIZATION_PLAN.md section 12): pass
    skip_overlap=True to pyannote.metrics.DiarizationErrorRate, which
    excludes reference regions with >=2 concurrent speakers from scoring
    entirely. This is pyannote's own published-comparison convention and
    the "what we can actually get" number given hayamimi emits one speaker
    per time span (see the Round 1 overlap-floor analysis).

    Requires pyannote.metrics (requirements-dev.txt); imported lazily so
    scoring/sweeping without it stays working.
    """
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def to_annotation(turns):
        ann = Annotation()
        for speaker, start, end in turns:
            ann[Segment(start, end)] = speaker
        return ann

    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    detail = metric(to_annotation(ref), to_annotation(hyp), detailed=True)
    total = detail["total"] or 1e-9  # guard empty reference (shouldn't happen on real data)
    return {
        "der_breakdown": detail["diarization error rate"],
        "miss": detail["missed detection"] / total,
        "false_alarm": detail["false alarm"] / total,
        "confusion": detail["confusion"] / total,
    }


def score_meeting(wav_path: str, rttm_path: str, collar: float,
                  min_silence: float = 0.35, max_speech: float = 12.0,
                  method: str = "baseline", diar_threshold: float = None,
                  sim_threshold: float = None, remap_threshold: float = None,
                  merge_enabled: bool | None = None, merge_threshold: float = None,
                  hysteresis_enabled: bool | None = None, hysteresis_min_hits: int = None,
                  min_duration_on: float = None, min_duration_off: float = None,
                  breakdown: bool = False, skip_overlap: bool = False,
                  vad_threshold: float = 0.5, min_remap_update_s: float = 0.0,
                  joint_remap: bool = False) -> dict:
    import simpleder

    ref = segments_to_der_tuples(parse_rttm(rttm_path))
    t0 = time.time()
    extra = {}
    if method == "refine_diarize":
        hyp_raw, extra = generate_diarize_hypothesis(
            wav_path, min_silence, max_speech, diar_threshold, sim_threshold, remap_threshold,
            merge_enabled, merge_threshold, hysteresis_enabled, hysteresis_min_hits,
            min_duration_on, min_duration_off, vad_threshold, min_remap_update_s, joint_remap)
    else:
        hyp_raw = generate_speaker_hypothesis(
            wav_path, min_silence, max_speech, hysteresis_enabled, hysteresis_min_hits,
            vad_threshold)
    decode_s = time.time() - t0
    hyp = segments_to_der_tuples(hyp_raw)

    der = simpleder.DER(ref, hyp, collar=collar)
    ref_speakers = {s for s, _, _ in ref}
    hyp_speakers = {s for s, _, _ in hyp}
    result = {
        "der": der,
        "n_ref_speakers": len(ref_speakers),
        "n_hyp_speakers": len(hyp_speakers),
        "n_ref_turns": len(ref),
        "n_hyp_segments": len(hyp),
        "decode_s": decode_s,
    }
    result.update(extra)
    if breakdown:
        result.update(der_breakdown(ref, hyp, collar, skip_overlap))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--meeting", default=None,
                    help="score only this meeting id (default: all in the manifest)")
    ap.add_argument("--collar", type=float, default=DEFAULT_COLLAR,
                    help=f"NIST-style collar in seconds around reference turn boundaries "
                         f"excluded from scoring (default {DEFAULT_COLLAR})")
    ap.add_argument("--min-silence", type=float, default=0.35,
                    help="VAD min_silence_duration, same knob as realtime_transcribe.py")
    ap.add_argument("--max-speech", type=float, default=12.0,
                    help="VAD max_speech_duration, same knob as realtime_transcribe.py")
    ap.add_argument("--vad-threshold", type=float, default=0.5, metavar="T",
                    help="Silero VAD speech-probability threshold (sherpa_onnx "
                         "SileroVadModelConfig.threshold), same knob as realtime_transcribe.py's "
                         "build_vad(). Default 0.5 matches current production/sherpa_onnx "
                         "default. Lower values flag more low-energy speech as speech at the "
                         "cost of more false alarms. docs/DIARIZATION_PLAN.md section 13 "
                         "(Round 3). The installed sherpa_onnx has no speech-padding knob to "
                         "pair with this (checked: SileroVadModelConfig fields are threshold, "
                         "min_silence_duration, min_speech_duration, max_speech_duration, "
                         "window_size, neg_threshold -- no speech_pad_ms), so there is no "
                         "--vad-pad flag.")
    ap.add_argument("--method", choices=["baseline", "refine_diarize"], default="baseline",
                    help="baseline: current --speakers (SpeakerLabeler only, docs/"
                         "DIARIZATION_PLAN.md section 6). refine_diarize: iteration "
                         "3-4's Refiner-group offline diarization + global remap")
    ap.add_argument("--diar-threshold", type=float, default=None, metavar="T",
                    help="FastClusteringConfig.threshold for --method refine_diarize "
                         "(default: diarize.DEFAULT_THRESHOLD). Local clustering within one "
                         "refine group only -- see --sim-threshold/--remap-threshold for the "
                         "global speaker_id.py threshold(s).")
    ap.add_argument("--sim-threshold", type=float, default=None, metavar="T",
                    help="speaker_id.SpeakerLabeler's fast-path threshold (label() calls that "
                         "build the VAD-segment-level fast hypothesis). Default: "
                         "speaker_id.SIM_THRESHOLD (0.45). Applies to both --method values.")
    ap.add_argument("--remap-threshold", type=float, default=None, metavar="T",
                    help="speaker_id.SpeakerLabeler's remap-path threshold, used only by "
                         "--method refine_diarize's local-cluster-to-global remap "
                         "(match_embedding() calls in generate_diarize_hypothesis). Default: "
                         "speaker_id.REMAP_THRESHOLD (0.35), independent of --sim-threshold. "
                         "See docs/DIARIZATION_PLAN.md section 8.")
    ap.add_argument("--breakdown", action="store_true",
                    help="also report Miss/False-Alarm/Confusion via pyannote.metrics "
                         "(requirements-dev.txt). Slower and heavier than the default "
                         "simpleder-only scoring -- see docs/DIARIZATION_PLAN.md section 4.")
    ap.add_argument("--merge-enabled", action=argparse.BooleanOptionalAction, default=None,
                    help="iteration 6 (docs/DIARIZATION_PLAN.md section 9) mitigation A: "
                         "periodically fold global centroids that have drifted together back "
                         "into one speaker. Only meaningful with --method refine_diarize (the "
                         "merge point is once per closed group). Default: speaker_id."
                         "SpeakerLabeler's own default (False -- section 9 found this mitigation "
                         "fragile, not adopted). Pass --no-merge-enabled to force off "
                         "explicitly even if the class default ever changes.")
    ap.add_argument("--merge-threshold", type=float, default=None, metavar="T",
                    help="cosine similarity above which two global centroids merge, when "
                         "--merge-enabled. Default: speaker_id.MERGE_THRESHOLD.")
    ap.add_argument("--hysteresis-enabled", action=argparse.BooleanOptionalAction, default=None,
                    help="iteration 6 (docs/DIARIZATION_PLAN.md section 9) mitigation B: a "
                         "newly opened global centroid displays under its nearest confirmed "
                         "speaker until it has matched --hysteresis-min-hits times. Applies to "
                         "both --method values. Default: speaker_id.SpeakerLabeler's own default "
                         "(False -- clean on the AMI sweep but rejected after "
                         "testdata/two_speakers.wav showed it can permanently swallow a real "
                         "speaker who only speaks once, see speaker_id.py). Pass "
                         "--hysteresis-enabled to measure the opt-in mitigation anyway.")
    ap.add_argument("--hysteresis-min-hits", type=int, default=None, metavar="N",
                    help="hits required to confirm a provisional speaker, when "
                         "hysteresis is enabled. Default: speaker_id.HYSTERESIS_MIN_HITS.")
    ap.add_argument("--min-duration-on", type=float, default=None, metavar="S",
                    help="diarize.GroupDiarizer's OfflineSpeakerDiarizationConfig."
                         "min_duration_on: drop speech turns shorter than this (seconds) "
                         "after segmentation. Only meaningful with --method refine_diarize. "
                         "Default: diarize.DEFAULT_MIN_DURATION_ON (0.3). See docs/"
                         "DIARIZATION_PLAN.md section 12 (Round 2).")
    ap.add_argument("--min-duration-off", type=float, default=None, metavar="S",
                    help="diarize.GroupDiarizer's OfflineSpeakerDiarizationConfig."
                         "min_duration_off: bridge silence gaps shorter than this (seconds) "
                         "within one local speaker turn. Only meaningful with --method "
                         "refine_diarize. Default: diarize.DEFAULT_MIN_DURATION_OFF (0.5). "
                         "See docs/DIARIZATION_PLAN.md section 12 (Round 2).")
    ap.add_argument("--min-remap-update-s", type=float, default=0.0, metavar="S",
                    help="Round 4 (docs/DIARIZATION_PLAN.md section 14) T2 experiment: a "
                         "refine-group local cluster shorter than this many seconds remaps "
                         "READ-ONLY (can still match an existing global centroid, but never "
                         "folds its embedding into that centroid's mean and never opens a "
                         "brand-new centroid on a miss -- falls back to the group's majority "
                         "fast-path label instead). Only meaningful with --method "
                         "refine_diarize. 0.0 (default) is a no-op, matching every earlier "
                         "round's behavior.")
    ap.add_argument("--joint-remap", action="store_true",
                    help="Round 5 (docs/DIARIZATION_PLAN.md section 15) T1 experiment: within "
                         "one refine group, solve the local-cluster-to-global remap jointly "
                         "(Hungarian assignment maximizing total similarity, scipy.optimize."
                         "linear_sum_assignment) instead of matching each local cluster "
                         "independently, so two distinct local clusters can't both land on the "
                         "same global speaker. Only meaningful with --method refine_diarize. Off "
                         "by default. See speaker_id.SpeakerLabeler.match_embeddings_joint().")
    ap.add_argument("--skip-overlap", action="store_true",
                    help="T2 (docs/DIARIZATION_PLAN.md section 12): exclude reference regions "
                         "with >=2 concurrent speakers from DER scoring (pyannote.metrics "
                         "skip_overlap=True). Only affects --breakdown's der_breakdown; "
                         "the primary simpleder DER is unaffected.")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"no manifest at {args.manifest} -- run scripts/make_diarset.py first",
              file=sys.stderr)
        sys.exit(1)

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.meeting:
        manifest = [e for e in manifest if e["meeting"] == args.meeting]
        if not manifest:
            print(f"meeting {args.meeting!r} not found in {args.manifest}", file=sys.stderr)
            sys.exit(1)

    mdir = os.path.dirname(args.manifest)
    rows = []
    for entry in manifest:
        wav_path = os.path.join(mdir, entry["wav"])
        rttm_path = os.path.join(mdir, entry["rttm"])
        result = score_meeting(wav_path, rttm_path, args.collar, args.min_silence, args.max_speech,
                               method=args.method, diar_threshold=args.diar_threshold,
                               sim_threshold=args.sim_threshold, remap_threshold=args.remap_threshold,
                               merge_enabled=args.merge_enabled, merge_threshold=args.merge_threshold,
                               hysteresis_enabled=args.hysteresis_enabled,
                               hysteresis_min_hits=args.hysteresis_min_hits,
                               min_duration_on=args.min_duration_on,
                               min_duration_off=args.min_duration_off,
                               breakdown=args.breakdown, skip_overlap=args.skip_overlap,
                               vad_threshold=args.vad_threshold,
                               min_remap_update_s=args.min_remap_update_s,
                               joint_remap=args.joint_remap)
        extra = ""
        if "diar_time_s" in result and result.get("audio_s"):
            rtf = result["diar_time_s"] / result["audio_s"]
            extra = (f"  diar={result['diar_time_s']:.1f}s over {result['n_groups']} groups "
                     f"(diar_rtf={rtf:.3f})")
        if args.breakdown:
            extra += (f"  [miss={result['miss'] * 100:.1f}% fa={result['false_alarm'] * 100:.1f}% "
                      f"confusion={result['confusion'] * 100:.1f}%]")
        if result.get("merge_history"):
            extra += f"  merged={result['merge_history']}"
        if result.get("centroid_open_counts"):
            extra += f"  opened_by={result['centroid_open_counts']}"
        print(f"[{entry['meeting']}] DER={result['der'] * 100:.1f}%  "
              f"ref_speakers={result['n_ref_speakers']} hyp_speakers={result['n_hyp_speakers']}  "
              f"ref_turns={result['n_ref_turns']} hyp_segments={result['n_hyp_segments']}  "
              f"decode={result['decode_s']:.1f}s{extra}")
        rows.append({**entry, **result})

    if rows:
        mean_der = sum(r["der"] for r in rows) / len(rows)
        print(f"\n=== mean DER over {len(rows)} meeting(s): {mean_der * 100:.1f}% "
              f"(collar={args.collar}s) ===")

    return rows


if __name__ == "__main__":
    main()
