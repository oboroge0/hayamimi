"""Diarization Error Rate (DER) scoring for hayamimi's speaker labeling.

docs/design/diarization.md iteration (2): score the current --speakers
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


DEFAULT_GLOBAL_RECLUSTER_THRESHOLD = 0.65  # Round 7, docs/design/diarization.md section 17
DEFAULT_GLOBAL_RECLUSTER_RELIABLE_S = 1.5  # same "reliable" cutoff Round 6 settled on


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
                                exclude_provisional_remap: bool = False,
                                global_recluster: bool = False,
                                global_recluster_threshold: float = DEFAULT_GLOBAL_RECLUSTER_THRESHOLD,
                                global_recluster_reliable_s: float = DEFAULT_GLOBAL_RECLUSTER_RELIABLE_S,
                                num_clusters_hint: str = "off",
                                num_clusters_hint_min_s: float = 0.0,
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
    and call count, for the RTF measurement docs/design/diarization.md
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
    docs/design/diarization.md section 8 (iteration 5) swept these
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
    Round 2 (docs/design/diarization.md section 12) sweeps these.

    min_remap_update_s (Round 4, docs/design/diarization.md section 14 T2):
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

    joint_remap (Round 5, docs/design/diarization.md section 15 T1): False
    (default) is a no-op -- each group's local clusters (other than any
    excluded by min_remap_update_s above) still remap independently via
    match_embedding(), exactly the pre-existing behavior. When True,
    those clusters are remapped together via
    speaker_id.SpeakerLabeler.match_embeddings_joint() (Hungarian
    assignment maximizing total similarity, constrained so distinct local
    clusters can't land on the same global speaker) -- see that method's
    docstring. Mirrors realtime_transcribe.Refiner._emit_turns() exactly,
    same as every other flag here.

    exclude_provisional_remap (Round 5, docs/design/diarization.md section 15
    T3, only run if T1 fails or leaves IS1009a confusion >8%): False
    (default) is a no-op. When True, every remap match_embedding() call
    below (both the min_remap_update_s read-only probe and the
    joint/independent long-cluster path) passes exclude_provisional=True,
    so a global centroid that hasn't yet been matched a second time can
    never be chosen as a remap target -- see match_embedding()'s
    exclude_provisional docstring.

    global_recluster (Round 7, docs/design/diarization.md section 17): False
    (default) is a no-op -- every local cluster's global label comes
    entirely from the incremental remap above, exactly as every earlier
    round's behavior. When True, this function ALSO keeps every
    remap-eligible local cluster's own embedding (plus its refine-group
    index as a cannot-link key and its speech duration), and once every
    group has been processed, re-clusters all of them at once with
    global_recluster.two_stage_cluster() (the same two-stage design Round 6
    proved out: only clusters with >= global_recluster_reliable_s of
    speech join the agglomerative merge at global_recluster_threshold
    cosine distance, two clusters from the same refine group can never
    merge, short clusters are assigned to the nearest resulting centroid
    afterward). The resulting cluster-id -> "S{n}" mapping REWRITES every
    eligible turn's label in the returned hypothesis; incremental labels
    are only used as this pass's input, not its output. This never changes
    turn timing, never changes which VAD/diarization spans exist, and
    never touches the fast path -- only which S{n} string a refine-group
    turn ends up with.

    num_clusters_hint (Round 9 Experiment A, docs/design/diarization.md
    section 19): "off" (default) is a no-op -- diarizer.process() is
    called with no override, exactly every prior round's behavior
    (FastClustering picks the cluster count itself, diarize.
    DEFAULT_THRESHOLD/-1). "confirmed" passes
    labeler.num_confirmed_speakers() (the session's confirmed-so-far
    global speaker count, as of just BEFORE this group's own remap --
    i.e. every earlier group's remap has already run) as this group's
    diarize.GroupDiarizer.process(num_clusters=) hint, but only when the
    hint is >= 1 (a session with zero confirmed speakers yet -- e.g. the
    very first group -- falls back to "off" for that one call; there is
    no informative hint to give). "confirmed-capped" additionally clamps
    that hint to min(confirmed, len(group)) -- len(group) (the number of
    fast-path VAD segments this refine group accumulated, known BEFORE
    diarization runs) is a cheap upper bound on how many distinct local
    turns pyannote segmentation could plausibly split this group into;
    this guards against forcing a hint larger than the group could ever
    contain. num_clusters_hint_min_s (default 0.0, i.e. no gate) is the
    "soft variant" the task asked for: the hint is only applied when this
    group's total VAD-detected speech duration (sum of (end-start) over
    its fast-path segments, in seconds -- NOT len(buf)/sr, which also
    counts preroll and internal silence gaps) is >= this many seconds.
    Below it, the group falls back to "off" for that call. This exists
    because a short group is unlikely to contain every session speaker
    yet, so forcing FastClustering toward the session's full confirmed
    count on a short group risks forcing a WRONG count -- the harmful
    failure mode Round 5 (section 15) already demonstrated for a related
    knob (see that section's caution against "forcing a wrong count").
    """
    import numpy as np
    from realtime_transcribe import GROUP_GAP_S, GROUP_MAX_S, SAMPLE_RATE, AudioHistory, build_vad, read_wave, wav_chunks
    from speaker_id import SIM_THRESHOLD, SpeakerLabeler
    from diarize import DEFAULT_MIN_DURATION_OFF, DEFAULT_MIN_DURATION_ON, DEFAULT_THRESHOLD, GroupDiarizer
    from global_recluster import pool_audio_for_group

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
    # Round 7 (section 17) global_recluster bookkeeping: hyp_keys[i] names
    # which (group_idx, local_id) hyp[i] came from, when that turn's label
    # is eligible to be rewritten by the post-hoc re-cluster below (None for
    # a turn that never had its own embedding -- the single-speaker-group
    # fallback, or a local cluster with no audio -- there is nothing to
    # re-cluster for those, so their incremental label is final either way).
    # recluster_entries accumulates one row per DISTINCT (group_idx,
    # local_id) that got a real embedding, across the whole session/eval --
    # exactly the "retain per-group local-cluster embeddings as the session
    # runs" T1 asks for. Built unconditionally (cheap: one array append per
    # local cluster) so turning global_recluster on/off doesn't change
    # anything about hypothesis generation up to this point -- only whether
    # the accumulated rows are ever clustered and used to rewrite hyp.
    hyp_keys: list[tuple[int, int] | None] = []
    recluster_entries: list[dict] = []
    diar_time = 0.0
    for group_idx, group in enumerate(groups):
        g_start, g_end = group[0][0], group[-1][1]
        buf = samples[g_start:g_end]
        fast_labels = [lbl for _, _, lbl in group]
        majority = max(set(fast_labels), key=fast_labels.count) if fast_labels else ""

        hint = None
        if num_clusters_hint != "off":
            # group entries are (seg_start, seg_end, label) sample indices;
            # sum of per-segment durations (not g_end-g_start, which also
            # counts internal silence gaps) is this group's actual
            # VAD-detected speech time -- see the docstring above.
            group_speech_s = sum(e - s for s, e, _ in group) / sr
            if group_speech_s >= num_clusters_hint_min_s:
                confirmed = labeler.num_confirmed_speakers()
                if confirmed >= 1:
                    hint = confirmed
                    if num_clusters_hint == "confirmed-capped":
                        hint = min(hint, len(group))

        t0 = time.time()
        try:
            raw = diarizer.process(buf, sr, num_clusters=hint)
        except Exception:
            raw = []
        diar_time += time.time() - t0

        turns = [(lid, s, e) for lid, s, e in raw if e - s >= 0.3]
        if len({lid for lid, _, _ in turns}) < 2:
            # single speaker (or diarizer declined): same fallback
            # Refiner._emit_turns() takes -- one majority-vote span.
            #
            # Round 8 (section 18) T1: this group still contributes a
            # group-level embedding to the re-cluster pool (local_id=-1
            # sentinel -- there is no real per-cluster id to key on) when
            # global_recluster is on, so its turn's label is rewritten
            # along with every other turn in the session instead of
            # permanently keeping its fast-path label. That fast-path/
            # untouched split (a group judged single-speaker never entered
            # the pool at all) is what caused Round 7's rejection -- see
            # global_recluster.py's module docstring.
            hyp.append((majority, g_start / sr, g_end / sr))
            if global_recluster:
                sample_turns = [(lid, int(round(s * sr)), int(round(e * sr)))
                                for lid, s, e in turns]
                pool_audio, pool_dur_s = pool_audio_for_group(
                    buf, sr, sample_turns, g_start, g_end)
                if len(pool_audio) > 0:
                    recluster_entries.append({
                        "group_idx": group_idx, "local_id": -1,
                        "embedding": labeler.embed(pool_audio, sr), "duration_s": pool_dur_s,
                    })
                    hyp_keys.append((group_idx, -1))
                else:
                    hyp_keys.append(None)
            else:
                hyp_keys.append(None)
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
            if global_recluster:
                cluster_dur_s = sum(e - s for lid, s, e in turns if lid == local_id)
                recluster_entries.append({
                    "group_idx": group_idx, "local_id": local_id,
                    "embedding": cluster_embs[local_id], "duration_s": cluster_dur_s,
                })

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
                    source="remap", exclude_provisional=exclude_provisional_remap)
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
                        source="remap", exclude_provisional=exclude_provisional_remap)

        # iteration 6 (section 9): this group's remap is exactly the "clean
        # copy" boundary merge_centroids() is meant for -- give it a chance
        # to fold any two global centroids that have drifted together
        # before the next group opens new ones against them.
        labeler.maybe_merge_centroids()

        for local_id, s, e in turns:
            hyp.append((global_label[local_id], (g_start + s * sr) / sr, (g_start + e * sr) / sr))
            hyp_keys.append((group_idx, local_id) if local_id in cluster_embs else None)

    global_recluster_time_s = 0.0
    n_recluster_entries = len(recluster_entries)
    n_recluster_clusters = 0
    if global_recluster and recluster_entries:
        from global_recluster import two_stage_cluster

        t0 = time.time()
        embeddings = np.stack([e["embedding"] for e in recluster_entries])
        group_ids = [e["group_idx"] for e in recluster_entries]
        durations = [e["duration_s"] for e in recluster_entries]
        labels = two_stage_cluster(embeddings, group_ids, durations,
                                   threshold=global_recluster_threshold,
                                   reliable_s=global_recluster_reliable_s)
        global_recluster_time_s = time.time() - t0
        n_recluster_clusters = int(labels.max()) + 1 if len(labels) else 0

        key_to_label: dict[tuple[int, int], str] = {
            (entry["group_idx"], entry["local_id"]): f"S{int(cluster_id) + 1}"
            for entry, cluster_id in zip(recluster_entries, labels)
        }
        hyp = [
            (key_to_label[key] if key is not None and key in key_to_label else label, s, e)
            for (label, s, e), key in zip(hyp, hyp_keys)
        ]

    stats = {
        "diar_time_s": diar_time,
        "n_groups": len(groups),
        "audio_s": len(samples) / sr,
        "merge_history": labeler.merge_history(),
        "global_recluster_time_s": global_recluster_time_s,
        "n_recluster_entries": n_recluster_entries,
        "n_recluster_clusters": n_recluster_clusters,
        # docs/design/diarization.md section 10.6 diagnostics: compare against
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
    call (docs/design/diarization.md section 4's two-tier plan: simpleder for
    the fast day-to-day sweep loop, pyannote.metrics only when the error
    breakdown itself is needed -- iteration 5, to tell whether the
    remaining over-splitting after the section-8 threshold tuning is
    boundary noise (Miss/FA) or genuine speaker confusion).

    skip_overlap (T2, docs/design/diarization.md section 12): pass
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
                  joint_remap: bool = False, exclude_provisional_remap: bool = False,
                  global_recluster: bool = False,
                  global_recluster_threshold: float = DEFAULT_GLOBAL_RECLUSTER_THRESHOLD,
                  num_clusters_hint: str = "off",
                  num_clusters_hint_min_s: float = 0.0) -> dict:
    import simpleder

    ref = segments_to_der_tuples(parse_rttm(rttm_path))
    t0 = time.time()
    extra = {}
    if method == "refine_diarize":
        hyp_raw, extra = generate_diarize_hypothesis(
            wav_path, min_silence, max_speech, diar_threshold, sim_threshold, remap_threshold,
            merge_enabled, merge_threshold, hysteresis_enabled, hysteresis_min_hits,
            min_duration_on, min_duration_off, vad_threshold, min_remap_update_s, joint_remap,
            exclude_provisional_remap, global_recluster, global_recluster_threshold,
            DEFAULT_GLOBAL_RECLUSTER_RELIABLE_S, num_clusters_hint, num_clusters_hint_min_s)
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
                         "cost of more false alarms. docs/design/diarization.md section 13 "
                         "(Round 3). The installed sherpa_onnx has no speech-padding knob to "
                         "pair with this (checked: SileroVadModelConfig fields are threshold, "
                         "min_silence_duration, min_speech_duration, max_speech_duration, "
                         "window_size, neg_threshold -- no speech_pad_ms), so there is no "
                         "--vad-pad flag.")
    ap.add_argument("--method", choices=["baseline", "refine_diarize"], default="baseline",
                    help="baseline: current --speakers (SpeakerLabeler only, docs/design/"
                         "diarization.md section 6). refine_diarize: iteration "
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
                         "See docs/design/diarization.md section 8.")
    ap.add_argument("--breakdown", action="store_true",
                    help="also report Miss/False-Alarm/Confusion via pyannote.metrics "
                         "(requirements-dev.txt). Slower and heavier than the default "
                         "simpleder-only scoring -- see docs/design/diarization.md section 4.")
    ap.add_argument("--merge-enabled", action=argparse.BooleanOptionalAction, default=None,
                    help="iteration 6 (docs/design/diarization.md section 9) mitigation A: "
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
                    help="iteration 6 (docs/design/diarization.md section 9) mitigation B: a "
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
                         "Default: diarize.DEFAULT_MIN_DURATION_ON (0.3). See docs/design/"
                         "diarization.md section 12 (Round 2).")
    ap.add_argument("--min-duration-off", type=float, default=None, metavar="S",
                    help="diarize.GroupDiarizer's OfflineSpeakerDiarizationConfig."
                         "min_duration_off: bridge silence gaps shorter than this (seconds) "
                         "within one local speaker turn. Only meaningful with --method "
                         "refine_diarize. Default: diarize.DEFAULT_MIN_DURATION_OFF (0.5). "
                         "See docs/design/diarization.md section 12 (Round 2).")
    ap.add_argument("--min-remap-update-s", type=float, default=0.0, metavar="S",
                    help="Round 4 (docs/design/diarization.md section 14) T2 experiment: a "
                         "refine-group local cluster shorter than this many seconds remaps "
                         "READ-ONLY (can still match an existing global centroid, but never "
                         "folds its embedding into that centroid's mean and never opens a "
                         "brand-new centroid on a miss -- falls back to the group's majority "
                         "fast-path label instead). Only meaningful with --method "
                         "refine_diarize. 0.0 (default) is a no-op, matching every earlier "
                         "round's behavior.")
    ap.add_argument("--joint-remap", action="store_true",
                    help="Round 5 (docs/design/diarization.md section 15) T1 experiment: within "
                         "one refine group, solve the local-cluster-to-global remap jointly "
                         "(Hungarian assignment maximizing total similarity, scipy.optimize."
                         "linear_sum_assignment) instead of matching each local cluster "
                         "independently, so two distinct local clusters can't both land on the "
                         "same global speaker. Only meaningful with --method refine_diarize. Off "
                         "by default. See speaker_id.SpeakerLabeler.match_embeddings_joint().")
    ap.add_argument("--exclude-provisional-remap", action="store_true",
                    help="Round 5 (docs/design/diarization.md section 15) T3 experiment: a global "
                         "centroid that hasn't yet been matched a second time (still provisional, "
                         "see speaker_id.PROVISIONAL_CONFIRM_HITS) is never chosen as a remap "
                         "target -- it may be 'stealing' a match that should have gone to a real, "
                         "already-recurring speaker. Only meaningful with --method refine_diarize. "
                         "Off by default. See speaker_id.SpeakerLabeler.match_embedding()'s "
                         "exclude_provisional docstring.")
    ap.add_argument("--global-recluster", action="store_true",
                    help="Round 7 (docs/design/diarization.md section 17) T1 experiment: at "
                         "session end, re-cluster every refine group's local-cluster "
                         "embeddings together (two-stage constrained agglomerative, same "
                         "design as eval_diar_overlap.py's Round 6 prototype, via "
                         "global_recluster.two_stage_cluster()) and rewrite the hypothesis's "
                         "labels with the result, instead of the incremental per-group remap. "
                         "Only meaningful with --method refine_diarize. Off by default.")
    ap.add_argument("--global-recluster-threshold", type=float,
                    default=DEFAULT_GLOBAL_RECLUSTER_THRESHOLD, metavar="T",
                    help=f"cosine-distance threshold for --global-recluster's agglomerative "
                         f"merge stage (default {DEFAULT_GLOBAL_RECLUSTER_THRESHOLD}, the "
                         f"value Round 6 found best on this same AMI subset).")
    ap.add_argument("--skip-overlap", action="store_true",
                    help="T2 (docs/design/diarization.md section 12): exclude reference regions "
                         "with >=2 concurrent speakers from DER scoring (pyannote.metrics "
                         "skip_overlap=True). Only affects --breakdown's der_breakdown; "
                         "the primary simpleder DER is unaffected.")
    ap.add_argument("--num-clusters-hint", choices=["off", "confirmed", "confirmed-capped"],
                    default="off",
                    help="Round 9 (docs/design/diarization.md section 19) Experiment A: pass a "
                         "num_clusters hint to diarize.GroupDiarizer's FastClustering, derived "
                         "from speaker_id.SpeakerLabeler.num_confirmed_speakers() at the moment "
                         "each refine group closes. 'off' (default) is a no-op. 'confirmed' uses "
                         "the confirmed count directly; 'confirmed-capped' additionally clamps it "
                         "to min(confirmed, len(group)) (the group's own fast-path VAD segment "
                         "count). Only meaningful with --method refine_diarize. See "
                         "generate_diarize_hypothesis()'s num_clusters_hint docstring.")
    ap.add_argument("--num-clusters-hint-min-s", type=float, default=0.0, metavar="S",
                    help="Only apply --num-clusters-hint when this refine group's total "
                         "VAD-detected speech duration (seconds) is >= this value; below it, "
                         "the group falls back to no hint. 0.0 (default) applies the hint to "
                         "every group. Guards against forcing a confirmed-speaker-count hint "
                         "onto a short group that plausibly doesn't contain every speaker yet.")
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
                               joint_remap=args.joint_remap,
                               exclude_provisional_remap=args.exclude_provisional_remap,
                               global_recluster=args.global_recluster,
                               global_recluster_threshold=args.global_recluster_threshold,
                               num_clusters_hint=args.num_clusters_hint,
                               num_clusters_hint_min_s=args.num_clusters_hint_min_s)
        extra = ""
        if "diar_time_s" in result and result.get("audio_s"):
            rtf = result["diar_time_s"] / result["audio_s"]
            extra = (f"  diar={result['diar_time_s']:.1f}s over {result['n_groups']} groups "
                     f"(diar_rtf={rtf:.3f})")
        if args.global_recluster and "global_recluster_time_s" in result:
            extra += (f"  recluster={result['global_recluster_time_s']:.2f}s "
                      f"({result['n_recluster_entries']} entries -> "
                      f"{result['n_recluster_clusters']} clusters)")
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
