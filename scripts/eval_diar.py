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
                                max_speech: float = 12.0) -> list[tuple[str, float, float]]:
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
    vad = build_vad(min_silence, max_speech)
    history = AudioHistory(sr)
    labeler = SpeakerLabeler()

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
                                diar_threshold: float = None) -> tuple[list[tuple[str, float, float]], dict]:
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
    """
    import numpy as np
    from realtime_transcribe import GROUP_GAP_S, GROUP_MAX_S, SAMPLE_RATE, AudioHistory, build_vad, read_wave, wav_chunks
    from speaker_id import SpeakerLabeler
    from diarize import DEFAULT_THRESHOLD, GroupDiarizer

    samples, sr = read_wave(wav_path, target_rate=SAMPLE_RATE)
    vad = build_vad(min_silence, max_speech)
    history = AudioHistory(sr)
    labeler = SpeakerLabeler()  # the single session-global centroid set
    diarizer = GroupDiarizer(threshold=DEFAULT_THRESHOLD if diar_threshold is None else diar_threshold)

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
        for local_id in local_ids:
            pieces = [buf[int(round(s * sr)):int(round(e * sr))]
                      for lid, s, e in turns if lid == local_id]
            cluster_audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            if len(cluster_audio) == 0:
                global_label[local_id] = majority
                continue
            emb = labeler.embed(cluster_audio, sr)
            global_label[local_id] = labeler.match_embedding(emb, update=True)

        for local_id, s, e in turns:
            hyp.append((global_label[local_id], (g_start + s * sr) / sr, (g_start + e * sr) / sr))

    stats = {
        "diar_time_s": diar_time,
        "n_groups": len(groups),
        "audio_s": len(samples) / sr,
    }
    return hyp, stats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_meeting(wav_path: str, rttm_path: str, collar: float,
                  min_silence: float = 0.35, max_speech: float = 12.0,
                  method: str = "baseline", diar_threshold: float = None) -> dict:
    import simpleder

    ref = segments_to_der_tuples(parse_rttm(rttm_path))
    t0 = time.time()
    extra = {}
    if method == "refine_diarize":
        hyp_raw, extra = generate_diarize_hypothesis(wav_path, min_silence, max_speech, diar_threshold)
    else:
        hyp_raw = generate_speaker_hypothesis(wav_path, min_silence, max_speech)
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
    ap.add_argument("--method", choices=["baseline", "refine_diarize"], default="baseline",
                    help="baseline: current --speakers (SpeakerLabeler only, docs/"
                         "DIARIZATION_PLAN.md section 6). refine_diarize: iteration "
                         "3-4's Refiner-group offline diarization + global remap")
    ap.add_argument("--diar-threshold", type=float, default=None, metavar="T",
                    help="FastClusteringConfig.threshold for --method refine_diarize "
                         "(default: diarize.DEFAULT_THRESHOLD)")
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
                               method=args.method, diar_threshold=args.diar_threshold)
        extra = ""
        if "diar_time_s" in result and result.get("audio_s"):
            rtf = result["diar_time_s"] / result["audio_s"]
            extra = (f"  diar={result['diar_time_s']:.1f}s over {result['n_groups']} groups "
                     f"(diar_rtf={rtf:.3f})")
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
