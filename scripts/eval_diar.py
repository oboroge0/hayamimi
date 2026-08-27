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
# Scoring
# ---------------------------------------------------------------------------

def score_meeting(wav_path: str, rttm_path: str, collar: float,
                  min_silence: float = 0.35, max_speech: float = 12.0) -> dict:
    import simpleder

    ref = segments_to_der_tuples(parse_rttm(rttm_path))
    t0 = time.time()
    hyp_raw = generate_speaker_hypothesis(wav_path, min_silence, max_speech)
    decode_s = time.time() - t0
    hyp = segments_to_der_tuples(hyp_raw)

    der = simpleder.DER(ref, hyp, collar=collar)
    ref_speakers = {s for s, _, _ in ref}
    hyp_speakers = {s for s, _, _ in hyp}
    return {
        "der": der,
        "n_ref_speakers": len(ref_speakers),
        "n_hyp_speakers": len(hyp_speakers),
        "n_ref_turns": len(ref),
        "n_hyp_segments": len(hyp),
        "decode_s": decode_s,
    }


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
        result = score_meeting(wav_path, rttm_path, args.collar, args.min_silence, args.max_speech)
        print(f"[{entry['meeting']}] DER={result['der'] * 100:.1f}%  "
              f"ref_speakers={result['n_ref_speakers']} hyp_speakers={result['n_hyp_speakers']}  "
              f"ref_turns={result['n_ref_turns']} hyp_segments={result['n_hyp_segments']}  "
              f"decode={result['decode_s']:.1f}s")
        rows.append({**entry, **result})

    if rows:
        mean_der = sum(r["der"] for r in rows) / len(rows)
        print(f"\n=== mean DER over {len(rows)} meeting(s): {mean_der * 100:.1f}% "
              f"(collar={args.collar}s) ===")

    return rows


if __name__ == "__main__":
    main()
