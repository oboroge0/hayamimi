"""Overlap-aware diarization PROTOTYPE + DER scoring (Round 6, eval-only).

docs/DIARIZATION_PLAN.md section 16. This is a measurement tool, not a
production path: nothing in realtime_transcribe.py / diarize.py calls it.

Why it exists
-------------
Every DER we have measured so far (sections 6-15) is bounded below by a
structural floor: hayamimi's hypothesis emits exactly one speaker per time
span, so every second of *overlapped* reference speech is a guaranteed
miss. On the AMI subset that floor is ~3.1pt of the 13.9% baseline.

Round 4 (section 14) established that sherpa-onnx's OfflineSpeakerDiarization
CANNOT emit overlap: it runs pyannote segmentation-3.0, whose powerset head
can represent up to 2 simultaneous speakers, but the C++ ExcludeOverlap()
discards those frames before anything is exposed to Python. The verdict was
"possible with work: run the segmentation ONNX ourselves".

This script does exactly that, from scratch (it is NOT the production
group/remap pipeline with a patch -- it is a whole-file diarizer):

  1. sliding 10s windows over the meeting, pyannote segmentation-3.0 run
     directly through onnxruntime (models/sherpa-onnx-pyannote-segmentation-3-0);
  2. powerset argmax -> per-frame multi-label activity (up to 2 of 3
     window-local speakers), overlap frames KEPT;
  3. one CAM++ embedding (scripts/speaker_id.SpeakerLabeler.embed, the same
     extractor production uses) per window-local speaker, computed over that
     speaker's *exclusive* (non-overlapped) frames so the embedding is not
     contaminated by the other talker;
  4. constrained agglomerative clustering (scipy, cosine, average linkage;
     two speakers found in the SAME window are forbidden from merging) to
     stitch window-local speakers into meeting-global speakers;
  5. per-frame majority vote across the overlapping windows -> meeting-level
     activity per global speaker -> segments, emitted WITH overlap;
  6. scored with pyannote.metrics DiarizationErrorRate at the same collar
     eval_diar.py uses, overlap INCLUDED, against the same AMI references.

Every hypothesis is scored twice, and both DERs are printed as
`pyannote/simpleder`: eval_diar.py's headline number is simpleder's while its
`--breakdown` miss/fa/confusion come from pyannote.metrics, and the two
metrics normalize differently enough to disagree by several points on the
very same hypothesis.

The same hypothesis is also re-scored the very same hypothesis with overlaps removed
(at each instant only the locally-dominant speaker survives). Comparing the
two isolates how much of any DER change comes from emitting overlap versus
from this script's different (full-file, offline) clustering.

Usage:
    python scripts/eval_diar_overlap.py                          # all meetings, default threshold
    python scripts/eval_diar_overlap.py --meeting ES2004a
    python scripts/eval_diar_overlap.py --thresholds 0.4,0.5,0.6 # sweep clustering
    python scripts/eval_diar_overlap.py --num-clusters 4         # oracle speaker count
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

import numpy as np  # noqa: E402

from eval_diar import (  # noqa: E402
    DEFAULT_COLLAR,
    MANIFEST_PATH,
    der_breakdown,
    parse_rttm,
    segments_to_der_tuples,
)
from global_recluster import (  # noqa: E402
    assign_by_centroid,
    cluster_reliable as cluster_local_speakers,
)

SEG_MODEL = os.path.join(ROOT, "models", "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")

# ---------------------------------------------------------------------------
# pyannote segmentation-3.0 constants. All of these are read back from the
# ONNX file's own metadata in load_segmentation() and asserted, rather than
# trusted blindly -- if k2-fsa ever re-exports the model with a different
# receptive field this script fails loudly instead of silently mis-timing
# every frame.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 160000       # 10s, the duration the model was trained on
FRAME_SHIFT = 270             # samples between consecutive output frames
NUM_CLASSES = 7
# pyannote's powerset encoding for (num_speakers=3, max_simultaneous=2), in
# the exact order the training-time Powerset module enumerates them:
# the empty set first, then all singletons, then all pairs.
POWERSET = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]

DEFAULT_HOP_S = 5.0           # 50% window overlap
DEFAULT_CLUSTER_THRESHOLD = 0.5   # cosine distance, scipy average linkage
DEFAULT_MIN_DURATION_ON = 0.3     # drop shorter emitted turns
DEFAULT_MIN_DURATION_OFF = 0.3    # bridge shorter within-speaker gaps
MIN_EMBED_S = 0.5             # below this much *exclusive* speech, a
                              # window-local speaker's embedding falls back to
                              # its overlap-contaminated full activity
DEFAULT_RELIABLE_S = 1.5      # exclusive speech a window-local speaker needs
                              # before its embedding is allowed to take part in
                              # the clustering merge (rather than merely being
                              # assigned to the nearest resulting centroid)


def load_segmentation(num_threads: int = 4):
    """Open the segmentation ONNX and verify its geometry matches our constants."""
    import onnxruntime

    if not os.path.exists(SEG_MODEL):
        raise FileNotFoundError(
            f"pyannote segmentation model missing: {SEG_MODEL} -- "
            "see scripts/diarize.py's download hint"
        )
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = num_threads
    sess = onnxruntime.InferenceSession(SEG_MODEL, opts,
                                        providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    assert int(meta["sample_rate"]) == SAMPLE_RATE, meta
    assert int(meta["window_size"]) == WINDOW_SAMPLES, meta
    assert int(meta["receptive_field_shift"]) == FRAME_SHIFT, meta
    assert int(meta["num_classes"]) == NUM_CLASSES, meta
    assert int(meta["num_speakers"]) == 3, meta
    assert int(meta["powerset_max_classes"]) == 2, meta
    return sess


def powerset_decode(logits: np.ndarray) -> np.ndarray:
    """(frames, 7) powerset log-posteriors -> (frames, 3) binary activity.

    The model's head is a log-softmax over the 7 powerset classes, so a plain
    argmax picks the single most likely *set* of simultaneously active local
    speakers. That set may contain two speakers -- which is precisely the
    information sherpa-onnx's ExcludeOverlap() throws away (section 14).
    """
    best = np.argmax(logits, axis=-1)
    out = np.zeros((logits.shape[0], 3), dtype=bool)
    for cls, speakers in enumerate(POWERSET):
        if not speakers:
            continue
        rows = best == cls
        for spk in speakers:
            out[rows, spk] = True
    return out


def segment_windows(samples: np.ndarray, sess, hop_samples: int, batch: int = 8):
    """Run the segmentation model over sliding windows.

    Yields (window_start_sample, activity) where activity is (frames, 3) bool.
    hop_samples must be a multiple of FRAME_SHIFT so window frame index i maps
    onto the meeting-global frame grid by a pure integer offset (no resampling
    or interpolation when the per-frame votes from overlapping windows are
    combined later).
    """
    assert hop_samples % FRAME_SHIFT == 0, hop_samples
    starts = list(range(0, max(len(samples) - WINDOW_SAMPLES, 0) + 1, hop_samples))
    if not starts:
        starts = [0]
    # cover the tail: a final zero-padded window anchored at the last hop
    tail = ((len(samples) - WINDOW_SAMPLES) // hop_samples + 1) * hop_samples
    if tail > starts[-1] and tail < len(samples):
        starts.append(tail)

    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        buf = np.zeros((len(chunk), 1, WINDOW_SAMPLES), dtype=np.float32)
        for j, start in enumerate(chunk):
            piece = samples[start:start + WINDOW_SAMPLES]
            buf[j, 0, :len(piece)] = piece
        logits = sess.run(None, {"x": buf})[0]
        for j, start in enumerate(chunk):
            yield start, powerset_decode(logits[j])


def window_speaker_audio(samples: np.ndarray, win_start: int, activity: np.ndarray,
                         spk: int, exclusive_only: bool) -> np.ndarray:
    """Concatenate the audio of one window-local speaker's active frames.

    exclusive_only=True keeps only frames where this speaker is the ONLY
    active one. That is what we embed with: a CAM++ embedding computed over
    frames where two people talk at once is a mixture, and mixtures cluster
    with neither speaker (this is the single biggest accuracy risk in the
    whole approach -- see section 16's assessment).
    """
    active = activity[:, spk]
    if exclusive_only:
        active = active & (activity.sum(axis=1) == 1)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return np.zeros(0, dtype=np.float32)
    pieces = []
    for i in idx:
        a = win_start + int(i) * FRAME_SHIFT
        pieces.append(samples[a:a + FRAME_SHIFT])
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def activity_to_segments(active: np.ndarray, frame_s: float,
                         min_on: float, min_off: float) -> list[tuple[float, float]]:
    """Boolean per-frame activity -> (start_s, end_s) spans, with the same two
    post-processing knobs sherpa-onnx's OfflineSpeakerDiarizationConfig has
    (min_duration_off bridges short internal gaps, then min_duration_on drops
    short surviving turns)."""
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    spans = [(float(s) * frame_s, float(e + 1) * frame_s) for s, e in zip(starts, ends)]

    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s - merged[-1][1] < min_off:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_on]


def strip_overlap(hyp: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Collapse a multi-label hypothesis to one speaker per instant.

    At every boundary-delimited slice where several speakers are active, the
    survivor is the one whose own containing segment is longest (ties broken
    by total speech in the meeting) -- i.e. the locally dominant talker, the
    same thing a single-label diarizer would have been forced to pick.
    """
    if not hyp:
        return []
    total: dict[str, float] = {}
    for spk, s, e in hyp:
        total[spk] = total.get(spk, 0.0) + (e - s)
    bounds = sorted({t for _, s, e in hyp for t in (s, e)})
    out: list[tuple[str, float, float]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        here = [(e - s, total[spk], spk) for spk, s, e in hyp if s <= mid < e]
        if not here:
            continue
        _, _, spk = max(here)
        if out and out[-1][0] == spk and abs(out[-1][2] - a) < 1e-9:
            out[-1] = (spk, out[-1][1], b)
        else:
            out.append((spk, a, b))
    return out


def overlap_fraction(ref: list[tuple[str, float, float]]) -> float:
    """The structural overlap floor: the fraction of reference speech time a
    single-label hypothesis is GUARANTEED to miss.

    A region where n>=2 speakers talk at once contributes (n-1)*duration of
    unavoidable missed detection to a hypothesis that can only name one
    speaker per instant. The denominator is total reference speech counting
    each speaker separately (sum of turn durations), which is exactly how
    pyannote.metrics normalizes miss/false-alarm/confusion -- so this number
    is directly comparable to the miss rates printed alongside it.
    """
    bounds = sorted({t for _, s, e in ref for t in (s, e)})
    total = sum(e - s for _, s, e in ref)
    ov = 0.0
    for a, b in zip(bounds, bounds[1:]):
        mid = (a + b) / 2
        n = sum(1 for _, s, e in ref if s <= mid < e)
        if n >= 2:
            ov += (b - a) * (n - 1)
    return ov / total if total else 0.0


def diarize_overlap(wav_path: str, hop_s: float = DEFAULT_HOP_S,
                    thresholds: list[float] = (DEFAULT_CLUSTER_THRESHOLD,),
                    num_clusters: int | None = None,
                    min_on: float = DEFAULT_MIN_DURATION_ON,
                    min_off: float = DEFAULT_MIN_DURATION_OFF,
                    threads: int = 4, method: str = "average",
                    reliable_s: float = DEFAULT_RELIABLE_S) -> tuple[dict[float, list], dict]:
    """Full-file overlap-aware diarization. Returns ({threshold: hyp}, stats).

    Segmentation and embedding are done once and shared across every
    clustering threshold -- only step 4-5 (cluster + aggregate) is repeated
    per threshold, which is what makes the sweep cheap.
    """
    from realtime_transcribe import read_wave
    from speaker_id import SpeakerLabeler

    samples, sr = read_wave(wav_path, target_rate=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    frame_s = FRAME_SHIFT / SAMPLE_RATE
    hop_frames = max(1, int(round(hop_s * SAMPLE_RATE / FRAME_SHIFT)))
    hop_samples = hop_frames * FRAME_SHIFT

    sess = load_segmentation(threads)
    labeler = SpeakerLabeler(threads=threads)

    t_seg = 0.0
    t_emb = 0.0
    windows: list[tuple[int, np.ndarray]] = []  # (frame offset, activity)
    embeddings: list[np.ndarray] = []
    window_ids: list[int] = []
    entries: list[tuple[int, int]] = []  # (window index, local speaker)
    exclusive_durations: list[float] = []

    t0 = time.time()
    seg_out = list(segment_windows(samples, sess, hop_samples))
    t_seg += time.time() - t0

    n_exclusive_short = 0
    for w, (win_start, activity) in enumerate(seg_out):
        windows.append((win_start // FRAME_SHIFT, activity))
        for spk in range(3):
            if not activity[:, spk].any():
                continue
            t0 = time.time()
            audio = window_speaker_audio(samples, win_start, activity, spk,
                                         exclusive_only=True)
            exclusive_s = len(audio) / SAMPLE_RATE
            if len(audio) < MIN_EMBED_S * SAMPLE_RATE:
                # not enough clean speech: fall back to this speaker's full
                # activity (overlap-contaminated, but better than dropping a
                # speaker who is genuinely present in this window)
                n_exclusive_short += 1
                audio = window_speaker_audio(samples, win_start, activity, spk,
                                             exclusive_only=False)
            if len(audio) < 0.2 * SAMPLE_RATE:
                t_emb += time.time() - t0
                continue
            emb = labeler.embed(audio, SAMPLE_RATE)
            t_emb += time.time() - t0
            embeddings.append(emb)
            window_ids.append(w)
            entries.append((w, spk))
            exclusive_durations.append(exclusive_s)

    emb_arr = np.asarray(embeddings, dtype=np.float32) if embeddings else np.zeros((0, 1),
                                                                                  dtype=np.float32)
    n_frames_total = int(np.ceil(len(samples) / FRAME_SHIFT))

    results: dict[float, list] = {}
    t_cluster = 0.0
    for threshold in thresholds:
        t0 = time.time()
        reliable = np.asarray(exclusive_durations) >= reliable_s if len(emb_arr) else \
            np.zeros(0, dtype=bool)
        if reliable.sum() < 2:  # nothing trustworthy to cluster: use everything
            reliable = np.ones(len(emb_arr), dtype=bool)
        labels_reliable = cluster_local_speakers(
            emb_arr[reliable], [w for w, keep in zip(window_ids, reliable) if keep],
            threshold, num_clusters, method)
        labels = assign_by_centroid(emb_arr, reliable, labels_reliable)
        n_global = int(labels.max()) + 1 if len(labels) else 0

        # per-frame vote accumulation across the overlapping windows
        votes = np.zeros((n_global, n_frames_total), dtype=np.int16)
        cover = np.zeros(n_frames_total, dtype=np.int16)
        assign: dict[tuple[int, int], int] = {e: int(g) for e, g in zip(entries, labels)}
        for w, (frame_off, activity) in enumerate(windows):
            nf = activity.shape[0]
            hi = min(frame_off + nf, n_frames_total)
            if hi <= frame_off:
                continue
            span = hi - frame_off
            cover[frame_off:hi] += 1
            for spk in range(3):
                g = assign.get((w, spk))
                if g is None:
                    continue
                votes[g, frame_off:hi] += activity[:span, spk]

        cover_safe = np.maximum(cover, 1)
        hyp: list[tuple[str, float, float]] = []
        for g in range(n_global):
            active = votes[g] * 2 >= cover_safe  # majority vote over covering windows
            active &= cover > 0
            for s, e in activity_to_segments(active, frame_s, min_on, min_off):
                hyp.append((f"P{g}", s, e))
        t_cluster += time.time() - t0
        results[threshold] = hyp

    stats = {
        "reliable_s": reliable_s,
        "n_reliable": int(np.asarray(exclusive_durations).__ge__(reliable_s).sum())
        if exclusive_durations else 0,
        "audio_s": len(samples) / SAMPLE_RATE,
        "n_windows": len(windows),
        "n_local_speakers": len(entries),
        "n_fallback_contaminated_embeddings": n_exclusive_short,
        "seg_time_s": t_seg,
        "embed_time_s": t_emb,
        "cluster_time_s": t_cluster,
    }
    return results, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--meeting", default=None)
    ap.add_argument("--collar", type=float, default=DEFAULT_COLLAR)
    ap.add_argument("--hop-s", type=float, default=DEFAULT_HOP_S,
                    help="sliding-window hop in seconds (window is fixed at 10s by the "
                         "model). Rounded to a whole number of 0.016875s output frames.")
    ap.add_argument("--thresholds", default=str(DEFAULT_CLUSTER_THRESHOLD),
                    help="comma-separated cosine-distance thresholds for the "
                         "cross-window agglomerative clustering; segmentation and "
                         "embedding are computed once and shared across all of them")
    ap.add_argument("--num-clusters", type=int, default=None,
                    help="force this many global speakers (oracle diagnostic); "
                         "overrides --thresholds' cut criterion")
    ap.add_argument("--min-duration-on", type=float, default=DEFAULT_MIN_DURATION_ON)
    ap.add_argument("--min-duration-off", type=float, default=DEFAULT_MIN_DURATION_OFF)
    ap.add_argument("--reliable-s", type=float, default=DEFAULT_RELIABLE_S,
                    help="exclusive speech a window-local speaker needs before its "
                         "embedding joins the clustering merge; shorter ones are assigned "
                         "to the nearest resulting centroid instead")
    ap.add_argument("--linkage", default="average", choices=["average", "complete", "single"],
                    help="scipy linkage method for the cross-window clustering")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json-out", default=None, help="write per-meeting rows here")
    args = ap.parse_args()

    import simpleder

    thresholds = [float(t) for t in args.thresholds.split(",")]

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    if args.meeting:
        manifest = [e for e in manifest if e["meeting"] == args.meeting]
        if not manifest:
            print(f"meeting {args.meeting!r} not found", file=sys.stderr)
            sys.exit(1)
    mdir = os.path.dirname(args.manifest)

    rows = []
    for entry in manifest:
        wav_path = os.path.join(mdir, entry["wav"])
        ref = segments_to_der_tuples(parse_rttm(os.path.join(mdir, entry["rttm"])))
        ov = overlap_fraction(ref)

        t0 = time.time()
        hyps, stats = diarize_overlap(wav_path, args.hop_s, thresholds, args.num_clusters,
                                      args.min_duration_on, args.min_duration_off,
                                      args.threads, args.linkage, args.reliable_s)
        wall = time.time() - t0

        for threshold in thresholds:
            hyp = segments_to_der_tuples(hyps[threshold])
            stripped = segments_to_der_tuples(strip_overlap(hyps[threshold]))
            on = der_breakdown(ref, hyp, args.collar)
            off = der_breakdown(ref, stripped, args.collar)
            # eval_diar.py's headline DER is simpleder's, not pyannote.metrics'
            # -- the two normalize differently (simpleder over the reference
            # speech *union*, pyannote over the sum of per-speaker turn
            # durations), so they disagree by several points on the same
            # hypothesis. Report both so this prototype can be compared
            # against the historical section 6-15 numbers on either metric
            # without mixing them up.
            on["der_simple"] = simpleder.DER(ref, hyp, collar=args.collar)
            off["der_simple"] = simpleder.DER(ref, stripped, collar=args.collar)
            n_spk = len({s for s, _, _ in hyp})
            row = {
                **entry, "threshold": threshold, "overlap_fraction": ov,
                "n_hyp_speakers": n_spk, "wall_s": wall, **stats,
                "overlap_on": on, "overlap_stripped": off,
            }
            rows.append(row)
            print(f"[{entry['meeting']}] t={threshold:.2f} ref_ov={ov * 100:.1f}%  "
                  f"hyp_speakers={n_spk}  "
                  f"OVERLAP-ON  DER={on['der_breakdown'] * 100:.1f}%/"
                  f"{on['der_simple'] * 100:.1f}% "
                  f"(miss={on['miss'] * 100:.1f} fa={on['false_alarm'] * 100:.1f} "
                  f"conf={on['confusion'] * 100:.1f})   "
                  f"STRIPPED  DER={off['der_breakdown'] * 100:.1f}%/"
                  f"{off['der_simple'] * 100:.1f}% "
                  f"(miss={off['miss'] * 100:.1f} fa={off['false_alarm'] * 100:.1f} "
                  f"conf={off['confusion'] * 100:.1f})")
        print(f"    wall={wall:.1f}s over {stats['audio_s']:.0f}s audio "
              f"(rtf={wall / stats['audio_s']:.3f}; seg={stats['seg_time_s']:.1f}s "
              f"embed={stats['embed_time_s']:.1f}s cluster={stats['cluster_time_s']:.1f}s) "
              f"windows={stats['n_windows']} local_speakers={stats['n_local_speakers']} "
              f"contaminated_embeds={stats['n_fallback_contaminated_embeddings']} "
              f"reliable={stats['n_reliable']}")

    for threshold in thresholds:
        sel = [r for r in rows if r["threshold"] == threshold]
        if not sel:
            continue
        mon = sum(r["overlap_on"]["der_breakdown"] for r in sel) / len(sel)
        moff = sum(r["overlap_stripped"]["der_breakdown"] for r in sel) / len(sel)
        son = sum(r["overlap_on"]["der_simple"] for r in sel) / len(sel)
        soff = sum(r["overlap_stripped"]["der_simple"] for r in sel) / len(sel)
        print(f"\n=== t={threshold:.2f} mean DER (pyannote/simpleder) over {len(sel)} "
              f"meeting(s): overlap-on {mon * 100:.1f}%/{son * 100:.1f}%  "
              f"overlap-stripped {moff * 100:.1f}%/{soff * 100:.1f}% "
              f"(collar={args.collar}s) ===")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    main()
