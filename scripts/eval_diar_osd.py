"""Dedicated-OSD-gate research for overlap-aware diarization (Track F, eval-only).

docs/design/diarization.md section 21. Like eval_diar_overlap.py (section 16)
and its Round 9 Experiment B pair_gate (section 20), this is a measurement
tool: nothing in realtime_transcribe.py / diarize.py calls it.

Background
----------
Section 16 built a whole-file overlap-aware diarization prototype on top of
pyannote segmentation-3.0's powerset head (unconditional argmax -> up to 2
simultaneous window-local speakers). Its overlap output's own net DER
contribution was small and, on low-overlap meetings, negative (-0.6pt
average, pyannote DER). Section 20 tried gating that argmax behind a single
per-frame confidence threshold on the pair class's own posterior (Bredin et
al.'s OSD-gate idea, retrofitted onto the SAME powerset head rather than a
model trained for OSD) and found the opposite of the literature's effect:
raising the threshold monotonically WORSENED the net contribution, because
the model's pair-class posterior is not reliably higher on true overlap
frames than on false-alarm ones -- mixed audio is intrinsically ambiguous to
this head, so a confidence floor discards true and false positives together.

This module explores the two research directions the section-20 write-up
left open, both without going and getting (or training) a model dedicated to
OSD:

  (b) "pseudo-OSD": a heavier-weight post-processing of the SAME powerset
      posteriors that section 20 gated -- but unlike section 20's
      single-frame instantaneous threshold, this adds temporal context and a
      joint per-speaker condition, i.e. the kind of structure a purpose-built
      OSD head would exploit implicitly:
        1. a JOINT marginal-probability floor: section 20 only ever looked
           at the argmax pair class's own posterior. This also requires both
           of the two speakers' individually-marginalized probabilities
           (summed over every powerset class that contains them, not just
           the pair class) to clear a floor -- catching the case where the
           model is confident SOME simultaneous-speech class fits better
           than a singleton, but lopsidedly (one speaker well-established,
           the other only weakly so), which the pair class's own posterior
           alone cannot distinguish from a jointly-confident true overlap.
        2. hysteresis (two-threshold Schmitt trigger) across the frame
           sequence WITHIN a window: entering "overlap-armed" state needs
           the pair class's own posterior >= hi_thresh, leaving it needs it
           to drop below lo_thresh (lo_thresh <= hi_thresh). A single-frame
           threshold (section 20) flickers frame to frame near its cutoff;
           this gives the decision inertia, matching the physical prior that
           real overlap does not blink on and off every 16.9ms.
        3. a minimum run-length on the (post-hysteresis) accepted overlap
           frames: any run shorter than min_run_frames is rejected back to
           the dominant single speaker, on the same reasoning
           activity_to_segments()'s min_duration_on already applies to
           emitted turns generally -- a would-be overlap span too short to
           be a real conversational event is more likely decode noise.
      pair_gate (section 20) is the min_run_frames=1, hi_thresh=lo_thresh,
      joint_floor=0.0 degenerate case of this (no memory, no joint
      condition); passing hi_thresh=lo_thresh=0.0, joint_floor=0.0,
      min_run_frames=1 degenerates further to section 16's unconditional
      argmax, and the test suite checks that equivalence directly.

  (c) survey of purpose-trained OSD models/heads published 2024-2026 (see
      docs/design/diarization.md section 21 for the write-up: pyannote's own
      dedicated OSD pipelines are gated on Hugging Face and inaccessible
      without an account+token this environment does not have; 3D-Speaker's
      "--include_overlap" is not a separate model at all -- it loads the
      same pyannote/segmentation-3.0 checkpoint this file already uses;
      NVIDIA's Sortformer is overlap-aware and CC-BY-4.0 but ONNX export is
      presently broken upstream for dynamic-slicing reasons; a 2025 WavLM-
      based speaker-aware progressive OSD paper has no released code or
      weights). None of these could be measured; no code lives here for (c).

Usage:
    python scripts/eval_diar_osd.py --mode none                       # sanity: reproduces section 16
    python scripts/eval_diar_osd.py --mode pseudo --hi-thresh 0.7 \
        --lo-thresh 0.5 --joint-floor 0.15 --min-run-frames 3
    python scripts/eval_diar_osd.py --mode pseudo --meeting ES2004a --json-out out.json
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
from eval_diar_overlap import (  # noqa: E402
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_HOP_S,
    DEFAULT_MIN_DURATION_OFF,
    DEFAULT_MIN_DURATION_ON,
    DEFAULT_RELIABLE_S,
    FRAME_SHIFT,
    MIN_EMBED_S,
    POWERSET,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    activity_to_segments,
    load_segmentation,
    overlap_fraction,
    strip_overlap,
    window_speaker_audio,
)
from global_recluster import (  # noqa: E402
    assign_by_centroid,
    cluster_reliable as cluster_local_speakers,
)

# Precomputed once, like eval_diar_overlap._PAIR_CLASSES/_SINGLE_OR_EMPTY_CLASSES.
_PAIR_CLASSES = [i for i, spk in enumerate(POWERSET) if len(spk) == 2]
_SINGLE_OR_EMPTY_CLASSES = [i for i, spk in enumerate(POWERSET) if len(spk) <= 1]
# (7, 3) membership matrix: MEMBERSHIP[c, spk] = 1 if spk in POWERSET[c]. Used
# to marginalize the 7-way softmax into a per-speaker "is this speaker active
# at all, in any class" probability -- probs @ MEMBERSHIP.
_MEMBERSHIP = np.zeros((len(POWERSET), 3), dtype=np.float32)
for _c, _spk in enumerate(POWERSET):
    for _s in _spk:
        _MEMBERSHIP[_c, _s] = 1.0

DEFAULT_HI_THRESH = 0.7
DEFAULT_LO_THRESH = 0.5
DEFAULT_JOINT_FLOOR = 0.15
DEFAULT_MIN_RUN_FRAMES = 3   # ~50.6ms at FRAME_SHIFT=270/16000Hz


def pseudo_osd_decode(logits: np.ndarray, hi_thresh: float, lo_thresh: float,
                      joint_floor: float, min_run_frames: int) -> np.ndarray:
    """(frames, 7) powerset log-posteriors -> (frames, 3) binary activity,
    gating pair-class argmax verdicts through hysteresis + a joint per-speaker
    marginal floor + a minimum run-length, instead of section 20's
    single-frame instantaneous threshold. See module docstring (b) for the
    reasoning behind each of the three gates. Like powerset_decode(), this
    only ever DOWNGRADES a pair verdict to a singleton/empty one -- it never
    invents overlap the plain argmax did not call in the first place.

    lo_thresh must be <= hi_thresh (a Schmitt trigger is meaningless
    otherwise); this is asserted rather than silently misbehaving.
    """
    assert lo_thresh <= hi_thresh, (lo_thresh, hi_thresh)
    assert min_run_frames >= 1, min_run_frames
    best = np.argmax(logits, axis=-1)
    probs = np.exp(logits)  # log-softmax -> softmax, monotonic (see
                            # eval_diar_overlap.powerset_decode's docstring)
    n = len(best)
    is_pair = np.isin(best, _PAIR_CLASSES)
    pair_conf = probs[np.arange(n), best]  # meaningful only where is_pair
    marginal = probs @ _MEMBERSHIP  # (frames, 3)

    # --- joint marginal floor: both speakers of the argmax pair class must
    # individually clear joint_floor, not just the pair class's own posterior.
    joint_ok = np.ones(n, dtype=bool)
    for cls in _PAIR_CLASSES:
        rows = best == cls
        if not rows.any():
            continue
        spk_a, spk_b = POWERSET[cls]
        ok = (marginal[rows, spk_a] >= joint_floor) & (marginal[rows, spk_b] >= joint_floor)
        joint_ok[rows] = ok

    # --- hysteresis (Schmitt trigger) over the pair-class posterior, frame
    # sequence order within this window. Sequential by construction (state
    # depends on the previous frame's state); a window is ~593 frames so this
    # python loop is a small fraction of the embedding-extraction cost that
    # dominates diarize_overlap() wall time (section 16).
    state = False
    armed = np.zeros(n, dtype=bool)
    for t in range(n):
        if is_pair[t] and pair_conf[t] >= hi_thresh:
            state = True
        elif not (is_pair[t] and pair_conf[t] >= lo_thresh):
            state = False
        armed[t] = state
    accepted_pair = is_pair & armed & joint_ok

    # --- minimum run-length: reject any accepted-pair run shorter than
    # min_run_frames back to False (activity_to_segments()'s min_duration_on
    # applies the analogous floor to emitted turns generally; this applies it
    # specifically to overlap spans before they ever reach that stage).
    if min_run_frames > 1 and accepted_pair.any():
        idx = np.flatnonzero(accepted_pair)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        run_starts = np.concatenate(([0], breaks + 1))
        run_ends = np.concatenate((breaks + 1, [len(idx)]))
        for rs, re in zip(run_starts, run_ends):
            if re - rs < min_run_frames:
                accepted_pair[idx[rs:re]] = False

    # frames whose argmax was a pair class but did not survive all three
    # gates get re-decided among the empty-set/singleton classes only --
    # "the dominant single speaker", same fallback section 20 uses.
    downgrade = is_pair & ~accepted_pair
    if downgrade.any():
        fallback_classes = np.asarray(_SINGLE_OR_EMPTY_CLASSES)
        restricted = probs[:, fallback_classes]
        fallback_best = fallback_classes[np.argmax(restricted, axis=-1)]
        best = np.where(downgrade, fallback_best, best)

    out = np.zeros((n, 3), dtype=bool)
    for cls, speakers in enumerate(POWERSET):
        if not speakers:
            continue
        rows = best == cls
        for spk in speakers:
            out[rows, spk] = True
    return out


def segment_windows_osd(samples: np.ndarray, sess, hop_samples: int, batch: int = 8,
                        mode: str = "pseudo", hi_thresh: float = DEFAULT_HI_THRESH,
                        lo_thresh: float = DEFAULT_LO_THRESH,
                        joint_floor: float = DEFAULT_JOINT_FLOOR,
                        min_run_frames: int = DEFAULT_MIN_RUN_FRAMES):
    """Same sliding-window contract as eval_diar_overlap.segment_windows, but
    decoded through pseudo_osd_decode() (mode="pseudo") instead of plain
    argmax (mode="none" -- the section-16 baseline, used to sanity-check this
    module reproduces it when all three gates are set to be no-ops)."""
    assert hop_samples % FRAME_SHIFT == 0, hop_samples
    starts = list(range(0, max(len(samples) - WINDOW_SAMPLES, 0) + 1, hop_samples))
    if not starts:
        starts = [0]
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
            if mode == "none":
                # section 16's unconditional argmax, reimplemented locally
                # (rather than importing powerset_decode) so this file has no
                # behavioral dependency on eval_diar_overlap.py beyond the
                # constants/helpers it explicitly imports.
                best = np.argmax(logits[j], axis=-1)
                act = np.zeros((logits[j].shape[0], 3), dtype=bool)
                for cls, speakers in enumerate(POWERSET):
                    if not speakers:
                        continue
                    rows = best == cls
                    for spk in speakers:
                        act[rows, spk] = True
                yield start, act
            else:
                yield start, pseudo_osd_decode(logits[j], hi_thresh, lo_thresh,
                                               joint_floor, min_run_frames)


def diarize_overlap_osd(wav_path: str, hop_s: float = DEFAULT_HOP_S,
                        thresholds: list[float] = (DEFAULT_CLUSTER_THRESHOLD,),
                        num_clusters: int | None = None,
                        min_on: float = DEFAULT_MIN_DURATION_ON,
                        min_off: float = DEFAULT_MIN_DURATION_OFF,
                        threads: int = 4, method: str = "average",
                        reliable_s: float = DEFAULT_RELIABLE_S,
                        mode: str = "pseudo", hi_thresh: float = DEFAULT_HI_THRESH,
                        lo_thresh: float = DEFAULT_LO_THRESH,
                        joint_floor: float = DEFAULT_JOINT_FLOOR,
                        min_run_frames: int = DEFAULT_MIN_RUN_FRAMES
                        ) -> tuple[dict[float, list], dict]:
    """Whole-file overlap-aware diarization identical in structure to
    eval_diar_overlap.diarize_overlap (segment -> exclusive-frame CAM++
    embeddings -> constrained clustering -> per-frame majority vote), except
    the segmentation stage runs through segment_windows_osd()'s pseudo-OSD
    gate rather than an unconditional/single-threshold-gated argmax. Like
    section 20's pair_gate, mode/hi_thresh/lo_thresh/joint_floor/
    min_run_frames are NOT free to sweep -- they change which frames count as
    overlap in the first place, so each combination needs its own
    segmentation + embedding pass. Only `thresholds` (the downstream
    clustering cut) is shared/free within one call.
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
    windows: list[tuple[int, np.ndarray]] = []
    embeddings: list[np.ndarray] = []
    window_ids: list[int] = []
    entries: list[tuple[int, int]] = []
    exclusive_durations: list[float] = []

    t0 = time.time()
    seg_out = list(segment_windows_osd(samples, sess, hop_samples, mode=mode,
                                       hi_thresh=hi_thresh, lo_thresh=lo_thresh,
                                       joint_floor=joint_floor,
                                       min_run_frames=min_run_frames))
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
        if reliable.sum() < 2:
            reliable = np.ones(len(emb_arr), dtype=bool)
        labels_reliable = cluster_local_speakers(
            emb_arr[reliable], [w for w, keep in zip(window_ids, reliable) if keep],
            threshold, num_clusters, method)
        labels = assign_by_centroid(emb_arr, reliable, labels_reliable)
        n_global = int(labels.max()) + 1 if len(labels) else 0

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
            active = votes[g] * 2 >= cover_safe
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
    ap.add_argument("--hop-s", type=float, default=DEFAULT_HOP_S)
    ap.add_argument("--thresholds", default=str(DEFAULT_CLUSTER_THRESHOLD),
                    help="comma-separated clustering thresholds, free to sweep within "
                         "one segmentation+embedding pass (same as eval_diar_overlap.py)")
    ap.add_argument("--num-clusters", type=int, default=None)
    ap.add_argument("--min-duration-on", type=float, default=DEFAULT_MIN_DURATION_ON)
    ap.add_argument("--min-duration-off", type=float, default=DEFAULT_MIN_DURATION_OFF)
    ap.add_argument("--reliable-s", type=float, default=DEFAULT_RELIABLE_S)
    ap.add_argument("--linkage", default="average", choices=["average", "complete", "single"])
    ap.add_argument("--mode", default="pseudo", choices=["none", "pseudo"],
                    help="'none' = section 16's unconditional argmax (sanity check this file "
                         "reproduces it); 'pseudo' = the three-gate OSD post-processing "
                         "(hysteresis + joint marginal floor + min run-length)")
    ap.add_argument("--hi-thresh", type=float, default=DEFAULT_HI_THRESH,
                    help="pair-class posterior needed to ENTER overlap-armed state")
    ap.add_argument("--lo-thresh", type=float, default=DEFAULT_LO_THRESH,
                    help="pair-class posterior below which overlap-armed state is left "
                         "(must be <= --hi-thresh)")
    ap.add_argument("--joint-floor", type=float, default=DEFAULT_JOINT_FLOOR,
                    help="both individual speakers' marginal probability (summed over every "
                         "powerset class containing them) must clear this")
    ap.add_argument("--min-run-frames", type=int, default=DEFAULT_MIN_RUN_FRAMES,
                    help="accepted-pair runs shorter than this (in 16.875ms frames) are "
                         "rejected back to the dominant single speaker")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json-out", default=None)
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
        hyps, stats = diarize_overlap_osd(
            wav_path, args.hop_s, thresholds, args.num_clusters,
            args.min_duration_on, args.min_duration_off, args.threads, args.linkage,
            args.reliable_s, args.mode, args.hi_thresh, args.lo_thresh, args.joint_floor,
            args.min_run_frames)
        wall = time.time() - t0

        for threshold in thresholds:
            hyp = segments_to_der_tuples(hyps[threshold])
            stripped = segments_to_der_tuples(strip_overlap(hyps[threshold]))
            on = der_breakdown(ref, hyp, args.collar)
            off = der_breakdown(ref, stripped, args.collar)
            on["der_simple"] = simpleder.DER(ref, hyp, collar=args.collar)
            off["der_simple"] = simpleder.DER(ref, stripped, collar=args.collar)
            n_spk = len({s for s, _, _ in hyp})
            row = {
                **entry, "threshold": threshold, "overlap_fraction": ov,
                "n_hyp_speakers": n_spk, "wall_s": wall, "mode": args.mode,
                "hi_thresh": args.hi_thresh, "lo_thresh": args.lo_thresh,
                "joint_floor": args.joint_floor, "min_run_frames": args.min_run_frames,
                **stats, "overlap_on": on, "overlap_stripped": off,
            }
            rows.append(row)
            print(f"[{entry['meeting']}] mode={args.mode} t={threshold:.2f} "
                  f"ref_ov={ov * 100:.1f}%  hyp_speakers={n_spk}  "
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
              f"embed={stats['embed_time_s']:.1f}s cluster={stats['cluster_time_s']:.1f}s)")

    for threshold in thresholds:
        sel = [r for r in rows if r["threshold"] == threshold]
        if not sel:
            continue
        mon = sum(r["overlap_on"]["der_breakdown"] for r in sel) / len(sel)
        moff = sum(r["overlap_stripped"]["der_breakdown"] for r in sel) / len(sel)
        son = sum(r["overlap_on"]["der_simple"] for r in sel) / len(sel)
        soff = sum(r["overlap_stripped"]["der_simple"] for r in sel) / len(sel)
        print(f"\n=== mode={args.mode} t={threshold:.2f} mean DER (pyannote/simpleder) over "
              f"{len(sel)} meeting(s): overlap-on {mon * 100:.1f}%/{son * 100:.1f}%  "
              f"overlap-stripped {moff * 100:.1f}%/{soff * 100:.1f}% "
              f"(net={100 * (mon - moff):+.1f}pt/{100 * (son - soff):+.1f}pt) "
              f"(collar={args.collar}s) ===")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    main()
