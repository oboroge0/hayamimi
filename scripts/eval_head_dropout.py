"""Measure ReazonSpeech zipformer "head dropout" and how well each onset
countermeasure prevents it.

Head dropout = the decode of a VAD speech segment loses the utterance's
leading characters. Silero's speech-start detection lags the true onset
(measured ~198ms behind on testdata/multi_sentence_ja.wav's first sentence,
docs/results/benchmarks.md 2026-08-31), so a segment handed to the recognizer as-is
can begin mid-phoneme and the zipformer never emits the first word.

This script reproduces the live path's VAD exactly (realtime_transcribe's
build_vad() defaults, 512-sample windows, flush at the end), takes the FIRST
speech segment of each clip -- onset loss is a first-segment phenomenon --
and decodes it under a grid of:

  decoding: greedy_search | modified_beam_search   (hayamimi ships beam)
  onset:    none          the VAD segment as-is
            preroll       up to PREROLL_S (1.0s) of the clip's REAL audio
                          before the onset, via AudioHistory.with_preroll
            silence300    300ms of zeros prepended
            silence1000   1.0s of zeros prepended

plus one reference row `production` = RoutedASR(forced_lang="ja") over the
pre-rolled audio, i.e. beam + pre-roll + split-retry, the shipped path.

The raw grid builds its own OfflineRecognizers with exactly
asr_engine._build_reazon's configuration (same files, model_type,
modeling_unit, threads) differing only in decoding_method, so no split-retry
or LID logic can interfere with what is being measured.

Scoring reuses eval_accuracy.normalize_ja (NFKC, punctuation and whitespace
stripped) and a character-level Levenshtein. Because the hypothesis covers
only the first VAD segment while the manifest reference covers the whole
clip, the alignment is END-FREE on the reference side: trailing reference
characters past the alignment's end cost nothing. Everything is then
measured on that aligned reference prefix.

Usage:
    python scripts/eval_head_dropout.py --limit 100
    python scripts/eval_head_dropout.py --limit 20 --determinism
    python scripts/eval_head_dropout.py --multi-sentence
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sherpa_onnx

import asr_engine
import realtime_transcribe as rt
from eval_accuracy import normalize_ja
from eval_common import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JA_DIR = os.path.join(ROOT, "testdata", "fleurs_bench", "ja")
MULTI_WAV = os.path.join(ROOT, "testdata", "multi_sentence_ja.wav")
OUT_JSON = os.path.join(ROOT, "docs", "eval", "head_dropout_results.json")

DECODERS = ["greedy_search", "modified_beam_search"]
ONSETS = ["none", "preroll", "silence300", "silence1000"]

# head dropout is "the alignment deletes >= HEAD_MIN_DEL reference characters
# before the first matched character", and only counts when the hypothesis is
# otherwise a real transcript of what it did decode (>= COVERAGE_MIN of the
# HYPOTHESIS's characters matched to reference characters). The denominator
# is deliberately the hypothesis, not the reference: a large head dropout
# removes reference characters by definition, so a reference-side denominator
# would reclassify the worst dropouts as "general error". A hypothesis below
# this bar is a general error, reported separately, not a head dropout.
HEAD_MIN_DEL = 2
COVERAGE_MIN = 0.60

# Many FLEURS recordings open with a sub-second click/breath/tone at ~0.1s
# that Silero emits as a speech segment of its own (min_speech_duration is
# 0.25s), so "the first VAD segment" is often not the utterance at all: 33 of
# 100 ja clips decoded their segment 0 as a 1-2 character noise token under
# every condition. Measure onset loss on the first segment that could carry
# an utterance instead. The rule is audio-only (duration), applied before any
# decoding and identically to every condition, so it cannot bias the
# comparison between conditions; the chosen index is recorded per clip.
MIN_UTTERANCE_S = 1.0


# ---------------------------------------------------------------------------
# Recognizers: _build_reazon's configuration, decoding_method varied
# ---------------------------------------------------------------------------

def build_reazon_with(decoding_method: str, threads: int):
    d = asr_engine.RZ_MODEL_DIR
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=asr_engine._find(d, "encoder-*.int8.onnx"),
        decoder=asr_engine._find(d, "decoder-*.int8.onnx"),
        joiner=asr_engine._find(d, "joiner-*.int8.onnx"),
        tokens=os.path.join(d, "tokens.txt"),
        num_threads=threads,
        model_type="zipformer",
        decoding_method=decoding_method,
        hotwords_file="",
        hotwords_score=2.0,
        modeling_unit="cjkchar",
    )


def decode(rec, samples, sample_rate):
    t0 = time.perf_counter()
    text = asr_engine.RoutedASR._decode(rec, samples, sample_rate)
    return text, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# VAD: the live path, verbatim
# ---------------------------------------------------------------------------

def vad_segments(samples, sample_rate):
    """Run the live capture VAD over `samples` the way run_stream() does.

    Returns a list of {"start", "samples", "preroll"} where "preroll" is what
    AudioHistory.with_preroll() -- fed the same 512-sample windows, in the
    same order, with the same clamp rules -- hands to the recognizer.
    """
    vad = rt.build_vad()
    history = rt.AudioHistory(sample_rate)
    segs = []

    def drain():
        while not vad.empty():
            seg = vad.front
            s = np.asarray(seg.samples, dtype=np.float32)
            segs.append({"start": int(seg.start), "samples": s,
                         "preroll": history.with_preroll(int(seg.start), s)})
            vad.pop()

    for chunk in rt.wav_chunks(samples, sample_rate, realtime=False):
        vad.accept_waveform(chunk)
        history.push(chunk)
        drain()
    vad.flush()
    drain()
    return segs


def first_utterance_index(segs, sample_rate):
    """Index of the first segment long enough to carry an utterance.

    Falls back to 0 when no segment reaches MIN_UTTERANCE_S. See that
    constant's comment for why segment 0 is not simply used.
    """
    for i, s in enumerate(segs):
        if len(s["samples"]) / sample_rate >= MIN_UTTERANCE_S:
            return i
    return 0


def onset_audio(seg, onset, sample_rate):
    if onset == "none":
        return seg["samples"]
    if onset == "preroll":
        return seg["preroll"]
    pad = {"silence300": 0.3, "silence1000": 1.0}[onset]
    return np.concatenate(
        [np.zeros(int(pad * sample_rate), dtype=np.float32), seg["samples"]])


# ---------------------------------------------------------------------------
# Scoring: end-free (on the reference side) character alignment
# ---------------------------------------------------------------------------

MATCH, SUB, DEL, INS = "M", "S", "D", "I"


def align_end_free(ref, hyp):
    """Align `hyp` to a PREFIX of `ref`; trailing reference is free.

    Returns (ops, ref_len, dist) where ops is the backtrace op sequence over
    the aligned reference prefix (DEL = a reference character the hypothesis
    does not have, INS = a hypothesis character the reference does not have).
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1,
                         prev[j - 1] + (0 if ri == hyp[j - 1] else 1))
    # end-free on the reference: pick the prefix length with the lowest cost
    best_i, best = 0, dp[0][m]
    for i in range(1, n + 1):
        if dp[i][m] < best:
            best_i, best = i, dp[i][m]

    ops = []
    i, j = best_i, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            ops.append(MATCH if ref[i - 1] == hyp[j - 1] else SUB)
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(DEL)
            i -= 1
        else:
            ops.append(INS)
            j -= 1
    ops.reverse()
    return ops, best_i, best


def score_pair(ref_raw, hyp_raw):
    ref, hyp = normalize_ja(ref_raw), normalize_ja(hyp_raw)
    ops, ref_len, dist = align_end_free(ref, hyp)
    matches = sum(1 for o in ops if o == MATCH)
    coverage = matches / len(hyp) if hyp else 0.0
    lead_del = 0
    for o in ops:
        if o == MATCH:
            break
        if o == DEL:
            lead_del += 1
    cer = dist / ref_len if ref_len else 0.0
    ok = coverage >= COVERAGE_MIN
    return {
        "cer": round(cer, 4), "coverage": round(coverage, 4), "lead_del": lead_del,
        "ref_len": ref_len, "hyp_len": len(hyp),
        "head": bool(ok and lead_del >= HEAD_MIN_DEL),
        "strict": bool(ok and lead_del >= 1),
        "general_error": not ok,
        "empty": not hyp,
    }


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def cond_name(dec, onset):
    return f"{'greedy' if dec == 'greedy_search' else 'beam'}/{onset}"


def make_production(threads):
    asr = asr_engine.RoutedASR(threads=threads, warmup=False, preload=False,
                               punctuate=False, forced_lang="ja")
    state = {"called": 0, "changed": 0}
    orig = asr._split_retry

    def wrapped(text, samples, sample_rate, lang, model):
        state["called"] += 1
        out = orig(text, samples, sample_rate, lang, model)
        if out != text:
            state["changed"] += 1
        return out

    asr._split_retry = wrapped
    return asr, state


def run_grid(entries, recs, asr, retry_state):
    per_clip = []
    for e in entries:
        path = os.path.join(JA_DIR, e["wav"])
        samples, sr = rt.read_wave(path)
        segs = vad_segments(samples, sr)
        row = {"wav": e["wav"], "ref": e["ref"], "n_segments": len(segs),
               "dur_s": round(len(samples) / sr, 3), "conditions": {}}
        if not segs:
            row["skipped"] = "no_vad_segment"
            per_clip.append(row)
            print(f"[{e['wav']}] NO VAD SEGMENT -- skipped")
            continue
        idx = first_utterance_index(segs, sr)
        seg = segs[idx]
        row["seg_index"] = idx
        row["seg_start_s"] = round(seg["start"] / sr, 3)
        row["seg_s"] = round(len(seg["samples"]) / sr, 3)
        row["preroll_s"] = round((len(seg["preroll"]) - len(seg["samples"])) / sr, 3)
        # sanity: none vs preroll differ ONLY by prepended samples
        assert len(seg["preroll"]) >= len(seg["samples"])
        assert np.array_equal(seg["preroll"][len(seg["preroll"]) - len(seg["samples"]):],
                              seg["samples"])

        for dec in DECODERS:
            for onset in ONSETS:
                text, ms = decode(recs[dec], onset_audio(seg, onset, sr), sr)
                s = score_pair(e["ref"], text)
                s["hyp"] = text
                s["ms"] = round(ms, 1)
                row["conditions"][cond_name(dec, onset)] = s
        if asr is not None:
            before = dict(retry_state)
            t0 = time.perf_counter()
            r = asr.transcribe(seg["preroll"], sr)
            ms = (time.perf_counter() - t0) * 1000.0
            s = score_pair(e["ref"], r["text"])
            s["hyp"] = r["text"]
            s["ms"] = round(ms, 1)
            s["split_retry_called"] = retry_state["called"] - before["called"]
            s["split_retry_changed"] = retry_state["changed"] - before["changed"]
            row["conditions"]["production"] = s
        per_clip.append(row)
        flags = "".join("H" if row["conditions"][cond_name(d, o)]["head"] else "."
                        for d in DECODERS for o in ONSETS)
        print(f"[{e['wav']}] segs={len(segs)} idx={idx} pre={row['preroll_s']:.2f}s {flags}")
    return per_clip


def is_onset_resolvable(clip):
    """Can the utterance's true beginning be reached from this segment at all?

    A FLEURS clip's reference covers the whole recording, but the segment
    under test covers only part of it, and Silero sometimes misses leading
    speech entirely (measured: clips whose only segment starts 2-4s in). On
    those clips every condition "deletes" the same leading reference
    characters -- a reference-coverage artifact, not an onset-handling
    failure -- so the absolute rates would be inflated for all of them alike.

    A clip counts as onset-resolvable when at least ONE condition lands on
    the reference's first character (lead_del == 0) with a hypothesis that is
    otherwise aligned. The criterion is a union over conditions, so it
    excludes only clips no condition can get right; it cannot favour any
    single condition.
    """
    return any(v["lead_del"] == 0 and v["coverage"] >= COVERAGE_MIN
               for v in clip["conditions"].values())


def summarize(per_clip, resolvable_only=False):
    names = [cond_name(d, o) for d in DECODERS for o in ONSETS] + ["production"]
    scored = [c for c in per_clip if c["conditions"]]
    if resolvable_only:
        scored = [c for c in scored if is_onset_resolvable(c)]
    out = {}
    for name in names:
        rows = [c["conditions"][name] for c in scored if name in c["conditions"]]
        if not rows:
            continue
        n = len(rows)
        out[name] = {
            "n": n,
            "head_dropout": sum(1 for r in rows if r["head"]),
            "strict_first_char": sum(1 for r in rows if r["strict"]),
            "general_error": sum(1 for r in rows if r["general_error"]),
            "empty_hyp": sum(1 for r in rows if r.get("empty")),
            "mean_lead_del": round(sum(r["lead_del"] for r in rows) / n, 2),
            "mean_cer": round(sum(r["cer"] for r in rows) / n, 4),
            "mean_ms": round(sum(r["ms"] for r in rows) / n, 1),
        }
        if any("split_retry_called" in r for r in rows):
            out[name]["split_retry_called"] = sum(r.get("split_retry_called", 0) for r in rows)
            out[name]["split_retry_changed"] = sum(r.get("split_retry_changed", 0) for r in rows)
    return out


def print_table(summary):
    print("\n| condition | n | head-dropout | strict | general err | mean CER | mean ms |")
    print("|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        print(f"| `{name}` | {s['n']} | {s['head_dropout']} | {s['strict_first_char']} | "
              f"{s['general_error']} | {s['mean_cer']:.4f} | {s['mean_ms']:.0f} |")


# ---------------------------------------------------------------------------
# multi_sentence_ja.wav: every segment, every condition, texts only
# ---------------------------------------------------------------------------

def run_multi(recs, asr):
    samples, sr = rt.read_wave(MULTI_WAV)
    segs = vad_segments(samples, sr)
    out = {"n_segments": len(segs), "dur_s": round(len(samples) / sr, 3), "segments": []}
    for idx, seg in enumerate(segs):
        row = {"index": idx, "start_s": round(seg["start"] / sr, 3),
               "seg_s": round(len(seg["samples"]) / sr, 3),
               "preroll_s": round((len(seg["preroll"]) - len(seg["samples"])) / sr, 3),
               "conditions": {}}
        for dec in DECODERS:
            for onset in ONSETS:
                text, ms = decode(recs[dec], onset_audio(seg, onset, sr), sr)
                row["conditions"][cond_name(dec, onset)] = {"hyp": text, "ms": round(ms, 1)}
        if asr is not None:
            r = asr.transcribe(seg["preroll"], sr)
            row["conditions"]["production"] = {"hyp": r["text"]}
        out["segments"].append(row)
        print(f"[multi seg{idx}] start={row['start_s']}s pre={row['preroll_s']}s")
        for k, v in row["conditions"].items():
            print(f"    {k:26} {v['hyp']}")
    return out


def _hist(values):
    h = {}
    for v in values:
        h[str(v)] = h.get(str(v), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: int(kv[0])))


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    data.update(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--multi-sentence", action="store_true",
                    help="run the grid on testdata/multi_sentence_ja.wav instead")
    ap.add_argument("--determinism", action="store_true",
                    help="run the grid twice and report whether hypotheses are identical")
    ap.add_argument("--no-production", action="store_true")
    args = ap.parse_args()

    recs = {d: build_reazon_with(d, args.threads) for d in DECODERS}
    asr, retry_state = (None, None) if args.no_production else make_production(args.threads)

    if args.multi_sentence:
        res = run_multi(recs, asr)
        _write(args.out, {"multi_sentence": res})
        print(f"\nwrote {args.out}")
        return

    entries = load_manifest(JA_DIR)[args.offset:args.offset + args.limit]
    t0 = time.perf_counter()
    per_clip = run_grid(entries, recs, asr, retry_state)
    wall = time.perf_counter() - t0
    summary = summarize(per_clip)
    summary_res = summarize(per_clip, resolvable_only=True)
    print("\nALL CLIPS")
    print_table(summary)
    print("\nONSET-RESOLVABLE SUBSET")
    print_table(summary_res)
    print(f"\nwall: {wall:.1f}s for {len(entries)} clips")

    payload = {
        "meta": {
            "dataset": "FLEURS ja test split (testdata/fleurs_bench/ja)",
            "clips": len(entries), "offset": args.offset,
            "threads": args.threads,
            "sherpa_onnx": sherpa_onnx.__version__,
            "python": sys.version.split()[0],
            "cpu": "AMD Ryzen 5 5600 (Windows 11)",
            "head_min_del": HEAD_MIN_DEL, "coverage_min": COVERAGE_MIN,
            "min_utterance_s": MIN_UTTERANCE_S,
            "wall_s": round(wall, 1),
        },
        "summary": summary,
        "summary_onset_resolvable": summary_res,
        "onset_resolvable_clips": [c["wav"] for c in per_clip
                                   if c["conditions"] and is_onset_resolvable(c)],
        "segment_counts": _hist([c["n_segments"] for c in per_clip]),
        "chosen_segment_index_counts": _hist(
            [c["seg_index"] for c in per_clip if "seg_index" in c]),
        "per_clip": per_clip,
    }

    if args.determinism:
        print("\n--- determinism: second pass ---")
        per_clip2 = run_grid(entries, recs, asr, retry_state)
        diffs = []
        for a, b in zip(per_clip, per_clip2):
            for k in a["conditions"]:
                if a["conditions"][k]["hyp"] != b["conditions"][k]["hyp"]:
                    diffs.append(f"{a['wav']} {k}")
        total = sum(len(c["conditions"]) for c in per_clip)
        print(f"determinism: {len(diffs)} differing hypotheses out of {total}")
        for d in diffs[:20]:
            print("  DIFF", d)
        payload["determinism"] = {"clips": len(entries), "compared": total,
                                  "differing": diffs}

    _write(args.out, payload)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
