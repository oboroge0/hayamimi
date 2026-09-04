"""Unified FLEURS test-split head-to-head: hayamimi production path vs
faster-whisper large-v3-turbo, same clips, same scoring, same CPU.

Data: testdata/fleurs_bench/<lang>/ built by scripts/make_fleursset.py
(5 languages x 100 clips, FLEURS test split).

Two engines, selected with --engine:

  - hayamimi: the real production RoutedASR path (auto LID -> tier routing
    -> decode -> ja punctuation), exactly like scripts/eval_engine.py.
    Default settings (second-opinion OFF). asr.reset_session() is called
    once per language block, same convention as eval_engine.py, so the
    engine's sticky-LID hysteresis doesn't charge the first clip of a
    language for "switching away" from a previous language's last clip.

  - turbo: faster-whisper large-v3-turbo, device=cpu, compute_type=int8,
    beam_size=1, language=<given>. Turbo is handed the correct language
    directly -- a documented asymmetry vs. hayamimi's auto-LID path, matching
    the conditions stated in docs/results/comparison.md.

Scoring: reuses eval_accuracy.wer_en / eval_accuracy.cer_ja exactly as
eval_engine.py does: en=WER, others=CER (yue additionally t2s-normalized via
opencc when available), so numbers are comparable in kind with the existing
scorecard.

Resumable: results are written incrementally to
testdata/fleurs_bench/results_<engine>.json (a dict keyed by "<lang>/<wav>").
Re-running skips clips that already have a recorded result. Use --lang and
--limit/--offset to bound a single invocation's wall time (turbo is the slow
engine; ~30 clips/call at RTF~1.4 on ~12s clips keeps a call under ~8 min).

Usage:
    python scripts/eval_fleurs_bench.py --engine hayamimi --lang ja
    python scripts/eval_fleurs_bench.py --engine turbo --lang ja --offset 0 --limit 30
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from eval_accuracy import cer_ja, wer_en
from eval_common import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_ROOT = os.path.join(ROOT, "testdata", "fleurs_bench")

LANGS = ["ja", "en", "zh", "ko", "yue"]

try:
    from opencc import OpenCC

    _t2s = OpenCC("t2s").convert
except Exception:
    _t2s = None


def score(lang: str, ref: str, hyp: str) -> float:
    if lang == "en":
        return wer_en(ref, hyp)
    r, h = ref, hyp
    if lang == "yue" and _t2s is not None:
        r, h = _t2s(r), _t2s(h)
    return cer_ja(r, h)[0]


def results_path(engine: str) -> str:
    return os.path.join(BENCH_ROOT, f"results_{engine}.json")


def load_results(engine: str) -> dict:
    p = results_path(engine)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(engine: str, results: dict):
    with open(results_path(engine), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def read_clip(mdir, wav_name):
    samples, sr = sf.read(os.path.join(mdir, wav_name), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def run_hayamimi(langs, results):
    from asr_engine import RoutedASR

    asr = RoutedASR(threads=6, preload=False)

    for lang in langs:
        mdir = os.path.join(BENCH_ROOT, lang)
        entries = load_manifest(mdir)
        pending = [e for e in entries if f"{lang}/{e['wav']}" not in results]
        if not pending:
            print(f"[hayamimi/{lang}] all {len(entries)} clips already scored, skipping")
            continue
        # Fresh language block -> reset sticky-LID state (see eval_engine.py).
        asr.reset_session()
        for e in entries:
            key = f"{lang}/{e['wav']}"
            if key in results:
                continue
            samples, sr = read_clip(mdir, e["wav"])
            dur = len(samples) / sr
            t0 = time.perf_counter()
            r = asr.transcribe(samples, sr)
            wall = time.perf_counter() - t0
            err = score(lang, e["ref"], r["text"])
            results[key] = {
                "lang": lang, "wav": e["wav"], "hyp": r["text"], "detected": r["lang"],
                "tier": r["tier"], "err": err, "rtf": wall / dur if dur > 0 else 0.0, "dur": dur,
            }
            print(f"[hayamimi] {key:16} lid={r['lang']:3} tier={r['tier']:4} "
                  f"err={err:.3f} rtf={results[key]['rtf']:.3f}")
            save_results("hayamimi", results)


def run_turbo(langs, results, offset, limit):
    from faster_whisper import WhisperModel

    print("Loading faster-whisper large-v3-turbo INT8 (CPU)...")
    t0 = time.perf_counter()
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    print(f"  load: {time.perf_counter() - t0:.2f}s")

    for lang in langs:
        mdir = os.path.join(BENCH_ROOT, lang)
        entries = load_manifest(mdir)
        window = entries[offset:offset + limit] if limit is not None else entries[offset:]
        for e in window:
            key = f"{lang}/{e['wav']}"
            if key in results:
                continue
            wav_path = os.path.join(mdir, e["wav"])
            samples, sr = read_clip(mdir, e["wav"])
            dur = len(samples) / sr
            t0 = time.perf_counter()
            segments, _info = model.transcribe(wav_path, language=lang, beam_size=1)
            hyp = "".join(seg.text for seg in segments).strip()
            wall = time.perf_counter() - t0
            err = score(lang, e["ref"], hyp)
            results[key] = {
                "lang": lang, "wav": e["wav"], "hyp": hyp,
                "err": err, "rtf": wall / dur if dur > 0 else 0.0, "dur": dur,
            }
            print(f"[turbo] {key:16} err={err:.3f} rtf={results[key]['rtf']:.3f}")
            save_results("turbo", results)


def print_summary(engine, langs, results):
    print(f"\n=== {engine} summary ===")
    for lang in langs:
        sub = [v for k, v in results.items() if v["lang"] == lang]
        if not sub:
            continue
        err = sum(r["err"] * r["dur"] for r in sub) / sum(r["dur"] for r in sub)
        rtf = sum(r["rtf"] for r in sub) / len(sub)
        print(f"| {lang} | n={len(sub)} | mean_err={err:.4f} | mean_rtf={rtf:.4f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["hayamimi", "turbo"])
    ap.add_argument("--lang", default="all", choices=LANGS + ["all"])
    ap.add_argument("--offset", type=int, default=0, help="turbo only: start index within language's manifest")
    ap.add_argument("--limit", type=int, default=None, help="turbo only: max clips to process this call")
    args = ap.parse_args()

    langs = LANGS if args.lang == "all" else [args.lang]
    results = load_results(args.engine)

    if args.engine == "hayamimi":
        run_hayamimi(langs, results)
    else:
        run_turbo(langs, results, args.offset, args.limit)

    print_summary(args.engine, langs, results)


if __name__ == "__main__":
    main()
