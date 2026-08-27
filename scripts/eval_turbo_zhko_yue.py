"""faster-whisper large-v3-turbo INT8 (CPU) on the zh/ko/yue real-speech sets.

Completes the 5-language head-to-head against hayamimi's production scorecard
(docs/SCORECARD.md): ja/en were already measured in docs/EVAL_REAL.md
(iteration #4-5); this script fills in zh, ko, yue using the same scoring
convention as scripts/eval_engine.py (cer_ja for zh/ko, cer_ja on
opencc t2s-normalized text for yue), so the two sides are directly
comparable.

Usage:
    python scripts/eval_turbo_zhko_yue.py
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from eval_accuracy import cer_ja
from eval_common import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFESTS = ["testdata/eval_real_zhko", "testdata/eval_real_yue"]

try:
    from opencc import OpenCC

    _t2s = OpenCC("t2s").convert
except Exception:
    _t2s = None


def score(lang: str, ref: str, hyp: str) -> float:
    if lang == "yue" and _t2s is not None:
        ref, hyp = _t2s(ref), _t2s(hyp)
    return cer_ja(ref, hyp)[0]


def main():
    from faster_whisper import WhisperModel

    print("Loading faster-whisper large-v3-turbo INT8 (CPU)...")
    t0 = time.perf_counter()
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    print(f"  load: {time.perf_counter() - t0:.2f}s")

    rows = []
    for rel in MANIFESTS:
        mdir = os.path.join(ROOT, rel)
        for e in load_manifest(mdir):
            wav_path = os.path.join(mdir, e["wav"])
            samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            dur = len(samples) / sr

            t0 = time.perf_counter()
            segments, _info = model.transcribe(wav_path, language=e["lang"], beam_size=1)
            hyp = "".join(seg.text for seg in segments).strip()
            dt = time.perf_counter() - t0

            err = score(e["lang"], e["ref"], hyp)
            rtf = dt / dur if dur > 0 else 0.0
            rows.append({"wav": e["wav"], "lang": e["lang"], "dur": dur, "err": err, "rtf": rtf, "hyp": hyp})
            print(f"{e['wav']:16} lang={e['lang']:3} err={err:.3f} rtf={rtf:.3f} hyp={hyp!r}")

    print("\n=== Aggregate (faster-whisper large-v3-turbo INT8, CPU) ===")
    for lang in ("zh", "ko", "yue"):
        sub = [r for r in rows if r["lang"] == lang]
        if not sub:
            continue
        err = sum(r["err"] * r["dur"] for r in sub) / sum(r["dur"] for r in sub)
        rtf = sum(r["rtf"] for r in sub) / len(sub)
        print(f"| {lang} | {len(sub)} | {err:.3f} | {rtf:.3f} |")


if __name__ == "__main__":
    main()
