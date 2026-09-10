"""English-path candidate evaluation for improvement track E.

Compares candidate sherpa-onnx Parakeet models for the "en" tier against the
current production model (Parakeet TDT 0.6B v3, which also carries the other
24 V3_LANGS European languages) on two English sets:

  - FLEURS en test, 100 clips (testdata/fleurs_bench/en, built by
    scripts/make_fleursset.py) -- WER via eval_accuracy.wer_en.
  - Real speech, 15 clips (testdata/eval_real, LibriSpeech dev-clean) --
    same scoring, this is the scorecard.md "en" row's source set.

Candidates (see docs/eval/en_candidates.md for the writeup):
  - v3   (baseline, already in production): models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
  - v2   (en-only, k2-fsa/sherpa-onnx asr-models release):
         models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
  - unified (en-only, 2026-04 release, NVIDIA parakeet-unified-en-0.6b):
         models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming

Each candidate is built the same way asr_engine.py builds v3
(OfflineRecognizer.from_transducer, model_type="nemo_transducer") -- all
three are NeMo RNNT/TDT transducer exports with encoder/decoder/joiner onnx
files, so the sherpa-onnx API surface is identical; only MODEL_DIR differs.

Threads are pinned to 2 (--threads, default 2) because five other
improvement tracks are evaluating in parallel on the same 6-core CPU -- RTF
numbers from this script are therefore "provisional, measured under
concurrent load" and documented as such, never presented as the engine's
uncontended RTF.

Usage:
    python scripts/eval_en_candidates.py --set fleurs --candidates v3,v2,unified
    python scripts/eval_en_candidates.py --set real --candidates v3,v2,unified
    python scripts/eval_en_candidates.py --set fleurs --candidates v2 --limit 10
    python scripts/eval_en_candidates.py --memory --candidates v2,unified

Results are checkpointed incrementally to
testdata/eval_en_candidates/results_<set>.json (dict keyed by
"<candidate>/<wav>") so a run can be split across multiple bounded
invocations and resumed.
"""
import argparse
import gc
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from eval_accuracy import wer_en
from eval_common import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FLEURS_EN_DIR = os.path.join(ROOT, "testdata", "fleurs_bench", "en")
REAL_EVAL_DIR = os.path.join(ROOT, "testdata", "eval_real")
RESULTS_DIR = os.path.join(ROOT, "testdata", "eval_en_candidates")

CANDIDATES = {
    "v3": os.path.join(MODELS_DIR, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"),
    "v2": os.path.join(MODELS_DIR, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
    "unified": os.path.join(MODELS_DIR, "sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming"),
}


def _find(model_dir: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


def build_recognizer(name: str, threads: int):
    import sherpa_onnx

    model_dir = CANDIDATES[name]
    encoder = _find(model_dir, "encoder*.onnx")
    decoder = _find(model_dir, "decoder*.onnx")
    joiner = _find(model_dir, "joiner*.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    if not (encoder and decoder and joiner and os.path.isfile(tokens)):
        raise FileNotFoundError(
            f"candidate {name!r}: missing transducer files under {model_dir} "
            f"(encoder={encoder!r} decoder={decoder!r} joiner={joiner!r})")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens,
        num_threads=threads, model_type="nemo_transducer",
    )


def decode(rec, samples, sr):
    stream = rec.create_stream()
    stream.accept_waveform(sr, samples)
    t0 = time.perf_counter()
    rec.decode_stream(stream)
    dt = time.perf_counter() - t0
    return stream.result.text, dt


def read_clip(mdir, wav_name):
    samples, sr = sf.read(os.path.join(mdir, wav_name), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def results_path(set_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"results_{set_name}.json")


def load_results(set_name: str) -> dict:
    p = results_path(set_name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(set_name: str, results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path(set_name), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_set(set_name, mdir, entries, candidates, threads, limit, offset, results):
    window = entries[offset:offset + limit] if limit is not None else entries[offset:]
    for cname in candidates:
        rec = None
        for e in window:
            key = f"{cname}/{e['wav']}"
            if key in results:
                continue
            if rec is None:
                print(f"[{set_name}/{cname}] loading recognizer from {CANDIDATES[cname]}")
                rec = build_recognizer(cname, threads)
            samples, sr = read_clip(mdir, e["wav"])
            dur = len(samples) / sr
            hyp, dt = decode(rec, samples, sr)
            err = wer_en(e["ref"], hyp)
            rtf = dt / dur if dur > 0 else 0.0
            results[key] = {
                "candidate": cname, "wav": e["wav"], "ref": e["ref"], "hyp": hyp,
                "err": err, "rtf": rtf, "dur": dur,
            }
            print(f"[{set_name}] {key:28} err={err:.4f} rtf={rtf:.3f} hyp={hyp!r}")
            save_results(set_name, results)
        del rec
        gc.collect()


def print_summary(set_name, candidates, results):
    print(f"\n=== {set_name} summary ===")
    for cname in candidates:
        sub = [v for v in results.values() if v["candidate"] == cname]
        if not sub:
            continue
        err = sum(r["err"] * r["dur"] for r in sub) / sum(r["dur"] for r in sub)
        rtf = sum(r["rtf"] for r in sub) / len(sub)
        print(f"| {cname:10} | n={len(sub):3} | WER={err:.4f} | mean_RTF={rtf:.4f} |")


def run_memory(candidates, threads):
    """Measure resident-set growth from loading each candidate alongside v3,
    the way RoutedASR would keep both resident under --max-resident.

    Prints RSS after: (1) process baseline, (2) v3 alone loaded+warmed,
    (3) v3 + candidate both loaded+warmed. The delta in step 3 minus step 2
    is the marginal cost of keeping the candidate resident next to v3.
    """
    import numpy as np
    import psutil

    proc = psutil.Process(os.getpid())

    def rss_mb():
        gc.collect()
        return proc.memory_info().rss / (1024 * 1024)

    silence = np.zeros(16000, dtype=np.float32)
    base = rss_mb()
    print(f"[memory] baseline RSS: {base:.1f} MB")

    v3 = build_recognizer("v3", threads)
    decode(v3, silence, 16000)
    after_v3 = rss_mb()
    print(f"[memory] +v3 loaded+warmed: {after_v3:.1f} MB (delta {after_v3 - base:.1f} MB)")

    for cname in candidates:
        if cname == "v3":
            continue
        rec = build_recognizer(cname, threads)
        decode(rec, silence, 16000)
        after_both = rss_mb()
        print(f"[memory] +v3 +{cname} loaded+warmed: {after_both:.1f} MB "
              f"(marginal +{after_both - after_v3:.1f} MB over v3 alone, "
              f"total delta {after_both - base:.1f} MB over process baseline)")
        del rec
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["fleurs", "real"], default="fleurs")
    ap.add_argument("--candidates", default="v3,v2,unified",
                     help="comma-separated subset of v3,v2,unified")
    ap.add_argument("--threads", type=int, default=2,
                     help="inference threads; kept low (default 2) because other "
                          "improvement tracks share this CPU during evaluation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--memory", action="store_true",
                     help="run the co-residency memory measurement instead of WER/RTF")
    args = ap.parse_args()

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    for c in candidates:
        if c not in CANDIDATES:
            ap.error(f"unknown candidate {c!r}, must be one of {sorted(CANDIDATES)}")

    if args.memory:
        run_memory(candidates, args.threads)
        return

    if args.set == "fleurs":
        mdir, manifest_dir = FLEURS_EN_DIR, FLEURS_EN_DIR
    else:
        mdir, manifest_dir = REAL_EVAL_DIR, REAL_EVAL_DIR

    entries = load_manifest(manifest_dir)
    if args.set == "real":
        entries = [e for e in entries if e["lang"] == "en"]

    results = load_results(args.set)
    run_set(args.set, mdir, entries, candidates, args.threads, args.limit, args.offset, results)
    print_summary(args.set, candidates, results)


if __name__ == "__main__":
    main()
