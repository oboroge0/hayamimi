"""Korean ASR candidate evaluation (improvement track D).

Compares candidate systems for the Korean route against the current
production config (SenseVoice small, auto-detected language, i.e. the "sv"
tier in scripts/asr_engine.py) on two sets:

  - FLEURS ko (test split), 100 clips: testdata/fleurs_bench/ko/
  - Real spoken Korean, 12 clips: testdata/eval_real_zhko/ (lang == "ko")

Candidates:
  a) zipformer-ko-gainnorm  - sherpa-onnx-zipformer-korean-2024-06-24 with a
     peak-amplitude gain-normalization preprocessing step (see
     docs/eval/eval_real_zhko.md: this model returns empty transcriptions on
     quiet clips; gain norm was found to fix that on 2 manually-checked clips).
  c) sv-forced-ko           - SenseVoice small with language="ko" pinned
     (rather than the production "" auto-detect setting).
  d) omnilingual            - Meta Omnilingual ASR 300M CTC, no lang hint
     (contrast/reference point, already used elsewhere in this repo).

The current production baseline (SenseVoice auto-detect) is NOT re-run here
for FLEURS: those 100 numbers already exist in
testdata/fleurs_bench/results_hayamimi.json (produced by
scripts/eval_fleurs_bench.py --engine hayamimi) and are reused via
current_sv_auto_fleurs_summary() below. It IS re-run for the 12-clip real
set for an apples-to-apples empty-output check, since that set is small.

Threads are capped at 2 (--threads, default 2) because other improvement
tracks are evaluating in parallel on the same 6-core machine; RTF numbers
here are therefore provisional/contended, not the repo's usual RTF figures
(which use 6 threads).

Usage:
    python scripts/eval_ko_candidates.py --threads 2
    python scripts/eval_ko_candidates.py --threads 2 --system sv-forced-ko --set fleurs
"""
import argparse
import glob
import json
import os
import sys
import time
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

from eval_accuracy import cer_ja
from eval_common import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FLEURS_KO_DIR = os.path.join(ROOT, "testdata", "fleurs_bench", "ko")
REAL_ZHKO_DIR = os.path.join(ROOT, "testdata", "eval_real_zhko")
RESULTS_PATH = os.path.join(ROOT, "testdata", "ko_candidates_results.json")

MODEL_KO_ZIPFORMER_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-zipformer-korean-2024-06-24")
MODEL_SV_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
MODEL_OMNI_DIR = os.path.join(MODELS_DIR, "omnilingual-300m-ctc-int8")

QUIET_CLIP_IDS = {"ko_08.wav", "ko_09.wav"}  # known low-amplitude clips, see eval_real_zhko.md


def _find(model_dir, pattern):
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


def _load_wav(path):
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, sr


def _peak_normalize(samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Scale so the clip's peak absolute amplitude hits target_peak.

    No-op (returns as-is) on silence/near-silence to avoid divide-by-huge-gain
    amplifying noise floor into garbage.
    """
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak < 1e-6:
        return samples
    gain = target_peak / peak
    return samples * gain


# ---------------------------------------------------------------------------
# Candidate systems
# ---------------------------------------------------------------------------

class ZipformerKoGainNorm:
    """sherpa-onnx-zipformer-korean-2024-06-24 (offline transducer, INT8),
    with peak-amplitude gain normalization applied before accept_waveform.
    Candidate (a): fixes the known empty-output-on-quiet-clips failure mode
    (docs/eval/eval_real_zhko.md) without touching the model itself.
    """

    key = "zipformer-ko-gainnorm"
    label = "Zipformer-ko + gain norm"

    def __init__(self, threads):
        self.threads = threads
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            encoder = _find(MODEL_KO_ZIPFORMER_DIR, "encoder*int8*.onnx")
            decoder = _find(MODEL_KO_ZIPFORMER_DIR, "decoder*int8*.onnx")
            joiner = _find(MODEL_KO_ZIPFORMER_DIR, "joiner*int8*.onnx")
            tokens = os.path.join(MODEL_KO_ZIPFORMER_DIR, "tokens.txt")
            assert encoder and decoder and joiner, f"missing transducer parts in {MODEL_KO_ZIPFORMER_DIR}"
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=self.threads,
                model_type="zipformer",
            )
        return self._rec

    def transcribe(self, samples, sr):
        samples = _peak_normalize(samples)
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


class SenseVoiceForcedKo:
    """SenseVoice small, language="ko" pinned (candidate c) instead of the
    production "" (auto-detect) setting used by asr_engine._build_sense_voice.
    """

    key = "sv-forced-ko"
    label = "SenseVoice (language=ko pinned)"

    def __init__(self, threads):
        self.threads = threads
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=_find(MODEL_SV_DIR, "model*.onnx"),
                tokens=os.path.join(MODEL_SV_DIR, "tokens.txt"),
                num_threads=self.threads,
                use_itn=True,
                language="ko",
            )
        return self._rec

    def transcribe(self, samples, sr):
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


class SenseVoiceAutoKo:
    """SenseVoice small, language="" (auto) -- the CURRENT production config
    (asr_engine._build_sense_voice). Re-run here only for the small real-12
    set (for the same empty-output check as the candidates); the FLEURS-100
    numbers are reused from testdata/fleurs_bench/results_hayamimi.json.
    """

    key = "sv-auto-ko"
    label = "SenseVoice (current production, language=auto)"

    def __init__(self, threads):
        self.threads = threads
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            self._rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=_find(MODEL_SV_DIR, "model*.onnx"),
                tokens=os.path.join(MODEL_SV_DIR, "tokens.txt"),
                num_threads=self.threads,
                use_itn=True,
                language="",
            )
        return self._rec

    def transcribe(self, samples, sr):
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


class OmnilingualKo:
    """Meta Omnilingual ASR 300M CTC INT8, no lang hint (candidate d)."""

    key = "omnilingual"
    label = "Omnilingual 300M CTC (no lang hint)"

    def __init__(self, threads):
        self.threads = threads
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            self._rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
                model=_find(MODEL_OMNI_DIR, "model*.onnx"),
                tokens=os.path.join(MODEL_OMNI_DIR, "tokens.txt"),
                num_threads=self.threads,
            )
        return self._rec

    def transcribe(self, samples, sr):
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


SYSTEMS = {
    "zipformer-ko-gainnorm": ZipformerKoGainNorm,
    "sv-forced-ko": SenseVoiceForcedKo,
    "sv-auto-ko": SenseVoiceAutoKo,
    "omnilingual": OmnilingualKo,
}


# ---------------------------------------------------------------------------
# Data sets
# ---------------------------------------------------------------------------

def load_fleurs_ko():
    entries = load_manifest(FLEURS_KO_DIR)
    return [{"wav": e["wav"], "ref": e["ref"], "path": os.path.join(FLEURS_KO_DIR, e["wav"])} for e in entries]


def load_real_ko():
    entries = load_manifest(REAL_ZHKO_DIR)
    return [
        {"wav": e["wav"], "ref": e["ref"], "path": os.path.join(REAL_ZHKO_DIR, e["wav"])}
        for e in entries if e["lang"] == "ko"
    ]


SETS = {"fleurs": load_fleurs_ko, "real": load_real_ko}


# ---------------------------------------------------------------------------
# Runner (resumable, same convention as eval_fleurs_bench.py)
# ---------------------------------------------------------------------------

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run(system_key, set_key, threads, results):
    system = SYSTEMS[system_key](threads)
    entries = SETS[set_key]()
    for e in entries:
        rkey = f"{system_key}/{set_key}/{e['wav']}"
        if rkey in results:
            continue
        samples, sr = _load_wav(e["path"])
        dur = len(samples) / sr if sr else 0.0
        hyp, dt = system.transcribe(samples, sr)
        score, dist, denom = cer_ja(e["ref"], hyp)
        results[rkey] = {
            "system": system_key, "set": set_key, "wav": e["wav"],
            "hyp": hyp, "cer": score, "dist": dist, "denom": denom,
            "rtf": (dt / dur) if dur > 0 else 0.0, "dur": dur,
            "empty": not hyp.strip(),
        }
        print(f"[{system_key}/{set_key}] {e['wav']:12} CER={score:.4f} rtf={results[rkey]['rtf']:.3f} "
              f"empty={results[rkey]['empty']} hyp={hyp!r}")
        save_results(results)


def summarize(results):
    print("\n=== Summary (micro-avg CER, mean RTF) ===")
    combos = sorted({(v["system"], v["set"]) for v in results.values()})
    for system_key, set_key in combos:
        sub = [v for v in results.values() if v["system"] == system_key and v["set"] == set_key]
        dist = sum(v["dist"] for v in sub)
        denom = sum(v["denom"] for v in sub)
        cer = dist / denom if denom else float("nan")
        rtf = sum(v["rtf"] for v in sub) / len(sub) if sub else float("nan")
        n_empty = sum(1 for v in sub if v["empty"])
        quiet_empty = [v["wav"] for v in sub if v["empty"] and v["wav"] in QUIET_CLIP_IDS]
        print(f"{system_key:24} {set_key:8} n={len(sub):3} CER={cer:.4f} mean_rtf={rtf:.4f} "
              f"empty={n_empty} quiet_empty={quiet_empty}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2,
                     help="inference threads per recognizer (default 2: capped because other "
                          "improvement tracks share this CPU during eval)")
    ap.add_argument("--system", default="all", choices=list(SYSTEMS) + ["all"])
    ap.add_argument("--set", default="all", choices=list(SETS) + ["all"])
    args = ap.parse_args()

    system_keys = list(SYSTEMS) if args.system == "all" else [args.system]
    set_keys = list(SETS) if args.set == "all" else [args.set]

    results = load_results()
    for system_key in system_keys:
        for set_key in set_keys:
            run(system_key, set_key, args.threads, results)

    summarize(results)


if __name__ == "__main__":
    main()
