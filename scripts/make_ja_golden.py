"""Regenerate tests/golden/ja/golden.json by running the real ja pipeline.

Problem: docs/spec/ja_pipeline.ja.md specifies the Japanese route for
re-implementation elsewhere, and a specification nobody executes rots. The
golden set is the executable half: eight FLEURS ja clips, and the exact text
`scripts/realtime_transcribe.py` emits for each of them today. A change that
moves those texts has to be a change someone decided to make.

What this runs is the pipeline, not a re-creation of it. It builds the same
objects `realtime_transcribe.main()` builds for

    python scripts/realtime_transcribe.py --wav X --no-realtime \\
        --mode single --lang ja --threads 4

(`RoutedASR(forced_lang="ja")`, `LiveVad`, `AudioHistory`, `Refiner`), drives
them through `run_stream()` and the same finalization `main()`'s `finish()`
does, and reads the results off the `EventHub` with `add_listener` rather
than scraping stdout.

Two things are deliberately left out of the CLI's wiring, neither of which
can reach the recorded text: `--translate` is not passed, so no
`TranslationWorker` is started (`drain_segments` only submits to one when it
exists, and an empty `TranslatorPool` translates nothing either way), and no
`ControlQueue` is polled, because nothing submits to it here.

Usage:
    python scripts/make_ja_golden.py            # rewrite golden.json
    python scripts/make_ja_golden.py --check    # run, compare, write nothing
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import string
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "tests", "golden", "ja")
GOLDEN_JSON = os.path.join(GOLDEN_DIR, "golden.json")
MANIFEST = os.path.join(ROOT, "testdata", "fleurs_bench", "ja", "manifest.json")

# The settings the golden run pins, i.e. the CLI line in the module docstring.
SETTINGS = {
    "cli": ("scripts/realtime_transcribe.py --wav <clip> --no-realtime "
            "--mode single --lang ja --threads 4"),
    "threads": 4,
    "mode": "single",
    "lang": "ja",
    "realtime": False,
    "partials": True,
    "refine": True,
    "punctuate": True,
    "itn": True,
    "max_resident": 3,
}

# CER above which tests/test_ja_golden.py fails a clip whose text is not an
# exact match. int8 ONNX kernels are not bit-reproducible across CPU
# microarchitectures (different vector widths take different code paths in
# onnxruntime), so a different machine can legitimately render a character
# or two differently while transcribing the same words. 1.0% of a ~40
# character reference is under half a character, so in practice it admits a
# single-character difference on the longest clips here and nothing more --
# a dropped word or a lost sentence is many times that. See
# tests/golden/ja/README.md.
CER_TOLERANCE = 0.01

_PUNCT_RE = re.compile(
    "[" + re.escape(string.punctuation)
    + "　-〿！-／：-＠［-｀｛-･" + "]"
)


def normalize_ja(text: str) -> str:
    """NFKC, drop punctuation and whitespace -- the same normalization
    scripts/eval_accuracy.py scores ja CER under, restated here because that
    module imports jiwer, which the test environment does not install."""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    return re.sub(r"\s+", "", text)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ja_cer(ref: str, hyp: str) -> float:
    r, h = normalize_ja(ref), normalize_ja(hyp)
    if not r:
        return 0.0
    return levenshtein(r, h) / len(r)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_wavs() -> list[str]:
    return sorted(f for f in os.listdir(GOLDEN_DIR) if f.endswith(".wav"))


def fleurs_references() -> dict:
    """wav basename -> (FLEURS id, reference text), from the bench manifest."""
    with open(MANIFEST, encoding="utf-8") as f:
        return {e["wav"]: (e["id"], e["ref"]) for e in json.load(f)}


def run_clip(wav_path: str) -> dict:
    """Run one clip through the production ja pipeline; return its events.

    Everything heavy is imported here rather than at module scope so this
    file stays importable (for ja_cer) without models, numpy or sherpa-onnx.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import realtime_transcribe as rt
    from asr_engine import RoutedASR
    from subtitle_server import EventHub

    hub = EventHub()
    events: list[dict] = []
    hub.add_listener(events.append)

    asr = RoutedASR(threads=SETTINGS["threads"],
                    max_resident=SETTINGS["max_resident"],
                    hotwords_file="", replace_file="",
                    lid_switch_confirm=2, dual_confirm=True,
                    forced_lang=SETTINGS["lang"],
                    ja_second_opinion=False,
                    on_event=hub.publish)
    asr.min_switch_s = 2.0
    live_vad = rt.LiveVad(0.35, 12.0)
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=SETTINGS["partials"], hub=hub)
    history = rt.AudioHistory(rt.SAMPLE_RATE)
    refiner = rt.Refiner(asr, history, rt.SAMPLE_RATE, printer, stats=stats)

    samples, sr = rt.read_wave(wav_path)
    rt.run_stream(rt.wav_chunks(samples, sr, realtime=False), live_vad, sr,
                  asr, stats, printer, refiner, history)
    # main()'s finish(sr), minus the speaker bookkeeping --speakers adds
    live_vad.flush()
    rt.drain_segments(live_vad, sr, asr, stats, printer, history, refiner=refiner)
    refiner.maybe_refine(0, force=True)
    refiner._task_queue.join()  # noqa: SLF001 -- exactly what finish() does

    finals = [e["text"] for e in events if e.get("type") == "final"]
    refines = [e["text"] for e in events if e.get("type") == "refine"]
    return {"finals": finals, "refines": refines,
            "audio_s": round(len(samples) / sr, 3)}


def build_golden() -> dict:
    import sherpa_onnx

    refs = fleurs_references()
    clips = []
    for wav in clip_wavs():
        path = os.path.join(GOLDEN_DIR, wav)
        fleurs_id, ref = refs[wav]
        result = run_clip(path)
        joined = "".join(result["finals"])
        clips.append({
            "wav": wav,
            "sha256": sha256_file(path),
            "audio_s": result["audio_s"],
            "fleurs_id": fleurs_id,
            "reference": ref,
            "finals": result["finals"],
            "refine": result["refines"][-1] if result["refines"] else None,
            "cer_finals_vs_reference": round(ja_cer(ref, joined), 4),
            "cer_refine_vs_reference": (
                round(ja_cer(ref, result["refines"][-1]), 4)
                if result["refines"] else None),
        })
        print(f"  {wav}  finals={result['finals']}  refine={clips[-1]['refine']}")
    return {
        "_generated_by": "scripts/make_ja_golden.py",
        "_purpose": (
            "Pinned output of hayamimi's Japanese route (docs/spec/ja_pipeline.ja.md) "
            "on eight FLEURS ja clips. tests/test_ja_golden.py re-runs the "
            "pipeline and compares; see tests/golden/ja/README.md for the "
            "exact-match / CER two-level rule."
        ),
        "_source": (
            "FLEURS (google/fleurs, ja_jp), Copyright Google, licensed "
            "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "Clips copied unmodified from testdata/fleurs_bench/ja/; the "
            "reference text is that set's manifest.json."
        ),
        "_recorded_on": {
            "date": datetime.date.today().isoformat(),
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sherpa_onnx": sherpa_onnx.__version__,
        },
        "_settings": SETTINGS,
        "_cer_tolerance": CER_TOLERANCE,
        "clips": clips,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="run the clips and report differences, writing nothing")
    ap.add_argument("--out", default=GOLDEN_JSON)
    args = ap.parse_args()

    print(f"running {len(clip_wavs())} clips through the ja pipeline ...")
    golden = build_golden()

    if args.check:
        with open(args.out, encoding="utf-8") as f:
            committed = json.load(f)
        differ = [(new["wav"], old["finals"], new["finals"])
                  for old, new in zip(committed["clips"], golden["clips"])
                  if old["finals"] != new["finals"]]
        for wav, old, new in differ:
            print(f"DIFF {wav}\n  committed: {old}\n  now      : {new}")
        print(f"\n{len(differ)} of {len(golden['clips'])} clips differ")
        raise SystemExit(1 if differ else 0)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
