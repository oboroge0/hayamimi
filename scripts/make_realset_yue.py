"""Build a REAL-SPEECH Cantonese (yue) evaluation set and run the eval harness.

Same approach as scripts/make_realset_zhko.py: pull real, human-spoken
utterances with reference transcripts from google/fleurs via the anonymous
Hugging Face "datasets-server" rows API (no auth needed).

  - Cantonese: google/fleurs, config "yue_hant_hk". The "test" split returns
    a datasets-server HTTP 500 (same class of failure as cmn_hans_cn's "test"
    split exceeding the parquet scan limit), so we use the "validation" split
    instead, which serves fine.

FLEURS yue references are written in Traditional Chinese (Cantonese
orthography, Hong Kong). Some ASR systems (e.g. SenseVoice) may emit
Simplified Chinese characters for the same words, which is an orthographic
difference, not a transcription error. We report both:
  - raw CER (straight Levenshtein over NFKC-normalized, punctuation/space
    stripped text)
  - t2s CER (reference converted Traditional->Simplified via the `opencc`
    package before comparing), which removes most of that scripting mismatch.

For each row we page through results, download the source WAV, convert to
16kHz mono s16 PCM WAV via ffmpeg, and keep utterances between MIN_DUR and
MAX_DUR_FALLBACK seconds.

Usage:
    python scripts/make_realset_yue.py                # build set + run eval
    python scripts/make_realset_yue.py --skip-build    # reuse existing wavs
    python scripts/make_realset_yue.py --skip-eval     # only (re)build the set
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
EVAL_REAL_DIR = os.path.join(ROOT, "testdata", "eval_real_yue")
MANIFEST_PATH = os.path.join(EVAL_REAL_DIR, "manifest.json")
DOCS_PATH = os.path.join(ROOT, "docs", "eval", "eval_real_yue.md")

sys.path.insert(0, SCRIPTS_DIR)

N_CLIPS = 12
MIN_DUR = 3.0
MAX_DUR = 9.0
MAX_DUR_FALLBACK = 15.0

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"

YUE_DATASET = "google/fleurs"
YUE_CONFIG = "yue_hant_hk"
YUE_SPLIT = "validation"  # "test" split returns datasets-server HTTP 500

PAGE_SIZE = 40
MAX_OFFSET = 400


def fetch_rows(dataset, config, split, offset, length):
    from urllib.parse import quote

    url = (
        f"{HF_ROWS_URL}?dataset={quote(dataset, safe='')}"
        f"&config={config}&split={split}&offset={offset}&length={length}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-faster-eval/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def download(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "whisper-faster-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)


def ffmpeg_to_wav16k(src_path, dst_path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", src_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav",
            dst_path,
        ],
        check=True,
    )


def wav_duration(path):
    import soundfile as sf
    return sf.info(path).duration


def collect_yue():
    os.makedirs(EVAL_REAL_DIR, exist_ok=True)
    kept = []
    tmpdir = tempfile.mkdtemp(prefix="realset_yue_")

    for max_dur in (MAX_DUR, MAX_DUR_FALLBACK):
        if len(kept) >= N_CLIPS:
            break
        offset = 0
        while len(kept) < N_CLIPS and offset < MAX_OFFSET:
            try:
                page = fetch_rows(YUE_DATASET, YUE_CONFIG, YUE_SPLIT, offset, PAGE_SIZE)
            except urllib.error.HTTPError as e:
                print(f"  [yue] fetch_rows offset={offset} failed: {e}")
                break
            rows = page.get("rows", [])
            if not rows:
                break
            for entry in rows:
                if len(kept) >= N_CLIPS:
                    break
                row = entry["row"]
                text = row["raw_transcription"].strip()
                if not text:
                    continue
                audio_url = row["audio"][0]["src"]
                idx = entry["row_idx"]
                raw_path = os.path.join(tmpdir, f"yue_{idx}.wav")
                try:
                    download(audio_url, raw_path)
                except Exception as e:
                    print(f"  [yue] row {idx} download failed: {e}")
                    continue

                n = len(kept) + 1
                wav_name = f"yue_{n:02d}.wav"
                wav_path = os.path.join(EVAL_REAL_DIR, wav_name)
                candidate_path = wav_path + ".candidate"
                try:
                    ffmpeg_to_wav16k(raw_path, candidate_path)
                except subprocess.CalledProcessError as e:
                    print(f"  [yue] row {idx} ffmpeg failed: {e}")
                    continue
                dur = wav_duration(candidate_path)
                already_have = {w for w, _, _ in kept}
                if MIN_DUR <= dur <= max_dur and wav_name not in already_have:
                    os.replace(candidate_path, wav_path)
                    kept.append((wav_name, text, dur))
                    print(f"  [yue] kept {wav_name} dur={dur:.2f}s ref={text[:40]!r}")
                else:
                    os.remove(candidate_path)
            offset += PAGE_SIZE

    if len(kept) < N_CLIPS:
        print(f"WARNING: only found {len(kept)}/{N_CLIPS} yue clips in duration "
              f"[{MIN_DUR},{MAX_DUR_FALLBACK}]s")
    return kept


def build_manifest():
    print("Collecting Cantonese clips from FLEURS (yue_hant_hk, validation split)...")
    yue = collect_yue()

    manifest = []
    for wav_name, text, _dur in yue:
        manifest.append({"wav": wav_name, "lang": "yue", "ref": text})

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_audio = sum(d for _, _, d in yue)
    print(f"Wrote {MANIFEST_PATH} with {len(manifest)} entries, total audio {total_audio:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

MODEL_WENET_DIR = os.path.join(
    ROOT, "models", "sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10"
)
MODEL_ZIPFORMER_DIR = os.path.join(ROOT, "models", "sherpa-onnx-zipformer-cantonese-2024-03-13")

THREADS = 6


def _find(model_dir, pattern):
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


class WenetYueSystem:
    """Dedicated Cantonese WeNet U2++ Conformer CTC, INT8 (ASLP-lab WSYue-ASR)."""

    name = "WenetYue-CTC"

    def __init__(self):
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            model = os.path.join(MODEL_WENET_DIR, "model.int8.onnx")
            assert os.path.exists(model), f"missing {model}"
            self._rec = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                model=model,
                tokens=os.path.join(MODEL_WENET_DIR, "tokens.txt"),
                num_threads=THREADS,
            )
        return self._rec

    def transcribe(self, wav_path, lang):
        import soundfile as sf

        samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


class ZipformerYueSystem:
    """Dedicated Cantonese zipformer transducer, INT8 (icefall zipformer-cantonese)."""

    name = "Zipformer-yue"

    def __init__(self):
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            encoder = _find(MODEL_ZIPFORMER_DIR, "encoder*int8*.onnx")
            decoder = _find(MODEL_ZIPFORMER_DIR, "decoder*int8*.onnx")
            joiner = _find(MODEL_ZIPFORMER_DIR, "joiner*int8*.onnx")
            tokens = os.path.join(MODEL_ZIPFORMER_DIR, "tokens.txt")
            assert encoder and decoder and joiner, f"missing transducer parts in {MODEL_ZIPFORMER_DIR}"
            self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                num_threads=THREADS,
                model_type="zipformer",
            )
        return self._rec

    def transcribe(self, wav_path, lang):
        import soundfile as sf

        samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


def run_eval():
    import soundfile as sf
    from eval_accuracy import SenseVoiceSystem, OmnilingualSystem, normalize_ja, levenshtein

    try:
        import opencc
        _t2s = opencc.OpenCC("t2s").convert
    except Exception as e:
        print(f"WARNING: opencc unavailable ({e}); t2s-normalized CER will be skipped")
        _t2s = None

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    def cer_raw(ref, hyp):
        r = normalize_ja(ref)
        h = normalize_ja(hyp)
        dist = levenshtein(r, h)
        rate = dist / len(r) if len(r) > 0 else 0.0
        return rate, dist, len(r)

    def cer_t2s(ref, hyp):
        # Convert BOTH ref and hyp to Simplified before comparing, so that a
        # simplified hypothesis vs traditional reference (or vice versa,
        # e.g. a model that mixes scripts) is not penalized for script choice.
        r = normalize_ja(_t2s(ref)) if _t2s else normalize_ja(ref)
        h = normalize_ja(_t2s(hyp)) if _t2s else normalize_ja(hyp)
        dist = levenshtein(r, h)
        rate = dist / len(r) if len(r) > 0 else 0.0
        return rate, dist, len(r)

    systems = [SenseVoiceSystem(), OmnilingualSystem(), WenetYueSystem(), ZipformerYueSystem()]

    results = []
    for entry in manifest:
        wav_path = os.path.join(EVAL_REAL_DIR, entry["wav"])
        lang = entry["lang"]
        ref = entry["ref"]
        audio_s = sf.info(wav_path).duration

        row = {"wav": entry["wav"], "lang": lang, "ref": ref, "audio_s": audio_s}
        for sys_obj in systems:
            # SenseVoice needs language="yue" explicitly; others ignore lang.
            hyp, dt = sys_obj.transcribe(wav_path, "yue")
            rtf = dt / audio_s if audio_s > 0 else 0.0
            score_raw, dist_raw, denom_raw = cer_raw(ref, hyp)
            score_t2s, dist_t2s, denom_t2s = cer_t2s(ref, hyp)
            row[sys_obj.name] = {
                "hyp": hyp,
                "score": score_raw, "dist": dist_raw, "denom": denom_raw,
                "score_t2s": score_t2s, "dist_t2s": dist_t2s, "denom_t2s": denom_t2s,
                "decode_s": dt, "rtf": rtf,
            }
            print(f"[{entry['wav']}] {sys_obj.name}: CER={score_raw:.4f} CER_t2s={score_t2s:.4f} "
                  f"rtf={rtf:.3f} hyp={hyp!r}")
        results.append(row)

    # Aggregate per system
    agg = {}
    for sys_obj in systems:
        name = sys_obj.name
        dist_sum = denom_sum = 0
        dist_t2s_sum = denom_t2s_sum = 0
        rtf_sum = 0.0
        rtf_n = 0
        for row in results:
            r = row[name]
            dist_sum += r["dist"]
            denom_sum += r["denom"]
            dist_t2s_sum += r["dist_t2s"]
            denom_t2s_sum += r["denom_t2s"]
            rtf_sum += r["rtf"]
            rtf_n += 1
        if rtf_n == 0:
            continue
        agg[name] = {
            "cer": dist_sum / denom_sum if denom_sum else float("nan"),
            "cer_t2s": dist_t2s_sum / denom_t2s_sum if denom_t2s_sum else float("nan"),
            "mean_rtf": rtf_sum / rtf_n,
            "n": rtf_n,
        }

    print("\n=== Aggregate (real speech, yue) ===")
    for name, a in sorted(agg.items()):
        print(f"{name}: CER={a['cer']:.4f}  CER_t2s={a['cer_t2s']:.4f}  mean_RTF={a['mean_rtf']:.4f}  n={a['n']}")

    write_markdown(results, agg, systems)
    print(f"\nWrote {DOCS_PATH}")


def write_markdown(results, agg, systems):
    lines = []
    lines.append("# ASR Accuracy Evaluation — Real Speech (Cantonese / yue)\n")
    n = len(results)
    total_audio = sum(r["audio_s"] for r in results)
    lines.append(
        f"Comparison of **SenseVoice** (multilingual, `language=\"yue\"`), **Omnilingual** "
        f"(multilingual, no lang hint), and two **dedicated Cantonese models** on a small set "
        f"of **real, human-spoken** Cantonese utterances ({n} clips, {total_audio:.1f}s total "
        f"audio, `testdata/eval_real_yue/`). Goal: determine whether SenseVoice's built-in "
        f"`yue` route is good enough, or whether a dedicated Cantonese model should be swapped "
        f"in, mirroring the existing zh/ko real-speech comparison in `docs/eval/eval_real_zhko.md`.\n"
    )
    lines.append(
        "## Data source\n\n"
        "- [FLEURS](https://huggingface.co/datasets/google/fleurs) (`google/fleurs`), config "
        "`yue_hant_hk` (Cantonese, Hong Kong, Traditional-Chinese orthography), read-aloud "
        "sentences recorded by native speakers, via the `datasets-server` anonymous rows API "
        "(no auth required).\n"
        "  - The `test` split returned an HTTP 500 from the datasets-server rows API (the same "
        "class of failure seen for `cmn_hans_cn`'s `test` split, which exceeds the server's "
        "parquet scan limit), so the `validation` split was used instead — same corpus/recording "
        "conditions, just a different held-out split.\n"
        "- Utterances were filtered to roughly 3-9s duration (widened to 3-15s only if needed). "
        "Source audio was converted to 16kHz mono 16-bit PCM WAV via ffmpeg.\n"
    )
    lines.append(
        "## Models\n\n"
        "- SenseVoice: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` "
        "(non-autoregressive, use_itn=True, `language=\"yue\"`) — the model currently routed "
        "for yue in this repo\n"
        "- Omnilingual: `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, no lang hint)\n"
        "- Dedicated Cantonese (WeNet CTC): `sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-"
        "cantonese-int8-2025-09-10` (ASLP-lab WSYue-ASR, U2++ Conformer CTC, INT8, "
        "`OfflineRecognizer.from_wenet_ctc`)\n"
        "- Dedicated Cantonese (zipformer): `sherpa-onnx-zipformer-cantonese-2024-03-13` "
        "(icefall zipformer transducer, INT8, `OfflineRecognizer.from_transducer(model_type="
        "\"zipformer\")`)\n"
    )
    lines.append(
        "## Metric\n\n"
        "Character error rate (CER), computed the same way as `cer_ja` in `scripts/eval_accuracy.py`: "
        "NFKC normalization, strip punctuation, strip all whitespace, then character-level "
        "Levenshtein distance / reference length (micro-averaged: total edits / total reference "
        "characters across all files). Two variants are reported:\n\n"
        "- **CER (raw)**: reference as-is (Traditional Chinese, per FLEURS `yue_hant_hk`) vs "
        "hypothesis as-is.\n"
        "- **CER (t2s)**: both reference and hypothesis passed through `opencc`'s `t2s` "
        "(Traditional-to-Simplified) converter before comparing, then normalized/diffed the same "
        "way. This removes script-choice mismatches (e.g. a model that emits Simplified "
        "characters for the same Cantonese words) from the error count, isolating actual "
        "mistranscriptions.\n"
    )

    def esc(s):
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines.append("\n## Per-file results\n")
    cols = [s.name for s in systems]
    header = "| file | ref |" + "".join(f" {c} hyp | {c} CER | {c} CER(t2s) | {c} RTF |" for c in cols)
    lines.append(header)
    lines.append("|---|---|" + "---|" * (4 * len(cols)))
    for row in results:
        cells = ""
        for c in cols:
            r = row[c]
            cells += f" {esc(r['hyp'])} | {r['score']:.3f} | {r['score_t2s']:.3f} | {r['rtf']:.3f} |"
        lines.append(f"| {row['wav']} | {esc(row['ref'])} |" + cells)

    lines.append("\n## Aggregate\n")
    lines.append("| system | CER raw (micro-avg) | CER t2s (micro-avg) | mean RTF | n |")
    lines.append("|---|---|---|---|---|")
    for name, a in sorted(agg.items()):
        lines.append(f"| {name} | {a['cer']:.4f} | {a['cer_t2s']:.4f} | {a['mean_rtf']:.4f} | {a['n']} |")

    lines.append("\n## Recommendation\n")
    sv = agg.get("SenseVoice")
    candidates = [n for n in ("WenetYue-CTC", "Zipformer-yue") if n in agg]
    if sv and candidates:
        best_name = min(candidates, key=lambda n: agg[n]["cer_t2s"])
        best = agg[best_name]
        delta = sv["cer_t2s"] - best["cer_t2s"]
        rel = (delta / sv["cer_t2s"] * 100.0) if sv["cer_t2s"] > 0 else 0.0
        if delta > 0 and rel >= 15:
            verdict = f"**Switch to {best_name}**"
            reason = (
                f"CER(t2s) drops from {sv['cer_t2s']*100:.1f}% (SenseVoice) to "
                f"{best['cer_t2s']*100:.1f}% ({best_name}), a {rel:.0f}% relative reduction in "
                f"errors, outweighing the RTF difference (SenseVoice {sv['mean_rtf']:.3f} vs "
                f"{best_name} {best['mean_rtf']:.3f})."
            )
        else:
            verdict = "**Keep SenseVoice**"
            reason = (
                f"CER(t2s) is {sv['cer_t2s']*100:.1f}% (SenseVoice) vs "
                f"{best['cer_t2s']*100:.1f}% (best dedicated candidate, {best_name}) — not a "
                f"clear enough win to justify adding a second model for yue, especially since "
                f"SenseVoice already covers 5 languages (zh/en/ja/ko/yue) in one model "
                f"(RTF: SenseVoice {sv['mean_rtf']:.3f} vs {best_name} {best['mean_rtf']:.3f})."
            )
        lines.append(f"- {verdict}. {reason}\n")

    lines.append("\n## Caveats\n")
    lines.append(
        "- **Small sample.** ~12 utterances. These numbers indicate relative system behavior, "
        "not statistically robust estimates — a single unusual sentence can swing the aggregate "
        "noticeably.\n"
    )
    lines.append(
        "- **Cantonese uses the `validation` split, not `test`**, purely due to a "
        "datasets-server HTTP 500 on the `test` split's rows API — both are held-out FLEURS "
        "splits recorded under the same conditions, so this should not bias the comparison.\n"
    )
    lines.append(
        "- **Script normalization is approximate.** The `opencc` t2s conversion is a "
        "well-established rule-based converter, but Cantonese-specific characters/idioms "
        "(e.g. \u5514\u55b0/\u5514\u5567, \u55da) don't always have a 1:1 Simplified mapping and can still "
        "register as errors post-conversion even when the transcription is semantically correct. "
        "The raw (non-t2s) CER column is also reported for transparency, but is expected to be "
        "inflated for any system that outputs Simplified characters against the Traditional "
        "FLEURS reference.\n"
    )
    lines.append(
        "- **FLEURS is read-aloud, single-speaker-per-clip speech**, not spontaneous/"
        "conversational or noisy far-field audio — real deployment conditions (accents, code-"
        "switching with English/Mandarin, background noise) may show larger gaps between "
        "systems than measured here.\n"
    )
    lines.append(
        "- **WenetYue-CTC and Zipformer-yue are Cantonese-only models** (plus some Mandarin/"
        "English coverage per their training data) — unlike SenseVoice, they cannot serve the "
        "other 4 languages this repo already routes through SenseVoice, so adopting one adds a "
        "second loaded model rather than replacing SenseVoice outright, unless yue is split into "
        "its own route.\n"
    )

    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        build_manifest()
    else:
        print(f"Skipping build, reusing {MANIFEST_PATH}")

    if not args.skip_eval:
        t0 = time.time()
        run_eval()
        print(f"Eval took {time.time() - t0:.1f}s")
    else:
        print("Skipping eval run.")


if __name__ == "__main__":
    main()
