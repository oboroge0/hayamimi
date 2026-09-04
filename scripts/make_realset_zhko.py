"""Build a REAL-SPEECH zh + ko evaluation set and run the eval harness on it.

Same approach as scripts/make_realset.py (ja/en): pull real, human-spoken
utterances with reference transcripts from google/fleurs via the anonymous
Hugging Face "datasets-server" rows API (no auth needed).

  - Chinese (Mandarin): google/fleurs, config "cmn_hans_cn". The "test" split
    hits a datasets-server "Parquet scan size limit exceeded" error (its test
    shard is too large for the server's row-read cap), so we use the
    "validation" split instead, which serves fine.
  - Korean: google/fleurs, config "ko_kr", split "test" (works fine as-is).

For each language we page through rows, download the source WAV, convert to
16kHz mono s16 PCM WAV via ffmpeg, and keep utterances between MIN_DUR and
MAX_DUR_FALLBACK seconds.

Usage:
    python scripts/make_realset_zhko.py                # build set + run eval
    python scripts/make_realset_zhko.py --skip-build    # reuse existing wavs
    python scripts/make_realset_zhko.py --skip-eval     # only (re)build the set
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
EVAL_REAL_DIR = os.path.join(ROOT, "testdata", "eval_real_zhko")
MANIFEST_PATH = os.path.join(EVAL_REAL_DIR, "manifest.json")
DOCS_PATH = os.path.join(ROOT, "docs", "eval", "eval_real_zhko.md")

sys.path.insert(0, SCRIPTS_DIR)

N_PER_LANG = 12
MIN_DUR = 3.0
MAX_DUR = 9.0
MAX_DUR_FALLBACK = 15.0

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"

ZH_DATASET = "google/fleurs"
ZH_CONFIG = "cmn_hans_cn"
ZH_SPLIT = "validation"  # "test" split exceeds datasets-server's parquet scan limit

KO_DATASET = "google/fleurs"
KO_CONFIG = "ko_kr"
KO_SPLIT = "test"

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


def collect_lang(lang, dataset, config, split, out_prefix, get_text, get_audio_url):
    os.makedirs(EVAL_REAL_DIR, exist_ok=True)
    kept = []
    tmpdir = tempfile.mkdtemp(prefix="realset_zhko_")

    for max_dur in (MAX_DUR, MAX_DUR_FALLBACK):
        if len(kept) >= N_PER_LANG:
            break
        offset = 0
        while len(kept) < N_PER_LANG and offset < MAX_OFFSET:
            try:
                page = fetch_rows(dataset, config, split, offset, PAGE_SIZE)
            except urllib.error.HTTPError as e:
                print(f"  [{lang}] fetch_rows offset={offset} failed: {e}")
                break
            rows = page.get("rows", [])
            if not rows:
                break
            for entry in rows:
                if len(kept) >= N_PER_LANG:
                    break
                row = entry["row"]
                text = get_text(row).strip()
                if not text:
                    continue
                audio_url, ext = get_audio_url(row)
                idx = entry["row_idx"]
                raw_path = os.path.join(tmpdir, f"{lang}_{idx}{ext}")
                try:
                    download(audio_url, raw_path)
                except Exception as e:
                    print(f"  [{lang}] row {idx} download failed: {e}")
                    continue

                n = len(kept) + 1
                wav_name = f"{out_prefix}_{n:02d}.wav"
                wav_path = os.path.join(EVAL_REAL_DIR, wav_name)
                candidate_path = wav_path + ".candidate"
                try:
                    ffmpeg_to_wav16k(raw_path, candidate_path)
                except subprocess.CalledProcessError as e:
                    print(f"  [{lang}] row {idx} ffmpeg failed: {e}")
                    continue
                dur = wav_duration(candidate_path)
                already_have = {w for w, _, _ in kept}
                if MIN_DUR <= dur <= max_dur and wav_name not in already_have:
                    os.replace(candidate_path, wav_path)
                    kept.append((wav_name, text, dur))
                    print(f"  [{lang}] kept {wav_name} dur={dur:.2f}s ref={text[:40]!r}")
                else:
                    os.remove(candidate_path)
            offset += PAGE_SIZE

    if len(kept) < N_PER_LANG:
        print(f"WARNING: only found {len(kept)}/{N_PER_LANG} {lang} clips in duration "
              f"[{MIN_DUR},{MAX_DUR_FALLBACK}]s")
    return kept


def build_manifest():
    print("Collecting Chinese (Mandarin) clips from FLEURS (cmn_hans_cn, validation split)...")
    zh = collect_lang(
        "zh", ZH_DATASET, ZH_CONFIG, ZH_SPLIT, "zh",
        get_text=lambda row: row["raw_transcription"],
        get_audio_url=lambda row: (row["audio"][0]["src"], ".wav"),
    )

    print("Collecting Korean clips from FLEURS (ko_kr, test split)...")
    ko = collect_lang(
        "ko", KO_DATASET, KO_CONFIG, KO_SPLIT, "ko",
        get_text=lambda row: row["raw_transcription"],
        get_audio_url=lambda row: (row["audio"][0]["src"], ".wav"),
    )

    manifest = []
    for wav_name, text, _dur in zh:
        manifest.append({"wav": wav_name, "lang": "zh", "ref": text})
    for wav_name, text, _dur in ko:
        manifest.append({"wav": wav_name, "lang": "ko", "ref": text})

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_audio = sum(d for _, _, d in zh) + sum(d for _, _, d in ko)
    print(f"Wrote {MANIFEST_PATH} with {len(manifest)} entries "
          f"({len(zh)} zh + {len(ko)} ko), total audio {total_audio:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

MODEL_ZH_DIR = os.path.join(ROOT, "models", "sherpa-onnx-paraformer-zh-int8-2025-10-07")
MODEL_KO_DIR = os.path.join(ROOT, "models", "sherpa-onnx-zipformer-korean-2024-06-24")

THREADS = 6


def _find(model_dir, pattern):
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


class ParaformerZhSystem:
    """Dedicated Chinese Paraformer INT8 model."""

    name = "Paraformer-zh"

    def __init__(self):
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            model = _find(MODEL_ZH_DIR, "model*.onnx")
            assert model, f"no onnx model found in {MODEL_ZH_DIR}"
            self._rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model,
                tokens=os.path.join(MODEL_ZH_DIR, "tokens.txt"),
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


class ZipformerKoSystem:
    """Dedicated Korean zipformer transducer model."""

    name = "Zipformer-ko"

    def __init__(self):
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            encoder = _find(MODEL_KO_DIR, "encoder*int8*.onnx") or _find(MODEL_KO_DIR, "encoder*.onnx")
            decoder = _find(MODEL_KO_DIR, "decoder*int8*.onnx") or _find(MODEL_KO_DIR, "decoder*.onnx")
            joiner = _find(MODEL_KO_DIR, "joiner*int8*.onnx") or _find(MODEL_KO_DIR, "joiner*.onnx")
            tokens = os.path.join(MODEL_KO_DIR, "tokens.txt")
            assert encoder and decoder and joiner, f"missing transducer parts in {MODEL_KO_DIR}"
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
    import re
    import soundfile as sf
    from eval_accuracy import SenseVoiceSystem, OmnilingualSystem, normalize_ja, levenshtein

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    def cer_zh(ref, hyp):
        r = normalize_ja(ref)  # NFKC + strip punctuation/space; language-agnostic for CJK char-level CER
        h = normalize_ja(hyp)
        dist = levenshtein(r, h)
        rate = dist / len(r) if len(r) > 0 else 0.0
        return rate, dist, len(r)

    def cer_ko(ref, hyp):
        # Korean: also strip all whitespace (already done by normalize_ja) so that
        # spacing convention differences between reference and hypothesis don't
        # dominate the character-level score.
        return cer_zh(ref, hyp)

    dedicated = {"zh": ParaformerZhSystem(), "ko": ZipformerKoSystem()}
    systems_common = [SenseVoiceSystem(), OmnilingualSystem()]

    results = []
    for entry in manifest:
        wav_path = os.path.join(EVAL_REAL_DIR, entry["wav"])
        lang = entry["lang"]
        ref = entry["ref"]
        audio_s = sf.info(wav_path).duration

        row = {"wav": entry["wav"], "lang": lang, "ref": ref, "audio_s": audio_s}
        all_systems = systems_common + [dedicated[lang]]
        for sys_obj in all_systems:
            hyp, dt = sys_obj.transcribe(wav_path, lang)
            rtf = dt / audio_s if audio_s > 0 else 0.0
            score, dist, denom = (cer_zh if lang == "zh" else cer_ko)(ref, hyp)
            row[sys_obj.name] = {
                "hyp": hyp, "score": score, "dist": dist, "denom": denom,
                "decode_s": dt, "rtf": rtf,
            }
            print(f"[{entry['wav']}] {sys_obj.name}: CER={score:.4f} rtf={rtf:.3f} hyp={hyp!r}")
        results.append(row)

    # Aggregate per (lang, system)
    sys_names = {"SenseVoice", "Omnilingual", "Paraformer-zh", "Zipformer-ko"}
    agg = {}
    for lang in ("zh", "ko"):
        for name in sys_names:
            dist_sum = denom_sum = 0
            rtf_sum = 0.0
            rtf_n = 0
            for row in results:
                if row["lang"] != lang or name not in row:
                    continue
                r = row[name]
                dist_sum += r["dist"]
                denom_sum += r["denom"]
                rtf_sum += r["rtf"]
                rtf_n += 1
            if rtf_n == 0:
                continue
            agg[(lang, name)] = {
                "cer": dist_sum / denom_sum if denom_sum else float("nan"),
                "mean_rtf": rtf_sum / rtf_n,
                "n": rtf_n,
            }

    print("\n=== Aggregate (real speech, zh+ko) ===")
    for (lang, name), a in sorted(agg.items()):
        print(f"{lang}/{name}: CER={a['cer']:.4f}  mean_RTF={a['mean_rtf']:.4f}  n={a['n']}")

    write_markdown(results, agg)
    print(f"\nWrote {DOCS_PATH}")


def write_markdown(results, agg):
    lines = []
    lines.append("# ASR Accuracy Evaluation — Real Speech (Chinese + Korean)\n")
    n_zh = sum(1 for r in results if r["lang"] == "zh")
    n_ko = sum(1 for r in results if r["lang"] == "ko")
    total_audio = sum(r["audio_s"] for r in results)
    lines.append(
        f"Comparison of **SenseVoice** (multilingual), **Omnilingual** (multilingual, no lang "
        f"hint), and a **dedicated per-language model** on a small set of **real, human-spoken** "
        f"utterances ({n_zh} Mandarin Chinese + {n_ko} Korean, {total_audio:.1f}s total audio, "
        f"`testdata/eval_real_zhko/`). Goal: determine whether swapping SenseVoice for a "
        f"dedicated per-language model is worthwhile for zh and ko, mirroring the existing "
        f"ja/en real-speech comparison in `docs/eval/eval_real.md`.\n"
    )
    lines.append(
        "## Data source\n\n"
        "- **Both languages**: [FLEURS](https://huggingface.co/datasets/google/fleurs) "
        "(`google/fleurs`), read-aloud sentences recorded by native speakers, via the "
        "`datasets-server` anonymous rows API (no auth required).\n"
        "  - Chinese: config `cmn_hans_cn`. The `test` split's parquet shard exceeds the "
        "datasets-server row-read cap (`Parquet error: Scan size limit exceeded`), so the "
        "`validation` split was used instead — same corpus/recording conditions, just a "
        "different held-out split.\n"
        "  - Korean: config `ko_kr`, `test` split (served without issue).\n"
        "- Utterances were filtered to roughly 3-9s duration (widened to 3-15s only if needed). "
        "Source audio was converted to 16kHz mono 16-bit PCM WAV via ffmpeg.\n"
    )
    lines.append(
        "## Models\n\n"
        "- SenseVoice: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` "
        "(non-autoregressive, use_itn=True, language forced per entry)\n"
        "- Omnilingual: `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, no lang hint)\n"
        "- Dedicated Chinese: `sherpa-onnx-paraformer-zh-int8-2025-10-07` "
        "(Paraformer, `OfflineRecognizer.from_paraformer`)\n"
        "- Dedicated Korean: `sherpa-onnx-zipformer-korean-2024-06-24` "
        "(Zipformer transducer INT8, `OfflineRecognizer.from_transducer(model_type=\"zipformer\")`)\n"
    )
    lines.append(
        "## Metric\n\n"
        "Character error rate (CER), computed the same way as `cer_ja` in `scripts/eval_accuracy.py`: "
        "NFKC normalization, strip punctuation, strip all whitespace, then character-level "
        "Levenshtein distance / reference length (micro-averaged: total edits / total reference "
        "characters across all files for that language+system). For Korean this also removes "
        "spacing (어절 분리) differences between reference and hypothesis, since ASR systems don't "
        "reliably reproduce reference spacing conventions and that would otherwise dominate the "
        "score independent of actual transcription quality.\n"
    )

    def esc(s):
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines.append("\n## Per-file results\n")
    for lang, label, dedicated_name in (("zh", "Chinese", "Paraformer-zh"), ("ko", "Korean", "Zipformer-ko")):
        lines.append(f"\n### {label}\n")
        cols = ["SenseVoice", "Omnilingual", dedicated_name]
        header = "| file | ref |" + "".join(f" {c} hyp | {c} CER | {c} RTF |" for c in cols)
        lines.append(header)
        lines.append("|---|---|" + "---|" * (3 * len(cols)))
        for row in results:
            if row["lang"] != lang:
                continue
            cells = ""
            for c in cols:
                r = row[c]
                cells += f" {esc(r['hyp'])} | {r['score']:.3f} | {r['rtf']:.3f} |"
            lines.append(f"| {row['wav']} | {esc(row['ref'])} |" + cells)

    lines.append("\n## Aggregate\n")
    lines.append("| lang | system | CER (micro-avg) | mean RTF | n |")
    lines.append("|---|---|---|---|---|")
    for (lang, name), a in sorted(agg.items()):
        lines.append(f"| {lang} | {name} | {a['cer']:.4f} | {a['mean_rtf']:.4f} | {a['n']} |")

    lines.append("\n## Recommendation\n")
    for lang, label, dedicated_name in (("zh", "Chinese", "Paraformer-zh"), ("ko", "Korean", "Zipformer-ko")):
        sv = agg.get((lang, "SenseVoice"))
        ded = agg.get((lang, dedicated_name))
        if sv and ded:
            delta = sv["cer"] - ded["cer"]
            rel = (delta / sv["cer"] * 100.0) if sv["cer"] > 0 else 0.0
            if delta > 0 and rel >= 15:
                verdict = f"**Switch to {dedicated_name}**"
                reason = (
                    f"CER drops from {sv['cer']*100:.1f}% (SenseVoice) to {ded['cer']*100:.1f}% "
                    f"({dedicated_name}), a {rel:.0f}% relative reduction in errors, "
                    f"clearly outweighing any RTF difference "
                    f"(SenseVoice {sv['mean_rtf']:.3f} vs {dedicated_name} {ded['mean_rtf']:.3f})."
                )
            else:
                verdict = "**Keep SenseVoice**"
                reason = (
                    f"CER is {sv['cer']*100:.1f}% (SenseVoice) vs {ded['cer']*100:.1f}% "
                    f"({dedicated_name}) — not a clear enough win to justify adding/switching "
                    f"a second model for {label}, especially considering SenseVoice already "
                    f"covers 5 languages in one model (RTF: SenseVoice {sv['mean_rtf']:.3f} vs "
                    f"{dedicated_name} {ded['mean_rtf']:.3f})."
                )
            lines.append(f"- **{label}**: {verdict}. {reason}\n")

    lines.append("\n## Caveats\n")
    lines.append(
        "- **Small sample.** ~12 utterances per language. These numbers indicate relative "
        "system behavior, not statistically robust estimates.\n"
    )
    lines.append(
        "- **Chinese uses the `validation` split, not `test`**, purely due to a datasets-server "
        "row-size limitation on the `test` parquet shard — both are held-out FLEURS splits "
        "recorded under the same conditions, so this should not bias the comparison.\n"
    )
    lines.append(
        "- **FLEURS is read-aloud, single-speaker-per-clip speech** (similar register to "
        "LibriSpeech), not spontaneous/conversational or noisy far-field audio — real deployment "
        "conditions may show larger or smaller gaps between systems than measured here.\n"
    )
    lines.append(
        "- **Korean CER strips spacing**, which is a deliberate deviation from the raw `cer_ja` "
        "normalization (which already strips whitespace) — noted here since Korean word-spacing "
        "(띄어쓰기) is linguistically meaningful, unlike Chinese/Japanese where it's absent.\n"
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
