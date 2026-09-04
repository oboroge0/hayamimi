"""Build a REAL-SPEECH evaluation set and run the existing eval harness on it.

Unlike testdata/eval/ (TTS-synthesized, scripts/make_testset.py), this script
pulls a small number of real, human-spoken utterances with reference
transcripts from two openly-accessible corpora, exposed anonymously through
the Hugging Face "datasets-server" rows API (no auth / no huggingface_hub
dependency required — plain HTTPS + urllib):

  - Japanese: ReazonSpeech (test split), via the mirror dataset
    "japanese-asr/ja_asr.reazonspeech_test" (audio + transcription columns).
    ReazonSpeech itself is built from Japanese TV broadcasts; this mirror
    exposes a small test split without gating.
  - English: LibriSpeech dev-clean ("clean"/"validation" config/split of
    "openslr/librispeech_asr"), read-aloud audiobook recordings, the
    standard ASR benchmark distributed by OpenSLR.

For each language we page through rows, download the source audio (FLAC),
convert to 16kHz mono s16 PCM WAV via ffmpeg, and keep utterances between
MIN_DUR and MAX_DUR seconds so the whole ~30-clip set stays a few minutes of
audio (Whisper large-v3-turbo on CPU runs at roughly 1-2x audio duration per
file).

Usage:
    python scripts/make_realset.py                # build set + run eval
    python scripts/make_realset.py --skip-build    # reuse existing wavs
    python scripts/make_realset.py --skip-eval     # only (re)build the set
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
EVAL_REAL_DIR = os.path.join(ROOT, "testdata", "eval_real")
MANIFEST_PATH = os.path.join(EVAL_REAL_DIR, "manifest.json")
DOCS_PATH = os.path.join(ROOT, "docs", "eval", "eval_real.md")

sys.path.insert(0, SCRIPTS_DIR)

N_PER_LANG = 15
MIN_DUR = 3.0
MAX_DUR = 9.0   # keep the set short; still within the 3-15s spec but biased
                # short so 15+15 clips stay in the ~2-4 minute total range.
MAX_DUR_FALLBACK = 15.0  # widened if not enough short clips are found

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"

JA_DATASET = "japanese-asr/ja_asr.reazonspeech_test"
JA_CONFIG = "default"
JA_SPLIT = "test"

EN_DATASET = "openslr/librispeech_asr"
EN_CONFIG = "clean"
EN_SPLIT = "validation"  # this is LibriSpeech dev-clean

PAGE_SIZE = 40
MAX_OFFSET = 400  # safety cap on how many rows we'll page through


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
    """Page through datasets-server rows, download+convert candidates, and
    keep N_PER_LANG clips with duration in [MIN_DUR, MAX_DUR] (widened to
    MAX_DUR_FALLBACK if the strict range doesn't yield enough)."""
    os.makedirs(EVAL_REAL_DIR, exist_ok=True)
    kept = []  # list of (wav_filename, ref_text, duration)
    offset = 0
    tmpdir = tempfile.mkdtemp(prefix="realset_")

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
    print("Collecting Japanese clips from ReazonSpeech (test split)...")
    ja = collect_lang(
        "ja", JA_DATASET, JA_CONFIG, JA_SPLIT, "ja",
        get_text=lambda row: row["transcription"],
        get_audio_url=lambda row: (row["audio"][0]["src"], ".flac"),
    )

    print("Collecting English clips from LibriSpeech dev-clean...")
    en = collect_lang(
        "en", EN_DATASET, EN_CONFIG, EN_SPLIT, "en",
        get_text=lambda row: row["text"],
        get_audio_url=lambda row: (row["audio"][0]["src"], ".flac"),
    )

    manifest = []
    for wav_name, text, _dur in ja:
        manifest.append({"wav": wav_name, "lang": "ja", "ref": text})
    for wav_name, text, _dur in en:
        manifest.append({"wav": wav_name, "lang": "en", "ref": text})

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_audio = sum(d for _, _, d in ja) + sum(d for _, _, d in en)
    print(f"Wrote {MANIFEST_PATH} with {len(manifest)} entries "
          f"({len(ja)} ja + {len(en)} en), total audio {total_audio:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# Eval runner (reuses system classes + metrics from eval_accuracy.py)
# ---------------------------------------------------------------------------

def run_eval():
    import soundfile as sf
    from eval_accuracy import (
        ParakeetSystem, SenseVoiceSystem, OmnilingualSystem, WhisperSystem,
        normalize_en, normalize_ja, wer_en, cer_ja,
    )
    import jiwer

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    systems = [ParakeetSystem(), SenseVoiceSystem(), OmnilingualSystem(), WhisperSystem()]

    results = []
    for entry in manifest:
        wav_path = os.path.join(EVAL_REAL_DIR, entry["wav"])
        lang = entry["lang"]
        ref = entry["ref"]
        audio_s = sf.info(wav_path).duration

        row = {"wav": entry["wav"], "lang": lang, "ref": ref, "audio_s": audio_s}
        for sys_obj in systems:
            hyp, dt = sys_obj.transcribe(wav_path, lang)
            rtf = dt / audio_s if audio_s > 0 else 0.0
            if lang == "en":
                score = wer_en(ref, hyp)
                dist, denom = None, None
            else:
                score, dist, denom = cer_ja(ref, hyp)
            row[sys_obj.name] = {
                "hyp": hyp, "score": score, "dist": dist, "denom": denom,
                "decode_s": dt, "rtf": rtf,
            }
            print(f"[{entry['wav']}] {sys_obj.name}: score={score:.4f} rtf={rtf:.3f} hyp={hyp!r}")
        results.append(row)

    agg = {}
    for sys_obj in systems:
        name = sys_obj.name
        en_num = en_den = 0.0
        ja_dist = ja_denom = 0
        rtf_sum = 0.0
        rtf_n = 0
        for row in results:
            r = row[name]
            rtf_sum += r["rtf"]
            rtf_n += 1
            if row["lang"] == "en":
                r_norm = normalize_en(row["ref"])
                h_norm = normalize_en(r["hyp"])
                r_words = r_norm.split()
                if len(r_words):
                    out = jiwer.process_words(r_norm, h_norm if h_norm else " ")
                    edits = out.substitutions + out.deletions + out.insertions
                    denom = out.substitutions + out.deletions + out.hits
                    en_num += edits
                    en_den += max(denom, 1)
            else:
                ja_dist += r["dist"]
                ja_denom += r["denom"]
        agg[name] = {
            "wer_en": (en_num / en_den) if en_den > 0 else float("nan"),
            "cer_ja": (ja_dist / ja_denom) if ja_denom > 0 else float("nan"),
            "mean_rtf": rtf_sum / rtf_n if rtf_n else float("nan"),
        }

    print("\n=== Aggregate (real speech) ===")
    for name, a in agg.items():
        print(f"{name}: WER-en={a['wer_en']:.4f}  CER-ja={a['cer_ja']:.4f}  mean_RTF={a['mean_rtf']:.4f}")

    write_markdown(results, systems, agg)
    print(f"\nWrote {DOCS_PATH}")


def write_markdown(results, systems, agg):
    lines = []
    lines.append("# ASR Accuracy Evaluation — Real Speech\n")
    n_ja = sum(1 for r in results if r["lang"] == "ja")
    n_en = sum(1 for r in results if r["lang"] == "en")
    total_audio = sum(r["audio_s"] for r in results)
    lines.append(
        f"Comparison of **Parakeet**, **SenseVoice**, **Omnilingual**, and "
        f"**faster-whisper large-v3-turbo** (INT8, CPU) on a small set of "
        f"**real, human-spoken** utterances ({n_ja} Japanese + {n_en} English, "
        f"{total_audio:.1f}s total audio, `testdata/eval_real/`) — as opposed "
        f"to the TTS-synthesized set in `testdata/eval/` (see `docs/eval/eval.md`).\n"
    )
    lines.append(
        "## Data sources\n\n"
        "- **Japanese**: [ReazonSpeech](https://huggingface.co/datasets/reazon-research/reazonspeech) "
        "test split, sourced from Japanese TV broadcast audio, via the openly-accessible mirror "
        "`japanese-asr/ja_asr.reazonspeech_test` on Hugging Face (fetched anonymously through the "
        "`datasets-server` rows API, no auth required).\n"
        "- **English**: [LibriSpeech](https://www.openslr.org/12/) dev-clean, read-aloud public-domain "
        "audiobook recordings, via `openslr/librispeech_asr` (`clean` config, `validation` split = dev-clean) "
        "on Hugging Face, same `datasets-server` rows API.\n"
        "- Utterances were filtered to roughly 3-9s duration (widened to 3-15s only if needed) to keep "
        "total audio short while still using real reference transcripts as-shipped by each corpus. "
        "Source FLAC audio was converted to 16kHz mono 16-bit PCM WAV via ffmpeg.\n"
    )
    lines.append(
        "- Japanese Parakeet model: `sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8` "
        "(NeMo CTC)\n"
        "- English/multilingual Parakeet model: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` "
        "(NeMo transducer)\n"
        "- SenseVoice: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` "
        "(non-autoregressive, use_itn=True, language forced per entry)\n"
        "- Omnilingual: `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, no lang hint)\n"
        "- Whisper: `large-v3-turbo`, `device=cpu`, `compute_type=int8`, `beam_size=1`, "
        "language forced per entry\n"
    )

    def esc(s):
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines.append("\n## Per-file results\n")
    sys_cols = "".join(f" {s.name} hyp | {s.name} metric | {s.name} RTF |" for s in systems)
    lines.append("| file | lang | ref |" + sys_cols)
    lines.append("|---|---|---|" + "---|" * (3 * len(systems)))
    for row in results:
        metric_name = "CER" if row["lang"] == "ja" else "WER"
        cells = ""
        for s in systems:
            r = row[s.name]
            cells += f" {esc(r['hyp'])} | {metric_name}={r['score']:.3f} | {r['rtf']:.3f} |"
        lines.append(f"| {row['wav']} | {row['lang']} | {esc(row['ref'])} |" + cells)

    lines.append("\n## Aggregate\n")
    lines.append("| system | WER (en, micro-avg) | CER (ja, micro-avg) | mean RTF |")
    lines.append("|---|---|---|---|")
    for name, a in agg.items():
        lines.append(f"| {name} | {a['wer_en']:.4f} | {a['cer_ja']:.4f} | {a['mean_rtf']:.4f} |")

    lines.append("\n## Comparison to the TTS baseline (`docs/eval/eval.md`)\n")
    lines.append(
        "TTS baseline (CER-ja% / WER-en%): Parakeet 9.1/13.7, SenseVoice 8.3/13.0, "
        "Omnilingual 10.3/15.1, Whisper-turbo 8.6/12.2. Real speech is expected to be "
        "harder across the board due to background noise/room acoustics, real prosody, "
        "disfluencies, and speaker variation absent from clean single-speaker TTS audio.\n"
    )

    lines.append("\n## Caveats\n")
    lines.append(
        "- **Small sample.** Only a handful of utterances per language were evaluated here "
        "(kept short deliberately so the full 4-system x 30-clip eval finishes in reasonable "
        "time on CPU). These numbers indicate relative system behavior, not statistically "
        "robust estimates.\n"
    )
    lines.append(
        "- **Reference transcript conventions differ by source.** LibriSpeech references are "
        "upper-cased with no punctuation (normalization lowercases and strips punctuation before "
        "scoring, so this does not bias WER). ReazonSpeech references come from broadcast-caption "
        "style Japanese text, which may include punctuation/formatting conventions not exactly "
        "matching what an ASR system would naturally output; normalization strips punctuation but "
        "cannot reconcile all such conventions.\n"
    )
    lines.append(
        "- **Domain mismatch remains.** ReazonSpeech audio is TV broadcast speech (news/variety "
        "style) and LibriSpeech is read audiobook narration — both are real, human speech, but "
        "still cleaner/more scripted than spontaneous conversational or noisy microphone input.\n"
    )

    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true", help="reuse existing testdata/eval_real/manifest.json")
    ap.add_argument("--skip-eval", action="store_true", help="only build the set, don't run the eval")
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
