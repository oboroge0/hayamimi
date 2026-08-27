#!/usr/bin/env python
"""Download all pretrained models hayamimi needs into models/.

Idempotent: any model whose target already exists is skipped, so re-running
this after an interrupted download only fetches what's missing.

Two model sets:
  --minimal   ja/en core only: ReazonSpeech (ja), whisper-tiny (LID + en
              fallback via VAD), Silero VAD, Japanese punctuation. ~1.1 GB.
  (default)   Everything the runtime routing in asr_engine.py can reach:
              minimal + zh/ko/yue/multilingual-EU/1600-language-fallback ASR,
              speaker embeddings, and ja->en/zh/ko plus en->ja translation.
              ~3.1 GB.

Add --eval-baselines to additionally fetch two extra models that are only
used as comparison baselines by scripts/eval_accuracy.py and
scripts/make_realset_zhko.py (not used by the live pipeline). ~1 GB more.

All models are pulled from their original publishers (via k2-fsa/sherpa-onnx's
GitHub release mirrors, or directly from Hugging Face). See
THIRD_PARTY_NOTICES.md for what you're agreeing to by downloading each one --
in particular, the ja->en translation model is CC BY-SA 4.0 (share-alike),
not permissive like the rest.
"""
import argparse
import io
import os
import sys
import tarfile
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

GITHUB_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASR_TAG = "asr-models"
# Yes, "recongition" -- that's the actual (misspelled) tag name upstream.
SPEAKER_TAG = "speaker-recongition-models"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _download_to(url: str, dest: str) -> None:
    """Stream url to dest with a progress print. Overwrites if dest exists."""
    req = urllib.request.Request(url, headers={"User-Agent": "hayamimi-download-models/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        tmp = dest + ".part"
        read = 0
        chunk = 1 << 20
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read += len(buf)
                if total:
                    pct = 100 * read / total
                    print(f"\r  {os.path.basename(dest)}: {pct:5.1f}% ({_human(read)}/{_human(total)})",
                          end="", flush=True)
                else:
                    print(f"\r  {os.path.basename(dest)}: {_human(read)}", end="", flush=True)
        print()
        os.replace(tmp, dest)


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hayamimi-download-models/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def download_file(url: str, dest_path: str, label: str) -> None:
    """Single-file model (e.g. silero_vad.onnx, campplus_sv.onnx)."""
    if os.path.exists(dest_path):
        print(f"[skip] {label} (already present: {dest_path})")
        return
    print(f"[get ] {label}")
    _download_to(url, dest_path)


def download_and_extract_tarbz2(url: str, dest_dir: str, label: str) -> None:
    """sherpa-onnx release tarball: extracts to a top-level dir matching the
    tarball's basename; we re-home that under models/<dest_dir_name>."""
    target = os.path.join(MODELS_DIR, dest_dir)
    if os.path.isdir(target):
        print(f"[skip] {label} (already present: {target})")
        return
    print(f"[get ] {label}")
    tmp = os.path.join(MODELS_DIR, f".{dest_dir}.tar.bz2.part")
    os.makedirs(MODELS_DIR, exist_ok=True)
    _download_to(url, tmp)
    print(f"  extracting -> {target}")
    with tarfile.open(tmp, "r:bz2") as tf:
        # sherpa-onnx tarballs have a single top-level directory; extract it
        # then rename to our canonical dest_dir name (usually the same, but
        # e.g. omnilingual's release name differs from our shorter dir name).
        members = tf.getmembers()
        top = members[0].name.split("/")[0]
        tf.extractall(MODELS_DIR)
    extracted = os.path.join(MODELS_DIR, top)
    if extracted != target:
        os.replace(extracted, target)
    os.remove(tmp)


def download_hf_repo(repo: str, dest_dir: str, label: str, ignore_patterns=None) -> None:
    """Snapshot-download a full Hugging Face repo (used for the Mojicast
    translation/punctuation models, which are distributed as small repos)."""
    target = os.path.join(MODELS_DIR, dest_dir)
    if os.path.isdir(target):
        print(f"[skip] {label} (already present: {target})")
        return
    print(f"[get ] {label}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ERROR: huggingface_hub not installed. `pip install huggingface_hub` "
              "or `pip install -r requirements.txt` and re-run.", file=sys.stderr)
        raise
    snapshot_download(repo_id=repo, local_dir=target, ignore_patterns=ignore_patterns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minimal", action="store_true",
                     help="only download the ja/en core (~1.1GB): ReazonSpeech, whisper-tiny, "
                          "Silero VAD, Japanese punctuation. Skips zh/ko/yue/EU/omnilingual "
                          "ASR, speaker embeddings, and translation models.")
    ap.add_argument("--eval-baselines", action="store_true",
                     help="also download 2 extra models (~1GB) used only by scripts/eval_accuracy.py "
                          "and scripts/make_realset_zhko.py as comparison baselines -- not needed "
                          "to run realtime_transcribe.py.")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    total_gb = "~1.1GB" if args.minimal else ("~4.1GB" if args.eval_baselines else "~3.1GB")
    print(f"hayamimi model download: this will fetch {total_gb} into {MODELS_DIR}")
    print("(see THIRD_PARTY_NOTICES.md for each model's license)\n")

    # --- ja/en core (--minimal stops after this block) ---
    download_and_extract_tarbz2(
        f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2",
        "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17",
        "ReazonSpeech k2 Zipformer (ja, primary ASR route)")

    download_and_extract_tarbz2(
        f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-whisper-tiny.tar.bz2",
        "sherpa-onnx-whisper-tiny",
        "whisper-tiny (spoken-language ID)")

    download_file(
        f"{GITHUB_RELEASES}/{ASR_TAG}/silero_vad.onnx",
        os.path.join(MODELS_DIR, "silero_vad.onnx"),
        "Silero VAD")

    # Punctuation model: only the fp32 export. The int8 export on this HF repo
    # (punct_bert.int8.onnx) was found non-functional (near-constant logits)
    # on onnxruntime 1.29 CPU EP during development -- see docs/PUNCT_JA.md.
    # Skipped here to save ~90MB and avoid confusion; punct_ja.py only ever
    # loads punct_bert.onnx.
    download_hf_repo(
        "ishiki-emo/mojicast-punct-onnx",
        "mojicast-punct-onnx",
        "Japanese punctuation restoration (Mojicast/tohoku-nlp, fp32 only)",
        ignore_patterns=["*.int8.onnx"])

    if args.minimal:
        print("\n--minimal done. zh/ko/yue/EU/omnilingual ASR, speaker labels, and "
              "translation are unavailable until you re-run without --minimal.")
        return

    # --- full multilingual routing ---
    download_and_extract_tarbz2(
        f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-paraformer-zh-int8-2025-10-07.tar.bz2",
        "sherpa-onnx-paraformer-zh-int8-2025-10-07",
        "Paraformer-zh (zh ASR)")

    # IMPORTANT: must be the 2024-07-17 export. The newer 2025-09-09 export
    # was found broken during development (see docs/BENCHMARKS.md) -- do not
    # substitute it.
    download_and_extract_tarbz2(
        f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        "SenseVoice small (ko/yue ASR) -- 2024-07-17 export, NOT 2025-09-09")

    download_and_extract_tarbz2(
        f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        "Parakeet TDT 0.6B v3 (en + 24 EU languages)")

    # Fetch the two runtime files directly. The GitHub release tarball is
    # ~1GB and its packaging has changed over time; direct publisher-mirror
    # files are both smaller and make interrupted/partial installs repairable.
    omni_repo = "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
    omni_dir = os.path.join(MODELS_DIR, "omnilingual-300m-ctc-int8")
    for filename in ("model.int8.onnx", "tokens.txt"):
        download_file(
            HF_RESOLVE.format(repo=omni_repo, path=filename),
            os.path.join(omni_dir, filename),
            f"Meta Omnilingual ASR 300M CTC {filename}",
        )

    download_file(
        f"{GITHUB_RELEASES}/{SPEAKER_TAG}/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        os.path.join(MODELS_DIR, "campplus_sv.onnx"),
        "CAM++ speaker embedding (--speakers)")

    download_hf_repo(
        "ishiki-emo/mojicast-m2m100-ct2",
        "mojicast-m2m100-ct2",
        "M2M-100 418M CTranslate2 (ja->zh/ko and en->ja translation, MIT)")

    download_hf_repo(
        "ishiki-emo/mojicast-fugumt-ja-en-ct2",
        "mojicast-fugumt-ja-en-ct2",
        "FuguMT CTranslate2 (ja->en translation, CC BY-SA 4.0 -- see THIRD_PARTY_NOTICES.md)")

    if args.eval_baselines:
        download_and_extract_tarbz2(
            f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8.tar.bz2",
            "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
            "Parakeet tdt_ctc 0.6B ja (eval baseline only)")

        download_and_extract_tarbz2(
            f"{GITHUB_RELEASES}/{ASR_TAG}/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2",
            "sherpa-onnx-zipformer-korean-2024-06-24",
            "Zipformer Korean (eval baseline only)")

    print("\nDone. Run `python scripts/realtime_transcribe.py --wav testdata/ja_test.wav` to smoke-test.")


if __name__ == "__main__":
    main()
