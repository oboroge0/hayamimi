"""Build the unified FLEURS test-split benchmark set (5 languages x 100 clips).

Data source: google/fleurs, TEST split, via the Hugging Face parquet-conversion
API (the datasets-server /rows API returns HTTP 500 for this dataset's test
splits on some configs, so this script bypasses it and reads the auto-
converted parquet shard directly with fsspec + pyarrow, using column
projection to keep memory/bandwidth bounded).

    https://huggingface.co/api/datasets/google/fleurs/parquet/<config>/test/0.parquet

Each of the 5 configs used here (ja_jp, en_us, cmn_hans_cn, ko_kr,
yue_hant_hk) fits in a single shard (0.parquet, 382-945 rows), so no
multi-shard paging is needed.

Selection rule (documented, deterministic): read the full parquet table's
`id`, `raw_transcription`, and `audio` columns, sort rows by the FLEURS `id`
field (stable sort -- ties, which do occur since a handful of ids repeat
across distinct recordings/speakers in the test split, are broken by the
row's original position in the parquet file), then take the FIRST 100 rows
of that sorted order. Rows with empty transcription or that fail to decode
are skipped and backfilled from the next row in sorted order, so exactly 100
clips are kept per language whenever the shard has enough usable rows.

Audio in the parquet is already 16kHz mono (verified per-config below); it is
written out as-is via soundfile, no resampling needed.

Output layout (matches eval_common.load_manifest's expected shape):
    testdata/fleurs_bench/<lang>/manifest.json  -- [{"wav", "ref", "id"}, ...]
    testdata/fleurs_bench/<lang>/<lang>_XXX.wav

Idempotent/resumable: if a language's manifest.json already has N_PER_LANG
entries and every referenced wav file exists on disk, that language is
skipped entirely (no network access). Use --force to rebuild anyway.

Usage:
    python scripts/make_fleursset.py --lang ja
    python scripts/make_fleursset.py --lang all
"""
import argparse
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "testdata", "fleurs_bench")

N_PER_LANG = 100

CONFIGS = {
    "ja": "ja_jp",
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
    "yue": "yue_hant_hk",
}

PARQUET_URL_TMPL = "https://huggingface.co/api/datasets/google/fleurs/parquet/{config}/test/0.parquet"


def lang_dir(lang):
    return os.path.join(OUT_ROOT, lang)


def manifest_path(lang):
    return os.path.join(lang_dir(lang), "manifest.json")


def already_built(lang):
    mpath = manifest_path(lang)
    if not os.path.exists(mpath):
        return False
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    if len(manifest) < N_PER_LANG:
        return False
    d = lang_dir(lang)
    return all(os.path.exists(os.path.join(d, e["wav"])) for e in manifest)


def fetch_table(config):
    import urllib.request

    import pyarrow.parquet as pq

    url = PARQUET_URL_TMPL.format(config=config)
    cache_path = os.path.join(OUT_ROOT, f".cache_{config}.parquet")
    os.makedirs(OUT_ROOT, exist_ok=True)
    if not os.path.exists(cache_path):
        print(f"  downloading {url} -> {cache_path} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "hayamimi-fleurs-bench/1.0"})
        # Large shards (e.g. yue_hant_hk, ~800+ rows with embedded audio) can
        # exceed fsspec's default async-http timeout when streamed via
        # pyarrow's dataset scanner; a plain blocking urlretrieve-style fetch
        # to a local file is more robust and also gives us a resumable cache.
        with urllib.request.urlopen(req, timeout=300) as resp, open(cache_path + ".tmp", "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(cache_path + ".tmp", cache_path)
    else:
        print(f"  reusing cached {cache_path}")
    tbl = pq.read_table(cache_path, columns=["id", "raw_transcription", "audio"])
    print(f"  got {tbl.num_rows} rows")
    return tbl


def build_lang(lang, force=False):
    if not force and already_built(lang):
        print(f"[{lang}] already built ({N_PER_LANG} clips present), skipping.")
        return

    config = CONFIGS[lang]
    os.makedirs(lang_dir(lang), exist_ok=True)

    tbl = fetch_table(config)
    ids = tbl.column("id").to_pylist()
    texts = tbl.column("raw_transcription").to_pylist()
    audio_col = tbl.column("audio").to_pylist()

    # stable sort by id; ties keep original (parquet row) order
    order = sorted(range(len(ids)), key=lambda i: ids[i])

    import soundfile as sf

    kept = []
    for i in order:
        if len(kept) >= N_PER_LANG:
            break
        text = (texts[i] or "").strip()
        if not text:
            continue
        audio = audio_col[i]
        raw_bytes = audio.get("bytes") if audio else None
        if not raw_bytes:
            continue
        try:
            data, sr = sf.read(io.BytesIO(raw_bytes))
        except Exception as e:
            print(f"  [{lang}] id={ids[i]} decode failed: {e}")
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            print(f"  [{lang}] id={ids[i]} unexpected sample rate {sr}, skipping")
            continue

        n = len(kept) + 1
        wav_name = f"{lang}_{n:03d}.wav"
        wav_path = os.path.join(lang_dir(lang), wav_name)
        sf.write(wav_path, data, sr, subtype="PCM_16")
        kept.append({"wav": wav_name, "ref": text, "id": ids[i]})

    if len(kept) < N_PER_LANG:
        print(f"WARNING: [{lang}] only found {len(kept)}/{N_PER_LANG} usable clips")

    with open(manifest_path(lang), "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"[{lang}] wrote {manifest_path(lang)} with {len(kept)} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=list(CONFIGS.keys()) + ["all"])
    ap.add_argument("--force", action="store_true", help="rebuild even if already complete")
    args = ap.parse_args()

    langs = list(CONFIGS.keys()) if args.lang == "all" else [args.lang]
    for lang in langs:
        build_lang(lang, force=args.force)


if __name__ == "__main__":
    main()
