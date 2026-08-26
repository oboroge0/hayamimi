"""FLEURS-based automatic quality check for scripts/translate_m2m.py targets.

FLEURS (google/fleurs) is n-way parallel: the same integer `id` refers to the
same underlying sentence across every language config in a given split, so
`ja_jp`'s `raw_transcription` for id N is a translation-equivalent reference
for target-language config `X`'s `raw_transcription` for the same id N.

This script:
  1. Pulls ja + target reference text for the `validation` split of FLEURS,
     matched by `id`.
  2. Runs each ja sentence through `TranslatorM2M(target_lang)`.
  3. Scores the output against the FLEURS reference with sacrebleu's chrF
     (character n-gram F-score -- more appropriate than BLEU for
     morphologically rich / non-whitespace-segmented targets like zh/ko/ja).
  4. Reports the target's chrF alongside baseline chrF for the two already-
     measured targets (zh, ko), as a point of comparison for whether a new
     target is "at the same level" as the existing VALIDATED_TARGETS.

Data source note: FLEURS' own parquet shards embed audio arrays, so a plain
`datasets-server` "rows" API request for some configs (e.g. es_419) exceeds
the server's 300MB per-request scan cap even for a handful of rows -- the
same "Parquet error: Scan size limit exceeded" issue documented in
scripts/make_realset_zhko.py for zh. Rather than depend on a config-specific
split workaround, this script reads the parquet files directly from the Hub
(not through datasets-server) with *column projection* (id + raw_transcription
only, no audio) via `fsspec` + `pyarrow`, which only fetches the byte ranges
for those two columns -- a few MB regardless of the file's total size. This
sidesteps the scan-size limit entirely and works uniformly for every target,
so this script does not need the `datasets` library. (`pyarrow` was added to
requirements-dev.txt; `fsspec`/`requests`/`aiohttp` were already present as
transitive deps of `huggingface_hub`.)

Usage:
    python scripts/eval_translate.py --targets zh,ko,es
    python scripts/eval_translate.py --targets es --n 80
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "refs%2Fconvert%2Fparquet"
FLEURS_SPLIT = "validation"

# FLEURS config id for each ISO 639-1 M2M-100 target code we know how to map.
# Add an entry here to evaluate a new target; anything else raises with a
# clear message rather than silently skipping.
FLEURS_CONFIG_BY_LANG = {
    "ja": "ja_jp",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
    "es": "es_419",
}

DEFAULT_N = 50
DEFAULT_SEED = 0


def _parquet_url(config: str, split: str) -> str:
    return (
        f"https://huggingface.co/datasets/{FLEURS_REPO}/resolve/"
        f"{FLEURS_REVISION}/{config}/{split}/0000.parquet"
    )


def load_fleurs_texts(config: str, split: str = FLEURS_SPLIT) -> "dict[int, str]":
    """id -> raw_transcription for one FLEURS config/split, via a column-projected
    remote parquet read (no audio bytes fetched)."""
    import fsspec
    import pyarrow.parquet as pq

    url = _parquet_url(config, split)
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read(columns=["id", "raw_transcription"])
    ids = table.column("id").to_pylist()
    texts = table.column("raw_transcription").to_pylist()
    return {i: t for i, t in zip(ids, texts) if t and t.strip()}


def build_pairs(target_lang: str, n: int, seed: int = DEFAULT_SEED):
    """Return up to n (ja_text, ref_text) pairs for target_lang, matched by FLEURS id."""
    import random

    if target_lang not in FLEURS_CONFIG_BY_LANG:
        raise ValueError(
            f"No FLEURS config mapping for target_lang={target_lang!r}. "
            f"Known: {sorted(FLEURS_CONFIG_BY_LANG)}. Add one to "
            f"FLEURS_CONFIG_BY_LANG in scripts/eval_translate.py."
        )

    ja_texts = load_fleurs_texts(FLEURS_CONFIG_BY_LANG["ja"])
    tgt_texts = load_fleurs_texts(FLEURS_CONFIG_BY_LANG[target_lang])

    common_ids = sorted(set(ja_texts) & set(tgt_texts))
    if not common_ids:
        raise RuntimeError(
            f"No common FLEURS ids between ja_jp and {FLEURS_CONFIG_BY_LANG[target_lang]} "
            f"on split={FLEURS_SPLIT!r} -- cannot build an eval set for {target_lang!r}."
        )

    rng = random.Random(seed)
    rng.shuffle(common_ids)
    chosen = common_ids[:n]
    return [(ja_texts[i], tgt_texts[i]) for i in chosen]


def run_eval(target_lang: str, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> dict:
    """Translate n ja->target_lang FLEURS sentences and score against reference chrF."""
    import sacrebleu

    from translate_m2m import TranslatorM2M

    pairs = build_pairs(target_lang, n, seed)
    print(f"[{target_lang}] {len(pairs)} ja/{target_lang} FLEURS pairs "
          f"(split={FLEURS_SPLIT}, seed={seed})", file=sys.stderr)

    t0 = time.perf_counter()
    translator = TranslatorM2M(target_lang)
    load_s = time.perf_counter() - t0

    hyps = []
    refs = []
    per_sentence = []
    for ja_text, ref_text in pairs:
        t0 = time.perf_counter()
        hyp = translator.translate(ja_text)
        dt_ms = (time.perf_counter() - t0) * 1000
        chrf = sacrebleu.sentence_chrf(hyp, [ref_text]).score
        hyps.append(hyp)
        refs.append(ref_text)
        per_sentence.append({
            "ja": ja_text, "ref": ref_text, "hyp": hyp,
            "chrf": chrf, "ms": dt_ms,
        })

    corpus_chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    mean_ms = sum(r["ms"] for r in per_sentence) / len(per_sentence)

    return {
        "target_lang": target_lang,
        "n": len(pairs),
        "load_s": load_s,
        "corpus_chrf": corpus_chrf,
        "mean_ms": mean_ms,
        "per_sentence": per_sentence,
    }


def print_report(result: dict) -> None:
    lang = result["target_lang"]
    print(f"\n=== ja -> {lang}: chrF={result['corpus_chrf']:.2f} "
          f"(n={result['n']}, mean {result['mean_ms']:.0f} ms/line, "
          f"model load {result['load_s']:.2f}s) ===")
    for r in result["per_sentence"][:5]:
        print(f"  ja:  {r['ja']}")
        print(f"  ref: {r['ref']}")
        print(f"  hyp: {r['hyp']}  [chrF={r['chrf']:.1f}]")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default="zh,ko", metavar="LANGS",
                     help="comma-separated M2M-100 target codes to evaluate (default zh,ko)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, metavar="N",
                     help=f"max sentences per target (default {DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="RNG seed for sampling which FLEURS ids to use")
    args = ap.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    results = {}
    for lang in targets:
        results[lang] = run_eval(lang, n=args.n, seed=args.seed)
        print_report(results[lang])

    print("\n=== Summary ===")
    for lang, r in results.items():
        print(f"{lang}: chrF={r['corpus_chrf']:.2f}  n={r['n']}  mean={r['mean_ms']:.0f} ms/line")

    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
