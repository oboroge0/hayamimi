"""Quantize the ja punctuation-restoration model (Mojicast BERT-char, see
docs/PUNCT_JA.md) to INT8 and measure the accuracy / size tradeoff, as a
follow-up mobile-sizing step to scripts/quantize_reazonspeech.py.

Background: the INT8 file shipped upstream on HF (`punct_bert.int8.onnx`)
was found non-functional in this environment (near-constant, token-
independent logits -- see docs/PUNCT_JA.md, "Important deviation"). This
script instead requantizes the verified-correct fp32 file ourselves with
`onnxruntime.quantization.quantize_dynamic`, the same technique that worked
for the ReazonSpeech ASR encoder (docs/MOBILE.md), and checks whether our
own INT8 export actually restores punctuation (unlike the upstream one).

Produces:
    models/mojicast-punct-onnx/quantized_ort/punct_bert.int8.onnx

Then evaluates fp32 vs int8 on the 15 ja clips in testdata/eval_real
(reusing their punctuated reference transcripts as ground truth -- no audio
is used here, just text): strip 、/。/？ from each ref to build unpunctuated
input, run PunctuatorJa.restore() with each model, and score:
  - punctuation-position F1 (、/。/？, exact mark type, aligned by base
    character position since restore() never alters non-punctuation chars)
  - whole-string CER *including* punctuation (Levenshtein / len(ref)),
    i.e. how close the fully-restored text is to the original transcript.

Usage:
    python scripts/quantize_punct.py            # quantize + evaluate
    python scripts/quantize_punct.py --skip-quantize  # eval only, reusing
        a previously-produced quantized_ort/punct_bert.int8.onnx
"""
import argparse
import json
import os
import sys
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from punct_ja import PunctuatorJa  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "mojicast-punct-onnx")
FP32_PATH = os.path.join(MODEL_DIR, "punct_bert.onnx")
QUANT_DIR = os.path.join(MODEL_DIR, "quantized_ort")
INT8_NAME = "punct_bert.int8.onnx"
INT8_PATH = os.path.join(QUANT_DIR, INT8_NAME)

# Additional candidate variants (see docs/MOBILE.md, "Punctuation model
# INT8" -- dynamic INT8 failed the large-scale FLEURS re-verification, these
# are the two follow-up candidates evaluated with the same harness).
VARIANT_FILENAMES = {
    "int8": INT8_NAME,  # dynamic INT8 (existing, kept for comparison)
    "fp16": "punct_bert.fp16.onnx",
    "int8-static": "punct_bert.int8-static.onnx",
}
PREPROC_NAME = "punct_bert.preproc.onnx"  # shape-inferred input to quantize_static

EVAL_DIR = os.path.join(ROOT, "testdata", "eval_real")
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.json")

TARGET_MARKS = ("、", "。", "？")

# FLEURS ja config/splits used for the larger re-verification eval set (see
# docs/MOBILE.md, "Punctuation model INT8" -- the n=15 testdata/eval_real
# sample was only ever a provisional check). FLEURS' ja_jp raw_transcription
# already carries natural 、/。/？ punctuation (unlike the ASR-oriented
# eval_real refs, FLEURS text wasn't written for this repo, so no filtering
# assumption should be taken on faith -- see build_fleurs_refs, which checks).
FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "refs%2Fconvert%2Fparquet"
FLEURS_JA_CONFIG = "ja_jp"
FLEURS_SPLITS = ("validation", "test")
FLEURS_DEFAULT_N = 250
FLEURS_DEFAULT_SEED = 0
FLEURS_MAX_CHARS = 450  # margin under the model's ~500-char practical limit


def _fleurs_parquet_url(config: str, split: str) -> str:
    return (
        f"https://huggingface.co/datasets/{FLEURS_REPO}/resolve/"
        f"{FLEURS_REVISION}/{config}/{split}/0000.parquet"
    )


def _load_fleurs_split(split: str) -> "dict[int, str]":
    """id -> raw_transcription for one FLEURS ja split, via a column-projected
    remote parquet read (no audio bytes fetched) -- same technique as
    scripts/eval_translate.py::load_fleurs_texts."""
    import fsspec
    import pyarrow.parquet as pq

    url = _fleurs_parquet_url(FLEURS_JA_CONFIG, split)
    with fsspec.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read(columns=["id", "raw_transcription"])
    ids = table.column("id").to_pylist()
    texts = table.column("raw_transcription").to_pylist()
    return {i: t for i, t in zip(ids, texts) if t and t.strip()}


def build_fleurs_refs(n: int = FLEURS_DEFAULT_N, seed: int = FLEURS_DEFAULT_SEED,
                       exclude: "set[str] | None" = None):
    """Build a length-diverse set of up to n punctuated ja reference
    sentences from FLEURS validation+test (raw_transcription already carries
    natural 、/。/？ punctuation). Filters to sentences that actually contain
    at least one target mark and fit under FLEURS_MAX_CHARS, then samples
    with length-decile stratification so the set isn't skewed toward the
    (more numerous) short sentences.

    `exclude`, if given, is a set of (already-stripped) sentence strings
    dropped from the candidate pool before sampling -- used to keep the
    int8-static calibration set disjoint from whatever sentences are used
    for accuracy evaluation (no calibration-on-the-eval-set leakage).
    """
    import random

    pool: dict[int, str] = {}
    for split in FLEURS_SPLITS:
        pool.update(_load_fleurs_split(split))

    seen_text = set()
    candidates = []
    for text in pool.values():
        norm = text.strip()
        if not norm or norm in seen_text:
            continue
        if not any(m in norm for m in TARGET_MARKS):
            continue
        if len(norm) > FLEURS_MAX_CHARS:
            continue
        if exclude and norm in exclude:
            continue
        seen_text.add(norm)
        candidates.append(norm)

    if not candidates:
        raise RuntimeError("No usable FLEURS ja sentences found (empty pool or all filtered out).")

    candidates.sort(key=len)
    n = min(n, len(candidates))

    # Stratify by length decile so short/medium/long sentences are all
    # represented roughly proportionally to their share of the pool, rather
    # than a plain random sample happening to skew toward the far more
    # numerous short sentences.
    num_buckets = 10
    buckets = [[] for _ in range(num_buckets)]
    for idx, text in enumerate(candidates):
        b = min(idx * num_buckets // len(candidates), num_buckets - 1)
        buckets[b].append(text)

    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)

    base = n // num_buckets
    remainder = n % num_buckets
    chosen = []
    leftover = []
    for i, b in enumerate(buckets):
        take = base + (1 if i < remainder else 0)
        chosen.extend(b[:take])
        leftover.extend(b[take:])
    # if some buckets were smaller than their quota (only possible with a
    # very small/uneven pool), top up from whatever's left over elsewhere
    if len(chosen) < n:
        rng.shuffle(leftover)
        chosen.extend(leftover[: n - len(chosen)])

    rng.shuffle(chosen)
    return chosen[:n]


def do_quantize_dynamic_int8():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(QUANT_DIR, exist_ok=True)
    print(f"[quantize int8 dynamic] {FP32_PATH} -> {INT8_PATH}")
    t0 = time.time()
    quantize_dynamic(
        model_input=FP32_PATH,
        model_output=INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    print(f"  done in {time.time() - t0:.1f}s")


def do_quantize_fp16():
    """onnxconverter-common float16 conversion (pip package added for this
    experiment; not yet in requirements.txt -- out of this task's scope,
    see report). keep_io_types=True keeps the graph's input_ids/
    attention_mask/logits IO as their original dtypes (int64 in, float32
    out) so punct_ja.py's PunctuatorJa needs no changes to consume this
    file -- only the internal weights/activations run in float16."""
    import onnx
    from onnxconverter_common import float16

    os.makedirs(QUANT_DIR, exist_ok=True)
    out_path = os.path.join(QUANT_DIR, VARIANT_FILENAMES["fp16"])
    print(f"[quantize fp16] {FP32_PATH} -> {out_path}")
    t0 = time.time()
    model = onnx.load(FP32_PATH)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, out_path)
    print(f"  done in {time.time() - t0:.1f}s")


class _PunctCalibrationReader:
    """onnxruntime.quantization CalibrationDataReader for punct_bert:
    feeds (input_ids, attention_mask) built the same way
    PunctuatorJa.restore() builds them, over a fixed list of calibration
    sentences (already-punctuated FLEURS text; punctuation is stripped
    here the same way strip_marks() does for eval, since calibration must
    see the same *unpunctuated* input distribution the model sees at
    inference time)."""

    def __init__(self, texts, punctuator):
        self._punctuator = punctuator
        self._iter = iter(texts)

    def get_next(self):
        text = next(self._iter, None)
        if text is None:
            return None
        stripped, _ = strip_marks(text)
        chars = self._punctuator._tokenize_chars(stripped)[: self._punctuator.max_chars]
        input_ids = ([self._punctuator.cls_id]
                     + [self._punctuator._char_id(c) for c in chars]
                     + [self._punctuator.sep_id])
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
        }

    def rewind(self):
        pass


def do_quantize_int8_static(calibration_texts):
    """Static (QDQ, calibrated) INT8 quantization via
    onnxruntime.quantization.quantize_static -- unlike quantize_dynamic
    (weights-only, activations computed in fp32 at runtime), this also
    quantizes activations using ranges observed over `calibration_texts`,
    which is the natural next experiment flagged in docs/MOBILE.md after
    dynamic INT8 failed the FLEURS re-verification (recall collapse)."""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    os.makedirs(QUANT_DIR, exist_ok=True)
    preproc_path = os.path.join(QUANT_DIR, PREPROC_NAME)
    out_path = os.path.join(QUANT_DIR, VARIANT_FILENAMES["int8-static"])

    print(f"[quantize int8-static] shape-inference preprocess {FP32_PATH} -> {preproc_path}")
    try:
        quant_pre_process(FP32_PATH, preproc_path)
    except Exception as e:
        # Symbolic shape inference chokes on this graph's position-embedding
        # Min(512, seq_len) broadcast pattern ("Incomplete symbolic shape
        # inference"). Fall back to the non-symbolic preprocessing path
        # (still does basic optimization + ONNX shape inference, just skips
        # the symbolic pass) -- quantize_static works fine off of this.
        print(f"  symbolic shape inference failed ({e}); retrying with "
              "skip_symbolic_shape=True")
        quant_pre_process(FP32_PATH, preproc_path, skip_symbolic_shape=True)

    # Tokenizer/vocab pipeline only -- loads an onnxruntime session on the
    # fp32 graph too, but we never call .restore()/.session.run() on it;
    # it's just reused for its _tokenize_chars/_char_id/cls_id/sep_id.
    tokenizer = PunctuatorJa(model_dir=MODEL_DIR, onnx_filename="punct_bert.onnx")
    reader = _PunctCalibrationReader(calibration_texts, tokenizer)

    print(f"[quantize int8-static] calibrating on {len(calibration_texts)} sentences...")
    t0 = time.time()
    quantize_static(
        model_input=preproc_path,
        model_output=out_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=False,
    )
    print(f"  done in {time.time() - t0:.1f}s")


def do_quantize(variant: str, calibration_texts=None):
    fp32_size = os.path.getsize(FP32_PATH)
    if variant == "int8":
        do_quantize_dynamic_int8()
    elif variant == "fp16":
        do_quantize_fp16()
    elif variant == "int8-static":
        do_quantize_int8_static(calibration_texts or [])
    else:
        raise ValueError(f"unknown variant: {variant}")

    out_path = os.path.join(QUANT_DIR, VARIANT_FILENAMES[variant])
    out_size = os.path.getsize(out_path)
    print(f"  {out_size / 1e6:.1f} MB (fp32 was {fp32_size / 1e6:.1f} MB)")
    return {"fp32": fp32_size, variant: out_size}


def existing_sizes(variant: str):
    out_path = os.path.join(QUANT_DIR, VARIANT_FILENAMES[variant])
    return {"fp32": os.path.getsize(FP32_PATH), variant: os.path.getsize(out_path)}


# ---------------------------------------------------------------------------
# Text alignment helpers
# ---------------------------------------------------------------------------

def strip_marks(text: str):
    """NFKC-normalize, then remove TARGET_MARKS, returning:
      - stripped: the unpunctuated text (model input)
      - marks_after: list, same length as stripped, marks_after[i] = the
        mark (or "") that immediately followed stripped[i] in the original
        text (only the first such mark; stacked marks are rare/absent here).
    """
    norm = unicodedata.normalize("NFKC", text)
    stripped = []
    marks_after = []
    for ch in norm:
        if ch in TARGET_MARKS:
            if stripped:
                if not marks_after[-1]:
                    marks_after[-1] = ch
            continue
        stripped.append(ch)
        marks_after.append("")
    return "".join(stripped), marks_after


def marks_from_restored(restored: str, base_len: int):
    """Parse a restore()-produced string back into per-base-char marks,
    assuming (as PunctuatorJa guarantees) every non-punctuation character
    from the input is preserved unchanged and in order. Returns a list of
    length base_len, same convention as strip_marks's marks_after."""
    marks_after = []
    for ch in restored:
        if ch in TARGET_MARKS:
            if marks_after:
                if not marks_after[-1]:
                    marks_after[-1] = ch
            continue
        marks_after.append("")
    # defensive: pad/truncate to base_len in case of an edge-case mismatch
    if len(marks_after) < base_len:
        marks_after += [""] * (base_len - len(marks_after))
    return marks_after[:base_len]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def punct_cer(ref: str, hyp: str):
    r = unicodedata.normalize("NFKC", ref)
    h = unicodedata.normalize("NFKC", hyp)
    dist = levenshtein(r, h)
    rate = dist / len(r) if r else 0.0
    return rate, dist, len(r)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_eval_real_refs():
    """ja reference transcripts (already punctuated) from the original
    15-clip testdata/eval_real/manifest.json sample."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return [(e["wav"], e["ref"]) for e in manifest if e["lang"] == "ja"]


def evaluate(onnx_filename: str, label: str, refs, latency: bool = False):
    """refs: list of (id, ref_text) pairs, ref_text already punctuated."""
    punctuator = PunctuatorJa(model_dir=MODEL_DIR, onnx_filename=onnx_filename)

    tp = fp = fn = 0
    total_dist = 0
    total_denom = 0
    rows = []
    latencies = []
    for ref_id, ref in refs:
        stripped, ref_marks = strip_marks(ref)
        if not stripped:
            continue
        t0 = time.time()
        hyp = punctuator.restore(stripped)
        if latency:
            latencies.append(time.time() - t0)
        hyp_marks = marks_from_restored(hyp, len(stripped))

        for rm, hm in zip(ref_marks, hyp_marks):
            if rm and hm and rm == hm:
                tp += 1
            elif rm and (not hm or hm != rm):
                fn += 1
                if hm:
                    fp += 1
            elif hm and not rm:
                fp += 1

        rate, dist, denom = punct_cer(ref, hyp)
        total_dist += dist
        total_denom += denom
        rows.append({"wav": ref_id, "ref": ref, "hyp": hyp, "cer": rate})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    micro_cer = total_dist / total_denom if total_denom else float("nan")
    mean_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else None

    print(f"  [{label}] n={len(rows)} P={precision:.4f} R={recall:.4f} "
          f"F1={f1:.4f} CER={micro_cer:.4f}"
          + (f" mean_latency={mean_latency_ms:.2f}ms" if mean_latency_ms is not None else ""))

    return {
        "label": label,
        "n": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": micro_cer,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_latency_ms": mean_latency_ms,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quantize", action="store_true")
    ap.add_argument("--variant", choices=["int8", "fp16", "int8-static"], default="int8",
                     help="quantization variant to produce/evaluate against fp32 "
                          "(default int8 = the original dynamic-INT8 recipe, which "
                          "already FAILED the fleurs n=250 re-verification -- see "
                          "docs/MOBILE.md. fp16 and int8-static are the follow-up "
                          "candidates)")
    ap.add_argument("--source", choices=["eval_real", "fleurs"], default="eval_real",
                     help="reference set to evaluate against (default eval_real, the "
                          "original 15-clip sample; fleurs is the larger re-verification set)")
    ap.add_argument("--n", type=int, default=FLEURS_DEFAULT_N,
                     help=f"max sentences to use from --source=fleurs (default {FLEURS_DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=FLEURS_DEFAULT_SEED,
                     help="RNG seed for fleurs length-stratified sampling")
    ap.add_argument("--calib-n", type=int, default=100,
                     help="number of FLEURS sentences for int8-static calibration "
                          "(default 100; ignored for other variants)")
    ap.add_argument("--calib-seed", type=int, default=777,
                     help="RNG seed for the calibration sample -- deliberately "
                          "different from --seed, and the eval sentences (when "
                          "--source=fleurs) are excluded from the calibration pool, "
                          "so calibration never sees the sentences used to score it")
    ap.add_argument("--latency", action="store_true",
                     help="also measure mean per-call restore() latency")
    args = ap.parse_args()

    if args.source == "fleurs":
        print(f"\n[source] building FLEURS ja eval set (n={args.n}, seed={args.seed})...")
        texts = build_fleurs_refs(n=args.n, seed=args.seed)
        refs = [(f"fleurs_{i:04d}", t) for i, t in enumerate(texts)]
        print(f"[source] {len(refs)} FLEURS ja sentences "
              f"(chars: min={min(len(t) for t in texts)}, "
              f"median={sorted(len(t) for t in texts)[len(texts) // 2]}, "
              f"max={max(len(t) for t in texts)})")
    else:
        refs = load_eval_real_refs()
        print(f"\n[source] {len(refs)} testdata/eval_real ja clips")

    calibration_texts = None
    if args.variant == "int8-static" and not args.skip_quantize:
        eval_texts = {t for _, t in refs} if args.source == "fleurs" else set()
        print(f"\n[calibration] building FLEURS calibration set "
              f"(n={args.calib_n}, seed={args.calib_seed}, "
              f"excluding {len(eval_texts)} eval sentences)...")
        calibration_texts = build_fleurs_refs(
            n=args.calib_n, seed=args.calib_seed, exclude=eval_texts
        )
        print(f"[calibration] {len(calibration_texts)} FLEURS ja sentences "
              "(disjoint from the eval set)")

    sizes = (existing_sizes(args.variant) if args.skip_quantize
              else do_quantize(args.variant, calibration_texts))

    print("\n=== sizes (bytes) ===")
    for k, v in sizes.items():
        print(f"  {k}: {v} ({v / 1e6:.1f} MB)")

    results = {}
    for filename, label in [("punct_bert.onnx", "fp32"),
                             (os.path.join("quantized_ort", VARIANT_FILENAMES[args.variant]),
                              args.variant)]:
        print(f"\n=== evaluating {label} ===")
        results[label] = evaluate(filename, label, refs, latency=args.latency)

    print("\n=== summary ===")
    for label, r in results.items():
        line = f"  {label}: n={r['n']} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} CER={r['cer']:.4f}"
        if r["mean_latency_ms"] is not None:
            line += f" latency={r['mean_latency_ms']:.2f}ms"
        print(line)

    fp32_f1 = results["fp32"]["f1"]
    variant_f1 = results[args.variant]["f1"]
    delta = variant_f1 - fp32_f1
    print(f"\n[verdict] {args.variant} F1 - fp32 F1 = {delta:+.4f} "
          f"({'PASS' if delta >= -0.02 else 'FAIL'}, threshold -0.02)")

    out = {
        "variant": args.variant,
        "source": args.source,
        "n": len(refs),
        "sizes": sizes,
        "results": {
            k: {"n": v["n"], "precision": v["precision"], "recall": v["recall"],
                "f1": v["f1"], "cer": v["cer"], "mean_latency_ms": v["mean_latency_ms"]}
            for k, v in results.items()
        },
        "f1_delta_variant_minus_fp32": delta,
    }
    out_path = os.path.join(ROOT, f"scratch_quantize_punct_results_{args.source}_{args.variant}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
