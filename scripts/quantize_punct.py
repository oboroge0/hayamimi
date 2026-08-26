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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from punct_ja import PunctuatorJa  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "mojicast-punct-onnx")
FP32_PATH = os.path.join(MODEL_DIR, "punct_bert.onnx")
QUANT_DIR = os.path.join(MODEL_DIR, "quantized_ort")
INT8_NAME = "punct_bert.int8.onnx"
INT8_PATH = os.path.join(QUANT_DIR, INT8_NAME)

EVAL_DIR = os.path.join(ROOT, "testdata", "eval_real")
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.json")

TARGET_MARKS = ("、", "。", "？")


def do_quantize():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(QUANT_DIR, exist_ok=True)
    fp32_size = os.path.getsize(FP32_PATH)
    print(f"[quantize] {FP32_PATH} -> {INT8_PATH}")
    t0 = time.time()
    quantize_dynamic(
        model_input=FP32_PATH,
        model_output=INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    int8_size = os.path.getsize(INT8_PATH)
    print(f"  done in {time.time() - t0:.1f}s, {int8_size / 1e6:.1f} MB "
          f"(fp32 was {fp32_size / 1e6:.1f} MB)")
    return {"fp32": fp32_size, "int8": int8_size}


def existing_sizes():
    return {"fp32": os.path.getsize(FP32_PATH), "int8": os.path.getsize(INT8_PATH)}


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

def evaluate(onnx_filename: str, label: str):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest = [e for e in manifest if e["lang"] == "ja"]

    punctuator = PunctuatorJa(model_dir=MODEL_DIR, onnx_filename=onnx_filename)

    tp = fp = fn = 0
    total_dist = 0
    total_denom = 0
    rows = []
    for entry in manifest:
        ref = entry["ref"]
        stripped, ref_marks = strip_marks(ref)
        if not stripped:
            continue
        hyp = punctuator.restore(stripped)
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
        rows.append({"wav": entry["wav"], "ref": ref, "hyp": hyp, "cer": rate})
        print(f"  [{label}] {entry['wav']}: cer={rate:.4f}")
        print(f"      ref: {ref}")
        print(f"      hyp: {hyp}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    micro_cer = total_dist / total_denom if total_denom else float("nan")

    return {
        "label": label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": micro_cer,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    sizes = existing_sizes() if args.skip_quantize else do_quantize()

    print("\n=== sizes (bytes) ===")
    for k, v in sizes.items():
        print(f"  {k}: {v} ({v / 1e6:.1f} MB)")

    results = {}
    for filename, label in [("punct_bert.onnx", "fp32"),
                             (os.path.join("quantized_ort", INT8_NAME), "int8")]:
        print(f"\n=== evaluating {label} ===")
        results[label] = evaluate(filename, label)

    print("\n=== summary ===")
    for label, r in results.items():
        print(f"  {label}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} CER={r['cer']:.4f}")

    out = {
        "sizes": sizes,
        "results": {
            k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1"], "cer": v["cer"]}
            for k, v in results.items()
        },
    }
    out_path = os.path.join(ROOT, "scratch_quantize_punct_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
