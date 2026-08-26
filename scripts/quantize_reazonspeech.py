"""Quantize the ReazonSpeech ja-en zipformer transducer to INT8 and measure
the accuracy / size tradeoff, as a first step toward a mobile-sized ja ASR
profile.

Produces two quantized variants (dynamic QInt8 via onnxruntime.quantization)
next to the original fp32 files:

  models/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/
    quantized_ort/full_int8/          encoder + decoder + joiner all int8
    quantized_ort/encoder_only_int8/  encoder int8, decoder/joiner left fp32
                                       (copied through unchanged)

Then decodes the 15 real-broadcast ja clips in testdata/eval_real (see
testdata/eval_real/manifest.json) with three sherpa-onnx OfflineRecognizer
configurations -- fp32 baseline, full_int8, encoder_only_int8 -- and reports
CER (reusing eval_accuracy.cer_ja) and RTF for each, plus on-disk size.

Usage:
    python scripts/quantize_reazonspeech.py            # quantize + evaluate
    python scripts/quantize_reazonspeech.py --skip-quantize  # eval only,
        reusing files already produced by a previous run
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_accuracy import cer_ja  # noqa: E402

MODELS_DIR = os.path.join(ROOT, "models")
RZ_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17")
QUANT_DIR = os.path.join(RZ_DIR, "quantized_ort")
FULL_INT8_DIR = os.path.join(QUANT_DIR, "full_int8")
ENCODER_ONLY_DIR = os.path.join(QUANT_DIR, "encoder_only_int8")

EVAL_DIR = os.path.join(ROOT, "testdata", "eval_real")
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.json")

COMPONENTS = ["encoder", "decoder", "joiner"]


def _find(model_dir, pattern):
    hits = glob.glob(os.path.join(model_dir, pattern))
    if not hits:
        raise FileNotFoundError(f"no match for {pattern!r} in {model_dir}")
    return hits[0]


def fp32_paths():
    return {
        "encoder": _find(RZ_DIR, "encoder-epoch-35-avg-1.onnx"),
        "decoder": _find(RZ_DIR, "decoder-epoch-35-avg-1.onnx"),
        "joiner": _find(RZ_DIR, "joiner-epoch-35-avg-1.onnx"),
    }


def quantized_name(component):
    src = fp32_paths()[component]
    base = os.path.basename(src)
    assert base.endswith(".onnx")
    return base[: -len(".onnx")] + ".int8.onnx"


def do_quantize():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(FULL_INT8_DIR, exist_ok=True)
    os.makedirs(ENCODER_ONLY_DIR, exist_ok=True)

    fp32 = fp32_paths()
    sizes = {}

    for component, src in fp32.items():
        sizes[f"{component}_fp32"] = os.path.getsize(src)
        out_name = quantized_name(component)
        full_out = os.path.join(FULL_INT8_DIR, out_name)
        print(f"[quantize] {component}: {src} -> {full_out}")
        t0 = time.time()
        quantize_dynamic(
            model_input=src,
            model_output=full_out,
            weight_type=QuantType.QInt8,
        )
        print(f"  done in {time.time() - t0:.1f}s, {os.path.getsize(full_out) / 1e6:.1f} MB")
        sizes[f"{component}_int8"] = os.path.getsize(full_out)

        eo_out = os.path.join(ENCODER_ONLY_DIR, out_name if component == "encoder" else os.path.basename(src))
        if component == "encoder":
            shutil.copyfile(full_out, eo_out)
        else:
            # decoder/joiner: sherpa-onnx-int8 knowledge says quantizing these
            # small non-conv heads on zipformer transducers tends to hurt CER
            # more than it saves in bytes, so the encoder-only profile keeps
            # them fp32.
            shutil.copyfile(src, eo_out)

    # tokens.txt needed alongside both quantized dirs for standalone loading
    for d in (FULL_INT8_DIR, ENCODER_ONLY_DIR):
        shutil.copyfile(os.path.join(RZ_DIR, "tokens.txt"), os.path.join(d, "tokens.txt"))

    return sizes


def existing_sizes():
    fp32 = fp32_paths()
    sizes = {}
    for component, src in fp32.items():
        sizes[f"{component}_fp32"] = os.path.getsize(src)
        full_out = os.path.join(FULL_INT8_DIR, quantized_name(component))
        sizes[f"{component}_int8"] = os.path.getsize(full_out)
    return sizes


def build_recognizer(encoder, decoder, joiner, tokens):
    import sherpa_onnx

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=4,
        model_type="zipformer",
        decoding_method="modified_beam_search",
        modeling_unit="cjkchar",
    )


def variant_recognizer(name):
    fp32 = fp32_paths()
    tokens = os.path.join(RZ_DIR, "tokens.txt")
    if name == "fp32":
        return build_recognizer(fp32["encoder"], fp32["decoder"], fp32["joiner"], tokens)
    if name == "full_int8":
        enc = os.path.join(FULL_INT8_DIR, quantized_name("encoder"))
        dec = os.path.join(FULL_INT8_DIR, quantized_name("decoder"))
        joi = os.path.join(FULL_INT8_DIR, quantized_name("joiner"))
        return build_recognizer(enc, dec, joi, os.path.join(FULL_INT8_DIR, "tokens.txt"))
    if name == "encoder_only_int8":
        enc = os.path.join(ENCODER_ONLY_DIR, quantized_name("encoder"))
        dec = os.path.join(ENCODER_ONLY_DIR, os.path.basename(fp32["decoder"]))
        joi = os.path.join(ENCODER_ONLY_DIR, os.path.basename(fp32["joiner"]))
        return build_recognizer(enc, dec, joi, os.path.join(ENCODER_ONLY_DIR, "tokens.txt"))
    raise ValueError(name)


def evaluate(name):
    import soundfile as sf
    import sherpa_onnx  # noqa: F401  (ensures clear error if missing)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest = [e for e in manifest if e["lang"] == "ja"]

    rec = variant_recognizer(name)

    total_dist = 0
    total_denom = 0
    total_decode_s = 0.0
    total_audio_s = 0.0
    rows = []
    for entry in manifest:
        wav_path = os.path.join(EVAL_DIR, entry["wav"])
        audio, sr = sf.read(wav_path, dtype="float32")
        audio_s = len(audio) / sr

        stream = rec.create_stream()
        t0 = time.time()
        stream.accept_waveform(sr, audio)
        rec.decode_stream(stream)
        dt = time.time() - t0
        hyp = stream.result.text

        score, dist, denom = cer_ja(entry["ref"], hyp)
        total_dist += dist
        total_denom += denom
        total_decode_s += dt
        total_audio_s += audio_s
        rows.append({"wav": entry["wav"], "cer": score, "rtf": dt / audio_s if audio_s else 0.0, "hyp": hyp})
        print(f"  [{name}] {entry['wav']}: cer={score:.4f} rtf={dt / audio_s:.4f} hyp={hyp!r}")

    micro_cer = total_dist / total_denom if total_denom else float("nan")
    mean_rtf = total_decode_s / total_audio_s if total_audio_s else float("nan")
    return {"variant": name, "cer": micro_cer, "rtf": mean_rtf, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quantize", action="store_true", help="reuse previously quantized files")
    args = ap.parse_args()

    if args.skip_quantize:
        sizes = existing_sizes()
    else:
        sizes = do_quantize()

    print("\n=== sizes (bytes) ===")
    for k, v in sizes.items():
        print(f"  {k}: {v} ({v / 1e6:.1f} MB)")

    results = {}
    for variant in ("fp32", "full_int8", "encoder_only_int8"):
        print(f"\n=== evaluating {variant} ===")
        results[variant] = evaluate(variant)

    print("\n=== summary ===")
    for variant, r in results.items():
        print(f"  {variant}: CER={r['cer']:.4f} RTF={r['rtf']:.4f}")

    out = {"sizes": sizes, "results": {k: {"cer": v["cer"], "rtf": v["rtf"]} for k, v in results.items()}}
    out_path = os.path.join(ROOT, "scratch_quantize_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
