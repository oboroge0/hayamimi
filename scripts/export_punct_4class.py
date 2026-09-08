"""Export the trained 4-class (+ none) ja punctuation token classifier
(<model-dir>/hf/, produced by scripts/train_punct_ja.py) to ONNX,
and produce a dynamic-INT8 quantized variant, following the same recipe as
scripts/quantize_punct.py uses for the existing mojicast model.

Outputs (under --model-dir):
    <model-dir>/punct_4class.onnx
    <model-dir>/quantized_ort/punct_4class.int8.onnx

Usage:
    .venv-train/Scripts/python scripts/export_punct_4class.py
    .venv-train/Scripts/python scripts/export_punct_4class.py \
        --model-dir models/punct-ja-4class      # the superseded first-round model
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(ROOT, "models", "punct-ja-4class-permissive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="reads <dir>/hf/, writes <dir>/punct_4class.onnx and "
                         "<dir>/quantized_ort/punct_4class.int8.onnx")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    hf_dir = os.path.join(model_dir, "hf")
    onnx_path = os.path.join(model_dir, "punct_4class.onnx")
    quant_dir = os.path.join(model_dir, "quantized_ort")
    int8_path = os.path.join(quant_dir, "punct_4class.int8.onnx")

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    print(f"[load] {hf_dir}")
    # eager attention: ModernBERT's default sdpa/flash path is not traceable
    # by torch.onnx.export, and it is what training used (scripts/train_punct_ja.py).
    model = AutoModelForTokenClassification.from_pretrained(
        hf_dir, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    model.eval()

    dummy_text = "これはエクスポート用のダミー文です"
    enc = tokenizer(dummy_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print(f"[export] -> {onnx_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    fp32_size = os.path.getsize(onnx_path)
    print(f"  {fp32_size / 1e6:.2f} MB")

    # sanity check: onnxruntime output matches torch output on the dummy input
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Check the export at the traced length *and* at a different one -- a
    # silently-static sequence axis would only show up on the second.
    for tag, text in (("traced-len", dummy_text),
                      ("other-len", "動的軸が本当に効いているかを確かめるための、"
                                    "もっと長いダミー文です")):
        enc_i = tokenizer(text, return_tensors="pt")
        ort_out = sess.run(None, {
            "input_ids": enc_i["input_ids"].numpy(),
            "attention_mask": enc_i["attention_mask"].numpy(),
        })[0]
        with torch.no_grad():
            torch_out = model(input_ids=enc_i["input_ids"],
                              attention_mask=enc_i["attention_mask"]).logits.numpy()
        max_diff = np.abs(ort_out - torch_out).max()
        print(f"  [sanity/{tag}] shape={ort_out.shape} "
              f"max |onnx - torch| logit diff = {max_diff:.6f}")
        assert max_diff < 1e-3, f"ONNX export diverges from the torch model ({tag})"

    print(f"\n[quantize int8 dynamic] {onnx_path} -> {int8_path}")
    os.makedirs(quant_dir, exist_ok=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(model_input=onnx_path, model_output=int8_path, weight_type=QuantType.QInt8)
    int8_size = os.path.getsize(int8_path)
    print(f"  {int8_size / 1e6:.2f} MB ({100 * (1 - int8_size / fp32_size):.1f}% smaller than fp32)")

    print("\n[sizes]")
    print(f"  fp32: {fp32_size / 1e6:.2f} MB")
    print(f"  int8: {int8_size / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
