"""Export speechbrain/lang-id-voxlingua107-ecapa to ONNX for LID-candidate (a).

Part of the track-A LID replacement evaluation (docs/eval/lid_candidates.md).
Must be run under the CPU-torch training venv, NOT the main sherpa-onnx venv:

    .venv-train/Scripts/python scripts/export_voxlingua_lid_onnx.py

Downloads speechbrain/lang-id-voxlingua107-ecapa (Apache-2.0) from Hugging
Face into .venv-train/pretrained/voxlingua107-ecapa/ (gitignored, not the
shared models/ junction -- this is a training-time artifact, not a runtime
one), then traces the encoder+classifier (waveform -> log-mel Fbank ->
mean/var norm -> ECAPA-TDNN embedding -> linear classifier -> softmax) into a
single ONNX graph with a dynamic time axis.

Output: models/voxlingua107-ecapa-lid-onnx/model.onnx + labels.json
(models/ is the shared junction; this is the only file this script writes
there, and the directory name is unique to this eval track).

The 107-way softmax this produces is the RAW VoxLingua107 output -- see
docs/eval/lid_candidates.md for the label mapping down to this project's
routed languages (ja/en/zh/ko/yue + 24 EU langs) used at eval time.
"""
import json
import os
import sys

WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(WORKTREE_ROOT, ".venv-train", "pretrained", "voxlingua107-ecapa")
OUT_DIR = os.path.join(WORKTREE_ROOT, "models", "voxlingua107-ecapa-lid-onnx")
HF_REPO = "speechbrain/lang-id-voxlingua107-ecapa"


def main():
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    print(f"loading {HF_REPO} (downloads to {PRETRAINED_DIR} if not cached)...")
    # LocalStrategy.COPY, not the default SYMLINK: creating symlinks needs
    # elevated privileges / Developer Mode on Windows (WinError 1314), which
    # this worktree's shell doesn't have.
    classifier = EncoderClassifier.from_hparams(
        source=HF_REPO, savedir=PRETRAINED_DIR, local_strategy=LocalStrategy.COPY)
    classifier.eval()

    compute_features = classifier.mods.compute_features
    mean_var_norm = classifier.mods.mean_var_norm
    embedding_model = classifier.mods.embedding_model
    out_classifier = classifier.mods.classifier
    label_encoder = classifier.hparams.label_encoder

    # label_encoder maps class index -> "xx: Full Name" strings (see
    # label_encoder.txt fetched during design: 107 classes, index 0..106,
    # no separate Cantonese class -- 'zh: Chinese' (index 106 in the
    # canonical encoder) is the closest tag to both Mandarin and Cantonese).
    ind2lab = label_encoder.ind2lab
    labels = [ind2lab[i] for i in range(len(ind2lab))]
    print(f"{len(labels)} VoxLingua107 classes")

    class VoxLinguaLidOnnx(torch.nn.Module):
        """waveform [1, T] float32 @16kHz -> [1, 107] softmax posteriors."""

        def __init__(self):
            super().__init__()
            self.compute_features = compute_features
            self.mean_var_norm = mean_var_norm
            self.embedding_model = embedding_model
            self.out_classifier = out_classifier

        def forward(self, wavs: torch.Tensor) -> torch.Tensor:
            lens = torch.ones(wavs.shape[0], device=wavs.device)
            feats = self.compute_features(wavs)
            feats = self.mean_var_norm(feats, lens)
            emb = self.embedding_model(feats, lens)
            out = self.out_classifier(emb).squeeze(1)  # [1, 107] logits
            return torch.nn.functional.softmax(out, dim=-1)

    wrapper = VoxLinguaLidOnnx()
    wrapper.eval()

    dummy = torch.randn(1, 16000 * 3)  # 3s dummy clip for tracing
    with torch.no_grad():
        ref_out = wrapper(dummy)

    os.makedirs(OUT_DIR, exist_ok=True)
    onnx_path = os.path.join(OUT_DIR, "model.onnx")
    print(f"exporting to {onnx_path} ...")
    torch.onnx.export(
        wrapper, (dummy,), onnx_path,
        input_names=["waveform"], output_names=["lang_probs"],
        dynamic_axes={"waveform": {1: "num_samples"}, "lang_probs": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
        dynamo=False,  # legacy TorchScript-based exporter: avoids the
                       # torch>=2.5 dynamo path's onnxscript dependency,
                       # which isn't installed in .venv-train.
    )

    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"wrote {onnx_path} ({size_mb:.1f} MB) and labels.json ({len(labels)} labels)")

    # Sanity-check the exported graph against the torch reference output on
    # the SAME dummy input (onnxruntime must also be installed in
    # .venv-train for this -- `pip install onnxruntime`).
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"waveform": dummy.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(onnx_out - ref_out.numpy())))
        print(f"onnxruntime vs torch max abs diff on dummy input: {max_abs_diff:.6f}")
        if max_abs_diff > 1e-3:
            print("WARNING: diff above 1e-3 -- export may be numerically off", file=sys.stderr)
    except ImportError:
        print("onnxruntime not installed in this venv -- skipping numeric sanity check "
              "(pip install onnxruntime in .venv-train to enable it)")


if __name__ == "__main__":
    main()
