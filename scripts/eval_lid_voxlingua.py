"""Measure LID candidates (a) and (d) against the same real-audio harness as
scripts/eval_lid_curve.py / scripts/eval_lid_candidates.py.

MUST run under .venv-train (torch CPU + speechbrain), NOT the main sherpa-onnx
venv -- this script never imports sherpa_onnx or asr_engine (which imports
sherpa_onnx at module scope), on purpose, so it works in an environment that
only has .venv-train's packages:

    .venv-train/Scripts/python scripts/eval_lid_voxlingua.py

Two candidates share one ECAPA-TDNN embedding extractor
(speechbrain/lang-id-voxlingua107-ecapa, Apache-2.0):

  candidate (a) "voxlingua107-ecapa-raw": the model's own pretrained 107-way
      softmax classifier, argmax mapped down to this project's routed
      languages (see LABEL_MAP). VoxLingua107's label set has NO separate
      Cantonese class (checked directly against the model's
      label_encoder.txt: 107 labels, 'zh: Chinese' only) -- so, like
      whisper-tiny, this candidate structurally cannot win on yue. Measured
      anyway per the track brief (comparison value even where a candidate
      is a known reject).

  candidate (d) "voxlingua107-ecapa-target-clf": the SAME embedding_model
      (frozen, its own pretrained 107-way head is NOT used here), followed
      by a from-scratch 5-way (ja/en/zh/ko/yue) logistic-regression head
      trained on a small FLEURS TRAIN split per language (see
      train_target_classifier()) -- never on testdata/eval_real*, which
      would be evaluation leakage. This is the one candidate that CAN have
      yue as its own class, because FLEURS ships yue_hant_hk as a distinct
      config.

ONNX: full end-to-end export (waveform -> ONNX graph) hit a torch.onnx
limitation exporting speechbrain's STFT-based Fbank front end ("STFT does
not currently support complex types" -- see scripts/export_voxlingua_lid_onnx.py's
docstring/git history for the attempt). Measurement here therefore runs
native torch CPU inference instead of onnxruntime; this is a slight
pessimistic bias on latency (torch's generic conv/BN op dispatch vs
onnxruntime's fused/quantized kernels), noted in docs/eval/lid_candidates.md.
Both candidates fail here anyway (raw: yue; target-clf: see the doc for its
actual verdict) so this was not chased further -- the fbank-front-end-in-numpy
+ ONNX-export-of-just-the-neural-net path documented in this file's design
notes is the way to unblock it if a future candidate needs production
integration.

Usage:
    .venv-train/Scripts/python scripts/eval_lid_voxlingua.py
    .venv-train/Scripts/python scripts/eval_lid_voxlingua.py --skip-fleurs-download
        (reuse a previously cached FLEURS pull under .venv-train/fleurs_cache/)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch

import eval_common
from lid_preprocessing import trim_lid_clip
from lid_target_clf import target_clf_predict

WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(WORKTREE_ROOT, ".venv-train", "pretrained", "voxlingua107-ecapa")
FLEURS_CACHE_DIR = os.path.join(WORKTREE_ROOT, ".venv-train", "fleurs_cache")
HF_REPO = "speechbrain/lang-id-voxlingua107-ecapa"

LANGS = ("ja", "en", "zh", "ko", "yue")
CLEAN_DIRS = ["testdata/eval_real", "testdata/eval_real_zhko", "testdata/eval_real_yue"]
NOISY_DIR = "testdata/eval_noisy/babble_snr10"
CONDITIONS = [("clean", CLEAN_DIRS), ("babble_snr10", [NOISY_DIR])]
BIN_SECONDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
THREADS = 2  # shared 6-core CPU, see the track brief

# VoxLingua107 label (from label_encoder.txt, format "xx: Full Name") -> this
# project's routed language code. VoxLingua107 uses the same ISO 639-1 codes
# ROUTABLE_LANGS already does for every one of them EXCEPT Cantonese, which
# it has no class for at all -- 'zh: Chinese' is the closest available tag,
# so yue is deliberately mapped there too (documents the structural miss
# rather than silently producing '?' for every yue clip).
LABEL_MAP = {
    "ja": "ja", "en": "en", "zh": "zh", "ko": "ko",
    "bg": "bg", "hr": "hr", "cs": "cs", "da": "da", "nl": "nl", "et": "et",
    "fi": "fi", "fr": "fr", "de": "de", "el": "el", "hu": "hu", "it": "it",
    "lv": "lv", "lt": "lt", "mt": "mt", "pl": "pl", "pt": "pt", "ro": "ro",
    "sk": "sk", "sl": "sl", "es": "es", "sv": "sv", "ru": "ru", "uk": "uk",
}

FLEURS_CONFIG = {"ja": "ja_jp", "en": "en_us", "zh": "cmn_hans_cn", "ko": "ko_kr", "yue": "yue_hant_hk"}
FLEURS_TRAIN_N = 40  # per language; kept small -- CPU-only, single eval track


def cache_path(root: str, name: str) -> str:
    return eval_common.cache_path(root, name)


def load_clips(root: str, dirs: list) -> list:
    clips = []
    for rel in dirs:
        mdir = os.path.join(root, rel)
        for e in eval_common.load_manifest(mdir):
            clips.append((rel, e["wav"], os.path.join(mdir, e["wav"]), e["lang"]))
    return clips


class Extractor:
    """Wraps speechbrain's frozen compute_features -> mean_var_norm ->
    embedding_model pipeline (candidate (a) and (d) share every bit of this;
    they differ only in what classifies the resulting 256-dim embedding)."""

    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        torch.set_num_threads(THREADS)
        print(f"loading {HF_REPO} ...")
        self.classifier = EncoderClassifier.from_hparams(
            source=HF_REPO, savedir=PRETRAINED_DIR, local_strategy=LocalStrategy.COPY)
        self.classifier.eval()
        mods = self.classifier.mods
        self.compute_features = mods.compute_features
        self.mean_var_norm = mods.mean_var_norm
        self.embedding_model = mods.embedding_model
        self.out_classifier = mods.classifier
        ind2lab = self.classifier.hparams.label_encoder.ind2lab
        # ind2lab values look like "ja: Japanese" -- keep just the code.
        self.raw_labels = [ind2lab[i].split(":")[0].strip() for i in range(len(ind2lab))]
        print(f"{len(self.raw_labels)} VoxLingua107 classes loaded")

    @torch.no_grad()
    def embed(self, samples: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(samples).float().unsqueeze(0)
        lens = torch.ones(1)
        feats = self.compute_features(wav)
        feats = self.mean_var_norm(feats, lens)
        emb = self.embedding_model(feats, lens)  # [1, 1, 256]
        return emb.squeeze(0).squeeze(0).numpy()

    @torch.no_grad()
    def raw_predict(self, samples: np.ndarray) -> tuple:
        """candidate (a): the model's own pretrained 107-way head."""
        wav = torch.from_numpy(samples).float().unsqueeze(0)
        lens = torch.ones(1)
        feats = self.compute_features(wav)
        feats = self.mean_var_norm(feats, lens)
        emb = self.embedding_model(feats, lens)
        logits = self.out_classifier(emb).squeeze(0).squeeze(0)
        probs = torch.softmax(logits, dim=-1).numpy()
        idx = int(np.argmax(probs))
        raw_label = self.raw_labels[idx]
        return LABEL_MAP.get(raw_label, raw_label), float(probs[idx])


# --- candidate (d): FLEURS-trained target-only classifier -------------------

FLEURS_PARQUET_URL = (
    "https://huggingface.co/datasets/google/fleurs/resolve/main/"
    "parquet-data/{config}/train-00000-of-00001.parquet"
)


def fleurs_train_samples(lang: str, n: int, skip_download: bool) -> list:
    """Returns n (samples_float32_16k,) arrays from lang's FLEURS TRAIN split
    (never testdata/eval_real* -- training on the eval set would be leakage).

    Cached under .venv-train/fleurs_cache/<lang>.npz so re-runs don't re-pull.

    Fetches via a direct HTTP range-read of ONE parquet row group (pyarrow +
    fsspec's https filesystem), not `datasets.load_dataset(streaming=True)`:
    that route was tried first and dropped for two reasons found during
    development -- (1) newer `datasets` needs the extra `torchcodec` package
    just to decode the audio column, and (2) FLEURS' parquet files store each
    split as 1-3 row groups of ~1000 rows each (audio-dominated, ~750MB/row
    group), so even the streaming loader has to fetch a whole row group
    before it can yield row 0 -- there is no cheaper partial-row-group read
    at the parquet format level. Talking to pyarrow directly gets the exact
    same one-time cost with far fewer moving parts (no torchcodec, no
    `datasets` version sensitivity), and only ever reads row group 0.
    """
    cache_file = os.path.join(FLEURS_CACHE_DIR, f"{lang}.npz")
    if os.path.exists(cache_file):
        # plain float32 arrays only (no allow_pickle=True needed/used) --
        # this cache is written by this same function, in this same script,
        # from this project's own .venv-train/fleurs_cache/.
        data = np.load(cache_file)
        return [data[f"arr_{i}"] for i in range(len(data.files))]
    if skip_download:
        raise RuntimeError(f"{cache_file} missing and --skip-fleurs-download was passed")

    import io

    import fsspec
    import pyarrow.parquet as pq
    import soundfile as sf

    config = FLEURS_CONFIG[lang]
    url = FLEURS_PARQUET_URL.format(config=config)
    print(f"fetching FLEURS train/{config} row group 0 for {n} {lang} clips "
          f"(one-time ~hundreds of MB over HTTP, cached after this)...")
    fs = fsspec.filesystem("https")
    with fs.open(url, "rb") as f:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(0, columns=["audio"])
    clips = []
    for row in table.to_pylist()[:n]:
        arr, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert sr == 16000, f"{lang} FLEURS clip: expected 16kHz, got {sr}"
        clips.append(arr)
    os.makedirs(FLEURS_CACHE_DIR, exist_ok=True)
    np.savez(cache_file, *clips)
    print(f"  cached {len(clips)} clips -> {cache_file}")
    return clips


def train_target_classifier(extractor: Extractor, skip_download: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    clf_path = os.path.join(FLEURS_CACHE_DIR, "target_clf.json")
    if os.path.exists(clf_path):
        print(f"reusing cached classifier at {clf_path}")
        with open(clf_path, encoding="utf-8") as f:
            return json.load(f)

    X, y = [], []
    for lang in LANGS:
        clips = fleurs_train_samples(lang, FLEURS_TRAIN_N, skip_download)
        for samples in clips:
            clip = trim_lid_clip(samples, 16000)  # same preprocessing production/eval use
            X.append(extractor.embed(clip))
            y.append(lang)
    X = np.array(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y)
    print(f"trained target-only classifier: train acc = {clf.score(Xs, y):.3f} on {len(y)} FLEURS-train clips")

    result = {
        "classes": list(clf.classes_),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_train": len(y),
    }
    os.makedirs(FLEURS_CACHE_DIR, exist_ok=True)
    with open(clf_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


# --- eval loop ----------------------------------------------------------------

def run(root: str, skip_fleurs_download: bool):
    extractor = Extractor()
    clf_params = train_target_classifier(extractor, skip_fleurs_download)

    raw_cache = eval_common.load_cache(cache_path(root, "_lid_candidates_voxlingua_cache.json"))
    # single shared cache file, keyed by backend name too, since both
    # candidates come out of this one script/run.

    for cond_name, dirs in CONDITIONS:
        clips = load_clips(root, dirs)
        for rel, wav_name, wav_path, true_lang in clips:
            samples, sr = sf.read(wav_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            assert sr == 16000, f"{wav_path}: expected 16kHz, got {sr}"
            dur = len(samples) / sr

            for length in BIN_SECONDS:
                if length > dur + 0.05:
                    continue
                clip = trim_lid_clip(samples[: int(length * sr)], sr, max_seconds=length)

                for backend_name, predict_fn in (
                    ("voxlingua107-ecapa-raw", lambda c: extractor.raw_predict(c)),
                    ("voxlingua107-ecapa-target-clf",
                     lambda c: target_clf_predict(extractor.embed(c), clf_params)),
                ):
                    key = f"{backend_name}::{cond_name}::{rel}/{wav_name}::{length}"
                    if key in raw_cache:
                        continue
                    t0 = time.perf_counter()
                    pred_lang, conf = predict_fn(clip)
                    ms = (time.perf_counter() - t0) * 1000
                    raw_cache[key] = {
                        "backend": backend_name, "condition": cond_name, "source": rel,
                        "wav": wav_name, "true_lang": true_lang, "length": length, "dur": dur,
                        "pred_lang": pred_lang, "confidence": conf, "latency_ms": ms,
                    }
                    print(f"  [{backend_name}][{cond_name}] {wav_name:10} L={length:>3.1f}s "
                          f"true={true_lang:3} pred={pred_lang:3} conf={conf:.2f} ({ms:.0f}ms)")
            eval_common.save_cache(cache_path(root, "_lid_candidates_voxlingua_cache.json"), raw_cache)

    print("\ndone. Now run (main venv): "
          ".venv/Scripts/python scripts/eval_lid_candidates.py --report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=WORKTREE_ROOT)
    ap.add_argument("--skip-fleurs-download", action="store_true",
                     help="reuse the cached .venv-train/fleurs_cache/*.npz pulls instead of "
                          "streaming from Hugging Face again")
    args = ap.parse_args()
    run(args.root, args.skip_fleurs_download)


if __name__ == "__main__":
    main()
