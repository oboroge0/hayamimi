"""Pure-numpy inference for LID candidate (d)'s target-language classifier.

Deliberately has NO torch/speechbrain/sherpa-onnx import: this is the one
piece of candidate (d) (docs/eval/lid_candidates.md) that both sides of the
venv split need --

  - scripts/eval_lid_voxlingua.py (runs under .venv-train, torch) trains the
    classifier and calls target_clf_predict() to score it during evaluation;
  - a future opt-in RoutedASR(lid_backend=...) production integration (main
    venv, sherpa-onnx + onnxruntime only, no torch) would call the exact
    same function against the exact same trained parameters, so the eval
    numbers in the doc are provably what production would do -- not a
    reimplementation that could silently drift.

The classifier itself is a standard scikit-learn StandardScaler +
LogisticRegression, small enough (5 classes x 256-dim ECAPA embedding) to
ship as plain JSON (see eval_lid_voxlingua.py's train_target_classifier())
rather than needing an ONNX export of its own.
"""
import numpy as np


def target_clf_predict(embedding: np.ndarray, clf_params: dict) -> tuple:
    """embedding: 256-dim ECAPA embedding (np.ndarray). clf_params: the dict
    train_target_classifier() in eval_lid_voxlingua.py produces/serializes
    (scaler mean/scale + logistic-regression coef/intercept/classes).
    Returns (predicted_lang, confidence 0..1)."""
    mean = np.asarray(clf_params["scaler_mean"])
    scale = np.asarray(clf_params["scaler_scale"])
    x = (embedding - mean) / scale
    coef = np.asarray(clf_params["coef"])   # [n_classes, n_features]
    intercept = np.asarray(clf_params["intercept"])
    logits = coef @ x + intercept
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = int(np.argmax(probs))
    return clf_params["classes"][idx], float(probs[idx])
