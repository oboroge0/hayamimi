"""Shared helpers for scripts/eval_*.py.

Two small conventions repeated verbatim across the eval scripts, factored
out so there's one place to fix if the JSON cache format or manifest layout
ever changes:

  - a JSON cache under <root>/testdata/<filename>, used by eval_noise.py and
    eval_lid_curve.py to checkpoint slow sweeps so re-runs only (re)compute
    missing conditions;
  - manifest.json loading, used by eval_engine.py, eval_noise.py,
    eval_singing.py, and eval_lid_curve.py to read a testdata manifest dir's
    list of {"wav": ..., "lang": ..., "ref": ...} entries.
"""
import json
import os


def cache_path(root: str, filename: str) -> str:
    return os.path.join(root, "testdata", filename)


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_manifest(mdir: str) -> list:
    """Load mdir/manifest.json: a list of {"wav", "lang", "ref"} entries."""
    with open(os.path.join(mdir, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)
