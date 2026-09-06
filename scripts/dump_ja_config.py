"""Print hayamimi's Japanese-route configuration as JSON.

Why this exists: docs/JA_PIPELINE.md specifies the ja route precisely enough
for another project to re-implement it (a C++ Godot GDExtension over
sherpa-onnx, at the time of writing). A document drifts away from the code
that it describes; this script reads the configuration out of the code
itself, and tests/test_ja_pipeline_spec.py compares its output against the
committed docs/ja_pipeline_spec.json so the drift is a failing test rather
than a wrong document.

No model is loaded and no recognizer is constructed. Every value below comes
from a module-level constant or a function's declared default, so this runs
on a checkout with no `models/` directory (which is exactly what CI has).

Usage:
    python scripts/dump_ja_config.py                  # configuration only
    python scripts/dump_ja_config.py --with-models    # + sha256 / byte sizes
    python scripts/dump_ja_config.py --with-models --out docs/ja_pipeline_spec.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import asr_engine  # noqa: E402
import itn_cjk  # noqa: E402
import punct_ja  # noqa: E402
import realtime_transcribe as rt  # noqa: E402

SCHEMA_VERSION = 1
GENERATED_BY = "scripts/dump_ja_config.py"

# realtime_transcribe.main()'s --threads default, which is what every
# documented ja invocation runs with. Restated here rather than read out of
# argparse: main() builds its parser inside the function, so getting at the
# default would mean running main()'s argument setup. tests/test_units.py
# would be the place to pin the two together if they ever drift.
CLI_THREADS_DEFAULT = 4

# Files that make up the ja route, as (model key, path relative to models/).
# The recognizer's four entries are resolved through the same glob patterns
# asr_engine._find() uses, so a re-export with a different epoch/avg suffix
# still resolves.
JA_ROUTE_MODELS = [
    ("recognizer.encoder", "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17",
     asr_engine.RZ_MODEL_FILES["encoder"]),
    ("recognizer.decoder", "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17",
     asr_engine.RZ_MODEL_FILES["decoder"]),
    ("recognizer.joiner", "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17",
     asr_engine.RZ_MODEL_FILES["joiner"]),
    ("recognizer.tokens", "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17",
     asr_engine.RZ_MODEL_FILES["tokens"]),
    ("vad", "", "silero_vad.onnx"),
    ("punctuation.model", "mojicast-punct-onnx", punct_ja.PUNCT_ONNX_FILENAME),
    ("punctuation.vocab", "mojicast-punct-onnx", "vocab.txt"),
]


def _default(func, name):
    """A function's declared default for `name`, so the dump cannot drift
    from the signature the pipeline actually calls."""
    value = inspect.signature(func).parameters[name].default
    if value is inspect.Parameter.empty:
        raise KeyError(f"{func.__name__}({name}=...) has no default")
    return value


def build_config() -> dict:
    """The ja route's effective configuration. Loads nothing."""
    return {
        "route": {
            # --mode single --lang ja: RoutedASR.forced_lang short-circuits
            # every LID / language-switch decision, so the ja route is the
            # only one a ja-only re-implementation needs.
            "forced_lang": "ja",
            "language_identification": False,
            "second_opinion_default": _default(
                asr_engine.RoutedASR.__init__, "ja_second_opinion"),
            "second_opinion_agree_threshold": asr_engine.SECOND_OPINION_THRESHOLD,
            # RoutedASR.transcribe()'s fixed tail, itn_cjk.py's module docstring
            "postprocessing_order": ["itn", "punctuation", "replacements"],
        },
        "recognizer": {
            "name": "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17",
            "api": "sherpa_onnx.OfflineRecognizer.from_transducer",
            "files": dict(asr_engine.RZ_MODEL_FILES),
            "model_type": asr_engine.RZ_MODEL_TYPE,
            "decoding_method": asr_engine.RZ_DECODING_METHOD,
            "modeling_unit": asr_engine.RZ_MODELING_UNIT,
            "num_threads": CLI_THREADS_DEFAULT,
            "hotwords_file": _default(asr_engine._build_reazon, "hotwords_file"),
            "hotwords_score": asr_engine.RZ_HOTWORDS_SCORE,
            "hotwords_usable": False,  # byte-level BPE tokens.txt, no bpe.model
            # sherpa-onnx FeatureExtractorConfig defaults, restated because a
            # re-implementation has to set them and they are invisible here.
            "feature_sample_rate": rt.SAMPLE_RATE,
            "feature_dim": 80,
        },
        "vad": {
            "model": "silero_vad.onnx",
            "threshold": _default(rt.build_vad, "vad_threshold"),
            "min_silence_duration": _default(rt.build_vad, "min_silence"),
            "min_speech_duration": rt.VAD_MIN_SPEECH_S,
            "max_speech_duration": _default(rt.build_vad, "max_speech"),
            "window_size": rt.WINDOW_SIZE,
            "sample_rate": rt.SAMPLE_RATE,
            "buffer_size_in_seconds": rt.VAD_BUFFER_S,
            "num_threads": rt.VAD_NUM_THREADS,
            "speech_padding_supported": False,  # sherpa-onnx 1.13.6 has no knob
        },
        "preroll": {
            "seconds": rt.PREROLL_S,
            "history_keep_seconds": _default(rt.AudioHistory.__init__, "keep_s"),
            "clamped_to_previous_segment_end": True,
            "source": "real audio from the rolling capture history",
        },
        "head_dropout_retry": {
            "min_buffer_seconds": asr_engine.SEGMENT_MIN_S,
            "min_silence_seconds": asr_engine.SEGMENT_MIN_SILENCE_S,
            "min_speech_seconds": asr_engine.SEGMENT_MIN_SPEECH_S,
            # Each piece is: SEGMENT_PAD_S of zero samples, then the VAD
            # span widened by SEGMENT_PAD_S of REAL audio on both sides,
            # then SEGMENT_PAD_S of zero samples again (_speech_pieces).
            "pad_seconds": asr_engine.SEGMENT_PAD_S,
            "pad_real_context_each_side": True,
            "pad_zero_samples_each_side": True,
            "sample_rate": asr_engine.SEGMENT_SAMPLE_RATE,
            "density_floor_cjk": asr_engine.DENSITY_FLOOR_CJK,
            "density_floor_latin": asr_engine.DENSITY_FLOOR_LATIN,
            "retry_tail_chars": asr_engine.RETRY_TAIL_CHARS,
            "retry_tail_match": asr_engine.RETRY_TAIL_MATCH,
            "fallback_chunk_seconds": asr_engine.SEGMENT_FALLBACK_CHUNK_S,
            "fallback_overlap_seconds": asr_engine.SEGMENT_FALLBACK_OVERLAP_S,
        },
        "punctuation": {
            "model": punct_ja.PUNCT_ONNX_FILENAME,
            "vocab": "vocab.txt",
            "input_names": list(punct_ja.PUNCT_INPUT_NAMES),
            "output_name": punct_ja.PUNCT_OUTPUT_NAME,
            "comma_threshold": punct_ja.PUNCT_COMMA_THRESHOLD,
            "period_threshold": punct_ja.PUNCT_PERIOD_THRESHOLD,
            "max_chars": punct_ja.PUNCT_MAX_CHARS,
            "force_final_period": punct_ja.PUNCT_FORCE_FINAL_PERIOD,
            "intra_op_num_threads": punct_ja.PUNCT_INTRA_OP_NUM_THREADS,
            "question_suffixes": list(punct_ja._QUESTION_SUFFIXES),
            # sorted so the dump is stable across runs (the source is a set)
            "punctuation_chars": sorted(punct_ja._JA_PUNCT_CHARS),
            "normalization": "NFKC",
        },
        "itn": {
            "enabled": "ja" in itn_cjk.APPLICABLE_LANGS,
            "module": "scripts/itn_cjk.py",
            "applicable_langs": sorted(itn_cjk.APPLICABLE_LANGS),
        },
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_models(models_dir: str) -> dict:
    """sha256 and byte size of every file the ja route loads.

    Returns {} when models_dir does not exist, so --with-models is safe to
    pass on a checkout without the (multi-gigabyte, untracked) models tree.
    """
    import glob

    if not os.path.isdir(models_dir):
        return {}
    out: dict = {}
    total = 0
    for key, subdir, pattern in JA_ROUTE_MODELS:
        base = os.path.join(models_dir, subdir) if subdir else models_dir
        hits = sorted(glob.glob(os.path.join(base, pattern)))
        if not hits:
            out[key] = {"file": None, "present": False}
            continue
        path = hits[0]
        size = os.path.getsize(path)
        total += size
        out[key] = {
            "file": os.path.relpath(path, models_dir).replace(os.sep, "/"),
            "present": True,
            "bytes": size,
            "sha256": _sha256(path),
        }
    out["_total_bytes"] = total
    return out


def build_spec(with_models: bool, models_dir: str) -> dict:
    spec = {
        "generated_by": GENERATED_BY,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "schema_version": SCHEMA_VERSION,
        "describes": "docs/JA_PIPELINE.md",
        "config": build_config(),
    }
    if with_models:
        spec["models"] = build_models(models_dir)
    return spec


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--with-models", action="store_true",
                    help="also hash the model files under models/ (slow: ~440MB read)")
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    args = ap.parse_args()

    spec = build_spec(args.with_models, args.models_dir)
    text = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote {args.out} ({len(text)} bytes)", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
