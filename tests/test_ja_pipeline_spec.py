"""docs/spec/ja_pipeline_spec.json must keep describing the code it came from.

The Japanese route is specified for re-implementation in docs/spec/ja_pipeline.ja.md,
with the numbers in machine-readable form in docs/spec/ja_pipeline_spec.json. Both
are only worth anything while they still match `scripts/`, so this is the
drift check: re-run the dump and compare.

Two halves, because they need different things on disk:

  * the CONFIGURATION half loads no model and reads no file under models/,
    so it always runs -- including in CI, which installs sherpa-onnx numpy
    scipy soundfile pytest and has no models/ and no testdata/;
  * the MODEL half (sha256 + byte size per file) can only be checked where
    models/ is actually populated, so it skips otherwise.

Regenerate after an intentional change:

    python scripts/dump_ja_config.py --with-models --out docs/spec/ja_pipeline_spec.json
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "docs", "spec", "ja_pipeline_spec.json")
MODELS_DIR = os.path.join(ROOT, "models")

# Deliberately NOT imported at module scope: scripts/dump_ja_config.py pulls
# in realtime_transcribe, hence sherpa_onnx. That import is fine in CI but
# collecting this file must not depend on it.
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _spec() -> dict:
    with open(SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _dump(with_models: bool = False) -> dict:
    import dump_ja_config

    return dump_ja_config.build_spec(with_models, MODELS_DIR)


def test_spec_file_exists_and_declares_its_generator():
    spec = _spec()
    assert spec["generated_by"] == "scripts/dump_ja_config.py"
    assert spec["schema_version"] == 1
    assert spec["generated_at"].endswith("Z")


def test_config_matches_the_live_dump():
    """The whole point: docs/spec/ja_pipeline_spec.json's `config` block is what
    scripts/ currently says, not what it said when someone wrote it down."""
    committed = _spec()["config"]
    live = _dump()["config"]
    assert live == committed, (
        "docs/spec/ja_pipeline_spec.json is stale. Regenerate it with\n"
        "  python scripts/dump_ja_config.py --with-models "
        "--out docs/spec/ja_pipeline_spec.json\n"
        "and update docs/spec/ja_pipeline.ja.md to match."
    )


def test_dump_needs_no_models_directory():
    """The dump has to work on a checkout with no models/ -- that is what
    makes the check above runnable in CI at all."""
    import dump_ja_config

    spec = dump_ja_config.build_spec(True, os.path.join(ROOT, "no_such_models_dir"))
    assert spec["models"] == {}
    assert spec["config"] == _spec()["config"]


@pytest.mark.skipif(not os.path.isdir(MODELS_DIR),
                    reason="models/ not present (CI / --minimal install)")
def test_model_hashes_match_when_models_are_present():
    committed = _spec().get("models")
    if not committed:
        pytest.skip("docs/spec/ja_pipeline_spec.json carries no model hashes")
    live = _dump(with_models=True)["models"]
    missing = [k for k, v in live.items()
               if isinstance(v, dict) and not v.get("present")]
    if missing:
        pytest.skip(f"ja-route model files not installed: {missing}")
    assert live == committed, (
        "the ja-route model files on disk differ from the ones "
        "docs/spec/ja_pipeline_spec.json was generated from"
    )
