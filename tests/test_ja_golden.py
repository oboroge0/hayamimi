"""Re-run the Japanese route over the golden clips and compare its output.

docs/JA_PIPELINE.md specifies the ja route so another project can rebuild it;
tests/golden/ja/golden.json is what this project's own implementation of that
specification actually produces. This test is what keeps the two honest: it
runs the real pipeline (via scripts/make_ja_golden.py, the same code that
recorded the file) and compares.

The comparison is two-level, and the reason is in tests/golden/ja/README.md:
int8 ONNX kernels are not bit-reproducible across CPU microarchitectures, so
demanding exact text everywhere would fail on a different machine for a
reason that is not a regression. So:

  * exact equality is reported (how many clips matched character for
    character), but is not itself the pass condition;
  * the pass condition is CER against the golden text, per clip, at most
    make_ja_golden.CER_TOLERANCE (1.0%).

Needs models/ and testdata/; skips cleanly without either, so it does not
run in CI (which installs sherpa-onnx numpy scipy soundfile pytest and has
neither directory).
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DIR = os.path.join(ROOT, "tests", "golden", "ja")
GOLDEN_JSON = os.path.join(GOLDEN_DIR, "golden.json")
MODELS_DIR = os.path.join(ROOT, "models")
MANIFEST = os.path.join(ROOT, "testdata", "fleurs_bench", "ja", "manifest.json")

sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _ja_route_present() -> bool:
    """models/ holds every file the ja route loads."""
    import glob

    rz = os.path.join(MODELS_DIR,
                      "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17")
    needed = [
        glob.glob(os.path.join(rz, "encoder-*.int8.onnx")),
        glob.glob(os.path.join(rz, "decoder-*.int8.onnx")),
        glob.glob(os.path.join(rz, "joiner-*.int8.onnx")),
        [os.path.join(rz, "tokens.txt")],
        [os.path.join(MODELS_DIR, "silero_vad.onnx")],
        [os.path.join(MODELS_DIR, "mojicast-punct-onnx", "punct_bert.onnx")],
    ]
    return all(hits and os.path.exists(hits[0]) for hits in needed)


needs_models = pytest.mark.skipif(
    not os.path.isdir(MODELS_DIR) or not _ja_route_present(),
    reason="ja route models not present under models/",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(GOLDEN_JSON),
    reason="tests/golden/ja/golden.json not present",
)


def _golden() -> dict:
    with open(GOLDEN_JSON, encoding="utf-8") as f:
        return json.load(f)


# --- always runnable: the recorded file's own integrity -------------------

@needs_golden
def test_golden_file_is_self_consistent():
    """Runs everywhere, including CI: the clips the file describes are the
    clips that are committed, and every one carries what a re-run needs."""
    golden = _golden()
    assert golden["_generated_by"] == "scripts/make_ja_golden.py"
    assert golden["_cer_tolerance"] > 0
    on_disk = sorted(f for f in os.listdir(GOLDEN_DIR) if f.endswith(".wav"))
    assert [c["wav"] for c in golden["clips"]] == on_disk
    for clip in golden["clips"]:
        assert clip["finals"], f"{clip['wav']}: no finals recorded"
        assert len(clip["sha256"]) == 64
        assert clip["reference"]


@needs_golden
def test_golden_wavs_are_the_ones_that_were_recorded():
    """A clip silently replaced would invalidate every text below it."""
    import make_ja_golden

    for clip in _golden()["clips"]:
        path = os.path.join(GOLDEN_DIR, clip["wav"])
        assert make_ja_golden.sha256_file(path) == clip["sha256"], (
            f"{clip['wav']} is not the file golden.json was recorded from")


@needs_golden
@pytest.mark.skipif(not os.path.exists(MANIFEST),
                    reason="testdata/fleurs_bench/ja not present")
def test_references_still_match_the_fleurs_manifest():
    import make_ja_golden

    refs = make_ja_golden.fleurs_references()
    for clip in _golden()["clips"]:
        fleurs_id, ref = refs[clip["wav"]]
        assert clip["fleurs_id"] == fleurs_id
        assert clip["reference"] == ref


# --- the real thing: rerun the pipeline -----------------------------------

@needs_golden
@needs_models
@pytest.mark.skipif(not os.path.exists(MANIFEST),
                    reason="testdata/fleurs_bench/ja not present")
def test_pipeline_still_produces_the_golden_text():
    import make_ja_golden

    golden = _golden()
    tolerance = golden["_cer_tolerance"]
    exact, failures, report = 0, [], []
    all_want, all_got = [], []

    for clip in golden["clips"]:
        path = os.path.join(GOLDEN_DIR, clip["wav"])
        result = make_ja_golden.run_clip(path)
        want, got = clip["finals"], result["finals"]
        if got == want:
            exact += 1
        cer = make_ja_golden.ja_cer("".join(want), "".join(got))
        all_want.append("".join(want))
        all_got.append("".join(got))
        report.append(f"{clip['wav']}: cer={cer:.4f} "
                      f"{'exact' if got == want else 'differs'}")
        if cer > tolerance:
            failures.append(f"{clip['wav']}: CER {cer:.4f} > {tolerance}\n"
                            f"  golden: {want}\n  now   : {got}")

    # Reported, not asserted. The per-clip gate above is the pass condition;
    # this is here because at these clip lengths (21-44 normalized
    # characters) a 1.0% per-clip threshold admits zero differing
    # characters, so a set-level figure is the only thing that says HOW far
    # off a failing run is. See tests/golden/ja/README.md.
    set_cer = make_ja_golden.ja_cer("".join(all_want), "".join(all_got))
    print(f"\ngolden ja: {exact}/{len(golden['clips'])} clips exact, "
          f"set-level CER {set_cer:.4f}")
    print("\n".join(report))
    assert not failures, "\n".join(failures)
