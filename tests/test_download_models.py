"""scripts/download_models.py: the selective tarball extractor must never
leave a half-filled model directory behind (2026-09-04 omni incident)."""
import os
import shutil
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import download_models  # noqa: E402

WANTED = {"model.int8.onnx", "tokens.txt"}


def _make_tarball(path, names):
    src = os.path.join(os.path.dirname(path), "src")
    top = os.path.join(src, "release-dir")
    os.makedirs(os.path.join(top, "test_wavs"), exist_ok=True)
    for name in names:
        with open(os.path.join(top, name), "wb") as f:
            f.write(b"x" * 16)
    with open(os.path.join(top, "test_wavs", "en.wav"), "wb") as f:
        f.write(b"RIFF")
    with tarfile.open(path, "w:bz2") as tf:
        tf.add(top, arcname="release-dir")
    shutil.rmtree(src)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(download_models, "MODELS_DIR", str(models))
    fetched = []

    def fake_download(url, dest):
        fetched.append(url)
        shutil.copyfile(url, dest)

    monkeypatch.setattr(download_models, "_download_to", fake_download)
    return tmp_path, models, fetched


def test_tarball_without_a_wanted_file_raises_and_leaves_no_partial_dir(sandbox):
    tmp, models, fetched = sandbox
    tar = str(tmp / "fp32.tar.bz2")
    _make_tarball(tar, ["model.onnx", "tokens.txt"])  # no model.int8.onnx
    with pytest.raises(RuntimeError, match="model.int8.onnx"):
        download_models.extract_members_only(tar, "omni", WANTED, "omni")
    assert not (models / "omni").exists(), "a partial directory would be skipped as complete forever"
    assert not any(n.endswith(".part") for n in os.listdir(models))


def test_complete_tarball_extracts_only_the_wanted_files(sandbox):
    tmp, models, fetched = sandbox
    tar = str(tmp / "int8.tar.bz2")
    _make_tarball(tar, ["model.int8.onnx", "tokens.txt", "README.md"])
    download_models.extract_members_only(tar, "omni", WANTED, "omni")
    assert set(os.listdir(models / "omni")) == WANTED
    assert not any(n.endswith(".part") for n in os.listdir(models))


def test_partial_dir_is_refetched_and_complete_dir_is_skipped(sandbox):
    tmp, models, fetched = sandbox
    tar = str(tmp / "int8.tar.bz2")
    _make_tarball(tar, ["model.int8.onnx", "tokens.txt"])
    (models / "omni").mkdir()
    (models / "omni" / "tokens.txt").write_bytes(b"old")  # the incident's leftover
    download_models.extract_members_only(tar, "omni", WANTED, "omni")
    assert fetched == [tar]
    assert set(os.listdir(models / "omni")) == WANTED
    download_models.extract_members_only(tar, "omni", WANTED, "omni")
    assert fetched == [tar], "a complete directory must not be downloaded again"
