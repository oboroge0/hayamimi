"""close() actually ends a pipeline's life: no threads left, nothing kept alive.

The bug these guard: `pytest tests` grew to ~19 GB RSS, all of it from
tests/test_ja_golden.py building one full pipeline per golden clip.
Neither RoutedASR nor Refiner could ever be collected, because each had
started a daemon thread that never returns -- Refiner._worker_loop's
`while True: queue.get()`, RoutedASR's preload -- and a running thread's
frame holds its owner. Refiner holds the RoutedASR, and the RoutedASR
holds several GB of sherpa-onnx recognizers, so eight clips meant eight
live engines.

Everything except the last test runs model-free (stubs in the same style
as tests/test_units.py), so CI covers the fix. The one test that runs the
real ja pipeline is skipped without models/ -- sherpa-onnx's C++ layer
calls exit() rather than raising when handed a missing model path, so
there is no "try it and see" here.
"""

import gc
import os
import sys
import threading
import time
import weakref

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import asr_engine  # noqa: E402
from realtime_transcribe import (AudioHistory, PartialPrinter,  # noqa: E402
                                 Refiner, TranslationWorker, TranslatorPool)

MODELS_DIR = os.path.join(ROOT, "models")
GOLDEN_DIR = os.path.join(ROOT, "tests", "golden", "ja")


@pytest.fixture
def no_thread_leak():
    """Fail the test if it leaves a thread running that wasn't there before.

    Identity-based rather than name-based: threading.Thread(target=...)
    names its threads "Thread-N", so there is nothing in the name to match
    _worker_loop or _preload_rest on. Comparing the live set before and
    after says the same thing and says it about every worker at once.
    """
    before = set(threading.enumerate())
    yield
    # a joined thread can linger in enumerate() for an instant on some
    # platforms; give it a moment before calling it a leak
    for _ in range(50):
        leaked = set(threading.enumerate()) - before
        if not leaked:
            break
        time.sleep(0.02)
    assert not leaked, f"threads still running after the test: {leaked}"


class _FakeAsr:
    """Stands in for RoutedASR in the Refiner tests: no models, no threads."""

    forced_lang = "ja"

    def _identify_lang(self, buf, sample_rate):
        return "ja"

    def transcribe(self, buf, sample_rate, known_lang=None, live=False, **kw):
        return {"text": "refined"}


def _refiner(asr=None) -> Refiner:
    sample_rate = 16000
    history = AudioHistory(sample_rate, keep_s=30.0)
    history.push(np.zeros(sample_rate * 3, dtype=np.float32))
    return Refiner(asr if asr is not None else _FakeAsr(), history, sample_rate,
                   PartialPrinter(enabled=False))


# ---- Refiner ---------------------------------------------------------------

def test_refiner_close_ends_the_worker_thread(no_thread_leak):
    refiner = _refiner()
    assert refiner._worker_thread.is_alive()
    refiner.close()
    assert not refiner._worker_thread.is_alive()


def test_refiner_close_is_idempotent(no_thread_leak):
    refiner = _refiner()
    refiner.close()
    refiner.close()  # must not hang on a second join or re-queue a sentinel
    assert not refiner._worker_thread.is_alive()


def test_refiner_close_runs_the_queued_backlog_first(no_thread_leak):
    """The stop sentinel goes in at the back of the FIFO, so close() is the
    same drain main()'s finish() did with _task_queue.join()."""
    refiner = _refiner()
    ran = []
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.2)
        ran.append("slow")

    refiner._task_queue.put(slow)
    started.wait(timeout=2.0)
    refiner._task_queue.put(lambda: ran.append("queued-after"))
    refiner.close()
    assert ran == ["slow", "queued-after"]


def test_refiner_refuses_work_after_close(no_thread_leak):
    refiner = _refiner()
    refiner.close()
    with pytest.raises(RuntimeError, match="closed"):
        refiner.add_span(0, 16000, "ja", "text", "")
    with pytest.raises(RuntimeError, match="closed"):
        # the deadlock case: a sync refine would wait forever on an Event
        # no worker is left to set
        refiner.maybe_refine(0, force=True)


def test_closing_the_refiner_releases_the_engine(no_thread_leak):
    """The leak itself, model-free: _worker_loop's frame held `self`, and
    `self` held the ASR engine, so neither could ever be collected."""
    asr = _FakeAsr()
    ref = weakref.ref(asr)
    refiner = _refiner(asr)
    del asr

    gc.collect()
    assert ref() is not None, "the live Refiner should still hold the engine"

    refiner.close()
    del refiner
    gc.collect()
    assert ref() is None, (
        "the engine outlived its closed Refiner -- something still holds a "
        "reference to it (this is the pytest-memory-leak bug)")


def test_refiner_close_closes_the_transcript_file(tmp_path, no_thread_leak):
    path = tmp_path / "transcript.txt"
    sample_rate = 16000
    history = AudioHistory(sample_rate, keep_s=30.0)
    refiner = Refiner(_FakeAsr(), history, sample_rate, PartialPrinter(enabled=False),
                      transcript_path=str(path))
    refiner.close()
    assert refiner._transcript is None


# ---- TranslationWorker -----------------------------------------------------

def test_translation_worker_close_ends_its_thread(no_thread_leak):
    worker = TranslationWorker(TranslatorPool(), _NullHub())
    assert worker._thread.is_alive()
    worker.close()
    worker.close()  # idempotent
    assert not worker._thread.is_alive()


def test_translation_worker_submit_after_close_is_a_no_op(no_thread_leak):
    worker = TranslationWorker(TranslatorPool(), _NullHub())
    worker.close()
    worker.submit("もう終わっている")  # must not raise, must not queue forever
    assert worker._q.empty()


class _NullHub:
    def publish(self, event):
        pass


# ---- RoutedASR (stubbed builders: no models, no sherpa-onnx) ---------------

@pytest.fixture
def stub_engine_builders(monkeypatch):
    """Make RoutedASR constructible without models/.

    sherpa-onnx's C++ layer exits the process on a missing model path, so
    every construction site has to be replaced, not just guarded: the LID
    build in __init__, the six _BUILDERS, the on-disk presence check, and
    the warm-up decode _preload_rest runs per tier. The punct/ko_spacer
    properties are replaced with plain None class attributes for the same
    reason (and to keep this test fast).
    """
    built = []

    def _builder(threads, _name="model"):
        time.sleep(0.05)  # a real tier takes ~2s; enough to be in flight
        built.append(_name)
        return object()

    monkeypatch.setattr(asr_engine, "_model_present", lambda name: True)
    monkeypatch.setattr(asr_engine, "_build_lid_guarded", lambda threads: object())
    monkeypatch.setattr(asr_engine, "_BUILDERS",
                        {name: (lambda threads, _n=name: _builder(threads, _n))
                         for name in asr_engine._BUILDERS})
    monkeypatch.setattr(asr_engine.RoutedASR, "_decode",
                        staticmethod(lambda rec, samples, sample_rate: ""))
    monkeypatch.setattr(asr_engine.RoutedASR, "punct", None)
    monkeypatch.setattr(asr_engine.RoutedASR, "ko_spacer", None)
    return built


def _stub_engine(**kw) -> asr_engine.RoutedASR:
    kw.setdefault("threads", 1)
    kw.setdefault("warmup", False)
    kw.setdefault("punctuate", False)
    kw.setdefault("max_resident", 3)
    return asr_engine.RoutedASR(**kw)


def test_engine_close_joins_the_preload_thread(stub_engine_builders, no_thread_leak):
    """close() called while _preload_rest is still loading tiers must not
    leave that thread running: it holds `self`, and through it every
    recognizer loaded so far."""
    asr = _stub_engine(preload=True)
    asr.close()
    assert asr._bg_threads == []
    assert asr.resident_models == []


def test_engine_close_drops_every_model_reference(stub_engine_builders, no_thread_leak):
    asr = _stub_engine(preload=False)
    asr._get("sv")
    assert asr.resident_models == ["sv"]
    asr.close()
    assert asr.resident_models == []
    assert asr.lid is None
    assert asr._punct is None and asr._ko_spacer is None


def test_engine_close_is_idempotent(stub_engine_builders, no_thread_leak):
    asr = _stub_engine(preload=True)
    asr.close()
    asr.close()


def test_engine_refuses_work_after_close(stub_engine_builders, no_thread_leak):
    asr = _stub_engine(preload=False)
    asr.close()
    silence = np.zeros(16000, dtype=np.float32)
    for call in (lambda: asr.transcribe(silence, 16000),
                 lambda: asr.partial(silence, 16000),
                 lambda: asr.identify(silence, 16000)):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_closed_engine_starts_no_new_background_work(stub_engine_builders,
                                                     no_thread_leak):
    asr = _stub_engine(preload=False)
    asr.close()
    assert asr._spawn_bg(lambda: None) is None


def test_closed_engine_is_collectable(stub_engine_builders, no_thread_leak):
    asr = _stub_engine(preload=True)
    ref = weakref.ref(asr)
    asr.close()
    del asr
    gc.collect()
    assert ref() is None, (
        "a closed RoutedASR is still reachable -- a background thread or "
        "another holder is keeping its recognizers alive")


# ---- the real pipeline, one clip, twice -----------------------------------

def _golden_wavs() -> list:
    if not os.path.isdir(GOLDEN_DIR):
        return []
    return sorted(f for f in os.listdir(GOLDEN_DIR) if f.endswith(".wav"))


@pytest.mark.skipif(not os.path.isdir(MODELS_DIR) or not _golden_wavs(),
                    reason="ja route models / golden clips not present")
def test_run_clip_leaves_no_engine_behind(no_thread_leak):
    """make_ja_golden.run_clip() is what tests/test_ja_golden.py calls once
    per clip. Two calls used to mean two live engines (~2.4 GB each) for
    the rest of the session; now the second call starts from where the
    first one did."""
    import make_ja_golden

    wav = os.path.join(GOLDEN_DIR, _golden_wavs()[0])
    for _ in range(2):
        result = make_ja_golden.run_clip(wav)
        assert result["finals"], "the clip produced no text -- wrong bug found"

    gc.collect()
    alive = [o for o in gc.get_objects() if isinstance(o, asr_engine.RoutedASR)]
    assert alive == [], (
        f"{len(alive)} RoutedASR instance(s) survived run_clip() -- each one "
        f"pins its sherpa-onnx recognizers, which is what took `pytest tests` "
        f"to ~19 GB")
