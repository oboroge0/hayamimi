"""Unit tests for GitHub issue #29's embedding API: EventHub pub/sub,
runtime setters on RoutedASR, TranslatorPool, LiveVad's deferred rebuild,
the various reset() methods, and SubtitleServer's /config and /reset
endpoints.

Run with: .venv/Scripts/python -m pytest tests -q

Most of this runs with no ASR/VAD models loaded at all (stubs stand in for
RoutedASR/the sherpa VAD/translators); the exception is the
RoutedASR-setter section, which constructs a real (but preload=False,
warmup=False) engine and so needs at least the whisper-tiny LID model
present on disk -- see `needs_lid_model` below, matching the skip pattern
tests/test_asr_segment.py already uses for its own model-dependent tests.
"""
import http.client
import json
import os
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import asr_engine  # noqa: E402
import realtime_transcribe as rt  # noqa: E402
import subtitle_server as ss  # noqa: E402
from asr_engine import RoutedASR  # noqa: E402

needs_lid_model = pytest.mark.skipif(
    not asr_engine._model_present("lid"),
    reason="whisper-tiny LID model not present under models/",
)


# =============================================================================
# EventHub: pub/sub + listener
# =============================================================================

def test_eventhub_publish_delivers_to_subscriber_in_order():
    hub = ss.EventHub()
    q = hub.subscribe()
    hub.publish({"type": "partial", "text": "a"})
    hub.publish({"type": "partial", "text": "b"})
    assert json.loads(q.get_nowait())["text"] == "a"
    assert json.loads(q.get_nowait())["text"] == "b"


def test_eventhub_replays_refine_events_to_new_subscribers():
    hub = ss.EventHub()
    hub.publish({"type": "refine", "text": "old news", "lang": "ja"})
    hub.publish({"type": "partial", "text": "not replayed"})  # only refine replays
    q = hub.subscribe()
    first = json.loads(q.get_nowait())
    assert first == {"type": "refine", "text": "old news", "lang": "ja"}
    assert q.empty()


def test_eventhub_unsubscribe_stops_delivery():
    hub = ss.EventHub()
    q = hub.subscribe()
    hub.unsubscribe(q)
    hub.publish({"type": "partial", "text": "x"})
    assert q.empty()


def test_eventhub_add_listener_receives_the_raw_dict_synchronously():
    hub = ss.EventHub()
    seen = []
    hub.add_listener(seen.append)
    event = {"type": "warning", "code": "x", "message": "y"}
    hub.publish(event)
    assert seen == [event]


def test_eventhub_multiple_listeners_all_receive_the_event():
    hub = ss.EventHub()
    a, b = [], []
    hub.add_listener(a.append)
    hub.add_listener(b.append)
    hub.publish({"type": "session_reset"})
    assert len(a) == 1 and len(b) == 1


def test_eventhub_listener_exception_is_swallowed_and_reported_once(capsys):
    hub = ss.EventHub()

    def boom(event):
        raise RuntimeError("listener is broken")

    hub.add_listener(boom)
    hub.publish({"type": "partial", "text": "1"})
    hub.publish({"type": "partial", "text": "2"})  # same listener, must not crash publish()

    err = capsys.readouterr().err
    assert err.count("event listener raised") == 1, (
        f"expected exactly one report for this listener across both publishes, got: {err!r}")


def test_eventhub_a_second_broken_listener_gets_its_own_report(capsys):
    hub = ss.EventHub()

    def boom_a(event):
        raise RuntimeError("a")

    def boom_b(event):
        raise RuntimeError("b")

    hub.add_listener(boom_a)
    hub.add_listener(boom_b)
    hub.publish({"type": "partial", "text": "1"})

    err = capsys.readouterr().err
    assert err.count("event listener raised") == 2


# =============================================================================
# drain_segments(): `final` event carries the new keys, `switched` logic
# =============================================================================

class _Segment:
    def __init__(self, start, samples):
        self.start = start
        self.samples = samples


class _OneShotVad:
    """Vad stub yielding exactly the segments it's constructed with, then
    reporting empty -- just enough of the sherpa VAD API for
    drain_segments()'s `while not vad.empty(): ... vad.pop()` loop."""

    def __init__(self, segments):
        self._q = list(segments)

    def empty(self):
        return not self._q

    @property
    def front(self):
        return self._q[0]

    def pop(self):
        return self._q.pop(0)


class _StubAsr:
    """Returns one canned transcribe() result per call, in order."""

    def __init__(self, results):
        self._results = list(results)

    def transcribe(self, samples, sample_rate, known_lang=None, speech_s=None):
        return self._results.pop(0)


def _final_event_for(lang: str, stats, hub) -> dict:
    sr = 16000
    samples = np.zeros(sr, dtype=np.float32)  # 1.0s
    vad = _OneShotVad([_Segment(0, samples)])
    asr = _StubAsr([{"text": f"hello in {lang}", "lang": lang, "tier": "rz",
                     "lid_ms": 12.0, "decode_ms": 34.0, "probe_ms": 0.0}])
    printer = rt.PartialPrinter(enabled=False, hub=hub)
    q = hub.subscribe()
    rt.drain_segments(vad, sr, asr, stats, printer)
    events = []
    while not q.empty():
        events.append(json.loads(q.get_nowait()))
    hub.unsubscribe(q)
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1, f"expected exactly one final event, got {finals}"
    return finals[0]


def test_final_event_carries_audio_s_lid_ms_decode_ms():
    hub = ss.EventHub()
    stats = rt.SessionStats()
    event = _final_event_for("ja", stats, hub)
    assert event["text"] == "hello in ja"
    assert event["lang"] == "ja"
    assert event["tier"] == "rz"
    assert event["audio_s"] == pytest.approx(1.0)
    assert event["lid_ms"] == pytest.approx(12.0)
    assert event["decode_ms"] == pytest.approx(34.0)


def test_final_event_switched_is_false_for_the_first_final():
    hub = ss.EventHub()
    stats = rt.SessionStats()
    event = _final_event_for("ja", stats, hub)
    assert event["switched"] is False


def test_final_event_switched_true_when_language_changes():
    hub = ss.EventHub()
    stats = rt.SessionStats()
    _final_event_for("ja", stats, hub)
    event = _final_event_for("en", stats, hub)
    assert event["switched"] is True


def test_final_event_switched_false_when_language_repeats():
    hub = ss.EventHub()
    stats = rt.SessionStats()
    _final_event_for("ja", stats, hub)
    _final_event_for("en", stats, hub)
    event = _final_event_for("en", stats, hub)
    assert event["switched"] is False


def test_session_stats_reset_clears_last_final_lang():
    hub = ss.EventHub()
    stats = rt.SessionStats()
    _final_event_for("en", stats, hub)
    assert stats.last_final_lang == "en"
    stats.reset()
    assert stats.last_final_lang is None
    # a reset session's first final must not appear "switched" against the
    # previous session's trailing language
    event = _final_event_for("ja", stats, hub)
    assert event["switched"] is False


# =============================================================================
# Live translations are published without --serve (no SubtitleServer at all)
# =============================================================================

class _FakeTranslator:
    def __init__(self, mapping):
        self._mapping = mapping

    def translate(self, text):
        return self._mapping.get(text, text)


def test_translation_worker_publishes_on_hub_with_no_http_server():
    hub = ss.EventHub()
    pool = rt.TranslatorPool(hub=hub)
    # bypass add_target()'s real model construction: stuff a fake
    # translator straight into the pool's own bookkeeping, the same way
    # tests/test_speaker_id.py's make_labeler() reaches into SpeakerLabeler
    # internals to avoid loading a real model.
    pool._translators["en"] = _FakeTranslator({"こんにちは": "hello"})
    pool._order.append("en")

    worker = rt.TranslationWorker(pool, hub)
    q = hub.subscribe()
    worker.submit("こんにちは")

    deadline = time.time() + 2.0
    event = None
    while time.time() < deadline:
        if not q.empty():
            candidate = json.loads(q.get_nowait())
            if candidate.get("type") == "translation":
                event = candidate
                break
        time.sleep(0.01)
    assert event == {"type": "translation", "lang": "en", "text": "hello"}


def test_translation_worker_is_a_silent_no_op_with_an_empty_pool():
    """main() always constructs the worker, even with no --translate
    targets -- submitting text with nothing in the pool must not publish
    anything or raise."""
    hub = ss.EventHub()
    pool = rt.TranslatorPool(hub=hub)
    worker = rt.TranslationWorker(pool, hub)
    q = hub.subscribe()
    worker.submit("text with no configured targets")
    time.sleep(0.1)
    assert q.empty()


# =============================================================================
# RoutedASR runtime setters
# =============================================================================

def _cheap_asr(**kwargs) -> RoutedASR:
    # not a test function itself -- every CALLER below is marked
    # @needs_lid_model, since RoutedASR.__init__ always builds the LID
    # model eagerly regardless of preload/warmup.
    kwargs.setdefault("threads", 1)
    kwargs.setdefault("warmup", False)
    kwargs.setdefault("preload", False)
    kwargs.setdefault("punctuate", False)
    return RoutedASR(**kwargs)


@needs_lid_model
def test_set_forced_lang_none_reverts_to_auto():
    asr = _cheap_asr(forced_lang="ja")
    assert asr.forced_lang == "ja"
    asr.set_forced_lang(None)
    assert asr.forced_lang is None


@needs_lid_model
def test_set_forced_lang_accepts_a_routable_code():
    asr = _cheap_asr()
    asr.set_forced_lang("en")
    assert asr.forced_lang == "en"


@needs_lid_model
def test_set_forced_lang_rejects_an_unroutable_code():
    asr = _cheap_asr()
    with pytest.raises(ValueError):
        asr.set_forced_lang("not-a-real-lang-code")
    assert asr.forced_lang is None  # rejected: unchanged


@needs_lid_model
def test_set_forced_lang_resets_sticky_and_pending_lid_state():
    asr = _cheap_asr()
    asr.last_lang = "zh"
    asr._pending_lang = "ko"
    asr._pending_count = 1
    asr.set_forced_lang("ja")
    assert asr.last_lang is None
    assert asr._pending_lang is None
    assert asr._pending_count == 0


@needs_lid_model
def test_set_dual_confirm_updates_the_flag():
    asr = _cheap_asr(dual_confirm=True)
    asr.set_dual_confirm(False)
    assert asr.dual_confirm is False
    asr.set_dual_confirm(True)
    assert asr.dual_confirm is True


@needs_lid_model
def test_set_lid_switch_confirm_accepts_valid_and_rejects_below_one():
    asr = _cheap_asr()
    asr.set_lid_switch_confirm(5)
    assert asr.lid_switch_confirm == 5
    with pytest.raises(ValueError):
        asr.set_lid_switch_confirm(0)


@needs_lid_model
def test_set_min_switch_s_accepts_valid_and_rejects_negative():
    asr = _cheap_asr()
    asr.set_min_switch_s(3.5)
    assert asr.min_switch_s == pytest.approx(3.5)
    with pytest.raises(ValueError):
        asr.set_min_switch_s(-0.1)


@needs_lid_model
def test_set_punctuate_off_then_on_is_a_noop_when_already_loaded():
    """Turning punctuate on while it's already on (never degraded) must not
    discard an already-loaded punct model."""
    asr = _cheap_asr(punctuate=True)
    sentinel = object()
    asr._punct = sentinel
    asr.set_punctuate(True)  # already on: no-op
    assert asr._punct is sentinel
    assert asr._punctuate is True


@needs_lid_model
def test_set_punctuate_reenable_after_degrade_clears_punct_for_retry():
    """A failure-degrade leaves _punctuate False (and _punct None, since
    the property only ever sets _punctuate False when it never
    successfully built one) -- re-enabling must let the next `.punct`
    access retry the load rather than silently staying degraded."""
    asr = _cheap_asr(punctuate=False)
    assert asr._punct is None
    asr.set_punctuate(True)
    assert asr._punctuate is True
    assert asr._punct is None  # nothing loaded yet, but no longer disabled

    asr.set_punctuate(False)
    assert asr._punctuate is False


# =============================================================================
# TranslatorPool.add_target/remove_target (translator construction stubbed)
# =============================================================================

def test_translator_pool_add_target_builds_once_and_emits_model_load(monkeypatch):
    calls = []

    def fake_build(lang):
        calls.append(lang)
        return _FakeTranslator({})

    monkeypatch.setattr(rt, "_build_translator", fake_build)
    hub = ss.EventHub()
    events = []
    hub.add_listener(events.append)
    pool = rt.TranslatorPool(hub=hub)

    pool.add_target("en")
    assert pool.targets == ["en"]
    assert calls == ["en"]

    model_load = [e for e in events if e["type"] == "model_load"]
    assert [e["phase"] for e in model_load] == ["start", "done"]
    assert all(e["model"] == "translator:en" for e in model_load)


def test_translator_pool_add_target_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(rt, "_build_translator", lambda lang: calls.append(lang) or _FakeTranslator({}))
    pool = rt.TranslatorPool()
    pool.add_target("en")
    pool.add_target("en")  # already a target: must not rebuild
    assert calls == ["en"]
    assert pool.targets == ["en"]


def test_translator_pool_remove_target_drops_it():
    pool = rt.TranslatorPool()
    pool._translators["en"] = _FakeTranslator({})
    pool._order.append("en")
    pool.remove_target("en")
    assert pool.targets == []
    assert pool.get("en") is None


def test_translator_pool_remove_target_missing_is_a_noop():
    pool = rt.TranslatorPool()
    pool.remove_target("never-added")  # must not raise
    assert pool.targets == []


def test_translator_pool_add_target_unsupported_raises_value_error(monkeypatch):
    def fake_build(lang):
        raise ValueError(f"unsupported translation target: {lang}")

    monkeypatch.setattr(rt, "_build_translator", fake_build)
    pool = rt.TranslatorPool()
    with pytest.raises(ValueError):
        pool.add_target("xx")
    assert pool.targets == []


def test_build_translators_degrades_unsupported_target_to_stderr(monkeypatch, capsys):
    def fake_build(lang):
        if lang == "bad":
            raise ValueError("unsupported translation target: bad")
        return _FakeTranslator({})

    monkeypatch.setattr(rt, "_build_translator", fake_build)
    pool = rt.TranslatorPool()
    rt.build_translators("en,bad", pool)
    assert pool.targets == ["en"]
    assert "unsupported translation target: bad" in capsys.readouterr().err


def test_translator_pool_items_and_targets_survive_concurrent_add_remove(monkeypatch):
    """Regression for GitHub issue #29 review round 1: items()/targets
    used to read self._order/self._translators without the lock
    add_target()/remove_target() hold, so a remove_target() landing mid-
    iteration could KeyError inside the live TranslationWorker's daemon
    thread -- which has no try/except around its loop, so that silently
    killed translation for the rest of the session. Hammers add/remove
    from one thread while several reader threads repeatedly call
    items()/targets; any exception in a reader is the regression."""
    monkeypatch.setattr(rt, "_build_translator", lambda lang: _FakeTranslator({}))

    pool = rt.TranslatorPool()
    langs = [f"l{i}" for i in range(12)]
    for lang in langs:
        pool.add_target(lang)

    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                for _lang, _tr in pool.items():
                    pass
                list(pool.targets)
            except Exception as exc:
                errors.append(exc)
                return

    def writer():
        for i in range(300):
            lang = langs[i % len(langs)]
            pool.remove_target(lang)
            pool.add_target(lang)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for r in readers:
        r.start()
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    writer_thread.join(timeout=15.0)
    stop.set()
    for r in readers:
        r.join(timeout=5.0)

    assert not any(r.is_alive() for r in readers), "a reader thread never noticed stop"
    assert errors == [], f"items()/targets raised under concurrent mutation: {errors!r}"


# =============================================================================
# LiveVad.set_sensitivity: deferred rebuild
# =============================================================================

class _FakeVad:
    """Stands in for sherpa_onnx.VoiceActivityDetector: each construction
    is a distinct object (identity-checked below to prove a rebuild
    happened), with a settable is_speech_detected() so tests can simulate
    "mid-utterance" vs. "between utterances"."""

    _next_id = 0

    def __init__(self, min_silence, max_speech, threshold):
        _FakeVad._next_id += 1
        self.id = _FakeVad._next_id
        self.min_silence = min_silence
        self.max_speech = max_speech
        self.threshold = threshold
        self.speech = False

    def is_speech_detected(self):
        return self.speech

    def accept_waveform(self, chunk):
        pass


@pytest.fixture
def fake_build_vad(monkeypatch):
    built = []

    def factory(min_silence=0.35, max_speech=12.0, vad_threshold=0.5):
        vad = _FakeVad(min_silence, max_speech, vad_threshold)
        built.append(vad)
        return vad

    monkeypatch.setattr(rt, "build_vad", factory)
    return built


def test_live_vad_applies_pending_change_at_a_safe_boundary(fake_build_vad):
    live = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    first = fake_build_vad[-1]
    first.speech = False  # not mid-utterance

    live.set_sensitivity(threshold=0.7)
    live.accept_waveform(np.zeros(512, dtype=np.float32))  # one chunk boundary

    assert len(fake_build_vad) == 2, "expected a rebuild once the boundary was safe"
    assert fake_build_vad[-1].threshold == pytest.approx(0.7)
    assert live.current_params()["threshold"] == pytest.approx(0.7)


def test_live_vad_defers_rebuild_while_speech_is_active(fake_build_vad):
    live = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    first = fake_build_vad[-1]
    first.speech = True  # mid-utterance

    live.set_sensitivity(threshold=0.9)
    live.accept_waveform(np.zeros(512, dtype=np.float32))
    assert len(fake_build_vad) == 1, "must not rebuild while a speech segment is in progress"
    assert live.current_params()["threshold"] == pytest.approx(0.5), "old params still active"

    # the segment closes; the NEXT chunk boundary is safe
    first.speech = False
    live.accept_waveform(np.zeros(512, dtype=np.float32))
    assert len(fake_build_vad) == 2
    assert fake_build_vad[-1].threshold == pytest.approx(0.9)


def test_live_vad_set_sensitivity_partial_args_keep_other_values(fake_build_vad):
    live = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    fake_build_vad[-1].speech = False
    live.set_sensitivity(min_silence=0.5)  # threshold/max_speech left alone
    live.accept_waveform(np.zeros(512, dtype=np.float32))
    params = live.current_params()
    assert params == {"threshold": 0.5, "min_silence": 0.5, "max_speech": 12.0}


# --- LiveVad.set_sensitivity: validation (GitHub issue #29 review round 1) --
#
# A bad value must raise ValueError synchronously, from set_sensitivity()
# itself, and never reach self._pending / build_vad() on the decode thread --
# these tests assert both halves: the exception, AND that current_params()
# (and, where relevant, the build count) is completely unchanged afterward.

@pytest.mark.parametrize("kwargs", [
    {"threshold": "abc"},       # wrong type: string
    {"threshold": True},        # wrong type: bool (a subclass of int -- must not pass as 1.0)
    {"threshold": -1},          # out of range: negative
    {"threshold": 0},           # out of range: not > 0
    {"threshold": 5.0},         # out of range: not <= 1
    {"min_silence": "abc"},
    {"min_silence": True},
    {"min_silence": -0.1},
    {"min_silence": 0},         # out of range: not > 0
    {"max_speech": "abc"},
    {"max_speech": True},
    {"max_speech": -1},
    {"max_speech": 0},
])
def test_live_vad_set_sensitivity_rejects_invalid_values(fake_build_vad, kwargs):
    live = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    with pytest.raises(ValueError):
        live.set_sensitivity(**kwargs)
    # rejected outright: nothing queued, nothing rebuilt
    assert live.current_params() == {"threshold": 0.5, "min_silence": 0.35, "max_speech": 12.0}
    assert len(fake_build_vad) == 1


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0.1}, {"threshold": 1.0}, {"threshold": 1},
    {"min_silence": 0.01}, {"min_silence": 5},
    {"max_speech": 0.5}, {"max_speech": 30},
])
def test_live_vad_set_sensitivity_accepts_valid_values(fake_build_vad, kwargs):
    live = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    live.set_sensitivity(**kwargs)  # must not raise


def test_live_vad_set_sensitivity_none_is_always_a_noop_no_change():
    """Every argument defaults to (and explicitly accepts) None, meaning
    "keep current" -- including the JSON-null case from POST /config,
    which apply_config() can't distinguish from "key absent" (payload.get()
    returns None either way). Neither must ever raise or change anything."""
    live = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    live.set_sensitivity(threshold=None, min_silence=None, max_speech=None)  # must not raise


def test_post_config_invalid_vad_value_returns_400_not_a_crash(running_server):
    """End-to-end: a bad "vad" value in POST /config must answer 400 from
    RuntimeControls.apply_config() -> LiveVad.set_sensitivity()'s
    validation, not reach build_vad() on some other thread later."""
    server, port = running_server
    live_vad = rt.LiveVad(min_silence=0.35, max_speech=12.0, vad_threshold=0.5)
    server.controls = ss.RuntimeControls(live_vad=live_vad)

    status, payload = _request(port, "POST", "/config", {"vad": {"threshold": "abc"}})
    assert status == 400
    assert "threshold" in payload["error"]
    assert live_vad.current_params()["threshold"] == pytest.approx(0.5)  # unchanged

    status, payload = _request(port, "POST", "/config", {"vad": {"min_silence": -1}})
    assert status == 400
    assert "min_silence" in payload["error"]


# =============================================================================
# SpeakerLabeler.reset(), AudioHistory.clear(), Refiner.reset()
# =============================================================================

def _make_labeler():
    """Bare SpeakerLabeler instance with no CAM++ model loaded, same
    object.__new__() pattern tests/test_speaker_id.py's make_labeler() uses
    -- reset() only touches the plain-Python bookkeeping attributes."""
    from speaker_id import SpeakerLabeler

    labeler = object.__new__(SpeakerLabeler)
    labeler._centroids = [np.array([1.0, 0.0], dtype=np.float32)]
    labeler._counts = [3]
    labeler._alias = {0: 0}
    labeler._merge_history = {"S2": "S1"}
    labeler._confirmed = [True]
    labeler._open_log = ["fast"]
    return labeler


def test_speaker_labeler_reset_clears_everything():
    labeler = _make_labeler()
    labeler.reset()
    assert labeler._centroids == []
    assert labeler._counts == []
    assert labeler._alias == {}
    assert labeler._merge_history == {}
    assert labeler._confirmed == []
    assert labeler._open_log == []


def test_audio_history_clear_empties_buffer_and_offsets():
    history = rt.AudioHistory(16000, keep_s=5.0)
    history.push(np.ones(8000, dtype=np.float32))
    assert len(history.buf) == 8000
    history.clear()
    assert len(history.buf) == 0
    assert history.offset == 0
    assert history.last_seg_end == 0


def test_refiner_reset_drops_spans_and_clears_history():
    sr = 16000
    history = rt.AudioHistory(sr, keep_s=5.0)
    history.push(np.ones(8000, dtype=np.float32))
    printer = rt.PartialPrinter(enabled=False)
    refiner = rt.Refiner(asr=object(), history=history, sample_rate=sr, printer=printer)
    refiner.spans = [(0, 1000, "ja", "text", "S1")]
    refiner._recluster_entries = [{"group_idx": 0, "local_id": 0, "embedding": None, "duration_s": 1.0}]
    refiner._recluster_group_idx = 3

    refiner.reset()

    assert refiner.spans == []
    assert len(refiner.history.buf) == 0
    assert refiner._recluster_entries == []
    assert refiner._recluster_group_idx == 0


def test_refiner_reset_writes_a_separator_into_an_open_transcript(tmp_path):
    path = tmp_path / "transcript.txt"
    path.write_text("first session line\n", encoding="utf-8")
    sr = 16000
    history = rt.AudioHistory(sr, keep_s=5.0)
    printer = rt.PartialPrinter(enabled=False)
    refiner = rt.Refiner(asr=object(), history=history, sample_rate=sr, printer=printer,
                         transcript_path=str(path))

    refiner.reset()
    refiner._transcript.write("second session line\n")
    refiner._transcript.flush()

    content = path.read_text(encoding="utf-8")
    assert "first session line" in content
    assert "session reset" in content
    assert "second session line" in content
    # the file must still be OPEN after reset (not closed and reopened) --
    # writing to a closed file object raises ValueError
    assert not refiner._transcript.closed


# =============================================================================
# reset_live_session(): ordering and event emission
# =============================================================================

def test_reset_live_session_emits_summary_then_reset_in_order():
    hub = ss.EventHub()
    events = []
    hub.add_listener(events.append)

    stats = rt.SessionStats()
    stats.segments = 5
    stats.total_audio_s = 12.0

    reset_calls = []

    class _StubAsrForReset:
        def reset_session(self):
            reset_calls.append("asr")

    rt.reset_live_session(_StubAsrForReset(), None, None, stats, hub)

    types = [e["type"] for e in events]
    assert types == ["session_summary", "session_reset"]
    assert events[0]["stats"]["segments"] == 5
    assert reset_calls == ["asr"]
    # stats itself must be reset AFTER being summarized, not before
    assert stats.segments == 0


def test_reset_live_session_tolerates_every_optional_piece_being_none():
    # no asr, no refiner, no labeler, no stats, no hub -- must not raise
    rt.reset_live_session(None, None, None, None, None)


def test_reset_live_session_flushes_pending_refine_before_summarizing():
    sr = 16000
    history = rt.AudioHistory(sr, keep_s=5.0)
    history.push(np.zeros(sr, dtype=np.float32))
    printer = rt.PartialPrinter(enabled=False)

    class _FakeAsr:
        forced_lang = None

        def _identify_lang(self, buf, sr):
            return "ja"

        def transcribe(self, buf, sr, known_lang=None, live=False):
            return {"text": "テスト文章です"}

        def reset_session(self):
            pass

    asr = _FakeAsr()
    stats = rt.SessionStats()
    # same stats instance the real main() wiring shares between Refiner and
    # reset_live_session() -- refine_groups_closed is bumped by Refiner
    # itself, synchronously, the moment maybe_refine() decides to actually
    # refine (see maybe_refine()'s docstring/body), which is what proves
    # the flush below ran BEFORE session_summary was captured.
    refiner = rt.Refiner(asr, history, sr, printer, stats=stats)
    refiner.spans = [(0, sr, "ja", "テスト文章です", "")]

    hub = ss.EventHub()
    summaries = []
    hub.add_listener(lambda e: summaries.append(e) if e["type"] == "session_summary" else None)

    rt.reset_live_session(asr, refiner, None, stats, hub)

    # the group that was still pending must have been flushed (closed) by
    # the reset BEFORE session_summary was captured, not silently
    # discarded by refiner.reset()'s own "drop whatever's left" behavior
    assert summaries[0]["stats"]["refine_groups_closed"] == 1
    assert refiner.spans == []
    assert stats.refine_groups_closed == 0  # stats.reset() zeroed it back out afterward


# =============================================================================
# ControlQueue: runs submitted work on the decode thread (review round 1)
# =============================================================================

def test_control_queue_submit_does_not_run_synchronously():
    control = rt.ControlQueue()
    ran = []
    done = control.submit(lambda: ran.append(1))
    assert ran == []
    assert not done.is_set()


def test_control_queue_poll_runs_queued_callables_in_fifo_order():
    control = rt.ControlQueue()
    order = []
    control.submit(lambda: order.append(1))
    control.submit(lambda: order.append(2))
    control.submit(lambda: order.append(3))
    control.poll()
    assert order == [1, 2, 3]


def test_control_queue_poll_with_nothing_queued_is_a_noop():
    control = rt.ControlQueue()
    control.poll()  # must not raise or block


def test_control_queue_poll_sets_the_event_after_running():
    control = rt.ControlQueue()
    done = control.submit(lambda: None)
    assert not done.is_set()
    control.poll()
    assert done.is_set()


def test_control_queue_poll_sets_event_and_continues_past_a_raising_callable(capsys):
    control = rt.ControlQueue()
    order = []

    def boom():
        raise RuntimeError("control task broke")

    done1 = control.submit(boom)
    done2 = control.submit(lambda: order.append("second"))
    control.poll()

    # a broken control task must not leave a caller blocked forever, and
    # must not stop the next queued task from running
    assert done1.is_set()
    assert done2.is_set()
    assert order == ["second"]
    assert "control task broke" in capsys.readouterr().err


def test_control_queue_task_runs_on_the_thread_that_calls_poll_not_the_submitter():
    """The core guarantee this class exists for: a callable submitted from
    one thread (e.g. an HTTP handler, for POST /reset) actually executes
    on whichever thread calls poll() (the decode thread), never on the
    submitter's own thread -- this is what keeps reset_live_session() off
    the HTTP handler thread and out of a race with run_stream()'s loop."""
    control = rt.ControlQueue()
    ran_on = []
    go = threading.Event()

    def task():
        ran_on.append(threading.current_thread())

    def poller():
        go.wait(timeout=3.0)
        control.poll()

    poller_thread = threading.Thread(target=poller)
    poller_thread.start()

    done = control.submit(task)
    assert not done.is_set()  # queued, not yet run

    go.set()
    assert done.wait(timeout=3.0)
    poller_thread.join(timeout=3.0)

    assert ran_on == [poller_thread]
    assert ran_on[0] is not threading.current_thread()


# =============================================================================
# run_stream(): control.poll() integration (review round 1)
# =============================================================================

class _NullVad:
    """Never produces a segment -- these tests are only about whether
    run_stream()'s chunk loop calls control.poll(), not about VAD/ASR
    behavior (same minimal-stub spirit as tests/test_process_safety.py's
    _StubVad)."""

    def accept_waveform(self, chunk):
        pass

    def empty(self):
        return True

    def is_speech_detected(self):
        return False

    def flush(self):
        pass


def test_run_stream_polls_control_once_per_chunk():
    sr = 16000
    n_chunks = 5
    samples = np.zeros(rt.WINDOW_SIZE * n_chunks, dtype=np.float32)
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=False)
    control = rt.ControlQueue()
    poll_calls = []
    original_poll = control.poll
    control.poll = lambda: (poll_calls.append(1), original_poll())

    rt.run_stream(rt.wav_chunks(samples, sr, realtime=False), _NullVad(), sr, None, stats,
                  printer, control=control)

    assert len(poll_calls) == n_chunks


def test_run_stream_without_control_is_unaffected():
    """Regression guard: omitting `control` (every pre-existing caller)
    must behave exactly as before -- no AttributeError, no behavior
    change."""
    sr = 16000
    samples = np.zeros(rt.WINDOW_SIZE * 3, dtype=np.float32)
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=False)
    rt.run_stream(rt.wav_chunks(samples, sr, realtime=False), _NullVad(), sr, None, stats,
                  printer)  # control omitted entirely


def test_run_stream_applies_a_submitted_task_on_the_decode_thread_not_the_submitter():
    """End-to-end version of the review round 1 fix: a task submitted
    from this (the test/"HTTP handler") thread runs on the thread that's
    actually driving run_stream(), not here -- proving a POST /reset
    wired the same way (main()'s RuntimeControls.reset_fn) can no longer
    race the decode loop."""
    sr = 16000
    samples = np.zeros(int(1.0 * sr), dtype=np.float32)
    stats = rt.SessionStats()
    printer = rt.PartialPrinter(enabled=False)
    control = rt.ControlQueue()

    ran_on = []
    control.submit(lambda: ran_on.append(threading.current_thread()))
    assert ran_on == []  # not run synchronously by submit() itself

    def run():
        rt.run_stream(rt.wav_chunks(samples, sr, realtime=False), _NullVad(), sr, None, stats,
                      printer, control=control)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert ran_on == [thread]
    assert ran_on[0] is not threading.current_thread()


# =============================================================================
# SubtitleServer: GET/POST /config, POST /reset via http.client
# =============================================================================

class _StubAsrForControls:
    def __init__(self):
        self.forced_lang = None
        self.dual_confirm = True
        self._punctuate = True
        self.lid_switch_confirm = 2
        self.min_switch_s = 2.0
        self.set_calls = []

    def set_forced_lang(self, lang):
        if lang not in (None, "ja", "en"):
            raise ValueError(f"unsupported language code: {lang!r}")
        self.forced_lang = lang
        self.set_calls.append(("lang", lang))

    def set_dual_confirm(self, enabled):
        self.dual_confirm = enabled

    def set_punctuate(self, enabled):
        self._punctuate = enabled

    def set_lid_switch_confirm(self, n):
        if n < 1:
            raise ValueError("lid_switch_confirm must be >= 1")
        self.lid_switch_confirm = n

    def set_min_switch_s(self, s):
        if s < 0:
            raise ValueError("min_switch_s must be >= 0")
        self.min_switch_s = s


@pytest.fixture
def running_server():
    hub = ss.EventHub()
    server = ss.SubtitleServer(port=0, hub=hub).start()
    port = server._httpd.server_address[1]
    yield server, port
    server._httpd.shutdown()
    server._httpd.server_close()


def _request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    payload = json.loads(resp.read().decode("utf-8"))
    conn.close()
    return resp.status, payload


def test_config_and_reset_404_without_controls_attached(running_server):
    server, port = running_server
    status, payload = _request(port, "GET", "/config")
    assert status == 404
    status, payload = _request(port, "POST", "/reset")
    assert status == 404


def test_get_config_reports_current_values(running_server):
    server, port = running_server
    asr = _StubAsrForControls()
    pool = rt.TranslatorPool()
    pool._translators["en"] = _FakeTranslator({})
    pool._order.append("en")
    server.controls = ss.RuntimeControls(asr=asr, translator_pool=pool)

    status, payload = _request(port, "GET", "/config")
    assert status == 200
    assert payload == {
        "lang": None, "dual_confirm": True, "punctuate": True,
        "lid_switch_confirm": 2, "min_switch_s": 2.0, "translate": ["en"], "vad": None,
    }


def test_post_config_applies_a_subset_of_keys(running_server):
    server, port = running_server
    asr = _StubAsrForControls()
    server.controls = ss.RuntimeControls(asr=asr)

    status, payload = _request(port, "POST", "/config", {"lang": "en", "dual_confirm": False})
    assert status == 200 and payload == {"ok": True}
    assert asr.forced_lang == "en"
    assert asr.dual_confirm is False


def test_post_config_invalid_value_returns_400_with_message(running_server):
    server, port = running_server
    asr = _StubAsrForControls()
    server.controls = ss.RuntimeControls(asr=asr)

    status, payload = _request(port, "POST", "/config", {"lang": "not-real"})
    assert status == 400
    assert "not-real" in payload["error"]
    assert asr.forced_lang is None  # rejected value never applied


def test_post_reset_calls_the_reset_function(running_server):
    server, port = running_server
    calls = []

    def reset_fn():
        calls.append(True)
        return True  # completed synchronously (see reset_fn's own contract)

    server.controls = ss.RuntimeControls(reset_fn=reset_fn)

    status, payload = _request(port, "POST", "/reset")
    assert status == 200 and payload == {"ok": True}
    assert calls == [True]


def test_post_reset_pending_returns_202(running_server):
    """reset_fn reporting False (still queued -- e.g. main()'s real
    wiring when a submitted ControlQueue task hasn't been polled by the
    decode thread within its timeout, GitHub issue #29 review round 1)
    must answer 202, not 200 and not a 4xx/5xx failure."""
    server, port = running_server
    server.controls = ss.RuntimeControls(reset_fn=lambda: False)

    status, payload = _request(port, "POST", "/reset")
    assert status == 202
    assert payload == {"ok": False, "pending": True}


def test_post_reset_without_reset_fn_returns_400(running_server):
    server, port = running_server
    server.controls = ss.RuntimeControls()  # attached, but no reset_fn wired

    status, payload = _request(port, "POST", "/reset")
    assert status == 400
