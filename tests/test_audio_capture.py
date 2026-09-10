"""Hardware-free regression tests for loopback capture and mixed input."""
import builtins
import io
import os
import queue
import sys
import threading
import time
import types
from collections import deque
from contextlib import contextmanager

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import realtime_transcribe as rt  # noqa: E402


class _RuntimeSys:
    """Keep test overrides local while following pytest's current streams."""

    def __getattr__(self, name):
        return getattr(sys, name)


@pytest.fixture(autouse=True)
def isolated_runtime_sys(monkeypatch):
    # Patching the real sys.platform breaks NumPy's lazy imports on Linux.
    monkeypatch.setattr(rt, "sys", _RuntimeSys())


@pytest.fixture(params=["linux", "win32"])
def fake_mix_backend(monkeypatch, request, isolated_runtime_sys):
    # Fake generators also need a fake caller-thread SoundCard preload.
    monkeypatch.setattr(rt.sys, "platform", request.param)
    monkeypatch.setattr(rt, "_import_soundcard", lambda: None)


@pytest.mark.parametrize("input_mode", ["speaker", "mix"])
def test_cli_rejects_loopback_on_non_windows_before_loading_models(monkeypatch, input_mode):
    monkeypatch.setattr(rt.sys, "platform", "linux")
    monkeypatch.setattr(rt.sys, "argv", ["realtime_transcribe.py", "--input", input_mode])
    output = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(rt.sys, "stdout", output)
    monkeypatch.setattr(rt, "RoutedASR", lambda **kwargs: pytest.fail("loaded ASR"))
    with pytest.raises(SystemExit) as exc:
        rt.main()
    assert exc.value.code == 2


def test_microphone_device_selection_uses_input_only_case_insensitive_match(monkeypatch):
    devices = [
        {"name": "USB output", "max_input_channels": 0},
        {"name": "USB microphone", "max_input_channels": 1},
        {"name": "Other mic", "max_input_channels": 1},
    ]
    fake_sd = types.SimpleNamespace(
        query_devices=lambda: devices,
        default=types.SimpleNamespace(device=[2, 0]),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert rt._resolve_mic_device("usb") == 1
    assert rt._resolve_mic_device() == 2
    fake_sd.default.device[0] = -1
    assert rt._resolve_mic_device() == 1
    with pytest.raises(RuntimeError, match="no input device matching"):
        rt._resolve_mic_device("missing")


@pytest.mark.parametrize("shape", [(512,), (512, 1), (511, 2)])
def test_loopback_rejects_corrupt_channel_or_window_shapes(shape):
    with pytest.raises(RuntimeError, match="WASAPI loopback returned"):
        rt._downmix_loopback(np.zeros(shape, dtype=np.float32))


class _FakeOle32:
    def __init__(self, result):
        self.result = result
        self.initialized = []
        self.uninitialized = 0

    def CoInitializeEx(self, reserved, mode):
        self.initialized.append((reserved, mode))
        return self.result

    def CoUninitialize(self):
        self.uninitialized += 1


@pytest.mark.parametrize("result", [0, 1])
def test_windows_com_scope_balances_every_success_result(result):
    ole32 = _FakeOle32(result)
    with rt._windows_com_initialized(ole32):
        assert ole32.initialized == [(None, 0)]
    assert ole32.uninitialized == 1


def test_windows_com_scope_uses_existing_changed_mode_apartment():
    ole32 = _FakeOle32(0x80010106)  # RPC_E_CHANGED_MODE
    with rt._windows_com_initialized(ole32):
        pass
    assert ole32.uninitialized == 0


def test_windows_com_scope_rejects_other_hresult_failures():
    ole32 = _FakeOle32(0x80004005)  # E_FAIL
    with pytest.raises(RuntimeError, match="0x80004005"):
        with rt._windows_com_initialized(ole32):
            pytest.fail("a failed CoInitializeEx must not enter the scope")
    assert ole32.uninitialized == 0


def test_speaker_capture_owns_com_objects_and_downmixes_native_channels(monkeypatch):
    main_thread = threading.get_ident()
    thread_state = threading.local()
    calls = []
    release_second_record = threading.Event()

    def require_com(operation):
        assert getattr(thread_state, "com_active", False), operation
        calls.append((operation, threading.get_ident()))

    @contextmanager
    def fake_com_scope():
        calls.append(("com_enter", threading.get_ident()))
        thread_state.com_active = True
        try:
            yield
        finally:
            require_com("com_exit")
            thread_state.com_active = False

    class Speaker:
        @property
        def id(self):
            require_com("speaker_id")
            return "speaker-id"

        @property
        def name(self):
            require_com("speaker_name")
            return "Fake Speakers"

    class Recorder:
        records = 0

        def __enter__(self):
            require_com("recorder_enter")
            return self

        def __exit__(self, *exc_info):
            require_com("recorder_exit")
            return False

        def record(self, numframes):
            require_com("record")
            assert numframes == rt.WINDOW_SIZE
            self.records += 1
            if self.records > 1:
                release_second_record.wait(timeout=1.0)
            left = np.full(numframes, 0.2, dtype=np.float32)
            right = np.full(numframes, 0.6, dtype=np.float32)
            return np.column_stack([left, right])

    class Microphone:
        @property
        def channels(self):
            require_com("microphone_channels")
            return 2

        def recorder(self, **kwargs):
            require_com("recorder_create")
            calls.append(("recorder_kwargs", kwargs))
            return Recorder()

    def default_speaker():
        require_com("default_speaker")
        return Speaker()

    def get_microphone(*, id, include_loopback):
        require_com("get_microphone")
        assert id == "speaker-id"
        assert include_loopback is True
        return Microphone()

    class FakeWarning(RuntimeWarning):
        pass

    fake_sc = types.SimpleNamespace(
        default_speaker=default_speaker,
        all_speakers=lambda: [Speaker()],
        get_microphone=get_microphone,
        SoundcardRuntimeWarning=FakeWarning,
    )
    monkeypatch.setattr(rt.sys, "platform", "win32")
    monkeypatch.setattr(rt, "_windows_com_initialized", fake_com_scope)
    monkeypatch.setitem(sys.modules, "soundcard", fake_sc)

    stop = threading.Event()
    chunks = rt.speaker_chunks(stop_event=stop, _timestamped=True)
    packet = next(chunks)
    assert isinstance(packet, rt._CapturedChunk)
    assert packet.end_ns > 0
    chunk = packet.samples
    np.testing.assert_allclose(chunk, np.full(rt.WINDOW_SIZE, 0.4, dtype=np.float32))

    stop.set()
    release_second_record.set()
    with pytest.raises(StopIteration):
        next(chunks)

    worker_ids = {thread_id for operation, thread_id in calls
                  if operation != "recorder_kwargs"}
    assert worker_ids == {next(iter(worker_ids))}
    assert main_thread not in worker_ids
    kwargs = next(value for operation, value in calls if operation == "recorder_kwargs")
    assert kwargs == {
        "samplerate": rt.SAMPLE_RATE,
        "channels": 2,
        "blocksize": rt.LOOPBACK_BLOCK_SIZE,
    }
    assert kwargs["blocksize"] > rt.WINDOW_SIZE
    operations = [operation for operation, _ in calls]
    assert operations.index("com_enter") < operations.index("default_speaker")
    assert operations.index("recorder_exit") < operations.index("com_exit")


def test_speaker_capture_rejects_non_windows_before_import_or_com(monkeypatch):
    monkeypatch.setattr(rt.sys, "platform", "linux")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "soundcard":
            pytest.fail("the non-Windows guard must run before importing soundcard")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        rt, "_windows_com_initialized",
        lambda: pytest.fail("the non-Windows path must not initialize COM"))

    with pytest.raises(RuntimeError, match="needs Windows"):
        next(rt.speaker_chunks())


def test_mic_timestamp_is_taken_in_the_capture_callback(monkeypatch):
    callback_thread = []

    class Channel:
        def copy(self):
            return np.full(rt.WINDOW_SIZE, 0.25, dtype=np.float32)

    class InputData:
        def __getitem__(self, key):
            assert key == (slice(None), 0)
            return Channel()

    class InputStream:
        def __init__(self, *args, callback=None, **kwargs):
            self.callback = callback

        def __enter__(self):
            callback_thread.append(threading.get_ident())
            self.callback(InputData(), rt.WINDOW_SIZE, None, None)
            return self

        def __exit__(self, *exc_info):
            return False

    fake_sd = types.SimpleNamespace(
        InputStream=InputStream,
        query_devices=lambda *args: ({"name": "fake mic", "max_input_channels": 1}
                                     if args else
                                     [{"name": "fake mic", "max_input_channels": 1}]),
        default=types.SimpleNamespace(device=[0, 0]),
        PortAudioError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stop = threading.Event()
    chunks = rt.mic_chunks(stop_event=stop, _timestamped=True)
    packet = next(chunks)
    stop.set()
    with pytest.raises(StopIteration):
        next(chunks)

    assert isinstance(packet, rt._CapturedChunk)
    assert packet.end_ns > 0
    np.testing.assert_allclose(packet.samples, 0.25)
    assert callback_thread == [threading.get_ident()]


def test_speaker_capture_failure_is_reported_with_endpoint_context(monkeypatch):
    @contextmanager
    def fake_com_scope():
        yield

    def default_speaker():
        raise RuntimeError("device vanished")

    fake_sc = types.SimpleNamespace(
        default_speaker=default_speaker,
        all_speakers=lambda: [],
        get_microphone=lambda **kwargs: None,
        SoundcardRuntimeWarning=RuntimeWarning,
    )
    monkeypatch.setattr(rt.sys, "platform", "win32")
    monkeypatch.setattr(rt, "_windows_com_initialized", fake_com_scope)
    monkeypatch.setitem(sys.modules, "soundcard", fake_sc)

    with pytest.raises(RuntimeError, match="default output device.*device vanished"):
        next(rt.speaker_chunks())


def _packet(end_ns, value):
    return rt._CapturedChunk(
        end_ns, np.full(rt.WINDOW_SIZE, value, dtype=np.float32))


def test_capture_timestamps_keep_audio_spacing_when_backend_returns_a_burst(monkeypatch):
    readings = iter([1_000_000_000, 1_000_000_001, 2_000_000_000])
    monkeypatch.setattr(rt.time, "perf_counter_ns", lambda: next(readings))

    first = rt._next_capture_end_ns(None)
    second = rt._next_capture_end_ns(first)
    third = rt._next_capture_end_ns(second)

    assert first == 1_000_000_000
    assert second == first + rt.CAPTURE_CHUNK_NS
    assert third == max(2_000_000_000, second + rt.CAPTURE_CHUNK_NS)


def test_put_latest_keeps_only_newest_bounded_items():
    items = queue.Queue(maxsize=3)
    for value in range(8):
        rt._put_latest(items, value)
    assert [items.get_nowait() for _ in range(3)] == [5, 6, 7]


def test_aligned_speaker_match_discards_stale_and_retains_future():
    stale = _packet(70, 0.1)
    near = _packet(98, 0.2)
    less_near = _packet(94, 0.3)
    future = _packet(130, 0.4)
    pending = deque([stale, less_near, near, future])

    chosen = rt._pop_aligned_speaker(pending, target_ns=100, tolerance_ns=10)

    assert chosen is near
    assert list(pending) == [future]
    assert rt._pop_aligned_speaker(pending, target_ns=100, tolerance_ns=10) is None
    assert list(pending) == [future], "far-future audio must remain for a later mic frame"


def test_wait_for_speaker_drains_packet_enqueued_as_done_becomes_true():
    target = time.perf_counter_ns()
    packets = queue.Queue()
    pending = deque()
    final_packet = _packet(target, 0.4)

    class Done:
        def is_set(self):
            packets.put_nowait(final_packet)
            return True

    chosen = rt._wait_for_aligned_speaker(
        packets, pending, Done(), target_ns=target)

    assert chosen is final_packet


def test_mix_stall_drops_both_sides_symmetrically_and_keeps_time_alignment(
        monkeypatch, fake_mix_backend):
    gate = threading.Event()
    mic_finished = threading.Event()
    speaker_finished = threading.Event()
    speaker_first_queued = threading.Event()
    base = time.perf_counter_ns()

    def fake_mic_chunks(*, stop_event, device_name, _timestamped):
        assert _timestamped is True
        assert speaker_first_queued.wait(5.0)
        yield _packet(base, 0.00)
        gate.wait(timeout=5.0)
        for index in range(1, 8):
            yield _packet(base + index * 10_000_000, index * 0.10)
        mic_finished.set()

    def fake_speaker_chunks(*, stop_event, device_name, _timestamped):
        assert _timestamped is True
        yield _packet(base, 0.01)
        # The drain loop put the yielded packet before requesting the next
        # generator value and reaching this line.
        speaker_first_queued.set()
        gate.wait(timeout=5.0)
        for index in range(1, 8):
            yield _packet(base + index * 10_000_000, index * 0.01)
        speaker_finished.set()

    monkeypatch.setattr(rt, "MIX_QUEUE_MAXLEN", 3)
    monkeypatch.setattr(rt, "mic_chunks", fake_mic_chunks)
    monkeypatch.setattr(rt, "speaker_chunks", fake_speaker_chunks)

    mixed = rt.mix_chunks()
    first = next(mixed)
    gate.set()
    assert mic_finished.wait(5.0)
    assert speaker_finished.wait(5.0)

    after_stall = [next(mixed) for _ in range(3)]
    with pytest.raises(StopIteration):
        next(mixed)

    np.testing.assert_allclose(first, 0.01)
    for actual, expected in zip(after_stall, [0.55, 0.66, 0.77]):
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_mix_waits_for_a_closer_packet_instead_of_consuming_older_candidate(
        monkeypatch, fake_mix_backend):
    target = time.perf_counter_ns() + 10_000_000_000
    old_was_queued = threading.Event()
    release_exact = threading.Event()
    result_ready = threading.Event()
    result = []

    def fake_mic_chunks(*, stop_event, device_name, _timestamped):
        yield _packet(target, 0.10)

    def fake_speaker_chunks(*, stop_event, device_name, _timestamped):
        yield _packet(target - 20_000_000, 0.20)
        # The drain loop has already put the yielded packet before asking the
        # generator for its next value and reaching this line.
        old_was_queued.set()
        release_exact.wait(timeout=5.0)
        yield _packet(target, 0.40)

    monkeypatch.setattr(rt, "mic_chunks", fake_mic_chunks)
    monkeypatch.setattr(rt, "speaker_chunks", fake_speaker_chunks)
    mixed = rt.mix_chunks()

    def consume_one():
        result.append(next(mixed))
        result_ready.set()

    consumer = threading.Thread(target=consume_one, daemon=True)
    consumer.start()
    assert old_was_queued.wait(5.0)
    assert not result_ready.wait(0.05), "mix consumed the older candidate too early"
    release_exact.set()
    assert result_ready.wait(5.0)
    consumer.join(timeout=5.0)
    mixed.close()

    np.testing.assert_allclose(result[0], 0.50, atol=1e-6)


def test_mix_does_not_retry_failed_soundcard_import_on_capture_thread(
        monkeypatch, capsys):
    base = time.perf_counter_ns()

    def failed_import():
        raise RuntimeError("S_FALSE rejected")

    def fake_mic_chunks(*, stop_event, device_name, _timestamped):
        yield _packet(base, 0.25)

    def speaker_must_not_start(**kwargs):
        pytest.fail("a failed caller-thread import must not be retried")
        yield  # pragma: no cover

    monkeypatch.setattr(rt.sys, "platform", "win32")
    monkeypatch.setattr(rt, "_import_soundcard", failed_import)
    monkeypatch.setattr(rt, "mic_chunks", fake_mic_chunks)
    monkeypatch.setattr(rt, "speaker_chunks", speaker_must_not_start)

    assert [float(chunk[0]) for chunk in rt.mix_chunks()] == [0.25]
    err = capsys.readouterr().err
    assert err.count("S_FALSE rejected") == 1


def test_mix_speaker_failure_falls_back_once_to_mic_only(
        monkeypatch, capsys, fake_mix_backend):
    base = time.perf_counter_ns()

    def fake_mic_chunks(*, stop_event, device_name, _timestamped):
        yield _packet(base, 0.25)
        yield _packet(base + 10_000_000, 0.50)

    def broken_speaker_chunks(*, stop_event, device_name, _timestamped):
        raise RuntimeError("loopback lost")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(rt, "mic_chunks", fake_mic_chunks)
    monkeypatch.setattr(rt, "speaker_chunks", broken_speaker_chunks)

    assert [float(chunk[0]) for chunk in rt.mix_chunks()] == [0.25, 0.5]
    assert capsys.readouterr().err.count("loopback lost") == 1


def test_mix_microphone_failure_is_fatal_and_stops_speaker(monkeypatch, fake_mix_backend):
    speaker_stopped = threading.Event()

    def broken_mic_chunks(*, stop_event, device_name, _timestamped):
        raise ValueError("microphone lost")
        yield  # pragma: no cover - makes this a generator

    def waiting_speaker_chunks(*, stop_event, device_name, _timestamped):
        while not stop_event.wait(0.01):
            if False:
                yield None
        speaker_stopped.set()

    monkeypatch.setattr(rt, "mic_chunks", broken_mic_chunks)
    monkeypatch.setattr(rt, "speaker_chunks", waiting_speaker_chunks)

    with pytest.raises(ValueError, match="microphone lost"):
        next(rt.mix_chunks())
    assert speaker_stopped.wait(1.0)
