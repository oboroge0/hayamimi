"""Real-time (or simulated real-time) multilingual transcription pipeline.

Audio -> Silero VAD (sherpa-onnx) -> RoutedASR (asr_engine.RoutedASR) -> print.

Usage:
    python scripts/realtime_transcribe.py --wav testdata/ja_test.wav --no-realtime
    python scripts/realtime_transcribe.py --wav testdata/ja_test.wav       # paced with sleeps
    python scripts/realtime_transcribe.py                                 # live microphone
"""
import argparse
import os
import queue
import sys
import threading
import time
import wave
from typing import Callable

import numpy as np
import sherpa_onnx

from asr_engine import ModelUnavailable, RoutedASR
from audio_utils import resample_linear
# subtitle_server.py only imports stdlib modules, so this is safe to import
# unconditionally even under .github/workflows/test.yml's CI install list
# (sherpa-onnx numpy scipy soundfile pytest, nothing else) -- GitHub issue
# #29 needs the EventHub available whether or not --serve's HTTP layer is
# requested.
from subtitle_server import EventHub, RuntimeControls, SubtitleServer

SAMPLE_RATE = 16000
WINDOW_SIZE = 512  # samples per VAD chunk, ~32ms @ 16kHz
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
VAD_MODEL = os.path.join(MODELS_DIR, "silero_vad.onnx")


def read_wave(path: str, target_rate: int = SAMPLE_RATE):
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2, f"{path}: expected 16-bit PCM"
        num_channels = f.getnchannels()
        sample_rate = f.getframerate()
        data = f.readframes(f.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)
    if sample_rate != target_rate:
        samples = resample_linear(samples, sample_rate, target_rate)
        sample_rate = target_rate
    return samples, sample_rate


def build_vad(min_silence: float = 0.35,
              max_speech: float = 12.0,
              vad_threshold: float = 0.5) -> sherpa_onnx.VoiceActivityDetector:
    # 0.35s endpointing measured CER-neutral vs 0.5s on real broadcast ja
    # (docs/BENCHMARKS.md iteration 9) and finalizes 150ms sooner. max_speech
    # force-splits breathless monologues (radio/game commentary hit 21s
    # segments) so finals stay timely; the refine pass re-merges the group.
    # vad_threshold is Silero's own speech-probability cutoff (sherpa_onnx's
    # SileroVadModelConfig default is 0.5). docs/DIARIZATION_PLAN.md section
    # 13 (Round 3) swept 0.40/0.30/0.20 on the AMI eval set: miss did drop
    # as expected, but confusion grew far more (offline diarization gets
    # more/noisier low-energy segments to cluster), so mean DER got *worse*
    # at every value tried (14.1% baseline -> 15.6-16.5%). Rejected --
    # this default (0.5) keeps current production behavior unchanged.
    # The installed sherpa_onnx (1.13.6) SileroVadModelConfig exposes no
    # speech-padding knob (no speech_pad_ms field) alongside threshold, so
    # there is nothing to plumb through for that half of T1.
    if not os.path.exists(VAD_MODEL):
        # sherpa-onnx's C++ layer calls the process's exit() -- not a
        # catchable Python exception -- when handed a missing model path
        # (see asr_engine._KEY_FILES' comment for the same landmine on the
        # recognizer side). Check first so an embedding app gets a
        # catchable ModelUnavailable instead of the whole process dying.
        raise ModelUnavailable("vad")
    cfg = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=VAD_MODEL,
            threshold=vad_threshold,
            min_silence_duration=min_silence,
            min_speech_duration=0.25,
            window_size=WINDOW_SIZE,
            max_speech_duration=max_speech,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=1,
    )
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)


class LiveVad:
    """Wraps the live-capture VAD so its sensitivity can change at runtime
    (GitHub issue #29's `POST /config` "vad" key).

    sherpa-onnx's VoiceActivityDetector exposes no setters for its config
    (see build_vad()'s docstring on vad_threshold) -- the only way to
    change min_silence/max_speech/threshold is to construct a brand new
    detector. Doing that while a speech segment is in progress would throw
    away that segment's accumulated audio and reset the endpointer mid-
    utterance, so a set_sensitivity() request is only actually applied at
    the next safe point: the next accept_waveform() call (one per audio
    chunk, called from run_stream()'s loop) that finds the detector NOT
    currently inside a detected speech segment. A request made mid-speech
    just waits until that segment closes.

    Implements the subset of sherpa_onnx.VoiceActivityDetector's API that
    run_stream()/drain_segments() actually use, so `LiveVad(...)` is a
    drop-in replacement for `build_vad(...)` at every call site.
    """

    def __init__(self, min_silence: float = 0.35, max_speech: float = 12.0,
                 vad_threshold: float = 0.5):
        self._min_silence = min_silence
        self._max_speech = max_speech
        self._threshold = vad_threshold
        self._vad = build_vad(min_silence, max_speech, vad_threshold)
        self._pending: dict | None = None  # requested-but-not-yet-applied params
        self._lock = threading.Lock()

    def current_params(self) -> dict:
        """Currently ACTIVE sensitivity (not a pending, not-yet-applied
        request -- see the class docstring), for GET /config."""
        return {"threshold": self._threshold, "min_silence": self._min_silence,
                "max_speech": self._max_speech}

    def set_sensitivity(self, threshold: float | None = None,
                        min_silence: float | None = None,
                        max_speech: float | None = None) -> None:
        """Request new sensitivity parameters. Any argument left as None
        keeps its current value (including an explicit JSON `null` coming
        through POST /config -- payload.get() can't tell "key absent" from
        "key present but null" apart, and both mean "no change" here,
        which is what a caller sending a partial `vad` object expects).
        Applied at the next safe chunk boundary (see the class docstring),
        not synchronously.

        Validated HERE, synchronously, so a bad value (GitHub issue #29
        review round 1: POST /config {"vad": {"threshold": "abc"}}, or a
        negative/zero/out-of-range number) raises ValueError immediately
        and is never queued into self._pending. The alternative --
        validating only when _maybe_apply_pending() later hands the value
        to build_vad() -- would crash the DECODE thread arbitrarily long
        after the HTTP request that caused it had already returned 200.
        """
        if threshold is not None:
            _require_number("threshold", threshold)
            if not (0 < threshold <= 1):
                raise ValueError(f"threshold must be > 0 and <= 1, got {threshold!r}")
        if min_silence is not None:
            _require_number("min_silence", min_silence)
            if not (min_silence > 0):
                raise ValueError(f"min_silence must be > 0, got {min_silence!r}")
        if max_speech is not None:
            _require_number("max_speech", max_speech)
            if not (max_speech > 0):
                raise ValueError(f"max_speech must be > 0, got {max_speech!r}")
        with self._lock:
            self._pending = {
                "threshold": self._threshold if threshold is None else threshold,
                "min_silence": self._min_silence if min_silence is None else min_silence,
                "max_speech": self._max_speech if max_speech is None else max_speech,
            }

    def _maybe_apply_pending(self) -> None:
        with self._lock:
            if self._pending is None:
                return
            if self._vad.is_speech_detected():
                return  # mid-speech: wait for this segment to close
            params = self._pending
            self._pending = None
        self._threshold = params["threshold"]
        self._min_silence = params["min_silence"]
        self._max_speech = params["max_speech"]
        self._vad = build_vad(self._min_silence, self._max_speech, self._threshold)

    # --- the subset of sherpa_onnx.VoiceActivityDetector's API used by
    # run_stream()/drain_segments() ---

    def accept_waveform(self, chunk) -> None:
        self._maybe_apply_pending()  # checked once per chunk: the "next safe point"
        self._vad.accept_waveform(chunk)

    def empty(self) -> bool:
        return self._vad.empty()

    @property
    def front(self):
        return self._vad.front

    def pop(self):
        return self._vad.pop()

    def is_speech_detected(self) -> bool:
        return self._vad.is_speech_detected()

    @property
    def current_segment(self):
        return self._vad.current_segment

    def flush(self):
        return self._vad.flush()


def _require_number(name: str, value) -> None:
    """bool is a subclass of int in Python (isinstance(True, int) is
    True), so it's excluded explicitly -- {"threshold": true} must be
    rejected, not silently accepted as 1.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")


RESET_TIMEOUT_S = 10.0  # how long POST /reset waits for a chunk boundary before answering 202


class ControlQueue:
    """Runs submitted callables on the DECODE thread instead of the
    caller's own thread (GitHub issue #29 review round 1).

    Session-mutating operations like reset_live_session() touch state the
    decode thread (run_stream()'s loop -- drain_segments(),
    Refiner.maybe_refine(), SpeakerLabeler.match_embedding()) reads and
    writes with no locking of its own, by design: none of those were built
    to be thread-safe against a concurrent mutation, the same way the raw
    sherpa VAD object LiveVad wraps isn't (see that class's own docstring
    for why IT defers instead of mutating in place). Running
    reset_live_session() directly on an HTTP handler thread (POST /reset)
    would race the decode thread -- SpeakerLabeler.reset() clearing
    _centroids/_counts mid-match_embedding() can raise IndexError there,
    and two threads both calling Refiner.maybe_refine() at once race on
    self.spans with nothing serializing them.

    submit(fn) queues a zero-argument callable and returns a
    threading.Event that is set once `fn` has actually run, ON the decode
    thread, via poll(). Any exception `fn` raises is caught, printed
    (never silently lost), and the Event is still set regardless -- a
    broken control operation must not leave a caller blocked on
    Event.wait() forever, the same "never let a control-plane problem
    break the decode loop, but never let it vanish silently either"
    principle RoutedASR._emit()/EventHub.publish() apply on the
    event-publishing side (see those methods' docstrings).

    poll() is meant to be called once per chunk by run_stream() -- the
    same "next safe point" granularity LiveVad's own deferred VAD rebuild
    uses -- and drains and runs every callable queued so far, in
    submission (FIFO) order. If nothing is currently driving run_stream()
    (no --wav/--input mic/ws in progress, or a --input ws session with no
    client sending audio -- see reset_live_session()'s docstring for what
    that means for POST /reset in that case), nobody calls poll() and a
    submitted callable simply sits queued until something does.
    """

    def __init__(self):
        self._q: "queue.Queue[tuple[Callable[[], None], threading.Event]]" = queue.Queue()

    def submit(self, fn: Callable[[], None]) -> threading.Event:
        done = threading.Event()
        self._q.put((fn, done))
        return done

    def poll(self) -> None:
        while True:
            try:
                fn, done = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception:
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
            finally:
                done.set()


def wav_chunks(samples: np.ndarray, sample_rate: int, realtime: bool):
    pos = 0
    n = len(samples)
    start = time.perf_counter()
    while pos < n:
        chunk = samples[pos:pos + WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            chunk = np.pad(chunk, (0, WINDOW_SIZE - len(chunk)))
        if realtime:
            # absolute-deadline pacing: naive per-chunk sleeps accumulate
            # ~15% drift on Windows (15.6ms timer granularity)
            delay = start + pos / sample_rate - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        yield chunk
        pos += WINDOW_SIZE


MIC_QUEUE_TIMEOUT_S = 0.1  # how often the generator wakes up to check stop_event


def mic_chunks(stop_event: "threading.Event | None" = None):
    """Yield mic input frames until `stop_event` is set.

    `q.get()` alone blocks forever with no chunk to hand back control to the
    caller, so an embedding app has no way to break out short of
    KeyboardInterrupt. Polling with a bounded timeout lets the generator
    notice stop_event between callbacks instead.
    """
    import sounddevice as sd

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=WINDOW_SIZE, callback=callback):
        while stop_event is None or not stop_event.is_set():
            try:
                yield q.get(timeout=MIC_QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue


def ws_chunks(ingest):
    """Re-chunk a ws_ingest.IngestServer's variable-sized network reads into
    fixed WINDOW_SIZE frames, the shape the VAD expects.

    Blocks on ingest.audio_q, same as mic_chunks() blocks on sounddevice's
    queue -- if the client disconnects, this just idles until the next one
    connects and starts pushing audio again; the pipeline stays alive.

    ingest.audio_q can also carry a non-ndarray "flush" sentinel (see
    ws_ingest.FLUSH), pushed when a streaming client disconnects. It is
    passed straight through to run_stream(), which treats any non-ndarray
    item as "flush the in-progress VAD segment now" -- otherwise a segment
    left open when the client vanishes would sit unfinalized forever.
    """
    leftover = np.zeros(0, dtype=np.float32)
    while True:
        item = ingest.audio_q.get()
        if not isinstance(item, np.ndarray):
            if len(leftover):
                yield np.pad(leftover, (0, WINDOW_SIZE - len(leftover)))
                leftover = np.zeros(0, dtype=np.float32)
            yield item
            continue
        leftover = np.concatenate([leftover, item])
        while len(leftover) >= WINDOW_SIZE:
            yield leftover[:WINDOW_SIZE]
            leftover = leftover[WINDOW_SIZE:]


PARTIAL_EVERY_S = 0.5   # decode a draft this often (in audio time) during speech
PARTIAL_WINDOW_S = 8.0  # cap draft decoding to the last N seconds of the utterance


class PartialPrinter:
    """Shows in-progress drafts; overwrites in place on a tty, one line otherwise."""

    def __init__(self, enabled: bool, hub: "EventHub | None" = None):
        self.enabled = enabled
        # main() always has a real hub to pass (GitHub issue #29: structured
        # events must flow whether or not --serve's HTTP layer exists); a
        # throwaway one here just keeps every other caller (existing tests,
        # ad-hoc scripts) working unchanged without having to construct one.
        self.hub = hub if hub is not None else EventHub()
        self._tty = sys.stdout.isatty()
        self._last_len = 0

    def show(self, text: str):
        if not self.enabled or not text:
            return
        self.hub.partial(text)
        if self._tty:
            pad = max(self._last_len - len(text), 0)
            print("\r~ " + text + " " * pad, end="", flush=True)
            self._last_len = len(text)
        else:
            print(f"~ {text}", flush=True)

    def clear(self):
        if self.enabled and self._tty and self._last_len:
            print("\r" + " " * (self._last_len + 2) + "\r", end="", flush=True)
            self._last_len = 0


class SessionStats:
    def __init__(self):
        self.total_audio_s = 0.0
        self.segments = 0
        self.latencies_ms: list[float] = []
        self.refine_lang_corrections = 0  # times the refine pass overruled the fast-path language
        # docs/DIARIZATION_PLAN.md section 10.6 diagnostics: eval_diar.py's
        # generate_diarize_hypothesis() groups VAD segments purely on the
        # silence-gap/max-length "due" condition (see group_segments()'s
        # docstring) and has no decoded text to split groups on a language
        # change, but production's Refiner.add_span() also force-flushes a
        # group the moment the (script-corrected) language differs from the
        # group in progress. Every extra group is an extra GroupDiarizer
        # call and an extra round of remap-path match_embedding() calls
        # (each a fresh chance to open a new global centroid at
        # remap_threshold), so counting groups actually closed in
        # production vs. how many were closed specifically because of a
        # language-boundary split (rather than silence/GROUP_MAX_S) tells
        # us whether over-grouping is a real contributor to the S1..S13
        # overcount, independent of eval_diar.py's simplified replica.
        self.refine_groups_closed = 0
        self.refine_lang_boundary_flushes = 0
        # The previous `final` event's language, so drain_segments() can
        # compute that event's `switched` field (GitHub issue #29's wire
        # format: true when this final's lang differs from the previous
        # final's, false for the first). Reset alongside everything else
        # so a new session after reset_live_session() doesn't report its
        # own first final as "switched" from the old session's last one.
        self.last_final_lang: str | None = None

    def summary(self) -> str:
        if not self.latencies_ms:
            return f"total_audio={self.total_audio_s:.1f}s segments=0"
        mean = sum(self.latencies_ms) / len(self.latencies_ms)
        return (f"total_audio={self.total_audio_s:.1f}s segments={self.segments} "
                f"mean_latency={mean:.0f}ms max_latency={max(self.latencies_ms):.0f}ms "
                f"refine_lang_corrections={self.refine_lang_corrections} "
                f"refine_groups_closed={self.refine_groups_closed} "
                f"refine_lang_boundary_flushes={self.refine_lang_boundary_flushes}")

    def reset(self) -> None:
        """Zero every counter, e.g. for reset_live_session(). Equivalent to
        a fresh SessionStats() -- there is no state here worth preserving
        across a reset."""
        self.__init__()


PREROLL_S = 1.0  # audio to prepend before the VAD's detected speech onset


class AudioHistory:
    """Rolling buffer of recent audio so finals can include pre-onset context."""

    def __init__(self, sample_rate: int, keep_s: float = 30.0):
        self.sr = sample_rate
        self.keep = int(keep_s * sample_rate)
        self.buf = np.zeros(0, dtype=np.float32)
        self.offset = 0  # absolute sample index of buf[0]
        self.last_seg_end = 0  # don't let preroll bleed into the previous utterance

    def push(self, chunk: np.ndarray):
        self.buf = np.concatenate([self.buf, chunk])
        if len(self.buf) > self.keep:
            drop = len(self.buf) - self.keep
            self.buf = self.buf[drop:]
            self.offset += drop

    def with_preroll(self, seg_start: int, seg_samples: np.ndarray) -> np.ndarray:
        want = max(seg_start - int(PREROLL_S * self.sr), self.last_seg_end, self.offset)
        pre = self.buf[want - self.offset:seg_start - self.offset]
        self.last_seg_end = seg_start + len(seg_samples)
        if len(pre) == 0:
            return seg_samples
        return np.concatenate([pre, seg_samples])

    def clear(self) -> None:
        """Drop all buffered audio, e.g. for reset_live_session(). The VAD
        driving run_stream() keeps counting samples from stream start
        regardless of this reset (rebuilding it is a separate, unrelated
        concern -- see LiveVad), so the first with_preroll() call after a
        clear() naturally finds nothing to prepend (its `want` bound falls
        outside the now-empty buf) rather than raising: numpy clamps an
        out-of-range slice to empty instead of erroring."""
        self.buf = np.zeros(0, dtype=np.float32)
        self.offset = 0
        self.last_seg_end = 0


def drain_segments(vad, sample_rate: int, asr: RoutedASR, stats: SessionStats,
                   printer: PartialPrinter, history: AudioHistory | None = None,
                   known_lang: str | None = None, refiner: "Refiner | None" = None,
                   translator_worker: "TranslationWorker | None" = None,
                   speaker_labeler=None) -> int:
    drained = 0
    while not vad.empty():
        segment = vad.front
        seg_end_time = time.perf_counter()  # segment-end reference point for latency
        samples = np.asarray(segment.samples, dtype=np.float32)
        seg_start, seg_end = segment.start, segment.start + len(samples)
        if history is not None:
            samples = history.with_preroll(seg_start, samples)
        vad.pop()
        drained += 1

        seg_s = len(samples) / sample_rate
        raw_speech_s = (seg_end - seg_start) / sample_rate  # without preroll
        # the early LID belongs to the utterance in progress; only the first
        # drained segment can safely claim it
        result = asr.transcribe(samples, sample_rate,
                                known_lang=known_lang if drained == 1 else None,
                                speech_s=raw_speech_s)
        latency_ms = (time.perf_counter() - seg_end_time) * 1000

        if not result["text"].strip():
            continue  # non-speech (jingle/SFX): no line, no speaker, no span

        # canonical_speaker ("S{n}", never "?") is what flows into the
        # refiner's spans -- majority voting and any other grouping logic
        # must only ever see the assignment SpeakerLabeler actually made.
        # display_speaker is the same label with issue #11's provisional
        # "?" suffix applied (docs/DIARIZATION_PLAN.md section 10.8, option
        # B) and is used ONLY for what gets printed/published right here --
        # assignment is untouched.
        canonical_speaker = ""
        display_speaker = ""
        if speaker_labeler is not None:
            canonical_speaker = speaker_labeler.label(samples, sample_rate, source="fast")
            display_speaker = speaker_labeler.display_label(canonical_speaker)
        speaker_tag = f"{display_speaker}|" if display_speaker else ""

        stats.segments += 1
        stats.latencies_ms.append(latency_ms)
        printer.clear()
        # GitHub issue #29: `switched` is true only when this final's
        # language differs from the PREVIOUS final's -- tracked on `stats`
        # (not asr.last_lang, which transcribe() has already advanced to
        # this same segment's language by the time we get here) so it
        # survives across drain_segments() calls for the life of the
        # session, and resets cleanly via SessionStats.reset().
        switched = (stats.last_final_lang is not None
                   and stats.last_final_lang != result["lang"])
        stats.last_final_lang = result["lang"]
        printer.hub.final(result["text"], result["lang"], display_speaker,
                          latency_ms, result.get("tier", ""),
                          audio_s=seg_s, lid_ms=result.get("lid_ms"),
                          decode_ms=result.get("decode_ms"), switched=switched)
        probe_part = f", probe={result['probe_ms']:.0f}ms" if result.get("probe_ms") else ""
        print(f"[{speaker_tag}{result['lang']}/{result.get('tier', '?')}] {result['text']}  "
              f"(seg={seg_s:.1f}s, lid={result['lid_ms']:.0f}ms{probe_part}, "
              f"decode={result['decode_ms']:.0f}ms, latency={latency_ms:.0f}ms)", flush=True)
        if translator_worker is not None and result["lang"] == "ja" and result["text"].strip():
            translator_worker.submit(result["text"])
        if refiner is not None:
            refiner.add_span(seg_start, seg_end, result["lang"], result["text"],
                             canonical_speaker)
    return drained


import re as _re


def digits_consistent(src: str, out: str) -> bool:
    """Every digit run in the source must survive into the translation.

    Guards against the MT models' number errors (500万円 -> "5万英镑"): a
    wrong number in a subtitle is worse than no translation. Kanji numerals
    carry no ASCII digits, so those lines pass through unguarded.
    """
    src_runs = _re.findall(r"\d+", src)
    if not src_runs:
        return True
    out_runs = set(_re.findall(r"\d+", out))
    return all(run in out_runs for run in src_runs)


def safe_translate(translator, text: str) -> str:
    """Translate one line; fall back to the source when numbers got mangled."""
    out = translator.translate(text)
    if out != text and not digits_consistent(text, out):
        return text
    return out


def translate_by_sentence(translator, text: str) -> str:
    """The MT models are trained on single sentences; feed one at a time."""
    sentences = [s for s in _re.split(r"(?<=[。！？!?])\s*", text) if s.strip()]
    out = []
    for s in sentences:
        en = safe_translate(translator, s)
        if en != s:
            out.append(en)
    return " ".join(out)


def _build_translator(lang: str):
    """One target language code -> a constructed translator instance.

    en uses the dedicated FuguMT module; any other target is accepted if
    M2M-100's vocabulary has a token for it (translate_m2m.
    is_supported_target()) -- ValueError otherwise. Only a subset of M2M
    targets have measured translation quality (translate_m2m.
    VALIDATED_TARGETS); TranslatorM2M itself prints a note to stderr for
    an unvalidated one but still constructs it.
    """
    if lang == "en":
        from translate_ja_en import TranslatorJaEn

        return TranslatorJaEn()
    from translate_m2m import TranslatorM2M, is_supported_target

    if not is_supported_target(lang):
        raise ValueError(f"unsupported translation target: {lang}")
    return TranslatorM2M(lang)


class TranslatorPool:
    """Shared, mutable set of ja->target translators (GitHub issue #29).

    Both the live TranslationWorker and the Refiner's second pass read
    from the SAME pool instance, so adding or removing a translation
    target at runtime (POST /config's "translate" key) applies to both
    passes at once instead of needing two separate APIs kept in sync.
    Translators are built lazily, one per add_target() call, each
    reporting a `model_load` event (start/done, timed) on `hub` if one was
    given -- the same event shape asr_engine.RoutedASR emits for its own
    recognizer builds, using "translator:<lang>" as the model name.

    `targets` is insertion order (oldest first), which is also the order
    TranslationWorker/Refiner print/publish each line's translations in.
    """

    def __init__(self, hub: "EventHub | None" = None):
        self._translators: dict[str, object] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self.hub = hub

    @property
    def targets(self) -> list[str]:
        # GitHub issue #29 review round 1: snapshotted under the lock, not
        # a bare `list(self._order)` -- add_target()/remove_target() both
        # mutate _order (append/remove) under this same lock, and reading
        # it while a mutation is in flight is a real hazard (CPython list
        # iteration/copy over a list another thread is resizing), not just
        # a theoretical one.
        with self._lock:
            return list(self._order)

    def __len__(self) -> int:
        return len(self._translators)

    def get(self, lang: str):
        return self._translators.get(lang)

    def items(self):
        """[(lang, translator), ...] in target-insertion order -- a
        drop-in for `dict.items()` at Refiner's existing iteration sites.

        Snapshotted under the lock (GitHub issue #29 review round 1): the
        live TranslationWorker iterates this on its own thread while
        add_target()/remove_target() can run concurrently from an HTTP
        handler thread (POST /config's "translate" key). Without the lock,
        a remove_target() landing between this reading `self._order` and
        looking up each `lang` in `self._translators` could KeyError --
        the worker thread has no try/except around its translation loop,
        so that would silently kill it and translations would stop for
        the rest of the session. Once snapshotted, the returned (lang,
        translator) pairs hold real translator object references, so a
        target removed a moment AFTER this call returns is harmless: the
        caller just finishes using the object it already has (at most one
        extra translation after the removal was requested), never a
        re-lookup by name that could miss.
        """
        with self._lock:
            return [(lang, self._translators[lang]) for lang in self._order]

    def add_target(self, lang: str) -> None:
        """Build and register a translator for `lang`. A no-op if `lang`
        is already a target (idempotent, matching set-like add semantics).
        Raises ValueError if `lang` isn't a supported translation target
        (see _build_translator) -- callers driving this from an external
        request (POST /config) should turn that into a 400; the CLI's own
        --translate startup path (build_translators() below) degrades it
        to a stderr note instead."""
        with self._lock:
            if lang in self._translators:
                return
            t0 = time.perf_counter()
            if self.hub is not None:
                self.hub.publish({"type": "model_load", "model": f"translator:{lang}",
                                  "phase": "start", "ms": None})
            translator = _build_translator(lang)  # ValueError propagates on an unsupported code
            ms = (time.perf_counter() - t0) * 1000
            if self.hub is not None:
                self.hub.publish({"type": "model_load", "model": f"translator:{lang}",
                                  "phase": "done", "ms": ms})
            self._translators[lang] = translator
            self._order.append(lang)

    def remove_target(self, lang: str) -> None:
        """Drop `lang` from the pool. A no-op if it wasn't a target."""
        with self._lock:
            if self._translators.pop(lang, None) is not None:
                self._order.remove(lang)


def build_translators(langs: str, pool: "TranslatorPool") -> None:
    """Populate `pool` from a "--translate en,zh,ko,es,..." style string.

    Unlike TranslatorPool.add_target() itself, an unsupported target here
    degrades to a stderr note and is skipped rather than raising -- this
    is the CLI startup path, which has always continued past a bad
    --translate code instead of aborting the whole session over it.
    """
    for lang in [x.strip() for x in langs.split(",") if x.strip()]:
        try:
            pool.add_target(lang)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)


class TranslationWorker:
    """Async ja->target translation of finalized lines (console display).

    Reads `pool.targets` fresh on every submitted line, so a target added
    or removed at runtime (TranslatorPool.add_target()/remove_target(),
    e.g. via POST /config) takes effect on the very next line with no
    restart needed. Always constructed by main() (even with no --translate
    targets at startup) precisely so a runtime-added target has a live
    worker already pulling ja finals to translate -- see main()'s wiring.
    """

    def __init__(self, pool: TranslatorPool, hub: "EventHub"):
        self._pool = pool
        self._hub = hub
        self._q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text: str):
        self._q.put(text)

    def _run(self):
        while True:
            text = self._q.get()
            for lang, tr in self._pool.items():
                out = safe_translate(tr, text)
                if out != text:  # fallback returns the source: nothing worth showing
                    print(f"[→{lang}] {out}", flush=True)
                    self._hub.publish({"type": "translation", "lang": lang, "text": out})


from asr_engine import (  # shared with the engine's live correction / refine dual-LID confirm
    REFINE_MIN_REGROUP_S, SV_LANGS, resolve_refine_lang, script_corrected_lang,
    sv_lid_tag,
)  # ModelUnavailable is imported at module top, alongside RoutedASR

GROUP_GAP_S = 2.0   # this much true silence closes an utterance group
GROUP_MAX_S = 25.0  # refine early rather than outgrow the audio history


class Refiner:
    """Second pass: re-decode a whole utterance group once the speaker pauses.

    Fast finals stay untouched; the refined text (measured ~23% relative CER
    better on real broadcast ja) goes to the console, the SSE stream, and the
    transcript file when one is requested.
    """

    def __init__(self, asr: RoutedASR, history: AudioHistory, sample_rate: int,
                 printer: PartialPrinter, transcript_path: str | None = None,
                 translators: "TranslatorPool | None" = None, stats: "SessionStats | None" = None,
                 speaker_labeler=None, diarizer=None, min_remap_update_s: float = 0.0,
                 joint_remap: bool = False, exclude_provisional_remap: bool = False,
                 global_recluster: bool = False,
                 global_recluster_threshold: float = 0.65,
                 num_clusters_hint: str = "off",
                 num_clusters_hint_min_s: float = 0.0):
        self.asr = asr
        self.history = history
        self.sr = sample_rate
        self.printer = printer
        # ja->target, synchronous per refine. Same TranslatorPool instance
        # main() hands the live TranslationWorker, so a runtime add/remove
        # target (POST /config) applies to both passes -- see that class's
        # docstring. Falls back to an empty pool (never translates
        # anything) rather than None so every call site below can use it
        # unconditionally (`self.translators.items()`/truthiness) without
        # an `is not None` guard.
        self.translators = translators if translators is not None else TranslatorPool()
        self.stats = stats
        # docs/DIARIZATION_PLAN.md iterations 3-4: when --speakers is on,
        # re-diarize each group's audio (group-local speaker turns) and
        # remap those local clusters onto speaker_labeler's session-global
        # S{n} centroids, so a refine group with a mid-group speaker change
        # prints one [refine/S{n}] line per turn instead of one majority-
        # vote line for the whole group. speaker_labeler is the SAME
        # instance the fast path uses (realtime_transcribe.py's --speakers
        # wiring), so global labels stay consistent between the two passes
        # and the fast path's centroids double as the diarizer's anchor set.
        self.speaker_labeler = speaker_labeler
        self.diarizer = diarizer
        # Round 4 (docs/DIARIZATION_PLAN.md section 14) T2 experiment: see
        # eval_diar.generate_diarize_hypothesis()'s min_remap_update_s
        # docstring. 0.0 (default) is a no-op.
        self.min_remap_update_s = min_remap_update_s
        # Round 5 (docs/DIARIZATION_PLAN.md section 15) T1 experiment: see
        # speaker_id.SpeakerLabeler.match_embeddings_joint()'s docstring.
        # False (default) is a no-op -- every local cluster still remaps
        # independently via match_embedding(), same as before.
        self.joint_remap = joint_remap
        # Round 5 (docs/DIARIZATION_PLAN.md section 15) T3 experiment: see
        # speaker_id.SpeakerLabeler.match_embedding()'s exclude_provisional
        # docstring. False (default) is a no-op.
        self.exclude_provisional_remap = exclude_provisional_remap
        # Round 7 (docs/DIARIZATION_PLAN.md section 17): see
        # eval_diar.generate_diarize_hypothesis()'s global_recluster
        # docstring for the full design. False (default) is a no-op: no
        # extra bookkeeping happens in _emit_turns() and live output is
        # therefore byte-identical to every earlier round's behavior.
        # global_recluster_entries accumulates one row per DISTINCT
        # (refine-group index, local diarization cluster id) that got a
        # real embedding, across the WHOLE SESSION -- unlike eval_diar.py
        # (one process per meeting), a live session can run indefinitely,
        # so this list grows for as long as the process runs; it is only
        # ever read (never rewritten mid-session) by run_global_recluster()
        # at shutdown. Deliberately NOT wired into any live/incremental
        # output path -- see run_global_recluster()'s docstring for why
        # "revise already-printed lines" is explicitly out of scope this
        # round.
        self.global_recluster = global_recluster
        self.global_recluster_threshold = global_recluster_threshold
        # Round 9 (docs/DIARIZATION_PLAN.md section 19) Experiment A: see
        # eval_diar.generate_diarize_hypothesis()'s num_clusters_hint
        # docstring -- same "off"/"confirmed"/"confirmed-capped" modes,
        # same min-speech-duration gate. "off" (default) is a no-op.
        self.num_clusters_hint = num_clusters_hint
        self.num_clusters_hint_min_s = num_clusters_hint_min_s
        self._recluster_entries: list[dict] = []
        self._recluster_group_idx = 0
        self.spans: list[tuple[int, int, str, str, str]] = []
        self._transcript = open(transcript_path, "a", encoding="utf-8") if transcript_path else None
        # A single FIFO worker (not "spawn a thread per refine") is what makes
        # output order safe: maybe_refine() is only ever called from the
        # single-threaded ingestion path (run_stream / add_span's language-
        # boundary flush), so the order tasks are *queued* in is always the
        # chronological order of the audio groups. Threading.Thread-per-call
        # loses that guarantee -- start() order is not lock-acquisition
        # order, so two nearly-simultaneous refines (a language-boundary
        # flush racing the next group's silence-triggered flush) could grab
        # the old _worker_lock in either order and print [refine/...] lines
        # out of chronological sequence. A single consumer thread draining a
        # Queue processes strictly in enqueue order, so this can't happen.
        self._task_queue: "queue.Queue" = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        while True:
            task = self._task_queue.get()
            try:
                task()
            except Exception:
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
            finally:
                self._task_queue.task_done()

    def reset(self) -> None:
        """Drop pending spans/group state and clear the rolling audio
        history, e.g. for a session reset (GitHub issue #29).

        Waits for the worker queue to drain FIRST so this never races an
        in-flight refine task still reading self.spans/self.history --
        this makes reset() safe to call on its own, but it means anything
        still pending here is simply DISCARDED, not decoded. Call this
        only after any refine output worth keeping has already been
        flushed (e.g. via maybe_refine(force=True) + a queue join, as
        module-level reset_live_session() does) if losing an in-progress
        group's text would be a problem.
        """
        self._task_queue.join()
        self.spans = []
        self.history.clear()
        self._recluster_entries = []
        self._recluster_group_idx = 0
        if self._transcript is not None:
            # visible marker in the transcript file, which stays open --
            # readers scrolling through it need some indication that
            # everything after this line belongs to a new, unrelated
            # session (new speakers, reset language state), not a
            # continuation of what came before.
            self._transcript.write("\n----- session reset -----\n\n")
            self._transcript.flush()

    def add_span(self, seg_start: int, seg_end: int, lang: str, text: str, speaker: str):
        """Append one finalized segment to the pending refine group.

        Splits the group first if this segment's (script-corrected)
        language differs from the group's current language. Without this,
        a group kept accumulating across a language change until the next
        silence gap or GROUP_MAX_S -- the "mixed" guard in maybe_refine
        already protected the DECODE from being corrupted (it skips
        re-decoding and falls back to the joined fast-path text), but the
        group still printed as a single refine line under one language tag,
        visually swallowing an en segment sandwiched between two ja ones
        into a "[refine/ja] ..." line. Refine groups now never cross a
        language boundary.
        """
        corrected = script_corrected_lang(lang, text)
        if self.spans:
            group_lang = script_corrected_lang(self.spans[-1][2], self.spans[-1][3])
            if corrected != group_lang:
                # flush the previous group off the hot path; it belongs to
                # a different language and must not accumulate this span
                if self.stats is not None:
                    self.stats.refine_lang_boundary_flushes += 1
                self.maybe_refine(seg_start, force=True, force_sync=False)
        self.spans.append((seg_start, seg_end, lang, text, speaker))

    MIN_TURN_S = 0.3  # shorter diarization turns aren't worth a separate ASR call

    def _emit_turns(self, buf: np.ndarray, refine_lang: str, fast_joined: str,
                    majority_speaker: str = "", num_clusters_hint_value: int | None = None
                    ) -> bool:
        """Multi-speaker refine path (docs/DIARIZATION_PLAN.md iterations 3-4).

        Re-diarizes this group's audio and, if it finds a genuine speaker
        change, decodes and prints one [refine/S{n}] line per turn instead
        of the single majority-vote line maybe_refine()'s caller falls back
        to. Each local diarization cluster is remapped onto the SAME global
        centroids self.speaker_labeler (the fast path) maintains, so S{n}
        stays consistent between the fast and refine passes.

        majority_speaker is maybe_refine()'s own majority-vote label over
        the group's fast-path per-segment labels -- the same fallback the
        caller would print on a decline. Only used as the read-only-remap
        fallback for a too-short local cluster when self.min_remap_update_s
        gates it (see that attribute's comment); ignored otherwise.

        num_clusters_hint_value (Round 9, docs/DIARIZATION_PLAN.md section
        19): maybe_refine() precomputes this per self.num_clusters_hint/
        self.num_clusters_hint_min_s (see their docstrings) using this
        group's fast-path span count/duration and self.speaker_labeler's
        confirmed-speaker count as of just before this call, then passes
        the result straight to self.diarizer.process(num_clusters=). None
        (self.num_clusters_hint == "off", or the group didn't qualify) is
        a no-op -- FastClustering picks the count itself, unchanged from
        every earlier round.

        Returns True if it printed turn-level output (caller should stop
        and skip the single-line fallback), False if it declined: no
        diarizer/labeler configured, diarization found only one speaker
        (the majority-vote line is just as good and cheaper), or the
        turn-decoded text came back suspiciously short (same "never lose
        content vs. the fast finals" guard maybe_refine() applies to the
        single-line path).
        """
        if self.diarizer is None or self.speaker_labeler is None:
            return False
        try:
            raw = self.diarizer.process(buf, self.sr, num_clusters=num_clusters_hint_value)
        except Exception as exc:
            message = f"diarization failed, falling back to single-speaker line: {exc}"
            print(f"[refine] {message}", file=sys.stderr)
            self.printer.hub.publish({"type": "warning", "code": "diarization_failed",
                                      "message": message})
            raw = []

        # Round 8 (docs/DIARIZATION_PLAN.md section 18) T1: this helper
        # records a group-level pool entry (local_id=-1 sentinel) for
        # whichever `turns` view was live at the moment this group declines,
        # when global_recluster is on -- mirrors eval_diar.
        # generate_diarize_hypothesis()'s pool-completeness fix (see
        # global_recluster.py's module docstring for why a group that never
        # contributes to the pool broke Round 7). Only feeds
        # run_global_recluster()'s end-of-session diagnostic mapping (see
        # that method's docstring for why live output is never revised), so
        # it changes no printed/published/written output either way. Kept
        # as a closure (not hoisted out) so the two decline points below can
        # each pass their own `turns` snapshot without duplicating the
        # bookkeeping.
        def record_pool_entry(turns_snapshot):
            if not self.global_recluster:
                return
            from global_recluster import pool_audio_for_group

            pool_audio, pool_dur_s = pool_audio_for_group(buf, self.sr, turns_snapshot)
            if len(pool_audio) > 0:
                self._recluster_entries.append({
                    "group_idx": self._recluster_group_idx, "local_id": -1,
                    "embedding": self.speaker_labeler.embed(pool_audio, self.sr),
                    "duration_s": pool_dur_s,
                })
                self._recluster_group_idx += 1

        if len({local for local, _, _ in raw}) < 2:
            # single speaker (or the diarizer declined): same fallback the
            # caller (maybe_refine) takes -- one majority-vote line. No
            # `turns` view exists yet at this decline point (unchanged from
            # pre-Round-8 control flow -- the duration filter below never
            # runs), so the pool entry falls back to the whole group buffer,
            # same as eval_diar.py's true-decline case.
            record_pool_entry([])
            return False

        turns = []  # (local_id, start_sample, end_sample) within buf
        for local_id, start_s, end_s in raw:
            start = max(0, int(round(start_s * self.sr)))
            end = min(len(buf), int(round(end_s * self.sr)))
            if end - start >= int(self.MIN_TURN_S * self.sr):
                turns.append((local_id, start, end))
        if len(turns) < 2:
            # duration filtering left fewer than 2 usable turns -- same
            # decline condition as before Round 8 (a plain turn count, not
            # a distinct-id count: two turns from the SAME local_id still
            # counts as "enough" here, unchanged). Whatever turns did
            # survive filtering are still real speech, so the pool entry
            # uses them instead of falling all the way back to the buffer.
            record_pool_entry(turns)
            return False

        # one representative embedding per local cluster, matched onto the
        # global centroid set -- this is the iteration-4 remap step.
        local_ids = sorted({t[0] for t in turns})
        cluster_embs = {}
        cluster_durs = {}
        for local_id in local_ids:
            cluster_audio = np.concatenate(
                [buf[start:end] for lid, start, end in turns if lid == local_id]
            )
            cluster_embs[local_id] = self.speaker_labeler.embed(cluster_audio, self.sr)
            cluster_durs[local_id] = sum(
                (end - start) / self.sr for lid, start, end in turns if lid == local_id)
            if self.global_recluster:
                self._recluster_entries.append({
                    "group_idx": self._recluster_group_idx, "local_id": local_id,
                    "embedding": cluster_embs[local_id], "duration_s": cluster_durs[local_id],
                })
        if self.global_recluster:
            self._recluster_group_idx += 1

        global_label = {}
        # min_remap_update_s (Round 4 T2) still gates short clusters to a
        # read-only probe, independent of joint_remap below -- see
        # min_remap_update_s's docstring. Only the remaining ("long
        # enough") clusters are eligible for the Round 5 T1 joint
        # assignment; a short cluster is never a joint-assignment
        # candidate (it can't fold into or claim a centroid either way).
        short_ids = [lid for lid in local_ids
                     if self.min_remap_update_s > 0 and cluster_durs[lid] < self.min_remap_update_s]
        for local_id in short_ids:
            probe = self.speaker_labeler.match_embedding(
                cluster_embs[local_id], update=False, threshold=self.speaker_labeler.remap_threshold,
                source="remap", exclude_provisional=self.exclude_provisional_remap)
            global_label[local_id] = probe if probe else majority_speaker

        joint_ids = [lid for lid in local_ids if lid not in short_ids]
        if joint_ids:
            if self.joint_remap:
                # Round 5 (docs/DIARIZATION_PLAN.md section 15) T1: solve
                # this group's remaining clusters' assignment jointly so
                # two distinct local clusters can't collapse onto the same
                # global speaker. match_embeddings_joint() itself falls
                # back to the old independent match_embedding() path for a
                # single-cluster input, so behavior is unchanged there.
                labels = self.speaker_labeler.match_embeddings_joint(
                    [cluster_embs[lid] for lid in joint_ids], update=True,
                    threshold=self.speaker_labeler.remap_threshold, source="remap")
                for local_id, label in zip(joint_ids, labels):
                    global_label[local_id] = label
            else:
                for local_id in joint_ids:
                    global_label[local_id] = self.speaker_labeler.match_embedding(
                        cluster_embs[local_id], update=True, threshold=self.speaker_labeler.remap_threshold,
                        source="remap", exclude_provisional=self.exclude_provisional_remap)

        # iteration 6 (docs/DIARIZATION_PLAN.md section 9): this refine
        # group's remap is the natural "clean copy" boundary -- give
        # maybe_merge_centroids() a chance to fold any two global speakers
        # that have drifted together before the next group opens more.
        # No-op unless --speaker-merge was passed.
        self.speaker_labeler.maybe_merge_centroids()

        outputs = []  # (global_label, turn_text, audio_s), in chronological turn order
        for local_id, start, end in turns:
            turn_text = self.asr.transcribe(buf[start:end], self.sr, known_lang=refine_lang,
                                            live=False)["text"]
            if turn_text.strip():
                outputs.append((global_label[local_id], turn_text, (end - start) / self.sr))

        total_text = " ".join(t for _, t, _ in outputs)
        if len(total_text.strip()) < 0.7 * len(fast_joined):
            return False

        for label, turn_text, turn_audio_s in outputs:
            # label is the canonical assignment from match_embedding() above
            # (unchanged); disp is only for what actually gets printed here
            # -- issue #11 / section 10.8 option B.
            disp = self.speaker_labeler.display_label(label) if label else label
            tag = f"{disp}|{refine_lang}" if disp else refine_lang
            print(f"[refine/{tag}] {turn_text}", flush=True)
            self.printer.hub.publish({"type": "refine", "text": turn_text,
                                      "lang": refine_lang, "speaker": disp,
                                      "audio_s": turn_audio_s})
            outs = []
            if self.translators and refine_lang == "ja":
                for tlang, tr in self.translators.items():
                    out = translate_by_sentence(tr, turn_text)
                    if out and out != turn_text:
                        print(f"[refine→{tlang}] {out}", flush=True)
                        outs.append((tlang, out))
            if self._transcript is not None:
                prefix = f"{disp}: " if disp else ""
                self._transcript.write(prefix + turn_text + "\n")
                for tlang, out in outs:
                    self._transcript.write(f"  →{tlang} {out}\n")
                self._transcript.flush()
        return True

    def run_global_recluster(self) -> dict:
        """Session-end (post-hoc) global re-cluster, Round 7 (docs/
        DIARIZATION_PLAN.md section 17). No-op unless self.global_recluster.

        Re-clusters every (refine-group, local-cluster) embedding
        accumulated by _emit_turns() across the whole session with
        global_recluster.two_stage_cluster() -- the same two-stage design
        Round 6 (eval_diar_overlap.py) proved out and eval_diar.py's
        --global-recluster reuses for the eval path. Returns a small stats
        dict (entry/cluster counts, wall time) and PRINTS a one-line
        summary of the resulting mapping, purely as a diagnostic -- this is
        explicitly NOT wired into live output, the SSE stream, or the
        transcript file.

        Why not: lines under the incremental S{n} labels have already been
        printed/published/written to the transcript by the time this runs
        (it's called at shutdown, after every group has already gone
        through _emit_turns() once). Retroactively revising them would need
        a "relabel" event/UX for every downstream consumer (console,
        subtitle overlay SSE, transcript file) to redraw already-shown
        lines under a new label -- an out-of-scope production/UX design
        decision this round's task explicitly defers (see the T1 docstring
        in eval_diar.generate_diarize_hypothesis()). Call this only after
        every already-queued refine task has finished (self._task_queue.
        join()), or the entry list may still be growing.
        """
        stats = {"time_s": 0.0, "n_entries": len(self._recluster_entries), "n_clusters": 0,
                 "mapping": {}}
        if not self.global_recluster or not self._recluster_entries:
            return stats
        from global_recluster import two_stage_cluster

        t0 = time.time()
        entries = self._recluster_entries
        embeddings = np.stack([e["embedding"] for e in entries])
        group_ids = [e["group_idx"] for e in entries]
        durations = [e["duration_s"] for e in entries]
        labels = two_stage_cluster(embeddings, group_ids, durations,
                                   threshold=self.global_recluster_threshold,
                                   reliable_s=1.5)
        stats["time_s"] = time.time() - t0
        stats["n_clusters"] = int(labels.max()) + 1 if len(labels) else 0
        stats["mapping"] = {
            f"group{e['group_idx']}/local{e['local_id']}": f"S{int(c) + 1}"
            for e, c in zip(entries, labels)
        }
        print(f"=== global re-cluster: {stats['n_entries']} local-cluster embeddings -> "
              f"{stats['n_clusters']} session-global speakers in {stats['time_s']:.2f}s "
              f"(diagnostic only, not applied to already-printed output) ===")
        # GitHub issue #29: published only when the diagnostic actually ran
        # (not on the early no-op return above), matching this docstring's
        # own "diagnostic only" framing -- an embedder that cares about
        # this gets the exact same mapping the console line summarizes.
        self.printer.hub.publish({"type": "recluster", **stats})
        return stats

    def maybe_refine(self, now_sample: int, force: bool = False, force_sync: bool | None = None):
        if not self.spans:
            return
        first_start = self.spans[0][0]
        last_end = self.spans[-1][1]
        due = (force
               or now_sample - last_end >= int(GROUP_GAP_S * self.sr)
               or last_end - first_start >= int(GROUP_MAX_S * self.sr))
        if not due:
            return
        # force=True normally means "run synchronously" (the shutdown/flush
        # path wants the transcript finished before the process exits);
        # force_sync=False overrides that for a forced-but-not-urgent flush
        # (a language-boundary split mid-stream) so it still goes through
        # the background thread and doesn't stall the hot path.
        run_sync = force if force_sync is None else force_sync
        lo = max(first_start - int(PREROLL_S * self.sr), self.history.offset)
        buf = self.history.buf[lo - self.history.offset:last_end - self.history.offset].copy()
        # LID tags lie under BGM; trust the script of the decoded text over
        # the tag (an "en" span full of kanji was misdetected Japanese, an
        # ALL-CAPS "ja" span was misdetected English).
        langs = [script_corrected_lang(lang, text)
                 for _, _, lang, text, _ in self.spans]
        lang = max(set(langs), key=langs.count)
        # a genuinely mixed-language group must not be re-decoded in one
        # language: the per-segment finals already used the right model per
        # language, and a majority-language re-decode mangles the minority
        # (docs/BENCHMARKS.md iteration 25). Keep the merge, skip the decode.
        mixed = len(set(langs)) > 1 and min(langs.count(l) for l in set(langs)) / len(langs) >= 0.25
        speakers = [sp for _, _, _, _, sp in self.spans if sp]
        speaker = max(set(speakers), key=speakers.count) if speakers else ""
        fast_joined = " ".join(t for _, _, _, t, _ in self.spans if t.strip())
        # Round 9 (docs/DIARIZATION_PLAN.md section 19) Experiment A: same
        # hint computation as eval_diar.generate_diarize_hypothesis(), done
        # here (before self.spans is cleared below) because this is the
        # only place both the group's own fast-path span list and
        # self.speaker_labeler's live confirmed-count are available
        # together. n_local_segments mirrors eval_diar.py's len(group)
        # cap for "confirmed-capped".
        num_clusters_hint_value = None
        if self.num_clusters_hint != "off" and self.speaker_labeler is not None:
            group_speech_s = sum(e - s for s, e, _, _, _ in self.spans) / self.sr
            if group_speech_s >= self.num_clusters_hint_min_s:
                confirmed = self.speaker_labeler.num_confirmed_speakers()
                if confirmed >= 1:
                    num_clusters_hint_value = confirmed
                    if self.num_clusters_hint == "confirmed-capped":
                        num_clusters_hint_value = min(confirmed, len(self.spans))
        self.spans = []
        if len(buf) < self.sr // 2:
            return
        if self.stats is not None:
            self.stats.refine_groups_closed += 1

        def work():
            # off the hot path: a refine of a 25s group takes ~0.5-1s and must
            # not delay the next utterance's instant final (soak test showed
            # 2.6s latency spikes when run inline). Runs on the single
            # _worker_thread, which serializes it against every other queued
            # refine in enqueue (== chronological) order -- see the comment
            # on _task_queue in __init__.
            refine_lang = lang
            if mixed:
                text = fast_joined
            else:
                # re-run LID on the merged (longer, higher-confidence)
                # audio: the fast path's per-segment majority vote used
                # only 2-4s clips, which docs/LID.md measured well below
                # whisper-tiny's high-confidence length for several
                # languages. But "longer" only holds when the group
                # really is a multi-segment utterance -- a group can be
                # a single short segment sitting alone between silence
                # gaps, and a lone whisper-tiny re-judgment on that is
                # no more reliable than the live path's single LID call
                # (real-mic incident: this flipped a correctly
                # dual-confirmed live "ko" back to a collapsed "ru").
                # So the re-judgment goes through the SAME dual-LID
                # confirmation as the live path: SenseVoice must agree,
                # and resolve_refine_lang additionally gates on the
                # group's total duration (see REFINE_MIN_REGROUP_S).
                group_duration_s = len(buf) / self.sr
                detected = self.asr._identify_lang(buf, self.sr)
                sv_lang = ""
                probe_text = None
                if detected != lang and group_duration_s >= REFINE_MIN_REGROUP_S:
                    try:
                        sv_rec = self.asr._get("sv")
                        probe_text, sv_tag = self.asr._decode_full(sv_rec, buf, self.sr)
                        sv_lang = sv_lid_tag(sv_tag)
                    except ModelUnavailable:
                        sv_lang = ""  # minimal install: no probe possible, no override
                resolved, changed = resolve_refine_lang(lang, detected, sv_lang, group_duration_s)
                if changed:
                    if resolved in SV_LANGS and probe_text is not None:
                        # resolve_refine_lang only sets changed=True when
                        # sv_lang == whisper_lang == resolved, so the SenseVoice
                        # probe above already decoded this exact buffer through
                        # the same route transcribe() would take for `resolved`
                        # (ko/yue both route to SenseVoice) -- reuse its text
                        # instead of a second SV pass, same reuse pattern as
                        # the live path's dual-LID confirmation probe. Apply
                        # the same post-processing transcribe() would (text
                        # replacements, and Kiwi ko word-spacing) so the
                        # result matches what a fresh transcribe() call would
                        # have produced.
                        text = self.asr._replace(probe_text)
                        if resolved == "ko" and text.strip() and self.asr.ko_spacer is not None:
                            try:
                                text = self.asr.ko_spacer.space(text, reset_whitespace=True)
                            except Exception:
                                pass
                    else:
                        text = self.asr.transcribe(buf, self.sr, known_lang=resolved,
                                                    live=False)["text"]
                    if text.strip():
                        refine_lang = resolved
                    else:
                        text = self.asr.transcribe(buf, self.sr, known_lang=lang,
                                                    live=False)["text"]
                else:
                    text = self.asr.transcribe(buf, self.sr, known_lang=lang,
                                                live=False)["text"]
            # a merged re-decode must never LOSE content; if it comes back
            # much shorter than the fast finals combined, trust those
            if len(text.strip()) < 0.7 * len(fast_joined):
                text = fast_joined
                refine_lang = lang
            if not text.strip():
                return
            if refine_lang != lang and self.stats is not None:
                self.stats.refine_lang_corrections += 1
                print(f"[refine] language corrected {lang}->{refine_lang}", flush=True)

            # docs/DIARIZATION_PLAN.md iterations 3-4: if the group actually
            # contains a speaker change, prefer per-turn output over the
            # single majority-vote line below. mixed-language groups skip
            # this (their "text" is already the joined fast finals, not a
            # coherent re-decode a turn-level re-split would make sense
            # against) and so does the fallback branch below when the
            # diarizer isn't available or found only one speaker.
            if not mixed and self._emit_turns(buf, refine_lang, fast_joined, speaker,
                                              num_clusters_hint_value):
                return

            # speaker is the majority vote over the group's canonical
            # per-segment labels (unchanged); disp is only for display --
            # issue #11 / section 10.8 option B.
            disp = (self.speaker_labeler.display_label(speaker)
                    if speaker and self.speaker_labeler is not None else speaker)
            tag = f"{disp}|{refine_lang}" if disp else refine_lang
            print(f"[refine/{tag}] {text}", flush=True)
            self.printer.hub.publish({"type": "refine", "text": text, "lang": refine_lang,
                                      "speaker": disp, "audio_s": len(buf) / self.sr})
            outs = []
            if self.translators and refine_lang == "ja":
                # synchronous here (we're already off the hot path) so the
                # transcript keeps source and translations adjacent, in
                # order. The MT models degrade on multi-sentence input,
                # so translate sentence by sentence.
                for tlang, tr in self.translators.items():
                    out = translate_by_sentence(tr, text)
                    if out and out != text:
                        print(f"[refine→{tlang}] {out}", flush=True)
                        outs.append((tlang, out))
            if self._transcript is not None:
                prefix = f"{disp}: " if disp else ""
                self._transcript.write(prefix + text + "\n")
                for tlang, out in outs:
                    self._transcript.write(f"  →{tlang} {out}\n")
                self._transcript.flush()

        # Both branches enqueue onto the same FIFO worker so cross-call
        # ordering is preserved even when a sync flush (shutdown, or a
        # language-boundary split queued via maybe_refine(..., force_sync=False)
        # just above it) lands close to an async one. force_sync=True still
        # blocks the caller until this task has actually run (the shutdown
        # path needs the transcript finished before the process exits) --
        # it just does so by waiting on the task rather than by running the
        # work inline, so it can't jump the queue ahead of an
        # already-queued-but-not-yet-run older group.
        if run_sync:
            done = threading.Event()

            def sync_work(work=work, done=done):
                try:
                    work()
                finally:
                    done.set()

            self._task_queue.put(sync_work)
            done.wait()
        else:
            self._task_queue.put(work)


def run_stream(chunks, vad, sample_rate: int, asr: RoutedASR, stats: SessionStats,
               printer: PartialPrinter, refiner: "Refiner | None" = None,
               history: AudioHistory | None = None,
               translator_worker: "TranslationWorker | None" = None,
               speaker_labeler=None, stop_event: "threading.Event | None" = None,
               control: "ControlQueue | None" = None):
    """Drive the VAD -> ASR pipeline over `chunks` until it's exhausted or
    `stop_event` is set.

    `stop_event` is the cancellation hook for an app embedding this module:
    the CLI itself still relies on KeyboardInterrupt (unchanged), but a
    caller running this in a background thread has no SIGINT to send, so it
    needs a way to ask the loop to stop that doesn't depend on the chunk
    source unblocking on its own. Checked once per chunk here; mic_chunks()
    additionally polls it directly since a plain `q.get()` would otherwise
    block indefinitely with nothing to yield.

    `control`, if given, is polled once per chunk (ControlQueue.poll()) --
    GitHub issue #29 review round 1: this is how a session-mutating
    operation submitted from another thread (POST /reset, via
    RuntimeControls.reset_fn) actually runs ON this thread instead of
    racing it. See ControlQueue's docstring.
    """
    audio_pos = 0.0
    last_partial = 0.0
    early_lang = None  # LID result computed mid-utterance so finals skip it
    if history is None:
        history = AudioHistory(sample_rate)
    for chunk in chunks:
        if stop_event is not None and stop_event.is_set():
            break
        if control is not None:
            control.poll()
        if not isinstance(chunk, np.ndarray):
            # non-ndarray = a "flush now" signal (ws_ingest.FLUSH on client
            # disconnect): finalize whatever the VAD has in progress instead
            # of waiting for silence that may never arrive.
            vad.flush()
            if drain_segments(vad, sample_rate, asr, stats, printer, history, early_lang,
                              refiner=refiner,
                              translator_worker=translator_worker,
                              speaker_labeler=speaker_labeler):
                early_lang = None
            if refiner is not None:
                refiner.maybe_refine(int(audio_pos * sample_rate), force=True)
            continue

        vad.accept_waveform(chunk)
        history.push(chunk)
        audio_pos += len(chunk) / sample_rate
        stats.total_audio_s += len(chunk) / sample_rate

        if vad.is_speech_detected() and audio_pos - last_partial >= PARTIAL_EVERY_S:
            last_partial = audio_pos
            cur = np.asarray(vad.current_segment.samples, dtype=np.float32)
            if len(cur) > int(PARTIAL_WINDOW_S * sample_rate):
                cur = cur[-int(PARTIAL_WINDOW_S * sample_rate):]
            # --mode single (asr.forced_lang set): every segment is forced to
            # one language, so the early-LID probe (whisper-tiny plus a
            # background model prefetch) would just be wasted work -- asr.partial()
            # already routes straight to forced_lang without needing a hint.
            if asr.forced_lang is None and early_lang is None and len(cur) >= int(2.0 * sample_rate):
                early_lang = asr.identify(cur, sample_rate)
            if printer.enabled and len(cur) >= sample_rate // 2:
                printer.show(asr.partial(cur, sample_rate, lang_hint=early_lang))

        if drain_segments(vad, sample_rate, asr, stats, printer, history, early_lang,
                          refiner=refiner,
                          translator_worker=translator_worker,
                          speaker_labeler=speaker_labeler):
            early_lang = None
        if refiner is not None and not vad.is_speech_detected():
            refiner.maybe_refine(int(audio_pos * sample_rate))


def _session_summary_event(stats: SessionStats, speaker_labeler=None) -> dict:
    """Build the `session_summary` event (GitHub issue #29): the same
    numbers main()'s shutdown block already prints, plus the same speaker
    diagnostics (when --speakers is on), as one JSON-able dict. Shared
    between shutdown and reset_live_session() so both report identically.

    `stats.latencies_ms`'s raw per-segment list is deliberately NOT
    included -- it is unbounded (grows for the life of a long-running
    session) and the console summary itself only ever reports its mean and
    max, so those are what this reports too.
    """
    speakers = None
    if speaker_labeler is not None:
        speakers = {
            "confirmed": speaker_labeler.num_confirmed_speakers(),
            "provisional": speaker_labeler.provisional_label_count(),
            "merges": speaker_labeler.merge_history(),
            "centroids": speaker_labeler.centroid_summary(),
        }
    return {
        "type": "session_summary",
        "stats": {
            "total_audio_s": stats.total_audio_s,
            "segments": stats.segments,
            "mean_latency_ms": (sum(stats.latencies_ms) / len(stats.latencies_ms)
                                if stats.latencies_ms else None),
            "max_latency_ms": max(stats.latencies_ms) if stats.latencies_ms else None,
            "refine_lang_corrections": stats.refine_lang_corrections,
            "refine_groups_closed": stats.refine_groups_closed,
            "refine_lang_boundary_flushes": stats.refine_lang_boundary_flushes,
        },
        "speakers": speakers,
    }


def reset_live_session(asr: "RoutedASR | None", refiner: "Refiner | None",
                       labeler, stats: "SessionStats | None",
                       hub: "EventHub | None", live_vad: "LiveVad | None" = None) -> None:
    """Reset a live session's mutable state WITHOUT rebuilding any loaded
    model (GitHub issue #29's `POST /reset`) -- a new conversation with the
    same process, same resident models, same VAD sensitivity, just no
    memory of the speakers/language/statistics that came before.

    Every parameter may be None (e.g. --no-refine leaves refiner=None, no
    --speakers leaves labeler=None), so this works for whatever subset of
    pieces a given session actually wired up.

    Order matters:
      1. Flush any refine work still pending (refiner.maybe_refine(force=
         True) + a queue join) so the outgoing session's tail isn't
         silently thrown away.
      2. Publish `session_summary` on `hub` -- summarizing the session
         that's ENDING, using its state as of right after the flush above
         (not the zeroed-out state reset leaves behind).
      3. Reset each piece's own state (labeler, refiner, stats).
      4. asr.reset_session() -- clears the sticky/pending LID state last,
         so nothing above it still needed the old session's language.
      5. Publish `session_reset` on `hub`, announcing the reset is done.

    `live_vad` is accepted for API symmetry with the other live-session
    pieces (and for a future caller that wants it) but is NOT touched here
    -- VAD sensitivity is an orthogonal, cross-session setting (see
    LiveVad's docstring), not part of a conversation's identity/state, so
    a session reset has no reason to change it.

    THREADING (GitHub issue #29 review round 1): this function is NOT
    thread-safe against a concurrently running run_stream() -- it mutates
    the same SpeakerLabeler/Refiner/SessionStats/RoutedASR state
    run_stream()'s decode loop reads and writes with no locking of its
    own (SpeakerLabeler.reset() clearing _centroids/_counts while
    match_embedding() is mid-lookup can raise IndexError there; two
    threads both calling Refiner.maybe_refine() race on self.spans).
    Call this only from the SAME thread that is driving run_stream() (an
    in-process embedder with a single-threaded pipeline can call it
    directly, e.g. between two separate run_stream() calls, or from
    inside a callback run_stream() itself invokes), or hand it to a
    ControlQueue that thread is polling instead
    (`control.submit(lambda: reset_live_session(...))`) -- main()'s own
    `POST /reset` wiring does exactly that (see RuntimeControls.reset_fn
    in main()) rather than calling this directly from the HTTP handler
    thread.
    """
    if refiner is not None:
        refiner.maybe_refine(0, force=True)
        refiner._task_queue.join()
    if hub is not None and stats is not None:
        hub.publish(_session_summary_event(stats, labeler))
    if labeler is not None:
        labeler.reset()
    if refiner is not None:
        refiner.reset()
    if stats is not None:
        stats.reset()
    if asr is not None:
        asr.reset_session()
    if hub is not None:
        hub.publish({"type": "session_reset"})


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="wav file to simulate streaming from (16kHz mono s16)")
    ap.add_argument("--no-realtime", action="store_true", help="don't sleep between chunks in --wav mode")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-partial", action="store_true", help="disable in-progress draft subtitles")
    ap.add_argument("--min-silence", type=float, default=0.35,
                    help="silence (s) that ends an utterance; lower = snappier finals, more splits")
    ap.add_argument("--max-speech", type=float, default=12.0,
                    help="force-finalize an utterance after this many seconds of continuous speech")
    ap.add_argument("--max-resident", type=int, default=3,
                    help="max non-tier0 models kept in memory (LRU eviction); 0 or less = unlimited")
    ap.add_argument("--serve", type=int, nargs="?", const=8833, default=None, metavar="PORT",
                    help="serve an OBS browser-source overlay at http://localhost:PORT (default 8833)")
    ap.add_argument("--no-refine", action="store_true",
                    help="disable the second-pass re-decode of utterance groups")
    ap.add_argument("--refine-ja-second-opinion", action="store_true",
                    help="refine pass only: also decode ja utterance groups with "
                         "parakeet-ja and adopt its text when both models agree "
                         "(needs download_models.py --eval-baselines; measured +~250MB RSS)")
    ap.add_argument("--refine-agree-threshold", type=float, default=None, metavar="CER",
                    help="agreement threshold for --refine-ja-second-opinion "
                         "(mutual CER between the two hypotheses; default 0.25)")
    ap.add_argument("--transcript", metavar="PATH",
                    help="append refined transcript lines to this file")
    ap.add_argument("--hotwords", metavar="PATH", default="",
                    help="hotword list (one per line) to bias Japanese decoding")
    ap.add_argument("--replace", metavar="PATH", default="",
                    help="user dictionary: 'wrong=right' per line, applied to all output")
    ap.add_argument("--mode", choices=["single", "balanced", "fast"], default="balanced",
                    help="language-switch policy preset (default balanced). single: fixed to "
                         "--lang, no LID/switching at all. balanced: dual-LID switch "
                         "confirmation for ja/en/zh/ko/yue via SenseVoice, length+repeat-count "
                         "hysteresis for other languages (docs/LID.md). fast: switch "
                         "immediately on any whisper-tiny detection, no confirmation "
                         "(equivalent to --lid-switch-confirm 1 --lang-switch-guard 0). "
                         "The individual --lang-switch-guard/--lid-switch-confirm flags below "
                         "still override the preset's values when passed explicitly.")
    ap.add_argument("--lang", metavar="CODE", default=None,
                    help="required by --mode single: force every segment to this language "
                         "code, skipping LID and switch logic entirely")
    ap.add_argument("--lang-switch-guard", type=float, default=None, metavar="SEC",
                    help="treat a new-language detection shorter than SEC as noise: it never "
                         "counts toward confirming a switch (see --lid-switch-confirm) and it "
                         "suppresses the omnilingual fallback on an empty decode "
                         "(0 disables; raise for single-language streams). Only used as the "
                         "fallback policy for languages SenseVoice can't confirm (see --mode). "
                         "Default depends on --mode (2.0 for balanced, 0 for fast).")
    ap.add_argument("--lid-switch-confirm", type=int, default=None, metavar="N",
                    help="consecutive same-language detections (each >= --lang-switch-guard "
                         "long) required before the session switches to a new language; "
                         "raise for stickier single-language sessions. Only used as the "
                         "fallback policy for languages SenseVoice can't confirm (see --mode). "
                         "Default depends on --mode (2 for balanced, 1 for fast).")
    ap.add_argument("--speakers", action="store_true",
                    help="label utterances with speaker ids (S1, S2, ...)")
    ap.add_argument("--speaker-remap-threshold", type=float, default=None, metavar="T",
                    help="cosine similarity threshold for the refine-path local-cluster-to-"
                         "global remap (speaker_id.SpeakerLabeler.remap_threshold), independent "
                         "of the fast-path SIM_THRESHOLD. Default: same as the fast path "
                         "(speaker_id.SIM_THRESHOLD). See docs/DIARIZATION_PLAN.md section 8.")
    ap.add_argument("--speaker-merge", action="store_true",
                    help="iteration 6 (docs/DIARIZATION_PLAN.md section 9) mitigation for "
                         "speaker-count overestimation: after each refine group's remap, fold "
                         "together any two global speaker centroids that have drifted close "
                         "enough to look like the same person. Off by default.")
    ap.add_argument("--speaker-merge-threshold", type=float, default=None, metavar="T",
                    help="cosine similarity above which two global centroids merge, when "
                         "--speaker-merge. Default: speaker_id.MERGE_THRESHOLD.")
    ap.add_argument("--speaker-hysteresis", action=argparse.BooleanOptionalAction, default=None,
                    help="iteration 6 (docs/DIARIZATION_PLAN.md section 9) mitigation for "
                         "speaker-count overestimation: a newly opened global speaker displays "
                         "under its nearest confirmed speaker's label until it has recurred "
                         "--speaker-hysteresis-min-hits times. Default: speaker_id."
                         "SpeakerLabeler's own default (False -- clean on the AMI meeting sweep "
                         "but rejected after testdata/two_speakers.wav showed it can "
                         "permanently swallow a real speaker who only speaks once, see "
                         "speaker_id.py). Pass --speaker-hysteresis to opt in anyway; useful "
                         "mainly for many-speaker meetings, not short 1-2 speaker sessions.")
    ap.add_argument("--speaker-hysteresis-min-hits", type=int, default=None, metavar="N",
                    help="hits required to confirm a provisional speaker, when "
                         "--speaker-hysteresis. Default: speaker_id.HYSTERESIS_MIN_HITS.")
    ap.add_argument("--speaker-min-remap-update-s", type=float, default=0.0, metavar="S",
                    help="Round 4 (docs/DIARIZATION_PLAN.md section 14) T2 experiment: a "
                         "refine-group local diarization cluster shorter than this many "
                         "seconds remaps READ-ONLY -- it can still match an existing global "
                         "speaker, but never folds its (likely noisier, since it's short) "
                         "embedding into that speaker's centroid and never opens a brand-new "
                         "S{n} on a miss, falling back to the group's majority fast-path label "
                         "instead. 0.0 (default) is a no-op.")
    ap.add_argument("--speaker-joint-remap", action="store_true",
                    help="Round 5 (docs/DIARIZATION_PLAN.md section 15) T1 experiment: within "
                         "one refine group, solve the local-cluster-to-global remap jointly "
                         "(Hungarian assignment maximizing total similarity) instead of matching "
                         "each local cluster independently, so two distinct local clusters "
                         "can't both land on the same global speaker. Off by default pending "
                         "measurement; see speaker_id.SpeakerLabeler.match_embeddings_joint().")
    ap.add_argument("--speaker-exclude-provisional-remap", action="store_true",
                    help="Round 5 (docs/DIARIZATION_PLAN.md section 15) T3 experiment: a global "
                         "speaker centroid that hasn't yet been matched a second time (still "
                         "provisional, see speaker_id.PROVISIONAL_CONFIRM_HITS) is never chosen "
                         "as a remap target -- it may be 'stealing' a match that should have gone "
                         "to a real, already-recurring speaker. Off by default. See "
                         "speaker_id.SpeakerLabeler.match_embedding()'s exclude_provisional "
                         "docstring.")
    ap.add_argument("--speaker-global-recluster", action="store_true",
                    help="Round 7 (docs/DIARIZATION_PLAN.md section 17) experiment: at "
                         "shutdown, re-cluster every refine group's local diarization cluster "
                         "embeddings accumulated over the whole session (two-stage constrained "
                         "agglomerative, same design as eval_diar_overlap.py's Round 6 "
                         "prototype) and print a diagnostic summary of the result. Off by "
                         "default. Does NOT change live output, the SSE stream, or the "
                         "transcript file -- see Refiner.run_global_recluster()'s docstring "
                         "for why revising already-printed labels is out of scope this round.")
    ap.add_argument("--speaker-global-recluster-threshold", type=float, default=0.65,
                    metavar="T",
                    help="cosine-distance threshold for --speaker-global-recluster's "
                         "agglomerative merge stage (default 0.65).")
    ap.add_argument("--speaker-num-clusters-hint",
                    choices=["off", "confirmed", "confirmed-capped"], default="off",
                    help="Round 9 (docs/DIARIZATION_PLAN.md section 19) Experiment A: pass a "
                         "num_clusters hint to diarize.GroupDiarizer's FastClustering, derived "
                         "from speaker_id.SpeakerLabeler.num_confirmed_speakers() at the moment "
                         "each refine group closes. 'off' (default) is a no-op. 'confirmed' uses "
                         "the confirmed count directly; 'confirmed-capped' additionally clamps it "
                         "to min(confirmed, this group's fast-path span count). See "
                         "Refiner._emit_turns()'s num_clusters_hint_value docstring.")
    ap.add_argument("--speaker-num-clusters-hint-min-s", type=float, default=0.0, metavar="S",
                    help="Only apply --speaker-num-clusters-hint when this refine group's total "
                         "VAD-detected speech duration (seconds) is >= this value; below it, the "
                         "group falls back to no hint. 0.0 (default) applies the hint to every "
                         "group.")
    ap.add_argument("--translate", nargs="?", const="en", default=None, metavar="LANGS",
                    help="translate Japanese lines to these languages, comma-separated "
                         "(default en). en=FuguMT; any other M2M-100 target code "
                         "(zh, ko, es, fr, de, ...) is accepted if the model's vocabulary "
                         "supports it. Only zh/ko have measured translation quality so far "
                         "-- other targets print an 'unvalidated' note to stderr, see "
                         "docs/TRANSLATE_M2M.md")
    ap.add_argument("--input", choices=["mic", "wav", "ws"], default=None,
                    help="audio source; default is mic, or wav if --wav is given")
    ap.add_argument("--ws-host", default="127.0.0.1", metavar="HOST",
                    help="bind host for --input ws (default 127.0.0.1, localhost-only; "
                         "pass --ws-host 0.0.0.0 to also accept connections from other "
                         "devices on the LAN, e.g. a phone or an ESP32 board -- do this "
                         "only on a network you trust, the /ingest endpoint has no auth)")
    ap.add_argument("--ws-port", type=int, default=8766, metavar="PORT",
                    help="port for the --input ws /ingest endpoint (default 8766)")
    args = ap.parse_args()

    if args.mode == "single" and not args.lang:
        ap.error("--mode single requires --lang CODE")
    # --mode bundles defaults for the two hysteresis knobs; an explicitly
    # passed --lang-switch-guard/--lid-switch-confirm still wins. "single"
    # has no entry here: forced_lang (set below) bypasses all switch/
    # hysteresis logic in asr_engine.RoutedASR.transcribe(), so these two
    # knobs are never read for that mode.
    mode_defaults = {"balanced": (2.0, 2), "fast": (0.0, 1)}
    default_guard, default_confirm = mode_defaults.get(args.mode, (2.0, 2))
    if args.lang_switch_guard is None:
        args.lang_switch_guard = default_guard
    if args.lid_switch_confirm is None:
        args.lid_switch_confirm = default_confirm

    # GitHub issue #29: the EventHub is created unconditionally -- every
    # structured event below is published on it whether or not --serve's
    # HTTP/SSE layer exists, so an app embedding this module can
    # hub.add_listener(...) and get model_load/warning/session_summary/etc.
    # without running a local HTTP server at all. --serve just adds that
    # HTTP layer as an optional front end over the same hub.
    hub = EventHub()
    # Review round 1: a POST /reset (or any other future control operation)
    # must run ON the decode thread, not the HTTP handler thread -- see
    # ControlQueue's docstring. Polled by every run_stream() call below.
    control = ControlQueue()
    server = None
    if args.serve:
        server = SubtitleServer(port=args.serve, hub=hub).start()
        print(f"subtitle overlay: http://localhost:{args.serve}/  (OBS browser source)",
              file=sys.stderr)

    print("loading models...", file=sys.stderr)
    try:
        asr = RoutedASR(threads=args.threads,
                        max_resident=args.max_resident if args.max_resident > 0 else None,
                        hotwords_file=args.hotwords, replace_file=args.replace,
                        lid_switch_confirm=max(args.lid_switch_confirm, 1),
                        dual_confirm=(args.mode != "fast"),
                        forced_lang=args.lang if args.mode == "single" else None,
                        ja_second_opinion=args.refine_ja_second_opinion,
                        on_event=hub.publish,
                        **({"agree_threshold": args.refine_agree_threshold}
                           if args.refine_agree_threshold is not None else {}))
        live_vad = LiveVad(args.min_silence, args.max_speech)
    except ModelUnavailable as exc:
        # A required model is missing under models/ -- asr_engine and
        # build_vad() both guard sherpa-onnx's native constructors so this
        # is a catchable exception, not the process silently exit()ing out
        # from inside the C++ layer. Fail with a clear message instead.
        ap.error(f"required model '{exc.name}' not found under models/ -- "
                 f"run scripts/download_models.py")
        return
    asr.min_switch_s = max(args.lang_switch_guard, 0.0)
    if server is not None:
        # lets /replacements and /itn_overrides on the subtitle server read
        # and update this engine's live postprocessing dictionaries
        server.asr = asr
    stats = SessionStats()
    printer = PartialPrinter(enabled=not args.no_partial, hub=hub)

    speaker_labeler = None
    diarizer = None
    if args.speakers:
        from speaker_id import SpeakerLabeler

        # None here means "use speaker_id.py's own default (REMAP_THRESHOLD)",
        # not "same as the fast-path threshold" -- only pass remap_threshold
        # through when --speaker-remap-threshold was actually given, so the
        # SpeakerLabeler constructor's own default applies otherwise.
        # --speaker-merge is a plain store_true: its class default (False,
        # not adopted -- see speaker_id.py section 9) is exactly what
        # omitting the flag should give, so there's nothing to distinguish
        # from "not passed". --speaker-hysteresis is tri-state
        # (None/True/False) for the same reason as --speaker-remap-threshold
        # above, even though its class default is also False (not adopted):
        # kept tri-state so --no-speaker-hysteresis stays available to
        # force it off explicitly if the class default ever changes.
        speaker_kwargs = {"merge_enabled": args.speaker_merge}
        if args.speaker_hysteresis is not None:
            speaker_kwargs["hysteresis_enabled"] = args.speaker_hysteresis
        if args.speaker_remap_threshold is not None:
            speaker_kwargs["remap_threshold"] = args.speaker_remap_threshold
        if args.speaker_merge_threshold is not None:
            speaker_kwargs["merge_threshold"] = args.speaker_merge_threshold
        if args.speaker_hysteresis_min_hits is not None:
            speaker_kwargs["hysteresis_min_hits"] = args.speaker_hysteresis_min_hits
        speaker_labeler = SpeakerLabeler(**speaker_kwargs)
        try:
            from diarize import GroupDiarizer

            diarizer = GroupDiarizer()
        except FileNotFoundError as exc:
            # missing pyannote segmentation model: --speakers still works
            # with the fast-path-only labeling (majority-vote in refine),
            # just without the per-turn refine split. Not fatal.
            print(f"[refine] speaker diarization disabled: {exc}", file=sys.stderr)

    # GitHub issue #29: the pool (and the worker reading from it) are
    # always constructed, even with no --translate targets at startup, so
    # a target added later at runtime (POST /config's "translate" key,
    # RuntimeControls.apply_config -> TranslatorPool.add_target) has a
    # live worker already pulling ja finals to translate -- without this,
    # a runtime-added target would sit in the pool with nothing ever
    # calling it. An empty pool costs nothing: TranslationWorker._run()
    # iterates zero targets per submitted line until one is added.
    translator_pool = TranslatorPool(hub=hub)
    if args.translate:
        print(f"loading translators ({args.translate})...", file=sys.stderr)
        build_translators(args.translate, translator_pool)
    translator_worker = TranslationWorker(translator_pool, hub)

    history = AudioHistory(SAMPLE_RATE)
    refiner = None if args.no_refine else Refiner(asr, history, SAMPLE_RATE, printer,
                                                  transcript_path=args.transcript,
                                                  translators=translator_pool, stats=stats,
                                                  speaker_labeler=speaker_labeler,
                                                  diarizer=diarizer,
                                                  min_remap_update_s=args.speaker_min_remap_update_s,
                                                  joint_remap=args.speaker_joint_remap,
                                                  exclude_provisional_remap=args.speaker_exclude_provisional_remap,
                                                  global_recluster=args.speaker_global_recluster,
                                                  global_recluster_threshold=args.speaker_global_recluster_threshold,
                                                  num_clusters_hint=args.speaker_num_clusters_hint,
                                                  num_clusters_hint_min_s=args.speaker_num_clusters_hint_min_s)

    if server is not None:
        # GitHub issue #29: wired only when --serve is running an HTTP
        # server for POST /config and POST /reset to actually reach --
        # RuntimeControls itself has no HTTP dependency, but there is no
        # endpoint to call it without --serve, and an app embedding this
        # module directly can just call these same setters/reset_live_
        # session() in-process without needing this wrapper at all.
        #
        # Review round 1: reset_live_session() is not safe to call
        # directly from this HTTP handler thread (see its own docstring),
        # so this SUBMITS it to `control` instead -- run_stream()'s decode
        # loop actually runs it, once its next per-chunk control.poll()
        # comes around -- and waits for that to happen. RESET_TIMEOUT_S
        # bounds the wait: if run_stream() isn't currently pulling chunks
        # to poll from (no --wav/mic/ws source active, or --input ws with
        # no client sending audio -- ws_chunks() blocks on the network
        # queue between chunks, so poll() isn't reached until one
        # arrives), the reset stays queued and this returns False so the
        # HTTP handler can answer 202 instead of hanging the request or
        # falsely claiming 200 before the reset actually ran.
        def _do_reset() -> bool:
            done = control.submit(
                lambda: reset_live_session(asr, refiner, speaker_labeler, stats, hub,
                                           live_vad=live_vad))
            return done.wait(timeout=RESET_TIMEOUT_S)

        server.controls = RuntimeControls(asr=asr, translator_pool=translator_pool,
                                          live_vad=live_vad, reset_fn=_do_reset)

    def finish(sr):
        live_vad.flush()
        drain_segments(live_vad, sr, asr, stats, printer, history,
                       refiner=refiner,
                       translator_worker=translator_worker,
                       speaker_labeler=speaker_labeler)
        if refiner is not None:
            refiner.maybe_refine(0, force=True)
            # maybe_refine(force=True) only blocks on the ONE task it just
            # enqueued; it returns instantly without enqueueing anything if
            # self.spans was already empty (the last group closed earlier,
            # asynchronously, during run_stream). On long --speakers audio
            # the worker thread routinely lags behind the fast path (per-
            # group diarization + per-turn re-decode is much slower than a
            # plain refine), so a backlog of already-queued-but-not-yet-run
            # groups can still be sitting in _task_queue at shutdown. Since
            # the worker is a daemon thread, exiting here would kill it
            # mid-backlog and silently drop those groups' refine output.
            # Block until every already-queued task has actually run.
            refiner._task_queue.join()
            refiner.run_global_recluster()

    input_mode = args.input or ("wav" if args.wav else "mic")
    # The CLI itself still relies on KeyboardInterrupt (unchanged behavior);
    # stop_event exists for an app embedding this module in a thread it
    # doesn't control via SIGINT -- it can call stop_event.set() from
    # anywhere to unwind run_stream cleanly. Set here too, in the except
    # handler, purely so the two cancellation paths converge on the same
    # state (run_stream has usually already unwound via the exception by
    # the time this runs).
    stop_event = threading.Event()

    try:
        hub.publish({"type": "session_start"})
        if input_mode == "wav":
            if not args.wav:
                ap.error("--input wav requires --wav PATH")
            samples, sr = read_wave(args.wav)  # resampled to SAMPLE_RATE if needed
            run_stream(wav_chunks(samples, sr, realtime=not args.no_realtime),
                       live_vad, sr, asr, stats, printer, refiner, history, translator_worker,
                       speaker_labeler, stop_event=stop_event, control=control)
            finish(sr)
        elif input_mode == "ws":
            from ws_ingest import INGEST_PATH, IngestServer

            ingest = IngestServer(args.ws_host, args.ws_port, sample_rate=SAMPLE_RATE,
                                  subtitle_server=server).start()
            print(f"ws ingest: ws://{args.ws_host}:{args.ws_port}{INGEST_PATH}  "
                  f"(JSON handshake, then binary pcm_s16le frames)", file=sys.stderr)
            run_stream(ws_chunks(ingest), live_vad, SAMPLE_RATE, asr, stats, printer, refiner,
                       history, translator_worker, speaker_labeler, stop_event=stop_event,
                       control=control)
        else:
            run_stream(mic_chunks(stop_event=stop_event), live_vad, SAMPLE_RATE, asr, stats,
                       printer, refiner, history, translator_worker, speaker_labeler,
                       stop_event=stop_event, control=control)
    except KeyboardInterrupt:
        stop_event.set()
        finish(SAMPLE_RATE)
    finally:
        print(f"\n=== session summary: {stats.summary()} ===")
        if speaker_labeler is not None:
            merges = speaker_labeler.merge_history()
            if merges:
                # docs/DIARIZATION_PLAN.md section 9 (design A): merges fold
                # centroids together but don't retroactively rewrite labels
                # already printed under the merged-away S{n} -- this table
                # is how a reader reconciles those older lines by hand.
                pairs = ", ".join(f"{old}->{new}" for old, new in sorted(merges.items()))
                print(f"=== speaker merges (old->current label): {pairs} ===")
            # docs/DIARIZATION_PLAN.md section 10.6 diagnostics: which path
            # (fast per-VAD-segment label() vs. refine-group remap
            # match_embedding()) opened each currently-live global centroid.
            print(f"=== speaker centroids opened by source: "
                  f"{speaker_labeler.centroid_open_counts()} ===")
            print(f"=== speaker centroid detail (label, opened_by, final_match_count): "
                  f"{speaker_labeler.centroid_summary()} ===")
            # issue #11 / docs/DIARIZATION_PLAN.md section 10.8 option B: how
            # many labels never got past their provisional "S{n}?" display
            # (never matched a second time by session end) -- rows already
            # printed under that provisional form are not retroactively
            # rewritten, so this is how a reader sees how many stayed that
            # way for the whole session.
            print(f"=== speaker labels still provisional at session end: "
                  f"{speaker_labeler.provisional_label_count()} ===")
        # GitHub issue #29: the same session_summary shape reset_live_
        # session() publishes mid-session, published here at real process
        # shutdown too, right after the console prints above so both
        # report identical numbers.
        hub.publish(_session_summary_event(stats, speaker_labeler))


if __name__ == "__main__":
    main()
