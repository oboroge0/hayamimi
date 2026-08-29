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

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import sherpa_onnx

from asr_engine import RoutedASR
from audio_utils import resample_linear

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


def mic_chunks():
    import sounddevice as sd

    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         blocksize=WINDOW_SIZE, callback=callback):
        while True:
            yield q.get()


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

    def __init__(self, enabled: bool, server=None):
        self.enabled = enabled
        self.server = server
        self._tty = sys.stdout.isatty()
        self._last_len = 0

    def show(self, text: str):
        if not self.enabled or not text:
            return
        if self.server is not None:
            self.server.partial(text)
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

    def summary(self) -> str:
        if not self.latencies_ms:
            return f"total_audio={self.total_audio_s:.1f}s segments=0"
        mean = sum(self.latencies_ms) / len(self.latencies_ms)
        return (f"total_audio={self.total_audio_s:.1f}s segments={self.segments} "
                f"mean_latency={mean:.0f}ms max_latency={max(self.latencies_ms):.0f}ms "
                f"refine_lang_corrections={self.refine_lang_corrections} "
                f"refine_groups_closed={self.refine_groups_closed} "
                f"refine_lang_boundary_flushes={self.refine_lang_boundary_flushes}")


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
        if printer.server is not None:
            printer.server.final(result["text"], result["lang"], display_speaker,
                                 latency_ms, result.get("tier", ""))
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


def build_translators(langs: str) -> dict:
    """"en,zh,ko,es,..." -> {lang: translator}.

    en uses the dedicated FuguMT module; any other target is accepted if
    M2M-100's vocabulary has a token for it (see
    translate_m2m.is_supported_target()). Only a subset of those targets have
    measured translation quality (translate_m2m.VALIDATED_TARGETS) --
    constructing a translator for an unvalidated target prints a note to
    stderr but still works.
    """
    out = {}
    for lang in [x.strip() for x in langs.split(",") if x.strip()]:
        if lang == "en":
            from translate_ja_en import TranslatorJaEn

            out["en"] = TranslatorJaEn()
        else:
            from translate_m2m import TranslatorM2M, is_supported_target

            if not is_supported_target(lang):
                print(f"unsupported translation target: {lang}", file=sys.stderr)
                continue
            out[lang] = TranslatorM2M(lang)
    return out


class TranslationWorker:
    """Async ja->target translation of finalized lines (console display)."""

    def __init__(self, translators: dict, server=None):
        self._translators = translators
        self._server = server
        self._q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text: str):
        self._q.put(text)

    def _run(self):
        while True:
            text = self._q.get()
            for lang, tr in self._translators.items():
                out = safe_translate(tr, text)
                if out != text:  # fallback returns the source: nothing worth showing
                    print(f"[→{lang}] {out}", flush=True)
                    if self._server is not None:
                        self._server.publish({"type": "translation", "lang": lang, "text": out})


from asr_engine import (  # shared with the engine's live correction / refine dual-LID confirm
    ModelUnavailable, REFINE_MIN_REGROUP_S, SV_LANGS, resolve_refine_lang, script_corrected_lang,
    sv_lid_tag,
)

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
                 translators: dict | None = None, stats: "SessionStats | None" = None,
                 speaker_labeler=None, diarizer=None, min_remap_update_s: float = 0.0,
                 joint_remap: bool = False, exclude_provisional_remap: bool = False,
                 global_recluster: bool = False,
                 global_recluster_threshold: float = 0.65):
        self.asr = asr
        self.history = history
        self.sr = sample_rate
        self.printer = printer
        self.translators = translators or {}  # ja->target, synchronous per refine
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
                    majority_speaker: str = "") -> bool:
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
            raw = self.diarizer.process(buf, self.sr)
        except Exception as exc:
            print(f"[refine] diarization failed, falling back to single-speaker line: {exc}",
                  file=sys.stderr)
            return False
        if len({local for local, _, _ in raw}) < 2:
            return False

        turns = []  # (local_id, start_sample, end_sample) within buf
        for local_id, start_s, end_s in raw:
            start = max(0, int(round(start_s * self.sr)))
            end = min(len(buf), int(round(end_s * self.sr)))
            if end - start >= int(self.MIN_TURN_S * self.sr):
                turns.append((local_id, start, end))
        if len(turns) < 2:
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

        outputs = []  # (global_label, turn_text), in chronological turn order
        for local_id, start, end in turns:
            turn_text = self.asr.transcribe(buf[start:end], self.sr, known_lang=refine_lang,
                                            live=False)["text"]
            if turn_text.strip():
                outputs.append((global_label[local_id], turn_text))

        total_text = " ".join(t for _, t in outputs)
        if len(total_text.strip()) < 0.7 * len(fast_joined):
            return False

        for label, turn_text in outputs:
            # label is the canonical assignment from match_embedding() above
            # (unchanged); disp is only for what actually gets printed here
            # -- issue #11 / section 10.8 option B.
            disp = self.speaker_labeler.display_label(label) if label else label
            tag = f"{disp}|{refine_lang}" if disp else refine_lang
            print(f"[refine/{tag}] {turn_text}", flush=True)
            if self.printer.server is not None:
                self.printer.server.publish({"type": "refine", "text": turn_text,
                                             "lang": refine_lang, "speaker": disp})
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
            if not mixed and self._emit_turns(buf, refine_lang, fast_joined, speaker):
                return

            # speaker is the majority vote over the group's canonical
            # per-segment labels (unchanged); disp is only for display --
            # issue #11 / section 10.8 option B.
            disp = (self.speaker_labeler.display_label(speaker)
                    if speaker and self.speaker_labeler is not None else speaker)
            tag = f"{disp}|{refine_lang}" if disp else refine_lang
            print(f"[refine/{tag}] {text}", flush=True)
            if self.printer.server is not None:
                self.printer.server.publish({"type": "refine", "text": text, "lang": refine_lang,
                                             "speaker": disp})
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
               speaker_labeler=None):
    audio_pos = 0.0
    last_partial = 0.0
    early_lang = None  # LID result computed mid-utterance so finals skip it
    if history is None:
        history = AudioHistory(sample_rate)
    for chunk in chunks:
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


def main():
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
    ap.add_argument("--translate", nargs="?", const="en", default=None, metavar="LANGS",
                    help="translate Japanese lines to these languages, comma-separated "
                         "(default en). en=FuguMT; any other M2M-100 target code "
                         "(zh, ko, es, fr, de, ...) is accepted if the model's vocabulary "
                         "supports it. Only zh/ko have measured translation quality so far "
                         "-- other targets print an 'unvalidated' note to stderr, see "
                         "docs/TRANSLATE_M2M.md")
    ap.add_argument("--input", choices=["mic", "wav", "ws"], default=None,
                    help="audio source; default is mic, or wav if --wav is given")
    ap.add_argument("--ws-host", default="0.0.0.0", metavar="HOST",
                    help="bind host for --input ws (default 0.0.0.0, so an ESP32/phone "
                         "on the LAN can reach it; use 127.0.0.1 to restrict to localhost)")
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

    server = None
    if args.serve:
        from subtitle_server import SubtitleServer

        server = SubtitleServer(port=args.serve).start()
        print(f"subtitle overlay: http://localhost:{args.serve}/  (OBS browser source)",
              file=sys.stderr)

    print("loading models...", file=sys.stderr)
    asr = RoutedASR(threads=args.threads,
                    max_resident=args.max_resident if args.max_resident > 0 else None,
                    hotwords_file=args.hotwords, replace_file=args.replace,
                    lid_switch_confirm=max(args.lid_switch_confirm, 1),
                    dual_confirm=(args.mode != "fast"),
                    forced_lang=args.lang if args.mode == "single" else None)
    asr.min_switch_s = max(args.lang_switch_guard, 0.0)
    vad = build_vad(args.min_silence, args.max_speech)
    stats = SessionStats()
    printer = PartialPrinter(enabled=not args.no_partial, server=server)

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

    translators = {}
    translator_worker = None
    if args.translate:
        print(f"loading translators ({args.translate})...", file=sys.stderr)
        translators = build_translators(args.translate)
        if translators:
            translator_worker = TranslationWorker(translators, server=server)

    history = AudioHistory(SAMPLE_RATE)
    refiner = None if args.no_refine else Refiner(asr, history, SAMPLE_RATE, printer,
                                                  transcript_path=args.transcript,
                                                  translators=translators, stats=stats,
                                                  speaker_labeler=speaker_labeler,
                                                  diarizer=diarizer,
                                                  min_remap_update_s=args.speaker_min_remap_update_s,
                                                  joint_remap=args.speaker_joint_remap,
                                                  exclude_provisional_remap=args.speaker_exclude_provisional_remap,
                                                  global_recluster=args.speaker_global_recluster,
                                                  global_recluster_threshold=args.speaker_global_recluster_threshold)

    def finish(sr):
        vad.flush()
        drain_segments(vad, sr, asr, stats, printer, history,
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

    try:
        if server is not None:
            server.publish({"type": "session_start"})
        if input_mode == "wav":
            if not args.wav:
                ap.error("--input wav requires --wav PATH")
            samples, sr = read_wave(args.wav)  # resampled to SAMPLE_RATE if needed
            run_stream(wav_chunks(samples, sr, realtime=not args.no_realtime),
                       vad, sr, asr, stats, printer, refiner, history, translator_worker,
                       speaker_labeler)
            finish(sr)
        elif input_mode == "ws":
            from ws_ingest import INGEST_PATH, IngestServer

            ingest = IngestServer(args.ws_host, args.ws_port, sample_rate=SAMPLE_RATE,
                                  subtitle_server=server).start()
            print(f"ws ingest: ws://{args.ws_host}:{args.ws_port}{INGEST_PATH}  "
                  f"(JSON handshake, then binary pcm_s16le frames)", file=sys.stderr)
            run_stream(ws_chunks(ingest), vad, SAMPLE_RATE, asr, stats, printer, refiner,
                       history, translator_worker, speaker_labeler)
        else:
            run_stream(mic_chunks(), vad, SAMPLE_RATE, asr, stats, printer, refiner, history,
                       translator_worker, speaker_labeler)
    except KeyboardInterrupt:
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


if __name__ == "__main__":
    main()
