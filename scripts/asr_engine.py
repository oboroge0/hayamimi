"""Routed multilingual ASR engine on top of sherpa-onnx.

Piper-style tiered catalog: a whisper-tiny spoken-language identifier routes
each audio segment to the best model for that language.

  tier 0  ja                   -> ReazonSpeech k2 zipformer (best real-speech ja, fastest)
  tier 1  zh                   -> Paraformer-zh (best real-speech zh)
  tier 1  ko/yue               -> SenseVoice small
  tier 2  en + 24 EU langs     -> Parakeet TDT v3 (casing + punctuation)
  tier 3  everything else      -> Omnilingual ASR 300M CTC (1600+ languages)

Models are loaded lazily on first use and, when `max_resident` is set, the
least-recently-used ones are unloaded so memory stays bounded no matter how
many languages a session wanders through.
"""
import glob
import os
import sys
import threading
import time

import numpy as np
import sherpa_onnx

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
V3_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
SV_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
OMNI_MODEL_DIR = os.path.join(MODELS_DIR, "omnilingual-300m-ctc-int8")
WHISPER_TINY_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-whisper-tiny")
RZ_MODEL_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17")
PARA_ZH_DIR = os.path.join(MODELS_DIR, "sherpa-onnx-paraformer-zh-int8-2025-10-07")

# ReazonSpeech ja-en zipformer: on real broadcast Japanese it beats even
# whisper-turbo (CER 8.6% vs 13.8%) at RTF 0.02. See docs/EVAL_REAL.md.
# English goes to v3 instead: rz outputs unpunctuated ALL-CAPS English
# (WER 1.6% vs v3's 2.5%, but v3's casing/punctuation reads far better).
RZ_LANGS = {"ja"}

# Paraformer-zh beats SenseVoice on real Chinese (CER 5.6% vs 7.5%); the
# dedicated Korean zipformer is worse (30%), so ko stays on SenseVoice.
# See docs/EVAL_REAL_ZHKO.md.
PARA_LANGS = {"zh"}

# SenseVoice small coverage (built-in ITN and punctuation).
SV_LANGS = {"ko", "yue"}

# Languages covered by the Parakeet-TDT-0.6B-v3 multilingual model.
V3_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
}

LID_MAX_SECONDS = 4.0  # only feed the first N seconds of a segment to the LID model


def _find(model_dir: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


def _build_reazon(threads: int, hotwords_file: str = "", hotwords_score: float = 2.0):
    # modified_beam_search: CER 8.6% -> 5.8% on real broadcast ja for +25%
    # decode time (still 37x realtime). v3/en showed no gain and stays greedy.
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=_find(RZ_MODEL_DIR, "encoder-*.int8.onnx"),
        decoder=_find(RZ_MODEL_DIR, "decoder-*.int8.onnx"),
        joiner=_find(RZ_MODEL_DIR, "joiner-*.int8.onnx"),
        tokens=os.path.join(RZ_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
        model_type="zipformer",
        decoding_method="modified_beam_search",
        hotwords_file=hotwords_file,
        hotwords_score=hotwords_score,
        modeling_unit="cjkchar",
    )


def _build_paraformer_zh(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=_find(PARA_ZH_DIR, "model*.onnx"),
        tokens=os.path.join(PARA_ZH_DIR, "tokens.txt"),
        num_threads=threads,
    )


def _build_sense_voice(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=_find(SV_MODEL_DIR, "model*.onnx"),
        tokens=os.path.join(SV_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
        use_itn=True,
        language="",  # auto: SenseVoice has its own internal LID for its 5 langs
    )


def _build_v3_recognizer(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=_find(V3_MODEL_DIR, "encoder*.onnx"),
        decoder=_find(V3_MODEL_DIR, "decoder*.onnx"),
        joiner=_find(V3_MODEL_DIR, "joiner*.onnx"),
        tokens=os.path.join(V3_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
        model_type="nemo_transducer",
    )


def _build_omnilingual(threads: int):
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=_find(OMNI_MODEL_DIR, "model*.onnx"),
        tokens=os.path.join(OMNI_MODEL_DIR, "tokens.txt"),
        num_threads=threads,
    )


def _build_lid(threads: int):
    whisper_cfg = sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
        encoder=_find(WHISPER_TINY_DIR, "tiny-encoder.int8.onnx"),
        decoder=_find(WHISPER_TINY_DIR, "tiny-decoder.int8.onnx"),
    )
    cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(whisper=whisper_cfg, num_threads=threads)
    return sherpa_onnx.SpokenLanguageIdentification(cfg)


def _has_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def script_corrected_lang(tagged: str, text: str) -> str:
    """Correct an LID tag that contradicts the script of the decoded text."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return tagged
    cjk = sum(1 for c in letters if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    hangul = sum(1 for c in letters if "가" <= c <= "힯")
    if hangul / len(letters) > 0.3 and tagged != "ko":
        return "ko"
    if tagged == "ko" and hangul == 0 and len(letters) >= 4:
        # SenseVoice Korean output is hangul; hangul-free "ko" is a mislabel
        return "zh" if cjk / len(letters) > 0.3 else "en"
    frac = cjk / len(letters)
    if frac > 0.3 and tagged not in ("ja", "zh", "yue", "ko"):
        return "ja"
    if frac < 0.05 and tagged == "ja" and len(letters) >= 8:
        return "en"
    return tagged


def resolve_sticky_lang(
    lang: str, last_lang: str | None, speech_s: float | None,
    min_switch_s: float, switch_confirm: int,
    pending_lang: str | None, pending_count: int,
) -> tuple[str, bool, str | None, int]:
    """Sticky-LID hysteresis: decide whether to accept a new LID detection
    as a real language switch, or hold the session's current language.

    A single new-language detection can be a babble-noise misfire
    (docs/NOISE.md -- whisper-tiny LID exposes no confidence score to
    threshold on) or a jingle/SFX blip (docs/VIDEO_TEST.md) rather than a
    genuine switch. A real switch -- the speaker changing language, or a new
    speaker -- repeats the SAME new language on the next detection, while a
    misfire lands on a random one. `switch_confirm` CONSECUTIVE detections
    of one new language are required before switching; staying on the
    current language needs no confirmation (asymmetric, so noise can't lock
    the session onto a wrong language). This costs a genuine switch at most
    `switch_confirm - 1` segments of latency.

    `min_switch_s` (--lang-switch-guard) is the noise filter on each
    individual candidate detection: a new-language segment shorter than this
    is presumed non-speech (jingle/SFX/misfire) and does NOT advance the
    switch_confirm counter at all -- it neither starts nor extends a
    pending candidate, and it doesn't reset one either, since a real
    candidate already accumulating shouldn't be wiped out by an unrelated
    short blip. This is what makes --lang-switch-guard actually control
    switch stickiness (GitHub issue #2): only detections at or above the
    guard length can ever confirm a switch. It also suppresses the
    omnilingual fallback for that segment, so a held language's empty
    decode isn't resurrected by it.

    Returns (resolved_lang, suppress_fallback, new_pending_lang, new_pending_count).
    """
    if last_lang is None or lang == last_lang:
        return lang, False, None, 0

    is_short = speech_s is not None and speech_s < min_switch_s
    if is_short:
        return last_lang, True, pending_lang, pending_count

    if lang == pending_lang:
        pending_count += 1
    else:
        pending_lang, pending_count = lang, 1

    if pending_count < switch_confirm:
        # Hold the session language for this segment; it's a genuine-speech
        # candidate (>= min_switch_s) merely decoded under the wrong tier's
        # model, so let the omni fallback have a shot if that specialist
        # draws a blank.
        return last_lang, False, pending_lang, pending_count

    return lang, False, None, 0


def _cjkchar_units(phrase: str) -> list[str]:
    """Split a hotword phrase the way sherpa-onnx's cjkchar modeling_unit
    encodes it: each CJK character becomes its own lookup unit, and runs of
    non-CJK, non-whitespace characters are grouped into whole-word units
    (matches the "屈 足 湖" / "GANKE FES" splits seen in sherpa-onnx's own
    "Cannot find ID for token" warnings)."""
    units: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            units.append("".join(buf))
            buf.clear()

    for ch in phrase:
        if ch.isspace():
            flush()
            continue
        if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "가" <= ch <= "힯":
            flush()
            units.append(ch)
        else:
            buf.append(ch)
    flush()
    return units


def _load_token_vocab(tokens_path: str) -> set[str]:
    vocab: set[str] = set()
    if not os.path.isfile(tokens_path):
        return vocab
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # "<token> <id>"; rsplit so a token that itself contains a
            # literal space (rare) still separates cleanly from the id.
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                vocab.add(parts[0])
    return vocab


def check_hotwords_encodable(hotwords_path: str, tokens_path: str) -> tuple[int, int]:
    """Return (total_hotwords, num_unencodable) for the cjkchar modeling_unit
    used by the ja (ReazonSpeech) tier.

    sherpa-onnx encodes each hotword by looking up its cjkchar units
    (see `_cjkchar_units`) directly against tokens.txt. ReazonSpeech's
    tokens.txt is byte-level BPE, not a cjkchar vocabulary, so every lookup
    normally misses -- sherpa-onnx only reports this as stderr warnings and
    still exits 0 (GitHub issue #1). This lets callers surface that loudly
    instead of leaving it buried in stderr.
    """
    if not hotwords_path or not os.path.isfile(hotwords_path):
        return 0, 0
    vocab = _load_token_vocab(tokens_path)
    total = 0
    bad = 0
    with open(hotwords_path, encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if not phrase or phrase.startswith("#"):
                continue
            total += 1
            units = _cjkchar_units(phrase)
            if not units or any(u not in vocab for u in units):
                bad += 1
    return total, bad


def _load_replacements(path: str) -> list[tuple[str, str]]:
    """User dictionary: one "wrong=right" (or tab/arrow-separated) pair per line."""
    if not path:
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("=", "	", "→"):
                if sep in line:
                    wrong, right = line.split(sep, 1)
                    pairs.append((wrong.strip(), right.strip()))
                    break
    return pairs


# a key file per model: checked BEFORE building, because sherpa-onnx's C++
# layer exits the process (not a catchable exception) on an empty model path
_KEY_FILES = {
    "rz": (RZ_MODEL_DIR, "encoder-*.int8.onnx"),
    "pz": (PARA_ZH_DIR, "model*.onnx"),
    "sv": (SV_MODEL_DIR, "model*.onnx"),
    "v3": (V3_MODEL_DIR, "encoder*.onnx"),
    "omni": (OMNI_MODEL_DIR, "model*.onnx"),
}


def _model_present(name: str) -> bool:
    d, pat = _KEY_FILES[name]
    return bool(_find(d, pat))


_BUILDERS = {
    "rz": _build_reazon,
    "pz": _build_paraformer_zh,
    "sv": _build_sense_voice,
    "v3": _build_v3_recognizer,
    "omni": _build_omnilingual,
}

# preload priority when a residency cap is in effect
_PRELOAD_ORDER = ("pz", "sv", "v3", "omni")


class ModelUnavailable(RuntimeError):
    """Raised when a model tier is not present on disk (--minimal install)."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class RoutedASR:
    """Lazily loads catalog models and routes each segment by detected language.

    max_resident bounds how many recognizers besides tier-0 ("rz", always kept:
    it is the ja/en primary and the default draft model) stay in memory; the
    least recently used one is dropped when the cap would be exceeded.
    """

    def __init__(self, threads: int = 4, warmup: bool = True, preload: bool = True,
                 max_resident: int | None = None, punctuate: bool = True,
                 hotwords_file: str = "", replace_file: str = "",
                 lid_switch_confirm: int = 2):
        self._threads = threads
        self._models: dict[str, object] = {}
        self._last_used: dict[str, float] = {}
        self._max_resident = max_resident
        self._punctuate = punctuate
        self._punct = None
        self._ko_spacer = None
        self._ko_spacer_ok = True
        self._hotwords_file = hotwords_file
        self._warn_hotwords_encodability(hotwords_file)
        self._replacements = _load_replacements(replace_file)
        self._load_lock = threading.Lock()  # punct + registry bookkeeping
        self._model_locks = {name: threading.Lock() for name in _BUILDERS}
        self.last_lang = None  # sticky language from the most recent final
        self._unavailable: set[str] = set()  # models missing on disk (--minimal installs)
        self._pending_lang = None   # candidate new language awaiting confirmation
        self._pending_count = 0
        self.lid_switch_confirm = lid_switch_confirm  # consecutive detections to accept a switch
        self.lid = _build_lid(threads)
        if warmup:
            # LID + tier-0 pay their one-time kernel/allocation costs here
            # so the first real segment isn't penalized.
            silence = np.zeros(16000, dtype=np.float32)
            self._identify_lang(silence, 16000)
            self._decode(self._get("rz"), silence, 16000)
        if preload:
            # pull the other tiers in on a daemon thread so the first
            # non-tier-0 utterance doesn't pay the ~2s model-load cost.
            threading.Thread(target=self._preload_rest, daemon=True).start()

    @staticmethod
    def _warn_hotwords_encodability(hotwords_file: str):
        """Print a loud, hard-to-miss warning if --hotwords entries can't be
        encoded against the ja (ReazonSpeech) tier's tokens.txt.

        sherpa-onnx only reports failed encodes as stderr warnings and still
        exits 0 with a normal-looking transcript, so this was silently doing
        nothing for every ReazonSpeech user until they happened to read
        stderr closely (GitHub issue #1). ReazonSpeech ships a byte-level
        BPE tokens.txt with no bpe.model, so there is currently no
        modeling_unit that encodes cjkchar-style hotwords against it; until
        that's fixed upstream (or hayamimi ships its own bpe.model), the
        best we can do is make the failure visible and point at --replace.
        """
        if not hotwords_file:
            return
        tokens_path = os.path.join(RZ_MODEL_DIR, "tokens.txt")
        total, bad = check_hotwords_encodable(hotwords_file, tokens_path)
        if bad == 0:
            return
        if bad == total:
            print(f"[hayamimi] WARNING: 0/{total} hotwords could be encoded for the ja "
                  f"tier (ReazonSpeech uses byte-level BPE tokens, incompatible with "
                  f"modeling_unit=cjkchar) -- --hotwords will have NO EFFECT on ja "
                  f"output. Use --replace for post-hoc find/replace instead.")
        else:
            print(f"[hayamimi] warning: {bad}/{total} hotwords cannot be encoded for "
                  f"the ja tier (ReazonSpeech uses byte-level BPE tokens); --hotwords "
                  f"will have no effect for these. Consider --replace instead.")

    def _preload_rest(self):
        if self._punctuate:
            self.punct  # first: ja finals need this almost immediately
        self.ko_spacer  # cheap (~1s); fixes SenseVoice's over-split Korean
        silence = np.zeros(16000, dtype=np.float32)
        budget = None if self._max_resident is None else self._max_resident
        for name in _PRELOAD_ORDER:
            if budget is not None and budget <= 0:
                break
            try:
                self._decode(self._get(name), silence, 16000)
            except ModelUnavailable:
                continue  # --minimal install: this tier simply isn't there
            if budget is not None:
                budget -= 1

    @property
    def punct(self):
        """Japanese punctuation restorer (BERT ONNX); None if unavailable."""
        if self._punct is None and self._punctuate:
            with self._load_lock:
                if self._punct is None:
                    try:
                        from punct_ja import PunctuatorJa

                        self._punct = PunctuatorJa()
                    except Exception:
                        self._punctuate = False  # missing model/deps: degrade quietly
        return self._punct

    @property
    def ko_spacer(self):
        """Kiwi morphological analyzer used to re-space Korean output."""
        if self._ko_spacer is None and self._ko_spacer_ok:
            with self._load_lock:
                if self._ko_spacer is None and self._ko_spacer_ok:
                    try:
                        from kiwipiepy import Kiwi

                        spacer = Kiwi()
                        spacer.space("한 국", reset_whitespace=True)  # warmup
                        self._ko_spacer = spacer
                    except Exception:
                        self._ko_spacer_ok = False  # missing dep: degrade quietly
        return self._ko_spacer

    def _get(self, name: str):
        if name in self._unavailable:
            raise ModelUnavailable(name)
        if name not in self._models and not _model_present(name):
            self._unavailable.add(name)
            print(f"[hayamimi] model '{name}' not found under models/ "
                  f"(minimal install?): routing falls back", file=sys.stderr)
            raise ModelUnavailable(name)
        rec = self._models.get(name)
        if rec is None:
            # per-model lock: loading v3 must not block a prefetch of omni
            with self._model_locks[name]:
                rec = self._models.get(name)
                if rec is None:
                    try:
                        if name == "rz":
                            rec = _build_reazon(self._threads, self._hotwords_file)
                        else:
                            rec = _BUILDERS[name](self._threads)
                    except Exception as exc:
                        # a --minimal install ships only some models: degrade
                        self._unavailable.add(name)
                        print(f"[hayamimi] model '{name}' unavailable "
                              f"(minimal install?): routing falls back",
                              file=sys.stderr)
                        raise ModelUnavailable(name) from exc
                    with self._load_lock:
                        self._evict_if_needed(incoming=name)
                        self._models[name] = rec
        self._last_used[name] = time.monotonic()
        return rec

    def _get_with_fallback(self, name: str) -> tuple[object, str]:
        for cand in (name, "rz", "sv", "v3", "pz", "omni"):
            try:
                return self._get(cand), cand
            except ModelUnavailable:
                continue
        raise RuntimeError(
            "no ASR models found under models/ -- run scripts/download_models.py")

    def _evict_if_needed(self, incoming: str):
        if self._max_resident is None or incoming == "rz":
            return
        resident = [n for n in self._models if n != "rz"]
        if len(resident) < self._max_resident:
            return
        victim = min(resident, key=lambda n: self._last_used.get(n, 0.0))
        del self._models[victim]

    @property
    def resident_models(self) -> list[str]:
        return sorted(self._models)

    def _identify_lang(self, samples: np.ndarray, sample_rate: int) -> str:
        clip = samples
        # skip the leading quiet (preroll padding): it eats into the 4s LID
        # window and cost the demo capture its first-utterance language
        loud = np.flatnonzero(np.abs(clip) > 0.015)
        if len(loud) and loud[0] > sample_rate // 10:
            clip = clip[max(loud[0] - sample_rate // 20, 0):]
        max_len = int(LID_MAX_SECONDS * sample_rate)
        if len(clip) > max_len:
            clip = clip[:max_len]
        stream = self.lid.create_stream()
        stream.accept_waveform(sample_rate, clip)
        return self.lid.compute(stream)

    @staticmethod
    def _decode_full(rec, samples: np.ndarray, sample_rate: int) -> tuple[str, str]:
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        rec.decode_stream(stream)
        text = stream.result.text
        # ReazonSpeech models emit TV-subtitle annotation brackets around
        # boundary words; they carry no speech content.
        for junk in ("［", "］", "〈", "〉"):
            text = text.replace(junk, "")
        return text, getattr(stream.result, "lang", "") or ""

    @classmethod
    def _decode(cls, rec, samples: np.ndarray, sample_rate: int) -> str:
        return cls._decode_full(rec, samples, sample_rate)[0]

    def _route(self, lang: str) -> tuple[object, str]:
        if lang in RZ_LANGS:
            return self._get_with_fallback("rz")
        if lang in PARA_LANGS:
            return self._get_with_fallback("pz")
        if lang in SV_LANGS:
            return self._get_with_fallback("sv")
        if lang in V3_LANGS:
            return self._get_with_fallback("v3")
        return self._get_with_fallback("omni")

    def partial(self, samples: np.ndarray, sample_rate: int,
                lang_hint: str | None = None) -> str:
        """Fast draft transcription of an in-progress utterance.

        Prefers the caller's early-LID hint for THIS utterance (so drafts
        switch language as soon as the mid-utterance LID fires, ~2s in);
        otherwise falls back to the session's sticky language. Without
        either, the tier-0 ja/en model drafts. Fixes Korean drafts staying
        blank after an English section (the sticky en model returns nothing
        for Korean speech).
        """
        if lang_hint is not None:
            # this utterance's language is confirmed: use its specialist
            rec, _ = self._route(lang_hint)
            return self._replace(self._decode(rec, samples, sample_rate))

        sticky = self.last_lang
        if sticky in V3_LANGS and sticky != "en":
            # EU languages: SenseVoice can't probe these; trust the session
            rec, _ = self._route(sticky)
            return self._replace(self._decode(rec, samples, sample_rate))

        # Before the early LID lands, never show the previous language's
        # guesswork (an English model romanizing Korean reads as garbage --
        # user feedback). SenseVoice runs its own per-utterance LID over
        # ja/zh/ko/yue/en, so probe with it and follow its tag: the very
        # first draft comes out in the right language.
        try:
            sv_text, sv_tag = self._decode_full(self._get("sv"), samples, sample_rate)
        except ModelUnavailable:
            rec = self._get_with_fallback("rz")[0]
            return self._replace(self._decode(rec, samples, sample_rate))
        if "ja" in sv_tag:
            try:
                rz_text = self._decode(self._get("rz"), samples, sample_rate)
                if rz_text.strip():
                    return self._replace(rz_text)
            except ModelUnavailable:
                pass
        return self._replace(sv_text)

    def _replace(self, text: str) -> str:
        for wrong, right in self._replacements:
            text = text.replace(wrong, right)
        return text

    def identify(self, samples: np.ndarray, sample_rate: int) -> str:
        """Public LID hook so callers can identify the language mid-utterance.

        Also kicks off a background prefetch of that language's model, so by
        the time the utterance finalizes the recognizer is already resident.
        """
        lang = self._identify_lang(samples, sample_rate)
        threading.Thread(target=self._route, args=(lang,), daemon=True).start()
        return lang

    min_switch_s = 2.0  # a shorter utterance can't establish a new language

    def reset_session(self):
        """Clear the sticky/pending language state.

        Call this between unrelated audio streams (a new recording, a new
        speaker with no continuity from the last one, an eval harness moving
        to the next independent clip set). Without it, the sticky-LID
        hysteresis in transcribe() would treat the first utterance of the
        new stream as a language SWITCH away from whatever the previous,
        unrelated stream last said -- costing it an extra confirmation
        segment for no reason, since there was never a real session to
        switch away from.
        """
        self.last_lang = None
        self._pending_lang = None
        self._pending_count = 0

    def transcribe(self, samples: np.ndarray, sample_rate: int,
                   known_lang: str | None = None, speech_s: float | None = None,
                   live: bool = True) -> dict:
        """live=False (e.g. the refine pass re-decoding past audio) must not
        touch the sticky/pending language state of the live stream."""
        if known_lang is not None and (speech_s is None or speech_s < 4.0):
            # trust the mid-utterance early LID only for short utterances; a
            # long one gives the full 4s window a chance to overrule the
            # guess made from its first 2 seconds (first-clip-per-language
            # errors in the demo capture came from exactly this)
            lang, lid_ms = known_lang, 0.0
        else:
            t0 = time.perf_counter()
            lang = self._identify_lang(samples, sample_rate)
            lid_ms = (time.perf_counter() - t0) * 1000

        suppress_fallback = False
        if not live:
            pass  # out-of-band decode: leave the live language state alone
        else:
            lang, suppress_fallback, self._pending_lang, self._pending_count = resolve_sticky_lang(
                lang, self.last_lang, speech_s, self.min_switch_s, self.lid_switch_confirm,
                self._pending_lang, self._pending_count,
            )

        t0 = time.perf_counter()
        sv = None
        if lang == "zh":
            # whisper-tiny LID labels Cantonese as "zh" (measured 0/12 correct
            # on FLEURS yue), so let SenseVoice's internal LID arbitrate: keep
            # its transcript for yue, re-decode with Paraformer for true zh.
            try:
                sv = self._get("sv")
            except ModelUnavailable:
                sv = None  # minimal install: plain routing below
        if sv is not None:
            text, sv_lang = self._decode_full(sv, samples, sample_rate)
            if "yue" in sv_lang:
                lang, tier = "yue", "sv"
            else:
                rec, tier = self._get_with_fallback("pz")
                text2 = self._decode(rec, samples, sample_rate)
                if text2.strip():
                    text = text2
        else:
            rec, tier = self._route(lang)
            text = self._decode(rec, samples, sample_rate)
        if not text.strip() and tier != "omni" and not suppress_fallback:
            # safety net: the specialist came back empty (likely LID mistake);
            # the 1600-language generalist gets the last word.
            text = self._decode(self._get("omni"), samples, sample_rate)
            tier = "omni"
        corrected = script_corrected_lang(lang, text)
        if live and text.strip() and corrected != lang:
            # the decoded script contradicts the LID tag (romaji-mangled
            # English under a ja tag, CJK under a non-CJK tag): re-decode
            # with the right model before anyone sees the final.
            if corrected == "ja" and not _has_kana(text) and "sv" not in self._unavailable:
                # han-only text is just as likely zh/yue as ja; let
                # SenseVoice's internal LID arbitrate instead of assuming
                # (assuming ja here cost yue 7.4% -> 24% CER, iteration 27)
                try:
                    sv2 = self._get("sv")
                except ModelUnavailable:
                    sv2 = None
                text2, sv_lang = (self._decode_full(sv2, samples, sample_rate)
                                  if sv2 is not None else ("", ""))
                if text2.strip():
                    if "yue" in sv_lang:
                        lang, tier, text = "yue", "sv", text2
                    elif "zh" in sv_lang:
                        text3 = self._decode(self._get_with_fallback("pz")[0], samples, sample_rate)
                        lang, tier, text = "zh", "pz", (text3 if text3.strip() else text2)
                    elif "ja" in sv_lang:
                        text3 = self._decode(self._get_with_fallback("rz")[0], samples, sample_rate)
                        lang, tier, text = "ja", "rz", (text3 if text3.strip() else text2)
                    elif "ko" in sv_lang:
                        lang, tier, text = "ko", "sv", text2
            else:
                rec2, tier2 = self._route(corrected)
                text2 = self._decode(rec2, samples, sample_rate)
                if text2.strip():
                    lang, tier, text = corrected, tier2, text2

        text = self._replace(text)
        if lang == "ko" and text.strip() and self.ko_spacer is not None:
            try:
                # SenseVoice emits a space between every token; Kiwi restores
                # real Korean word spacing (docs/BENCHMARKS.md iteration 20)
                text = self.ko_spacer.space(text, reset_whitespace=True)
            except Exception:
                pass
        if lang == "ja" and text.strip() and self.punct is not None:
            try:
                text = self.punct.restore(text)
            except Exception:
                pass  # a punctuation failure must never lose the transcription
        decode_ms = (time.perf_counter() - t0) * 1000

        if live and text.strip():  # empty results must not poison the sticky language
            self.last_lang = lang
        return {"text": text, "lang": lang, "tier": tier, "lid_ms": lid_ms, "decode_ms": decode_ms}
