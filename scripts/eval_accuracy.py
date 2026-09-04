"""ASR accuracy evaluation harness.

Compares two systems on the TTS test set produced by scripts/make_testset.py:
  a) Parakeet (sherpa-onnx): Japanese NeMo CTC model for lang=="ja",
     multilingual NeMo transducer (v3) model for lang=="en".
  b) faster-whisper large-v3-turbo, INT8, CPU.

Metrics:
  - English: WER via jiwer, with basic normalization (lowercase, strip
    punctuation, collapse whitespace).
  - Japanese: CER (character error rate) via a direct Levenshtein
    implementation, with NFKC normalization + stripping of punctuation
    and whitespace.

Also records per-file decode time and RTF (decode_time / audio_duration).

Usage:
    python scripts/eval_accuracy.py
"""
import glob
import json
import os
import re
import string
import sys
import time
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

import jiwer
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "testdata", "eval")
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.json")
DOCS_PATH = os.path.join(ROOT, "docs", "eval", "eval.md")

MODEL_JA_DIR = os.path.join(ROOT, "models", "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8")
MODEL_EN_DIR = os.path.join(ROOT, "models", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")

THREADS = 6

# ---------------------------------------------------------------------------
# Recognizer construction (mirrors scripts/bench_offline.py, proven pattern)
# ---------------------------------------------------------------------------

def _find(model_dir, pattern):
    hits = glob.glob(os.path.join(model_dir, pattern))
    return hits[0] if hits else ""


def build_parakeet_ja():
    import sherpa_onnx

    model = _find(MODEL_JA_DIR, "model*.onnx")
    tokens = os.path.join(MODEL_JA_DIR, "tokens.txt")
    assert model, f"no onnx model found in {MODEL_JA_DIR}"
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model,
        tokens=tokens,
        num_threads=THREADS,
    )


def build_parakeet_en():
    import sherpa_onnx

    encoder = _find(MODEL_EN_DIR, "encoder*.onnx")
    decoder = _find(MODEL_EN_DIR, "decoder*.onnx")
    joiner = _find(MODEL_EN_DIR, "joiner*.onnx")
    tokens = os.path.join(MODEL_EN_DIR, "tokens.txt")
    assert encoder and decoder and joiner, f"missing transducer parts in {MODEL_EN_DIR}"
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=THREADS,
        model_type="nemo_transducer",
    )


class ParakeetSystem:
    """Lazily loads the ja/en Parakeet recognizers on first use."""

    name = "Parakeet"

    def __init__(self):
        self._ja = None
        self._en = None

    def _rec(self, lang):
        if lang == "ja":
            if self._ja is None:
                self._ja = build_parakeet_ja()
            return self._ja
        elif lang == "en":
            if self._en is None:
                self._en = build_parakeet_en()
            return self._en
        raise ValueError(f"unsupported lang: {lang}")

    def transcribe(self, wav_path, lang):
        samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        rec = self._rec(lang)
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


MODEL_SV_DIR = os.path.join(ROOT, "models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")


class SenseVoiceSystem:
    """SenseVoice small INT8 (Mojicast's multilingual model). One model for ja+en."""

    name = "SenseVoice"

    def __init__(self):
        self._rec = {}

    def _get(self, lang):
        # language is fixed at construction time in sherpa-onnx, so keep one
        # recognizer per language (the model itself is shared on disk).
        if lang not in self._rec:
            import sherpa_onnx

            self._rec[lang] = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=_find(MODEL_SV_DIR, "model*.onnx"),
                tokens=os.path.join(MODEL_SV_DIR, "tokens.txt"),
                num_threads=THREADS,
                language=lang,
                use_itn=True,
            )
        return self._rec[lang]

    def transcribe(self, wav_path, lang):
        samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        rec = self._get(lang)
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


MODEL_OMNI_DIR = os.path.join(ROOT, "models", "omnilingual-300m-ctc-int8")


class OmnilingualSystem:
    """Meta Omnilingual ASR 300M CTC INT8 — one model, 1600+ languages, no lang hint."""

    name = "Omnilingual"

    def __init__(self):
        self._rec = None

    def _get(self):
        if self._rec is None:
            import sherpa_onnx

            self._rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
                model=_find(MODEL_OMNI_DIR, "model*.onnx"),
                tokens=os.path.join(MODEL_OMNI_DIR, "tokens.txt"),
                num_threads=THREADS,
            )
        return self._rec

    def transcribe(self, wav_path, lang):
        samples, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        rec = self._get()
        stream = rec.create_stream()
        stream.accept_waveform(sr, samples)
        t0 = time.perf_counter()
        rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        return stream.result.text, dt


class WhisperSystem:
    """Wraps faster-whisper large-v3-turbo, INT8, CPU. Loaded once, lazily."""

    name = "Whisper-turbo"

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, wav_path, lang):
        model = self._get_model()
        t0 = time.perf_counter()
        segments, _info = model.transcribe(wav_path, language=lang, beam_size=1)
        text = "".join(seg.text for seg in segments)
        dt = time.perf_counter() - t0
        return text.strip(), dt


# ---------------------------------------------------------------------------
# Normalization + metrics
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(
    "[" + re.escape(string.punctuation) + "\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65" + "]"
)


def normalize_en(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ja(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    return text


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = cur
    return prev[-1]


def wer_en(ref: str, hyp: str) -> float:
    r = normalize_en(ref)
    h = normalize_en(hyp)
    if len(r.split()) == 0:
        return 0.0
    return jiwer.wer(r, h)


def cer_ja(ref: str, hyp: str):
    r = normalize_ja(ref)
    h = normalize_ja(hyp)
    dist = levenshtein(r, h)
    rate = dist / len(r) if len(r) > 0 else 0.0
    return rate, dist, len(r)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    systems = [ParakeetSystem(), SenseVoiceSystem(), OmnilingualSystem(), WhisperSystem()]

    results = []  # list of dicts, one per (file, system)
    for entry in manifest:
        wav_path = os.path.join(EVAL_DIR, entry["wav"])
        lang = entry["lang"]
        ref = entry["ref"]
        audio_s = sf.info(wav_path).duration

        row = {"wav": entry["wav"], "lang": lang, "ref": ref, "audio_s": audio_s}
        for sys_obj in systems:
            hyp, dt = sys_obj.transcribe(wav_path, lang)
            rtf = dt / audio_s if audio_s > 0 else 0.0
            if lang == "en":
                score = wer_en(ref, hyp)
                dist, denom = None, None
            else:
                score, dist, denom = cer_ja(ref, hyp)
            row[sys_obj.name] = {
                "hyp": hyp,
                "score": score,
                "dist": dist,
                "denom": denom,
                "decode_s": dt,
                "rtf": rtf,
            }
            print(f"[{entry['wav']}] {sys_obj.name}: score={score:.4f} rtf={rtf:.3f} hyp={hyp!r}")
        results.append(row)

    # ---- Aggregate (micro-averaged) ----
    agg = {}
    for sys_obj in systems:
        name = sys_obj.name
        en_num = en_den = 0.0
        ja_dist = ja_denom = 0
        rtf_sum = 0.0
        rtf_n = 0
        for row in results:
            r = row[name]
            rtf_sum += r["rtf"]
            rtf_n += 1
            if row["lang"] == "en":
                r_norm = normalize_en(row["ref"])
                h_norm = normalize_en(r["hyp"])
                r_words = r_norm.split()
                if len(r_words):
                    out = jiwer.process_words(r_norm, h_norm if h_norm else " ")
                    edits = out.substitutions + out.deletions + out.insertions
                    denom = out.substitutions + out.deletions + out.hits
                    en_num += edits
                    en_den += max(denom, 1)
            else:
                ja_dist += r["dist"]
                ja_denom += r["denom"]
        agg[name] = {
            "wer_en": (en_num / en_den) if en_den > 0 else float("nan"),
            "cer_ja": (ja_dist / ja_denom) if ja_denom > 0 else float("nan"),
            "mean_rtf": rtf_sum / rtf_n if rtf_n else float("nan"),
        }

    # ---- Print comparison table ----
    print("\n=== Per-file results ===")
    header = f"{'file':10} {'lang':4} {'system':14} {'metric':8} {'score':8} {'decode_s':9} {'rtf':7}"
    print(header)
    print("-" * len(header))
    for row in results:
        for sys_obj in systems:
            r = row[sys_obj.name]
            metric = "CER" if row["lang"] == "ja" else "WER"
            print(f"{row['wav']:10} {row['lang']:4} {sys_obj.name:14} {metric:8} {r['score']:.4f}   {r['decode_s']:.3f}     {r['rtf']:.3f}")

    print("\n=== Aggregate ===")
    for name, a in agg.items():
        print(f"{name}: WER-en={a['wer_en']:.4f}  CER-ja={a['cer_ja']:.4f}  mean_RTF={a['mean_rtf']:.4f}")

    write_markdown(results, systems, agg)
    print(f"\nWrote {DOCS_PATH}")


def write_markdown(results, systems, agg):
    lines = []
    lines.append("# ASR Accuracy Evaluation\n")
    lines.append(
        "Comparison of **Parakeet** (sherpa-onnx NeMo models) vs **faster-whisper "
        "large-v3-turbo** (INT8, CPU) on a small TTS-generated test set "
        "(8 Japanese + 8 English sentences, `testdata/eval/`).\n"
    )
    lines.append(
        "- Japanese Parakeet model: `sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8` "
        "(NeMo CTC)\n"
        "- English/multilingual Parakeet model: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` "
        "(NeMo transducer)\n"
        "- SenseVoice: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` "
        "(non-autoregressive, use_itn=True, language forced per entry; the model Mojicast uses)\n"
        "- Whisper: `large-v3-turbo`, `device=cpu`, `compute_type=int8`, `beam_size=1`, "
        "language forced per entry\n"
    )

    def esc(s):
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines.append("\n## Per-file results\n")
    sys_cols = "".join(f" {s.name} hyp | {s.name} metric | {s.name} RTF |" for s in systems)
    lines.append("| file | lang | ref |" + sys_cols)
    lines.append("|---|---|---|" + "---|" * (3 * len(systems)))
    for row in results:
        metric_name = "CER" if row["lang"] == "ja" else "WER"
        cells = ""
        for s in systems:
            r = row[s.name]
            cells += f" {esc(r['hyp'])} | {metric_name}={r['score']:.3f} | {r['rtf']:.3f} |"
        lines.append(f"| {row['wav']} | {row['lang']} | {esc(row['ref'])} |" + cells)

    lines.append("\n## Aggregate\n")
    lines.append("| system | WER (en, micro-avg) | CER (ja, micro-avg) | mean RTF |")
    lines.append("|---|---|---|---|")
    for name, a in agg.items():
        lines.append(f"| {name} | {a['wer_en']:.4f} | {a['cer_ja']:.4f} | {a['mean_rtf']:.4f} |")

    lines.append("\n## Caveats\n")
    lines.append(
        "- **TTS audio is easier than real speech.** These references were synthesized "
        "with edge-tts and are clean, single-speaker, noise-free recordings with no "
        "disfluencies, overlaps, accents, or background noise. Real-world microphone "
        "input will show meaningfully higher error rates than reported here.\n"
    )
    lines.append(
        "- **Small sample.** Only 8 Japanese and 8 English sentences were evaluated. "
        "These numbers are indicative of relative system behavior, not statistically "
        "robust estimates — a single unusual sentence can swing the aggregate noticeably.\n"
    )
    lines.append(
        "- **Residual normalization noise from number/date formatting.** TTS references "
        "were written with a mix of kanji numerals, half-width digits, and spelled-out "
        "forms (e.g. \u4e09\u6642 vs 3\u6642, \"twenty three\" vs \"23\"), while ASR systems often "
        "output digits regardless of how the reference was written. NFKC normalization plus "
        "punctuation/space stripping only fixes full-width vs half-width digit differences; "
        "it does not reconcile spelled-out numbers/dates against digit forms or vice versa. "
        "Some of the measured CER/WER is this formatting mismatch rather than an actual "
        "mistranscription.\n"
    )

    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
