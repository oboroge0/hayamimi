"""Japanese -> English subtitle translation module.

Uses FuguMT ja-en (staka/fugumt-ja-en), converted to CTranslate2 by the
Mojicast project (ishiki-emo/mojicast-fugumt-ja-en-ct2, CC BY-SA 4.0).
See docs/design/translate.md for model source, license, and measured latency.

Design notes (following Mojicast's TRANSLATION_REPORT.md):
- no_repeat_ngram_size is used to suppress repetition loops (e.g. "Thank
  thank thank..."). Measured n=1 here, not the reference's n=3 -- see
  docs/design/translate.md for why.
- beam_size chosen for a good latency/quality tradeoff (see docs/design/translate.md).
- Any exception, or empty/garbage output, falls back to returning the
  original Japanese text unchanged -- a subtitle line must never go blank.
"""

from __future__ import annotations

import os
import time

import ctranslate2
import sentencepiece as spm

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mojicast-fugumt-ja-en-ct2")

BEAM_SIZE = 5
# NOTE: measured on this model conversion, no_repeat_ngram_size=3 (as used by
# Mojicast's original report) did NOT eliminate repetition loops -- it still
# produced runs like "Thank thank thank thank you...". no_repeat_ngram_size=1
# (block any single-token repeat) was required to meaningfully suppress them.
# See docs/design/translate.md for the measurements behind this choice.
NO_REPEAT_NGRAM_SIZE = 1
MAX_DECODING_LENGTH_CAP = 150
MAX_DECODING_LENGTH_MIN = 30
MAX_DECODING_LENGTH_PER_SOURCE_TOKEN = 6
MAX_DECODING_LENGTH_BASE = 20


class TranslatorJaEn:
    """Loads the FuguMT ja->en CTranslate2 model once and translates lines."""

    def __init__(self, model_dir: str = _MODEL_DIR, device: str = "cpu", compute_type: str = "int8"):
        self.model_dir = model_dir

        source_spm_path = os.path.join(model_dir, "source.spm")
        target_spm_path = os.path.join(model_dir, "target.spm")

        self._sp_source = spm.SentencePieceProcessor(model_file=source_spm_path)
        self._sp_target = spm.SentencePieceProcessor(model_file=target_spm_path)

        self._translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)

    def translate(self, text: str) -> str:
        """Translate a single line of Japanese text to English.

        Returns the original text unchanged (never raises, never blanks
        the line) if input is empty/whitespace-only, or if translation
        fails or produces empty/garbage output.
        """
        if text is None:
            return text

        stripped = text.strip()
        if not stripped:
            return text

        try:
            tokens = self._sp_source.encode(stripped, out_type=str)
            if not tokens:
                return text

            # Cap decode length relative to the source length. Without this,
            # a degenerate hypothesis can run all the way to a large fixed
            # max_decoding_length, turning one bad line into a multi-second
            # stall -- unacceptable for a live subtitle pipeline.
            max_decoding_length = min(
                MAX_DECODING_LENGTH_CAP,
                max(
                    MAX_DECODING_LENGTH_MIN,
                    len(tokens) * MAX_DECODING_LENGTH_PER_SOURCE_TOKEN + MAX_DECODING_LENGTH_BASE,
                ),
            )

            results = self._translator.translate_batch(
                [tokens],
                beam_size=BEAM_SIZE,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                max_decoding_length=max_decoding_length,
            )
            out_tokens = results[0].hypotheses[0]
            if not out_tokens:
                return text

            translated = self._sp_target.decode(out_tokens).strip()
            if not translated:
                return text

            return translated
        except Exception:
            return text


def _smoke_test() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    samples = [
        "本日はお集まりいただきありがとうございます。",  # business
        "今日はマジで疲れたわー。",  # casual
        "会議は午後3時から始まります。",  # business/schedule
        "このプロジェクトの予算は500万円です。",  # numbers
        "明日は雨が降ると思いますか?",  # question
        "先週末、家族と一緒に近くの山に登って、久しぶりに自然の中でゆっくりとした時間を過ごすことができました。",  # long line
        "ありがとうございます、それでは次のスライドに移ります。",  # business
        "This is already in English, so what happens?",  # already English
    ]

    print(f"Loading model from: {_MODEL_DIR}")
    t0 = time.perf_counter()
    translator = TranslatorJaEn()
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s (beam_size={BEAM_SIZE}, no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE})\n")

    latencies = []
    for line in samples:
        t0 = time.perf_counter()
        result = translator.translate(line)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        print(f"[{elapsed_ms:7.1f} ms] {line!r}\n           -> {result!r}\n")

    mean_ms = sum(latencies) / len(latencies)
    print(f"Mean latency: {mean_ms:.1f} ms/line over {len(latencies)} lines")
    print(f"Min: {min(latencies):.1f} ms, Max: {max(latencies):.1f} ms")


if __name__ == "__main__":
    _smoke_test()
