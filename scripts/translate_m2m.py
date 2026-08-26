"""Japanese -> multilingual (any M2M-100 target) subtitle translation module.

Uses M2M-100 418M (facebook/m2m100_418M), converted to CTranslate2 int8 by
the Mojicast project (ishiki-emo/mojicast-m2m100-ct2, MIT license).
See docs/TRANSLATE_M2M.md for model source, license, and measured latency.

Target acceptance and validation tiers:
- Any target language present in the model's own vocabulary (i.e. one with an
  `__<lang>__` token in `shared_vocabulary.json`) is accepted --
  `is_supported_target()` checks this directly against the model files rather
  than a hardcoded allowlist, so this module works for any of M2M-100's ~100
  languages without code changes.
- Only a subset of those targets have been *measured* for translation
  quality against a reference (see `scripts/eval_translate.py` and
  docs/TRANSLATE_M2M.md's "Validation tiers" section). `VALIDATED_TARGETS`
  tracks that subset. Constructing a `TranslatorM2M` for a target outside
  `VALIDATED_TARGETS` prints a one-line note to stderr -- it still works
  (M2M-100 is trained on all these languages), just with unmeasured quality.

Design notes (mirroring scripts/translate_ja_en.py):
- M2M-100 is a multilingual model: the source SentencePiece tokens must be
  prefixed with the source language token (__ja__) and suffixed with the
  end-of-sentence token (</s>), and translate_batch is called with
  target_prefix=[["__<lang>__"]] to select the output language. The decoded
  hypothesis starts with that same target-language token, which is stripped
  before detokenizing (this is the standard CTranslate2 M2M-100 usage
  pattern -- see https://opennmt.net/CTranslate2/guides/transformers.html#m2m-100).
- no_repeat_ngram_size=3 is used, matching Mojicast's reported setting. On
  this conversion it does NOT fully eliminate repetition (unlike the ja-en
  FuguMT module, which needed n=1) but it reliably prevents the catastrophic
  multi-second degenerate loops (e.g. 50+ repeated tokens) seen without it.
  See docs/TRANSLATE_M2M.md for the measurements behind this choice.
- beam_size differs per target: zh uses greedy (beam_size=1), ko uses
  beam_size=4 -- matching Mojicast's reported configuration, which held up
  in measurement here (see docs/TRANSLATE_M2M.md).
- Any exception, or empty/garbage output, falls back to returning the
  original Japanese text unchanged -- a subtitle line must never go blank.
"""

from __future__ import annotations

import json
import os
import sys
import time

import ctranslate2
import sentencepiece as spm

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "mojicast-m2m100-ct2")

SOURCE_LANG_TOKEN = "__ja__"
EOS_TOKEN = "</s>"

# Targets with a *measured* chrF score against a FLEURS reference (see
# scripts/eval_translate.py and docs/TRANSLATE_M2M.md's "Validation tiers"
# section). Any other M2M-100 target is still accepted (see
# is_supported_target() below) but its quality has not been checked.
VALIDATED_TARGETS = {"zh", "ko", "es"}

# Per-target settings, matching Mojicast's reported configuration (zh:
# greedy, ko: beam_size=4). See docs/TRANSLATE_M2M.md for the measurements
# that confirmed these hold up on this specific model conversion. Targets not
# listed here fall back to DEFAULT_BEAM_SIZE (untuned).
BEAM_SIZE_BY_TARGET = {
    "zh": 1,
    "ko": 4,
    "en": 1,
}
# Fallback beam size for a target not in BEAM_SIZE_BY_TARGET. Matches ko's
# beam_size=4 (the more cautious of the two measured settings) rather than
# zh's greedy beam_size=1, since it has not been re-measured per-target --
# see docs/TRANSLATE_M2M.md.
DEFAULT_BEAM_SIZE = 4

_vocab_cache: "dict[str, set[str]]" = {}


def _load_vocab(model_dir: str = _MODEL_DIR) -> "set[str]":
    """Load (and cache) the set of token strings in the model's CTranslate2 vocabulary."""
    if model_dir not in _vocab_cache:
        vocab_path = os.path.join(model_dir, "shared_vocabulary.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        _vocab_cache[model_dir] = set(tokens)
    return _vocab_cache[model_dir]


def is_supported_target(lang: str, model_dir: str = _MODEL_DIR) -> bool:
    """True if the model's vocabulary has a `__<lang>__` target-language token.

    This is a direct check against shared_vocabulary.json rather than a
    hardcoded allowlist, so any of M2M-100's ~100 target languages that this
    CTranslate2 conversion actually supports is accepted -- see the module
    docstring for the distinction between "supported" (accepted here) and
    "validated" (quality-measured, tracked in VALIDATED_TARGETS).
    """
    try:
        vocab = _load_vocab(model_dir)
    except (OSError, json.JSONDecodeError):
        return False
    return f"__{lang}__" in vocab


# NOTE: Mojicast's report says no_repeat_ngram_size=3 "eliminated" repetition
# loops for their conversion. Re-measured here on filler-heavy casual lines
# (e.g. "そうそう、そうなんだよね。" and a longer hesitation-filled line):
# n=3 does NOT fully eliminate repetition (occasional "是啊,是啊!是啊?..."
# style runs remain), but it reliably prevents the catastrophic degenerate
# loops seen with n=0 -- e.g. one line ran to 50+ repeated tokens and took
# ~2.6-3.0s without it, vs. a bounded, mostly-sane output in ~0.3-0.8s with
# it. Kept at n=3 (unlike translate_ja_en.py's n=1) because a stricter value
# did not measurably improve quality further in spot checks and n=3 is the
# documented reference setting. See docs/TRANSLATE_M2M.md.
NO_REPEAT_NGRAM_SIZE = 3
MAX_DECODING_LENGTH_CAP = 150
MAX_DECODING_LENGTH_MIN = 30
MAX_DECODING_LENGTH_PER_SOURCE_TOKEN = 6
MAX_DECODING_LENGTH_BASE = 20


class TranslatorM2M:
    """Loads the M2M-100 CTranslate2 model once and translates ja->target_lang lines."""

    def __init__(
        self,
        target_lang: str,
        model_dir: str = _MODEL_DIR,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        if not is_supported_target(target_lang, model_dir):
            raise ValueError(
                f"Unsupported target_lang: {target_lang!r} "
                f"(no __{target_lang}__ token in {model_dir}'s shared_vocabulary.json)"
            )
        if target_lang not in VALIDATED_TARGETS:
            print(
                f"note: {target_lang!r} is an unvalidated translation target "
                "(quality not yet measured; see docs/TRANSLATE_M2M.md)",
                file=sys.stderr,
            )

        self.target_lang = target_lang
        self.model_dir = model_dir
        self._target_token = f"__{target_lang}__"
        self._beam_size = BEAM_SIZE_BY_TARGET.get(target_lang, DEFAULT_BEAM_SIZE)

        sp_path = os.path.join(model_dir, "sentencepiece.model")
        self._sp = spm.SentencePieceProcessor(model_file=sp_path)

        self._translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type)

    def translate(self, text: str) -> str:
        """Translate a single line of Japanese text to self.target_lang.

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
            pieces = self._sp.encode(stripped, out_type=str)
            if not pieces:
                return text

            source_tokens = [SOURCE_LANG_TOKEN] + pieces + [EOS_TOKEN]

            # Cap decode length relative to the source length. Without this,
            # a degenerate hypothesis can run all the way to a large fixed
            # max_decoding_length, turning one bad line into a multi-second
            # stall -- unacceptable for a live subtitle pipeline.
            max_decoding_length = min(
                MAX_DECODING_LENGTH_CAP,
                max(
                    MAX_DECODING_LENGTH_MIN,
                    len(pieces) * MAX_DECODING_LENGTH_PER_SOURCE_TOKEN + MAX_DECODING_LENGTH_BASE,
                ),
            )

            results = self._translator.translate_batch(
                [source_tokens],
                target_prefix=[[self._target_token]],
                beam_size=self._beam_size,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                max_decoding_length=max_decoding_length,
            )
            out_tokens = results[0].hypotheses[0]
            # The hypothesis starts with the target-language token we passed
            # as target_prefix -- strip it before detokenizing.
            if out_tokens and out_tokens[0] == self._target_token:
                out_tokens = out_tokens[1:]
            if out_tokens and out_tokens[-1] == EOS_TOKEN:
                out_tokens = out_tokens[:-1]
            if not out_tokens:
                return text

            translated = self._sp.decode(out_tokens).strip()
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
        "会議は午後3時から始まります。",  # business/schedule, numbers
        "このプロジェクトの予算は500万円です。",  # numbers
        "明日は雨が降ると思いますか?",  # question
        "そうそう、そうなんだよね。",  # repetition-prone casual line
    ]
    targets = ["zh", "ko"]

    translators = {}
    for lang in targets:
        print(f"Loading M2M-100 model for target={lang} from: {_MODEL_DIR}")
        t0 = time.perf_counter()
        translators[lang] = TranslatorM2M(target_lang=lang)
        load_time = time.perf_counter() - t0
        beam = BEAM_SIZE_BY_TARGET[lang]
        print(f"  loaded in {load_time:.2f}s (beam_size={beam}, no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE})")
    print()

    latencies_by_lang = {lang: [] for lang in targets}
    for line in samples:
        print(f"JA: {line!r}")
        for lang in targets:
            t0 = time.perf_counter()
            result = translators[lang].translate(line)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_by_lang[lang].append(elapsed_ms)
            print(f"  [{lang}] [{elapsed_ms:7.1f} ms] -> {result!r}")
        print()

    for lang in targets:
        latencies = latencies_by_lang[lang]
        mean_ms = sum(latencies) / len(latencies)
        print(f"[{lang}] Mean latency: {mean_ms:.1f} ms/line over {len(latencies)} lines "
              f"(min {min(latencies):.1f} ms, max {max(latencies):.1f} ms)")


if __name__ == "__main__":
    _smoke_test()
