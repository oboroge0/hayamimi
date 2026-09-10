"""Candidate translation backends for the Track B evaluation
(docs/eval/translate_candidates.md): replacement candidates for FuguMT
(ja->en, CC BY-SA 4.0) and/or M2M-100 418M (ja->zh/ko/es, MIT).

This module does NOT change scripts/translate_ja_en.py's or
scripts/translate_m2m.py's default behavior -- it only adds new,
separately-instantiated translator classes/factories used by
scripts/eval_translate.py's --backend flag. Nothing here is wired into
realtime_transcribe.py.

Backends (see docs/eval/translate_candidates.md for full results/decision):
  - "fugumt"        existing TranslatorJaEn (ja->en only) -- baseline
  - "m2m"           existing TranslatorM2M, default model dir -- baseline
  - "m2m100-1.2b"   facebook/m2m100_1.2B, converted to CTranslate2 int8
                     (models/m2m100-1.2B-ct2), reusing TranslatorM2M's code
                     via model_dir override (same tokenizer/target-token
                     scheme as the 418M model)
  - "opus-mt"       Helsinki-NLP/opus-mt-ja-<X> (Apache-2.0), one dedicated
                     bilingual model per target -- only "en" and "es" exist
                     upstream (no official ja->zh or ja->ko opus-mt model)
  - "lmt60"         NiuTrans/LMT-60-0.6B (Apache-2.0, 2026), a Qwen3-based
                     decoder-only MT model prompted for translation and
                     converted to a CTranslate2 Generator (int8)

Conversion commands (from the worktree root, using .venv-train which has
torch/transformers -- see docs/eval/translate_candidates.md):

    python -m ctranslate2.converters.transformers \\
        --model Helsinki-NLP/opus-mt-ja-en --quantization int8 \\
        --copy_files source.spm target.spm vocab.json \\
        --output_dir models/opus-mt-ja-en-ct2

    python -m ctranslate2.converters.transformers \\
        --model Helsinki-NLP/opus-mt-ja-es --quantization int8 \\
        --copy_files source.spm target.spm vocab.json \\
        --output_dir models/opus-mt-ja-es-ct2

    python -m ctranslate2.converters.transformers \\
        --model facebook/m2m100_1.2B --quantization int8 \\
        --copy_files sentencepiece.bpe.model --low_cpu_mem_usage \\
        --output_dir models/m2m100-1.2B-ct2

    python -m ctranslate2.converters.transformers \\
        --model NiuTrans/LMT-60-0.6B --quantization int8 --low_cpu_mem_usage \\
        --copy_files tokenizer.json tokenizer_config.json vocab.json \\
            merges.txt special_tokens_map.json chat_template.jinja \\
            added_tokens.json generation_config.json \\
        --output_dir models/lmt60-0.6b-ct2
"""

from __future__ import annotations

import os

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

M2M100_1_2B_DIR = os.path.join(_MODELS_DIR, "m2m100-1.2B-ct2")
OPUS_MT_JA_EN_DIR = os.path.join(_MODELS_DIR, "opus-mt-ja-en-ct2")
OPUS_MT_JA_ES_DIR = os.path.join(_MODELS_DIR, "opus-mt-ja-es-ct2")
LMT60_DIR = os.path.join(_MODELS_DIR, "lmt60-0.6b-ct2")

# opus-mt is one bilingual model per pair; Helsinki-NLP only publishes an
# official ja->en and ja->es direction (no ja->zh / ja->ko model exists
# upstream -- see docs/eval/translate_candidates.md for the search that
# confirmed this).
OPUS_MT_DIR_BY_TARGET = {
    "en": OPUS_MT_JA_EN_DIR,
    "es": OPUS_MT_JA_ES_DIR,
}

LANG_NAMES = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "es": "Spanish",
}

# Same decode-length-cap philosophy as translate_ja_en.py / translate_m2m.py:
# bound the worst case rather than let a degenerate hypothesis run to a large
# fixed default -- a multi-second stall is unacceptable for a live subtitle
# pipeline even during an offline eval run.
_MT_MAX_DECODING_LENGTH_CAP = 150
_MT_MAX_DECODING_LENGTH_MIN = 30
_MT_MAX_DECODING_LENGTH_PER_SOURCE_TOKEN = 6
_MT_MAX_DECODING_LENGTH_BASE = 20

# LMT-60 is prompted (source text + instructions), so its useful "signal" per
# generated token is lower and headroom is a bit more generous than the pure
# seq2seq MT modules above; still capped, never left at ctranslate2's raw
# default.
_LLM_MAX_DECODING_LENGTH_CAP = 200
_LLM_MAX_DECODING_LENGTH_MIN = 40
_LLM_MAX_DECODING_LENGTH_PER_SOURCE_TOKEN = 8
_LLM_MAX_DECODING_LENGTH_BASE = 30


def _capped_max_length(n_source_tokens: int, cap: int, min_len: int, per_token: int, base: int) -> int:
    return min(cap, max(min_len, n_source_tokens * per_token + base))


class TranslatorMarianPair:
    """Generic single-language-pair Marian/OPUS-MT CTranslate2 wrapper (ja->X).

    Same tokenization scheme and never-blank fallback behavior as
    scripts/translate_ja_en.py's TranslatorJaEn (which is itself a Marian
    model, FuguMT) -- this class is parameterized so it can point at any
    converted opus-mt-ja-<X> directory instead of being hardcoded to one
    model/target pair.
    """

    def __init__(
        self,
        model_dir: str,
        beam_size: int = 5,
        no_repeat_ngram_size: int = 3,
        device: str = "cpu",
        compute_type: str = "int8",
        intra_threads: int = 0,
    ):
        import ctranslate2
        import sentencepiece as spm

        self.model_dir = model_dir
        self._beam_size = beam_size
        self._no_repeat_ngram_size = no_repeat_ngram_size
        self._sp_source = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "source.spm"))
        self._sp_target = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "target.spm"))
        self._translator = ctranslate2.Translator(
            model_dir, device=device, compute_type=compute_type, intra_threads=intra_threads
        )

    def translate(self, text: str) -> str:
        if text is None:
            return text
        stripped = text.strip()
        if not stripped:
            return text
        try:
            tokens = self._sp_source.encode(stripped, out_type=str)
            if not tokens:
                return text
            max_decoding_length = _capped_max_length(
                len(tokens),
                _MT_MAX_DECODING_LENGTH_CAP,
                _MT_MAX_DECODING_LENGTH_MIN,
                _MT_MAX_DECODING_LENGTH_PER_SOURCE_TOKEN,
                _MT_MAX_DECODING_LENGTH_BASE,
            )
            results = self._translator.translate_batch(
                [tokens],
                beam_size=self._beam_size,
                no_repeat_ngram_size=self._no_repeat_ngram_size,
                max_decoding_length=max_decoding_length,
            )
            out_tokens = results[0].hypotheses[0]
            if not out_tokens:
                return text
            translated = self._sp_target.decode(out_tokens).strip()
            return translated or text
        except Exception:
            return text


class TranslatorLMT60:
    """NiuTrans/LMT-60-0.6B (Qwen3-based decoder-only MT model, Apache-2.0),
    converted to a CTranslate2 Generator (int8). See
    docs/eval/translate_candidates.md for the full writeup.

    Unlike the encoder-decoder candidates above, this is a decoder-only LLM
    prompted with the model card's documented translation template
    ("Translate the following text from <SRC> into <TGT>:\\n<SRC>: <text>\\n<TGT>:"
    wrapped in the model's own ChatML-style turns) and generated with
    ``<think>\\n\\n</think>\\n\\n`` forced immediately after the assistant turn
    to disable Qwen3's default chain-of-thought mode -- without this, the
    model prepends a reasoning block before the actual translation, which
    only adds latency here (this is what the model's own chat template does
    when ``enable_thinking=False`` is set; reproduced by hand since this
    module avoids a `transformers`/Jinja dependency in the runtime venv).
    """

    def __init__(
        self,
        target_lang: str,
        model_dir: str = LMT60_DIR,
        source_lang: str = "ja",
        beam_size: int = 1,
        device: str = "cpu",
        compute_type: str = "int8",
        intra_threads: int = 0,
    ):
        import ctranslate2
        from tokenizers import Tokenizer

        if target_lang not in LANG_NAMES or source_lang not in LANG_NAMES:
            raise ValueError(
                f"Unknown language code (source={source_lang!r}, target={target_lang!r}); "
                f"add it to LANG_NAMES in scripts/translate_candidates.py."
            )

        self.model_dir = model_dir
        self.target_lang = target_lang
        self.source_lang = source_lang
        self._beam_size = beam_size
        self._src_name = LANG_NAMES[source_lang]
        self._tgt_name = LANG_NAMES[target_lang]

        self._tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._generator = ctranslate2.Generator(
            model_dir, device=device, compute_type=compute_type, intra_threads=intra_threads
        )

    def _build_prompt(self, text: str) -> str:
        user = (
            f"Translate the following text from {self._src_name} into {self._tgt_name}:\n"
            f"{self._src_name}: {text}\n{self._tgt_name}:"
        )
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def translate(self, text: str) -> str:
        if text is None:
            return text
        stripped = text.strip()
        if not stripped:
            return text
        try:
            src_tokens = self._tok.encode(stripped, add_special_tokens=False).tokens
            n_src = len(src_tokens) if src_tokens else 1

            prompt = self._build_prompt(stripped)
            prompt_tokens = self._tok.encode(prompt, add_special_tokens=False).tokens
            if not prompt_tokens:
                return text

            max_length = _capped_max_length(
                n_src,
                _LLM_MAX_DECODING_LENGTH_CAP,
                _LLM_MAX_DECODING_LENGTH_MIN,
                _LLM_MAX_DECODING_LENGTH_PER_SOURCE_TOKEN,
                _LLM_MAX_DECODING_LENGTH_BASE,
            )
            results = self._generator.generate_batch(
                [prompt_tokens],
                max_length=max_length,
                beam_size=self._beam_size,
                end_token=["<|im_end|>", "<|endoftext|>"],
                include_prompt_in_result=False,
            )
            out_ids = results[0].sequences_ids[0]
            if not out_ids:
                return text
            translated = self._tok.decode(out_ids, skip_special_tokens=True).strip()
            return translated or text
        except Exception:
            return text


def make_translator(backend: str, target_lang: str, intra_threads: int = 2):
    """Factory used by scripts/eval_translate.py's --backend flag.

    Raises ValueError for a backend/target_lang combination that doesn't
    exist (e.g. opus-mt has no ja->zh model) rather than silently falling
    back to something else.
    """
    if backend == "fugumt":
        if target_lang != "en":
            raise ValueError("backend='fugumt' only supports target_lang='en'")
        from translate_ja_en import TranslatorJaEn

        return TranslatorJaEn(intra_threads=intra_threads)

    if backend == "m2m":
        from translate_m2m import TranslatorM2M

        return TranslatorM2M(target_lang, intra_threads=intra_threads)

    if backend == "m2m100-1.2b":
        from translate_m2m import TranslatorM2M

        return TranslatorM2M(target_lang, model_dir=M2M100_1_2B_DIR, intra_threads=intra_threads)

    if backend == "opus-mt":
        if target_lang not in OPUS_MT_DIR_BY_TARGET:
            raise ValueError(
                f"backend='opus-mt' has no ja->{target_lang} model "
                f"(only {sorted(OPUS_MT_DIR_BY_TARGET)} exist upstream from Helsinki-NLP)"
            )
        return TranslatorMarianPair(OPUS_MT_DIR_BY_TARGET[target_lang], intra_threads=intra_threads)

    if backend == "lmt60":
        return TranslatorLMT60(target_lang, intra_threads=intra_threads)

    raise ValueError(f"Unknown backend: {backend!r}")
