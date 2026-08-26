"""Japanese punctuation restoration for ASR output.

Standalone module: loads a small BERT-char token-classification ONNX model
(from the OSS Mojicast project, see docs/PUNCT_JA.md) and inserts 、(comma)
and 。(period) into raw, unpunctuated Japanese text such as what a streaming
ASR engine (e.g. ReazonSpeech k2 zipformer) typically emits.

Usage:
    from punct_ja import PunctuatorJa
    p = PunctuatorJa()
    p.restore("明日の会議は午後三時から始まります資料の準備をお願いします")
    # -> "明日の会議は午後三時から始まります。資料の準備をお願いします。"

This module is intentionally self-contained and does NOT import or modify
asr_engine.py / realtime_transcribe.py -- integration is done elsewhere.
"""

from __future__ import annotations

import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    import fugashi
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "fugashi is required for the Mojicast punctuation tokenizer. "
        "Install with: pip install fugashi unidic-lite"
    ) from e


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR.parent / "models" / "mojicast-punct-onnx"

# Trailing/leading punctuation that we should not duplicate insertion after.
_JA_PUNCT_CHARS = set("。、！？!?…「」『』（）()【】・,.\n")

# Heuristic suffixes that strongly indicate a question -- the model only
# predicts comma/period (see docs/PUNCT_JA.md), so "?" is added on top via
# this simple rule-based check on the sentence-final mora, not by the model.
_QUESTION_SUFFIXES = (
    "ですか", "ますか", "でしょうか", "かな", "かしら", "かい", "の",
    "だろうか", "でしたか", "ましたか",
)


class PunctuatorJa:
    """Restores Japanese 、/。 punctuation in unpunctuated text.

    Uses a char-level BERT token-classification model (Tohoku BERT-base
    Japanese-char-v3 body + Mojicast punctuation head, ONNX int8), run via
    onnxruntime on CPU. The model predicts, for every input character,
    whether a comma (読点 、) and/or period (句点 。) should follow it.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        comma_threshold: float = 0.5,
        period_threshold: float = 0.5,
        max_chars: int = 500,
        force_final_period: bool = True,
        onnx_filename: str = "punct_bert.onnx",
    ):
        model_dir = Path(model_dir)
        # NOTE: the int8 dynamic-quantized file shipped in the upstream HF
        # repo (punct_bert.int8.onnx) was verified to produce near-constant,
        # token-independent logits on this onnxruntime build (1.29.0) --
        # i.e. it does not actually restore punctuation. The fp32 file
        # (punct_bert.onnx) is used by default. See docs/PUNCT_JA.md for
        # details, and docs/MOBILE.md for a from-scratch INT8 requantization
        # (scripts/quantize_punct.py) that IS verified functional -- pass
        # onnx_filename="quantized_ort/punct_bert.int8.onnx" (or wherever it
        # was written) to load that instead.
        onnx_path = model_dir / onnx_filename
        vocab_path = model_dir / "vocab.txt"
        if not onnx_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                f"Punctuation model files not found under {model_dir}. "
                "Expected punct_bert.int8.onnx and vocab.txt "
                "(see docs/PUNCT_JA.md for download instructions)."
            )

        self.comma_threshold = comma_threshold
        self.period_threshold = period_threshold
        self.max_chars = max_chars
        self.force_final_period = force_final_period

        # --- vocab: standard BERT vocab.txt, line index == token id ---
        self.vocab: dict[str, int] = {}
        with open(vocab_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = idx
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

        # --- mecab word splitter (matches BertJapaneseTokenizer's
        # MecabTokenizer + CharacterTokenizer pipeline: NFKC-normalize,
        # split into mecab morphemes, then split each morpheme into
        # individual characters, mapping OOV chars to [UNK]) ---
        self._tagger = fugashi.Tagger()

        # --- onnxruntime session ---
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    # ------------------------------------------------------------------
    def _tokenize_chars(self, text: str) -> list[str]:
        """Return the list of surface characters as the model tokenizer
        would see them (NFKC-normalized, mecab-segmented then char-split).
        The concatenation of this list equals the NFKC-normalized text."""
        norm = unicodedata.normalize("NFKC", text)
        chars: list[str] = []
        for word in self._tagger(norm):
            chars.extend(list(word.surface))
        return chars

    def _char_id(self, ch: str) -> int:
        return self.vocab.get(ch, self.unk_id)

    # ------------------------------------------------------------------
    def restore(self, text: str) -> str:
        """Insert 、/。 into `text` and return the punctuated string."""
        text = text.strip()
        if not text:
            return text

        chars = self._tokenize_chars(text)
        if not chars:
            return text
        # chunk defensively (model max_position_embeddings=512, CLS+SEP+chars)
        chars = chars[: self.max_chars]

        input_ids = [self.cls_id] + [self._char_id(c) for c in chars] + [self.sep_id]
        attention_mask = [1] * len(input_ids)

        ids_np = np.array([input_ids], dtype=np.int64)
        mask_np = np.array([attention_mask], dtype=np.int64)

        logits = self.session.run(
            ["logits"], {"input_ids": ids_np, "attention_mask": mask_np}
        )[0][0]  # -> (seq_len, 2)

        probs = 1.0 / (1.0 + np.exp(-logits))

        out_parts: list[str] = []
        n = len(chars)
        for i, ch in enumerate(chars):
            out_parts.append(ch)
            # logits[i+1] corresponds to the position right after chars[i]
            # (index 0 is [CLS]).
            comma_p, period_p = probs[i + 1]
            is_last = i == n - 1
            next_ch = chars[i + 1] if not is_last else ""
            if ch in _JA_PUNCT_CHARS:
                continue
            if next_ch in _JA_PUNCT_CHARS:
                # don't double up before existing punctuation
                continue
            if period_p >= self.period_threshold and (
                not is_last or self.force_final_period
            ):
                out_parts.append("。")
            elif comma_p >= self.comma_threshold:
                out_parts.append("、")

        result = "".join(out_parts)
        if self.force_final_period and result and result[-1] not in _JA_PUNCT_CHARS:
            result += "。"

        result = self._apply_question_marks(result)
        return result

    @staticmethod
    def _apply_question_marks(text: str) -> str:
        """Rule-based post-pass: turn a sentence-final '。' into '？' when
        the sentence ends in a common question suffix. The ONNX model only
        classifies comma/period, so this is a simple heuristic layered on
        top (see docs/PUNCT_JA.md, "known limitations")."""
        segments = text.split("。")
        # split() on "a。b。" -> ["a", "b", ""]; a trailing "" means the
        # text ended with "。" (so there's nothing after the last mark).
        out = []
        for seg in segments:
            if not seg:
                continue
            mark = "？" if seg.endswith(_QUESTION_SUFFIXES) else "。"
            out.append(seg + mark)
        return "".join(out)


# ------------------------------------------------------------------------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    samples = [
        "明日の会議は午後三時から始まります資料の準備をお願いします",
        "今日めっちゃ疲れたわもう寝る",
        "これって本当に大丈夫なんですか",
    ]

    print("Loading PunctuatorJa...")
    t0 = time.perf_counter()
    punctuator = PunctuatorJa()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.1f} ms\n")

    for s in samples:
        t0 = time.perf_counter()
        out = punctuator.restore(s)
        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"IN : {s}")
        print(f"OUT: {out}")
        print(f"latency: {dt_ms:.2f} ms\n")
