"""Shared Japanese text normalisation for the punctuation pipeline.

unicodedata.normalize("NFKC") folds the fullwidth marks ？ and ！ to ASCII
? and !, which silently defeats every fullwidth-only mark check downstream
(training labels, F1 scoring, calibration text). This module holds the one
NFKC that protects them, so the trainer, the corpus builder, the 4-class eval
and quantize_punct.py all normalise text the same way. Dependency-free.
"""
import unicodedata

# private-use-area placeholders that survive NFKC untouched
_Q_SENTINEL = ""
_E_SENTINEL = ""

def safe_nfkc(text: str) -> str:
    """See module docstring / docs/eval/punct_retrain.md -- plain NFKC
    folds fullwidth "？"/"！" to ASCII "?"/"!", which breaks fullwidth-only
    mark-membership checks downstream. Protect them with PUA sentinels."""
    text = text.replace("？", _Q_SENTINEL).replace("！", _E_SENTINEL)
    text = unicodedata.normalize("NFKC", text)
    return text.replace(_Q_SENTINEL, "？").replace(_E_SENTINEL, "！")


# backwards-compatible alias for callers that imported the private name
_safe_nfkc = safe_nfkc
