"""Build the Dart<->Python parity fixture for the mobile punctuation port.

Problem: `mobile/hayamimi_core` re-implements `scripts/punct_ja.py`'s
Japanese punctuation restorer in Dart (see docs/PUNCT_JA.md for the model
itself, and the package README for why the port exists). A
re-implementation is only trustworthy if it is pinned to the reference
implementation on real sentences, not on hand-written examples that both
sides were written to satisfy.

What this script does: it takes Japanese sentences from the FLEURS ja
benchmark set, strips their punctuation to build ASR-like input, and
records for each one (a) the token ids `PunctuatorJa` feeds the model and
(b) the string `PunctuatorJa.restore()` returns with the fp16 model
loaded. The Dart test `test/punct/punct_ja_parity_test.dart` replays the
file and asserts both, character for character.

Why the token ids are recorded separately from the output: the Dart port
has no MeCab and no `unicodedata.normalize`, so its tokenizer is the part
most likely to drift. Recording ids lets the Dart side check tokenization
without loading a 182 MB model, so that half of the parity stays covered
on machines that do not have the model file.

The script also re-verifies the assumption the Dart tokenizer is built on:
that MeCab's only observable effect in this pipeline is dropping
whitespace, i.e. that NFKC-normalising the text and removing whitespace
reproduces `PunctuatorJa._tokenize_chars` exactly.

Usage:
    python scripts/make_punct_fixture.py
    python scripts/make_punct_fixture.py --n 40 --out <path>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from punct_ja import PunctuatorJa  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "mojicast-punct-onnx")
FP16_NAME = os.path.join("quantized_ort", "punct_bert.fp16.onnx")
FLEURS_JA_DIR = os.path.join(ROOT, "testdata", "fleurs_bench", "ja")
DEFAULT_OUT = os.path.join(
    ROOT, "mobile", "hayamimi_core", "test", "fixtures", "punct_ja_parity.json"
)

# Marks removed from the FLEURS reference text to build unpunctuated,
# ASR-like input -- the same three marks scripts/quantize_punct.py scores.
STRIPPED_MARKS = "、。？"

# Cases FLEURS ja does not contain but the port still has to get right:
# the question-mark rule (FLEURS ja is news/wiki prose and holds no
# question sentences at all), the empty / whitespace-only /
# punctuation-only early returns, half-width kana and full-width ASCII
# (which only NFKC folds), and input longer than `max_chars` (the longest
# FLEURS ja reference is 133 characters, so nothing there reaches the
# 500-character cut). The over-length case is built from FLEURS text at
# run time, see main().
EXTRA_CASES = [
    ("empty", ""),
    ("whitespace_only", "   "),
    ("punctuation_only", "。、"),
    ("question_ka", "これって本当に大丈夫なんですか"),
    ("question_then_clause", "もう準備はできましたか今日中に送りますね"),
    ("two_clauses", "明日の会議は午後三時から始まります資料の準備をお願いします"),
    ("casual", "今日めっちゃ疲れたわもう寝る"),
    ("already_punctuated", "こんにちは。元気ですか？"),
    ("halfwidth_kana_fullwidth_ascii", "ｱｼﾀﾉ会議ﾊ１５ジカラデス"),
    ("period_inside_word", "モーニング娘。のコンサートに行きました"),
]


def strip_marks(text: str) -> str:
    return "".join(c for c in text if c not in STRIPPED_MARKS)


def simple_chars(text: str) -> list[str]:
    """The rule the Dart tokenizer implements: NFKC, drop whitespace, then
    split into code points."""
    return [c for c in unicodedata.normalize("NFKC", text) if not c.isspace()]


def load_fleurs_refs() -> list[str]:
    path = os.path.join(FLEURS_JA_DIR, "manifest.json")
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    # The manifest has one row per wav and FLEURS repeats a sentence across
    # speakers, so de-duplicate. sorted() keeps the selection reproducible.
    return sorted({e["ref"] for e in entries})


def verify_tokenizer_assumption(punct: PunctuatorJa, texts: list[str]) -> int:
    """Compare PunctuatorJa._tokenize_chars (NFKC + MeCab + char split)
    against the simpler NFKC + drop-whitespace rule. Returns the mismatch
    count and prints the first few, so a regression is visible instead of
    being silently baked into the fixture."""
    mismatches = 0
    for text in texts:
        got = punct._tokenize_chars(text)  # noqa: SLF001 - reference impl
        want = simple_chars(text)
        if got != want:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH on {text[:40]!r}")
                print("    mecab : " + "".join(got)[:80])
                print("    simple: " + "".join(want)[:80])
    return mismatches


def select_inputs(refs: list[str], n: int) -> list[str]:
    """Pick `n` FLEURS sentences, keeping every one that contains digits or
    Latin letters (those exercise NFKC's full-width folding) and spreading
    the rest evenly over the length range."""
    inputs = [strip_marks(r) for r in refs]
    interesting = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ]")
    by_len = sorted(inputs, key=lambda s: (len(s), s))
    must_keep = [s for s in by_len if interesting.search(s)]
    rest = [s for s in by_len if not interesting.search(s)]
    need = n - len(must_keep)
    if need < 0:
        raise SystemExit(
            f"{len(must_keep)} sentences contain digits/Latin but only {n} "
            "were requested; raise --n"
        )
    if need >= len(rest):
        chosen = must_keep + rest
    else:
        step = (len(rest) - 1) / max(need - 1, 1) if need > 1 else 0
        idx = sorted({round(i * step) for i in range(need)})
        while len(idx) < need:  # rounding collisions
            for j in range(len(rest)):
                if j not in idx:
                    idx.append(j)
                    break
            idx = sorted(set(idx))
        chosen = must_keep + [rest[i] for i in idx[:need]]
    chosen = sorted(set(chosen), key=lambda s: (len(s), s))
    if len(chosen) != n:
        raise SystemExit(f"selected {len(chosen)} sentences, wanted {n}")
    return chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Build the punctuation parity fixture.")
    ap.add_argument("--n", type=int, default=40, help="FLEURS sentences to take")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--onnx",
        default=FP16_NAME,
        help="model file, relative to models/mojicast-punct-onnx",
    )
    args = ap.parse_args()

    print(f"Loading PunctuatorJa with {args.onnx} ...")
    punct = PunctuatorJa(model_dir=MODEL_DIR, onnx_filename=args.onnx)

    refs = load_fleurs_refs()
    print(f"FLEURS ja: {len(refs)} unique reference sentences")

    print("Verifying the NFKC + drop-whitespace tokenizer assumption ...")
    probe = refs + [strip_marks(r) for r in refs]
    probe += [text for _, text in EXTRA_CASES]
    mismatches = verify_tokenizer_assumption(punct, probe)
    print(f"  {len(probe)} texts checked, {mismatches} mismatches")
    if mismatches:
        raise SystemExit(
            "The Dart tokenizer rule no longer matches MeCab; fix the port "
            "before regenerating the fixture."
        )

    inputs = select_inputs(refs, args.n)

    extras = list(EXTRA_CASES)
    long_input = ""
    for s in inputs:
        long_input += s
        if len(long_input) > punct.max_chars + 60:
            break
    extras.append(("over_max_chars", long_input))

    cases = []
    for i, text in enumerate(inputs):
        cases.append({"name": f"fleurs_{i:02d}", "source": "fleurs", "input": text})
    for name, text in extras:
        cases.append({"name": name, "source": "synthetic", "input": text})

    for case in cases:
        text = case["input"]
        chars = punct._tokenize_chars(text.strip())  # noqa: SLF001
        chars = chars[: punct.max_chars]
        case["input_ids"] = (
            [punct.cls_id]
            + [punct._char_id(c) for c in chars]  # noqa: SLF001
            + [punct.sep_id]
        )
        case["expected"] = punct.restore(text)

    payload = {
        "_generated_by": "scripts/make_punct_fixture.py",
        "_purpose": (
            "Python-vs-Dart parity for the Japanese punctuation restorer. "
            "'input_ids' is the full model input ([CLS] + characters + "
            "[SEP]) that scripts/punct_ja.py builds; 'expected' is what its "
            "restore() returns with the model named in _model."
        ),
        "_source": (
            "Sentences named fleurs_* come from the FLEURS corpus "
            "(google/fleurs, ja_jp), Copyright Google, licensed CC BY 4.0 "
            "(https://creativecommons.org/licenses/by/4.0/), via "
            "testdata/fleurs_bench/ja/manifest.json, with the marks "
            "、。？ removed to make unpunctuated, ASR-like "
            "input. Cases named otherwise are synthetic, written for this "
            "fixture."
        ),
        "_model": "models/mojicast-punct-onnx/"
        + args.onnx.replace(os.sep, "/").replace("\\", "/"),
        "_onnxruntime": __import__("onnxruntime").__version__,
        "_unicodedata": unicodedata.unidata_version,
        "_thresholds": {
            "comma": punct.comma_threshold,
            "period": punct.period_threshold,
            "max_chars": punct.max_chars,
            "force_final_period": punct.force_final_period,
        },
        "cases": cases,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # One line per case: small enough to diff, compact enough to stay well
    # under the size a test fixture should be.
    head = {k: v for k, v in payload.items() if k != "cases"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},")
    lines.append('  "cases": [')
    for i, case in enumerate(cases):
        tail = "," if i < len(cases) - 1 else ""
        lines.append(
            "    " + json.dumps(case, ensure_ascii=False, separators=(",", ":")) + tail
        )
    lines.append("  ]")
    lines.append("}")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    size = os.path.getsize(args.out)
    print(f"\nwrote {args.out} ({len(cases)} cases, {size / 1024:.1f} KB)")
    for case in cases[:3] + cases[-5:]:
        print(f"  {case['name']:<32} {case['input'][:26]}  ->  {case['expected'][:32]}")


if __name__ == "__main__":
    main()
