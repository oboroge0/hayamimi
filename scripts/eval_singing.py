"""Singing-voice scorecard: how much does the engine degrade on sung audio?

Runs the production RoutedASR path over testdata/eval_singing/ (sung clips:
ja from PJS, ko/en from CSD) and, for ja, also over the paired *spoken*
recordings of the same lyrics (testdata/eval_singing_speech/, PJS speech
twins) -- a controlled sung-vs-spoken comparison with singer and text fixed.

Scoring: en=WER, ko=CER as usual. ja refs are hiragana lyric syllables from
PJS musicxml, so ja is scored in *kana space*: both reference and hypothesis
are converted to hiragana with pykakasi before CER. The kana conversion adds
some noise on the hypothesis side (kanji reading errors), but it applies
equally to the sung and spoken sides, so the sung-vs-spoken delta stays fair.

Usage:
    python scripts/eval_singing.py [--root H:\\path\\to\\hayamimi]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from asr_engine import RoutedASR
from eval_accuracy import cer_ja, wer_en, normalize_ja, levenshtein
from eval_common import load_manifest

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_kakasi = None


def to_hira(text: str) -> str:
    global _kakasi
    if _kakasi is None:
        import pykakasi

        _kakasi = pykakasi.kakasi()
    return "".join(item["hira"] for item in _kakasi.convert(text))


def cer_kana(ref: str, hyp: str) -> float:
    r = to_hira(normalize_ja(ref))
    h = to_hira(normalize_ja(hyp))
    if not r:
        return 0.0
    return levenshtein(r, h) / len(r)


def score(lang: str, ref: str, hyp: str) -> float:
    if lang == "en":
        return wer_en(ref, hyp)
    if lang == "ja":
        return cer_kana(ref, hyp)
    return cer_ja(ref, hyp)[0]


def run_manifest(asr, mdir):
    rows = []
    for e in load_manifest(mdir):
        # Each clip is an independent recording; don't let sticky-LID
        # hysteresis carry language state across clips.
        if hasattr(asr, "reset_session"):
            asr.reset_session()
        samples, sr = sf.read(os.path.join(mdir, e["wav"]), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        dur = len(samples) / sr
        t0 = time.perf_counter()
        r = asr.transcribe(samples, sr)
        wall = time.perf_counter() - t0
        rows.append({
            "wav": e["wav"], "lang": e["lang"], "detected": r["lang"],
            "tier": r["tier"], "err": score(e["lang"], e["ref"], r["text"]),
            "rtf": wall / dur, "dur": dur, "hyp": r["text"], "ref": e["ref"],
        })
        print(f"{os.path.basename(mdir):24} {e['wav']:12} true={e['lang']:3} "
              f"lid={r['lang']:3} tier={r['tier']:4} err={rows[-1]['err']:.3f} "
              f"rtf={rows[-1]['rtf']:.3f}")
    return rows


def summarize(rows, lang):
    sub = [r for r in rows if r["lang"] == lang]
    if not sub:
        return None
    lid_ok = sum(1 for r in sub if r["detected"] == r["lang"])
    tiers = sorted({r["tier"] for r in sub},
                   key=lambda t: -sum(1 for r in sub if r["tier"] == t))
    err = sum(r["err"] * r["dur"] for r in sub) / sum(r["dur"] for r in sub)
    rtf = sum(r["rtf"] for r in sub) / len(sub)
    return {"clips": len(sub), "lid": f"{lid_ok}/{len(sub)}",
            "tiers": "+".join(tiers), "err": err, "rtf": rtf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    sing_dir = os.path.join(args.root, "testdata", "eval_singing")
    speech_dir = os.path.join(args.root, "testdata", "eval_singing_speech")
    docs_path = os.path.join(DEFAULT_ROOT, "docs", "SINGING.md")

    asr = RoutedASR(threads=6, preload=False)

    print("== singing clips ==")
    sung = run_manifest(asr, sing_dir)
    print("\n== spoken twins (ja/PJS) ==")
    spoken = run_manifest(asr, speech_dir) if os.path.isdir(speech_dir) else []

    lines = [
        "# Singing-Voice Scorecard\n",
        "本番経路 (LID→ルーティング→デコード) を歌唱音声で採点。",
        "ja=PJS(アカペラ自作曲、かな空間CER・pykakasi正規化)、ko/en=CSD(童謡アカペラ、CER/WER)。",
        "ja は同一歌唱者・同一歌詞の朗読版 (PJS speech) との対照比較付き。\n",
        "## 歌唱 vs 話し言葉ベースライン\n",
        "| lang | clips | LID正解 | 主tier | mean err (歌唱) | mean err (朗読/話し言葉基準) | mean RTF |",
        "|---|---|---|---|---|---|---|",
    ]

    # ja speech twins measured here; en/ko/zh read-aloud baseline comes from
    # docs/SCORECARD.md (eval_real), quoted for context.
    baseline = {"ja": None, "en": 0.023, "ko": 0.081}
    ja_spoken = summarize(spoken, "ja")
    if ja_spoken:
        baseline["ja"] = ja_spoken["err"]

    for lang in ("ja", "en", "ko"):
        s = summarize(sung, lang)
        if not s:
            continue
        base = baseline.get(lang)
        base_s = f"{base:.3f}" + (" (朗読ペア実測)" if lang == "ja" else " (SCORECARD)") \
            if base is not None else "-"
        lines.append(f"| {lang} | {s['clips']} | {s['lid']} | {s['tiers']} | "
                     f"{s['err']:.3f} | {base_s} | {s['rtf']:.3f} |")

    lines.append("\n## ja: 歌唱 vs 朗読（同一人物・同一歌詞、かな空間CER）\n")
    if spoken:
        lines.append("| wav | err 朗読 | err 歌唱 | Δ | 歌唱LID | 歌唱tier |")
        lines.append("|---|---|---|---|---|---|")
        spoken_by = {r["wav"]: r for r in spoken}
        for r in [x for x in sung if x["lang"] == "ja"]:
            sp = spoken_by.get(r["wav"])
            if not sp:
                continue
            lines.append(f"| {r['wav']} | {sp['err']:.3f} | {r['err']:.3f} | "
                         f"{r['err'] - sp['err']:+.3f} | {r['detected']} | {r['tier']} |")

    mis = [r for r in sung if r["detected"] != r["lang"]]
    lines.append("\n## 歌唱でのLID誤判定\n")
    if mis:
        lines.append("| wav | true | detected | tier | err |")
        lines.append("|---|---|---|---|---|")
        for r in mis:
            lines.append(f"| {r['wav']} | {r['lang']} | {r['detected']} | {r['tier']} | {r['err']:.3f} |")
    else:
        lines.append("誤判定なし。")

    with open(docs_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {docs_path}")


if __name__ == "__main__":
    main()
