"""End-to-end engine scorecard.

Runs the production RoutedASR path (whisper-tiny LID -> tier routing ->
decode -> ja punctuation) over every real-speech manifest and reports, per
language: LID accuracy, tier distribution, CER/WER, and RTF.

Unlike eval_accuracy.py (which scores models in isolation with the language
forced), this measures what the engine actually does, misroutes included.

Usage:
    python scripts/eval_engine.py
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from asr_engine import RoutedASR
from eval_accuracy import cer_ja, wer_en
from eval_common import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(ROOT, "docs", "SCORECARD.md")

MANIFESTS = [
    ("testdata/eval_real", None),        # ja + en
    ("testdata/eval_real_zhko", None),   # zh + ko
    ("testdata/eval_real_yue", None),    # yue
]

try:
    from opencc import OpenCC

    _t2s = OpenCC("t2s").convert
except Exception:
    _t2s = None


def score(lang: str, ref: str, hyp: str) -> float:
    if lang == "en":
        return wer_en(ref, hyp)
    if lang == "yue" and _t2s is not None:
        ref, hyp = _t2s(ref), _t2s(hyp)
    return cer_ja(ref, hyp)[0]


def main():
    asr = RoutedASR(threads=6, preload=False)

    rows = []
    for rel, _ in MANIFESTS:
        mdir = os.path.join(ROOT, rel)
        # Each manifest is an independent recording set, not a continuation
        # of the previous one's session -- reset the sticky-LID state so the
        # engine's language-switch hysteresis (RoutedASR.transcribe /
        # resolve_dual_confirm / resolve_sticky_lang in asr_engine.py)
        # doesn't charge the first clip of eval_real_zhko or eval_real_yue
        # for "switching away" from whatever language the previous
        # manifest's last clip happened to end on. Same convention as
        # eval_noise.py's per-language-block reset.
        asr.reset_session()
        for e in load_manifest(mdir):
            samples, sr = sf.read(os.path.join(mdir, e["wav"]), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            dur = len(samples) / sr
            t0 = time.perf_counter()
            r = asr.transcribe(samples, sr)  # LID included: the real path
            wall = time.perf_counter() - t0
            rows.append({
                "wav": e["wav"], "lang": e["lang"], "detected": r["lang"],
                "tier": r["tier"], "err": score(e["lang"], e["ref"], r["text"]),
                "rtf": wall / dur, "dur": dur,
            })
            print(f"{e['wav']:16} true={e['lang']:3} lid={r['lang']:3} tier={r['tier']:4} "
                  f"err={rows[-1]['err']:.3f} rtf={rows[-1]['rtf']:.3f}")

    lines = ["# Engine Scorecard (end-to-end, real speech)\n",
             "本番経路 (LID→ルーティング→デコード→ja句読点) のエンドツーエンド採点。",
             "単発クリップのためプリロール・二段パスは含まない。metricは en=WER, 他=CER"
             + ("（yueはt2s正規化）" if _t2s else "（yueは生CER: opencc無し）") + "。\n",
             "| lang | clips | LID正解 | 主tier | mean err | mean RTF |", "|---|---|---|---|---|---|"]
    print()
    for lang in ("ja", "en", "zh", "ko", "yue"):
        sub = [r for r in rows if r["lang"] == lang]
        if not sub:
            continue
        lid_ok = sum(1 for r in sub if r["detected"] == r["lang"])
        tiers = sorted({r["tier"] for r in sub},
                       key=lambda t: -sum(1 for r in sub if r["tier"] == t))
        err = sum(r["err"] * r["dur"] for r in sub) / sum(r["dur"] for r in sub)
        rtf = sum(r["rtf"] for r in sub) / len(sub)
        line = f"| {lang} | {len(sub)} | {lid_ok}/{len(sub)} | {'+'.join(tiers)} | {err:.3f} | {rtf:.3f} |"
        lines.append(line)
        print(line)

    mis = [r for r in rows if r["detected"] != r["lang"]]
    lines.append("\n## LID誤判定の内訳\n")
    if mis:
        lines.append("| wav | true | detected | tier | err |")
        lines.append("|---|---|---|---|---|")
        for r in mis:
            lines.append(f"| {r['wav']} | {r['lang']} | {r['detected']} | {r['tier']} | {r['err']:.3f} |")
    else:
        lines.append("誤判定なし。")

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {DOCS_PATH}")


if __name__ == "__main__":
    main()
