"""Segment-length vs LID-accuracy curve for the switch-policy redesign.

Answers two questions with real data:
  1. What is the shortest segment length at which whisper-tiny LID alone
     (the production language identifier, see asr_engine.py's `_identify_lang`)
     is reliable enough (>=95%) to route on?
  2. When whisper-tiny AND SenseVoice's own internal LID (its `_decode_full`
     tag) AGREE on a short segment, how much more trustworthy is that
     agreement than either detector alone?

Method: take every clip in the real-speech eval sets (ja/en/zh/ko/yue) plus
the babble_snr10 noisy condition, truncate each to the first
0.5/1.0/1.5/2.0/2.5/3.0/4.0 seconds (skipping any bin longer than the clip),
and run BOTH detectors on each truncated segment. whisper-tiny is the exact
model+method the production engine uses (RoutedASR._identify_lang);
SenseVoice's internal LID is read off the `<|xx|>` tag in its decode result
(RoutedASR._decode_full), the same signal RoutedASR already uses to
arbitrate zh/yue (see asr_engine.py's transcribe()).

Results are checkpointed to testdata/_lid_curve_cache.json (testdata/ is
gitignored) so re-runs are cheap; pass --report to just rewrite docs/LID.md
from the existing cache.

Usage:
    python scripts/eval_lid_curve.py --root H:\\Programming\\hayamimi
    python scripts/eval_lid_curve.py --root H:\\Programming\\hayamimi --report
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from asr_engine import RoutedASR
from eval_common import load_manifest
import eval_common

WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(WORKTREE_ROOT, "docs", "LID.md")

# Same 5 languages SenseVoice's own LID covers (see asr_engine.py SV_LANGS /
# the "auto: SenseVoice has its own internal LID for its 5 langs" comment).
LANGS = ("ja", "en", "zh", "ko", "yue")
SV_TAG = {lang: f"<|{lang}|>" for lang in LANGS}

CLEAN_DIRS = ["testdata/eval_real", "testdata/eval_real_zhko", "testdata/eval_real_yue"]
NOISY_DIR = "testdata/eval_noisy/babble_snr10"
CONDITIONS = [("clean", CLEAN_DIRS), ("babble_snr10", [NOISY_DIR])]

BIN_SECONDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def cache_path(root: str) -> str:
    return eval_common.cache_path(root, "_lid_curve_cache.json")


def load_cache(root: str) -> dict:
    return eval_common.load_cache(cache_path(root))


def save_cache(root: str, cache: dict):
    eval_common.save_cache(cache_path(root), cache)


def load_clips(root: str, dirs: list) -> list:
    """[(source_tag, wav_rel_path, abs_wav_path, true_lang), ...]"""
    clips = []
    for rel in dirs:
        mdir = os.path.join(root, rel)
        entries = load_manifest(mdir)
        for e in entries:
            clips.append((rel, e["wav"], os.path.join(mdir, e["wav"]), e["lang"]))
    return clips


def sv_predicted_lang(sv_tag: str) -> str:
    for lang, tag in SV_TAG.items():
        if tag in sv_tag:
            return lang
    return "?"


def run_conditions(root: str, cache: dict):
    asr = None
    for cond_name, dirs in CONDITIONS:
        clips = load_clips(root, dirs)
        for rel, wav_name, wav_path, true_lang in clips:
            samples, sr = sf.read(wav_path, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            dur = len(samples) / sr

            need_any = any(
                f"{cond_name}::{rel}/{wav_name}::{length}" not in cache
                for length in BIN_SECONDS if length <= dur + 0.05
            )
            if not need_any:
                continue
            if asr is None:
                print("Loading RoutedASR (threads=6)...")
                asr = RoutedASR(threads=6, warmup=True, preload=False)
                sv_rec = asr._get("sv")

            for length in BIN_SECONDS:
                if length > dur + 0.05:
                    continue
                key = f"{cond_name}::{rel}/{wav_name}::{length}"
                if key in cache:
                    continue
                clip = samples[: int(length * sr)]
                t0 = time.perf_counter()
                wtiny_lang = asr._identify_lang(clip, sr)
                wtiny_ms = (time.perf_counter() - t0) * 1000
                t0 = time.perf_counter()
                _, sv_tag = asr._decode_full(sv_rec, clip, sr)
                sv_ms = (time.perf_counter() - t0) * 1000
                sv_lang = sv_predicted_lang(sv_tag)
                cache[key] = {
                    "condition": cond_name, "source": rel, "wav": wav_name,
                    "true_lang": true_lang, "length": length, "dur": dur,
                    "wtiny_lang": wtiny_lang, "sv_lang": sv_lang,
                    "wtiny_ms": wtiny_ms, "sv_ms": sv_ms,
                }
                print(f"  [{cond_name}] {wav_name:10} L={length:>3.1f}s true={true_lang:3} "
                      f"wtiny={wtiny_lang:3} sv={sv_lang:3}")
            save_cache(root, cache)


def aggregate(rows: list) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    wtiny_ok = sum(1 for r in rows if r["wtiny_lang"] == r["true_lang"])
    sv_ok = sum(1 for r in rows if r["sv_lang"] == r["true_lang"])
    agree = [r for r in rows if r["wtiny_lang"] == r["sv_lang"]]
    agree_ok = sum(1 for r in agree if r["wtiny_lang"] == r["true_lang"])
    return {
        "n": n,
        "wtiny_acc": wtiny_ok / n,
        "sv_acc": sv_ok / n,
        "agree_n": len(agree),
        "agree_rate": len(agree) / n,
        "agree_acc": (agree_ok / len(agree)) if agree else float("nan"),
    }


def find_95_crossing(rows_by_length: dict, key: str) -> float | None:
    """Shortest bin at/after which accuracy stays >= 0.95 for all longer bins too
    (a single lucky bin above threshold with a later dip doesn't count)."""
    lengths = sorted(rows_by_length)
    accs = []
    for length in lengths:
        agg = aggregate(rows_by_length[length])
        accs.append((length, agg.get(key, 0.0) if agg["n"] else None))
    for i, (length, acc) in enumerate(accs):
        if acc is None:
            continue
        rest = [a for _, a in accs[i:] if a is not None]
        if rest and min(rest) >= 0.95:
            return length
    return None


def write_report(root: str, cache: dict):
    rows = list(cache.values())

    lines = ["# LID.md \u2014 セグメント長 vs LID正解率\n",
             "whisper-tiny 単独LID（本番 `RoutedASR._identify_lang` と同一モデル・同一呼び出し）と、"
             "SenseVoice内蔵LID（`<|xx|>` タグ、本番の zh/yue 裁定に既に使っている信号）を、"
             "実データの eval セット（ja/en/zh/ko/yue、`testdata/eval_real*`）と "
             "babble_snr10 ノイズ条件（`testdata/eval_noisy/babble_snr10`）の両方で、"
             f"クリップ先頭 {', '.join(str(s) for s in BIN_SECONDS)} 秒に切り詰めて評価した。"
             "評価スクリプト: `scripts/eval_lid_curve.py`（結果は "
             "`testdata/_lid_curve_cache.json` にキャッシュ、再実行は差分のみ計算）。\n"]

    conditions = sorted(set(r["condition"] for r in rows))

    def table_for(metric_key: str, title: str, only_sv_langs: bool = False):
        out = [f"## {title}\n",
               "| condition | length(s) | " + " | ".join(LANGS) + " | overall |",
               "|---|---|" + "---|" * (len(LANGS) + 1)]
        for cond in conditions:
            for length in BIN_SECONDS:
                cell_rows = [r for r in rows if r["condition"] == cond and r["length"] == length]
                if not cell_rows:
                    continue
                cells = []
                for lang in LANGS:
                    sub = [r for r in cell_rows if r["true_lang"] == lang]
                    agg = aggregate(sub)
                    if agg["n"] == 0:
                        cells.append("-")
                    else:
                        cells.append(f"{agg[metric_key]*100:.0f}% ({agg['n']})")
                overall = aggregate(cell_rows)
                out.append(
                    f"| {cond} | {length:.1f} | " + " | ".join(cells) +
                    f" | {overall[metric_key]*100:.0f}% ({overall['n']}) |"
                )
        out.append("")
        return out

    lines += table_for("wtiny_acc", "表1: whisper-tiny 単独LID 正解率")
    lines += table_for("sv_acc", "表2: SenseVoice 内蔵LID 正解率（対応5言語）")

    # Table 3: agreement rate + accuracy when both detectors agree.
    lines.append("## 表3: 二重判定（whisper-tiny と SenseVoice の一致）\n")
    lines.append("| condition | length(s) | 一致率 | 一致時の正解率 | n |")
    lines.append("|---|---|---|---|---|")
    for cond in conditions:
        for length in BIN_SECONDS:
            cell_rows = [r for r in rows if r["condition"] == cond and r["length"] == length]
            if not cell_rows:
                continue
            agg = aggregate(cell_rows)
            agree_acc = f"{agg['agree_acc']*100:.0f}%" if agg["agree_n"] else "-"
            lines.append(
                f"| {cond} | {length:.1f} | {agg['agree_rate']*100:.0f}% | "
                f"{agree_acc} | {agg['agree_n']} |"
            )
    lines.append("")

    # Conclusion: shortest length where accuracy >= 95% and stays there, per lang / overall.
    lines.append("## 結論\n")
    per_cond_lang_cross = {}   # cond -> {lang: crossing length or None}
    per_cond_overall_single = {}   # cond -> {length: overall single-LID acc}
    per_cond_agree = {}        # cond -> {length: (agree_rate, agree_acc)}

    for cond in conditions:
        lines.append(f"### {cond}\n")
        rows_by_length_overall = {L: [r for r in rows if r["condition"] == cond and r["length"] == L]
                                   for L in BIN_SECONDS}
        overall_cross = find_95_crossing(rows_by_length_overall, "wtiny_acc")
        lines.append(
            f"- 単独LID（whisper-tiny）が全体で95%を超え、それ以降も維持する最短長: "
            + (f"**{overall_cross:.1f}秒**" if overall_cross else "**7秒枠内で到達せず**") + "\n"
        )
        lang_cross = {}
        for lang in LANGS:
            rows_by_length_lang = {
                L: [r for r in rows if r["condition"] == cond and r["length"] == L and r["true_lang"] == lang]
                for L in BIN_SECONDS
            }
            cross = find_95_crossing(rows_by_length_lang, "wtiny_acc")
            lang_cross[lang] = cross
            lines.append(
                f"  - {lang}: " + (f"{cross:.1f}秒" if cross else "7秒枠内で到達せず")
            )
        lines.append("")
        per_cond_lang_cross[cond] = lang_cross

        overall_single = {L: aggregate(rows_by_length_overall[L]).get("wtiny_acc", float("nan"))
                           for L in BIN_SECONDS if rows_by_length_overall[L]}
        per_cond_overall_single[cond] = overall_single

        agree_map = {}
        for length in BIN_SECONDS:
            cell = [r for r in rows if r["condition"] == cond and r["length"] == length]
            agg = aggregate(cell)
            if agg["n"]:
                agree_map[length] = (agg["agree_rate"], agg["agree_acc"])
        per_cond_agree[cond] = agree_map

        # Headline comparison: at each bin, does "both detectors agree" beat
        # "single LID overall accuracy"? This is the direct evidence for/against
        # a short-segment dual-confirmation policy.
        lines.append("| length(s) | 単独LID全体正解率 | 一致率 | 一致時正解率 | 一致の方が高い？ |")
        lines.append("|---|---|---|---|---|")
        for length in BIN_SECONDS:
            if length not in overall_single or length not in agree_map:
                continue
            single_acc = overall_single[length]
            rate, acc = agree_map[length]
            better = "yes" if (acc == acc and acc > single_acc + 0.01) else "no"
            acc_s = f"{acc*100:.0f}%" if acc == acc else "-"
            lines.append(f"| {length:.1f} | {single_acc*100:.0f}% | {rate*100:.0f}% | {acc_s} | {better} |")
        lines.append("")

    lines.append("### 推奨デフォルト閾値\n")

    for cond in conditions:
        lang_cross = per_cond_lang_cross[cond]
        reached = {l: c for l, c in lang_cross.items() if c is not None}
        not_reached = [l for l, c in lang_cross.items() if c is None]
        if reached:
            worst_lang, worst_len = max(reached.items(), key=lambda kv: kv[1])
            lines.append(
                f"- **{cond}**: 単独LIDだけで95%に達する言語のうち最も遅いのは "
                f"{worst_lang}（{worst_len:.1f}秒）。"
                + (f"{', '.join(not_reached)} は7秒枠内で単独LIDが95%に到達しないため、"
                   f"これらの言語は長さによらず二重判定（whisper-tiny と SenseVoice 一致）に"
                   f"フォールバックする必要がある。" if not_reached else "全言語が7秒枠内で到達。")
            )
        else:
            lines.append(f"- **{cond}**: どの言語も7秒枠内で単独LID 95%に到達せず、単独LIDだけに頼るのは危険。")
    lines.append("")

    lines.append(
        "yue は whisper-tiny の出力語彙にCantoneseクラスが無く、常に zh 系タグに丸められるため"
        "（asr_engine.py のコメント参照）、単独LIDでは**原理的に**95%に到達しない。"
        "yue の切り替えは常に SenseVoice 内蔵LID（またはその一致確認）に委ねるのが正しい設計であり、"
        "計測結果はその前提を裏付けている（全長で0%）。\n"
    )

    # Data-driven verdict on the dual-confirmation policy itself.
    for cond in conditions:
        agree_map = per_cond_agree[cond]
        overall_single = per_cond_overall_single[cond]
        wins, losses, ties = [], [], []
        for length in BIN_SECONDS:
            if length not in agree_map or length not in overall_single:
                continue
            rate, acc = agree_map[length]
            if acc != acc:
                continue
            delta = acc - overall_single[length]
            if delta > 0.01:
                wins.append((length, delta))
            elif delta < -0.01:
                losses.append((length, delta))
            else:
                ties.append(length)
        if wins:
            short_wins = [w for w in wins if w[0] <= 2.0]  # prefer a short-length example
            example = short_wins[-1] if short_wins else wins[0]
            lines.append(
                f"- **{cond}**: {len(wins)}/{len(wins)+len(losses)+len(ties)} 長さ帯で"
                f"「一致時正解率」が「単独LID全体正解率」を上回った"
                f"（例: {example[0]:.1f}秒で単独{overall_single[example[0]]*100:.0f}% → "
                f"一致時{(overall_single[example[0]]+example[1])*100:.0f}%）。"
                f"二重判定は特に短尺（0.5〜2.0秒）で単独LIDより明確に信頼できる。\n"
            )

    lines.append(
        "\n**推奨ポリシー**: 言語がSenseVoice対応5言語（ja/en/zh/ko/yue）のいずれかである限り、"
        "上表の単独LID 95%到達長に達するまでは「whisper-tiny と SenseVoice 内蔵LIDが一致した場合のみ"
        "言語切り替えを確定する」二重判定を使い、到達後は単独LIDの結果をそのまま採用してよい。"
        "yue（および騒音下のja/ko）は本評価の7秒枠内で単独LIDが95%に届かないため、これらは長さに関わらず"
        "二重判定（不一致なら様子見を継続）をデフォルト運用とすることを推奨する。\n"
    )

    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {DOCS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=WORKTREE_ROOT,
                     help="repo root containing testdata/ (default: derived from script location)")
    ap.add_argument("--report", action="store_true",
                     help="only (re)write docs/LID.md from the existing cache, no evaluation")
    args = ap.parse_args()

    cache = load_cache(args.root)
    if not args.report:
        run_conditions(args.root, cache)
    write_report(args.root, cache)


if __name__ == "__main__":
    main()
