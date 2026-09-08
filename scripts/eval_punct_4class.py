"""Evaluate the 4-class (+ none) ja punctuation token classifier
(models/punct-ja-4class-permissive/ by default) against the same FLEURS ja
harness used for the existing mojicast fp32/int8 comparison
(scripts/quantize_punct.py), per improvement track C's acceptance criteria
(docs/eval/punct_retrain.md).

Reuses scripts/quantize_punct.py's FLEURS fetch/sampling
(build_fleurs_refs) and testdata/eval_real loader, but scores with a
*fixed* strip_marks/marks_from_restored: the originals NFKC-normalize the
text before checking membership in a fullwidth-only TARGET_MARKS set,
which silently folds "？"/"！" (U+FF1F/U+FF01) to ASCII "?"/"!" and drops
them from the ground truth entirely -- see _safe_nfkc below and
docs/eval/punct_retrain.md for the full writeup of this bug (found while
building this task's training labels). This script's TARGET_MARKS also
adds "！", which the existing script never tracked (the old model can't
predict it).

Runs in the repo's normal .venv: it drives PunctuatorJa4Class from
scripts/punct_ja.py (onnxruntime + `tokenizers`), so no torch/transformers
is needed, and the numbers below describe the code path that actually
ships rather than a parallel reimplementation of it.

Usage:
    .venv/Scripts/python scripts/eval_punct_4class.py --variant fp32 --n 250 --latency
    .venv/Scripts/python scripts/eval_punct_4class.py --variant int8 --n 250 --latency
    .venv/Scripts/python scripts/eval_punct_4class.py --variant fp32 --question-set
    .venv/Scripts/python scripts/eval_punct_4class.py --variant baseline --n 250 --latency
        # the existing shipped mojicast model (scripts/punct_ja.py::PunctuatorJa),
        # scored with this script's fixed scorer for a like-for-like comparison
    .venv/Scripts/python scripts/eval_punct_4class.py --variant fp32 --n 250 \
        --model-dir models/punct-ja-4class    # the superseded first-round model
"""
import argparse
import json
import os
import sys
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from quantize_punct import build_fleurs_refs, load_eval_real_refs  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    """Same implementation as scripts/eval_accuracy.py::levenshtein, inlined
    here to avoid pulling that module's heavy audio deps (soundfile etc.,
    not installed in .venv-train) into this text-only eval script."""
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
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]

DEFAULT_MODEL_DIR = os.path.join(ROOT, "models", "punct-ja-4class-permissive")
FP32_ONNX_NAME = "punct_4class.onnx"
INT8_ONNX_NAME = os.path.join("quantized_ort", "punct_4class.int8.onnx")

LABELS = ["O", "、", "。", "？", "！"]
TARGET_MARKS = ("、", "。", "？", "！")

_Q_SENTINEL = ""
_E_SENTINEL = ""


def _safe_nfkc(text: str) -> str:
    """See module docstring / docs/eval/punct_retrain.md -- plain NFKC
    folds fullwidth "？"/"！" to ASCII "?"/"!", which breaks fullwidth-only
    mark-membership checks downstream. Protect them with PUA sentinels."""
    text = text.replace("？", _Q_SENTINEL).replace("！", _E_SENTINEL)
    text = unicodedata.normalize("NFKC", text)
    return text.replace(_Q_SENTINEL, "？").replace(_E_SENTINEL, "！")


def strip_marks(text: str, marks=TARGET_MARKS):
    norm = _safe_nfkc(text)
    stripped, marks_after = [], []
    for ch in norm:
        if ch in marks:
            if stripped and not marks_after[-1]:
                marks_after[-1] = ch
            continue
        stripped.append(ch)
        marks_after.append("")
    return "".join(stripped), marks_after


def marks_from_restored(restored: str, base_len: int, marks=TARGET_MARKS):
    marks_after = []
    for ch in restored:
        if ch in marks:
            if marks_after and not marks_after[-1]:
                marks_after[-1] = ch
            continue
        marks_after.append("")
    if len(marks_after) < base_len:
        marks_after += [""] * (base_len - len(marks_after))
    return marks_after[:base_len]


def punct_cer(ref: str, hyp: str):
    r = _safe_nfkc(ref)
    h = _safe_nfkc(hyp)
    dist = levenshtein(r, h)
    return (dist / len(r) if r else 0.0), dist, len(r)


def build_punctuator(variant, model_dir, num_threads):
    """Return a `.restore(text) -> punctuated text` callable for `variant`.

    fp32/int8 both go through PunctuatorJa4Class -- the same class callers
    would use -- rather than a private copy of its logic, so a bug in the
    shipped reconstruction path shows up in these numbers instead of hiding
    behind an eval-only reimplementation.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    if variant == "baseline":
        from punct_ja import PunctuatorJa
        p = PunctuatorJa()
        print(f"[model] shipped mojicast fp32 ({p.session.get_modelmeta().graph_name or 'punct_bert.onnx'})")
        return p.restore

    from punct_ja import PunctuatorJa4Class

    onnx_filename = FP32_ONNX_NAME if variant == "fp32" else INT8_ONNX_NAME
    print(f"[model] {os.path.join(model_dir, onnx_filename)}")
    p = PunctuatorJa4Class(model_dir=model_dir, onnx_filename=onnx_filename,
                           num_threads=num_threads)
    return p.restore


def evaluate(restore_fn, label, refs, latency=False):
    tp = fp = fn = 0
    per_mark = {m: {"tp": 0, "fp": 0, "fn": 0} for m in TARGET_MARKS}
    total_dist = total_denom = 0
    latencies = []
    rows = []
    for ref_id, ref in refs:
        stripped, ref_marks = strip_marks(ref)
        if not stripped:
            continue
        t0 = time.time()
        hyp = restore_fn(stripped)
        if latency:
            latencies.append(time.time() - t0)
        hyp_marks = marks_from_restored(hyp, len(stripped))

        for rm, hm in zip(ref_marks, hyp_marks):
            if rm and hm and rm == hm:
                tp += 1
                per_mark[rm]["tp"] += 1
            elif rm and (not hm or hm != rm):
                fn += 1
                per_mark[rm]["fn"] += 1
                if hm:
                    fp += 1
                    per_mark[hm]["fp"] += 1
            elif hm and not rm:
                fp += 1
                per_mark[hm]["fp"] += 1

        rate, dist, denom = punct_cer(ref, hyp)
        total_dist += dist
        total_denom += denom
        rows.append({"id": ref_id, "ref": ref, "hyp": hyp, "cer": rate})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    micro_cer = total_dist / total_denom if total_denom else float("nan")
    mean_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else None

    per_mark_metrics = {}
    for m, c in per_mark.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_mark_metrics[m] = {"precision": p, "recall": r, "f1": f, "support": c["tp"] + c["fn"]}

    print(f"  [{label}] n={len(rows)} P={precision:.4f} R={recall:.4f} F1={f1:.4f} "
          f"CER={micro_cer:.4f}" + (f" latency={mean_latency_ms:.2f}ms" if mean_latency_ms else ""))
    for m, mm in per_mark_metrics.items():
        print(f"      {m}: P={mm['precision']:.4f} R={mm['recall']:.4f} F1={mm['f1']:.4f} "
              f"support={mm['support']}")

    return {
        "label": label, "n": len(rows), "precision": precision, "recall": recall, "f1": f1,
        "cer": micro_cer, "mean_latency_ms": mean_latency_ms, "per_mark": per_mark_metrics,
        "rows": rows,
    }


def build_question_set(min_n=50, seed=0):
    """FLEURS ja sentences ending in "か。" or "？" (the acceptance
    criterion's question-recall eval set). Draws from a larger FLEURS pool
    than the main 250-sentence eval (disjoint sampling isn't required here
    since this measures a different, ？-specific metric, not F1 on the
    same distribution) and synthesizes extra ones if under 50."""
    pool = build_fleurs_refs(n=100000, seed=seed)  # effectively "all" after build_fleurs_refs's own cap
    qs = [t for t in pool if t.endswith(("か。", "？"))]
    n_real = len(qs)
    if len(qs) < min_n:
        # FLEURS ja (read-aloud, largely encyclopedic/declarative text) has
        # *zero* sentences ending in "か。"/"？" in the 456-sentence pool
        # (checked directly) -- the task brief's own fallback plan for
        # exactly this case ("50文に満たなければ自作で補う"). All 50+ below
        # are self-authored and distinct (no cycling/repeats), covering a
        # spread of common Japanese question forms (か/でしょうか/かな/の,
        # short and long, with and without 、-clauses).
        synth = [
            "これは正しいですか？", "何時に始まりますか？", "本当に大丈夫なんですか？",
            "どこに行けばいいのでしょうか？", "誰が担当するのですか？", "なぜそうなったのですか？",
            "いつ終わる予定ですか？", "それはどういう意味ですか？", "もう一度説明してもらえますか？",
            "何が問題なのでしょうか？", "彼は本当に来るのですか？", "これで合っていますか？",
            "次はどうすればいいですか？", "その理由は何ですか？", "どちらを選べばいいですか？",
            "本当にそれで良いのですか？", "誰に聞けばわかりますか？", "いつまでに終わらせればいいですか？",
            "これはいくらですか？", "どのくらい時間がかかりますか？",
            "会議は何時から始まりますか？", "資料はもう届いていますか？", "この道で合っていますか？",
            "そちらの天気はどうですか？", "明日は休みですか？", "これは誰の意見ですか？",
            "その計画はいつ実行するのですか？", "駅までどれくらいかかりますか？",
            "予約はもう済んでいますか？", "本当にそれでいいと思いますか？",
            "何か手伝えることはありますか？", "この機能はどう使うのですか？",
            "会場はどこになりますか？", "参加費はいくらかかりますか？",
            "彼女は今どこにいるのですか？", "その話は本当なのですか？",
            "次の電車は何分後に来ますか？", "この件について誰に相談すればいいですか？",
            "今日の会議に出席しますか？", "そのファイルはもう保存しましたか？",
            "結果はいつわかりますか？", "この商品はまだ在庫がありますか？",
            "その提案に賛成ですか、それとも反対ですか？", "今から向かっても間に合いますか？",
            "この文章に間違いはありますか？", "サポートにはどこから連絡すればいいですか？",
            "写真はもう撮り終わりましたか？", "契約書にはもうサインしましたか？",
            "この設定で問題ないでしょうか？", "他に質問はありますか？",
            "本当にこれだけで十分なのかな？", "彼は納得しているのだろうか？",
        ]
        assert len(synth) == len(set(synth)), "synthetic question templates must be unique"
        i = 0
        while len(qs) < min_n and i < len(synth):
            if synth[i] not in qs:
                qs.append(synth[i])
            i += 1
    print(f"  [question-set] {n_real} real FLEURS question-form sentences, "
          f"{len(qs) - n_real} self-authored added (target n={min_n})")
    return qs[:max(min_n, len(qs))]


def build_exclaim_set():
    """A self-authored ！-terminated eval set, the ！ counterpart of
    build_question_set().

    FLEURS ja is read-aloud encyclopedic text and contains a single ！ in
    the whole 250-sentence eval draw, so it cannot say anything about the
    ！ class either way. ！ is not an acceptance criterion, but round 1
    trained it on ~600 self-authored templates and scored F1 0.23 on its
    own held-out split, so round 2 -- which has real ！ from web text --
    owes a number rather than a shrug. These sentences are written for
    this eval and are disjoint from the training corpus's domain (they
    are not web text), so this measures generalization, not recall of
    memorized templates.
    """
    return [
        "本当にすごい景色でした！", "それは絶対にやめたほうがいいですよ！",
        "みんなで力を合わせて頑張りましょう！", "今日は本当に楽しかったです！",
        "危ないから、そこに近づかないでください！", "おめでとうございます！",
        "早くしないと電車に乗り遅れますよ！", "こんなに嬉しいことはありません！",
        "もう一度だけ挑戦してみます！", "ずっと会いたかったです！",
        "ようやく完成しました！", "こんな結果になるとは思いませんでした！",
        "本当にありがとうございました！", "絶対に忘れません！",
        "気をつけて帰ってくださいね！", "やっと春が来ました！",
        "その話、初めて聞きました！", "とんでもない大失敗でした！",
        "これ以上は待てません！", "信じられないくらい美味しかったです！",
        "全員無事でよかったです！", "また一緒に行きましょう！",
        "本当に助かりました！", "こんな偶然ってあるんですね！",
        "予想をはるかに超える出来栄えです！", "急いで支度をしてください！",
        "ついに目標を達成しました！", "みなさん、おはようございます！",
        "こんなに寒い朝は久しぶりです！", "その色、とても似合っていますよ！",
        "もう限界です、少し休ませてください！", "全部間違えました、すみません！",
        "今すぐ確認してみます！", "こんなところで会うなんて驚きました！",
        "みなさん、準備はできましたか、始めますよ！",
        "最後まで諦めずに走り切りました！", "本当に立派な仕事ぶりでした！",
        "この景色を見られただけでも来た甲斐がありました！",
        "手伝ってくれて本当に感謝しています！", "まさか優勝できるとは！",
        "いや、やっぱりやめておきます！",
        "静かにしてください、赤ちゃんが寝ています！",
        "こんなに長い行列は見たことがありません！", "無事に到着しました！",
        "その提案、とてもいいと思います！", "雨が降ってきました、傘を持ってきて！",
        "一生の思い出になりました！", "本当にお疲れさまでした！",
        "こんな素敵な贈り物をいただけるなんて！", "さあ、出発しましょう！",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fp32", "int8", "baseline"], default="fp32",
                    help="fp32/int8: the 4-class model under --model-dir. "
                         "baseline: the shipped mojicast PunctuatorJa, scored with "
                         "the same fixed scorer.")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--source", choices=["eval_real", "fleurs"], default="fleurs")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--question-set", action="store_true",
                     help="evaluate ？ recall on the FLEURS question-only set instead of "
                          "the main n-sentence set")
    ap.add_argument("--exclaim-set", action="store_true",
                     help="evaluate ！ on a self-authored exclamation set (quality note, "
                          "not an acceptance criterion -- see build_exclaim_set)")
    ap.add_argument("--threads", type=int, default=2,
                     help="onnxruntime intra_op_num_threads (default 2 -- other eval "
                          "tracks run in parallel on this CPU)")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    restore = build_punctuator(args.variant, model_dir, args.threads)

    if args.exclaim_set:
        texts = build_exclaim_set()
        refs = [(f"e_{i:04d}", t) for i, t in enumerate(texts)]
        print(f"\n[source] {len(refs)} self-authored ！ sentences")
    elif args.question_set:
        print("\n[source] building FLEURS ja question-only set...")
        texts = build_question_set()
        refs = [(f"q_{i:04d}", t) for i, t in enumerate(texts)]
        print(f"[source] {len(refs)} question sentences")
    elif args.source == "fleurs":
        print(f"\n[source] building FLEURS ja eval set (n={args.n}, seed={args.seed})...")
        texts = build_fleurs_refs(n=args.n, seed=args.seed)
        refs = [(f"fleurs_{i:04d}", t) for i, t in enumerate(texts)]
        print(f"[source] {len(refs)} FLEURS ja sentences")
    else:
        refs = load_eval_real_refs()
        print(f"\n[source] {len(refs)} testdata/eval_real ja clips")

    result = evaluate(restore, args.variant, refs, latency=args.latency)

    if args.question_set:
        q_recall = result["per_mark"]["？"]["recall"]
        print(f"\n[question-set verdict] ？ recall = {q_recall:.4f} "
              f"({'PASS' if q_recall >= 0.7 else 'FAIL'}, threshold 0.7)")
    if args.exclaim_set:
        e = result["per_mark"]["！"]
        print(f"\n[exclaim-set] ！ P={e['precision']:.4f} R={e['recall']:.4f} "
              f"F1={e['f1']:.4f} (quality note, not an acceptance criterion)")

    tag = ("question_set" if args.question_set
           else "exclaim_set" if args.exclaim_set else args.source)
    # The model directory is part of the filename: comparing the round-1 and
    # round-2 models means running the same --variant twice, and without it
    # the second run silently overwrites the first one's numbers.
    if args.variant == "baseline":
        model_tag, reported_dir = "mojicast", "models/mojicast-punct-onnx"
    else:
        model_tag, reported_dir = os.path.basename(model_dir), model_dir
    out_dir = os.path.join(ROOT, "testdata", "punct_4class_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_{model_tag}_{args.variant}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_dir": reported_dir, "variant": args.variant, "n": result["n"],
            "precision": result["precision"], "recall": result["recall"],
            "f1": result["f1"], "cer": result["cer"],
            "mean_latency_ms": result["mean_latency_ms"], "per_mark": result["per_mark"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
