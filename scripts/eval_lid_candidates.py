"""LID replacement-candidate evaluation (improvement track A).

Scores candidate spoken-language-identification detectors against the exact
same real-audio harness scripts/eval_lid_curve.py uses for the production
whisper-tiny LID and SenseVoice's internal LID (clean + babble_snr10,
testdata/eval_real* + testdata/eval_real_zhko + testdata/eval_real_yue,
truncated to 0.5..4.0s), so the numbers in docs/eval/lid_candidates.md are
directly comparable to docs/eval/lid.md's tables 1-2. It does NOT touch
eval_lid_curve.py or docs/eval/lid.md -- those stay exactly as published.

Candidates measurable from this script alone (sherpa-onnx only, main venv):
  whisper-base   -- candidate (b): same whisper LID technique as production,
                     bigger checkpoint.
  sensevoice     -- candidate (c): contrast baseline, SenseVoice's own
                     internal LID tag alone (already used by production as
                     the second half of dual-confirm; this table asks "how
                     good is it ALONE"). Reproduces docs/eval/lid.md's table
                     2 under this script's own harness as a consistency check.

Candidates (a) speechbrain/lang-id-voxlingua107-ecapa and (d) ECAPA
embedding + target-only classifier require torch/speechbrain and are
measured by the separate scripts/eval_lid_voxlingua.py, run under
.venv-train (see that script's docstring). This script only READS their
result caches (testdata/_lid_candidates_voxlingua_cache.json) to fold them
into the same report -- it never imports torch.

Usage:
    .venv/Scripts/python scripts/eval_lid_candidates.py --backend whisper-base
    .venv/Scripts/python scripts/eval_lid_candidates.py --backend sensevoice
    .venv/Scripts/python scripts/eval_lid_candidates.py --backend all
    .venv/Scripts/python scripts/eval_lid_candidates.py --report   # just rewrite the doc
"""
import argparse
import os
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sherpa_onnx
import soundfile as sf

import asr_engine
import eval_common
from eval_lid_curve import BIN_SECONDS, CONDITIONS, LANGS, SV_TAG, aggregate, load_clips

WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(WORKTREE_ROOT, "docs", "eval", "lid_candidates.md")
THREADS = 2  # 6-core CPU is shared with 5 other parallel eval tracks -- see the track brief

WHISPER_BASE_DIR = os.path.join(asr_engine.MODELS_DIR, "sherpa-onnx-whisper-base")
VOXLINGUA_CACHE_NAME = "_lid_candidates_voxlingua_cache.json"

# scripts/eval_lid_voxlingua.py runs under .venv-train (this script never
# does -- see its own docstring), so it can't be asked for its model size at
# report time; these paths are where it downloads/trains its artifacts, read
# straight off disk here instead (both are gitignored, worktree-local).
VOXLINGUA_PRETRAINED_DIR = os.path.join(WORKTREE_ROOT, ".venv-train", "pretrained", "voxlingua107-ecapa")
VOXLINGUA_FLEURS_CACHE_DIR = os.path.join(WORKTREE_ROOT, ".venv-train", "fleurs_cache")
VOXLINGUA_MODEL_PATHS = {
    # candidate (a): embedding + the model's own pretrained 107-way head.
    "voxlingua107-ecapa-raw": [
        os.path.join(VOXLINGUA_PRETRAINED_DIR, "embedding_model.ckpt"),
        os.path.join(VOXLINGUA_PRETRAINED_DIR, "classifier.ckpt"),
    ],
    # candidate (d): the SAME embedding, but the tiny from-scratch 5-way
    # head instead of the 107-way one (no classifier.ckpt needed).
    "voxlingua107-ecapa-target-clf": [
        os.path.join(VOXLINGUA_PRETRAINED_DIR, "embedding_model.ckpt"),
        os.path.join(VOXLINGUA_FLEURS_CACHE_DIR, "target_clf.json"),
    ],
}


def sv_predicted_lang(sv_tag: str) -> str:
    for lang, tag in SV_TAG.items():
        if tag in sv_tag:
            return lang
    return "?"


# --- pluggable backends ------------------------------------------------------
#
# Every backend exposes identify(clip, sr) -> (pred_lang, confidence|None).
# confidence is a 0..1 posterior when the backend can produce one (criterion
# 6 in the track brief), None when it can't (whisper's sherpa-onnx binding
# exposes only the argmax language string, same limitation production
# whisper-tiny already has -- see docs/eval/lid.md).

class WhisperBaseBackend:
    name = "whisper-base"
    model_dirs = [WHISPER_BASE_DIR]

    def __init__(self, threads: int = THREADS):
        encoder = asr_engine._find(WHISPER_BASE_DIR, "base-encoder.int8.onnx")
        decoder = asr_engine._find(WHISPER_BASE_DIR, "base-decoder.int8.onnx")
        if not encoder or not decoder:
            raise FileNotFoundError(
                f"whisper-base model not found under {WHISPER_BASE_DIR} -- "
                f"run scripts/download_models.py --lid-candidates")
        whisper_cfg = sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=encoder, decoder=decoder)
        cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(whisper=whisper_cfg, num_threads=threads)
        self.lid = sherpa_onnx.SpokenLanguageIdentification(cfg)

    def identify(self, clip, sr):
        stream = self.lid.create_stream()
        stream.accept_waveform(sr, clip)
        return self.lid.compute(stream), None


class SenseVoiceBackend:
    name = "sensevoice"
    model_dirs = [asr_engine.SV_MODEL_DIR]

    def __init__(self, threads: int = THREADS):
        self.rec = asr_engine._build_sense_voice(threads)

    def identify(self, clip, sr):
        _, tag = asr_engine.RoutedASR._decode_full(self.rec, clip, sr)
        return sv_predicted_lang(tag), None


BACKENDS = {
    "whisper-base": WhisperBaseBackend,
    "sensevoice": SenseVoiceBackend,
}


def dir_size_bytes(paths: list) -> int:
    total = 0
    for p in paths:
        if os.path.isfile(p):
            total += os.path.getsize(p)
            continue
        for root, _, files in os.walk(p):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    return total


# --- eval loop ----------------------------------------------------------------

def cache_path(root: str, backend_name: str) -> str:
    return eval_common.cache_path(root, f"_lid_candidates_{backend_name}_cache.json")


def run_backend(root: str, backend_name: str, cache: dict):
    backend = None
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
            if backend is None:
                print(f"Loading {backend_name} (threads={THREADS})...")
                backend = BACKENDS[backend_name](threads=THREADS)

            for length in BIN_SECONDS:
                if length > dur + 0.05:
                    continue
                key = f"{cond_name}::{rel}/{wav_name}::{length}"
                if key in cache:
                    continue
                clip = asr_engine.trim_lid_clip(samples[: int(length * sr)], sr, max_seconds=length)
                t0 = time.perf_counter()
                pred_lang, conf = backend.identify(clip, sr)
                ms = (time.perf_counter() - t0) * 1000
                cache[key] = {
                    "backend": backend_name, "condition": cond_name, "source": rel,
                    "wav": wav_name, "true_lang": true_lang, "length": length, "dur": dur,
                    "pred_lang": pred_lang, "confidence": conf, "latency_ms": ms,
                }
                print(f"  [{backend_name}][{cond_name}] {wav_name:10} L={length:>3.1f}s "
                      f"true={true_lang:3} pred={pred_lang:3} ({ms:.0f}ms)")
            eval_common.save_cache(cache_path(root, backend_name), cache)


def model_size_mb(backend_name: str) -> "float | None":
    if backend_name in BACKENDS:
        return dir_size_bytes(BACKENDS[backend_name].model_dirs) / 1e6
    if backend_name in VOXLINGUA_MODEL_PATHS:
        paths = [p for p in VOXLINGUA_MODEL_PATHS[backend_name] if os.path.exists(p)]
        return dir_size_bytes(paths) / 1e6 if paths else None
    return None


def latency_p50_ms(rows: list) -> "float | None":
    four_s = [r["latency_ms"] for r in rows if r["length"] == 4.0 and r.get("latency_ms") is not None]
    return statistics.median(four_s) if four_s else None


# --- report -------------------------------------------------------------------

CRITERIA_HEADER = (
    "| # | 基準 | 閾値 |\n|---|---|---|\n"
    "| 1 | クリーン2秒 正解率 | ≥90% |\n"
    "| 2 | babble_snr10 2秒 正解率 | ≥80% |\n"
    "| 3 | yueを独立クラスとして判定 | 可能 |\n"
    "| 4 | モデルサイズ | ≤100MB |\n"
    "| 5 | 4秒入力の判定レイテンシ(CPU) | ≤140ms |\n"
    "| 6 | 事後確率(信頼度)を返せる | 可能 |\n"
)


def acc_at(rows, condition, length, lang=None):
    sub = [r for r in rows if r["condition"] == condition and r["length"] == length]
    if lang:
        sub = [r for r in sub if r["true_lang"] == lang]
    agg = aggregate([{**r, "wtiny_lang": r["pred_lang"], "sv_lang": r["pred_lang"]} for r in sub])
    return agg


def table_for(rows, title):
    out = [f"### {title}\n",
           "| condition | length(s) | " + " | ".join(LANGS) + " | overall |",
           "|---|---|" + "---|" * (len(LANGS) + 1)]
    conditions = sorted(set(r["condition"] for r in rows))
    for cond in conditions:
        for length in BIN_SECONDS:
            cell_rows = [r for r in rows if r["condition"] == cond and r["length"] == length]
            if not cell_rows:
                continue
            cells = []
            for lang in LANGS:
                agg = acc_at(rows, cond, length, lang)
                cells.append(f"{agg['wtiny_acc']*100:.0f}% ({agg['n']})" if agg["n"] else "-")
            overall = acc_at(rows, cond, length)
            out.append(f"| {cond} | {length:.1f} | " + " | ".join(cells) +
                       f" | {overall['wtiny_acc']*100:.0f}% ({overall['n']}) |")
    out.append("")
    return out


def verdict_for(candidate_name, clean2, babble2, has_yue, size_mb, latency_ms, has_confidence):
    checks = {
        "1. clean 2s >= 90%": clean2 is not None and clean2 >= 0.90,
        "2. babble_snr10 2s >= 80%": babble2 is not None and babble2 >= 0.80,
        "3. yue independent class": has_yue,
        "4. size <= 100MB": size_mb is not None and size_mb <= 100,
        "5. 4s latency <= 140ms": latency_ms is not None and latency_ms <= 140,
        "6. returns confidence": has_confidence,
    }
    adopted = all(checks.values())
    lines = [f"**{'採用' if adopted else '不採用'}**\n"]
    for label, ok in checks.items():
        lines.append(f"- [{'x' if ok else ' '}] {label}")
    return adopted, "\n".join(lines)


def write_report(root: str):
    all_rows_by_backend = {}
    for name in BACKENDS:
        cache = eval_common.load_cache(cache_path(root, name))
        if cache:
            all_rows_by_backend[name] = list(cache.values())

    vox_cache = eval_common.load_cache(eval_common.cache_path(root, VOXLINGUA_CACHE_NAME))
    for row in vox_cache.values():
        all_rows_by_backend.setdefault(row["backend"], []).append(row)

    lines = ["# LID候補評価 (改善トラックA)\n",
             "whisper-tiny単独LID(現行本番)を置き換える候補を、`docs/eval/lid.md`と同一の実データ"
             "ハーネス(`scripts/eval_lid_curve.py`のtestdata/eval_real*・babble_snr10・0.5〜4.0秒bin)"
             "で評価した。既存の`docs/eval/lid.md`・`scripts/eval_lid_curve.py`は変更していない。\n",
             "## 採用基準(全部満たしたら採用)\n", CRITERIA_HEADER]

    lines.append("## 候補一覧\n")
    lines.append("| 候補 | 説明 | ライセンス | 出典 |")
    lines.append("|---|---|---|---|")
    lines.append("| (b) whisper-base | whisper-tinyと同じsherpa-onnx LID手法、大きいチェックポイント | MIT (OpenAI Whisper) | k2-fsa/sherpa-onnx release `sherpa-onnx-whisper-base.tar.bz2` |")
    lines.append("| (c) SenseVoice内蔵LID(対照) | 本番で二重判定の片翼として既に使用中の信号を単独評価 | Apache-2.0 | 既存 `models/sherpa-onnx-sense-voice-*` |")
    lines.append("| (a) speechbrain VoxLingua107-ECAPA | 107言語話者言語識別、ECAPA-TDNN埋め込み+線形分類器 | Apache-2.0 | HF `speechbrain/lang-id-voxlingua107-ecapa` |")
    lines.append("| (d) ECAPA埋め込み + 対象言語のみの軽量分類器 | (a)の埋め込みを凍結し、FLEURS train少量でja/en/zh/ko/yueのみのロジスティック回帰を学習 | Apache-2.0 (埋め込み) + 自前(分類器) | FLEURS train (ja_jp/en_us/cmn_hans_cn/ko_kr/yue_hant_hk) |")
    lines.append("")

    order = ["whisper-base", "sensevoice", "voxlingua107-ecapa-raw", "voxlingua107-ecapa-target-clf"]
    display_name = {
        "whisper-base": "(b) whisper-base",
        "sensevoice": "(c) SenseVoice内蔵LID(対照)",
        "voxlingua107-ecapa-raw": "(a) speechbrain VoxLingua107-ECAPA (107-way, zhへ丸め)",
        "voxlingua107-ecapa-target-clf": "(d) ECAPA埋め込み + 対象言語(ja/en/zh/ko/yue)専用分類器",
    }
    yue_support = {
        "whisper-base": False,   # OpenAI Whisper has no separate Cantonese token
        "sensevoice": True,      # already the production yue arbiter
        "voxlingua107-ecapa-raw": False,  # VoxLingua107's label set has no yue class either
        "voxlingua107-ecapa-target-clf": True,  # trained with yue as its own class (FLEURS yue_hant_hk)
    }
    has_confidence = {
        "whisper-base": False,   # sherpa-onnx's SpokenLanguageIdentification returns argmax only
        "sensevoice": False,     # tag extraction, no posterior exposed
        "voxlingua107-ecapa-raw": True,   # softmax over 107 classes
        "voxlingua107-ecapa-target-clf": True,  # softmax over 5 classes
    }

    notes = {
        "whisper-base": "sherpa-onnxのSpokenLanguageIdentificationはargmax言語コードのみを返し、"
                        "内部で計算している言語softmaxの生logitsにアクセスできない(信頼度基準を満たせない構造的理由)。",
        "sensevoice": "このモデルはko/yueのASR tierとして本番で常時ロード済み -- LID用途単独で追加される"
                     "モデルサイズはゼロ。基準4(サイズ)を独立評価するのはこの候補では不公平だが、"
                     "基準1・6(クリーン2秒正解率・信頼度)を満たさないため採用可否には影響しない。",
        "voxlingua107-ecapa-raw": "label_encoder.txt(107クラス)を直接確認: 'zh: Chinese'はあるが"
                                  "Cantonese/yueに対応するクラスが存在しない。whisper-tinyと同じ理由"
                                  "(モデルの語彙にyueが無い)で構造的に基準3を満たせない。",
        "voxlingua107-ecapa-target-clf": "FLEURS train (yue_hant_hk含む5言語, 各40クリップ)で学習した"
                                          "ロジスティック回帰をECAPA埋め込みの上に載せた版。yueを学習データの"
                                          "時点で独立クラスとして扱えるため、(a)と同じ埋め込みを使いながら"
                                          "基準3の構造的制約を回避できる唯一の候補。",
    }
    onnx_note = (
        "\n**ONNX化について**: (a)/(d)はscripts/export_voxlingua_lid_onnx.pyでフルパイプライン"
        "(波形→ONNX)のエクスポートを試みたが、speechbrainのFbank前段が使うSTFTがtorch.onnxの"
        "現行エクスポータでcomplex型非対応のため失敗した(`STFT does not currently support complex "
        "types`)。この評価では代わりに.venv-train上のネイティブtorch CPU推論で計測している(onnxruntime"
        "より遅くなる可能性がある保守的な値)。本番組み込みが必要になった場合は、STFTをDFT行列の"
        "matmulで書き換える(complex型を避ける定番の回避策)か、特徴量抽出をnumpyで実装し埋め込み"
        "ネットワークだけをONNX化する経路がある。\n"
    )

    verdicts = []
    for name in order:
        rows = all_rows_by_backend.get(name)
        lines.append(f"## {display_name[name]}\n")
        if not rows:
            lines.append("_未計測 (このスクリプト/scripts/eval_lid_voxlingua.pyをまだ実行していない)_\n")
            continue
        lines += table_for(rows, "正解率 (言語別 x 秒数)")

        clean2 = acc_at(rows, "clean", 2.0)
        babble2 = acc_at(rows, "babble_snr10", 2.0)
        clean2_acc = clean2["wtiny_acc"] if clean2["n"] else None
        babble2_acc = babble2["wtiny_acc"] if babble2["n"] else None
        size_mb = model_size_mb(name)
        if size_mb is None:
            size_mb = next((r.get("model_size_mb") for r in rows if r.get("model_size_mb")), None)
        latency_ms = latency_p50_ms(rows)
        if latency_ms is None:
            latency_candidates = [r.get("latency_p50_ms") for r in rows if r.get("latency_p50_ms")]
            latency_ms = latency_candidates[0] if latency_candidates else None

        lines.append(f"- clean 2秒 正解率: {f'{clean2_acc*100:.0f}%' if clean2_acc is not None else '-'}")
        lines.append(f"- babble_snr10 2秒 正解率: {f'{babble2_acc*100:.0f}%' if babble2_acc is not None else '-'}")
        lines.append(f"- モデルサイズ: {f'{size_mb:.1f}MB' if size_mb is not None else '未計測'}")
        lines.append(f"- 4秒入力レイテンシ(中央値, threads={THREADS}, 暫定・並列実行中の計測): "
                     f"{f'{latency_ms:.0f}ms' if latency_ms is not None else '未計測'}")
        lines.append(f"- yueを独立クラスとして判定: {'可能' if yue_support[name] else '不可(zh等に丸められる)'}")
        lines.append(f"- 信頼度(事後確率)を返せる: {'可能' if has_confidence[name] else '不可'}")
        lines.append("")

        adopted, verdict_text = verdict_for(
            display_name[name], clean2_acc, babble2_acc, yue_support[name],
            size_mb, latency_ms, has_confidence[name])
        lines.append("### 採用判定\n")
        lines.append(verdict_text)
        lines.append("")
        if name in notes:
            lines.append(f"> {notes[name]}\n")
        if name in ("voxlingua107-ecapa-raw", "voxlingua107-ecapa-target-clf"):
            lines.append(onnx_note)
        verdicts.append((display_name[name], adopted))

    lines.append("## 結論\n")
    for name, adopted in verdicts:
        lines.append(f"- {name}: {'**採用**' if adopted else '不採用'}")
    lines.append("")

    adopted_names = [name for name, ok in verdicts if ok]
    all_measured = len(verdicts) == len(order)
    if not all_measured:
        lines.append(
            "_一部の候補が未計測のため、以下の推奨は暫定。scripts/eval_lid_voxlingua.py の完走後に "
            "`--report` で再生成すること。_\n"
        )
    elif adopted_names:
        lines.append(
            f"**推奨**: {', '.join(adopted_names)} が採用基準を全て満たした。scripts/asr_engine.py の "
            f"`RoutedASR(lid_backend=...)` opt-in、`realtime_transcribe.py` の `--lid-backend` フラグとして "
            f"組み込み、既定値(whisper-tiny)は変更しない。最終判断はメイン(オーケストレーター)に委ねる。\n"
        )
    else:
        lines.append(
            "**推奨**: どの候補も採用基準を全て満たさなかった。現行の whisper-tiny + SenseVoice "
            "二重判定(docs/eval/lid.md の推奨ポリシー)を維持する。yueを独立クラスとして判定できる "
            "候補は (c) SenseVoice内蔵LID(対照、既に本番で使用中)と (d) ECAPA+対象言語分類器のみ -- "
            "(d) は基準3をクリアしても他の基準(上表参照)で落ちている場合、今回のFLEURS train 40件/言語 "
            "という小規模データでの学習が精度上限の一因である可能性があり、学習データを増やす追試は "
            "価値があるかもしれない(このevalは3トラック分の候補測定が目的であり、そのハイパーパラメータ "
            "探索までは対象外)。\n"
        )

    lines.append("## 再現コマンド\n")
    lines.append("```")
    lines.append(".venv/Scripts/python scripts/download_models.py --lid-candidates")
    lines.append(".venv/Scripts/python scripts/eval_lid_candidates.py --backend whisper-base")
    lines.append(".venv/Scripts/python scripts/eval_lid_candidates.py --backend sensevoice")
    lines.append(".venv-train/Scripts/python scripts/export_voxlingua_lid_onnx.py")
    lines.append(".venv-train/Scripts/python scripts/eval_lid_voxlingua.py")
    lines.append(".venv/Scripts/python scripts/eval_lid_candidates.py --report")
    lines.append("```\n")

    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {DOCS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=WORKTREE_ROOT)
    ap.add_argument("--backend", choices=list(BACKENDS) + ["all"], default=None)
    ap.add_argument("--report", action="store_true",
                     help="only (re)write docs/eval/lid_candidates.md from existing caches")
    args = ap.parse_args()

    if not args.report:
        names = list(BACKENDS) if args.backend in (None, "all") else [args.backend]
        if not names:
            ap.error("pass --backend or --report")
        for name in names:
            cache = eval_common.load_cache(cache_path(args.root, name))
            run_backend(args.root, name, cache)

    write_report(args.root)


if __name__ == "__main__":
    main()
