"""Noise-robustness scorecard for the production RoutedASR path.

Runs the same "real path" evaluation as scripts/eval_engine.py (LID ->
tier routing -> decode -> ja punctuation) over the clean real-speech eval
sets AND every noisy condition produced by scripts/make_noisyset.py, then
reports per (lang, noise, snr): LID accuracy, mean err, mean RTF, and the
degradation delta versus that language's clean baseline.

Noise conditions (see scripts/make_noisyset.py docstring for exact
definitions): white/pink/babble noise types x SNR in {20,10,5,0} dB,
speech-RMS-relative SNR, seed=42, fully reproducible.

Because the full sweep (13 conditions x ~65 clips) is slow on CPU, results
are checkpointed to a JSON cache (testdata/_noise_eval_cache.json under
--root; testdata/ is already gitignored) so this script can be re-run
multiple times, optionally scoped with --noise/--snr, and it will only
(re)compute conditions missing from the cache. Pass --report to (re)write
docs/NOISE.md from whatever is in the cache without evaluating anything.

GTCRN denoiser A/B (--denoise):
    Pass --denoise to run the SAME condition set through sherpa-onnx's
    OfflineSpeechDenoiser (GTCRN model, models/gtcrn/gtcrn_simple.onnx under
    --root) before handing samples to RoutedASR. Each denoised condition is
    cached under a separate key ("<condition>+denoise") alongside the plain
    result, so both sides of the A/B are available at once for docs/NOISE.md
    to compare. The denoiser's own processing time is measured separately
    (denoise_rtf) and also folded into the row's total rtf (denoise time +
    decode time, over audio duration) so --denoise numbers are directly
    comparable to the non-denoised RTF column.

Usage:
    python scripts/eval_noise.py --root H:\\Programming\\hayamimi
    python scripts/eval_noise.py --root H:\\Programming\\hayamimi --noise white
    python scripts/eval_noise.py --root H:\\Programming\\hayamimi --snr 0 5
    python scripts/eval_noise.py --root H:\\Programming\\hayamimi --report
    python scripts/eval_noise.py --root H:\\Programming\\hayamimi --denoise
    python scripts/eval_noise.py --root H:\\Programming\\hayamimi --denoise --noise babble
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf

from asr_engine import RoutedASR
from eval_accuracy import cer_ja, wer_en
from eval_common import load_manifest
import eval_common

WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(WORKTREE_ROOT, "docs", "NOISE.md")

CLEAN_DIRS = ["testdata/eval_real", "testdata/eval_real_zhko", "testdata/eval_real_yue"]
NOISE_TYPES = ["white", "pink", "babble"]
SNR_LEVELS_DB = [20, 10, 5, 0]
SEED = 42  # must match scripts/make_noisyset.py

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


def cache_path(root: str) -> str:
    return eval_common.cache_path(root, "_noise_eval_cache.json")


def load_cache(root: str) -> dict:
    return eval_common.load_cache(cache_path(root))


def save_cache(root: str, cache: dict):
    eval_common.save_cache(cache_path(root), cache)


def condition_dir(root: str, noise: str, snr) -> str:
    if noise == "clean":
        return None  # clean uses CLEAN_DIRS, handled separately
    return os.path.join(root, "testdata", "eval_noisy", f"{noise}_snr{snr}")


GTCRN_MODEL_REL = os.path.join("models", "gtcrn", "gtcrn_simple.onnx")


def build_denoiser(root: str):
    """GTCRN speech denoiser (sherpa-onnx OfflineSpeechDenoiser). Model
    downloaded from the sherpa-onnx 'speech-enhancement-models' GitHub
    release (MIT, 16kHz, ~0.5MB, RTF ~0.07 per upstream docs)."""
    import sherpa_onnx

    model_path = os.path.join(root, GTCRN_MODEL_REL)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GTCRN model not found at {model_path}. Download it with e.g.:\n"
            f"  curl -L -o {model_path} "
            f"https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            f"speech-enhancement-models/gtcrn_simple.onnx"
        )
    gtcrn_cfg = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=model_path)
    model_cfg = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
        gtcrn=gtcrn_cfg, num_threads=1, debug=False, provider="cpu")
    cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(model=model_cfg)
    return sherpa_onnx.OfflineSpeechDenoiser(cfg)


def eval_manifest_dir(asr: RoutedASR, mdir: str, denoiser=None):
    """Run every clip in mdir/manifest.json through the routed engine,
    return a list of per-clip result rows. If denoiser is given, each clip
    is denoised (GTCRN) first; the denoiser's own time is tracked separately
    as denoise_rtf, and rtf covers denoise+decode combined."""
    entries = load_manifest(mdir)
    rows = []
    prev_lang = None
    for e in entries:
        if e["lang"] != prev_lang:
            # Each language's clip block is an independent recording, not a
            # continuous multi-lingual conversation -- reset the sticky-LID
            # session so the engine's language-switch hysteresis (see
            # RoutedASR.transcribe / resolve_dual_confirm in asr_engine.py --
            # ja/en/zh/ko/yue, all the languages evaluated here, are
            # DUAL_CONFIRM_LANGS) doesn't charge the first clip of a new
            # block for "switching away" from an unrelated previous block's
            # language. Real in-session switch latency is covered separately
            # by tests/test_units.py's resolve_dual_confirm tests.
            asr.reset_session()
            prev_lang = e["lang"]
        wav_path = os.path.join(mdir, e["wav"])
        samples, sr = sf.read(wav_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        dur = len(samples) / sr

        denoise_s = 0.0
        if denoiser is not None:
            t0 = time.perf_counter()
            den = denoiser(samples, sr)
            denoise_s = time.perf_counter() - t0
            samples, sr = den.samples, den.sample_rate

        t0 = time.perf_counter()
        r = asr.transcribe(samples, sr)
        decode_s = time.perf_counter() - t0
        rows.append({
            "wav": e["wav"], "lang": e["lang"], "detected": r["lang"],
            "tier": r["tier"], "err": score(e["lang"], e["ref"], r["text"]),
            "rtf": (denoise_s + decode_s) / dur, "denoise_rtf": denoise_s / dur,
            "dur": dur,
        })
        print(f"  {e['wav']:16} true={e['lang']:3} lid={r['lang']:3} tier={r['tier']:4} "
              f"err={rows[-1]['err']:.3f} rtf={rows[-1]['rtf']:.3f}"
              + (f" denoise_rtf={rows[-1]['denoise_rtf']:.3f}" if denoiser is not None else ""))
    return rows


def condition_key(noise, snr, denoise: bool = False) -> str:
    base = "clean" if noise == "clean" else f"{noise}_snr{snr}"
    return f"{base}+denoise" if denoise else base


def run_conditions(root: str, conditions, cache: dict, asr_holder: list, denoise: bool,
                    denoiser_holder: list):
    for noise, snr in conditions:
        key = condition_key(noise, snr, denoise)
        if key in cache:
            print(f"[{key}] cached, skipping ({len(cache[key])} rows)")
            continue

        if asr_holder[0] is None:
            print("Loading RoutedASR (threads=6)...")
            asr_holder[0] = RoutedASR(threads=6, preload=False)
        asr = asr_holder[0]

        if denoise and denoiser_holder[0] is None:
            print("Loading GTCRN OfflineSpeechDenoiser...")
            denoiser_holder[0] = build_denoiser(root)
        denoiser = denoiser_holder[0] if denoise else None

        print(f"[{key}] evaluating...")
        if noise == "clean":
            rows = []
            for rel in CLEAN_DIRS:
                mdir = os.path.join(root, rel)
                rows.extend(eval_manifest_dir(asr, mdir, denoiser))
        else:
            mdir = condition_dir(root, noise, snr)
            if not os.path.exists(os.path.join(mdir, "manifest.json")):
                print(f"  WARNING: {mdir} missing manifest.json, skipping "
                      f"(run scripts/make_noisyset.py first)")
                continue
            rows = eval_manifest_dir(asr, mdir, denoiser)

        cache[key] = rows
        save_cache(root, cache)
        print(f"[{key}] done, {len(rows)} rows cached")


def aggregate(rows):
    lid_ok = sum(1 for r in rows if r["detected"] == r["lang"])
    err = sum(r["err"] * r["dur"] for r in rows) / sum(r["dur"] for r in rows)
    rtf = sum(r["rtf"] for r in rows) / len(rows)
    agg = {"clips": len(rows), "lid_acc": lid_ok / len(rows), "err": err, "rtf": rtf}
    if rows and "denoise_rtf" in rows[0]:
        agg["denoise_rtf"] = sum(r["denoise_rtf"] for r in rows) / len(rows)
    return agg


def write_report(root: str, cache: dict):
    langs = ("ja", "en", "zh", "ko", "yue")

    clean_by_lang = {}
    if "clean" in cache:
        for lang in langs:
            sub = [r for r in cache["clean"] if r["lang"] == lang]
            if sub:
                clean_by_lang[lang] = aggregate(sub)

    lines = ["# Noise Robustness Scorecard\n",
             "本番経路 (LID→ルーティング→デコード→ja句読点) を、クリーン音声と "
             "white/pink/babble ノイズ (SNR 20/10/5/0dB) を重ねた音声の両方で評価。"
             "metricは en=WER, 他=CER" + ("（yueはt2s正規化）" if _t2s else "（yueは生CER: opencc無し）") + "。\n",
             "## ノイズ生成条件\n",
             f"- SNRは対象音声のRMSパワー基準: `10*log10(P_speech/P_noise) == SNR_dB`\n"
             f"- noise種別: white（ホワイトノイズ）, pink（1/fピンクノイズ）, "
             f"babble（同一プールの別言語クリップを2〜3本ランダムに重ねた話し声ノイズ、"
             f"対象と同言語は使用しない）\n"
             f"- SNRレベル: {', '.join(str(s) for s in SNR_LEVELS_DB)} dB\n"
             f"- 乱数seed: {SEED}（固定・再現可能、`scripts/make_noisyset.py` 参照）\n"
             f"- クリッピング防止: 混合波形のピークが1.0を超える場合は全体をスケールダウン\n"
             f"- 生成スクリプト: `scripts/make_noisyset.py` / 評価スクリプト: `scripts/eval_noise.py`\n",
             "## 結果\n",
             "| lang | condition | clips | LID正解率 | mean err | Δerr vs clean | mean RTF |",
             "|---|---|---|---|---|---|---|"]

    conditions = [("clean", None)] + [(n, s) for n in NOISE_TYPES for s in SNR_LEVELS_DB]
    for lang in langs:
        for noise, snr in conditions:
            key = condition_key(noise, snr)
            if key not in cache:
                continue
            sub = [r for r in cache[key] if r["lang"] == lang]
            if not sub:
                continue
            agg = aggregate(sub)
            cond_label = "clean" if noise == "clean" else f"{noise} snr={snr}dB"
            delta = ""
            if lang in clean_by_lang and noise != "clean":
                delta = f"{agg['err'] - clean_by_lang[lang]['err']:+.3f}"
            lines.append(
                f"| {lang} | {cond_label} | {agg['clips']} | "
                f"{int(round(agg['lid_acc'] * agg['clips']))}/{agg['clips']} | "
                f"{agg['err']:.3f} | {delta} | {agg['rtf']:.3f} |"
            )

    gtcrn_lines, conclusion_stats = build_gtcrn_section(cache, langs, conditions)
    lines.extend(gtcrn_lines)

    os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {DOCS_PATH}")
    return conclusion_stats


def build_gtcrn_section(cache: dict, langs, conditions):
    """Build the GTCRN denoiser A/B section: per (lang, condition) err/LID
    with vs without denoising, plus a data-driven conclusion paragraph.
    Only conditions present in the cache on BOTH sides (plain + "+denoise")
    are shown, since the A/B needs a matched pair."""
    lines = ["\n## GTCRN デノイザ A/B\n",
             "sherpa-onnx 1.13.6 の `OfflineSpeechDenoiser`（GTCRN, `models/gtcrn/gtcrn_simple.onnx`, "
             "MIT, 16kHz）で各クリップを前処理してから本番経路に通した場合の比較。"
             "denoiser RTFはGTCRN単体の処理時間（音声長比）、mean RTFはdenoise+decode合計。\n",
             "| lang | condition | clips | LID (無/有) | err (無) | err (有) | Δerr | denoiser RTF |",
             "|---|---|---|---|---|---|---|---|"]

    noisy_deltas = []      # (err_no_denoise - err_with_denoise) for noisy conditions, + = improvement
    clean_deltas = []      # same, for clean (side-effect check)
    lid_regressions = 0    # count of (lang, condition) where LID acc got worse with denoise
    lid_improvements = 0
    denoise_rtfs = []

    for lang in langs:
        for noise, snr in conditions:
            key_plain = condition_key(noise, snr, denoise=False)
            key_den = condition_key(noise, snr, denoise=True)
            if key_plain not in cache or key_den not in cache:
                continue
            sub_plain = [r for r in cache[key_plain] if r["lang"] == lang]
            sub_den = [r for r in cache[key_den] if r["lang"] == lang]
            if not sub_plain or not sub_den:
                continue
            agg_plain = aggregate(sub_plain)
            agg_den = aggregate(sub_den)
            cond_label = "clean" if noise == "clean" else f"{noise} snr={snr}dB"
            delta_err = agg_plain["err"] - agg_den["err"]  # positive = denoise helped
            n = agg_plain["clips"]
            lid_no = int(round(agg_plain["lid_acc"] * n))
            lid_yes = int(round(agg_den["lid_acc"] * n))
            lines.append(
                f"| {lang} | {cond_label} | {n} | {lid_no}/{n} -> {lid_yes}/{n} | "
                f"{agg_plain['err']:.3f} | {agg_den['err']:.3f} | {delta_err:+.3f} | "
                f"{agg_den.get('denoise_rtf', float('nan')):.3f} |"
            )
            if noise == "clean":
                clean_deltas.append(delta_err)
            else:
                noisy_deltas.append(delta_err)
            if lid_yes < lid_no:
                lid_regressions += 1
            elif lid_yes > lid_no:
                lid_improvements += 1
            if "denoise_rtf" in agg_den:
                denoise_rtfs.append(agg_den["denoise_rtf"])

    stats = {
        "noisy_deltas": noisy_deltas, "clean_deltas": clean_deltas,
        "lid_regressions": lid_regressions, "lid_improvements": lid_improvements,
        "denoise_rtfs": denoise_rtfs,
    }

    lines.append("\n### 結論\n")
    lines.append(write_conclusion(stats))
    return lines, stats


def write_conclusion(stats: dict) -> str:
    noisy = stats["noisy_deltas"]
    clean = stats["clean_deltas"]
    if not noisy and not clean:
        return ("A/Bペアが揃っていないため結論なし（--denoise で全条件を評価してから "
                "--report を実行してください）。\n")

    mean_noisy = sum(noisy) / len(noisy) if noisy else float("nan")
    improved = sum(1 for d in noisy if d > 0.005)
    worsened = sum(1 for d in noisy if d < -0.005)
    flat = len(noisy) - improved - worsened
    mean_clean = sum(clean) / len(clean) if clean else float("nan")
    mean_denoise_rtf = (sum(stats["denoise_rtfs"]) / len(stats["denoise_rtfs"])
                         if stats["denoise_rtfs"] else float("nan"))

    parts = [
        f"ノイズ条件 {len(noisy)}件中、denoiseでerrが改善したのは{improved}件、"
        f"悪化したのは{worsened}件、ほぼ変化なしが{flat}件（平均Δerr={mean_noisy:+.3f}、"
        f"正=改善）。",
        f"クリーン音声側の副作用: 平均Δerr={mean_clean:+.3f}"
        + ("（悪化＝デノイザが健全な音声に悪影響）" if mean_clean < -0.005
           else "（実質的な悪化なし）" if not (mean_clean != mean_clean) else "") + "。",
        f"LIDは{stats['lid_regressions']}件で悪化、{stats['lid_improvements']}件で改善。",
        f"GTCRN自体のRTFは平均{mean_denoise_rtf:.3f}と軽量。",
    ]

    if worsened > improved and mean_clean < -0.005:
        verdict = ("**不採用が妥当**: ノイズ条件でも改善より悪化が多く、クリーン音声にも悪影響が出ている。"
                    "前処理として常時オンにする根拠がない。")
    elif worsened > improved:
        verdict = ("**不採用寄り**: クリーン音声への悪影響は限定的だが、ノイズ条件でも悪化が改善を上回っており、"
                    "常時オンにする価値は薄い。SNR推定などで悪条件に限定しても収支はプラスにならない可能性が高い。")
    elif mean_clean < -0.005:
        verdict = ("**条件付き採用**: ノイズ条件では改善が優勢だが、クリーン音声を悪化させるため常時オンは避け、"
                    "SNR推定などで低SNR時のみデノイザを通す条件付き運用が妥当。")
    else:
        verdict = ("**採用が妥当**: ノイズ条件で改善が優勢かつクリーン音声への悪影響もほぼ無いため、"
                    "前処理として常時オンにしてよい。")

    return " ".join(parts) + "\n\n" + verdict + "\n"


def parse_conditions(noise_filter, snr_filter):
    noises = noise_filter if noise_filter else ["clean"] + NOISE_TYPES
    snrs = snr_filter if snr_filter else SNR_LEVELS_DB
    out = []
    if "clean" in noises:
        out.append(("clean", None))
    for n in noises:
        if n == "clean":
            continue
        for s in snrs:
            out.append((n, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--root", default=default_root,
                     help="repo root containing testdata/ (default: derived from script location)")
    ap.add_argument("--noise", nargs="*", choices=["clean"] + NOISE_TYPES,
                     help="restrict to these noise types (default: all incl. clean)")
    ap.add_argument("--snr", nargs="*", type=int, choices=SNR_LEVELS_DB,
                     help="restrict to these SNR levels in dB (default: all)")
    ap.add_argument("--report", action="store_true",
                     help="only (re)write docs/NOISE.md from the existing cache, no evaluation")
    ap.add_argument("--denoise", action="store_true",
                     help="run the selected conditions through the GTCRN speech denoiser "
                          "before RoutedASR (results cached under separate '+denoise' keys)")
    args = ap.parse_args()

    cache = load_cache(args.root)

    if not args.report:
        conditions = parse_conditions(args.noise, args.snr)
        asr_holder = [None]
        denoiser_holder = [None]
        run_conditions(args.root, conditions, cache, asr_holder, args.denoise, denoiser_holder)

    write_report(args.root, cache)


if __name__ == "__main__":
    main()
