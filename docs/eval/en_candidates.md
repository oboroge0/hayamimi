# en_candidates.md — 英語経路を Parakeet v2 系に振り替える評価(改善トラックE)

現行の en 経路は多言語 Parakeet TDT 0.6B v3 int8
(`models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`)。v3 は en に加えて他 24 の
V3_LANGS(欧州言語)も担当しているため、en 専用モデルに切り替えれば欧州言語側の
精度を犠牲にせずに en だけ底上げできる可能性がある、というのがこのトラックの狙い。

**結論を先に書く: 採用。** en-only候補 `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`
(以下 v2)が採用基準を全て満たした。`scripts/asr_engine.py` に opt-in の en 専用
tier として実装済み(既定は現状維持で v3 のまま、`--en-tier v2` / `en_tier="v2"`
で切り替え)。

## 候補

| 候補 | 由来 | 公開時期 | サイズ(int8) |
|---|---|---|---|
| v3(対照、現行本番) | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 2025-08 | 487MB (tar.bz2) |
| **v2(採用)** | [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | 2025-08 | 482MB (tar.bz2) |
| unified(参考、不採用) | [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | 2026-04 | 501MB (tar.bz2、non-streaming int8) |

3つとも k2-fsa/sherpa-onnx の `asr-models` GitHub Release から取得した int8
tarball(NeMo RNNT/TDT transducer、encoder/decoder/joiner の3ファイル構成)。
sherpa-onnx 側の呼び出しは全く同じ形(`OfflineRecognizer.from_transducer(...,
model_type="nemo_transducer")`)で、モデルディレクトリが違うだけ。

v2 の tarball 名は `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2`
(v2-fp16 版も存在するが CPU int8 運用の方針に合わないため対象外)。unified は
`sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2`
(2025〜2026年公開の英語専用モデルの追加候補として選定。ストリーミング版もある
が今回の評価はオフライン decode のみなので non-streaming を使用)。

### ライセンス

| 候補 | ライセンス | 再配布 |
|---|---|---|
| v3 | CC-BY-4.0 | 可(既存 `THIRD_PARTY_NOTICES.md` に記載済み) |
| **v2** | **CC-BY-4.0** | 可(v3と同じ、追加の帰属表示ファイル不要) |
| unified | NVIDIA Open Model License Agreement | 可(改変物含め再配布可。ただし帰属表示として "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" を記載した Notice ファイルの同梱が必要 — v3/v2のCC-BY-4.0より一手間多い) |

v2 は v3 と同じ CC-BY-4.0 なので、`THIRD_PARTY_NOTICES.md` に一行追加するだけで
既存の運用パターンをそのまま踏襲できる。unified はライセンス自体は再配布可能だが、
別ライセンス体系(Notice ファイル要件)が増えることが不採用の一因。

## 評価方法

- `scripts/eval_en_candidates.py`(新規)。`scripts/eval_accuracy.py` の
  `wer_en` をそのまま再利用し、3候補を同一の `from_transducer` 呼び出しで
  ビルドして同一クリップ集合を decode する。
- データセット:
  - **FLEURS en test 100本**(`testdata/fleurs_bench/en`、`docs/results/benchmarks.md`
    2026-09-01節と同一セット)
  - **実音声 en 15本**(`testdata/eval_real`、LibriSpeech dev-clean、
    `docs/results/scorecard.md` の en 行と同一セット)
- **スレッド数 2**(`--threads 2`、既定値)。他5トラックが同じ6コアCPUで並列
  評価中のため、RTFは**暫定値(並列実行中の計測)**。
- 採点前の正規化は `eval_accuracy.normalize_en`(小文字化・NFKC・句読点除去・
  空白圧縮)+ jiwer.wer。v3 の結果が `docs/results/benchmarks.md` の既存値
  (FLEURS en WER 0.1004)と完全一致することを確認済み — ハーネスの検算はここで
  取れている。

## 結果

### FLEURS en test, 100本

| 候補 | WER | mean RTF(暫定) |
|---|---|---|
| v3(対照) | 0.1004 | 0.141 |
| **v2** | **0.0664** | 0.099 |
| unified | 0.0726 | 0.089 |

### 実音声 en(LibriSpeech dev-clean), 15本

| 候補 | WER | mean RTF(暫定) |
|---|---|---|
| v3(対照) | 0.0226 | 0.089 |
| **v2** | **0.0133** | 0.097 |
| unified | 0.0071 | 0.098 |

v2・unified とも FLEURS/実音声の両方で v3 を明確に上回った。特に FLEURS の
固有名詞("Goethe, Fichte" — v3/unifiedは "Gose/Goth, Fischt/Ficht" と誤る場面が
あったが v2 は正しく書き起こした)で v2 が良い結果を出すクリップが複数見られた。

### 出力例(大文字小文字・句読点の維持を確認)

| クリップ | 参照 | v3 | v2 | unified |
|---|---|---|---|---|
| fleurs en_090 | (Easter vigil の記述) | `...often hold in feast of vigil on Saturday night...` | `...often hold an Easter vital on Saturday night...` | `...often hold an Easter vigil on Saturday night...` |
| eval_real en_14 | "At one thirty he went to Rector's for lunch..." | `At one thirty he went to Rectors for lunch, and when he returned a messenger was waiting for him.` | `At one thirty he went to Rectors for lunch, and when he returned a messenger was waiting for him.` | `At one thirty he went to Rector's for lunch, and when he returned a messenger was waiting for him.` |

3候補とも大文字化・カンマ・ピリオドが安定して出力されている(v3と同じ品質)。
アポストロフィの再現(`Rector's`)は unified が一番良く、v2/v3は"Rectors"と
アポストロフィを落とす傾向がわずかにある — が WER には影響しない程度。

### メモリ(v3と同時常駐時の増分)

`scripts/eval_en_candidates.py --memory` で計測(モデルのみロード、
`RoutedASR`本体の他tierは含まない単純計測)。

| 状態 | RSS | 差分 |
|---|---|---|
| プロセス起動直後 | 33.0 MB | - |
| v3 ロード+ウォームアップ | 735.4 MB | +702.3 MB (対起動時) |
| v3 + v2 両方ロード+ウォームアップ | 1401.5 MB | **+666.1 MB (v2の限界増分)** / 総計+1368.4 MB |
| v3 + unified 両方ロード+ウォームアップ | 1411.9 MB | +676.5 MB (unifiedの限界増分) / 総計+1378.9 MB |

v2・unifiedともほぼ同じメモリコスト(v2がわずかに軽い)。v3+v2を両方常駐させても
2GB(`--max-resident`のLRU想定枠)以内に収まる。ただしこれは実行中モデル本体
のみの測定で、本番の`RoutedASR`(whisper-tiny LID・他tierも含む)にv2を足すと
実際の増分はこれより大きい可能性がある点に注意。

## 採用基準チェック

| # | 基準 | v2 | unified |
|---|---|---|---|
| 1 | FLEURS en WER ≤ 8.0% | ✅ 6.64% | ✅ 7.26% |
| 2 | 実音声 en WER ≤ 2.3%(退行なし) | ✅ 1.33% | ✅ 0.71% |
| 3 | 大文字小文字・句読点付き出力の維持 | ✅ | ✅ |
| 4 | RTF < 0.2(暫定値) | ✅ 0.10(fleurs)/0.10(real) | ✅ 0.09/0.10 |
| 5 | ライセンス再配布可能 | ✅ CC-BY-4.0 | ✅ NVIDIA Open Model License(Notice要) |
| 6 | v3併存時の常駐メモリ増分が2GB以内 | ✅ 合計1.4GB | ✅ 合計1.4GB |

両候補とも全基準を満たした。

## 採用判断: v2

v2 を採用し、unified は評価のみで見送った。理由:

1. **FLEURS(100本、統計的に安定した方のセット)でv2の方がWERが低い**
   (6.64% vs 7.26%)。実音声(15本)ではunifiedがやや良いが、サンプル数が
   少なく差が小さい(0.71% vs 1.33%、絶対差0.6pt)。
2. **ライセンスがv3と同じCC-BY-4.0**で、`THIRD_PARTY_NOTICES.md`の既存パターン
   をそのまま踏襲できる。unifiedのNVIDIA Open Model License Agreementは
   再配布可能だが帰属表示ファイル(Notice)の同梱義務が別途あり、運用上の
   手間が一つ増える。
3. メモリコストはほぼ同等(v2がわずかに軽い)。
4. トラック指示で名指しされている主候補であり、"存在すれば追加"扱いの
   unifiedより優先度が高い。

unified は 2025〜2026年公開の英語専用モデルという条件を満たす良い候補ではある
ため、将来 real-speech 側の精度をより重視する判断になった場合の代替として
記録だけ残す(コードへの組み込みはしていない)。

## 実装

`scripts/asr_engine.py`:

- `V2_MODEL_DIR` 定数、`_build_v2_recognizer()`(v3と同じ`from_transducer`呼び出し、
  モデルディレクトリのみ違う)。
- `_KEY_FILES["v2"]` / `_BUILDERS["v2"]` に登録(既存の流儀通り)。
- `_PRELOAD_ORDER` には**追加しない**(`pja`と同様、opt-inモデルは既定では
  プリロードしない — 使わないセッションが無駄にロード時間を払わないため)。
- `EN_TIERS = ("v3", "v2")` と純粋関数 `resolve_en_tier()` を追加(モデル
  ロードなしでユニットテスト可能)。
- `RoutedASR.__init__(..., en_tier: str = "v3", ...)` を追加。既定は`"v3"`
  (現状維持)。
- `RoutedASR._route()`: `lang == "en" and self._en_tier == "v2"` の場合のみ
  `v2` にルーティング。`V3_LANGS`自体は変更していないので、`en_tier`を
  指定しない全ての既存呼び出し元は今まで通り`en`→`v3`のまま。
- モデルが未ダウンロードの場合は`v2`→`v3`の順で後退する
  (`_get_with_fallback("v2", prefer=("v3",))`)。既定チェーン
  (`rz`→`sv`→`v3`→…)にそのまま乗せると先頭の`rz`(ReazonSpeech、英語は
  大文字・句読点なし)に落ちてしまうため、`v3`を明示的に先に試す。
  `model_fallback`イベント(requested=`v2`, used=`v3`)が1回だけ出て、
  プロセスは落ちない(tests/test_units.py の回帰テストで固定)。

`scripts/realtime_transcribe.py`:

- `--en-tier {v3,v2}`(既定 `v3`)を追加し、`RoutedASR(en_tier=args.en_tier)`
  に渡すだけ。

`scripts/download_models.py`:

- `--en-parakeet-v2`(opt-in、既定では落とさない)で
  `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2`を取得。`--minimal`
  実行後には効かない(`--eval-baselines`と同じ配置)。

`THIRD_PARTY_NOTICES.md`:

- v2の行を追加(CC-BY-4.0、v3と同じ表内)。

`tests/test_units.py`:

- `test_resolve_en_tier_accepts_known_values` / `_rejects_unknown_value`
- `test_route_en_stays_on_v3_by_default`
- `test_route_en_switches_to_v2_when_opted_in`
- `test_route_other_v3_langs_unaffected_by_en_tier`(en以外のV3_LANGS言語は
  en_tier="v2"でもv3のまま、という回帰防止)

全て既存の流儀(`RoutedASR`をインスタンス化せず、必要な属性だけ持つstubに
対して`unbound method`を呼ぶ)に合わせた、モデルロードなしの高速ユニット
テスト。

## 再現手順

```bash
# v2 (採用): download_models.py に組み込み済み
python scripts/download_models.py --en-parakeet-v2

# unified (参考、コードには組み込んでいない): 手動取得
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming.tar.bz2
# を models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming/ に展開

# 精度・RTF計測
python scripts/eval_en_candidates.py --set fleurs --candidates v3,v2,unified
python scripts/eval_en_candidates.py --set real --candidates v3,v2,unified

# メモリ計測
python scripts/eval_en_candidates.py --memory --candidates v3,v2,unified

# 本番経路でv2を試す
python scripts/realtime_transcribe.py --en-tier v2 --wav testdata/eval_real/en_01.wav
```

結果JSONは `testdata/eval_en_candidates/results_{fleurs,real}.json`
(git非追跡)に随時チェックポイントされる。再実行は既に採点済みのクリップを
スキップする(resumable)。

## 制約

- `--max-resident` の枠を1つ余計に消費する: 既定では en と欧州24言語が v3 を共有するが、`--en-tier v2` では v2 と v3 の2認識器になる。上限が小さい設定（例 `--max-resident 1`）で英語と仏語などを行き来すると、v2/v3 が互いを追い出して毎回モデル再ロード（数秒）が走る。`docs/guide/tuning.md` の `--max-resident` 行に併記した。

- FLEURS/実音声ともread-aloud寄りの音声で、雑音・話者交代・オーバーラップは
  含まない(`docs/results/benchmarks.md`と同じ制約)。
- RTFは他5トラックと同一CPUを共有した並列実行下の暫定値。専有環境での
  再計測は未実施。
- メモリ計測は「モデル単体をロードしてウォームアップした場合」の値。本番の
  `RoutedASR`が保持する他tier(rz/whisper-tiny LID等)込みの実測ではない。
- v2採用後もv3は引き続き他24のV3_LANGS(欧州言語)を担当するため、v3自体は
  削除・置換していない(en_tierという分岐を追加しただけ)。
