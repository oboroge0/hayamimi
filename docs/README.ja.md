# hayamimi ドキュメント索引

English: [README.md](README.md)

`docs/`以下は、書かれた時期ではなく「何のために読むか」で分けてあります。
各項目には、その文書が何か、何語で書かれているか、そして**結論**（最新の状態に
保つもの）なのか**日付つきの記録**（ある実行のスナップショットで、後から
書き換えないもの）なのかを書いています。

**言語方針。** 英語を主、日本語を併記とし、`README.md` + `README.ja.md`の形を
とります。この方針より前に日本語で書かれた文書は日本語のままで、下の表では`ja`と
記しています。古いという意味ではなく、単に未翻訳です。新しい文書は英語で書き、
日本語の読み手に必要なものには`.ja.md`を添えてください。

## 使う

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`../README.md`](../README.md) / [`../README.ja.md`](../README.ja.md) | en / ja | 結論 | hayamimiとは何か、クイックスタート、CLIリファレンス、実測性能、既知の制限。 |
| [`../mobile/README.md`](../mobile/README.md) | en | 結論 | デモアプリ`hayamimi_mobile`とその動かし方。 |

## 組み込む

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`guide/embedding.md`](guide/embedding.md) / [`guide/embedding.ja.md`](guide/embedding.ja.md) | en / ja | 結論 | Pythonエンジンを自分のアプリから動かす方法。importできる部品、構造化イベントの一覧、`POST /config`、`POST /reset`。 |
| [`../mobile/hayamimi_core/README.md`](../mobile/hayamimi_core/README.md) / [`../mobile/hayamimi_core/README.ja.md`](../mobile/hayamimi_core/README.ja.md) | en / ja | 結論 | Flutter/Dartパッケージのガイド。初期化、モデル配置、実行時設定、スレッドモデル、プラットフォーム状況。 |
| [`spec/ja_pipeline.md`](spec/ja_pipeline.md) / [`spec/ja_pipeline.ja.md`](spec/ja_pipeline.ja.md) | en / ja | 結論 | 他言語で再実装できる粒度まで書いた日本語経路の仕様。 |
| [`spec/ja_pipeline_spec.json`](spec/ja_pipeline_spec.json) | — | 結論 | 同じ設定の機械可読版。`scripts/dump_ja_config.py`が生成し、`tests/test_ja_pipeline_spec.py`が差分を見るので、仕様とコードが黙って乖離しない。 |

## 調整する

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`guide/tuning.md`](guide/tuning.md) / [`guide/tuning.ja.md`](guide/tuning.ja.md) | en / ja | 結論 | 両実装のユーザー向けつまみ全部。既定値、変更する場所、その既定値を決めた記録へのリンク。意図的に公開していない内部タイムアウトも併記。 |

## 現在の数字

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`results/scorecard.md`](results/scorecard.md) | ja | 結論 | 本番経路の実音声エンドツーエンド精度。言語ごとに、使ったデータと採点条件つき。 |
| [`results/comparison.md`](results/comparison.md) | ja | 結論 | 既存モデル・サービスとの比較。同一音声での直接対決と、公表値との比較を分けて記載。 |
| [`results/benchmarks.md`](results/benchmarks.md) | ja | 日付つき記録 | 約30件の改善イテレーション記録。何を試し、何が測れ、採用したかどうか。このプロジェクトの設計史そのもので、不採用の結果も残っている。 |

## 実験の記録

特定の実行のスナップショットです。コードが先に進んでも更新しません。当時どう
測って何が出たかを読むためのものです。

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`eval/eval.md`](eval/eval.md) | en | 日付つき記録 | TTS合成のja/enセットでのParakeet対faster-whisper large-v3-turbo。最初の精度計測で、下の実音声版に取って代わられた。 |
| [`eval/eval_real.md`](eval/eval_real.md) | en | 日付つき記録 | 同じ比較を実際の人間の発話（ja 15本 + en 15本）で。TTSセットが誤導的だと分かった回。 |
| [`eval/eval_real_zhko.md`](eval/eval_real_zhko.md) | en | 日付つき記録 | 実音声のzh + ko。SenseVoiceを言語専用モデルに置き換える価値があるか。 |
| [`eval/eval_real_yue.md`](eval/eval_real_yue.md) | en | 日付つき記録 | 同じ問いを広東語で。 |
| [`eval/noise.md`](eval/noise.md) | ja | 日付つき記録 | white/pink/babbleノイズ SNR 20/10/5/0dB下での本番経路。ここから二重LID確認の方針が出た。 |
| [`eval/singing.md`](eval/singing.md) | ja | 日付つき記録 | 歌唱音声（ja/ko/en）。同一歌唱者の朗読版を対照に置いている。 |
| [`eval/lid.md`](eval/lid.md) | ja | 日付つき記録 | セグメント長に対するLID正解率。whisper-tiny単独とSenseVoice内蔵LIDを、クリーンとノイズ条件の両方で。切替確認ポリシーの根拠。 |
| [`eval/video_test.md`](eval/video_test.md) | ja | 日付つき記録 | 実YouTube動画6本をフルパイプラインに通した2026-08-24の記録。ノイズ抑制の作業はここから始まった。 |
| [`eval/head_dropout.md`](eval/head_dropout.md) | ja | 日付つき記録 | 発話冒頭がどれだけ落ちるかの実測。プリロールと「疑わしいときだけ再試行」の根拠。 |
| [`eval/head_dropout_results.json`](eval/head_dropout_results.json) | — | 日付つき記録 | その計測のクリップ単位の生出力。`scripts/eval_head_dropout.py`が書き出す。 |

## 設計の経緯

ある判断に至るまでの調査です。判断しながら書いているので長く、選ばなかった案も
そのまま残っています。

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`design/goals.md`](design/goals.md) | ja | 結論 | このプロジェクトが目指すもの（2026-08-23合意）。目標レイテンシ、目標精度、スコープの境界。 |
| [`design/diarization.md`](design/diarization.md) | ja | 日付つき記録 | 完全な話者分離の調査。AMIに対する9ラウンドのチューニングで約3,000行。多くのラウンドは「実測のうえ不採用」で終わり、その理由も書いてある。 |
| [`design/mobile_quantization.md`](design/mobile_quantization.md) | en | 日付つき記録 | スマホ向けjaモデルのINT8/fp16量子化とサイズ計測、iPhone実機の測定、句読点モデルのfp32据え置きを決めた回帰。 |
| [`design/punct_ja.md`](design/punct_ja.md) | en | 結論 | 日本語句読点復元。モデル、上流からの意図的な逸脱、既知の限界。 |
| [`design/translate.md`](design/translate.md) | en | 日付つき記録 | FuguMTによるja→en字幕翻訳と、実測した失敗例。 |
| [`design/translate_m2m.md`](design/translate_m2m.md) | en | 日付つき記録 | M2M-100によるja→zh/ko/esほかと、品質の上限がどこにあるか。 |

## 検証手順

実機で確かめる手順と、前回やったときに何が起きたか。

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`verify/ios.md`](verify/ios.md) | ja | 結論 | Macから`mobile/`をiPhone 15実機に入れて確認する手順。 |
| [`verify/android_emulator.md`](verify/android_emulator.md) | en | 日付つき記録 | デコードワーカーとDart句読点を確かめた2026-09-01のAndroidエミュレータ3回分。3件のセグメンテーション欠陥もここで出た。旧`docs/MOBILE.md`から分離。 |

## 初期リサーチ（2026-08-23時点）

コードを書く前に方向を決めるために書いたものです。エコシステムの現状説明としてでは
なく、当時の判断の理由として残しています。中の数値は他プロジェクトからの引用で、
ここで実測したものではありません。

| 文書 | 言語 | 種別 | 内容 |
|---|---|---|---|
| [`research/00-summary.md`](research/00-summary.md) | ja | 日付つき記録 | 下の3本から出した結論。 |
| [`research/01-whisper-ecosystem.md`](research/01-whisper-ecosystem.md) | ja | 日付つき記録 | 2026年8月時点のWhisperと派生実装。 |
| [`research/02-optimization-techniques.md`](research/02-optimization-techniques.md) | ja | 日付つき記録 | Whisper系ASRの高速化技術サーベイ。 |
| [`research/03-competitor-asr.md`](research/03-competitor-asr.md) | ja | 日付つき記録 | 非Whisper系ASRモデルの速度と精度。 |

## 旧パス

2026-09-03に全文書を移動しました。古いリンクを持っている場合の対応表です。

| 旧パス | 新パス |
|---|---|
| `docs/BENCHMARKS.md` | [`docs/results/benchmarks.md`](results/benchmarks.md) |
| `docs/COMPARISON.md` | [`docs/results/comparison.md`](results/comparison.md) |
| `docs/SCORECARD.md` | [`docs/results/scorecard.md`](results/scorecard.md) |
| `docs/EVAL.md` | [`docs/eval/eval.md`](eval/eval.md) |
| `docs/EVAL_REAL.md` | [`docs/eval/eval_real.md`](eval/eval_real.md) |
| `docs/EVAL_REAL_ZHKO.md` | [`docs/eval/eval_real_zhko.md`](eval/eval_real_zhko.md) |
| `docs/EVAL_REAL_YUE.md` | [`docs/eval/eval_real_yue.md`](eval/eval_real_yue.md) |
| `docs/NOISE.md` | [`docs/eval/noise.md`](eval/noise.md) |
| `docs/SINGING.md` | [`docs/eval/singing.md`](eval/singing.md) |
| `docs/LID.md` | [`docs/eval/lid.md`](eval/lid.md) |
| `docs/VIDEO_TEST.md` | [`docs/eval/video_test.md`](eval/video_test.md) |
| `docs/HEAD_DROPOUT.md` | [`docs/eval/head_dropout.md`](eval/head_dropout.md) |
| `docs/GOALS.md` | [`docs/design/goals.md`](design/goals.md) |
| `docs/DIARIZATION_PLAN.md` | [`docs/design/diarization.md`](design/diarization.md) |
| `docs/PUNCT_JA.md` | [`docs/design/punct_ja.md`](design/punct_ja.md) |
| `docs/TRANSLATE.md` | [`docs/design/translate.md`](design/translate.md) |
| `docs/TRANSLATE_M2M.md` | [`docs/design/translate_m2m.md`](design/translate_m2m.md) |
| `docs/MOBILE.md` | 分割: [`docs/design/mobile_quantization.md`](design/mobile_quantization.md) + [`docs/verify/android_emulator.md`](verify/android_emulator.md) |
| `docs/IOS_VERIFY.md` | [`docs/verify/ios.md`](verify/ios.md) |
| `docs/JA_PIPELINE.md` | [`docs/spec/ja_pipeline.md`](spec/ja_pipeline.md) |
| `docs/ja_pipeline_spec.json` | [`docs/spec/ja_pipeline_spec.json`](spec/ja_pipeline_spec.json) |
| `README.md`の「Embedding in another app」「Embedding: runtime control and structured events」 | [`docs/guide/embedding.md`](guide/embedding.md) |
| `README.ja.md`の「他アプリへの組み込み」「組み込み: 実行時制御と構造化イベント」 | [`docs/guide/embedding.ja.md`](guide/embedding.ja.md) |

`docs/research/`と`docs/images/`は移動していません。
