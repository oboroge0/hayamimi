# hayamimi (早耳)

[![tests](https://github.com/oboroge0/hayamimi/actions/workflows/test.yml/badge.svg)](https://github.com/oboroge0/hayamimi/actions/workflows/test.yml) [![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/oboroge0/hayamimi)](https://github.com/oboroge0/hayamimi/releases)

**CPUだけで動くリアルタイム多言語音声認識。** GPUもクラウドAPIも使わず、メモリ2GB未満で、
ライブ字幕からブラウザ表示、話者ラベル、翻訳字幕まで動きます。

English README is at [README.md](README.md).

名前の「早耳」は、情報を聞きつけるのが早い人のこと。このツールも耳が早く、
話している最中から字幕が出て、話し終えると**約100ms**で確定します。

## なぜ作ったか

CPUだけでリアルタイム音声認識をやろうとすると、普通はWhisperのような汎用モデルを1つ選び、
その精度で我慢することになります。hayamimiは発想を変えて、発話ごとに言語を判定し、
その言語がいちばん得意なモデルに振り分けます。モデルはすべてINT8量子化のONNX形式で、
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)の上で動くため、PyTorchもCUDAも要りません。

実際のテレビ放送の日本語音声で測ると、この方式で**CER 5.8%**が出ます（`docs/SCORECARD.md`）。
同じ音声で`whisper-large-v3-turbo`は13.8%なので、誤りは半分以下。速度は6コアの
デスクトップCPUで実時間の10〜50倍です。

## 機能

| 機能 | 内容 |
|---|---|
| 5ルート言語カタログ | 日/中/韓/広東語/英+欧州24言語をそれぞれ最適な専用モデルへ。それ以外（約1600言語）はMeta Omnilingual ASRにフォールバック |
| 速報字幕 | 発話中に約0.5秒ごとにドラフトテキストが更新される |
| 高速確定 | 話し終えてから約100msで確定文が出る（日本語。他言語は`docs/GOALS.md`参照） |
| 二段パス補正 | 2秒の無音後、直近の発話群を一括再デコードし高精度な「清書」トランスクリプトを生成（実放送ja CER 15.5%→12.0%） |
| 話者ラベル | `--speakers`でCAM++話者埋め込みを使いS1/S2/...とタグ付け（ターンテイキング方式、フル話者分離ではない） |
| 翻訳 | `--translate en,zh,ko,es,...`で日本語行をリアルタイム翻訳（en=FuguMT。それ以外はモデルの語彙が対応していれば任意のM2M-100ターゲット言語コードを受け付ける。zh/ko/esは品質を実測済み、詳細はdocs/TRANSLATE_M2M.md） |
| ホットワード/置換辞書 | `--hotwords`で固有名詞認識を強化（現状jaルートには効果なし。既知の制限を参照）、`--replace`は事後の検索置換でどのルートでも有効 |
| OBSオーバーレイ+ダッシュボード | `--serve`でローカルHTTPサーバーを起動、ブラウザソースオーバーレイとライブダッシュボードを提供 |
| ネットワーク音声入力 | `--input ws`でWebSocket経由のマイク音声（スマホ、ESP32/スタックチャン等）を受け付け、`--serve`のダッシュボード/オーバーレイにもそのまま流れる |
| メモリ上限管理 | LRUモデル退避で常駐モデルを上限内（既定<2GB）に制御 |
| CPUのみ | すべてのモデルがsherpa-onnx経由の量子化ONNXとして動作。GPU・PyTorch不要 |

## デモUI

`--serve`を付けるとローカルサーバーが立ち、ブラウザから3つのページを開けます。

- **`http://localhost:8833/dashboard`**: ライブダッシュボード。発話中のテキスト、
  確定した行（言語バッジと話者、行ごとの応答時間つき）、その下に翻訳、右側に清書版の
  トランスクリプトが流れます。
- **`http://localhost:8833/`**: OBS用のシンプルなオーバーレイ。OBSのブラウザソースに
  このURLを入れると配信画面に字幕が載ります。確定行と発話中の行は別の行として
  表示され、`?show=final` / `?show=partial` を付けるとどちらか片方だけになるので、
  OBS上で別ソースとして自由に配置できます。
- **`http://localhost:8833/transcript`**: 清書トランスクリプトだけを流すページ。

![dashboard](docs/images/dashboard.png)

🎬 **[デモ動画を見る](https://github.com/oboroge0/hayamimi/releases/download/v0.1.0/hayamimi_demo.mp4)**: 実際の4言語音声（日英韓中）を文字起こししたときの記録を、そのまま再生した動画です。

## ネットワーク音声入力

`--input ws`を付けると、ローカルマイクの代わりにWebSocket経由で音声を受け付ける
モードになります。スマホやESP32ベースのスタックチャンからマイク音声をLAN越しに
送って、hayamimiの通常パイプラインでそのまま文字起こしできます。

```bash
.venv/Scripts/python scripts/realtime_transcribe.py --input ws --serve
# -> ws://<host>:8766/ingest が音声を受け付け、http://localhost:8833/dashboard に結果が出る
```

プロトコル: `/ingest`に接続し、最初にJSONテキストフレームを1通送ります
（`{"sr": 16000, "format": "pcm_s16le", "channels": 1}`）。以降は生の
`pcm_s16le`音声をバイナリフレームで連続送信します。16kHz以外はサーバー側で
リサンプリングし、ダッシュボードのSSEストリームと同じpartial/final/translation/refine
のJSONイベントを返すので、クライアント側で独自に字幕表示することもできます。
音声を送るクライアントは同時に1本だけ受け付けます。`scripts/ws_mic_client.py`は
外部ライブラリ不要の参照クライアント（wavファイルを実時間ペースで送信）で、
スマホ/ESP32クライアントを作るときのテンプレートにもなります。

## 動作環境

Python 3.10以上と、PATHの通ったffmpegが必要です。開発と検証は**Windows 11**で行いました。
使っているランタイムはどれもクロスプラットフォームなのでmacOS/Linuxでも動くはずですが、
実機での通し確認はまだできていません。動いた・動かなかったの報告を歓迎します。

## クイックスタート

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py

# macOS / Linux
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py

# マイクからリアルタイム認識
.venv/Scripts/python scripts/realtime_transcribe.py     # Windows
.venv/bin/python scripts/realtime_transcribe.py          # macOS/Linux

# ダッシュボード + OBSオーバーレイ付き
.venv/Scripts/python scripts/realtime_transcribe.py --serve
# -> ブラウザで http://localhost:8833/dashboard を開く
```

`scripts/download_models.py`は約3.1GB分のモデルを`models/`（git管理外）にダウンロードします。
`--minimal`を付けると日本語と英語だけの約1.1GB構成になります。
各モデルのライセンスは`THIRD_PARTY_NOTICES.md`にまとめてあります。

## CLIリファレンス

フラグはすべて`scripts/realtime_transcribe.py`のものです。

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--wav PATH` | マイク入力 | マイクの代わりに16kHzモノラルWAVファイルからのストリーミングをシミュレート |
| `--no-realtime` | オフ | `--wav`使用時、チャンク間でスリープしない（高速バッチ処理） |
| `--input {mic,wav,ws}` | mic、`--wav`指定時はwav | 音声入力元。`ws`はネットワーク経由で音声を受け付ける（上記参照） |
| `--ws-host HOST` | `0.0.0.0` | `--input ws`の`/ingest`エンドポイントのバインドホスト |
| `--ws-port PORT` | 8766 | `--input ws`の`/ingest`エンドポイントのポート |
| `--threads N` | 4 | モデルごとの推論スレッド数 |
| `--no-partial` | オフ | 発話中の速報字幕を無効化 |
| `--min-silence SEC` | 0.35 | 発話終了とみなす無音時間。小さいほど確定が速くなるが分割も増える |
| `--max-speech SEC` | 12.0 | 連続発話がこの秒数を超えたら強制的に確定させる |
| `--max-resident N` | 3 | tier0以外で常駐させるモデル数の上限（LRU退避）。`0`以下で無制限 |
| `--serve [PORT]` | オフ、8833 | `http://localhost:PORT`でダッシュボード+OBSオーバーレイを配信 |
| `--no-refine` | オフ | 発話群の二段パス再デコードを無効化 |
| `--transcript PATH` | なし | 清書済みトランスクリプト行をこのファイルに追記 |
| `--hotwords PATH` | なし | 固有名詞側にデコードを寄せるホットワード一覧（1行1語）。**現状jaルートには効果なし**（ReazonSpeechのbyte-level BPEなtokens.txtはエンコードできず、起動時に失敗数を警告表示）。jaの固有名詞には`--replace`を使ってください |
| `--replace PATH` | なし | ユーザー辞書。`誤=正`形式、1行1組。全出力に適用 |
| `--lang-switch-guard SEC` | 2.0 | この秒数未満の新言語判定はノイズとみなす：スイッチ確定（`--lid-switch-confirm`参照）には一切カウントされず、空デコード時のomnilingualフォールバックも抑制する（`0`で無効） |
| `--lid-switch-confirm N` | 2 | セッション言語を実際に切り替えるのに必要な、連続した新言語判定の回数（各判定は`--lang-switch-guard`秒以上）。大きくするほど切り替えが粘る |
| `--speakers` | オフ | 発話に話者ID（S1, S2, ...）をラベル付け |
| `--translate [LANGS]` | オフ、`en` | 日本語行をカンマ区切りの言語に翻訳。`en`は専用のFuguMTモジュール、それ以外（`zh`/`ko`/`es`/`fr`など）はモデルの語彙が対応していれば受け付ける。`zh`/`ko`/`es`以外は品質未実測の旨をnoteで表示、詳細はdocs/TRANSLATE_M2M.md |

## アーキテクチャ

```
                          ┌─────────────┐
  マイク/wav ───────────▶ │ Silero VAD  │  0.35秒終端検出 + 0.8秒プリロール
                          └──────┬──────┘
                                 │ 発話セグメント
                                 ▼
                   ┌───────────────────────────┐
                   │ whisper-tiny 言語判定       │  セグメント受信中の最初の
                   │ (+ 文字種仲裁)              │  約4秒間で実行
                   └─────────────┬─────────────┘
                                 │ 言語タグ
                 ┌───────────────┼────────────────┬─────────────┬──────────────┐
                 ▼               ▼                ▼             ▼              ▼
             ┌───────┐      ┌─────────┐      ┌──────────┐  ┌─────────┐   ┌──────────┐
             │  ja   │      │   zh    │      │  ko/yue  │  │ en + 24 │   │  約1600  │
             │Reazon │      │Paraformer│      │SenseVoice│  │欧州言語 │   │  その他  │
             │Speech │      │   -zh   │      │  small   │  │Parakeet │   │Omnilingual│
             │Zipform│      │         │      │          │  │TDT v3   │   │  ASR     │
             └───┬───┘      └────┬────┘      └────┬─────┘  └────┬────┘   └────┬─────┘
                 └───────────────┴────────────────┴─────────────┴─────────────┘
                                                │
                    速報字幕（約0.5秒ごと）        │   確定（発話終了から約0.1秒）
                     ◀───────────────────────────┴───────────────────────▶
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                 ┌────────────────┐   ┌──────────────────┐   ┌────────────────┐
                 │ 日本語句読点付与  │   │ 話者ラベリング      │   │  翻訳           │
                 │ (BERT復元)      │   │ (CAM++, --speakers)│   │ (FuguMT/M2M-100)│
                 └────────────────┘   └──────────────────┘   └────────────────┘
                                                │
                       2秒無音: 直近の発話群を一括再デコード（二段パス補正）
                                                │
                                                ▼
                            ダッシュボード / OBSオーバーレイ / トランスクリプトファイル
```

モデルは最初に必要になったときに読み込みます。上限（`--max-resident`）を超えたら
長く使っていないものから外すので、何か国語話してもメモリは上限を超えません。

## 実測パフォーマンス

本番と同じ経路（言語判定からデコード、日本語の句読点付与まで）を、実音声のクリップ単位で
採点した結果です。`en`はWER、それ以外はCER。測り方の詳細は`docs/SCORECARD.md`にあります。

| 言語 | クリップ数 | LID正解率 | ルート | 平均誤り率 | 平均RTF |
|---|---|---|---|---|---|
| ja | 15 | 15/15 | ReazonSpeech | 7.5% | 0.071 |
| en | 15 | 15/15 | Parakeet v3 | 2.3% | 0.109 |
| zh | 12 | 12/12 | Paraformer-zh | 5.3% | 0.102 |
| ko | 12 | 12/12 | SenseVoice | 8.1% | 0.062 |
| yue | 12 | 12/12 | SenseVoice | 6.1% | 0.061 |

どの言語もCPU単体で実時間の9〜16倍の速さです。開発中に何を試して何を捨てたかは、
30回分の改善記録ごと`docs/BENCHMARKS.md`に残してあります。

主な実測値:

- **日本語CER 5.8%**（ビームサーチ使用時）。同じ実放送音声で`whisper-large-v3-turbo`は13.8%。
- **確定までの平均が約100ms**（日本語、句読点込み）。5言語混在で全機能を有効にしても平均236ms。
- **メモリ2GB未満**（`--max-resident 3`のとき）。`--max-resident 2`なら1.35GB。

## 既知の制限

- **1文の中に複数言語が混ざる発話には対応していません。** 発話1つにつき言語を1つ選ぶ方式のため、
  日本語の文に英語フレーズが挟まると、そこが化けるか消えます。文単位で言語が切り替わる分には
  問題なく追従します（通訳が日英交互に話すような場面は得意です）。
- **効果音やBGMの直後の短い発話は、言語判定を外すことがあります。** ガード機構
  （`--lang-switch-guard`と`--lid-switch-confirm`の組み合わせ）である程度抑えていますが、
  起動直後の1発話目などは外すことがあります。どの程度外すかは`docs/BENCHMARKS.md`の
  イテレーション#29に実測があります。`--lid-switch-confirm 1 --lang-switch-guard 0`で
  この粘着ヒステリシスを完全に無効化できます（判定ごとに即切り替え）。ノイズ耐性と
  引き換えに応答性を最大化する設定で、ロック機構自体が自分のセットアップに合っているか
  検証したいときに使えます。
- **`--hotwords`は現状jaルート（ReazonSpeech）に効きません。** ReazonSpeechの
  `tokens.txt`はbyte-level BPEで、hayamimiがホットワードのエンコードに使う
  `modeling_unit=cjkchar`と非互換のため、全ホットワードがエンコードに失敗します
  （sherpa-onnx側はstderr警告のみで終了コード0のまま動くため気づきにくい問題でした
  ―GitHub Issue #1）。現在はhayamimi起動時に失敗数を警告表示するようにしています。
  jaの固有名詞には`--replace`を使ってください。根本解決にはReazonSpeech向けの
  `bpe.model`（現状未配布）か、自前のbyte-BPEホットワードエンコーダが必要で、
  今後の課題としています。
- **同時に話す2人は分離できません。** `--speakers`は発話ごとに「誰の声か」を推定する方式で、
  声が重なった区間は1人分のラベルにまとまります。
- **翻訳は小型モデルなりの品質です。** とくに中国語・韓国語への翻訳は数値や金額を
  間違えることがあるため、数字が壊れた訳文は原文をそのまま出す安全装置を入れています。
  実際の失敗例を`docs/TRANSLATE.md`と`docs/TRANSLATE_M2M.md`に載せているので、
  数字が大事な用途では先に見てください。
- **作者の環境以外での検証はまだ多くありません。** 手元で数値が再現しない場合は
  Issueで教えてもらえると助かります。

## ライセンス

ソースコードはMITライセンスです（`LICENSE`）。モデルの重みはこのリポジトリに含まれておらず、
`scripts/download_models.py`が各配布元からダウンロードします。モデルごとのライセンスは
`THIRD_PARTY_NOTICES.md`にまとめました。

1つだけ注意が必要なモデルがあります。日→英翻訳モデル（`mojicast-fugumt-ja-en-ct2`、
`--translate en`で使用）は**CC BY-SA 4.0**です。このモデルの重みを再配布するときは
クレジット表示と同ライセンスでの公開が必要になります。hayamimi本体のコードや、
`--translate`のそれ以外のターゲットが使うM2M-100（MIT）には影響しません。

## クレジット

hayamimiは次のプロジェクトの成果を借りて動いています。

- [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): 全モデルを動かしている推論エンジン。
- [ReazonSpeech](https://research.reazon.jp/)（Reazon Human Interaction Lab）: 日本語認識の主力モデル。
  日本語の精度はこのモデルに支えられています。
- [NVIDIA NeMo / Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): 英語と欧州24言語。
- [Meta AI Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr): 約1600言語を
  受け止めるフォールバック。
- [FunASR / SenseVoice](https://github.com/FunAudioLLM/SenseVoice)（Alibaba DAMO Academy）:
  中国語、韓国語、広東語の認識。
- [Mojicast](https://github.com/ishiki-emo/mojicast)（ishiki-emo）: ライブ字幕パイプラインの
  設計で多くを参考にしました。句読点モデルと翻訳モデルの変換版もこのプロジェクトの配布物です。
  完全オフラインで動く配信字幕アプリとして、Mojicast自体もおすすめです。
- [Silero VAD](https://github.com/snakers4/silero-vad): 発話区間の検出。
- [3D-Speaker](https://github.com/modelscope/3D-Speaker)（Alibaba DAMO Academy）:
  `--speakers`で使う話者埋め込みモデル。
- [Kiwi](https://github.com/bab2min/kiwipiepy): 韓国語の分かち書きを直す形態素解析器。

## Contributing

`CONTRIBUTING.md`を見てください。
