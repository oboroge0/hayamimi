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

実際のテレビ放送の日本語音声で測ると、この方式で**CER 3.8%**が出ます（2026-09-01再計測、冒頭欠落修正と数字正規化を含む現行パイプライン。`docs/SCORECARD.md`）。
同じ音声で`whisper-large-v3-turbo`は13.8%なので、誤りは半分以下。速度は6コアの
デスクトップCPUで実時間の10〜50倍です。
Whisper系・クラウドSTT API・他のローカルモデルとの比較（hayamimiが負けている言語も含めて）は
`docs/COMPARISON.md`にまとめています。

## 機能

| 機能 | 内容 |
|---|---|
| 5ルート言語カタログ | 日/中/韓/広東語/英+欧州24言語をそれぞれ最適な専用モデルへ。それ以外（約1600言語）はMeta Omnilingual ASRにフォールバック |
| 速報字幕 | 発話中に約0.5秒ごとにドラフトテキストが更新される |
| 高速確定 | 話し終えてから約100msで確定文が出る（日本語。他言語は`docs/GOALS.md`参照） |
| 二段パス補正 | 2秒の無音後、直近の発話群を一括再デコードし高精度な「清書」トランスクリプトを生成（実放送ja CER 15.5%→12.0%） |
| 話者ラベル | `--speakers`でCAM++埋め込みによりリアルタイムにS1/S2/...をタグ付けし、清書パスでpyannote segmentation-3.0による本格的な再分離をかけ直して同じS{n}に対応付け（AMI 5会議でDER平均25.7%→13.9%、詳細はLimitations参照） |
| 翻訳 | `--translate en,zh,ko,es,...`で日本語行をリアルタイム翻訳（en=FuguMT。それ以外はモデルの語彙が対応していれば任意のM2M-100ターゲット言語コードを受け付ける。zh/ko/esは品質を実測済み、詳細はdocs/TRANSLATE_M2M.md） |
| ホットワード/置換辞書 | `--hotwords`で固有名詞認識を強化（現状jaルートには効果なし。既知の制限を参照）、`--replace`は事後の検索置換でどのルートでも有効 |
| 漢数字→算用数字変換 | 位取りの漢数字・点小数・桁読みの数字列を保守的に算用数字へ変換（日/中/広東語のみ）。慣用句・固有名詞（一番、一緒、九州など）は既定で変換しない（`scripts/itn_cjk.py`） |
| 実行時辞書API | `RoutedASR.set_replacements()` / `set_itn_overrides()`でセッション中に置換辞書・ITN例外/強制指定を差し替え可能。`--serve`使用時は`GET`/`POST /replacements`・`/itn_overrides`でも同じことができる |
| 構造化イベント + 実行時設定 | パイプラインの各段階（確定文、翻訳、モデル読み込み、警告、セッション集計...）が`EventHub`に発行され、組み込み先アプリはこれを直接listenできる。`--serve`時はさらに`GET`/`POST /config`と`POST /reset`が使え、プロセスを再起動せずに言語・翻訳・VAD設定の変更やセッションのリセットができる（詳細は後述の「組み込み: 実行時制御と構造化イベント」参照） |
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

`--input ws`は既定で`127.0.0.1`にバインドするので、エンドポイントは何もしなければ
localhostからしかアクセスできません。LAN上の他端末を受け付けたいときだけ
`--ws-host 0.0.0.0`を明示してください（`/ingest`に認証はないので、信頼できる
ネットワークでのみ使うこと）。バインドしたアドレスは起動時にstderrへ出力されます。

## 他アプリへの組み込み

`scripts/realtime_transcribe.py`の各部品（`RoutedASR`、`build_vad`、`run_stream`）は
CLI専用ではなくimportして使えます。`RoutedASR(...)`と`build_vad(...)`は、
モデルパスが見つからない場合にsherpa-onnxのC++層が呼ぶ`exit()`（catchできない）
の代わりに、catchできる`asr_engine.ModelUnavailable`を送出します。また
`run_stream(..., stop_event=threading.Eventのインスタンス)`で`threading.Event`型の
停止トークンを受け付けるので、ホストアプリが自前のスレッドでパイプラインを回している
場合でも、CLIのプロセスにしか効かない`KeyboardInterrupt`に頼らず
`stop_event.set()`だけできれいに止められます。

## 組み込み: 実行時制御と構造化イベント

上の仕組みだけでは埋まらない穴が二つ残っていました。ひとつは、どの言語に固定するか、
翻訳するかどうか、VAD（音声区間検出。発話の開始と終了を判定する部品）の感度を
どのくらいにするか、といったセッション設定がコンストラクタ時にしか渡せなかったこと
です。ユーザーが途中で気が変わったホストアプリは、エンジンを丸ごと作り直すしか
ありませんでした。もうひとつは、構造化されたセッションイベントが`--serve`を
付けたときにしか存在しなかったことです。`--serve`なしでは、モデルの読み込みに
失敗したことや、発話言語が切り替わったことをプログラムから知る手段がstderrと
コンソールの自由形式の診断行を読み取るしかなく、これはプログラムから確実にパース
できるものではありませんでした。

どちらも同じ仕組みで解決しています。`realtime_transcribe.main()`（および
`subtitle_server.EventHub`を直接組み立てるアプリ）は、`--serve`のHTTPサーバーの
有無にかかわらず常にイベントハブを作るようになり、パイプラインの各段階はすべて
そこへ発行されます。`--serve`は同じハブの上に追加のHTTPサーバーと、
`subtitle_server.RuntimeControls`にひもづく3つのJSONエンドポイントを乗せる
だけの存在になりました。

**HTTPなしでイベントを受け取る。** このモジュールを直接importするアプリなら、
`--serve`はおろか`SubtitleServer`すら要りません。`hub.add_listener(callback)`は、
publishのたびに生のイベントdictを渡して同期的に呼び出すコールバックを登録します。
`hub.subscribe()`（前述のダッシュボード/オーバーレイが`/events`のSSEとして内部で
消費しているのと同じもので、`scripts/ws_ingest.py`はこれをそのままWebSocket
クライアントへミラーしている）は、同じイベント列をJSON文字列の`queue.Queue`として
返すので、コールバックではなくポーリングしたい側にはこちらが向いています。

**イベントの種類。** すべてのイベントは`type`キーを持つJSONオブジェクトです。

| `type` | 形 | いつ発行されるか |
|---|---|---|
| `session_start` | `{"type":"session_start"}` | セッション開始時に1回 |
| `partial` | `{"type":"partial","text":str}` | 発話中のドラフトが更新されたとき（約0.5秒おき） |
| `final` | `{"type":"final","text":str,"lang":str,"speaker":str,"latency_ms":float\|null,"tier":str,"audio_s":float,"lid_ms":float\|null,"decode_ms":float\|null,"switched":bool}` | VADのセグメントが確定したとき。`switched`は直前の確定文と`lang`が異なる場合だけtrue（セッション最初の確定文ではfalse） |
| `translation` | `{"type":"translation","lang":str,"text":str}` | `--translate`のターゲット言語が日本語の確定文・清書行の翻訳を終えたとき |
| `refine` | `{"type":"refine","text":str,"lang":str,"speaker":str,"audio_s":float}` | 発話群の二段目再デコード（清書）が出たとき |
| `model_load` | `{"type":"model_load","model":str,"phase":"start"\|"done","ms":float\|null}` | 認識器/LID/句読点/翻訳モデルの読み込みが始まった・終わったとき。`model`はエンジン内部の短縮名（`rz`/`pz`/`sv`/`v3`/`omni`/`pja`/`lid`/`punct`、または`translator:<言語コード>`） |
| `model_fallback` | `{"type":"model_fallback","requested":str,"used":str,"reason":str}` | 要求されたモデルが無く（`--minimal`インストールなど）別のティアに振り替えたとき。同じセッション内で同じ要求モデルについては1回だけ発行 |
| `warning` | `{"type":"warning","code":str,"message":str}` | 致命的ではない劣化状態。`code`は`hotwords_unencodable`・`segmentation_vad_unavailable`・`second_opinion_unavailable`・`diarization_failed`のいずれか |
| `session_summary` | `{"type":"session_summary","stats":{...},"speakers":{...}\|null}` | プロセス終了時、およびセッションリセット直前。`stats`はコンソールの`=== session summary: ... ===`行と同じ数値、`speakers`は`--speakers`使用時のみ話者診断行と同じ内容（それ以外は`null`） |
| `recluster` | `{"type":"recluster","time_s":float,"n_entries":int,"n_clusters":int,"mapping":{...}}` | `--speaker-global-recluster`のセッション末診断が実際に走ったとき |
| `session_reset` | `{"type":"session_reset"}` | `POST /reset`（または`reset_live_session()`の直接呼び出し）が完了したとき |

**HTTP経由の実行時制御（`--serve`時のみ）。** 前述の`/replacements`・`/itn_overrides`に加えて
3つのエンドポイントがあります。

```bash
# 現在のセッション設定を読む
curl http://localhost:8833/config

# セッションを英語に固定し、言語切替時にSenseVoice自身のLIDとの一致を
# 必須としないようにする
curl -X POST http://localhost:8833/config \
  -d '{"lang": "en", "dual_confirm": false}'

# 韓国語をリアルタイム翻訳の対象に追加する（ここに列挙しなかったターゲットは
# 削除される。他の設定キーと同様、集合全体を置き換える動作）
curl -X POST http://localhost:8833/config -d '{"translate": ["en", "ko"]}'

# VADの感度を下げる: セグメントが確定するまで待つ無音時間を長くする
curl -X POST http://localhost:8833/config -d '{"vad": {"min_silence": 0.6}}'

# 会話をリセットする: 話者と言語切替の状態は忘れるが、モデルは常駐したまま
# （再読み込みのコストなし）
curl -X POST http://localhost:8833/reset
```

`GET /config`は`{"lang": null|str, "dual_confirm": bool, "punctuate": bool,
"lid_switch_confirm": int, "min_switch_s": float, "translate": [str, ...],
"vad": {"threshold": float, "min_silence": float, "max_speech": float}}`を
返します。`POST /config`はこれらのキーの任意の部分集合を受け付け、対応する
`RoutedASR`/`TranslatorPool`/VADのsetterへそれぞれ渡します。値が不正な場合
（ルーティング不能な言語コード、負の`min_switch_s`、未対応の翻訳ターゲット
など）は、そのsetter自身のエラーメッセージ付きで`400`を返し、部分的な適用や
値の暗黙のクランプはしません。VADの感度だけは即座には反映されません。
sherpa-onnxにはその場で設定を変えるAPIが無いため、`vad`の変更は検出器が
発話区間の途中でないタイミングまで遅延され、そこで作り直されます。誰かが
話している最中に来た変更要求は、リクエストの瞬間ではなく、その人が話し終えた
タイミングで反映されます。

`POST /reset`は実行中セッションの話者セントロイド、言語の確定/保留状態、
清書・セッションの統計をクリアします。モデルは一切再読み込みしないので、
次のセグメントはプロセスを起動し直したかのように始まります。1つの長時間
プロセスの中で無関係な収録が切り替わる場面（配信のゲスト交代、来場者ごとに
リセットするキオスク端末など）で、モデル再読み込みのコストを払わずに済みます。

リセット処理自体はHTTPハンドラのスレッドでは実行されません。話者・言語の
状態はデコードスレッドが自前のロックなしで読み書きしているため、別スレッド
から直接リセットを走らせると、デコード中のスレッド側で例外が飛ぶ場合がある
とレビューで判明しました。代わりにキューに積んでおき、そのスレッドが音声
チャンクの間の安全なタイミングに達した時点で実行します。1チャンクは
せいぜい数十ミリ秒なので、通常はほぼ瞬時に反映されます。`POST /reset`は
リセットが実際に走った時点で`200 {"ok": true}`を返し、10秒経っても安全な
タイミングが来なければ`202 {"ok": false, "pending": true}`を返します
（まだキューに残っているだけで、後で必ず適用されます）。これが主に効いて
くるのは`--input ws`のときです。音声を送っているクライアントがいなければ
チャンク境界自体が発生しないので、そのあいだに要求したリセットは音声が
再開するまで反映されません。

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
| `--ws-host HOST` | `127.0.0.1` | `--input ws`の`/ingest`エンドポイントのバインドホスト。LANクライアントを受け付けるには`0.0.0.0`を指定 |
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
| `--mode {single,balanced,fast}` | `balanced` | 言語切替のプリセット。`balanced`は2つの言語判定器が一致した時だけ切り替える（`docs/LID.md`参照）。`single`は`--lang`で指定した言語に固定し自動切替を行わない。`fast`は判定のたびに即切り替えるv0.2.0以前相当の動作。下の個別フラグはプリセットより優先される |
| `--lang-switch-guard SEC` | 2.0 | この秒数未満の新言語判定はノイズとみなす：スイッチ確定（`--lid-switch-confirm`参照）には一切カウントされず、空デコード時のomnilingualフォールバックも抑制する（`0`で無効） |
| `--lid-switch-confirm N` | 2 | セッション言語を実際に切り替えるのに必要な、連続した新言語判定の回数（各判定は`--lang-switch-guard`秒以上）。大きくするほど切り替えが粘る |
| `--speakers` | オフ | 発話に話者ID（S1, S2, ...）をラベル付け。清書パスでpyannote segmentation-3.0による再分離をかけ直す |
| `--speaker-remap-threshold T` | 0.35 | 清書パスの分離結果をセッション全体のS{n}ラベルへ対応付ける際のコサイン類似度閾値（速報パスは従来の0.45のまま） |
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

- **日本語CER 3.8%**。同じ実放送音声で`whisper-large-v3-turbo`は13.8%。オプションの清書時二段照合（`--refine-ja-second-opinion`）は別の実放送50分セットで4.0%。
- **確定までの平均が約100ms**（日本語、句読点込み）。5言語混在で全機能を有効にしても平均236ms。
- **メモリ2GB未満**（`--max-resident 3`のとき）。`--max-resident 2`なら1.35GB。

## 既知の制限

- **1文の中に複数言語が混ざる発話には対応していません。** 発話1つにつき言語を1つ選ぶ方式のため、
  日本語の文に英語フレーズが挟まると、そこが化けるか消えます。文単位で言語が切り替わる分には
  問題なく追従します（通訳が日英交互に話すような場面は得意です）。
- **効果音やBGMの直後の短い発話は、言語判定を外すことがあります。** ガード機構
  （`--lang-switch-guard`と`--lid-switch-confirm`の組み合わせ）である程度抑えていますが、
  判定と文字化けが偶然「別言語っぽく」揃ってしまうケースは死角として残っています。
  どの程度外すかは`docs/BENCHMARKS.md`のイテレーション#29に実測があります。
  `--lid-switch-confirm 1 --lang-switch-guard 0`でこの粘着ヒステリシスを完全に
  無効化できます（判定ごとに即切り替え）。ノイズ耐性と引き換えに応答性を最大化する設定で、
  ロック機構自体が自分のセットアップに合っているか検証したいときに使えます。
- **セッションの最初の発話は、必ずSenseVoiceで確認してから言語を決めます**
  （`docs/NOISE.md`の二重判定ポリシー節を参照）。これにより、whisper-tinyの
  ブートストラップ誤判定が対応モデルの無い言語へセッションを丸投げすることは
  なくなりました。SenseVoice対応5言語（ja/en/zh/ko/yue）以外（欧州語や`--minimal`
  未対応言語）は、`--lang-switch-guard`長のセグメントが`--lid-switch-confirm`回
  連続するまでセッション言語として確定しないため、正当な欧州語セッションも
  起動直後は即決ではなく少し遅れて確定します。
- **`--hotwords`は現状jaルート（ReazonSpeech）に効きません。** ReazonSpeechの
  `tokens.txt`はbyte-level BPEで、hayamimiがホットワードのエンコードに使う
  `modeling_unit=cjkchar`と非互換のため、全ホットワードがエンコードに失敗します
  （sherpa-onnx側はstderr警告のみで終了コード0のまま動くため気づきにくい問題でした
  ―GitHub Issue #1）。現在はhayamimi起動時に失敗数を警告表示するようにしています。
  jaの固有名詞には`--replace`を使ってください。根本解決にはReazonSpeech向けの
  `bpe.model`（現状未配布）か、自前のbyte-BPEホットワードエンコーダが必要で、
  今後の課題としています。
- **同時に話す2人は分離できません。** 清書パスでも同じで、pyannote segmentation-3.0は
  重複区間そのものは検出できますが、hayamimiは重複を考慮した書き起こし（who-said-what
  の重複割当）まではやっておらず、先に処理した方の話者ラベルが勝ちます。清書パスで
  変わったのは別のところです: 従来は無音区切り(または25秒)で確定した1グループの中で
  話者が何度切り替わっても、確定済み各行の多数決で1つのラベルにまとめていました。
  今は各グループをpyannote segmentation-3.0とリアルタイムパスと同じCAM++埋め込みで
  再分離し、そのローカルクラスタをセッション全体のグローバルS{n}へ対応付け直すため、
  1グループ内で複数回話者が入れ替わっても`[refine/S{n}]`の行ごとに正しく分かれます。
  AMI会議データ（計50分、CC BY 4.0、collar 0.25秒、`docs/DIARIZATION_PLAN.md`§8）で
  実測したところ、平均DERはリアルタイムパスのみの25.7%から13.9%まで下がりました。
  参照話者数は各会議4人ですが、hayamimiの推定はまだ過大（4〜8人）で、`--speakers`は
  「だいたい合っているターン分け」であって正確な話者数のソースとしては使えません。
  この過大表示を画面上で緩和するため、初出の話者は仮表示（`S5?`）で出て、2回目に
  再登場した時点で正式な`S{n}`に確定します。一度きりで終わった話者は`?`付きのまま
  セッション終了まで残ります（詳細は`docs/DIARIZATION_PLAN.md`§11）。
- **翻訳は小型モデルなりの品質です。** とくに中国語・韓国語への翻訳は数値や金額を
  間違えることがあるため、数字が壊れた訳文は原文をそのまま出す安全装置を入れています。
  実際の失敗例を`docs/TRANSLATE.md`と`docs/TRANSLATE_M2M.md`に載せているので、
  数字が大事な用途では先に見てください。
- **複数の文をまとめてデコードすると、先頭の文が失われることがあります。** オフライン
  認識器が複数発話のバッファを1発話に潰してしまう現象です。hayamimiは「話した長さの
  わりに文字数が少なすぎる」という症状を見つけたときだけ、そのバッファを0.35秒以上の
  内部無音で分割して decode し直し、実際に文が戻ったときだけ採用します。逃げ切られる
  のは2種類——分割できる無音がないほど詰めて話された場合（head-dropout調査で見つかった
  clip-324系）と、落ちた量が少なくて文字数が正常に見えてしまう場合です。
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
