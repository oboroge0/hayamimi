# 他アプリへの組み込み（Pythonエンジン）

2026-09-03にリポジトリ直下の`README.ja.md`から切り出したもの。以下の2節は
そのままの文面なので、文中の「前述」「上の」はREADME側を指しています
（[`../../README.ja.md`](../../README.ja.md)）。

Flutter/Dart側の組み込みガイドは
[`mobile/hayamimi_core/README.md`](../../mobile/hayamimi_core/README.md)に
あります。このファイルはPythonエンジンについてだけ書いています。

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
