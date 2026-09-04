# チューニング一覧

hayamimiの両実装について、ユーザーが触れるつまみを、既定値・変更する場所・
その既定値を決めた記録つきで並べたものです。推定値は入れていません。既定値は
すべて行に書いたコードから読み出したもので、「根拠」欄のリンクはその値を決めた
実測（または設計メモ）です。根拠欄が空の行は、上流ライブラリの既定値をそのまま
使っていて、このプロジェクトでは代案を測っていないという意味です。

対象は独立した2つの実装で、設定ファイルは共有していません。

- **Pythonエンジン** — `scripts/realtime_transcribe.py`（CLI）、
  `scripts/asr_engine.py`（`RoutedASR`）、`scripts/subtitle_server.py`
  （`POST /config`）。
- **`hayamimi_core`**（Flutter/Dart） — `mobile/hayamimi_core/lib/live/*.dart`。
  パッケージ自体のガイドは
  [`mobile/hayamimi_core/README.md`](../../mobile/hayamimi_core/README.md)。

日本語ルートの解決済み設定をJSONで出すには`python scripts/dump_ja_config.py`。
コミット済みのコピーは
[`../spec/ja_pipeline_spec.json`](../spec/ja_pipeline_spec.json)、それが記述する
仕様書は[`../spec/ja_pipeline.ja.md`](../spec/ja_pipeline.ja.md)です。

## Pythonエンジン

「`RoutedASR`のsetter」は動作中の`asr_engine.RoutedASR`インスタンスのメソッドで、
組み込みアプリから呼べます（[`embedding.ja.md`](embedding.ja.md)）。
「`POST /config`のキー」は`--serve`のHTTPエンドポイントが受け付けるキーです。
3つとも埋まっているつまみはセッション中に変更でき、CLIだけのつまみはプロセスの
起動時に固定されます。

### 認識とルーティング

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| `--threads` | `4` | CLIフラグ、`RoutedASR(threads=...)` | 各sherpa-onnx認識器に渡すintra-opスレッド数 | |
| `--max-resident` | `3` | CLIフラグ、`RoutedASR(max_resident=...)` | 常駐するja/enティア以外に何個の認識器をメモリに残すか。超えるとLRUで落とす | [benchmarks](../results/benchmarks.md) イテレーション#3（RAM実測）、#7（LRUアンロード） |
| `--mode {single,balanced,fast}` | `balanced` | CLIフラグ | 言語切替ポリシーのプリセット。下の2フラグと`dual_confirm`をまとめて決める | [lid.md](../eval/lid.md) |
| `--lang CODE` | `None` | CLIフラグ、`RoutedASR.set_forced_lang()`、`POST /config`の`lang` | 全セグメントを1言語に固定しLIDを完全に飛ばす（`--mode single`で必須） | |
| `--lang-switch-guard SEC` | `2.0`（`balanced`）、`0.0`（`fast`）。`realtime_transcribe.py`の`mode_defaults`。`RoutedASR.min_switch_s`自体は`2.0` | CLIフラグ、`RoutedASR.set_min_switch_s()`、`POST /config`の`min_switch_s` | これより短い別言語の検出は切替の根拠に数えない | [benchmarks](../results/benchmarks.md) イテレーション#29、[lid.md](../eval/lid.md) |
| `--lid-switch-confirm N` | `2`（`balanced`）、`1`（`fast`） | CLIフラグ、`RoutedASR.set_lid_switch_confirm()`、`POST /config`の`lid_switch_confirm` | セッションが言語を切り替えるまでに必要な同一言語検出の連続回数 | [benchmarks](../results/benchmarks.md) イテレーション#29、[lid.md](../eval/lid.md) |
| 二重LID確認 | オン（`RoutedASR(dual_confirm=True)`。`--mode fast`ではオフ） | `RoutedASR.set_dual_confirm()`、`POST /config`の`dual_confirm` | ja/en/zh/ko/yueへの切替時にSenseVoice自身のLIDとwhisper-tinyの一致を要求するか | [noise.md](../eval/noise.md)、[lid.md](../eval/lid.md) |
| `asr_engine.LID_MAX_SECONDS` | `4.0`秒 | モジュール定数 | セグメント冒頭の何秒をLIDモデルに渡すか | [lid.md](../eval/lid.md) |
| `--hotwords PATH` | `""` | CLIフラグ、`RoutedASR(hotwords_file=...)` | sherpa-onnxの認識器レベルのホットワード。**jaティアには効かない**（byte-BPEの`tokens.txt`と`cjkchar`のmodeling unitが非互換） | [README.ja.md](../../README.ja.md)の「既知の制限」 |
| `asr_engine.RZ_HOTWORDS_SCORE` | `2.0` | モジュール定数 | エンコードできた場合のホットワードのスコア加算 | |

### 発話区間の切り出し（VAD）

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| `--min-silence SEC` | `0.35`秒 | CLIフラグ、`POST /config`の`vad.min_silence` | セグメントが確定するまでに必要な無音の長さ | [benchmarks](../results/benchmarks.md) イテレーション#9: 0.5→0.35はCER変化なしで確定が150ms早い |
| `--max-speech SEC` | `12.0`秒 | CLIフラグ、`POST /config`の`vad.max_speech` | 区切りのない発話をVADが強制的に閉じるまでの長さ | [benchmarks](../results/benchmarks.md) イテレーション#23 |
| VADの`threshold` | `0.5` | `POST /config`の`vad.threshold`のみ（CLIフラグなし） | Sileroの音声確率のしきい値。高いほど鈍感 | [diarization.md](../design/diarization.md) §13（Round 3のスイープ） |
| `realtime_transcribe.VAD_MIN_SPEECH_S` | `0.25`秒 | モジュール定数 | これより短い音声区間は捨てる | |
| `realtime_transcribe.WINDOW_SIZE` | `512`サンプル（16kHzで約32ms） | モジュール定数 | Sileroに渡す1フレームの長さ | |
| `realtime_transcribe.VAD_BUFFER_S` | `30.0`秒 | モジュール定数 | 検出器内部のリングバッファ | |
| `realtime_transcribe.PREROLL_S` | `1.0`秒 | モジュール定数 | VADが検出した発話開始の手前に足す音声。直前セグメントの終端でクランプ | [head_dropout.md](../eval/head_dropout.md) |

`POST /config`の`vad`だけは即時に反映されません。sherpa-onnxにその場で書き換える
APIが無いため、検出器が発話区間の途中でないタイミングで作り直されます。詳細は
[`embedding.ja.md`](embedding.ja.md)。

### ドラフトと清書パス

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| `--no-partial` | オフ（ドラフトは有効） | CLIフラグ | 発話中の暫定字幕 | [benchmarks](../results/benchmarks.md) イテレーション#2 |
| `realtime_transcribe.PARTIAL_EVERY_S` | `0.5`秒 | モジュール定数 | 発話中にドラフトを再デコードする間隔 | |
| `realtime_transcribe.PARTIAL_WINDOW_S` | `8.0`秒 | モジュール定数 | ドラフトが再処理する末尾ウィンドウ | |
| `--no-refine` | オフ（清書は有効） | CLIフラグ | 発話群の二段目再デコード | [benchmarks](../results/benchmarks.md) イテレーション#10 |
| `realtime_transcribe.GROUP_GAP_S` | `2.0`秒 | モジュール定数 | 発話群を閉じる無音の長さ | |
| `realtime_transcribe.GROUP_MAX_S` | `25.0`秒 | モジュール定数 | 発話群を早めに閉じる長さ | |
| `asr_engine.REFINE_MIN_REGROUP_S` | `2.5`秒 | モジュール定数 | 再グループ化する価値のある最短の群 | |
| `--refine-ja-second-opinion` | オフ | CLIフラグ、`RoutedASR(ja_second_opinion=...)` | ja清書でparakeet-jaをセカンドオピニオンとして走らせる | [eval_real.md](../eval/eval_real.md) |
| `--refine-agree-threshold CER` | `0.25`（`asr_engine.SECOND_OPINION_THRESHOLD`） | CLIフラグ、`RoutedASR(agree_threshold=...)` | 2つのja認識器が「一致した」とみなすCER距離 | |
| 冒頭欠落の再試行 | オン。`asr_engine.SEGMENT_MIN_S` `4.0`秒、`SEGMENT_MIN_SILENCE_S` `0.35`秒、`SEGMENT_MIN_SPEECH_S` `0.25`秒、`SEGMENT_PAD_S` `0.35`秒 | モジュール定数 | 疑わしいデコードを内部無音で分割して再試行する条件 | [head_dropout.md](../eval/head_dropout.md)、[benchmarks](../results/benchmarks.md) 2026-08-31 |

### テキストの後処理

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| 日本語の句読点付与 | オン | `RoutedASR(punctuate=...)`、`RoutedASR.set_punctuate()`、`POST /config`の`punctuate` | ja出力への、。？の挿入 | [punct_ja.md](../design/punct_ja.md) |
| `--replace PATH` | `""` | CLIフラグ、`RoutedASR.set_replacements()`、`POST /replacements` | 最後に適用する文字列置換（ja固有名詞のホットワード代替） | [benchmarks](../results/benchmarks.md) イテレーション#14 |
| ITNの上書き | 空 | `RoutedASR.set_itn_overrides()`、`POST /itn_overrides` | CJK逆正規化（漢数字→算用数字）の例外指定 | [benchmarks](../results/benchmarks.md) イテレーション#17 |
| `--translate [LANGS]` | オフ。フラグのみなら`en` | CLIフラグ、`POST /config`の`translate`（集合全体を置換） | ja行のリアルタイム翻訳。`en`はFuguMT、それ以外はM2M-100 | [translate.md](../design/translate.md)、[translate_m2m.md](../design/translate_m2m.md) |

後処理の順序はITN→句読点→置換に固定です（`dump_ja_config.py`の
`postprocessing_order`）。

### 話者ラベリング（`--speakers`）

ここのしきい値はすべてAMI 5会議（計50分、CC BY 4.0、collar 0.25秒）で
スイープしています。節番号は[`diarization.md`](../design/diarization.md)のものです。

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| `--speakers` | オフ | CLIフラグ | 確定文・清書行への話者ラベル（`S1`、`S2`…） | [diarization.md](../design/diarization.md) §6-8 |
| `speaker_id.SIM_THRESHOLD` | `0.45` | モジュール定数 | 速報パスで既存話者に合流するコサイン類似度 | [diarization.md](../design/diarization.md) §8 |
| `--speaker-remap-threshold T` | `speaker_id.REMAP_THRESHOLD` = `0.35` | CLIフラグ | 清書パスのローカルクラスタ→グローバル対応付けのしきい値（速報パスとは別に調整） | [diarization.md](../design/diarization.md) §8 |
| `diarize.DEFAULT_THRESHOLD` | `0.5` | モジュール定数 | 清書グループ内のクラスタリングしきい値 | [diarization.md](../design/diarization.md) §13 |
| `diarize.DEFAULT_MIN_DURATION_ON` | `0.3`秒 | モジュール定数 | セグメンテーションモデルが報告する最短の発話区間 | [diarization.md](../design/diarization.md) §12（Round 2） |
| `diarize.DEFAULT_MIN_DURATION_OFF` | `0.5`秒 | モジュール定数 | 2つの区間を分ける最短の無音 | [diarization.md](../design/diarization.md) §12 |
| `--speaker-merge` / `--speaker-merge-threshold T` | オフ / `speaker_id.MERGE_THRESHOLD` = `0.80` | CLIフラグ | 過分割された話者のセントロイド統合。**実測のうえ不採用** | [diarization.md](../design/diarization.md) §9 |
| `--speaker-hysteresis` / `--speaker-hysteresis-min-hits N` | オフ / `speaker_id.HYSTERESIS_MIN_HITS` = `2` | CLIフラグ | 新規話者のヒステリシス。**実測のうえ不採用** | [diarization.md](../design/diarization.md) §9 |
| `speaker_id.PROVISIONAL_CONFIRM_HITS` | `2` | モジュール定数 | 仮ラベル`S5?`が`S5`に確定するまでの出現回数 | [diarization.md](../design/diarization.md) §11 |
| `--speaker-min-remap-update-s S` | `0.0`（no-op） | CLIフラグ | S秒未満のクラスタを読み取り専用で対応付ける | [diarization.md](../design/diarization.md) §14（Round 4 T2） |
| `--speaker-joint-remap` | オフ | CLIフラグ | グループ内の対応付けをハンガリアン法で同時に解く | [diarization.md](../design/diarization.md) §15（Round 5 T1） |
| `--speaker-exclude-provisional-remap` | オフ | CLIフラグ | 仮のままのセントロイドを対応付け先にしない | [diarization.md](../design/diarization.md) §15（T3） |
| `--speaker-global-recluster` | オフ | CLIフラグ | セッション終了時の再クラスタリング診断。**実測のうえ不採用**（confusion +22.5pt） | [diarization.md](../design/diarization.md) §17 |
| `--speaker-global-recluster-threshold T` | `0.65` | CLIフラグ | その診断の凝集型マージのしきい値 | [diarization.md](../design/diarization.md) §17 |
| `--speaker-num-clusters-hint` / `--speaker-num-clusters-hint-min-s` | `off` / `0.0` | CLIフラグ | `FastClustering`へのクラスタ数ヒント。**実測のうえ不採用** | [diarization.md](../design/diarization.md) §19 |
| `speaker_id.MAX_EMBED_SECONDS` | `6.0`秒 | モジュール定数 | 1つの話者埋め込みに入れる音声長の上限 | |

### サーバーと入出力

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| `--serve [PORT]` | オフ。フラグのみなら`8833` | CLIフラグ | HTTPダッシュボード、SSEの`/events`、JSON制御エンドポイント | [embedding.ja.md](embedding.ja.md) |
| `--input {mic,wav,ws}` | 自動（`--wav`があれば`wav`、なければ`mic`） | CLIフラグ | 音声の入力元 | |
| `--ws-host` | `127.0.0.1` | CLIフラグ | `--input ws`のバインドアドレス。`/ingest`に認証はない | |
| `--ws-port` | `8766` | CLIフラグ | `--input ws`のポート | |
| `--transcript PATH` | オフ | CLIフラグ | 確定文と清書をファイルに追記 | |
| `realtime_transcribe.RESET_TIMEOUT_S` | `10.0`秒 | モジュール定数 | `POST /reset`が`202`を返すまでにチャンク境界を待つ時間 | [embedding.ja.md](embedding.ja.md) |

## `hayamimi_core`（Flutter/Dart）

パッケージ自身のリファレンスは
[`mobile/hayamimi_core/README.md`](../../mobile/hayamimi_core/README.md)です。
この表は同じつまみを1か所に集め、各既定値の出どころを添えたものです。
「コンストラクタ引数」は`LiveTranscriber`/`HayamimiLive`のコンストラクタ、
「実行時setter」は同名のプロパティへの再代入で、次の判定時またはバッファ書き込み時
から効きます（不正な値＝非正または非有限は即座に`ArgumentError`）。

| つまみ | 既定値 | 変更する場所 | 何に効くか | 根拠 |
|---|---|---|---|---|
| `draftIntervalSeconds` | `1.0`秒（`defaultDraftIntervalSeconds`、`live/draft_pass.dart`） | コンストラクタ引数、実行時setter | セグメント進行中にドラフトを再デコードする間隔 | [android_emulator.md](../verify/android_emulator.md) |
| `draftWindowSeconds` | `8.0`秒（`defaultDraftWindowSeconds`） | コンストラクタ引数、実行時setter | ドラフトが再処理する末尾ウィンドウ | [android_emulator.md](../verify/android_emulator.md) |
| `minDraftAudioSeconds` | `0.25`秒（`defaultMinDraftAudioSeconds`） | コンストラクタ引数、実行時setter | ドラフトを走らせる価値のある最短音声長 | |
| `autoRefineSilenceSeconds` | `4.0`秒（`defaultAutoRefineSilenceSeconds`、`live/refine_pass.dart`） | コンストラクタ引数、実行時setter | 自動清書を発火させる無音の長さ | [android_emulator.md](../verify/android_emulator.md) run 3 |
| `autoRefineMaxBufferedSeconds` | `20.0`秒（`defaultAutoRefineMaxBufferedSeconds`） | コンストラクタ引数、実行時setter | 無音が来なくても自動清書を発火させるバッファ長の上限 | |
| `refineBufferMaxSeconds` | `60.0`秒（`defaultRefineBufferMaxSeconds`） | コンストラクタ引数、実行時setter | 清書バッファが最古のセグメントを捨て始めるまでの上限 | |
| `prerollSeconds` | `1.0`秒（`defaultPrerollSeconds`、`live/preroll.dart`） | コンストラクタ引数、実行時setter | セグメント開始の手前に足す音声。`0`で無効 | [android_emulator.md](../verify/android_emulator.md) run 1（`資料は昨日送りました`が`昨日は昨日送りました`になった）、デスクトップ側は[head_dropout.md](../eval/head_dropout.md) |
| `defaultPrerollKeepSeconds` | `30.0`秒（`live/preroll.dart`） | モジュール定数 | プリロール用に保持する過去音声の長さ | |
| `VadSensitivity.threshold` | `0.5` | `start(vadSensitivity:)`、`setVadSensitivity()` | Sileroの音声確率しきい値（sherpa-onnxの既定値。デスクトップ側も触っていない） | [diarization.md](../design/diarization.md) §13 |
| `VadSensitivity.minSilenceSeconds` | `0.35`秒 | `start(vadSensitivity:)`、`setVadSensitivity()` | セグメント確定までの無音長。**sherpa-onnxの0.5ではない**。0.5だとエミュレータ実行で日本語3文が1つの6.13秒セグメントに融合し、認識器は最後の1文しか返さなかった | [android_emulator.md](../verify/android_emulator.md) run 1、[benchmarks](../results/benchmarks.md) イテレーション#9 |
| `VadSensitivity.minSpeechSeconds` | `0.25`秒 | `start(vadSensitivity:)`、`setVadSensitivity()` | これより短い音声はデコード前に捨てる（sherpa-onnxの既定値） | |
| `VadSensitivity.maxSpeechSeconds` | `12.0`秒 | `start(vadSensitivity:)`、`setVadSensitivity()` | 保証ではなく目安。5.0秒に設定したセッションでも6.134秒のセグメントが出たので、これを前提にバッファやタイムアウトを設計しないこと | [android_emulator.md](../verify/android_emulator.md) run 1 |
| `defaultMinDecodeDurationSeconds` | `0.2`秒（`live/speech_segment_filter.dart`） | モジュール定数 | これより短いセグメントはデコードしない | |
| `decodingMethod` | `null`。プレーン経路は`greedy_search`、`RoutingProfile.jaSenseVoice`のjaティアは`modified_beam_search` | `start()` / `startDebugWavStream()`の引数 | 探索アルゴリズム。デスクトップ本番の値は`modified_beam_search` | [mobile_quantization.md](../design/mobile_quantization.md) |
| `hotwordsFile` / `hotwordsScore` | `null` / `1.5` | `start()` / `startDebugWavStream()`の引数。**実行時setterなし**（認識器の構築時に焼き込まれる） | プレーン経路とルーテッドjaティアの認識器レベルのホットワード | |
| `defaultPunctNumThreads` | `2`（`live/ja_punctuation.dart`） | `JaPunctuation`のコンストラクタ | 句読点モデルのONNX intra-opスレッド数 | [android_emulator.md](../verify/android_emulator.md) run 3（ウォーム時、11〜14文字1行あたり約38〜49ms） |
| `JaPunctuation(applyToFinals:)` | `true` | `JaPunctuation`のコンストラクタ | 清書だけでなく速報の確定文にも句読点を付けるか。`false`で清書のみに戻る | [android_emulator.md](../verify/android_emulator.md) run 3 |
| `SubtitleBroadcastServer.defaultPort` | `8833`（`lib/server/subtitle_broadcast_server.dart`） | コンストラクタ引数 | LAN配信のポート。Python側`--serve`の既定と揃えてある | |

### 意図的に公開していないもの

コンストラクタ引数もsetterも無い内部タイムアウトと安全弁です。存在を知って
つまみを探さずに済むように並べてあります。変えるにはパッケージ自体の修正が必要です。

| 定数 | 値 | 場所 | なぜつまみにしないか |
|---|---|---|---|
| `_decodeDrainTimeout` | 10秒 | `live/live_transcriber.dart` | `stop()`がデコードワーカーの残作業を待つ上限。1セグメントの所要時間（1秒未満）に対しては十分長く、画面が閉じるのを待つ人間に対しては十分短い。チューニング用ではなく、ハングしたワーカーに対する歯止め。 |
| `defaultDecodeWorkerShutdownTimeout` | 3秒 | `live/decode_worker.dart` | ワーカーisolateの終了ackを待つ上限。ここまで返事が無いワーカーは戻ってこないデコードの中で固まっているので、待つ意味がもう無い。テストから参照できるよう名前付き`const`にしてあるだけで、公開つまみではない。 |
| `_autoRefineCheckInterval` | 1秒 | `live/live_transcriber.dart` | 自動清書の判定を回す間隔。判定は数回の比較なので、1秒刻みでも応答性は十分でバッテリーにはほぼ影響しない。 |
| `_minRefineAudioSeconds` | 0.5秒 | `live/live_transcriber.dart` | 清書パスを走らせる価値のある最短音声長。デスクトップの`Refiner`の`len(buf) < sr // 2`と同じ考え方で、1秒未満を再デコードしてもフルのデコード費用がかかるだけで、すでに出ている確定文を上回れない。 |
