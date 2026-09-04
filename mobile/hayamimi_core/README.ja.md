# hayamimi_core

English: [README.md](README.md)

[hayamimi](https://github.com/oboroge0/hayamimi) のモバイル向け音声認識パイプラインを、
`mobile/` のデモアプリから切り出して再利用可能にしたコアパッケージ。他の Flutter アプリ
(例: スマートグラス連携アプリ) が、デモアプリの UI を引き込まずにライブ字幕を組み込める。

3 つの独立した部品を提供する。いずれも同じ
[`SubtitleEvent`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/server/subtitle_event.dart)
ワイヤーフォーマット (デスクトップの `scripts/subtitle_server.py` と OBS オーバーレイが使う
`partial`/`final`/`translation`/`refine` と同じ JSON 形式) をしゃべる。

- **`HayamimiLive`** — 端末上での文字起こし。マイク → Silero VAD →
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) のオフライン ASR モデル、
  という経路で、任意の 2 パス目「清書」(refine, 再デコード) がある。
- **`HayamimiRemote`** — この端末のマイクを PC 上で動く hayamimi サーバー
  (`--input ws --serve`) にストリーミングし、返ってくる字幕イベントを受け取るだけの薄いクライアント。
  PC 側のフルパイプライン (多言語ルーティング・refine・翻訳) が実際の認識を行う。
- **`SubtitleBroadcastServer`** — 上記どちらかを SSE + 透過オーバーレイページとして
  LAN に再配信する小さな HTTP サーバー。同一ネットワーク上の OBS やブラウザから購読できる。

`LiveTranscriber`・`RemoteTranscriber`・`BenchRunner`・VAD/PCM ヘルパーなど、より低レベルな
部品も export されている。上の 2 つのファサードより細かく制御したい場合は
[`lib/hayamimi_core.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/hayamimi_core.dart)
の export 一覧を見ること。

**詳細なリファレンス** (イベント一覧の全項目、リモートモードの詳細、API ドキュメント) は
[README.md](README.md) を参照。このファイルは最初の組み込みに必要なものだけをまとめた
短縮版で、250〜400 行を目標にしている。

## インストール

```yaml
dependencies:
  hayamimi_core: ^0.1.0
  # または、このリポジトリを直接追うなら:
  # hayamimi_core:
  #   path: ../path/to/hayamimi/mobile/hayamimi_core
  # hayamimi_core:
  #   git:
  #     url: https://github.com/oboroge0/hayamimi.git
  #     path: mobile/hayamimi_core
```

```dart
import 'package:hayamimi_core/hayamimi_core.dart';
```

## 初期化とプラットフォーム権限

### `sherpa_onnx.initBindings()`

`sherpa_onnx` (このパッケージの依存先) は FFI のバインディング表を isolate ごとに持つので、
そのネイティブオブジェクトに触れる isolate は最初に自分で初期化しておく必要がある。
`runApp` の前に、アプリのメイン isolate で一度だけ呼ぶ:

```dart
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  sherpa_onnx.initBindings();
  runApp(const MyApp());
}
```

このパッケージ自身のバックグラウンド isolate (decode worker と、VAD を組み立てる短命な
isolate) は自分で `initBindings()` を呼ぶので、呼び出し側が気にする必要はない。ただし VAD は
組み立て後に呼び出し側の isolate で動き続けるので、そこでも先に初期化しておくこと。
呼び忘れると、sherpa-onnx のオブジェクトに最初に触れた瞬間にネイティブクラッシュか
「failed to lookup symbol」エラーになる (親切な Dart 例外にはならない)。

### プラットフォーム権限

`sherpa_onnx` と `record` (どちらもこのパッケージの依存先) には通常のプラットフォーム設定が
必要: Android の `RECORD_AUDIO` 権限
(`android/app/src/main/AndroidManifest.xml`)、iOS の
`NSMicrophoneUsageDescription` (`ios/Runner/Info.plist`)。動作する例はこのリポジトリの
[`mobile/`](https://github.com/oboroge0/hayamimi/tree/main/mobile) アプリか、このパッケージ自身の
[`example/`](https://github.com/oboroge0/hayamimi/tree/main/mobile/hayamimi_core/example)
を参照。`HayamimiRemote` も同じ `RECORD_AUDIO`/`NSMicrophoneUsageDescription`
(こちらもマイクをストリーミングする。デコード先が端末内か PC かの違いだけ) に加えて
`INTERNET` が必要。Android のデバッグビルドでは暗黙に付与されるが、リリースの manifest には
明示しておくこと。`SubtitleBroadcastServer` を使う場合はさらにもう 1 つ、iOS のローカル
ネットワーク権限が要る (下記「LAN への配信」参照)。

## 端末上での字幕: HayamimiLive

エラー処理や widget の配線を省いた最小の骨格。動く完全な例が
[`example/`](https://github.com/oboroge0/hayamimi/tree/main/mobile/hayamimi_core/example)
にある (pub.dev 標準の `example/` レイアウト)。モデルファイルの配置は下の
「モデル配置」を参照。

```dart
import 'package:hayamimi_core/hayamimi_core.dart';

final live = HayamimiLive();

live.events.listen((event) {
  switch (event) {
    case PartialSubtitleEvent(:final text):
      print('draft: $text');
    case FinalSubtitleEvent(:final text, :final lang):
      print('final [$lang]: $text');
    case RefineSubtitleEvent(:final text):
      print('refine (清書): $text');
    default:
      break;
  }
});

await live.start(
  modelDir: '/path/to/model',        // encoder/decoder/joiner/tokens.txt
  vadModelPath: '/path/to/silero_vad.onnx',
  // 多言語ルーティング (任意): セグメントごとに ReazonSpeech (ja) か
  // SenseVoice (en/zh/ko/yue) に振り分ける。単一言語固定モデルの代わりに使う。
  routingProfile: RoutingProfile.jaSenseVoice,
  senseVoiceModelDir: '/path/to/sense_voice',
  lidModelDir: '/path/to/lid',
);

// ... あとで、ボタンタップ時など:
await live.refineNow();   // 手動で「清書」再デコードを走らせる

// ... 終了時:
await live.stop();
await live.dispose();
```

`HayamimiLive(textTransform: ...)` (セッション中も差し替え可能な `live.textTransform` でも可)
で、各テキストが `SubtitleEvent` になる直前にホスト側で書き換えられる。CJK の逆テキスト
正規化 (漢数字→アラビア数字、デスクトップ側の `scripts/itn_cjk.py`) やユーザー置換辞書は
Dart 側にまだ移植されていないので、必要なら自前でここに実装すること。

## モデル配置

`HayamimiLive` はディスク上の sherpa-onnx モデルファイルを必要とする。このパッケージ自体は
何もバンドルしないが、ダウンローダを同梱しているので取得・検証・展開・配置を自分で書く必要はない。

### 構成 (実測)

2 つの対応プロファイルがあり、どちらも **int8** モデルを使う。実機 iPhone 15 では int8 が
**RTF 0.013** (ja zipformer、`modified_beam_search`) — 同じ int8 ファイルをデスクトップ
x86-64 CPU で動かすより約 4.8 倍速く、計測した中で fp32 より常に速い。つまり ARM 端末では
int8 が最小かつ最速で、大きい fp16/fp32 を積む速度上の理由はない。詳細な測定条件は
[`docs/design/mobile_quantization.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/design/mobile_quantization.md)
の「On-device (iPhone 15) verification」を参照。

| プロファイル | 端末上のモデル | ディスク | 対応言語 |
|---|---|---|---|
| `RoutingProfile.jaOnly` (既定) | ReazonSpeech zipformer int8 + Silero VAD | 約 72 MB | ja |
| `RoutingProfile.jaSenseVoice` | + SenseVoice small int8 + whisper-tiny int8 (LID プローブ) | 約 396 MB | ja / en / zh / ko / yue |

補足:

- 多言語プロファイルは 3 モデルを同時にロードする (LRU 的な破棄はしない)。396 MB は
  数 GB の RAM がある端末を前提にした、サイズと言語カバレッジのトレードオフ。
  言語切り替えが要らないなら ja-only から始めること。
- このパッケージに同梱されている refine (「清書」) の既定値はスマホ向けにチューニング済み
  (数値は英語版 README の "Pacing knobs" を参照)。
- 日本語向けの句読点復元 (デスクトップ側の BERT-char モデル、fp16 で 181.8 MB) は
  refine パスに組み込まれているが (下記「日本語の句読点復元」参照)、**どちらのダウンロード
  プロファイルにも含まれていない**。モデルファイルはまだローカルのビルド成果物で、
  ホストアプリが自分で端末に置く必要がある (下記参照)。

### ディレクトリ配置

`downloadProfile` と手動配置はどちらも 1 つの `<targetDir>` (例: アプリの Documents
ディレクトリ) の下にこの配置を作る。`HayamimiLive.start`/`LiveTranscriber.start` の
`modelDir`/`vadModelPath`/`senseVoiceModelDir`/`lidModelDir` はこれを読む:

| ディレクトリ | 必要とするプロファイル | 中身 |
|---|---|---|
| `<targetDir>/model/` | 全プロファイル | ReazonSpeech ja zipformer の `encoder`/`decoder`/`joiner` (`.int8.onnx`) + `tokens.txt` |
| `<targetDir>/vad/` | 全プロファイル (`HayamimiLive` のみ。`HayamimiRemote` にローカル VAD はない) | `silero_vad.onnx` |
| `<targetDir>/sense_voice/` | `RoutingProfile.jaSenseVoice` のみ | `model.int8.onnx` + `tokens.txt` |
| `<targetDir>/lid/` | `RoutingProfile.jaSenseVoice` のみ | `tiny-encoder.int8.onnx` + `tiny-decoder.int8.onnx` |
| `JaPunctuation` に渡す任意のパス | 日本語句読点、任意、全プロファイルで使える | `punct_bert.fp16.onnx` (181.8 MB) + `vocab.txt` (28 KB) |

### 自動ダウンロード

```dart
import 'package:path_provider/path_provider.dart';

final docsDir = await getApplicationDocumentsDirectory();
await downloadProfile(ModelProfile.jaOnly, docsDir.path); // または .jaSenseVoice

final live = HayamimiLive();
await live.start(
  modelDir: '${docsDir.path}/model',
  vadModelPath: '${docsDir.path}/vad/silero_vad.onnx',
);
```

`downloadProfile` は sherpa-onnx の GitHub リリース (`asr-models` タグ) から各アセットを
ダウンロードし、sha256 を検証し、複数精度を含むアーカイブからは `int8` のメンバーだけを
展開して上の表どおりの配置に置く。冪等なので、毎回のアプリ起動時に無条件で呼んでよい
(既にディスクにあるものはチェックサムで再検証し、欠けているか壊れているものだけ再取得する)。

日本語句読点モデル (`punct_bert.fp16.onnx` + `vocab.txt`) はダウンロードプロファイルの
対象外で、ローカルの `python scripts/quantize_punct.py --variant fp16` (メインの hayamimi
リポジトリ側) が作るビルド成果物。`ModelDownloader` にはまだエントリがないので、手動で
`JaPunctuation(modelPath: ..., vocabPath: ...)` に渡す場所へコピーすること (詳細は
[`docs/design/punct_ja.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/design/punct_ja.md))。

モデルファイルはすべてサードパーティの成果物で、それぞれ独自のライセンスを持つ (このパッケージ
自体のコードは MIT)。公開元・ライセンス・出所は
[`THIRD_PARTY_NOTICES.md`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/THIRD_PARTY_NOTICES.md)
を参照。

## 実行時設定のつまみ

以下のペーシング/VAD/デコードの既定値はスマホ向けにチューニングされているが、
組み込み先のアプリが常に同じ形とは限らない (騒がしい会場では鈍感な VAD が要るし、
講義向けアプリはもっと長い draft ウィンドウが欲しいかもしれない)。issue
[#29](https://github.com/oboroge0/hayamimi/issues/29) でこれらを外から変更できるようにした。

### ペーシング

draft (「発話中の暫定字幕」) と refine (「清書」) の各パスは、`lib/live/draft_pass.dart` と
`lib/live/refine_pass.dart` の `default*` 定数群が制御する。この 6 つは
`LiveTranscriber`/`HayamimiLive` のコンストラクタ引数であると同時に、セッション中でも
再代入できる同名プロパティになっている。setter は次回の due-check かバッファ書き込みから
効き、ネイティブモデルには触れない。不正な値 (0 以下や非有限) は即座に `ArgumentError` を
投げる:

| つまみ | 既定値 | 実行時変更 | 効果 |
|---|---|---|---|
| `draftIntervalSeconds` | 1.0s | 可 | VAD セグメントが進行中のとき、draft 再デコードを走らせる間隔 |
| `draftWindowSeconds` | 8.0s | 可 | draft デコードが再処理する末尾の音声ウィンドウ。長い発話でも draft が遅くならないようにする |
| `minDraftAudioSeconds` | 0.25s | 可 | draft デコードを走らせる価値があるとみなす最小蓄積音声長 |
| `autoRefineSilenceSeconds` | 4.0s | 可 | `autoRefineEnabled` が有効なとき、auto-refine を起動する無音間隔 |
| `autoRefineMaxBufferedSeconds` | 20.0s | 可 | 無音がなくても auto-refine を起動するバッファ長の上限 |
| `refineBufferMaxSeconds` | 60.0s | 可 | refine バッファが保持できる音声の上限。超えると一番古いセグメントから捨てる |
| `prerollSeconds` | 1.0s | 可 | 発話区間の**検出された開始点より前**の音声を、どれだけ一緒にデコードするか。ここだけ `0` が有効な値 (「プリロールなし」を意味する) |

```dart
final live = HayamimiLive(autoRefineSilenceSeconds: 2.0);
live.draftWindowSeconds = 4.0; // セッション中でも変更可
```

**`prerollSeconds` がある理由。** Silero VAD は発話が始まってから少し遅れて気づくので、
渡されるサンプルが最初の単語の途中から始まることがある。Android エミュレータでは
`資料は昨日送りました` が `昨日は昨日送りました` に化け、別の文からは `あしたの` が
先頭で消えた。プリロールは検出点の直前の音声を前に足す (デスクトップの `AudioHistory` /
`PREROLL_S` と同じ考え方で、2 セグメントがその間の隙間を両方とも含まないようクランプ済み)。
前に足した音声はその後セグメントの一部として扱われ、デコード・`audio_s`・refine バッファの
すべてに含まれる。`0` にすると VAD が区切った範囲をそのまま正確にデコードする。

### デコード方式

`start`/`startDebugWavStream` の `decodingMethod` 引数で `'greedy_search'` (速い) /
`'modified_beam_search'` (CPU を余分に使うが概ね高精度) を切り替えられる。`null` (既定) なら
単一モデル経路は sherpa-onnx の既定 `'greedy_search'`、`RoutingProfile.jaSenseVoice` の
ja 層はデスクトップ本番と同じ `'modified_beam_search'` のまま。SenseVoice には効かない。

### VAD 感度

`VadSensitivity` は Silero VAD の 4 つのつまみを外に出す: `threshold`
(発話確率のしきい値、高いほど鈍感)、`minSilenceSeconds` (セグメントが確定するまでに
必要な無音の長さ)、`minSpeechSeconds` (これより短い断片はデコード前に捨てる)、
`maxSpeechSeconds` (VAD が発話の途中でもセグメントを打ち切るまでのおおよその長さ)。

```dart
await live.start(
  modelDir: modelDir,
  vadModelPath: vadPath,
  vadSensitivity: VadSensitivity(threshold: 0.65, minSilenceSeconds: 0.8),
);

// ...セッション中、再起動なしで:
await live.setVadSensitivity(VadSensitivity(threshold: 0.4));
```

**既定値は sherpa-onnx のものではなく、デスクトップパイプラインのもの。**
`minSilenceSeconds` は **0.35s**、`maxSpeechSeconds` は **12.0s**
(`scripts/realtime_transcribe.py` と一致)。`threshold` (0.5) と `minSpeechSeconds` (0.25s) は
sherpa-onnx 自身の値。理由: sherpa-onnx 既定の 0.5s 無音しきい値だと、Android エミュレータで
3 文の日本語音声 (ポーズが 0.5 秒弱) が 1 つの 6.13 秒セグメントにまとまり、この日本語認識器は
複数発話を含むセグメントからは最後の 1 文しか返さない — 2 文が出力から消えた。0.35s では
同じ音声が 3 セグメントに分割され、3 文とも出た。実放送日本語では 0.35s は 0.5s より精度が
悪化せず、150ms 早く確定表示できると記録されている。

`maxSpeechSeconds` は保証ではなく目安。5.0s に設定したセッションでも 6.134s のセグメントが
出た実測がある。`VadSensitivity(minSilenceSeconds: 0.5, maxSpeechSeconds: 5.0)` でこの
パッケージ以前の既定値に戻せる。

### ホットワード

`start`/`startDebugWavStream` は `hotwordsFile`/`hotwordsScore` も受け付ける。
sherpa-onnx 自身の認識器レベルのホットワードバイアスで、単一モデル経路とルーティングされた
`RoutingProfile.jaSenseVoice` の ja 層に適用される (SenseVoice には効かない)。
ペーシングつまみや VAD 感度と違い、実行時の setter はない。認識器を組み立てるときに
コンパイルされるので、変更するには `start` をやり直す必要がある。

### 新しい「会話」をモデル再ロードなしで始める

`resetSession()` は今の会話に関するものだけをクリアする: refine バッファ、進行中の draft、
(ルーティングされたセッションなら) 現在ロックされている言語。ロード済みのネイティブモデルには
一切触れない。セッションが動いていなければ何もしない。

```dart
await live.resetSession(); // 会話だけ新しくする。ロード済みモデルはそのまま
```

## 日本語の句読点復元

デスクトップパイプラインでは refine (「清書」) パスのテキストが `scripts/punct_ja.py` を
必ず通るので、デスクトップの字幕は文として読める。このパッケージには対応するものが
なかったので、refine の出力は句読点なしの一続きの文字列だった。issue
[#15](https://github.com/oboroge0/hayamimi/issues/15) がこのギャップを埋めるためのもので、
希望するセッションについては解消済み。必要なモデルファイルはどちらのダウンロード
プロファイルにも含まれない (上の「モデル配置」参照)。

**有効にする方法。** `start` (エミュレータで確認するなら `startDebugWavStream`) に
`JaPunctuation` を渡す:

```dart
await live.start(
  modelDir: modelDir,
  vadModelPath: vadModelPath,
  punctuation: JaPunctuation(
    modelPath: '$punctDir/punct_bert.fp16.onnx',
    vocabPath: '$punctDir/vocab.txt',
    // デスクトップホストでは libraryPath も指定する:
    // libraryPath: '<...>/onnxruntime.dll'
  ),
);
```

既定の `null` は無効のままで、この引数が存在しなかった頃と同じ挙動。

**有効にすると何が変わるか。**

- **refine と final が対象。draft は対象外。** `RefineSubtitleEvent`・`FinalSubtitleEvent`
  (`refineEntries`/`entries` も) に 、。？ が付く。`PartialSubtitleEvent`/`drafts` は
  認識器のテキストのまま変わらない — draft は未完成の文への当て推量になるため。
- **各行に印が付く。** `punctuated` フィールド (JSON も `"punctuated"`) で、句読点モデルが
  書いたテキストか認識器がそのまま出したテキストかを消費側が区別できる。
- **どの行が対象か。** `RoutingProfile.jaSenseVoice` は日本語と判定された行だけに適用。
  単一モデルセッションには言語タグがないので、渡すこと自体が「このモデルは日本語を
  書き起こす」という宣言になる。**日本語以外のモデルを使うセッションには渡さないこと。**
- **どこで動くか。** decode worker isolate の中、認識器のあとにロードされる。
- **メモリと失敗時の挙動。** モデルはセッション中 181.8 MB 常駐 (認識器の重みに加えて最大
  396 MB)。ロード失敗はファイル名を含むメッセージで `start()` ごと失敗させる — 黙って
  句読点なしで動き続けることはしない。
- **`textTransform` は句読点が付いた後、最後に実行される。**

**なぜ final も句読点付きなのか。** refine はバッファ全体を 1 発話として再デコードするが、
この日本語認識器は複数発話音声の**最後の発話しか残さない**ので、2 セグメント以上の
グループは大抵 1 文しか返さない。`isRefineTextTooShort` がこれを検知して句読点付きの
final をつなげたものを代わりに出すので、final も句読点付きにしてある。コストは発話ごとに
句読点モデルを走らせること — Android エミュレータの実測で**11〜14 文字の行あたり約
40〜50 ms** (機能オン/オフの直接比較)。`JaPunctuation(..., applyToFinals: false)` で
断ることもでき、その場合 final は認識器そのまま、fallback した refine は
`punctuated: false` を報告する。

`PunctuatorJa` クラス自体は公開されたままで、ライブパイプライン以外が作ったテキストに
句読点を打ちたい呼び出し側からも使える:

```dart
final punctuator = await PunctuatorJa.load(
  modelPath: '<...>/punct_bert.fp16.onnx',
  vocabPath: '<...>/vocab.txt',
);
punctuator.restore('明日の会議は午後三時から始まります資料の準備をお願いします');
// -> 明日の会議は午後三時から始まります。資料の準備をお願いします。
punctuator.dispose();   // ネイティブセッションなので自動解放されない
```

1 インスタンスは 1 つの ONNX Runtime セッションで、複数の isolate から同時に使うのは
安全ではない。ライブパイプラインが自分専用のものを decode worker の中に持っているのはこのため。

このモデルは `sherpa_onnx` が音声認識用に既に同梱している ONNX Runtime を、`dart:ffi` で
直接叩いて動かす (別ランタイムを追加すると Android で 2 つの `libonnxruntime.so` が
1 プロセスに入りクラッシュする既知の問題があるため)。プラットフォームごとの検証状況は
下の「プラットフォーム状況」表の「日本語句読点」列を参照。ONNX Runtime の呼び出し順、
NFKC/MeCab の扱い、パリティテストの詳細は英語版 README の
"How it reaches ONNX Runtime, and why that way" を参照。

## スレッドモデル

sherpa-onnx が行うことはすべて**同期的な** FFI 呼び出しで、完了するまで次の Dart の行が
実行されない。Dart の isolate はシングルスレッドなので、この呼び出しの間はその isolate の
他のこと (Flutter の描画を含む) が止まる。

| 処理 | 動く場所 |
|---|---|
| マイク入力 | `record` プラグイン自身のプラットフォームスレッド。音声チャンクは呼び出し側の isolate にストリームとして届く |
| Silero VAD (32ms フレームごとに `acceptWaveform` を 1 回) | 呼び出し側の isolate |
| 認識器の構築 | decode worker isolate (自分でロードする) |
| VAD の構築 | 短命なバックグラウンド isolate (組み立ててハンドルを渡す) |
| すべてのデコード (セグメントごとの final、draft、refine、ルーティングセッションの言語判定 whisper-tiny) | decode worker isolate |
| 日本語句読点モデルの構築 (要求されたとき) | decode worker isolate、認識器のあと |
| refine 結果への句読点復元 | decode worker isolate、そのデコード直後 |
| `entries`/`drafts`/`refineEntries`/`decoding`/`errors`/`modelLoads`/`sessionResets` の発行 | 呼び出し側の isolate |

セッション 1 つにつき decode worker を 1 つ持つ。`start` が生成し `stop`/`dispose` で終了する。
worker は 1 度に 1 件ずつ処理する (以前の同期コードが自動的に持っていた「デコードは同時に
1 つだけ」という保証と同じ)。final は必ずキューに積まれ捨てられない。draft は他の仕事が
進行中なら捨てられる (final が出たセグメントの遅れた draft も破棄)。refine は合流する:
既に待っている refine があれば新規要求はその future を返す。

**デコードのレイテンシ自体は変わらない。** 仕事が動く場所が変わっただけ。
`LiveTranscriptEntry.latencyMs` は今も worker 内で計測した同じデコード時間を報告する
(メッセージ往復時間は含まない)。変わったのは、呼び出し側の isolate が待たなくてよくなったこと。

**エラー処理。** セッションは自分の失敗を `HayamimiLive.events` の `ErrorSubtitleEvent`
(`LiveTranscriber.errors` なら `LiveTranscriberException`) で報告する。`start`/`connect`
自体は不正な呼び出しに今も同期的に例外を投げる。decode worker isolate が死んだ場合は
セッションが停止し、マイク入力が破棄され、オブジェクトは再度 `start` できる状態で残る。

## プラットフォーム状況

このリリース時点で実際に動かした実績。「動くはず」という主張ではない。詳細な記録は
[`docs/design/mobile_quantization.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/design/mobile_quantization.md)
と
[`docs/verify/ios.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/verify/ios.md)。

| プラットフォーム | ASR ベンチ/ライブマイク | 多言語ルーティング | リモートモード | 日本語句読点 |
|---|---|---|---|---|
| **iOS** (実機 iPhone 15) | **検証済み。** ベンチ RTF 0.013 (ja int8、`modified_beam_search`、単発計測)。Live タブは実マイク入力を実機で最初から最後まで書き起こし、発熱や UI のカクつきの報告なし。 | 実音声で ja/en の切り替えをライブで確認。バッジ切り替えの精度は室内に複数の話者がいたため混在した結果になっており、「動く」以上の意味は持たせないこと。 | **未実施** — このリポジトリの持ち主がこの検証セッションではスコープ外とした (Mac の Wi-Fi IP が通常のプライベート LAN 範囲外だったため)。既知の失敗ではない。 | **未確認、既定で拒否。** |
| **Android** | **エミュレータのみ。** `sherpa_onnx.dart` 経由で AVD 上のロード/精度を検証 (同じ ja 15 クリップで CER 6.19% 対 PC の 5.50%、+0.69pp の差は onnxruntime のカーネル差に起因すると見ている)。エミュレータの RTF は*ホスト PC の* CPU を仮想化した数字で、実機の数字ではない。decode worker isolate は x86_64 エミュレータ (API 35) で検証済み: 起動、sherpa-onnx バインディングの初期化、全モデルの構築、isolate 境界を跨いだセッション全体のデコード、1 プロセス内で 3 回の再構築、いずれもクラッシュなし。**実機 Android デバイスでの検証はまだない。** | エミュレータでの manifest バッチ評価のみ検証。ライブマイク経路は Android では未実施 (iOS のみ)。`RoutingProfile.jaSenseVoice` は下記エミュレータ実行では検証していない (`jaOnly` のみ実行)。 | Android 固有の検証は未実施。 | **x86_64 エミュレータ (API 35) で検証済み**: ja の refine が Python 参照実装と同じ文末位置に 。 を付けて `punctuated: true` で返り、句読点モデルなしの対照実行では `punctuated: false` になった。**実機 ARM デバイスでは未実施。** |
| **Windows** (デスクトップホスト) | このパッケージの対応プラットフォームではない (`pubspec.yaml` の `platforms:` 参照)。このリポジトリ自体の `flutter analyze`/`flutter test` ゲートと下記の句読点パリティテストの実行にのみ使用。 | n/a | n/a | **検証済み** — Python 参照実装と 51 件の記録済みケースで一致 (`flutter test`)。 |
| **macOS / Linux** | 対応プラットフォームではない。 | n/a | n/a | 明示的な `libraryPath` が必要。未実施。 |

**既知の制限**: refine が長いグループを再デコードすると、この ReazonSpeech
トランスデューサは最後の 1 文しか返さないことがある (英語版 README の
"Known limitation: a refine over a long group can lose its leading sentences" 参照)。
VAD の既定値が短いポーズで分割するようになったのはこの対策の一つで、
`isRefineTextTooShort` によるフォールバックも組み込まれている。デスクトップパイプラインの
分割再試行 (`_looks_truncated` / `_split_retry`) はこのパッケージにはまだ移植されていない。

## さらに詳しく

- イベントの全項目 (`partial`/`final`/`translation`/`refine`/`error`/`model_load`/
  `session_reset` の JSON 形式と発行タイミング)、`HayamimiRemote` の詳細、
  パッケージレイアウト、dartdoc API リファレンスへのリンクは [README.md](README.md) を参照。
