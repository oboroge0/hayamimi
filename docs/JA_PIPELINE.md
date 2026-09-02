# 日本語経路の実装仕様 (JA_PIPELINE)

## 目的と適用範囲

hayamimi の日本語経路を、別プロジェクトが別言語で作り直せる程度に書き下ろした文書。
想定している再実装先は **C++ の Godot GDExtension** で、音声認識と VAD は
sherpa-onnx が配布するバイナリをそのままリンクし、句読点モデルは sherpa-onnx が同梱する
ONNX Runtime を直接叩いて動かす構成である。

この文書に書いてある数値と手順は、すべて `scripts/` の実装か、このリポジトリに既に
記録されている実測から取っている。推測は書かない。まだ測っていないことは
「未検証事項」に分けて書く。

対象は `--mode single --lang ja` の経路だけである。hayamimi 本体は 5 層の
言語ルーティング (`docs/LID.md`) を持つが、日本語しか扱わない再実装にはそのどれも要らない。
`RoutedASR.forced_lang` が `"ja"` に設定されていると、言語判定 (LID)、言語切替の確認、
zh/yue の仲裁、文字種による再デコードのすべてが `transcribe()` の中で丸ごと飛ばされる
(`scripts/asr_engine.py` の `transcribe()` 冒頭と、その中の `if self.forced_lang is not None:`
の分岐)。したがって日本語専用の実装が用意すべきモデルは、認識器・VAD・句読点の 3 つだけになる。

用語をここで一度だけ定義しておく。

- **VAD** (voice activity detection): 音声の中から発話区間だけを切り出す処理。ここでは
  Silero VAD を sherpa-onnx 経由で使う。
- **プリロール (pre-roll)**: VAD が検出した発話開始点より前の**実音声**を、認識器に渡す
  バッファの先頭に足すこと。後述の冒頭欠落対策の主役。
- **ITN** (inverse text normalization, 逆テキスト正規化): 読み上げ表記を書き言葉表記に
  戻す処理。ここでは漢数字をアラビア数字にする (`千九百四十年` → `1940年`)。
- **final / refine**: `final` は VAD セグメント 1 個ごとに即座に出す確定字幕、
  `refine` は発話グループ全体を後からまとめて再デコードし直した清書。
- **CER** (character error rate, 文字誤り率): 正解文に対する編集距離を正解文の長さで割った値。

## モデル一覧

日本語経路が読み込むファイルは 7 個。sha256 とバイト数は
`scripts/dump_ja_config.py --with-models` が実ファイルから計算したもので、
機械可読な形は `docs/ja_pipeline_spec.json` の `models` ブロックにある。

| 役割 | ファイル (models/ 以下) | bytes | sha256 | ライセンス / 出所 |
|---|---|---:|---|---|
| 認識器 encoder | `sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/encoder-epoch-35-avg-1.int8.onnx` | 70,876,409 | `ead1579e118b821a767242a8eb9272634b0e63ba16f8dfc4d126732406eae268` | Apache-2.0 / Reazon Human Interaction Lab、[reazon-research/reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2)、パッケージングは k2-fsa/sherpa-onnx |
| 認識器 decoder | `.../decoder-epoch-35-avg-1.int8.onnx` | 1,308,690 | `d0179db78a2e65445c5c3dc41e94c62068fc539fe4e45060e32f438cca76432f` | 同上 |
| 認識器 joiner | `.../joiner-epoch-35-avg-1.int8.onnx` | 1,033,417 | `c7f4ba40a8ae307a6c30b5c06e2570add04466bcb45bab62699f0ec5d00ed495` | 同上 |
| 認識器 tokens | `.../tokens.txt` | 26,631 | `144f8a4f639373a1bdf7eabb2437482ef64b0cc5db24ad27cce65f293e4faa24` | 同上 |
| VAD | `silero_vad.onnx` | 643,854 | `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6` | MIT / [snakers4/silero-vad](https://github.com/snakers4/silero-vad)、パッケージングは k2-fsa/sherpa-onnx |
| 句読点 モデル (fp32) | `mojicast-punct-onnx/punct_bert.onnx` | 363,501,157 | `4ed3c28ede4792526c6abab9101a3b6c304ab09fd0bda4e318e0acb2b7008e63` | Apache-2.0 / 本体 [tohoku-nlp/bert-base-japanese-char-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v3) + ヘッド [bobfromjapan/bert_japanese_punctuation](https://huggingface.co/bobfromjapan/bert_japanese_punctuation)、ONNX 化 [ishiki-emo/mojicast-punct-onnx](https://huggingface.co/ishiki-emo/mojicast-punct-onnx) |
| 句読点 vocab | `mojicast-punct-onnx/vocab.txt` | 27,928 | `57411bcac5e9559f2aa4d316a2217289048cb40fe23187b02a81aeb3e5d61cf3` | 同上 |

**合計 437,418,086 bytes (417.2 MiB)**。句読点モデルだけで 83% を占める。

### 上流の INT8 句読点モデルは使ってはいけない

Hugging Face の `ishiki-emo/mojicast-punct-onnx` は `punct_bert.int8.onnx` (動的 INT8) を
既定として案内している。hayamimi はこれを試して**壊れていることを確認した**。
onnxruntime 1.29.0 の CPU EP 上で、乱数トークン列を入れても実文を入れても logits が
ほぼ一定 (読点・句点の確率がどこでも 0.31〜0.36 の範囲) で、入力に反応しない。
つまり句読点を復元しない。原因は特定していない (壊れたエクスポート、アップロードミス、
特定の onnxruntime ビルドとの非互換のいずれか)。詳細は `docs/PUNCT_JA.md`。

**再実装ではこのファイルを使わないこと。** 使うなら fp32 (`punct_bert.onnx`) か、
下の fp16 のどちらかにする。

### fp32 と fp16 のどちらを選ぶか

hayamimi 自身で量子化をやり直した 3 通りを、FLEURS ja 250 文 (seed 0) で
`scripts/quantize_punct.py` により採点した。合格条件は事前宣言で「F1 が fp32 比 −0.02 以内」。

| variant | サイズ | P | R | F1 | 判定 |
|---|---:|---:|---:|---:|---|
| fp32 (`punct_bert.onnx`) | 363.5 MB | 0.8724 | 0.4831 | 0.6218 | 基準 |
| 動的 INT8 | 91.4 MB | 0.9638 | 0.3757 | 0.5407 | **不合格** (−0.0812) |
| 静的 INT8 (QDQ, 較正あり) | 91.3 MB | 0.9275 | 0.3432 | 0.5010 | **不合格** (−0.1208) |
| **fp16** | **181.8 MB** | 0.8724 | 0.4831 | **0.6218** | **合格 (差 ±0.0000)** |

fp16 は `onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`
による全グラフ変換で、入出力は int64/float32 のまま残る。この 250 文と `eval_real` の
両方で fp32 と**予測が 1 個も変わらなかった**。INT8 側はどちらも precision が上がって
recall が崩れるという同じ壊れ方をしており、この BERT-char ヘッドの閾値付近の判断境界を
量子化すると起きる性質だと見ている (レシピの問題ではない)。

判断:

- **PC (デスクトップ) では fp32 を使う。** サイズ制約がないので、わざわざ変換する理由がない。
  `scripts/punct_ja.py` の既定はこれ。
- **メモリやパッケージサイズに制約がある実装では fp16 を使う。** 精度は fp32 と同一、
  サイズは半分。合計は 255,720,140 bytes (243.9 MiB) まで落ちる。
  Android エミュレータでの実動作確認は fp16 で行っている (後述)。
- fp16 の速度については注意がある。この Windows/x86 機では fp16 の方が **10 倍遅い**
  (`restore()` 平均 532 ms 対 fp32 の 52〜78 ms)。x86 の CPU EP に fp16 の演算経路がなく、
  演算ごとに fp32 へキャストして戻しているためで、NEON fp16 を持つ ARM では話が変わる。
  実機で測り直すまで、この遅さを ARM に外挿してはいけない。

## 認識器設定

`scripts/asr_engine.py` の `_build_reazon()` が唯一の構築点。sherpa-onnx の
`OfflineRecognizer.from_transducer` に渡す値は全部これで、C++ の
`OfflineRecognizerConfig` に一対一で対応する。

| 項目 | 値 | 出所 |
|---|---|---|
| `model_type` | `"zipformer"` | `asr_engine.RZ_MODEL_TYPE` |
| `decoding_method` | `"modified_beam_search"` | `asr_engine.RZ_DECODING_METHOD` |
| `modeling_unit` | `"cjkchar"` | `asr_engine.RZ_MODELING_UNIT` |
| `num_threads` | `4` | CLI `--threads` の既定値 |
| `hotwords_file` | `""` (未使用) | `_build_reazon` の既定 |
| `hotwords_score` | `2.0` | `asr_engine.RZ_HOTWORDS_SCORE` |
| feature `sampling_rate` | `16000` | sherpa-onnx の `FeatureExtractorConfig` 既定 |
| feature `feature_dim` | `80` | 同上 (80 次元 fbank) |

特徴量は sherpa-onnx の既定のままなので Python 側では何も指定していない。だが C++ で
`OfflineRecognizerConfig` を自分で組む場合は明示することになるので、値をここに書いておく:
**16 kHz、80 次元 log-Mel フィルタバンク**。入力音声も 16 kHz モノラルに揃える
(`realtime_transcribe.read_wave()` が違うレートを線形補間でリサンプルしている)。

### なぜ modified_beam_search か

`greedy_search` から替えたのは実測に基づく。実放送日本語で **CER 8.6% → 5.8%**、
デコード時間は +25% (それでも実時間の 37 倍速)。英語の v3 tier では改善が出なかったので
そちらは greedy のままにしてある (`_build_reazon` のコメント)。

### ホットワードは事実上使えない

`--hotwords` は API としては通っているが、この認識器では効かない。ReazonSpeech の
`tokens.txt` はバイトレベル BPE で、`bpe.model` が同梱されていない。そのため
`cjkchar` を含めどの `modeling_unit` でもホットワードをエンコードできない。
sherpa-onnx はエンコード失敗を stderr の警告として出すだけで終了コードは 0、
字幕も普通に出てしまうので、気付かないまま何も効いていない状態になる (GitHub issue #1)。
hayamimi は起動時に `tokens.txt` に対してエンコード可能性を検査して、
できないときは大きな警告を出し、`warning` イベント (`code: "hotwords_unencodable"`) も
publish する (`RoutedASR._warn_hotwords_encodability`)。

**再実装への指示**: 固有名詞対策はホットワードではなく、後段の置換辞書
(`--replace`。下の「`forced_lang="ja"` の経路」の手順 10) でやること。ホットワード機能を UI に出すなら、この制限を書くか、
そもそも出さない方がよい。

## VAD 設定

`scripts/realtime_transcribe.py` の `build_vad()` が唯一の構築点。

| 項目 | 値 | 定数 |
|---|---|---|
| モデル | `silero_vad.onnx` | `VAD_MODEL` |
| `threshold` | `0.5` | `build_vad(vad_threshold=...)` の既定 |
| `min_silence_duration` | `0.35` 秒 | `build_vad(min_silence=...)` の既定 |
| `min_speech_duration` | `0.25` 秒 | `VAD_MIN_SPEECH_S` |
| `max_speech_duration` | `12.0` 秒 | `build_vad(max_speech=...)` の既定 |
| `window_size` | `512` サンプル (16 kHz で約 32 ms) | `WINDOW_SIZE` |
| `sample_rate` | `16000` | `SAMPLE_RATE` |
| `buffer_size_in_seconds` | `30.0` | `VAD_BUFFER_S` |
| `num_threads` | `1` | `VAD_NUM_THREADS` |

音声は 512 サンプルちょうどの塊で `accept_waveform()` に渡し、そのつど `empty()` を見て
溜まったセグメントを `front` / `pop` で取り出す。

判断の根拠:

- **`min_silence` 0.35 秒**: 実放送 ja 15 クリップで 0.5 → 0.35 → 0.30 と振って
  CER が変わらなかった (分割数は 17 → 19 に増えた)。精度コストがゼロで確定表示が
  一律 150 ms 早くなるので 0.35 を採用した (`docs/BENCHMARKS.md` 改善イテレーション#9)。
- **`threshold` 0.5**: AMI 評価セットで 0.40 / 0.30 / 0.20 を掃引したところ、
  miss は期待どおり減ったが confusion がそれ以上に増え、平均 DER が全ての値で悪化した
  (14.1% → 15.6〜16.5%)。下げる案は却下し、Silero の既定 0.5 のままにしてある
  (`docs/DIARIZATION_PLAN.md` section 13)。
- **`max_speech` 12 秒**: 息継ぎのない実況で 21 秒のセグメントが出て確定が遅れたので入れた。
  ただし Android エミュレータでの実測で、sherpa-onnx はこの値を強制分割ではなく
  「目安」として扱うことが分かっている (5.0 秒設定で 6.134 秒のセグメントが出た)。
  遅延の上限としては当てにできない。

**sherpa-onnx 1.13.6 の `SileroVadModelConfig` には発話前後のパディング設定がない**
(`speech_pad_ms` に相当するフィールドが存在しない)。したがって「VAD 側で冒頭に余白を足す」
という解決は、この構成では取れない。次章の対策が必要になるのはこのためである。

## 冒頭欠落と対策

### 現象

この ReazonSpeech zipformer は、発話がバッファの先頭 (サンプル 0) から始まっていると、
先頭のトークンを落とすことがある。落ちるだけでなく、別の語に化けることもある。

hayamimi 側で観測した具体例:

- オフライン分割の実装中、断片が発話の立ち上がりちょうどから始まっていたとき、
  `東京の天気は晴れです` の `東京の` が丸ごと消えた (`_speech_pieces()` のコメント)。
- ライブ経路の検証で、プリロールを 0 にすると `資料は昨日送りました` が
  `昨日は昨日送りました` になった (`docs/BENCHMARKS.md` 2026-09-01)。
- Android エミュレータでも同じことが再現した。プリロール 0 で
  `昨日は昨日送りました` / `会議は十時からです` (`あしたの` が欠落)、
  プリロール 1.0 秒でどちらも desktop と同じ `資料は昨日送りました` /
  `あしたの会議は十時からです` になった (`docs/MOBILE.md`、`agent/feature/core-release`
  ブランチの "Android emulator verification, run 2")。この run は制御実験になっていて、
  VAD の値だけ直したパス (`r2-preroll0`) と両方直したパス (`r1-defaults`) を
  別々に走らせ、どちらの修正がどちらの効果を出したかを切り分けている。

Silero の発話開始検出が真の立ち上がりより遅れることも実測してある。上記フィクスチャの
第 1 文で **198 ms 遅れ**、第 2・3 文は ±25 ms 以内だった。ただし壊れ方は遅れ量に
比例しない。遅れが 18 ms しかないセグメントでも冒頭語が化けた。つまり
「VAD の遅れを補正する」より「先頭に何かを置く」ことが効いている。

### 対策 1 (主): 実音声のプリロール 1.0 秒

hayamimi の標準対策。`PREROLL_S = 1.0` 秒。

`realtime_transcribe.AudioHistory` が直近 30 秒の入力音声をリングバッファで保持し、
`with_preroll(seg_start, seg_samples)` が VAD セグメントの前に最大 1.0 秒の**実音声**を
足してから認識器に渡す。足す量は 3 方向にクランプされる:

```
want = max(seg_start - PREROLL_S * sr,   # 1 秒より前には遡らない
           self.last_seg_end,            # 直前のセグメントには食い込まない
           self.offset)                  # 履歴に残っている範囲を超えない
```

`last_seg_end` のクランプが重要で、これがないと前の発話の末尾を巻き込んで二重に認識する。
実際に足された量は状況依存で、Android の測定では 0.198 / 0.356 / 0.420 秒だった
(1.0 秒フルではない)。

効果は大きい。実放送 ja 15 クリップの VAD 経由 CER が **プリロールなし 40.2% →
プリロール 0.8 秒で 15.5%** (`docs/BENCHMARKS.md` 改善イテレーション#9。当時は 0.8 秒、
現在の実装は 1.0 秒)。クリップ丸ごとのオフラインデコード (参考上限) が 8.6% なので、
残差 15.5% − 8.6% はストリーミング分割そのものによる文脈喪失で、これは別の問題である。

この不変条件は `tests/test_asr_segment.py::test_live_path_preroll_keeps_utterance_initial_words`
で固定してある。3 文のフィクスチャを VAD → プリロール → デコードの実経路に通し、
各文の冒頭語が残ることを要求する。

### 対策 2 (副): 疑わしいときだけの分割再試行

v0.3.1 で入れた。`_looks_truncated()` と `_split_retry()` (`scripts/asr_engine.py`)。

長いバッファを内部の無音で切って 1 発話ずつデコードし直すと、冒頭を落としたクリップは
直る (FLEURS ja clip 15 は CER 0.67 → 0.11)。しかし**無条件にやると全体では損**で、
外部の FLEURS 5×100 A/B では ja 8.6% → 9.9%、en 9.4% → 10.2%、ko 8.1% → 9.1% と悪化した
(ja では 16 クリップが改善して 26 クリップが悪化した。断片の境界にまたがる語が消え、
短い断片ほどデコードの条件が悪くなるため)。

そこで**再試行**にしてある。バッファ全体を今までどおりデコードし、その結果が
冒頭を落としたように見えるときだけ分割を試し、分割結果が明らかに良いときだけ採用する。

- 疑いの判定 (`_looks_truncated`): 発話秒あたりの英数字数 (句読点は数えない) が
  `DENSITY_FLOOR_CJK = 2.4` 未満なら疑う。FLEURS ja 60 クリップの実測が 3.46〜14.22
  (中央値 6.54)、既知の欠落クリップが 1.70 で、2.4 はその谷の対数中点にあたる。
  非 CJK は `DENSITY_FLOOR_LATIN = 6.0`。バッファが `SEGMENT_MIN_S = 4.0` 秒以下なら
  そもそも疑わない。空文字列も疑わない。
- 分割 (`_speech_pieces`): 同じ Silero VAD を使い、`SEGMENT_MIN_SILENCE_S = 0.35` 秒
  (ライブ VAD の `min_silence` と**わざと同じ値**) 以上の無音で切る。同じ値にしてあるので、
  ライブ VAD が作ったセグメントは必ず 1 個にしか割れず、再試行が自動的に無効になる。
- 断片の作り方: 各断片は `SEGMENT_PAD_S = 0.35` 秒の**ゼロサンプル** + VAD スパンを前後
  0.35 秒の**実音声**で広げたもの + 0.35 秒のゼロサンプル、という構成。先頭の無音を
  合成するのは、実音声の側だけでは VAD の遅れを吸収しきれず、断片が発話の立ち上がりから
  始まってしまうことがあるため。
- 採用の判定 (`_retry_is_better`): 再試行結果が (a) 元より長く、密度が健全な範囲に戻り、
  かつ (b) 元のテキストの末尾 `RETRY_TAIL_CHARS = 12` 文字のうち
  `RETRY_TAIL_MATCH = 0.6` 以上が再試行結果の中に (最長共通部分文字列として) 現れる、
  の両方を満たしたときだけ。元のデコードは発話の**生き残った末尾**なので、
  本物の復旧ならそれは残っているはずだ、という理屈。迷ったら元を残す。

**これはライブ経路には効かない** (効かなくてよい)。ライブ経路は VAD セグメント 1 個ずつ
デコードするので内部に切れる無音がなく、短くて密度も高いので疑いの門も通らない。
オフラインでファイル全体を 1 回で流し込むときの保険である。

### 代替案: 300 ms の無音を前置する

別プロジェクトが同じ現象を greedy_search で観測し、**300 ms の無音をバッファ先頭に足す**
ことで回避している。そちらの VAD は発話開始前の音声を保持していないので、実音声の
プリロールが構造上できず、合成無音が唯一の選択肢だった。

**hayamimi はプリロール (実音声 1.0 秒) を標準とする。** 記録に残っている実測が
そちら側にしかないためで、無音前置が劣ると測ったわけではない。実音声を持っている
実装ならプリロールを、持っていない実装なら無音前置を選ぶ、という切り分けでよい。
なお hayamimi もオフライン分割の断片には合成無音を使っている (上記 `SEGMENT_PAD_S`)。
実音声とゼロサンプルは排他ではなく、両方使う場面がある。

### 発生率の実測

> **プレースホルダ**: greedy_search / modified_beam_search × 対策なし / プリロール /
> 300 ms 無音前置 の 6 条件で、冒頭欠落の発生率を測る作業が別途進行中である。
> **計測結果は `docs/HEAD_DROPOUT.md` を参照。数値は後でここに転記する。**
> この表を埋めるまで、条件間の優劣を数値で語らないこと。

| 条件 | 発生率 | 備考 |
|---|---|---|
| greedy_search / 対策なし | (未転記) | |
| greedy_search / プリロール 1.0 s | (未転記) | |
| greedy_search / 無音 300 ms | (未転記) | |
| modified_beam_search / 対策なし | (未転記) | |
| modified_beam_search / プリロール 1.0 s | (未転記) | |
| modified_beam_search / 無音 300 ms | (未転記) | |

### 再現手順

`testdata/multi_sentence_ja.wav` (edge-tts で合成した 3 文、間隔 0.5 秒、6.26 秒、
16 kHz モノラル。助走の無音なしで発話が立ち上がる) を使う。

現行構成での正解:

```
python scripts/realtime_transcribe.py --wav testdata/multi_sentence_ja.wav \
    --no-realtime --mode single --lang ja --threads 4
```

```
[ja/rz] 東京の天気は晴れです。       (seg=1.8s)
[ja/rz] あしたの会議は十時からです。  (seg=2.4s)
[ja/rz] 資料は昨日送りました。        (seg=2.1s)
```

`realtime_transcribe.PREROLL_S` を 0 にして同じコマンドを流すと、第 3 文が
`昨日は昨日送りました` に化ける。これが対策 1 の効いている証拠になる。

同じファイルで、VAD を通さず丸ごと 1 回でデコードすると `資料は昨日送りました` の
1 文しか返らない。これは Windows x86 でも Android x86_64 でも同じで、
**この認識器は複数発話を含む音声を渡すと最後の 1 発話しか返さない**。
VAD による分割は最適化ではなく正しさのために必須である。

## `forced_lang="ja"` の経路

音声入力から確定テキストまでの呼び出し順。括弧内は実装の場所。

1. **16 kHz モノラル float32 に揃える** (`read_wave` / マイク入力)。
2. **512 サンプルずつ VAD に渡す** (`run_stream`)。同時に `AudioHistory.push()` で
   リングバッファにも積む。
3. **セグメントが溜まったら取り出す** (`drain_segments`: `vad.empty()` → `vad.front` → `vad.pop()`)。
4. **プリロールを前置する** (`AudioHistory.with_preroll`)。ここで渡す音声が確定する。
5. **zipformer でデコードする** (`RoutedASR.transcribe` → `_route("ja")` → `_decode`)。
   `forced_lang` が設定されているので LID も言語切替も走らない。
6. **冒頭欠落の疑い判定と分割再試行** (`_looks_truncated` → `_split_retry`)。
   ライブ経路では実質 no-op。**任意**。
7. **second-opinion ゲート** (`_maybe_second_opinion`)。parakeet-ja でもう一度デコードして、
   両者の相互 CER が 0.25 以下なら parakeet の結果を採る。**既定で無効**
   (`ja_second_opinion=False`)、有効にしても refine パスにしか適用されない。
   モデルが 1 個増えて RSS が約 250 MB 増える。**任意**。
8. **CJK ITN** (`itn_cjk.convert(text, "ja")`)。漢数字 → アラビア数字。
9. **句読点復元** (`punct_ja.PunctuatorJa.restore`)。
10. **ユーザ置換辞書** (`RoutedASR._replace`)。`--replace` で与えた `wrong=right` を適用する。
    **常に最後**なので、ITN や句読点が作った結果も上書きできる。**任意**。

8 → 9 → 10 の順序は固定で、`scripts/itn_cjk.py` の module docstring が根拠を書いている。
ITN を先にするのは、句読点復元が正規化済みの数字を見るようにするため (小数点で効く)。
置換辞書を最後にするのは、ユーザの指定が常に勝つようにするため。

### 段階の必須 / 任意

| 段階 | 必須か | 落とすとどうなるか |
|---|---|---|
| VAD 分割 | **必須** | 複数発話を渡すと最後の 1 文しか返らない |
| プリロール | **必須に近い** | 実放送 ja で CER 15.5% → 40.2% |
| zipformer デコード | **必須** | — |
| 分割再試行 | 任意 | 長いオフラインバッファで冒頭の文を落とす場合がある |
| second opinion | 任意 (既定 off) | 精度が少し下がるがモデル 1 個ぶん軽い |
| ITN | 任意 | 漢数字のまま出る (`千九百四十年`) |
| 句読点復元 | 任意 | 句読点なしの平文が出る |
| 置換辞書 | 任意 | 固有名詞の誤りを直す手段がなくなる |

### C++ 側の最小構成

日本語専用の再実装が最低限持つべきもの:

- sherpa-onnx の `OfflineRecognizer` を上表の設定で 1 個。
- sherpa-onnx の `VoiceActivityDetector` を上表の設定で 1 個。
- 1.0 秒ぶん (16 kHz なら 16000 サンプル) を保持できる音声リングバッファと、
  「直前のセグメント末尾」を覚えておく変数 1 個。実装は `AudioHistory` が 30 行程度なので
  そのまま移せる。
- 句読点モデル用の ONNX Runtime セッション 1 個 (次章)。
- ITN は純粋な文字列処理で依存がない。`scripts/itn_cjk.py` は 239 行、外部依存ゼロ、
  正規表現の量指定子はすべて有界 (catastrophic backtracking がない) なので、
  そのまま移植できる。

分割再試行と second opinion は最初から入れなくてよい。

## 句読点復元の C++ 再現仕様

`scripts/punct_ja.py` の `PunctuatorJa` の完全仕様。Dart 移植が
`agent/feature/core-release` ブランチにあるので、C++ を書く前にそちらを読むと早い
(下の「参照実装」)。

### モデルの入出力

char-level BERT のトークン分類。

| | 名前 | 型 | 形 |
|---|---|---|---|
| 入力 1 | `input_ids` | int64 | `[1, seq]` |
| 入力 2 | `attention_mask` | int64 | `[1, seq]` (全部 1) |
| 出力 | `logits` | float32 | `[1, seq, 2]` |

`token_type_ids` は要らない (モデルが要求しない)。バッチも組まない (1 文字列 1 呼び出し)。

`logits[0][i]` は列 0 が**読点 (、)**、列 1 が**句点 (。)** の logit。位置 `i` は
「`input_ids[i]` の**直後**」を意味する。`input_ids[0]` は `[CLS]` なので、
文字 `chars[i]` に対応する logit は `logits[0][i+1]` になる。ここを 1 ずらすと
句読点が 1 文字ずつずれるという分かりにくい壊れ方をする。

fp16 モデルも `keep_io_types=True` で変換してあるので入出力の型は変わらない。
出力の要素型が float32 でなければ**別のエクスポートを掴んでいる**ということなので、
そのまま float32 として読まずにエラーにすること (Dart 版はそう実装してある)。

### 語彙

`vocab.txt` は標準的な BERT の vocab で、**行番号がそのままトークン ID** になる
(0 始まり)。行末の改行だけ剥がして、それ以外は加工しない。必要な特殊トークンは
`[PAD]` `[UNK]` `[CLS]` `[SEP]` の 4 つで、いずれも vocab から引く (ID をハードコードしない)。
語彙にない文字は `[UNK]` に落とす。

### トークン化 (MeCab は要らない)

元の `BertJapaneseTokenizer` は「NFKC 正規化 → MeCab で形態素分割 → 各形態素を 1 文字ずつに
分割」というパイプラインで、`scripts/punct_ja.py` は fugashi + unidic-lite でそれを
再現している。しかし**形態素はその場で 1 文字ずつにばらされる**ので、MeCab の唯一の
観測可能な効果は**空白が落ちること**だけである。

これは検証してある。`scripts/make_punct_fixture.py` が

```
NFKC 正規化 → 空白文字を除去 → コードポイントに分割
```

という規則と MeCab 版を突き合わせ、**102 個のテキストで不一致 0 件**だった。内訳は
FLEURS ja の参照文 (46 種) とその句読点を剥がしたもの、それに合成ケース 10 個
(空文字列、空白のみ、句読点のみ、半角カナ、全角 ASCII、`モーニング娘。` を含む文など)。
`eval_real` の ja 15 文でも一致している。

**したがって C++ 実装は MeCab を入れなくてよい。** 上の 3 段の規則で十分である。
移植したら同じ突き合わせを一度やって、0 件を自分の目で確認すること。

`[CLS]` + 文字列 + `[SEP]` が最終的な `input_ids`。文字数は **500 文字** で切る
(`PUNCT_MAX_CHARS`)。モデルの位置埋め込みは 512 で、`[CLS]`/`[SEP]` の 2 個ぶんの余裕を
見た値。超過ぶんはチャンク分割せず**黙って捨てる**。1〜2 文の発話単位で呼ぶ前提なので
実用上は当たらない (FLEURS ja の最長参照文が 133 文字)。

### NFKC を飛ばすとどうなるか

飛ばしてはいけない。全角の英数字 (`１５`) や半角カナ (`ｱｼﾀ`) がそのまま語彙引きに行き、
`[UNK]` だらけになってモデルの予測が壊れる。逆に、NFKC を掛けたぶん**出力も正規化後の
文字列になる**ことを受け入れる必要がある。入力の全角英数字は半角で返る。

### 後処理の順序

`restore()` の中身をそのままの順序で:

1. 入力を `strip()`。空なら**そのまま返す** (モデルを呼ばない)。
2. トークン化 (上記)。結果が空ならそのまま返す。500 文字で切る。
3. `input_ids` / `attention_mask` を組んでモデルを実行。
4. `probs = sigmoid(logits)`。
5. 文字 `chars[i]` を順に出力しながら、各文字の後に印を入れるか決める。
   `comma_p, period_p = probs[i+1]` として:
   - `chars[i]` 自身が句読点集合に含まれるなら **何もしない** (連続させない)。
   - 次の文字 `chars[i+1]` が句読点集合に含まれるなら **何もしない** (直前に重ねない)。
   - `period_p >= 0.5` なら `。` を入れる。ただし最後の文字のときは
     `force_final_period` が真のときだけ。
   - そうでなく `comma_p >= 0.5` なら `、` を入れる。
   句点と読点が両方閾値を超えたら**句点が勝つ**。
6. `force_final_period` が真で、結果の末尾が句読点集合に含まれないなら `。` を足す。
7. 疑問符の規則 (下記) を掛ける。

閾値は読点・句点とも **0.5**。`force_final_period` の既定は **真**。

句読点集合 (`_JA_PUNCT_CHARS`) はこの 21 文字:

```
。 、 ！ ？ ! ? … 「 」 『 』 （ ） ( ) 【 】 ・ , . \n
```

半角と全角が両方入っているのは、NFKC が `！？（）` を `!?()` に畳んでからこの集合を
引くため。片方だけだと判定が漏れる。

### 疑問符の規則

モデルは読点と句点の 2 クラスしか持たない。`？` はモデルの出力ではなく、
文末の語尾を見る手書きの規則である (`_apply_question_marks`)。

実装: 出力を `。` で `split` し、空でない各セグメントについて、末尾が下の接尾辞の
いずれかで終わるなら `？` を、そうでなければ `。` を付け直して連結する。

接尾辞は**この 10 個**、この順序 (`_QUESTION_SUFFIXES`):

```
ですか  ますか  でしょうか  かな  かしら  かい  の  だろうか  でしたか  ましたか
```

限界も明記しておく。この規則は両方向に外す。イントネーションだけの疑問文は取れないし、
準体助詞の `の` で終わる平叙文には誤爆する。`！` は一切扱わない。それでも移植側は
**この 10 個を一字一句このまま**持つこと。desktop と同じテキストが出ることが、
移植が正しいことの唯一の確認手段だからである。

### 既知の癖: 全角 `？` で終わる入力

入力が全角 `？` で終わっていると、出力が `?。` になる。NFKC が `？` を半角 `?` に畳み、
その `?` は句読点集合に入っているので `_apply_question_marks` の `。` 付け足しが
別の経路で動いてしまう。バグだが、desktop がそうなっている以上、
**移植側も同じ挙動にしておく** (直すなら両方同時に直す)。

### ONNX Runtime の呼び出し順

sherpa-onnx が同梱している `libonnxruntime.so` / `onnxruntime.dll` をそのまま使う。
2 つ目の ONNX Runtime を同じプロセスに入れると落ちる
([sherpa-onnx#3261](https://github.com/k2-fsa/sherpa-onnx/issues/3261))。

エントリポイントは `OrtGetApiBase()`。そこから `GetApi(version)` で `OrtApi` の
関数テーブルを取る。Dart 版が実際に呼んでいるのは `OrtApi` の 27 メンバ
(それに `OrtApiBase` の 2 個と `OrtAllocator` の 3 個):

- 生存管理: `CreateEnv`, `CreateSessionOptions`, `SetIntraOpNumThreads`,
  `SetInterOpNumThreads`, `CreateSession`, `CreateCpuMemoryInfo`,
  `GetAllocatorWithDefaultOptions`
- グラフの検査: `SessionGetInputCount`, `SessionGetOutputCount`,
  `SessionGetInputName`, `SessionGetOutputName`
- 推論: `CreateTensorWithDataAsOrtValue`, `Run`, `GetTensorTypeAndShape`,
  `GetTensorElementType`, `GetDimensionsCount`, `GetDimensions`, `GetTensorMutableData`
- エラー: `GetErrorCode`, `GetErrorMessage`
- 解放: `ReleaseStatus`, `ReleaseSessionOptions`, `ReleaseValue`,
  `ReleaseTensorTypeAndShapeInfo`, `ReleaseMemoryInfo`, `ReleaseSession`, `ReleaseEnv`
- `OrtApiBase`: `GetApi`, `GetVersionString` / `OrtAllocator`: `Alloc`, `Free`, `Info`

順序と注意点:

1. `CreateEnv(ORT_LOGGING_LEVEL_WARNING, "hayamimi_punct", &env)`。
2. `CreateSessionOptions(&options)` → `SetIntraOpNumThreads(options, 2)`
   → `SetInterOpNumThreads(options, 1)`。認識器と同居するのでスレッドは絞る
   (Python 側は intra-op 4 だが、認識器と別プロセスではないので実装に合わせて決めてよい)。
   Execution provider は**登録しない**。何も登録しなければ CPU provider に落ちる。
3. **`CreateSession(env, model_path, options, &session)` はファイルパスで呼ぶ**。
   バイト列から読むと 182〜364 MB をいったんヒープに載せて runtime にコピーし直すことになる。
   パスで渡せば runtime が自分で mmap する。
   **`model_path` の型は `ORTCHAR_T*` で、Windows では `wchar_t*`、それ以外では `char*`**。
   ここを間違えると「ファイルがない」という分かりにくいエラーになる。
4. `CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info)`。
5. グラフの検証 (任意だが推奨): `SessionGetInputCount` / `SessionGetInputName` で
   入力が `input_ids`, `attention_mask` の順で始まること、`SessionGetOutputName` に
   `logits` があることを確認する。名前は ORT のアロケータが確保して返すので、
   **同じアロケータの `Free` で返す** (自前の free を使わない)。
   違うモデルを掴んだときに `Run` の不可解なエラーではなく読めるメッセージで落ちる。
6. 推論ごとに: `CreateTensorWithDataAsOrtValue` で `input_ids` と `attention_mask` の
   int64 テンソルを 2 個作る (shape は `{1, seq}`、要素型 `ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64`)。
   → `Run(session, nullptr, input_names, inputs, 2, output_names, 1, outputs)`。
7. 出力: `GetTensorTypeAndShape` → `GetTensorElementType` が float32 か確認 →
   `GetDimensionsCount` が 3 か確認 → `GetDimensions` が `{1, seq, 2}` か確認 →
   `GetTensorMutableData` でポインタを取る。**バッファは `OrtValue` の所有物**なので、
   使い終わる前に `ReleaseValue` してはいけない。コピーしてから解放する。
8. 毎回 `ReleaseValue` (入力 2 個 + 出力 1 個)、`ReleaseTensorTypeAndShapeInfo`。
   終了時に `ReleaseMemoryInfo` → `ReleaseSession` → `ReleaseEnv`。
   `GetAllocatorWithDefaultOptions` が返すアロケータはプロセス所有なので**解放しない**。
9. どの API も `OrtStatus*` を返す。非 NULL がエラー。`GetErrorCode` / `GetErrorMessage` で
   内容を取ってから **必ず `ReleaseStatus`** する。

### ONNX Runtime の互換性について分かっていること

- **Android x86_64 エミュレータ (API 35)**: sherpa_onnx が APK に入れる
  `lib/x86_64/libonnxruntime.so` (25,000,408 bytes) を `dlopen` 相当で開き、
  `OrtGetApiBase` が解決でき、C API version 11 を要求して通り、runtime は自分を
  **ONNX Runtime 1.27.1** と名乗った。APK 内の `libonnxruntime.so` は 1 個だけで、
  2 つ目は追加されていない。その上で 181.8 MB の fp16 モデルを実際に読み込んで
  `restore()` を回している。
- **Windows**: `sherpa_onnx_windows` が同梱する `onnxruntime.dll` (同じく 1.27.1) で
  同じことを確認している。
- **iOS: 未確認**。Dart 版は iOS では `OrtLibrary.processSymbols` (プロセスに既にロード
  されているシンボルを引く) 経路を想定しているが、実機でもシミュレータでも走らせていない。
  Dart 版は既定で iOS での読み込みを拒否する実装になっている。

sherpa-onnx 1.13.6 が同梱する ONNX Runtime が 1.27.1 だという事実は、Android と Windows の
両方で runtime 自身に名乗らせて確認したものである。

### 参照実装とパリティ用フィクスチャ

`agent/feature/core-release` ブランチに Dart 移植がある。C++ を書く前に読むとよい。

| ファイル | 中身 |
|---|---|
| `mobile/hayamimi_core/lib/punct/ort_bindings.dart` | `OrtApi` の関数テーブル定義。使うメンバだけ実型で、残りは `Pointer<Void>` のプレースホルダ。`OrtApi` は追記のみという ABI 約束があるので、古いヘッダから作った表が新しい runtime でも通る理由もここに書いてある |
| `mobile/hayamimi_core/lib/punct/punct_ort_session.dart` | 上の呼び出し順の実装。`ORTCHAR_T` の扱いもここ |
| `mobile/hayamimi_core/lib/punct/punct_ja_tokenizer.dart` | vocab 読み込みと MeCab なしトークン化 |
| `mobile/hayamimi_core/lib/punct/punct_ja_text.dart` | 閾値・印の挿入・疑問符規則。句読点集合と接尾辞リストの定数もここ |
| `mobile/hayamimi_core/test/fixtures/punct_ja_parity.json` | **パリティ用フィクスチャ** |
| `scripts/make_punct_fixture.py` | それを生成するスクリプト (このブランチにもある) |

フィクスチャの形式は、ヘッダ (生成元・出典・使ったモデル・onnxruntime のバージョン・
`unicodedata` のバージョン・閾値) と `cases` 配列。各ケースは:

```json
{"name":"fleurs_00","source":"fleurs",
 "input":"海の下は薄く高地の下は厚くなっています",
 "input_ids":[2,3348,464,...,3],
 "expected":"海の下は薄く高地の下は厚くなっています。"}
```

`input_ids` を別に記録しているのが肝で、これがあると**モデルを持っていない環境でも
トークン化だけを検証できる**。移植でいちばん壊れやすいのがトークナイザなので、
182 MB のモデルなしで半分をカバーできるのは大きい。C++ 側も同じファイルを読んで
同じ 2 段 (ids と最終文字列) を突き合わせるべきである。

FLEURS 由来のケースは CC BY 4.0 で、`、。？` を除去して ASR 風の無句読点入力にしてある。
FLEURS ja には疑問文が 1 つもないので、疑問符規則・空文字列・句読点のみ・半角カナ・
全角 ASCII・500 文字超過は合成ケースで補ってある。

## ゴールデンテスト

`tests/golden/ja/` に FLEURS ja の 8 クリップと、日本語経路が現在それらに対して出す
テキストが `golden.json` として置いてある。詳細は `tests/golden/ja/README.md`。

要点だけ:

- 生成は `scripts/make_ja_golden.py`。`realtime_transcribe.main()` が
  `--wav X --no-realtime --mode single --lang ja --threads 4` で組むのと同じ
  オブジェクト (`RoutedASR(forced_lang="ja")`, `LiveVad`, `AudioHistory`, `Refiner`) を
  組み、`EventHub.add_listener` で結果を受け取る。stdout は読まない。
- 検証は `tests/test_ja_golden.py`。**2 段階の判定**で、
  (1) 完全一致した本数は報告するだけ、(2) 合否は**ゴールデンテキストに対する CER が
  1.0% 以下**かどうかで決める。int8 の ONNX カーネルは CPU のマイクロアーキテクチャを
  またぐと bit 再現しない (onnxruntime がベクトル幅に応じて別のコード経路を選ぶ) ので、
  完全一致を要求すると回帰でない理由で他のマシンで落ちる、というのが 2 段階にした理由である。
- **ただし、この 8 本の長さでは 1.0% は 1 文字も許さない。** 正規化後の長さが
  21〜44 文字なので、1 文字違うと CER は 2.3〜4.8% になる。つまり現状のこの閾値は
  完全一致と同じ判定になっている。それでも残してあるのは、失敗時に差分ではなく数値が出ること、
  長いクリップを足したときに意味を持つこと、そして「テキストを固定し、1% 未満のぶれは許す」
  という意図そのものを再実装側に伝えるためである。テストは 8 本を連結したセット全体の CER
  (正規化後 291 文字なので 1.0% は約 2.9 文字) も出すが、そちらは表示するだけで判定には使わない。
- 記録したマシンでは 8 本とも完全一致し、独立した 3 回の実行で同じ結果になった。
  別の CPU で無害な 1 文字ずれが出るようなら、直すのはゴールデンではなく閾値の方で、
  そのときは実測を書き残すこと。
- `models/` か `testdata/` がない環境では skip する。CI は両方持っていない。

再実装側でこのゴールデンをそのまま使うのは勧めない (デコーダのビット再現性が
そこまで期待できない)。使うなら CER の閾値を実測で決め直すこと。

## 設定ダンプと CI チェック

文書は放っておくとコードから離れる。そこで数値は 1 か所から機械的に出している。

- `scripts/dump_ja_config.py` が、**モデルを一切ロードせずに**日本語経路の実効設定を
  JSON で出す。値はすべてモジュール定数か関数の宣言済み既定値から取っており
  (`inspect.signature` を使っている箇所もある)、`models/` がない環境でも動く。
  `--with-models` を付けると `models/` を読んで sha256 とバイト数も足す。
- その出力を `docs/ja_pipeline_spec.json` に固定してある。再生成は:

  ```
  python scripts/dump_ja_config.py --with-models --out docs/ja_pipeline_spec.json
  ```

- `tests/test_ja_pipeline_spec.py` が、その `config` ブロックを毎回ダンプし直して
  committed のものと比較する。これが**ドリフト検出**である。`models` ブロックは
  `models/` がある環境でだけ比較し、ない環境では skip する。

この仕組みのために、コード側の裸のリテラルをモジュール定数に括り出した
(`asr_engine.RZ_MODEL_TYPE` / `RZ_DECODING_METHOD` / `RZ_MODELING_UNIT` /
`RZ_HOTWORDS_SCORE` / `RZ_MODEL_FILES`、`realtime_transcribe.VAD_MIN_SPEECH_S` /
`VAD_BUFFER_S` / `VAD_NUM_THREADS`、`punct_ja.PUNCT_*`)。値は 1 つも変えていない。
`punct_ja.py` は numpy / onnxruntime / fugashi の import をメソッド内に移してあり、
定数を読むだけなら 3 つとも要らない。

## 未検証事項

この文書に書いてあることのうち、まだ測っていない・確かめていないもの。

- **冒頭欠落の発生率**。greedy / beam × 対策なし / プリロール / 無音 300 ms の 6 条件の
  表は空のままである (上記プレースホルダ)。計測は別途進行中で、結果は
  `docs/HEAD_DROPOUT.md` に出る。**そこから転記するまで、条件間の優劣を数値で語らないこと。**
- **無音 300 ms 前置と実音声プリロールの直接比較**。他プロジェクトが無音前置で回避している
  という事実は分かっているが、hayamimi はそれを自分で測っていない。
  「プリロールを標準とする」という判断は、実測が残っているのがプリロール側だけ、
  という理由による。無音前置が劣ると測ったわけではない。
- **iOS**。句読点モデルを ONNX Runtime 経由で動かす経路は iOS で一度も走らせていない。
  Dart 版は既定で拒否する。
- **実 ARM デバイス**。Android の確認はすべて x86_64 エミュレータ (ホストは Ryzen 5 5600) で、
  fp16 の速度はいちばん外挿してはいけない数字である。x86 の CPU EP には fp16 の演算経路が
  なく、演算ごとに fp32 へキャストしている。NEON fp16 を持つ ARM では別の結果になる。
- **fp16 句読点モデルの ARM 上での実測レイテンシ**。同上。この PC では fp32 の約 10 倍遅い。
- **句読点復元の単体レイテンシ (Android)**。refine の合計時間から final の時間を引いた
  差分でしか出していない (11〜14 文字で 25〜35 ms)。直接の計測ではない。
- **`--replace` 置換辞書のこの経路での効果**。機構は実装されているが、
  日本語経路単体での効果測定はしていない。
- **ゴールデンテストの CER 閾値 1.0% が別 CPU で妥当かどうか**。別のマシンで走らせて
  いないので、閾値が厳しすぎるか緩すぎるかは分かっていない。
- **500 文字超過時の切り捨て**。1〜2 文の発話では当たらないという前提で放置してある。
  長文を流し込む使い方をするなら、チャンク分割を自分で足す必要がある。
