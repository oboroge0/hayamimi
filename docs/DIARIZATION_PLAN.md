# 話者分離（フルダイアライゼーション）設計調査

コードは書かず、調査と設計のみ。実装は本ドキュメントを土台に別イテレーションで行う。

## 1. 現状実装の正確な把握

### 構成

- `scripts/speaker_id.py` の `SpeakerLabeler`: CAM++ 話者埋め込み
  (`campplus_sv.onnx`, 3D-Speaker/Alibaba DAMO, 28MB) を VAD 確定セグメント
  ごとに1回計算し、セッション内で走っているセントロイド集合と比較する。
  コサイン類似度が `SIM_THRESHOLD=0.45` 以上なら最も近いセントロイドに割り当てて
  移動平均で更新、それ未満なら新規話者 `S{N+1}` を開く（`speaker_id.py:20-55`）。
- `scripts/realtime_transcribe.py::drain_segments()` が VAD 確定セグメントの
  サンプル列をそのまま `speaker_labeler.label()` に渡し、ASR の確定行と一緒に
  `S{n}|lang` のタグを付けて表示・配信する（`realtime_transcribe.py:232-234`）。
- 二段階目の `Refiner`（無音 `GROUP_GAP_S=2.0s` またはグループ長
  `GROUP_MAX_S=25.0s` で確定した「発話グループ」をまとめて再デコードする清書パス）
  は、話者ラベルを **多数決** で決める: グループ内の各セグメントに付いた
  `S{n}` のうち最頻値をグループ全体の話者として採用する
  （`realtime_transcribe.py:442-443`）。清書は音声認識精度の再デコードだけを
  行い、話者について独自の判定はしない。

### 限界（正確に言語化)

1. **セグメント単位が粗い**: 話者判定の単位は VAD の1確定セグメント
   (`min_speech_duration=0.25s`, `min_silence_duration=0.35s`) であり、
   1セグメント内に複数話者が混在する場合は分離できない。
   README にも明記の通り「同時発話（オーバーラップ）は1ラベルになる」。
2. **清書グループ内のターンチェンジが失われる**: `Refiner` は
   `GROUP_GAP_S=2.0s` 以上の無音がない限りセグメントを1グループに束ねる。
   このグループ内で話者が切り替わっても（相槌の応酬など、無音ギャップが
   短い場合）、清書結果には多数決で決めた1つの話者ラベルしか残らない。
   高速パス（`drain_segments` の確定行）には正しいセグメント単位のラベルが
   出るが、清書済みトランスクリプト・字幕・SSE配信の最終行はこの多数決に
   従うため、**清書版では話者の切り替えが消える**ケースがある。
3. **話者数の扱いはオンライン・オープンセット**: 明示的な話者数指定はなく、
   閾値 `0.45` を跨ぐたびに新規話者が増え続ける。上限がないため、
   長時間セッションで環境ノイズや同一話者の音響ゆらぎにより
   ラベルが分裂（同一人物が S1 と S3 に割れる）するリスクがある。
   逆に閾値を跨がなければ別人が同一ラベルに吸収される。
   セッション再起動やターゲット話者のリセット機構もない。
4. **ラベルが入れ替わる条件**: `_centroids` はプロセス内メモリのみで、
   セグメント順に貪欲でオンライン更新される。話者Aが長く発話してから
   何セグメントか置いて再登場した場合、その間に閾値未満の似た声色
   （BGM混じり、ノイズ、感情変化）が入るとセントロイドが徐々にドリフトし、
   後半で別話者と誤認識されうる。ヒステリシスや再クラスタリングは無い。
5. **CAM++ は 6 秒でサチる** (`MAX_EMBED_SECONDS=6.0`) ため長いセグメントは
   先頭のみで埋め込みを計算しており、これ自体は妥当な設計だが、
   長い発話ほど短い音声で判定していることになる。

### 現状のまとめ

現行方式は「ターンテイキング型の会話（対談・インタビューなど、
発話がほぼ交互に来る）」には十分実用的だが、①同時発話の分離、
②話者数の自動確定、③清書後もセグメント単位のラベルを保持、
の3点が本格的なダイアライゼーションに向けて不足している。

## 2. sherpa-onnx のダイアライゼーション対応

`.venv`（`H:\Programming\hayamimi\.venv`, sherpa-onnx **1.13.6**）の実体を
`dir()`/`help()` で確認済み。以下のクラスが利用可能:

```
sherpa_onnx.OfflineSpeakerDiarization
sherpa_onnx.OfflineSpeakerDiarizationConfig
sherpa_onnx.OfflineSpeakerDiarizationResult
sherpa_onnx.OfflineSpeakerDiarizationSegment
sherpa_onnx.FastClustering / FastClusteringConfig
sherpa_onnx.OfflineSpeakerSegmentationModelConfig
  (pyannote: OfflineSpeakerSegmentationPyannoteModelConfig)
```

### API の実際の形

```python
config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model="sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        ),
        num_threads=2,
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model="campplus_sv.onnx",   # 既存の --speakers と同じモデルを流用できる
        num_threads=2,
    ),
    clustering=sherpa_onnx.FastClusteringConfig(
        num_clusters=-1,      # -1 = 話者数を自動推定（threshold で決まる）
        threshold=0.5,
    ),
    min_duration_on=0.3,
    min_duration_off=0.5,
)
sd = sherpa_onnx.OfflineSpeakerDiarization(config)
result = sd.process(samples)          # samples: 1次元float32、sample_rate は sd.sample_rate 依存
segments = result.sort_by_start_time()  # [(speaker, start, end), ...]
```

重要な点:

- **完全にオフラインAPI**: `process()` は音声配列全体を受け取り一括処理する
  設計で、ストリーミング/チャンク投入用のAPIは無い（`process(samples, callback=...)`
  の `callback` は進捗コールバックであって、逐次投入用ではない）。
  リアルタイム対応は自前でウィンドウ分割・逐次呼び出しする必要がある
  （詳細は §3）。
- **embedding に既存の CAM++ をそのまま使い回せる**: `embedding` フィールドは
  `SpeakerEmbeddingExtractorConfig` 型で、`speaker_id.py` が既に使っている
  `campplus_sv.onnx` と完全に同じ型・同じモデルを渡せる。
  **新規ダウンロードが必要なのは segmentation モデルだけ**で、
  embedding モデルの追加コストはゼロ。
- **話者数自動推定**: `FastClusteringConfig.num_clusters=-1` で `threshold`
  ベースのクラスタ数自動決定になる（閾値が小さいほど話者数が増える方向）。
  既知の話者数がある場合は `num_clusters=N` を明示できる。

### モデルの入手性・ライセンス・サイズ

- **pyannote segmentation-3.0** (GitHub リリース
  `k2-fsa/sherpa-onnx` タグ `speaker-segmentation-models` の
  `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`): **6,958,444 バイト
  (≈6.6MB, 圧縮)**。元モデルは Hugging Face の
  [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
  および ONNX 変換版
  [`onnx-community/pyannote-segmentation-3.0`](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)
  （model.onnx 単体 ≈6MB）で、**ライセンスは MIT**。SincNet フロントエンド +
  双方向LSTM の軽量モデルで、現行の `campplus_sv.onnx`(28MB) より軽い。
- 同リリースには他に `sherpa-onnx-reverb-diarization-v1.tar.bz2`
  (≈10.9MB, Reverb社の埋め込みモデル込み) と
  `sherpa-onnx-reverb-diarization-v2.tar.bz2` (≈254MB, NeMo系、大型) もあるが、
  hayamimi はすでに CAM++ を運用しているため **pyannote segmentation-3.0 +
  既存 CAM++ embedding** の組み合わせが最小追加コストで最有力。
- テスト用音声も同リリースに同梱 (`0-four-speakers-zh.wav` など)。
  ライセンスは元動画/録音依存のため、redistribute する場合は要確認。
- CPU実行速度の公式ベンチマーク数値は sherpa-onnx 公式ドキュメントに
  明記が見当たらなかった（オフライン処理という性質上、公式は「バッチで回せば
  十分速い」という説明に留まる）。pyannote segmentation-3.0 自体は
  SincNet+BiLSTM の小型モデルで、hayamimi が既に CPU で回している
  Parakeet(RTF 0.02〜0.11) や CAM++ 埋め込み抽出と比べても軽量な部類。
  **公式の定量CPU実測レポートは見つからなかった** ため、Iteration ②で
  hayamimi 自身の環境で実測するのが正しい進め方（§5参照）。

### 現行 `testdata/two_speakers.wav` の出所

- リポジトリでは **git 管理外**（`.gitignore` に `testdata/` 指定あり）。
  コミット履歴・コード内のどこにも出典の記載が見つからなかった。
  開発者手元のアドホックなテスト音声と推測され、**由来不明・ライセンス不明**。
  再配布や公開評価データとして使うのは避け、個人のスモークテスト用途に
  留めるべき。DERの正式な計測には §4 のCC-BYデータセットを使う。

## 3. リアルタイム分離の設計パターン

### 定石の整理

| 方式 | 概要 | 長所 | 短所 |
|---|---|---|---|
| **スライディングウィンドウ再クラスタリング** | 直近N秒のバッファに対し毎回オフライン分離をやり直し、ラベルを前回結果とハンガリアン法などで対応付け直す | 実装が単純、各ウィンドウは高精度なオフラインアルゴリズムをそのまま使える | ウィンドウ境界をまたぐ話者の一貫性維持が別途必要、計算コストが再帰的（毎回全体をやり直す） |
| **埋め込みのオンラインクラスタリング**（本プロジェクトの現行方式はこれに近い） | セグメントごとに埋め込みを計算し、逐次到着するデータ点をオンラインでクラスタに割り当てる（貪欲最近傍・オンラインk-means・インクリメンタルAHC・VBx等） | レイテンシ最小、追加モデル不要な場合が多い | クラスタ数決定・ドリフト・再クラスタリングの仕組みが必要（現行実装が抱える限界そのもの） |
| **確定セグメント単位の逐次クラスタ割当＋清書での再クラスタリング**（two-pass） | 高速パスは軽量なオンライン割当のまま、清書（無音区切り or 定期）でそのグループ音声をオフラインの高精度分離器（pyannote segmentation + AHC/スペクトラルクラスタリング）にかけ直し、ラベルをセッション全体のセントロイド集合に再マッピングする | hayamimi の既存 two-pass 清書アーキテクチャ（`Refiner`）にそのまま乗る。高速パスの応答性を落とさず、清書だけ精度を上げられる。オーバーラップ検出も清書側でだけ有効化できる | 清書のタイミング（無音2秒 or 25秒）でしか高精度化されない。清書内で決めたローカルクラスタIDをセッション全体のグローバルラベル（S1/S2...）に一貫してマッピングし直すロジックが要る |

### hayamimi への推薦

**「確定セグメント単位の逐次クラスタ割当＋清書での再クラスタリング」を推薦する。**

理由:

1. hayamimi はすでに VAD 確定セグメント → 高速パス → 無音/長さトリガーの
   清書（`Refiner`）という two-pass 構造を持っており、これに話者分離を
   重ねるのがアーキテクチャ的に自然（新しいパイプライン段を作らずに済む）。
   `Refiner.maybe_refine()` は既にグループ全体の音声バッファ
   (`self.history.buf[...]`) を持っているため、
   `OfflineSpeakerDiarization.process()` をそのまま groupバッファに対して
   呼べる。
2. `OfflineSpeakerDiarization` はオフライン専用APIなので、
   「スライディングウィンドウ全体を毎回丸ごと再分離する」設計とは
   そもそも相性が良い。清書グループ（数秒〜25秒）は「ウィンドウ」として
   ちょうど良いサイズで、hayamimi のGROUP_GAP_S/GROUP_MAX_Sの設計と一致する。
3. 高速パス（現行 `SpeakerLabeler`）は変更不要、または軽微な改善に留めて
   応答性を守る。精度が必要なのは「最終的に画面・字幕・書き起こしに残る
   清書結果」なので、そこだけ本格化するのが費用対効果が高い。
4. VBx やオンラインk-meansのような真の逐次クラスタリングアルゴリズムを
   ゼロから実装するより、sherpa-onnxが提供する `FastClustering`
   （既にAHC相当の実装をラップ済み）を清書境界で呼ぶ方が実装・保守コストが低い。

### 具体的な設計

- **高速パス（変更小）**: 現行 `SpeakerLabeler` のまま。ただし
  ラベルのドリフト対策として、後述のグローバル再マッピング結果を
  センロイドにフィードバックする経路を用意する（Iteration④）。
- **清書パス（新規）**:
  1. `Refiner.maybe_refine()` がグループのbufを確定した時点で、
     `OfflineSpeakerDiarization.process(buf)` を呼び、グループ内の
     セグメント単位の話者境界（`segments: speaker, start, end`）を得る。
  2. 各 diarization セグメントの中心埋め込み（`FastClustering` が内部で
     使う CAM++ embedding は今回 process() 内で計算済みだが、外部からは
     取得できないため、境界だけ貰って embedding は自前で再計算するか、
     `speaker_id.py` の `SpeakerLabeler` にある既存メソッドを流用して
     再計算する）を、セッション全体のグローバルセントロイド集合
     （高速パスが保持しているもの）に対して最近傍マッチングし、
     グループローカルな話者ID（0,1,2...）をセッションのグローバルラベル
     （S1, S2...）に対応付け直す。
  3. 清書テキストを diarization の話者境界でさらに分割し、
     `[refine/S1] ... [refine/S2] ...` のように **清書結果でも
     ターン単位のラベルを保持**する（現行の多数決を廃止）。
     これが現状の限界②を解消する中心的な変更点。
  4. オーバーラップ区間（pyannote segmentationは重複区間も検出できる）は
     当面「先着優先 or 両方表示」の単純規則で良い。完全な overlap-aware
     文字起こし（who-said-what の重複割当）はスコープ外とする。

### なぜ「毎セグメントでオフライン分離器を回す」を採用しないか

セグメント長は 0.25〜数秒と短く、pyannote segmentation + FastClustering の
オーバーヘッドをセグメント単位で払うのは高速パスのレイテンシ予算に見合わない。
清書境界（数秒〜25秒に1回）でだけ払うのがRTF予算的に妥当。

## 4. 評価

### DER (Diarization Error Rate) の標準計測

DER は Miss（参照にある発話を検出できなかった時間）、False Alarm（参照に
無い発話を誤検出した時間）、Confusion（検出はできたが話者を間違えた時間）
の合計を参照音声長で割った値。NIST RT の慣習に従い、通常
**collar 0.25秒**（発話境界前後を採点対象外にする）と、重複発話区間を
含めるかどうかの2条件で報告するのが一般的。

### 採点ライブラリの選定

- **[pyannote.metrics](https://github.com/pyannote/pyannote-metrics)**
  (`pip install pyannote-metrics`): DERの標準実装
  (`pyannote.metrics.diarization.DiarizationErrorRate`)。Miss/FA/Confusion
  の内訳まで取れる。ただし依存が重い（`pyannote.core`, `pandas`,
  `sortedcontainers` 等）。CPU・低依存を志向する hayamimi の理念とはやや
  ズレるが、**開発時の評価専用の依存**として `requirements-dev` 的な
  位置づけで導入するのは妥当（実行時の推論パスには入らない）。
- **[simpleder](https://pypi.org/project/simpleder/)**
  (`pip install simpleder`, Apache-2.0, 依存は `scipy` のみ):
  DERの総合値だけを軽量に計算できる。内訳（FA/Miss/Confusion）は出ないが、
  素早い実測ループ（Iteration②③⑤の反復）にはこちらで十分。
  導入コストが最小で、CI/バッチ評価スクリプトに組み込みやすい。

**推薦**: 日常のイテレーション（DERの上下だけ見たい）は `simpleder`、
最終的な特性分析（どのタイプの誤りが多いか）が必要になった時だけ
`pyannote.metrics` を追加投入する、の二段構えとする。

### CC-BY系で入手容易な多話者評価データ

- **[AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)**
  ([ライセンス: CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)):
  100時間、英語4人会議、ヘッドセットmix音声＋正解話者アノテーション付き。
  クリーンな会議室録音で、話者分離の標準ベンチマークとして広く使われる
  （"AMI headset-mix" は多くの diarization 論文の定番テストセット）。
  **推薦: まずこれをベースラインに使う。** 学術的に確立した参照値との
  比較がしやすい。
- **[VoxConverse](https://github.com/joonson/voxconverse)**
  (CC BY 4.0、YouTube由来の"in the wild"音声、開発セット20.3h/216本、
  テストセット53.7h/310本、話者数1〜20+）: BGM・雑音・話者数のばらつきが
  大きく、hayamimi が実際に想定する「配信・対談・雑談」的な音響条件に近い。
  AMIより難しく、実運用に近い誤り傾向を見るのに向く。
- 両データセットとも英語中心で日本語コンテンツを含まない。
  DER自体は言語非依存の指標（音声区間と話者境界の一致度）なので技術的な
  妥当性は損なわれないが、**hayamimi の主戦場である日本語コンテンツでの
  実地確認には別途、開発者自身が権利を持つ/許諾を得た日本語多人数音声
  （録音同意の取れた対談等）でのスポットチェックを併用すべき**。
  `testdata/two_speakers.wav`（出所不明）はこの用途にも使わない。

### 導入コスト

- `simpleder`: 数分（`pip install simpleder`、DER計算関数を1つ呼ぶだけ）。
- AMIサブセット取得: ヘッドセットmixの数ファイルだけなら数百MB、
  スクリプトで自動ダウンロード可能（`groups.inf.ed.ac.uk/ami/download/`
  はミーティングID単位でダウンロードでき、全量を取る必要はない）。
- VoxConverse: 開発セットの音声+RTTM正解ラベルはGitHubリポジトリと
  音声アーカイブから取得。RTTM形式はpyannote.metrics/simpladerどちらも
  パース可能（RTTMパーサは自作 or `pyannote.database` 経由）。

## 5. 段階的実装計画

各イテレーションは1〜2時間で完結する粒度に分解。

### ① 評価基盤: DER採点 + AMIサブセット取得スクリプト

- `scripts/eval_diarization.py`（新規、既存の `eval_*.py` 群の命名慣習に
  合わせる）: RTTM（または簡易JSON）形式の参照ラベルと、hayamimiの出力
  （セグメント+話者ラベルのリスト）を受け取り `simpleder` でDERを計算する。
- `scripts/make_diarizationset.py`（既存の `make_realset.py` 等に倣う）:
  AMI headset-mixから数会議分（例: 5〜10ファイル、各10〜30分程度）を
  ダウンロードし、正解RTTMと共に `testdata/eval_diarization/` に配置する
  スクリプト（`testdata/` はgit管理外なので配置は安全）。
- 完了条件: 適当な2ファイル（参照=仮説と同じ）でDER=0%が出ることを確認。

### ② 現方式のDERベースライン実測

- ①のAMIサブセットに対し、現行 `--speakers`（`SpeakerLabeler` の
  ナイーブな逐次最近傍割当）をそのまま走らせてDERを計測する。
- `docs/BENCHMARKS.md` の慣習に合わせて表形式で記録（音声、話者数、
  DER、内訳が取れれば pyannote.metrics で追加計測）。
- 完了条件: ベースラインDER（おそらく20〜40%程度になると予想。
  現行は同時発話非対応・話者数無制限オープンセットのため悪化しやすい）
  が数値として残る。

### ③ 清書パスへの `OfflineSpeakerDiarization` 統合（グループ単位）

- `Refiner.maybe_refine()` の `work()` 内で、清書対象グループの `buf` に
  対し `OfflineSpeakerDiarization.process(buf)` を呼び、セグメント単位の
  話者境界を得る。
- pyannote segmentation-3.0 モデルを `download_models.py` に追加
  (`SPEAKER_TAG` 相当のリリースタグから取得、embedding は既存
  `campplus_sv.onnx` を再利用)。
- まずは「グループローカルなクラスタID（0,1,2...）をそのまま
  `[refine/spk0]` のように出す」だけの最小実装で動作確認する
  （グローバルなS1/S2への統一マッピングは④に回す）。
- 完了条件: ①②のAMIサブセットで、清書結果のみのDERが計測でき、
  ②のベースラインと比較できる。

### ④ グローバルラベルへの再マッピング（セッション一貫性）

- ③で得たグループローカルの話者クラスタの代表embeddingを、
  高速パス（`SpeakerLabeler`）が保持しているグローバルセントロイド集合に
  最近傍マッチングし、`S{n}` にマッピングし直す。
- 高速パスの結果と清書結果の間でラベルの不整合（高速パスではS1、
  清書ではS2、のような食い違い）が出るケースを洗い出し、
  どちらを正とするか（清書を正とし、SSE配信済みの高速パス行を
  後から上書きするか、両方残すか）の方針を決めて実装する。
- 完了条件: セッション全体を通して同一話者が一貫したS番号を保つこと
  （複数回の清書グループをまたいで）をAMIサブセットの手動確認 or
  DER計測で確認。

### ⑤ 実測比較とチューニング

- ①②③④を通して、AMI・VoxConverse両方でDERを比較し、
  `FastClusteringConfig.threshold`・`SIM_THRESHOLD`（高速パス）・
  `GROUP_GAP_S`（清書粒度）を振ってチューニングする。
- RTFの実測（清書パスに追加した diarization 処理がGOALSのRTF<0.2予算に
  収まるか、清書はリアルタイム制約が高速パスほど厳しくないため許容幅は
  広いが計測はしておく）。
- `docs/BENCHMARKS.md` 追記、README の Limitations 更新
  （「同時発話は分離できない」は残る点も明記: pyannote segmentationは
  オーバーラップを検出できるが、本計画のスコープでは overlap-aware な
  文字起こし統合まではやらない）。
- 完了条件: 最終DER・RTFが記録され、次の意思決定（overlap対応や
  ストリーミング化をさらに進めるか）ができる状態になる。

## 6. ベースライン実測 (イテレーション①②)

`scripts/make_diarset.py` で AMI headset-mix から dev/test 計5会議
(各600秒、開始60秒地点からのスライス。冒頭1分は準備中の雑音が多いため除外)
を `testdata/eval_diar/` に整備し、`scripts/eval_diar.py` で
現行 `--speakers`（`SpeakerLabeler` のオンライン最近傍割当。VADは
`realtime_transcribe.build_vad()` と同一設定、ASRデコードは省略し
VAD+CAM++埋め込みのみの軽量経路で本番の `SpeakerLabeler` クラスを
そのまま呼び出す）のDERを実測した。採点は `simpleder.DER(ref, hyp,
collar=0.25)`（NIST RT慣習の0.25秒collar）。

### 会議別DER

| 会議 | split | 話者数(ref) | 話者数(hyp) | ref発話数 | hyp発話数 | DER |
|---|---|---|---|---|---|---|
| ES2011a | dev | 4 | 9 | 83 | 78 | 32.1% |
| IS1008a | dev | 4 | 6 | 107 | 56 | 18.2% |
| ES2004a | test | 4 | 8 | 131 | 81 | 28.5% |
| IS1009a | test | 4 | 6 | 145 | 64 | 32.9% |
| TS3003a | test | 4 | 6 | 62 | 92 | 16.7% |
| **平均** | | | | | | **25.7%** |

VAD+CAM++埋め込みのみの処理時間は600秒音声あたり約14.5秒
(RTF ≈ 0.024)で、ASRを含まない分ここは全く問題にならない。

### 観察された壊れ方

1. **話者数の過大推定が一貫している**: 全会議で参照4話者に対し
   仮説は6〜9話者。予想通り §1限界③（オープンセット・上限なしの
   新規話者生成）がそのまま表れており、`SIM_THRESHOLD=0.45`が
   実会議の音響条件（マイク距離差、被り、環境ノイズ）では
   同一話者の埋め込みゆらぎを「別話者」と誤判定しやすいことを示す。
   ES2011aが最悪（4→9）、IS1008a/IS1009a/TS3003aは4→6に留まっており、
   会議ごとの発話スタイル（早い切り替え・被り発話の頻度）に
   感度が高いと見られる。
2. **発話数はref/hypでほぼ同オーダーだが、hypがrefを上回る会議もある**
   (TS3003a: ref62 vs hyp92)。VADの`min_silence_duration=0.35s`
   基準のセグメント分割がRTTMの発話境界と一致しないケースがあり、
   1つの参照発話がVAD側で複数セグメントに割れる、または逆に
   短い間投詞的発話がVADでは1セグメントに吸収される、といった
   ミスアラインメントがDERのMiss/FA成分に寄与していると考えられる
   （`simpleder`はMiss/FA/Confusionの内訳を出さないため、
   §4で触れた`pyannote.metrics`を投入すればここの切り分けが可能）。
3. **DERのばらつきが大きい (16.7%〜32.9%)**: 会議固有の話し方
   （TS3003aはターンテイキングが比較的整っているのか良好、
   IS1009aは被り発話が多いのか悪化）に強く依存しており、
   単一の閾値`SIM_THRESHOLD`では会議横断で安定しないことが数値でも
   裏付けられた。これは§3で推薦した「清書パスでのオフライン
   再分離」（イテレーション③）が対処すべき核心的な弱点そのもの。
4. **予想レンジ内**: §5の完了条件で見込んだ「20〜40%程度」に
   平均25.7%は収まった。同時発話非対応・話者数無制限オープンセット
   という設計上の弱点がそのまま数値に出ている。

### イテレーション③への引き継ぎ

- ベースライン（現行 `SpeakerLabeler` 単体、平均DER 25.7%、
  話者数過大推定が主要因）に対し、清書パスへの
  `OfflineSpeakerDiarization`統合後のDERを同じ5会議・同じcollar設定
  (0.25s) で比較すること。`scripts/eval_diar.py --meeting <ID>`で
  会議単位の再測定ができる。
- 話者数過大推定（限界③）が最大の誤り要因である可能性が高いため、
  ③の実装では `FastClusteringConfig` の `threshold`（またはAHCの
  停止基準）を、まず現行`SIM_THRESHOLD=0.45`と同等〜やや緩い設定で
  試し、DERの話者数(`n_hyp_speakers`)がrefの4に近づくかを
  最初のチェックポイントにするとよい。
- `pyannote.metrics`（Miss/FA/Confusion内訳）の追加投入は、
  ③でも過大推定が残る場合に「境界のズレ(Miss/FA)」と
  「話者取り違え(Confusion)」のどちらが支配的かを切り分けるために
  イテレーション⑤で検討する。
- 評価セット (`testdata/eval_diar/`, 5会議・50分) はgit非追跡なので、
  ③④⑤の実装エージェントは各自 `python scripts/make_diarset.py` を
  実行してから `eval_diar.py` を使うこと（`--skip-existing`で
  再ダウンロードを避けられる）。

## 7. イテレーション③④実装結果: Refinerグループ単位の再分離 + グローバル対応付け

`scripts/diarize.py`（新規、`GroupDiarizer`）で
`sherpa_onnx.OfflineSpeakerDiarization`（pyannote segmentation-3.0 +
既存CAM++ embedding + `FastClustering`）をラップし、
`realtime_transcribe.Refiner._emit_turns()` から呼び出す形で統合した。
`Refiner.maybe_refine()` がグループの `buf` を確定した時点で
`GroupDiarizer.process(buf, sr)` を呼び、ローカル話者クラスタ（0,1,2...）を
セグメント境界とともに取得。クラスタごとの代表埋め込み
（`speaker_id.SpeakerLabeler.embed()`、そのクラスタに属する区間の音声を
連結して計算）を、高速パスが保持する**同一インスタンスのグローバル
セントロイド集合**に `SpeakerLabeler.match_embedding(update=True)` で
最近傍マッチングし、`S{n}` へ対応付け直す。分離で2話者以上見つかった
グループはターンごとに個別ASR再デコード＋個別 `[refine/S{n}]` 行を出力し
（`GROUP_MAX_S=25s`のグループ内でも話者交代を保持）、
1話者しか見つからない場合や分離に失敗した場合は元の多数決1行に
フォールバックする（`fast_joined` に対する「再デコードが短すぎたら
fast pathの結合テキストを信頼する」既存ガードも踏襲）。
`--speakers` 有効時のみ動作し、速報パス（`drain_segments`）は無変更。
pyannote segmentation-3.0 モデルは `models/sherpa-onnx-pyannote-segmentation-3-0/`
に配置（`models/` はgit管理外、`download_models.py` への追加はしていない
-- 今回は直接DLで導入、⑤で判断）。モデル欠如時は `--speakers` は
従来の多数決動作にフォールバックし、起動は失敗しない。

`scripts/eval_diar.py` に `--method refine_diarize` を追加。
VAD→高速`SpeakerLabeler`割当→`Refiner`と同じ無音/長さ規則
（`GROUP_GAP_S`/`GROUP_MAX_S`）でのグループ化→グループ単位の
`GroupDiarizer`分離→グローバル対応付け、という本番`Refiner._emit_turns()`と
同じロジックをASRデコード抜きで再現し、DERとRTFを測定できるようにした
（グループ化のロジック自体は `group_segments()` として純粋関数に切り出し、
`tests/test_diar_eval.py` でユニットテスト済み）。

### DER比較（AMI 5会議、collar=0.25s）

| 会議 | ref話者数 | ベースラインDER (①②) | 新方式DER (threshold=0.5) | 新方式DER (0.6) | 新方式DER (0.7) |
|---|---|---|---|---|---|
| ES2011a | 4 | 32.1% | 18.0% | 18.0% | 17.6% |
| IS1008a | 4 | 18.2% | 4.2% | 4.2% | 4.2% |
| ES2004a | 4 | 28.5% | 17.8% | 17.7% | 17.7% |
| IS1009a | 4 | 32.9% | 17.3% | 17.1% | 17.4% |
| TS3003a | 4 | 16.7% | 14.1% | 14.1% | 13.4% |
| **平均** | | **25.7%** | **14.3%** | **14.2%** | **14.1%** |

`FastClusteringConfig.threshold`（`diarize.DEFAULT_THRESHOLD`、清書グループ内
のローカル分離にのみ効く）を0.5/0.6/0.7の3点で粗く振ったが、
平均DERはほぼ横ばい（14.1〜14.3%）で、この閾値は支配的な要因ではない
ことが分かった。デフォルトは `0.5` のまま据え置いた（有意な差がない中で、
過去に元設計案で示していた値を維持）。

### 推定話者数（グローバル、セッション通し）

| 会議 | ref | ベースライン(①②) | 新方式 threshold=0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| ES2011a | 4 | 9 | 8 | 8 | 6 |
| IS1008a | 4 | 6 | 10 | 11 | 11 |
| ES2004a | 4 | 8 | 11 | 10 | 11 |
| IS1009a | 4 | 6 | 9 | 8 | 8 |
| TS3003a | 4 | 6 | 8 | 8 | 8 |

**チェックポイント①（推定話者数の4への収束）は達成していない**:
DERは大幅改善した一方（平均25.7%→14.1〜14.3%）、グローバル話者数は
むしろ悪化した会議もある（IS1008a: 6→10〜11、ES2004a: 8→10〜11）。
原因を切り分けると、`FastClusteringConfig.threshold`（清書グループ内の
ローカル分離）をいくら振ってもグローバル話者数がほぼ動かないことから、
支配的な要因は清書グループのローカル分離ではなく、**そのローカル
クラスタをグローバルへ対応付ける `SpeakerLabeler.match_embedding()` 側の
閾値（`speaker_id.SIM_THRESHOLD=0.45`、④で意図的に「既存クラスの再利用を
優先」する設計として流用した値）** と見られる。理由は2つ考えられる:
(a) 清書グループごとに複数のローカルクラスタ埋め込みをグローバル
照合するようになった分、単純にグローバル照合の**試行回数**が
ベースライン（VADセグメント単位1回のみ）より増え、0.45を割る
「別話者」判定の機会が増えた。(b) ローカルクラスタの代表埋め込みは
そのクラスタに属する区間だけを連結した音声から計算するため、
セッション全体を通した音響条件のブレに対してベースラインの
移動平均センロイドより脆弱な可能性がある。
DERそのものは大幅に改善しているため（同じグループ内のターン境界を
正しく分離できていることがConfusion/Missの減少に効いている）、
実害は「同一人物が複数のS番号に分裂する」点に限定されるが、
これは製品の「セッション通しでラベルが一貫すること」という
④の完了条件を完全には満たしていない。

### 処理時間（RTF、清書グループ単位）

| 会議 | グループ数 | 分離処理時間(threshold=0.5) | 音声長 | diar RTF |
|---|---|---|---|---|
| ES2011a | 32 | 19.9s | 600s | 0.033 |
| IS1008a | 22 | 40.6s | 600s | 0.068 |
| ES2004a | 39 | 18.4s | 600s | 0.031 |
| IS1009a | 24 | 31.1s | 600s | 0.052 |
| TS3003a | 28 | 31.2s | 600s | 0.052 |

グループあたり平均 0.5〜1.9秒程度（音声長換算 RTF 0.03〜0.07）。
清書は非ホットパス（バックグラウンドワーカースレッドで直列実行、
次の確定行の表示はブロックしない）なので実用上は問題ないが、
1グループの分離自体は決して軽くはない（清書デコードそのものと
同程度〜それ以上のコストがかかることもある）ため、⑤でRTF全体の
予算内に収まるか改めて確認する価値がある。

### ⑤への引き継ぎ

- **最優先課題**: グローバル話者数の過大推定は解消していない。
  `speaker_id.SIM_THRESHOLD=0.45`（グローバル対応付けの閾値）を
  `diarize.DEFAULT_THRESHOLD` とは独立に振る実験が必要。特に
  「ローカルクラスタの代表埋め込みをグローバルへ照合する際だけ
  閾値を緩める」といった非対称な調整が有効か検証する価値がある。
- `pyannote.metrics`（Miss/FA/Confusion内訳）を投入すれば、
  DER改善の内訳（ターン境界精度の改善なのか、話者取り違えの
  減少なのか）を定量化できる。現状は`simpleder`の合算値のみ。
- `download_models.py` へのpyannote segmentation-3.0追加は未実施
  （今回は直接DLで導入）。本番運用に載せるなら追加が必要。
- オーバーラップ区間の扱い（先着優先）は本イテレーションでも
  未着手のまま（スコープ外として据え置き、§3の設計方針通り）。

## 8. イテレーション⑤実装結果: グローバル閾値の分離チューニング + 内訳分析

### 実験設計

§7の引き継ぎ通り、グローバル話者数過大推定の主因は`FastClusteringConfig.threshold`
（清書グループ内のローカル分離）ではなく`speaker_id.SIM_THRESHOLD=0.45`
（ローカルクラスタ代表埋め込みをグローバルセントロイドへ対応付ける際の閾値）
と当たりを付けていたため、まずこの1点を検証した。

`speaker_id.SpeakerLabeler`に2つ目の閾値`remap_threshold`を追加した
（`__init__`のキーワード引数、デフォルトは新設の`REMAP_THRESHOLD=0.35`）。
`match_embedding(emb, update, threshold=None)`が`threshold`引数を受け取れるよう
にし、`None`のときだけ`self._threshold`（従来の`SIM_THRESHOLD`、速報パスの
`label()`が使う値）にフォールバックする。清書パスの対応付け呼び出し
（`realtime_transcribe.Refiner._emit_turns()`と`eval_diar.py`の
`generate_diarize_hypothesis()`の両方）は`threshold=labeler.remap_threshold`を
明示的に渡すよう変更した。これにより**速報パス（`label()`、VADセグメント単位で
高頻度に呼ばれる）と清書パスのグローバル対応付け（グループごとのローカル
クラスタ単位で低頻度に呼ばれる）を別々の閾値で運用できる**ようになった
(`scripts/speaker_id.py:16-79`, `scripts/eval_diar.py --sim-threshold`/
`--remap-threshold`, `scripts/realtime_transcribe.py --speaker-remap-threshold`)。

dev会議（ES2011a, IS1008a）で以下を振った:

1. 対称スイープ: `sim_threshold`(=`SIM_THRESHOLD`扱い)を0.30/0.35/0.40/0.45/0.50で
   振り、`remap_threshold`は同じ値にフォールバック（従来の単一閾値運用と同等）。
2. 非対称スイープ: `sim_threshold=0.45`（速報パス据え置き、回帰リスクゼロ）に固定し、
   `remap_threshold`だけを0.30/0.35/0.40で振る。

### dev会議での結果

対称スイープ（`diar_threshold=0.5`固定）:

| sim=remap閾値 | ES2011a DER | ES2011a 話者数 | IS1008a DER | IS1008a 話者数 |
|---|---|---|---|---|
| 0.30 | 48.5% | 3 | 4.0% | 5 |
| 0.35 | 18.6% | 4 | 4.0% | 6 |
| 0.40 | 18.0% | 5 | 4.0% | 6 |
| 0.45(旧デフォルト) | 18.0% | 8 | 4.2% | 10 |
| 0.50 | 19.4% | 10 | 4.4% | 12 |

0.30まで下げるとES2011aが崩壊する（DER 48.5%、話者数が3に潰れる＝過少推定側に
振り切れる）。0.35〜0.40が対称スイープの中では最良で、話者数もrefの4に近づくが、
DERはIS1008aでは横ばい（各閾値でDER自体はほぼ4.0〜4.4%と鈍感、話者数だけが動く）。

非対称スイープ（`sim_threshold=0.45`固定、`diar_threshold=0.5`固定）:

| remap閾値 | ES2011a DER | ES2011a 話者数 | IS1008a DER | IS1008a 話者数 |
|---|---|---|---|---|
| 0.30 | 17.7% | 5 | 4.0% | 6 |
| **0.35** | **16.8%** | **4** | **4.0%** | 7 |
| 0.40 | 17.8% | 7 | 4.0% | 7 |

`remap_threshold=0.35`（速報パスは0.45のまま）が対称スイープの最良値
（`sim=remap=0.35`: ES2011a DER 18.6%）を上回った（ES2011a DER 16.8%、話者数は
参照通り4に一致）。速報パス閾値を動かさない分、清書の恩恵を受けない場面
（`--speakers`使用時でも清書前に表示される確定行など）への影響もない。
**採用: `remap_threshold=0.35`、`SIM_THRESHOLD`(速報パス)は0.45のまま据え置き。**

### test会議での汎化確認

選定した`remap_threshold=0.35`（`sim_threshold`はデフォルトの0.45、
`diar_threshold`もデフォルトの0.5）を、dev会議選定に使っていない残り3会議
（test split: ES2004a, IS1009a, TS3003a）で検証した。

| 会議 | split | ref話者数 | §7時点DER(remap=0.45) | §7時点話者数 | ⑤採用後DER(remap=0.35) | ⑤採用後話者数 |
|---|---|---|---|---|---|---|
| ES2011a | dev | 4 | 18.0% | 8 | 16.8% | 4 |
| IS1008a | dev | 4 | 4.2% | 10 | 4.0% | 7 |
| ES2004a | test | 4 | 17.8% | 10〜11 | 17.8% | 8 |
| IS1009a | test | 4 | 17.3% | 8〜9 | 16.9% | 6 |
| TS3003a | test | 4 | 14.1% | 8 | 14.1% | 7 |
| **平均** | | | **14.3%*** | | **13.9%** | |

(*§7表のthreshold=0.5列の平均。dev/test全5会議、collar=0.25s。)

test会議でもDERは横ばい〜微改善（悪化した会議はゼロ）、話者数は全会議で
過大推定が縮小した（ES2004a 10〜11→8、IS1009a 8〜9→6、TS3003a 8→7）。
dev会議で選んだ設定がtest会議でも一貫して効いており、dev/testの過学習は
見られない。ただし**話者数の過大推定は解消しきれていない**（refの4に対し
6〜8で下げ止まり）。速報パス閾値0.45を維持したことで速報パス自体の性質
（同一閾値0.45での新規話者生成しやすさ）はそのまま残っており、清書パスの
対応付け閾値だけを緩めても速報パスが最初に開いた「分裂気味の」グローバル
セントロイド集合自体は変えられないため、と考えられる（根本対応には
速報パス側のセントロイド管理そのもの、例えば定期的な再クラスタリングや
ヒステリシス付き閾値が必要で、これは本イテレーションのスコープ外）。

### pyannote.metrics による内訳分析

`pyannote.metrics==4.1`を`requirements-dev.txt`に追加し
（`pip install pyannote.metrics`で`pyannote.core`/`pyannote.database`/
`pandas`/`scikit-learn`ごと入る、simpladerより重い依存。§4で書いた通り
「日常はsimpleder、内訳が要る時だけpyannote.metrics」の二段構え通りの位置づけ）、
`scripts/eval_diar.py`に`der_breakdown()`関数と`--breakdown`フラグを追加した
(`pyannote.metrics.diarization.DiarizationErrorRate(collar=...)`を
`detailed=True`で呼び、Miss/FalseAlarm/Confusionを参照総発話時間で正規化して返す)。
`tests/test_diar_eval.py`に2件のユニットテストを追加（同一ref/hypでゼロ、
純粋なConfusionケースと純粋なMissケースが正しく分離されることを確認）。

採用設定（`remap_threshold=0.35`, `sim_threshold=0.45`, `diar_threshold=0.5`）で
AMI 5会議の内訳を実測:

| 会議 | DER(simpleder) | Miss | False Alarm | Confusion |
|---|---|---|---|---|
| ES2011a | 16.8% | 11.3% | 3.5% | 5.9% |
| IS1008a | 4.0% | 2.6% | 3.5% | 0.7% |
| ES2004a | 17.8% | 14.5% | 3.7% | 5.0% |
| IS1009a | 16.9% | 6.8% | 4.5% | 12.4% |
| TS3003a | 14.1% | 8.7% | 6.5% | 0.7% |

（pyannote.metricsのDER定義はsimpladerと実装が異なるため個々の内訳の合計は
simpladerのDER%と厳密には一致しない。overlap区間の扱いなど採点細部の差異による
もので、傾向を見る内訳比較としては問題ない。）

**結論: 5会議中4会議はMiss（検出漏れ）がConfusion（話者取り違え）を上回る**
（ES2011a: 11.3%>5.9%、IS1008a: 2.6%>0.7%、ES2004a: 14.5%>5.0%、
TS3003a: 8.7%>0.7%）。唯一IS1009aだけConfusionがMissを上回る（12.4%>6.8%）。

これは§1限界②で挙げていた「話者過大推定」問題の実害を見る上で重要な情報:
グローバル話者数はrefの4に対し6〜8で過大推定のままだが、それによる
**DERへの実害（Confusion）は多くの会議で小さい**。過大推定で生まれた余分な
S{n}ラベルは、既存の話者と時間的に大きく重ならない区間（境界のズレで
別クラスタに割れた小さな発話片など）に付くことが多く、参照話者への
誤帰属（＝真の意味での話者取り違え）としてはあまりカウントされていない
と読める。DERの主成分はむしろMiss（VADのセグメント境界とRTTMの参照境界の
ずれ、pyannote segmentationの発話境界検出誤差）であり、§6で立てていた
仮説（VAD境界と参照境界のミスアラインメントがMiss/FAに寄与）が内訳計測で
裏付けられた形になる。IS1009aだけConfusionが優勢なのは、この会議固有の
話者交代パターン（早口・被り発話が多い、§6の観察2と整合）が影響している
可能性が高い。

### 本番反映と回帰確認

- `speaker_id.py`: `REMAP_THRESHOLD=0.35`を新設し`SpeakerLabeler.__init__`の
  `remap_threshold`デフォルトに設定。`SIM_THRESHOLD=0.45`（速報パス）は無変更。
- `realtime_transcribe.py`: `--speaker-remap-threshold`CLIオプションを追加
  （未指定時は`speaker_id.REMAP_THRESHOLD`が効く。明示的に値を渡した場合のみ
  上書き）。`Refiner._emit_turns()`のグローバル対応付け呼び出しを
  `threshold=self.speaker_labeler.remap_threshold`を渡す形に変更。
- `eval_diar.py`: `--sim-threshold`/`--remap-threshold`CLIオプションを追加、
  `--breakdown`でpyannote.metrics内訳を追加出力。
- **速報パス回帰確認**: `--method baseline`（`SIM_THRESHOLD`のみを使う経路、
  `remap_threshold`は一切関与しない）を5会議で再実行し、§6の元の数値と
  完全一致することを確認した（ES2011a 32.1%, IS1008a 18.2%, ES2004a 28.5%,
  IS1009a 32.9%, TS3003a 16.7%, 平均25.7% -- 1桁まで全て一致）。
  `remap_threshold`の追加が速報パス・ベースライン経路に一切影響しないことの
  直接的な裏付け。
- **清書パス最終確認**: `--method refine_diarize`（フラグ無し=新デフォルト）を
  5会議で再実行し、平均DER 13.9%（会議別: 16.8/4.0/17.8/16.9/14.1%）を確認。

### 最終DERサマリー（ベースライン→③④→⑤、AMI 5会議、collar=0.25s）

| 段階 | 手法 | 平均DER | 話者数（refは全会議4） |
|---|---|---|---|
| §6 ベースライン | 速報パスのみ（`SpeakerLabeler`単体） | 25.7% | 6〜9 |
| §7 ③④ | 清書パス統合、`diar_threshold=0.5`、単一閾値0.45 | 14.1〜14.3% | 8〜11（悪化した会議あり） |
| §8 ⑤（採用） | 清書パス、`remap_threshold=0.35`分離 | **13.9%** | 4〜8（全会議で改善） |

### 残課題

- **話者数の過大推定は残る**（refの4に対し4〜8）。§8の非対称閾値調整は
  DERと話者数の両方を改善したが、根治には至っていない。次に効きそうな
  方向性: (a) 速報パスのセントロイド管理自体の改善（定期的な再クラスタリング、
  ヒステリシス付き閾値、セッション終盤の全体再クラスタリング）、
  (b) ローカルクラスタ代表埋め込みの計算方法自体の改善（現状はクラスタに
  属する区間を単純連結してから1回埋め込み計算 -- 複数区間の埋め込みを
  個別計算して平均する方が安定する可能性）。いずれも本イテレーションの
  スコープ外（§5当初計画のイテレーション⑤で完結させる粒度を優先した）。
- オーバーラップ区間の扱い（先着優先）は引き続き未着手（§3設計方針通り、
  スコープ外として据え置き）。
- `download_models.py`にpyannote segmentation-3.0の取得を追加した
  (`SEGMENTATION_TAG="speaker-segmentation-models"`、既存の`campplus_sv.onnx`
  取得の直後、`--speakers`使用時のみ関与しモデル欠如時は速報パスのみへ
  フォールバックするため必須化はしていない)。README英日の`--speakers`説明を
  実測DERと清書パスの挙動変化に合わせて更新した。

## 9. イテレーション⑥実装結果: 速報パスのセントロイド管理(セントロイドマージ/新規話者ヒステリシス) -- 両案とも不採用

### 背景と目的

§8の残課題: グローバル話者数はrefの4に対し4〜8で過大推定のまま。§8の分析で
「速報パス（`SpeakerLabeler.label()`、`SIM_THRESHOLD=0.45`）が0.45閾値未満の
ミスのたびに新規グローバルセントロイドを開き続けるのが原因、清書パスの
対応付け閾値（`remap_threshold`）だけを緩めても速報パスが最初に開いた
分裂気味のセントロイド集合自体は変えられない」と特定していた。§8の
pyannote.metrics内訳分析では5会議中4会議でMiss（境界ノイズ）がConfusion
（話者取り違え）を上回っており、過大推定の**DERへの実害は小さい**ことも
分かっていた。つまり本イテレーションの主目的はDERの改善そのものではなく
**表示される話者数の正しさ**（ユーザー体験の問題）。

`speaker_id.SpeakerLabeler`に2つの独立した緩和策をコンストラクタ引数で
on/off可能な形で実装した（`scripts/speaker_id.py`）:

- **A. セントロイドの定期マージ**（`merge_enabled`, `merge_threshold`,
  既定`MERGE_THRESHOLD=0.80`）: `merge_centroids()`がグローバルセントロイド
  同士のコサイン類似度が閾値を超えるペアを畳み込み統合する。呼び出しタイミングは
  「清書グループが1つ閉じた直後」（`realtime_transcribe.Refiner._emit_turns()`の
  ローカル対応付け直後、`eval_diar.py`の`generate_diarize_hypothesis()`の
  グループループ末尾）で、`maybe_merge_centroids()`が無効時は即noop。統合は
  常に番号の若い方（作成が早い方）が生き残り、統合先インデックスは
  `_alias`辞書に記録して`_centroids`/`_counts`からは物理削除しない（既出ラベルの
  インデックスがずれて過去に出力済みのS{n}表記が壊れるのを防ぐため）。過去に
  出力済みの行のラベルは遡及修正しない（イベント再送が必要になるためスコープ外
  ―― これは当初計画通り）が、以後のマッチはすべて生き残ったセントロイドに
  収束する。`merge_history()`で累積の`{旧ラベル: 現行ラベル}`対応表を取得でき、
  `realtime_transcribe.py`はセッション終了時にこれをサマリー行として出力する。
- **B. 新規話者の開設ヒステリシス**（`hysteresis_enabled`,
  `hysteresis_min_hits`、既定`HYSTERESIS_MIN_HITS=2`）: 速報パスのミスで
  新規に開いたセントロイドは「仮」状態になり、自分自身が最良マッチとして
  `hysteresis_min_hits`回選ばれるまで、表示上は最も近い「確定済み」話者の
  ラベルにフォールバックする（内部的にはちゃんと独自セントロイドとして育つ）。
  セッション最初の話者はフォールバック先が存在しないため即座に確定される。

`scripts/eval_diar.py`と`scripts/realtime_transcribe.py`の両方にCLIフラグ
（`--merge-enabled`/`--merge-threshold`/`--hysteresis-enabled`/
`--hysteresis-min-hits`、realtime側は`--speaker-merge*`/`--speaker-hysteresis*`）
を追加し、`None`未指定時は`SpeakerLabeler`自身の既定値にフォールバックする
tri-state（`argparse.BooleanOptionalAction`、§8の`remap_threshold=None`と
同じ設計）にした。

### dev会議での比較（ES2011a, IS1008a）

各設定を複数回実行して確認した通り、このパイプラインには実行間で若干の
非決定性がある（ONNX Runtimeのマルチスレッド実行順序が閾値境界近くの
浮動小数点比較に影響するとみられる。baseline単体でもES2011aは実行ごとに
4話者/DER16.8〜17.0%と5話者/DER17.5〜17.7%の間で揺れた）。以下は複数回実行の
代表値。

**A（セントロイドマージ）閾値スイープ、ES2011a:**

| merge_threshold | DER | 話者数 | 備考 |
|---|---|---|---|
| 0.55〜0.60 | 49.4〜50.3% | 8 | **破局的**: 別人のセントロイド同士を大量誤統合（`merged={'S2':'S1','S3':'S1','S6':'S1','S7':'S1'}`） |
| 0.65 | 16.8% | 4 | baselineと一致（=無効時と同じ、統合が発生しない安全な高め設定） |
| 0.70〜0.75 | 17.5〜17.7% | 5 | ほぼ無変化 |
| 0.80（既定） | 17.7% | 5 | 統合0件、実質no-op |

IS1008aでは0.60〜0.70のどの閾値でも統合が一度も発生せず（`merged`が空）、
話者数7のまま変化しなかった。

**結論: Aは有効範囲がナイフエッジ**（0.55〜0.60は破局的崩壊、0.65以上は
ほぼ無効化）で、安全に効く閾値帯が実質存在しない。ES2011aで唯一効いて
見えた0.65もbaselineと数値が完全一致しており、統合が起きていない
（=無効と同じ）ことを示唆する。IS1008aでは効果ゼロ。**不採用。**

**B（新規話者ヒステリシス）`hysteresis_min_hits`スイープ:**

| min_hits | ES2011a DER | ES2011a 話者数 | IS1008a DER | IS1008a 話者数 |
|---|---|---|---|---|
| なし（baseline） | 16.8〜17.7%（揺れ） | 4〜5（揺れ） | 4.0% | 7 |
| 2（既定） | 17.5% | 5 | 4.0% | **5**（改善） |
| 3 | 19.5%（再現性あり、2回実行で一致） | 5（改善なし） | 5.2% | **4**（ref一致） |
| 4 | 18.7% | 3（過少推定側に振れる） | 5.2% | 4 |

`min_hits=2`はIS1008aで話者数を7→5に改善しつつDERはbaseline同等
（4.0%のまま）。`min_hits=3`はIS1008aでref通り4に一致するがES2011aで
DERが確実に悪化（16.8〜17.7%→19.5%、話者数の改善は伴わない）。

### test会議での汎化確認（ES2004a, IS1009a, TS3003a）

`hysteresis_min_hits=2`（既定候補）:

| 会議 | baseline DER/話者数 | hysteresis(2) DER/話者数 |
|---|---|---|
| ES2004a | 17.8% / 8 | 17.8% / 8（変化なし） |
| IS1009a | 16.9% / 6 | 17.0% / 6（誤差範囲） |
| TS3003a | 14.1% / 7 | 14.1% / **5**（改善） |

`hysteresis_min_hits=3`:

| 会議 | DER/話者数 |
|---|---|
| ES2004a | **24.7%** / 8（DER大幅悪化、話者数改善なし） |
| IS1009a | 16.8% / 5（改善） |
| TS3003a | 14.2% / **4**（ref一致） |

`min_hits=3`はTS3003aでref通り4に一致する魅力的な結果を出す一方、
ES2004aで+6.9ptのDER悪化を話者数の改善なしに引き起こす（8→8のまま）。
`min_hits=2`はdev/test全5会議でDER悪化なし、IS1008a・TS3003aの2会議で
話者数改善という、より安全な設定に見えた。ここまでは§8と同じ「dev会議で
選んで確認する」ワークフローに沿っており、`hysteresis_enabled=True,
min_hits=2`を採用候補としてデフォルト化する計画だった。

### 実運用回帰確認で判明した本命の欠陥: 短い2話者会話での話者消失

§5当初計画の「速報パスの単純ケース＝1〜2話者会話で回帰がないことを、
two_speakers系の既存スモークかユニットテストで担保」の通り、
`testdata/two_speakers.wav`（`tests/test_diarize.py`が2話者と検証済みの
実録音）に対して`GroupDiarizer`でセグメントを切り出し、時系列順に
`SpeakerLabeler.label()`へ通すテストを追加した
(`tests/test_speaker_id.py::test_fast_path_default_finds_both_real_speakers`/
`test_hysteresis_can_swallow_a_rare_real_speaker`)。

この録音は4セグメント中、話者0が1回（約3秒）しか発話しない
（話者1, 話者1, 話者0, 話者1の順）。`hysteresis_min_hits=2`で走らせると:

```
no-hysteresis (旧挙動):  ['S1', 'S1', 'S2', 'S1']  -- 2話者を正しく区別
hysteresis(min_hits=2):  ['S1', 'S1', 'S1', 'S1']  -- 話者0が永久にS1へ吸収される
```

話者0は仮センロイドとして1回しか自分自身にマッチしないため、
`hysteresis_min_hits=2`に達することがなく、確定済み話者（S1）への
フォールバック表示のまま最後まで抜け出せない。これは**AMI会議での
過大推定緩和よりも深刻な失敗モード**である: `speaker_id.py`冒頭の
モジュールdocstringが明記する通り、この機能の主用途は
「turn-taking conversations」（多くは1〜2話者）であり、そこで一度しか
発言しない実在の話者が黙って消える方が、多人数会議でのS{n}過剰よりも
ユーザー体験上の実害が大きいと判断した。

### 結論: A・Bとも実装はするが、デフォルトはどちらも無効のまま（不採用）

- **A（マージ）**: 安全に効く閾値帯がない（ナイフエッジ、破局的崩壊リスク）。
  `merge_enabled`既定`False`。
- **B（ヒステリシス）**: AMI会議の评価だけを見れば`min_hits=2`は
  無害〜改善だったが、実運用の主戦場である短い1〜2話者会話で実話者を
  消してしまう欠陥が`two_speakers.wav`回帰テストで見つかった。
  `hysteresis_enabled`既定`False`。

両方とも`SpeakerLabeler`のコンストラクタ引数・CLIフラグとしては実装済みで
使用可能（多人数会議を主用途とし、この trade-off を許容できる利用者は
オプトインできる）。デフォルト動作は§8終了時点から変更なし
（`remap_threshold=0.35`, `SIM_THRESHOLD=0.45`, `merge_enabled=False`,
`hysteresis_enabled=False`）。5会議の平均DERも§8と同一の傾向を維持している
ことを確認済み（本イテレーションはコードパスを変更していないため、
デフォルト設定でのDERは実行間の非決定性の範囲内で§8の数値
―― 平均13.9% ―― と一致する）。

### 話者数まとめ（本イテレーションで判明した内容、参考値）

| 会議 | ref話者数 | baseline話者数(§8時点) | A最良ケース | B(min_hits=2)話者数 | 採用後話者数 |
|---|---|---|---|---|---|
| ES2011a | 4 | 4〜5（非決定性あり） | 効果なし（no-op化） | 5 | 4〜5（変更なし） |
| IS1008a | 4 | 7 | 効果なし | 5 | 7（変更なし） |
| ES2004a | 4 | 8 | 未計測（Aは全体不採用のためtest会議は未走査） | 8（変化なし） | 8（変更なし） |
| IS1009a | 4 | 6 | 未計測 | 6（誤差範囲） | 6（変更なし） |
| TS3003a | 4 | 7 | 未計測 | 5 | 7（変更なし） |

「採用後話者数」列が§8終了時点から変わっていないのは、A・Bとも不採用で
デフォルト動作を変更しなかったため。話者数過大推定という§8の残課題は
本イテレーションでは解決できなかった、というのが最終的な結論。

### 残課題（次イテレーションへの申し送り）

- 話者数過大推定は依然未解決（refの4に対し4〜8）。本イテレーションで
  「セントロイド管理を後から直す」系のアプローチ2つ（定期マージ、
  ヒステリシス）を両方試し、どちらも実運用で安全に使えるほど頑健では
  ないと判明した。次に試す価値がありそうな方向性:
  - ヒステリシスの「確定」判定を回数ベースではなく、セッション終了時に
    未確定のまま残った仮センロイドを見直す一括後処理にする
    （早口の一度きりの発話をリアルタイムに待たせず、清書パスの
    タイミングで「本当に新規話者か、既存話者のノイズか」を再判定する）。
  - マージの閾値をコサイン類似度の絶対値ではなく、各センロイドの
    サンプル数・分散を加味した統計的な近さ（例:
    信頼区間が重なるか）にする、単純な固定閾値のナイフエッジ問題を
    避けられる可能性がある。
  - §8の残課題(b)で挙げていた「ローカルクラスタ代表埋め込みの計算方法
    自体の改善」（複数区間を個別embedしてから平均する）は本イテレーション
    でも未着手のまま。
- オーバーラップ区間の扱い（先着優先）は引き続き未着手。

## 10. フルパイプライン検証（本番経路: VAD→ASR→清書→グループ再分離→ターン分割出力）

### 背景と目的

§6〜§9のDER測定は`scripts/eval_diar.py`のASR抜きの軽量経路（VADで切った
セグメントに直接埋め込み・クラスタリングをかけるだけ）で行っており、
実際にユーザーが使う`scripts/realtime_transcribe.py`の本番経路
（VAD確定セグメント→ASR速報→`Refiner`の清書グループ化→
`GroupDiarizer`によるグループ内再分離→ターン単位の再デコード→
`[refine/S{n}]`出力）は通していなかった。朝のPR提案前の最終QAとして、
`--speakers`ありでAMI会議2本（各10分、4話者）を実際に完走させ、
ターン分割・レイテンシ・キュー挙動・メモリを確認した。

### 実行環境

- コマンド: `python scripts/realtime_transcribe.py --wav <file> --no-realtime
  [--speakers] --mode balanced`
- 対象: `testdata/eval_diar/IS1008a.wav`, `testdata/eval_diar/ES2011a.wav`
  （各600秒・16kHz mono、リファレンス話者数4）
- 計測: 子プロセスの壁時計時間、`psutil`によるプロセスツリー合計RSSの
  ピーク、`stdout`から`session summary`行の`mean_latency`/`max_latency`
  （速報パスのASRレイテンシ）

### 10.1 通し実行結果

| ファイル | `--speakers` | 完走 | wall time | peak RSS | segments (速報) | refineライン数 |
|---|---|---|---|---|---|---|
| IS1008a | なし | ✅ | 430.6s | 3749MB | 54 | 22 |
| IS1008a | あり | ✅（後述バグ発見・修正後は完全） | 474.8s | 3752MB | 54 | 70→71(修正後) |
| ES2011a | なし | ✅ | 218.3s | 3287MB | 75 | 36 |
| ES2011a | あり | ✅ | 247.3s | 3503MB | 75 | 66 |

2本ともクラッシュ・ハング・未捕捉例外なし（`returncode=0`）。ただし
IS1008aの`--speakers`実行で、完走はするが**会議終盤の内容が清書出力
から消える**という実害のあるバグを発見した（10.2）。

### 10.2 発見したバグ: シャットダウン時に清書ワーカーのバックログが
   ドレインされず終盤の内容が無音のまま消える（修正済み）

**症状**: IS1008a に `--speakers` を付けて通しで実行すると、`returncode=0`・
例外一切なしで完走するにもかかわらず、会議終盤（「finance/marketing/
Christineとのやり取り」に相当する約30秒分、速報パスには
`[S1|en/v3] Because uh she is doing the design ... Do we already have a cast`
として正しく出ている区間）の清書 `[refine/...]` 行が**一切出力されない**。
`--speakers`なしの同一ファイル・同一区間では、`[refine/en] ...`として
一字一句一致する内容が正しく出力される。

**再現条件**: 90秒に切り出した末尾クリップ単体では再現せず、10分フル尺
でのみ再現した。`Refiner._worker_loop`に`try/except`+`traceback.print_exc()`
を仕込んでフル尺を再実行しても例外は一切出力されなかった
（`scripts/realtime_transcribe.py`のこの診断ハンドラは調査用に追加し、
そのまま防御的措置として残した）。

**根本原因**（`scripts/realtime_transcribe.py`）: `Refiner`は単一の
FIFOワーカースレッド（daemon）で`_task_queue`を処理する。シャットダウン
処理の`finish()`は

```python
if refiner is not None:
    refiner.maybe_refine(0, force=True)
```

だけを呼んでいた。`maybe_refine(force=True)`は「今エンキューする**その
1タスクだけ**」を`threading.Event`で待つ（`done.wait()`）。しかも
`self.spans`が既に空なら**何もエンキューせず即return**する
（`if not self.spans: return`）。`--speakers`は話者ごとの再分離＋
ターンごとの再デコードが発生するぶん通常の清書より大幅に重く
（1グループあたり0.5〜1s程度で済む通常清書に対し、ダイアライゼーション＋
複数ターンの個別ASR再デコードが乗る）、10分尺では速報パス（メイン
スレッド）がワーカースレッドの処理速度を上回って先に進み、
`run_stream`終了時点で**すでに非同期（`force_sync=False`）でキューに
積まれたまま未処理のグループがバックログとして残っている**ケースが
生じる。この状態で最後のグループがちょうど`run_stream`中に非同期フラッシュ
済み（＝`self.spans`が空）だと、`finish()`の`maybe_refine(force=True)`は
何もせず即座に戻り、`done.wait()`による待機も発生しない。その直後に
メインスレッドは`session summary`を出力してプロセスを終了し、daemon
ワーカースレッドはバックログを処理しきる前に強制終了される――
これが例外なし・`returncode=0`・内容だけ消える、という観測と完全に一致する。

**修正**（`scripts/realtime_transcribe.py`の`finish()`）: `maybe_refine`の
直後に`refiner._task_queue.join()`を追加し、既にキューに積まれている
全タスク（同期・非同期問わず）の完了を待ってから終了するようにした。

```python
if refiner is not None:
    refiner.maybe_refine(0, force=True)
    refiner._task_queue.join()
```

**検証**: 修正後にIS1008a・ES2011aを`--speakers`ありで再実行し、両方とも
最終グループの内容が欠落なく`[refine/...]`に出力されることを確認した
（IS1008aは71行、末尾が`[refine/S1|en] ... Do we already have a cast`で
速報パスの最終行と一致）。回帰テストとして
`tests/test_units.py::test_refiner_shutdown_must_drain_full_backlog_not_just_last_task`
を追加し、モデルなしで「非同期フラッシュ直後は`self.spans`が空になる →
`maybe_refine(force=True)`は待たずに戻る → `_task_queue.join()`を呼んで
初めてバックログの出力が観測される」という再発防止のガード条件を固定した。
`pytest tests -q`は127件全通過（既存126件＋新規1件）。

### 10.3 ターン分割出力の実例

修正後のIS1008a（`--speakers`）から、1つの清書グループ内で複数話者に
正しく分割された実例:

```
[refine/S1|en] new project which we are going to discuss now so I want to
  introduce first of all the names and the colleagues here
[refine/S1|en] And what you're uh doing?
[refine/S2|en] Uh sure. My name is Agnes and I'm an user usability user
  interface designer.
[refine/S3|en] My name is Ed and I do accounting.
[refine/S1|en] Uh how do you spell your name uh E D. E. D, okay.
[refine/S3|en] E D.
[refine/S4|en] Do you also do marketing?
```

`--speakers`なしでは同じ区間が単一の多数決ラベル1行に潰れる
（§1で記載した既知の限界）のに対し、`--speakers`ありでは相槌の応酬
（"E D." のような1〜2語の短いターン）まで含めて話者ごとに分離できている。
清書テキスト自体の崩壊（単語の欠落・文字化け）は目視で確認した範囲では
見られなかった。ただし後述10.6の通り話者数の過大推定（S1〜S7、
ES2011aはS1〜S13相当）は残っている。

### 10.4 清書レイテンシへの影響

`session summary`の`mean_latency`/`max_latency`は速報パス（VAD確定
セグメント単位のASR）のレイテンシで、`--speakers`は速報パスに
埋め込み計算1回を追加するのみなので影響は小さいと予想した通りだった:

| ファイル | 指標 | `--speakers`なし | `--speakers`あり | 差分 |
|---|---|---|---|---|
| IS1008a | mean_latency | 783ms | 838ms | +55ms |
| IS1008a | max_latency | 4318ms | 4353ms | +35ms |
| ES2011a | mean_latency | 456ms | 479ms | +23ms |
| ES2011a | max_latency | 2256ms | 2206ms | -50ms |

速報パスのレイテンシはほぼ変わらない。コストが乗るのは清書パス側で、
壁時計全体の増分（IS1008a: 430.6s→474.8s、+44.2s / ES2011a:
218.3s→247.3s、+29.0s、いずれも`--no-realtime`での処理時間差）が
それに相当する。清書ワーカーは速報パスと並行して裏で走る設計なので
体感の速報表示には影響しないが、10.2のバグが示す通り**清書が速報パスに
追いつききれないバックログが実際に発生する**ため、清書の「最終的な
到着」は録音直後ではなく、10〜30秒以上遅れて（場合によっては音声終了後
まで）続くことがある。

### 10.5 FIFOワーカーとの相互作用（キュー詰まり）

10.2の通り、キューが「詰まる」こと自体は実際に起きていた
（バックログが未処理のまま残る）。ただし本番のFIFO設計自体は健全で、
シャットダウン時に`_task_queue.join()`で待てば必ず全件処理される
（デッドロックやスタベーションは観測されず、単に「終了処理が
バックログを待たずに先に終わっていた」というシャットダウン手順の
バグだった）。稼働中（ストリーミング中）の詰まりについては、キューの
深さを直接ログするインストルメンテーションはコードベースに無く、
今回は間接的に（清書行の出力タイミングの空白区間・壁時計の増分から）
確認した。次回の作業では`_task_queue.qsize()`をサマリーに含める
などバックログの深さを直接観測できるようにしておくと、今回のような
バグの切り分けが速くなる。

### 10.6 RSS/安定性

| ファイル | `--speakers`なし peak RSS | `--speakers`あり peak RSS | 差分 |
|---|---|---|---|
| IS1008a | 3749MB | 3752MB | +3MB |
| ES2011a | 3287MB | 3503MB | +216MB |

差分はモデル込みプロセスツリー全体（ASR・LID・翻訳ワーカー等含む）の
ピークで、CAM++埋め込みモデル自体は28MBと小さいため、話者追加に伴う
メモリ増分は誤差〜数百MB程度に収まっている。10分尺2本を通しても
メモリリークを疑う継続的な増加傾向は見られなかった（本測定はプロセス
単位のピークのみで、時系列のRSS推移は取っていない）。

過大推定の副作用として、ES2011aの`--speakers`実行ではセントロイドが
S1〜S13相当まで開いた（リファレンス4話者）。これは§9で記録した
軽量DER測定（`eval_diar.py`、ref4に対し4〜5）よりも明確に悪い数字で、
本番の清書パスが`_emit_turns()`経由で`speaker_labeler.embed()`/
`match_embedding()`を清書グループごと・ターンごとに追加で呼ぶため、
軽量経路の測定より頻繁にセントロイドマッチングが走り、閾値を跨ぐ
機会そのものが増えているのが一因とみられる（未検証の推測。§9の
残課題「話者数過大推定」がフルパイプラインではむしろ悪化しうる、
という新しい懸念として申し送る）。

### 10.7 実用判断・申し送り

- **10.2のバグは修正済み・回帰テスト追加済みで、PRに含めてよい状態**。
  修正前の状態でPRを出していた場合、長尺会議の`--speakers`利用で
  終盤の清書が無言で消えるという、ユーザーから見て気づきにくい
  重大な体験劣化を埋め込むところだった。
- ターン分割の質そのもの（10.3）は実用的で、テキストの崩壊も
  見られない。相槌レベルの短いターンまで拾えている。
- 話者数の過大推定（10.6）は§8/§9で既知の課題だが、フルパイプラインで
  測ると軽量DER評価より悪化して見える会議があった。DERへの実害は
  §8で小さいと確認済みだが、**表示される話者ラベルの数がユーザー体験に
  直結する**ことを踏まえると、次イテレーションの優先課題として
  引き続き重い。
- レイテンシ・メモリの両面で、`--speakers`を本番投入すること自体への
  障害は見当たらなかった（速報パスの体感速度は変わらず、清書の遅延も
  実用範囲内、メモリ増分も小さい）。
- 申し送り: キューの深さを可観測にする計装がないため、同種の
  バックログ関連の不具合は次回も気づきにくい。`_task_queue.qsize()`や
  最終グループのタイムスタンプをサマリーに出す等の軽量な計装を
  今後のイテレーションで検討する価値がある。

### 10.8 話者数過大推定（10.6）の根本原因調査: 配線漏れではなく、
     「本番の表示」と「DER評価のハイポセシス構成」が別物であることに
     起因する測定アーティファクト（設計に根差す・修正なし）

10.6の「未検証の推測」を検証するため、`speaker_id.SpeakerLabeler`に
セントロイドの開設元（`fast`=速報パスの`label()`、`remap`=清書パスの
`_emit_turns()`/`generate_diarize_hypothesis()`の局所クラスタ→
グローバル対応付け）と最終マッチ回数を記録する計装
（`match_embedding()`の`source`引数、`centroid_open_counts()`、
`centroid_summary()`）を追加し、`realtime_transcribe.py`にも
清書グループのクローズ回数（`refine_groups_closed`）と言語境界による
強制フラッシュ回数（`refine_lang_boundary_flushes`）を追加した上で、
疑うべき候補を順に潰した。

**候補①（remap閾値の配線漏れ）: 否定**。`realtime_transcribe.py`の
`--speaker-remap-threshold`未指定時は`speaker_kwargs`に`remap_threshold`
キー自体を入れず、`SpeakerLabeler.__init__`のデフォルト引数
`REMAP_THRESHOLD`（0.35）がそのまま使われる（`eval_diar.py`の
`generate_diarize_hypothesis`も同じtri-state idiomで同じデフォルトに
落ちる）。コードを読む限り両経路は完全に同じ値に配線されており、
閾値の食い違いは無かった。

**候補②（速報パスと清書パスの二重開設）: 「バグ」ではなく設計通り、
ただし過大推定の主因でもなかった**。速報パス（`label()`,
SIM_THRESHOLD=0.45）と清書パスのremap（`match_embedding(threshold=
remap_threshold)`, REMAP_THRESHOLD=0.35）が同一のグローバルセントロイド
集合に書き込むこと自体は`speaker_id.py`の`__init__`docstringに明記された
意図通りの設計で、`eval_diar.py`の`generate_diarize_hypothesis`も
（`realtime_transcribe.Refiner._emit_turns()`を"exactly"再現すると
docstringに書かれている通り）全く同じ二重構造を持つ。実測でも
ES2011aの開設元内訳は本番`{'fast': 11, 'remap': 2}`（計13）、
eval側`{'fast': 9}`（remapは0開設、計9）で、二重構造自体は両経路
共通であり差の主因ではない。

**候補③（ターン別再デコード時の余分なembed呼び出し）: 否定**。
`_emit_turns()`のembed呼び出しはローカルクラスタごとに1回（`local_id`の
ユニーク数だけ）で、`eval_diar.py`の同等ループと呼び出し回数・引数とも
一致している。ターンのASR再デコード（`asr.transcribe()`）自体は
ターンごとに走るが、これは`speaker_labeler`へは一切触れない
（読み取りも書き込みもしない）ことをコードで確認した。

**実際に見つかった2つの寄与要因**（いずれも「配線ミス」ではなく、
`eval_diar.py`が本番の挙動を意図的に単純化して再現している評価ハーネス
であること自体に起因する構造的な差）:

1. **グループの過分割（小さいが実在する寄与）**。`eval_diar.py`の
   `group_segments()`はVADセグメント全件（無音・SFX誤検出も含む）を
   境界点として使うのに対し、本番の`realtime_transcribe.drain_segments()`
   はASRが空文字列を返したVADセグメントを`Refiner.add_span()`に一切
   渡さない（そのセグメントは清書グループの一部にならない）。ところが
   `Refiner.maybe_refine()`の無音ギャップ判定（`due`）は`run_stream()`が
   渡す**実際の経過サンプル数**（スキップされたセグメントの分も含む
   実時間）を基準にしており、「最後に採用されたスパンの終端」からの
   ギャップを見る。そのため、スキップされたセグメント自身の開始・終了が
   境界点として使えるeval側では閾値未満のギャップでも、それが使えない
   本番側では同じ実時間ギャップが`GROUP_GAP_S`（2.0秒）を超えて
   しまい、余分なグループ分割が起きうる。ES2011aの実測でも
   本番`refine_groups_closed=36`に対しeval`n_groups=32`
   （`refine_lang_boundary_flushes=0`だったので言語境界分割は無関係、
   純粋にこの機構による差）。分割が増えれば清書remapの呼び出し回数も
   増え、新規センロイドを開く機会がわずかに増える
   （実測: ES2011aの`remap`経由の開設は本番2件・eval0件）。
   回帰テスト:
   `tests/test_diar_eval.py::test_production_over_splits_relative_to_eval_replica_on_a_skipped_vad_segment`
   がこの機構を最小構成で再現し固定した。

2. **主因: DERハイポセシスの構成方法が「一発限りの外れ値センロイド」を
   構造的に握りつぶすのに対し、本番のコンソール出力は握りつぶさない**。
   `centroid_summary()`でES2011aの本番実行を検査すると、開設された
   13個のセントロイドのうち実に6個（S1, S9〜S13）が
   `final_match_count == 1`（＝開設した瞬間の1回しかマッチしていない
   「一発限りの外れ値」）、さらに2個（S4=2回, S6/S7=各1回のremap開設）も
   ほぼ同様に希少だった。本番は`drain_segments()`が非空テキストの
   VADセグメントごとに`[S{n}|...]`を**無条件に**print するため、この
   種の一発限りの外れ値も開設された瞬間にそのままユーザーへ見える形で
   出力される。一方`eval_diar.py`のDERハイポセシスは、各清書グループに
   つき（a）グループ内速報ラベルの**多数決**、または（b）ローカル
   ダイアライゼーションで2話者以上見つかった場合のみターンごとの
   remapラベル、のいずれかしか出力に含めない構成になっている
   （`generate_diarize_hypothesis()`参照）ため、一度もどのグループの
   多数決にもならず、ローカルクラスタとしても再発見されなかった
   一発限りのセントロイドは、**開設されたこと自体がスコアリング対象の
   ハイポセシスに一切現れない**。実測でもES2011aのeval側は9個の
   センロイドを開設していながら最終`hyp_speakers=4`（5個は一度も
   ハイポセシスに現れなかった）。逆にIS1008aでは開設した7個が
   偶然すべて何らかのグループの多数決を取り、eval側`hyp_speakers=7`が
   本番の可視話者数（S1〜S7）と一致した——つまり両者の食い違いの
   大きさは、たまたまどれだけの一発限りセントロイドがグループの
   多数決を取れるかという**運**に左右される、測定方法起因のばらつき
   だと分かる。回帰テスト:
   `tests/test_speaker_id.py::test_centroid_summary_flags_one_off_opens_with_a_final_match_count_of_one`
   がこの「一発限りは`final_match_count == 1`」という不変条件を固定した。

**結論**: §10.6の懸念（フルパイプラインの方が軽量DER評価より話者数
過大推定が悪化しうる）は事実として確認されたが、原因は配線漏れでは
なく、(1) グループ過分割によるわずかな開設機会の増加と、(2) それより
支配的な要因として、DER評価のハイポセシス構成方法が一発限りの外れ値
センロイドを構造的に握りつぶすのに対し本番コンソールは握りつぶさない、
という**測定方法の非対称性**である。したがって「DERが13.9%前後で
安定している」こと自体は本番のクラスタリング品質の実害を必ずしも
過小評価していないが、**ユーザーに見える話者ラベルの数**はDERが
測っているものより悪く見えることがある、という事実は申し送りとして
重い。

**現実的な緩和策の選択肢**（いずれも未採用・デフォルト変更なし。
どれもトレードオフがあり、単独の正解ではない）:

- **A. `--speaker-hysteresis`を本番の既定運用として案内する**。§9で
  実装済みの新規話者ヒステリシス（`hysteresis_min_hits`回再出現するまで
  新規S{n}として表示しない）は、まさに「一発限りの外れ値を表示前に
  握りつぶす」ことをそのまま実現する機構であり、この節で見つかった
  主因に対してピンポイントに効く。ただし§9で既に確認済みの通り、
  短い2話者録音で2番目の話者が1回しか発話しない場合にその話者を
  永久に1人目のラベルへ吸収してしまう副作用がある
  （`tests/test_speaker_id.py::test_hysteresis_can_swallow_a_rare_real_speaker`）。
  多人数会議での利用が主目的なユーザーへのオプトインとして案内する
  のは妥当だが、デフォルトにするには短時間・少人数録音での回帰が
  未解決のまま。
- **B. 新規センロイドを「開く」条件と「表示する」条件を分離する**
  （今回のAとは別の実装アプローチ）。ヒステリシスは既存センロイドの
  マッチングロジックそのものを変えるため2話者録音の副作用を持つが、
  「新規に開いたセントロイドは、その場では`S{n}`として表示しつつ
  内部的には`final_match_count`が閾値（例: 2）に達するまで`provisional`
  のまま残し、一定時間内に再発しなかった一発限りの行だけを事後的に
  usoの警告やマージ候補として扱う」というような**表示層のみの緩和**
  であれば、話者そのものを取り違えるリスクを避けつつ、コンソールでの
  ノイズだけを減らせる可能性がある。未実装・未検証。
- **C. VADセグメントの短さ・信頼度でフィルタする**。一発限りの外れ値の
  多くは短い/ノイズの多いセグメント（相槌、咳、被り、SFX）に由来する
  可能性が高い（今回の計測では個々のセグメント長・SNRまでは相関を
  取れていない）。新規センロイドを**開く**条件にセグメント長や
  ASR確信度の下限を追加すれば、Bと同様に既存話者の再発見には影響せず、
  ノイズ由来の新規開設だけを減らせる可能性がある。未実装・未検証、
  かつAMIコーパスでの効果測定が必要。
- **D. 現状維持＋UX側での説明**。DERは実質的な書き起こし品質の指標
  として妥当な値（13.9%前後）を保っているため、「表示される話者数が
  やや過大」であることを既知の制限としてドキュメント（README等）に
  明記し、ユーザーには`--speaker-hysteresis`または`--speaker-merge`
  （いずれもオフがデフォルト）を多人数会議向けオプトインとして
  案内するに留める、という判断もあり得る。

次イテレーションで緩和策を実装する場合は、A/Bの優先順位（Aは既存
実装の展開で低コスト、Bは新規実装だが2話者録音への副作用を避けられる
可能性がある）と、Cのための短時間セグメント統計の追加計測から
始めるのが妥当と考える。

## 11. イテレーション⑦実装結果: 表示レイヤのみの仮ラベル（issue #11、§10.8選択肢B）

### 背景

§10.8で特定した過大表示の主因は、一発限りの外れ値センロイド（開設後
二度とマッチしない）がDERハイポセシスには現れない一方、本番コンソールは
開設された瞬間に無条件でそのS{n}をprintすることだった。§9で試した2案
（セントロイド定期マージ・新規話者ヒステリシス）はどちらも**割当ロジック
自体を変える**ため、2話者会話で1回しか発話しない実話者を永久に他方へ
吸収する欠陥（`test_hysteresis_can_swallow_a_rare_real_speaker`）を生み
不採用になった。本イテレーションは§10.8の選択肢B「割当は一切変えず、
表示だけを変える」を実装した。

### 実装

`scripts/speaker_id.py`の`SpeakerLabeler`に、既存の`_counts`（センロイド
ごとの累積マッチ回数、`match_embedding()`が更新）を読むだけの3メソッドを
追加した。`match_embedding()`/`label()`自体は一切変更していない。

- `is_provisional(label)`: そのラベルのセントロイドが
  `PROVISIONAL_CONFIRM_HITS`（定数、既定2）回未満しかマッチしていなければ
  `True`。§9のヒステリシス`hysteresis_min_hits`既定値と同じ2だが、
  割当を変えない別機構なので独立した定数として定義した。
- `display_label(label)`: provisionalなら`f"{label}?"`（例: `S5?`）、
  確定済みなら`label`をそのまま返す。
- `provisional_label_count()`: セッション終了時点でまだprovisionalの
  ままのラベル数（マージ済みでエイリアス化された死んだスロットは
  除外）。セッションサマリー行として出力。

`_counts`は単調増加（減ることがない）ため、一度確定したラベルは
以後ずっと確定のままで、§9のヒステリシスのような「表示が後から
不安定になる」問題は起きない。

`scripts/realtime_transcribe.py`の3箇所（速報パスの`drain_segments()`、
清書のターン別出力`_emit_turns()`、清書の多数決フォールバック
`maybe_refine()`）すべてで、**内部で使う正規ラベル（グルーピング・
多数決の対象）と、印字・SSE配信・トランスクリプト書き込みに使う表示
ラベルを明確に分離**した: `Refiner.add_span()`に渡すラベルは常に
`speaker_labeler.label()`の生の戻り値（正規ラベル）のままで、
`display_label()`は`print()`/`server.publish()`/トランスクリプト書き込み
の直前でのみ呼ぶ。これにより多数決やDERハイポセシス構成（§10.8の
eval側ロジック、`eval_diar.py`は今回一切変更していない）は完全に
影響を受けない。

過去に印字済みの行は遡及書き換えしない（§9のマージ`merge_history()`と
同じ制約——SSE/コンソールに一度送った行を後から差し替える機構が無い）。
これは設計上の判断として明示的に受け入れ、セッションサマリーに
「セッション終了時点でまだprovisionalのラベル数」を出力することで、
読者が後から「このセッションでは何個が仮表示のまま終わったか」を
把握できるようにした。

### 割当ロジック不変の確認

`scripts/eval_diar.py`は今回のイテレーションで一切変更していない
（`speaker_id.py`への変更は既存コードの変更なしの追加メソッドのみ、
`git diff`で確認）。`--method refine_diarize`でES2011aを再実行した
結果は`DER=17.7%  ref_speakers=4 hyp_speakers=5 ... opened_by={'fast': 9}`
で、§9の記録値（`hysteresis_min_hits`既定2適用前のbaseline、
`DER 16.8〜17.7%（揺れ）/ 話者数4〜5`）と完全に一致する範囲に収まった。
`pytest tests -q`は141件全通過（既存132件＋本イテレーションの新規9件）。

### ES2011a本番実行での確定話者数 before/after

`python scripts/realtime_transcribe.py --wav testdata/eval_diar/ES2011a.wav
--no-realtime --speakers --mode balanced`を実行（§10.8と同じセントロイド
開設内訳: `opened_by={'fast': 11, 'remap': 2}`、計13個、§10.8の実測値と
一致——センロイド開設自体は本イテレーションで変わっていないことの再確認）。

| | before（§10.8まで） | after（本イテレーション） |
|---|---|---|
| コンソールに出るS{n}の総数 | 13（S1〜S13、全て無条件で確定済みとして表示） | 13ラベル中5個が確定済み表示（S2,S3,S4,S5,S8）、8個が仮表示`S{n}?`（S1?,S6?,S7?,S9?,S10?,S11?,S12?,S13?） |
| 「一発限りの外れ値」の見た目 | 他の実話者と区別つかない`S9`のような表示 | `S9?`のように一目で仮だとわかる表示 |
| セッションサマリー | なし | `speaker labels still provisional at session end: 8` |

実際のログ抜粋（`final_match_count`が1のセントロイドが仮表示になっている
ことの確認）:

```
[S1?|en/v3] Uh  (seg=1.9s, ...)
[refine/S6?|en] Mm.
[S9?|en/v3] Um  (seg=1.8s, ...)
[S13?|en/v3] This is not.  (seg=2.1s, ...)
```

`centroid_summary()`の内訳: `[('S1','fast',1), ('S2','fast',64),
('S3','fast',24), ('S4','fast',2), ('S5','fast',3), ('S6','remap',1),
('S7','remap',1), ('S8','fast',5), ('S9','fast',1), ('S10','fast',1),
('S11','fast',1), ('S12','fast',1), ('S13','fast',1)]` ——
`final_match_count>=2`（確定済み）はS2/S3/S4/S5/S8の5個のみで、
これが「実際に複数回発話が確認された、信頼できる話者ラベル」の実数に
近い。残り8個は一発限りで、今回の変更後は全て`?`付きの仮表示になる。

### 2話者ケースの確認

`testdata/two_speakers.wav`（実録音、2話者中1人は4セグメント中1回
=約3秒しか発話しない）に対する新規テスト
`test_provisional_display_does_not_reproduce_section_9s_swallowing_bug`
で、割当（`label()`の戻り値、正規ラベル）が引き続き2種類以上に分かれる
こと、かつ表示ラベル（`display_label()`適用後）も2種類以上に分かれる
こと（希少な話者が仮表示`S2?`のまま残るだけで、§9のヒステリシスのように
もう一方のラベルへ吸収されて1種類に潰れることがない）を固定した。
これは実装の構造上（`display_label()`は`_counts`を読むだけで
`_centroids`/`_counts`/`_alias`を一切変更しない）自明に成り立つはずだが、
§9の教訓を踏まえて明示的な回帰テストとして固定した。

### 仮表示の表記仕様

- 新規に開設されたセントロイドは、開設された瞬間から`final_match_count`
  （自分自身への再マッチ回数）が`PROVISIONAL_CONFIRM_HITS`（既定2）に
  達するまで、表示名の末尾に`?`を付けて仮表示する（例: 4番目に開設された
  センロイドなら`S4?`）。
- 2回目のマッチで即座に確定し、以後同セッション内では二度と`?`付きに
  戻らない（`_counts`は単調増加のため）。
- 過去に印字・配信済みの行は遡及的に書き換えない。確定前に出た行は
  `?`付きのまま残る。セッション終了時点でまだ確定していないラベル数は
  セッションサマリーの`speaker labels still provisional at session end`
  行で報告する。
- 適用範囲: 速報パス（`drain_segments`のコンソール出力・
  `server.final()`のSSE）、清書のターン別出力（`_emit_turns`の
  `[refine/...]`行・SSE・トランスクリプトファイル）、清書の多数決
  フォールバック行（`maybe_refine`の`[refine/...]`行・SSE・
  トランスクリプトファイル）の全てに一貫して適用。
- `Refiner.add_span()`に渡るラベルと多数決対象は常に正規ラベル
  （`?`なし）のまま。`eval_diar.py`のDERハイポセシス構成は本変更の
  影響を受けない。

### 残課題

- 表示レイヤの緩和だけでは、根本の「一発限りセントロイドが多数開設される」
  こと自体（§10.8の主因分析）は変わらない。今回の変更は「過大表示された
  ラベルをユーザーが仮とわかる形で見分けられるようにする」ものであり、
  センロイド開設数そのものを減らす施策ではない。
- `PROVISIONAL_CONFIRM_HITS`は§9の`HYSTERESIS_MIN_HITS`既定値と同じ2に
  揃えたが、独立してチューニングされていない。表示のみの変更で実害が
  小さいため、本イテレーションでは2固定のまま様子見とした。

## 出典

- [Speaker Diarization — sherpa-onnx docs](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html)
- [k2-fsa/sherpa-onnx offline-speaker-diarization.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py)
- [k2-fsa/sherpa-onnx releases: speaker-segmentation-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)（`gh api` で実サイズ確認: `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` = 6,958,444 bytes）
- [pyannote/segmentation-3.0 (Hugging Face)](https://huggingface.co/pyannote/segmentation-3.0)
- [onnx-community/pyannote-segmentation-3.0 (Hugging Face)](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)
- [3D-Speaker (modelscope, CAM++)](https://github.com/modelscope/3D-Speaker)
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) / [AMI License (CC BY 4.0)](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- [VoxConverse (joonson/voxconverse)](https://github.com/joonson/voxconverse)
- [pyannote.metrics (GitHub)](https://github.com/pyannote/pyannote-metrics) / [pyannote-metrics (PyPI)](https://pypi.org/project/pyannote-metrics/) (導入・実測: 本イテレーション⑤)
- [simpleder (PyPI)](https://pypi.org/project/simpleder/)
- `.venv` 内 `sherpa_onnx` 1.13.6 実体の `dir()`/`help()` 出力（本調査で直接確認）
- リポジトリ内: `scripts/speaker_id.py`, `scripts/realtime_transcribe.py`,
  `scripts/download_models.py`, `README.md`, `docs/GOALS.md`,
  `docs/BENCHMARKS.md`
