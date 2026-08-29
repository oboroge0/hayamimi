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

## 12. Round 2: min_duration_on/off の露出とスイープ、overlap除外採点、TS3003a miss診断

### 背景

Round 1（§8〜§11、DER 14.1%が到達点）で判明していたが手を付けていなかった
3点を今回のラウンドで潰す。

- **overlap（同時発話区間）がmissの構造的な下限になっている**: reference側の
  overlap比率（≥2話者が同時に発話している時間の割合）はES2004a 15.1%、
  ES2011a 10.1%、IS1009a 12.8%、IS1008a 2.4%、TS3003a 2.1%。hayamimiは
  1区間=1話者ラベルしか出さないため、overlap比率とmissはほぼ1:1で連動する
  （TS3003aだけはoverlapが2.1%しかないのにmissが8.7%あり、overlapだけでは
  説明できない未解明のmissが残っている）。
- **`min_silence_duration=0.5`にすると全体DERが13.5%まで下がるが、内訳を見ると
  伸びているのはConfusionで、Missではない**（IS1009a confusion 12.4%→10.2%）。
  ただし`min_silence_duration`は本番のライブVAD自体の設定で、0.35s→0.5sに
  上げると字幕確定が0.15s遅れる。ライブ側のレイテンシを犠牲にしてまで
  この設定をデフォルトにするのは割に合わないため、**本番デフォルト変更は
  見送り、同じconfusion改善を別の経路で得られないか探す**、というのが
  Round 1の宿題だった。
- `scripts/diarize.py`の`GroupDiarizer`は、清書グループの音声だけを対象にした
  オフライン再分離（`OfflineSpeakerDiarizationConfig`）を持ち、そこにも
  `min_duration_on=0.3`/`min_duration_off=0.5`がハードコードされていた。
  これはライブVADとは別物（清書グループが確定した*後*、そのグループの
  バッファだけに対して働く）なので、ここを動かしてもライブの字幕確定
  レイテンシには影響しない。Round 1の宿題（confusion改善をライブ非依存の
  経路で得る）の候補として、まずこの2値を露出してスイープする。

タスクは4つ: (T1) `min_duration_on`/`min_duration_off`の露出とスイープ、
(T2) overlap区間を除外したDER採点オプションの追加、(T3) TS3003aの
未解明missの内訳診断、(T4) 本セクションでのまとめ。

### T1: min_duration_on/off の露出とスイープ

`diarize.GroupDiarizer.__init__`に`min_duration_on`/`min_duration_off`を
コンストラクタ引数として追加した（デフォルトは従来通り0.3/0.5、
`DEFAULT_MIN_DURATION_ON`/`DEFAULT_MIN_DURATION_OFF`として定数化）。
`scripts/eval_diar.py`に`--min-duration-on`/`--min-duration-off`を、
既存の`--diar-threshold`と同じ配線パターン（`None`はモジュールデフォルトに
フォールバック）で追加した(`scripts/diarize.py`, `scripts/eval_diar.py`)。

採用基準（Round 1から継続）: 平均DERが**0.3pt以上改善**かつ**どの会議も
0.5pt超の悪化なし**かつ**ライブ経路のレイテンシに影響しない**ことの3条件を
すべて満たした場合のみ新デフォルトに採用する。それ以外は「試したが不採用」
としてオプションのまま残す。

AMI 5会議、collar=0.25s、`--method refine_diarize --breakdown`、他は
デフォルト（`remap_threshold=0.35`, `sim_threshold=0.45`, `diar_threshold=0.5`,
`min_silence=0.35`）での実測:

| 設定 | 平均DER | ES2011a | IS1008a | ES2004a | IS1009a | TS3003a |
|---|---|---|---|---|---|---|
| baseline（Round 1到達点） | 14.1% | 17.7% | 4.0% | 17.8% | 16.9% | 14.1% |
| min_duration_off=0.25 | 14.1% | 16.9% | **4.7%** | 17.8% | 17.2% | 14.0% |
| min_duration_off=0.35 | 14.0% | 16.9% | 4.3% | 17.8% | 17.0% | 14.0% |
| min_duration_on=0.15 | 14.1%（baselineと完全一致） | 17.7% | 4.0% | 17.8% | 16.9% | 14.1% |

内訳（Miss/FalseAlarm/Confusion、参照総発話時間比）:

| 設定 | ES2011a | IS1008a | ES2004a | IS1009a | TS3003a |
|---|---|---|---|---|---|
| baseline | miss11.3/fa3.5/conf6.8 | miss2.6/fa3.5/conf0.7 | miss14.5/fa3.7/conf5.0 | miss6.8/fa4.5/conf12.4 | miss8.7/fa6.5/conf0.7 |
| off=0.25 | miss11.5/fa3.5/conf5.8 | miss3.3/fa3.2/conf0.7 | miss14.6/fa3.5/conf4.9 | miss7.7/fa4.0/conf11.9 | miss8.8/fa6.0/conf0.8 |
| off=0.35 | miss11.5/fa3.5/conf5.8 | miss2.8/fa3.4/conf0.7 | miss14.5/fa3.6/conf5.0 | miss7.4/fa4.1/conf12.0 | miss8.8/fa6.0/conf0.8 |
| on=0.15 | miss11.3/fa3.5/conf6.8 | miss2.6/fa3.5/conf0.7 | miss14.5/fa3.7/conf5.0 | miss6.8/fa4.5/conf12.4 | miss8.7/fa6.5/conf0.7 |

**`min_duration_on=0.15`は全会議・全指標がbaselineと完全一致** —
清書グループ内で0.3秒未満の短い発話ターンがそもそもほとんど存在しない
（下げても拾うものがない）ことを意味する。これは後述T3の診断結果
（min_duration_onで説明できるmissは実質ゼロ）とも整合する。

`min_duration_off`は動きはするが小さい。0.25側はIS1008aが4.0%→4.7%
（+0.7pt、採用基準の0.5pt超悪化に抵触）、平均も横ばい（14.1%→14.1%、
実測は14.10%台での微増）で**却下**。0.35側は平均14.1%→14.0%
（0.1pt改善、採用基準の0.3pt未満）、どの会議も0.5pt超の悪化はないが
**改善幅が採用基準に届かず却下**。confusionへの効き方もES2011aで
6.8%→5.8%と多少動くが、Round 1の`min_silence=0.5`ほどの明確な
confusion改善（IS1009a 12.4%→10.2%）には届かなかった。

**コンボ実行は行わなかった**: タスク設計では「min_duration_offで改善が
見えたら最良ペアで1回だけコンボを回す」としていたが、off=0.25/0.35
いずれも採用基準（0.3pt以上）を満たす改善を示さず、on=0.15は無効
だったため、組み合わせる意味のある候補が無かった。

**結論（T1）: `min_duration_on`・`min_duration_off`とも、デフォルト変更は
不採用。オプションとして`--min-duration-on`/`--min-duration-off`は
残すが、本番の`GroupDiarizer`呼び出しはデフォルト値（0.3/0.5）のまま
据え置く。** Round 1の宿題だった「`min_silence=0.5`と同じconfusion改善を
ライブ非依存の経路で得る」は、この2値では達成できなかった —
理由はT3の診断で明らかになる通り、清書グループ内の短時間ターン自体が
そもそも少ないため。

### T2: overlap区間を除外したDER採点オプション

`der_breakdown()`に`skip_overlap`引数を追加し、
`pyannote.metrics.diarization.DiarizationErrorRate(collar=..., skip_overlap=...)`
にそのまま渡すようにした。`eval_diar.py`に`--skip-overlap`フラグを追加
(`scripts/eval_diar.py`)。**`--skip-overlap`は`--breakdown`の内訳
（Miss/FalseAlarm/Confusion、およびpyannote側DER値）だけに効く
オプションで、`simpleder`ベースの主指標（`DER=`として毎行先頭に出る値）
には影響しない** — 二つのDER実装を混同しないよう、意図的にそう設計した。

baseline設定（`remap_threshold=0.35`等、他はデフォルト）でoverlap区間を
除外した内訳:

| 会議 | Miss | FalseAlarm | Confusion | 内訳合計(≒overlap除外DER) |
|---|---|---|---|---|
| ES2011a | 9.9% | 4.2% | 5.7% | 19.8% |
| IS1008a | 2.0% | 3.6% | 0.6% | 6.2% |
| ES2004a | 9.7% | 4.6% | 1.7% | 16.0% |
| IS1009a | 3.6% | 5.4% | 9.8% | 18.8% |
| TS3003a | 7.7% | 6.7% | 0.5% | 14.9% |
| **平均** | | | | **15.1%** |

overlap込み（§8実測、baseline）の内訳合計は平均18.2%
（(21.6+6.8+23.2+23.7+15.9)/5、内訳は本セクションT1表参照）だったので、
overlapを除外すると内訳合計で**約3.1pt下がる**。会議別ではoverlap比率が
高いES2004a（15.1%）とIS1009a（12.8%）で下がり幅が大きい
（ES2004a: miss14.5→9.7、IS1009a: miss6.8→3.6/confusion12.4→9.8）。
overlap比率が低いIS1008a（2.4%）・TS3003a（2.1%）はほぼ動かない。
§1で挙げていた「overlapがmissの構造的下限」という見立てが、
overlap除外採点でも定量的に裏付けられた形。

**結論（T2）: `--skip-overlap`は採点オプションとして採用（本番挙動には
無関係、評価スクリプトの機能追加のみ）。overlap除外DER（内訳合計で
平均15.1%）は「hayamimiの1区間1話者という設計を所与としたときに
実際に狙える上限」であり、overlap込みのDER 14.1%との差分約3.1ptは、
overlap自体に手を入れない限り解消できない構造的な床であることが
確認された。**

### T3: TS3003aの未解明missの診断

Round 1の観察: TS3003aはoverlap比率がわずか2.1%（5会議中最小）にも
かかわらずmissが8.7%ある。他の会議はoverlap比率とmissがほぼ1:1で
連動するのに対し、TS3003aだけ「overlapで説明できないmiss」が残っている。
このmissの内訳を3カテゴリに分けて秒数で診断した（比較対象として
overlap比率が高い側のES2011aも同時に計測）。

**手法**: 生成した清書グループ（`group_segments()`と同じロジック）の
音声スパン単位で、(a) そもそもどの清書グループにも入らなかった
（＝ライブVADが一度も音声として検出しなかった）参照発話時間、
(b) グループには入ったが`GroupDiarizer`のraw出力に現れず、
`min_duration_on`を0.3→0.02まで下げると新たに現れる時間
（＝`min_duration_on`フィルタに直接起因する分だけを分離）、
(c) それ以外（グループには入り、`min_duration_on`を下げても回収できない
が、最終的な仮説には反映されなかった時間 — pyannote segmentationの
ターン境界誤差や、`eval_diar.py`側の0.3秒未満ターン破棄など、
「境界のずれ／清書内部の検出漏れ」の寄せ集め）に分類した。
（この診断スクリプト自体はcollarを掛けない単純な区間差分で計測しており、
collar付き・pyannote.metrics換算のmiss%（Round 1の8.7%等）とは
定義が異なるため厳密には一致しない。カテゴリ間の内訳比率を見る
相対比較として使う。）

| 会議 | 参照発話合計 | 実測miss(uncollared) | (a) VAD未検出 | (b) min_duration_on起因 | (c) 境界ズレ／清書内部その他 |
|---|---|---|---|---|---|
| TS3003a | 490.6s | 40.3s (8.2%) | 27.8s (5.7%) | 0.2s (0.0%) | 12.3s (2.5%) |
| ES2011a | 394.9s | 38.7s (9.8%) | 30.2s (7.6%) | 0.0s (0.0%) | 8.5s (2.1%) |

**結論（T3）: どちらの会議も`min_duration_on`起因のmissは実質ゼロ
（0.0〜0.2秒）** — T1で`min_duration_on=0.15`が全指標でbaselineと
完全一致した理由がここで裏付けられた。miss の大半（TS3003aで
5.7pt/8.2pt≒70%、ES2011aで7.6pt/9.8pt≒78%）は**(a) ライブSilero VADが
そもそも音声として検出せず、清書グループにすら入らなかった区間**に
起因する。残り（TS3003a 2.5pt、ES2011a 2.1pt）は(c)境界ズレ／清書内部の
検出漏れで、こちらはグループ後のpyannote segmentation側の問題であり
`min_duration_on`では触れない領域。

TS3003aのoverlap比率がわずか2.1%なのにmissが8.7%（Round 1の
collarベース実測）ある理由は、overlapではなく**ライブVAD側の検出漏れ
（カテゴリa）が主因**という診断結果になる。fix自体は今回のスコープ外
だが、Round 3の優先候補は「清書グループ内のオフライン診断パラメータ
（`min_duration_on`/`off`）」ではなく、**ライブSilero VADの感度側**
（`min_speech_duration`、VAD確率しきい値、または2段VAD構成そのもの）
であることが今回の診断で示された。

### 採用/不採用まとめ

| 施策 | 変更内容 | 平均DERへの効果 | 判定 |
|---|---|---|---|
| `min_duration_off=0.25` | 0.5→0.25 | ±0.0pt、IS1008a +0.7pt悪化 | **不採用**（悪化基準に抵触） |
| `min_duration_off=0.35` | 0.5→0.35 | -0.1pt改善 | **不採用**（改善0.3pt未満） |
| `min_duration_on=0.15` | 0.3→0.15 | 効果なし（全指標一致） | **不採用**（効果なし） |
| `--skip-overlap`採点オプション | 新規CLIフラグ | 主指標DERには非影響、内訳のみ | **採用**（評価スクリプト機能として） |

Round 1の宿題「`min_silence=0.5`と同じconfusion改善をライブ非依存の
経路で得る」は本ラウンドでは未達成のまま持ち越し。T3の診断結果から、
Round 3では清書側のオフラインパラメータではなく、ライブVADの感度
チューニング（本番の字幕確定レイテンシとのトレードオフを伴う）を
検討対象にすべきと判断する。

### 残課題

- ライブSilero VADの感度パラメータ（`min_speech_duration`、VAD確率閾値）
  のスイープは未実施。§1で見送った理由と同じく、感度を上げると
  誤検出（FA）が増える可能性があり、Miss改善とのトレードオフを
  T3の分析と同じ粒度（VAD検出漏れ vs 境界ズレ）で見る必要がある。
- T3の(c)「境界ズレ／清書内部その他」はTS3003a 12.3秒・ES2011a 8.5秒と
  カテゴリ(a)より小さいが無視できる量ではない。pyannote segmentationの
  ターン境界がRTTM参照とどれだけずれているか、秒単位でさらに分解する
  診断は未実施（今回はcollarなしの粗い差分までに留めた）。
- `--skip-overlap`はDER内訳のみに効くよう意図的に設計したが、
  「overlap区間を積極的に2話者ラベルで出す」機能自体（overlap detection）
  はhayamimiに存在しない。overlap除外DER 15.1% と overlap込みDER 14.1%
  の差分3.1ptを縮めるには、overlap detection自体の実装が必要になり、
  スコープが大きい（Round 3以降の検討事項）。

## 13. Round 3: ライブSilero VAD検出しきい値のスイープ（issue #11続き、§12 T3の申し送り）

### 背景

Round 2（§12）のT3診断で、TS3003a・ES2011aの実際のmissの70〜78%は
**ライブSilero VADが一度も音声として検出せず、清書グループにすら
入らなかった区間**（TS3003a 27.8s/490.6s、ES2011a 30.2s/394.9s）に
起因することが分かった。境界ズレ／清書内部その他は2〜2.5%程度、
`min_duration_on`起因は実質ゼロ。§12はこの結果から「Round 3では
清書側のオフラインパラメータではなく、ライブVADの感度チューニングを
検討対象にすべき」と申し送っていた。IS1009aの confusion 12.4%（5会議中
最大）は本ラウンドの対象外（VAD側のmiss改善に絞る）。

タスクは4つ: (T1) `scripts/realtime_transcribe.py`の`build_vad()`が
持つSilero VADの感度ノブの露出、(T2) スイープ、(T3) 採用候補が出た
場合のみのfaサニティゲート、(T4) 本セクションでのまとめ。

### T1: ライブVAD感度ノブの露出

`build_vad()`に`vad_threshold`引数（デフォルト0.5 = 現行本番動作と同じ）
を追加し、`sherpa_onnx.SileroVadModelConfig.threshold`にそのまま渡すよう
にした。`eval_diar.py`に`--vad-threshold`を、既存の`--min-silence`/
`--max-speech`と同じ配線パターンで追加した
(`scripts/realtime_transcribe.py`, `scripts/eval_diar.py`)。

もう一つ探すはずだった「speech-padding（onset/offset側の余白）ノブ」は、
インストール済み`sherpa_onnx`（1.13.6）の`SileroVadModelConfig`を実際に
`dir()`で確認したところ**存在しなかった**: 持っているフィールドは
`threshold`, `min_silence_duration`, `min_speech_duration`,
`max_speech_duration`, `window_size`, `neg_threshold`の6つのみで、
`speech_pad_ms`に相当するものはない。従って`--vad-pad`フラグは追加せず、
このバージョンでは`threshold`単独が唯一の実効ノブとなる。

### T2: スイープ

AMI 5会議、collar=0.25s、`--method refine_diarize --breakdown`、他は
デフォルト（`remap_threshold=0.35`, `sim_threshold=0.45`,
`diar_threshold=0.5`, `min_silence=0.35`）での実測。baseline行は
本ラウンドで再実測した`--vad-threshold 0.5`（Round 1到達点14.1%との
差は実行間ノイズ、後述）。

| vad-threshold | 平均DER | ES2011a | IS1008a | ES2004a | IS1009a | TS3003a |
|---|---|---|---|---|---|---|
| 0.50（baseline再実測） | 13.9% | 17.0% | 4.0% | 17.8% | 16.9% | 14.1% |
| 0.40 | 16.5% | 33.7%（**+16.7pt**） | 4.7%（+0.7pt） | 17.2%（-0.6pt） | 14.9%（-2.0pt） | 11.9%（-2.2pt） |
| 0.30 | 15.8% | 32.0%（**+15.0pt**） | 4.0%（±0.0pt） | 16.2%（-1.6pt） | 14.1%（-2.8pt） | 12.8%（-1.3pt） |
| 0.20 | 15.6% | 20.8%（+3.8pt） | 4.0%（±0.0pt） | 18.2%（+0.4pt） | 19.7%（+2.8pt） | 15.3%（+1.2pt） |

内訳（Miss/FalseAlarm/Confusion、参照総発話時間比、`--breakdown`実測）:

| vad-threshold | ES2011a | IS1008a | ES2004a | IS1009a | TS3003a | 平均miss / 平均fa / 平均confusion |
|---|---|---|---|---|---|---|
| 0.50 | miss11.3/fa3.5/conf6.1 | miss2.6/fa3.5/conf0.7 | miss14.5/fa3.7/conf5.0 | miss6.8/fa4.5/conf12.4 | miss8.7/fa6.5/conf0.7 | 8.8 / 4.3 / 5.0 |
| 0.40 | miss9.0/fa4.4/**conf25.2** | miss2.7/fa3.8/conf1.2 | miss12.2/fa6.0/conf5.5 | miss6.9/fa4.6/conf10.1 | miss6.2/fa6.5/conf0.9 | 7.4 / 5.1 / **8.6** |
| 0.30 | miss7.6/fa4.8/**conf24.2** | miss2.0/fa3.2/conf1.1 | miss10.6/fa6.2/conf5.3 | miss6.5/fa5.1/conf8.7 | miss4.3/fa8.0/conf2.1 | 6.2 / 5.5 / **8.3** |
| 0.20 | miss7.3/fa7.2/conf9.7 | miss2.0/fa3.2/conf1.1 | miss10.9/fa8.0/conf5.2 | miss6.4/fa8.1/conf10.5 | miss6.2/fa9.9/conf1.1 | 6.6 / **7.3** / 5.5 |

**miss自体はT3診断どおり下がる**（平均8.8%→7.4%→6.2%→6.6%、
`vad_threshold`を下げるほどライブVADが拾う音声が増えるという狙い通りの
挙動）。一方で**faも単調に増える**（平均4.3%→5.1%→5.5%→7.3%、想定の
トレードオフ）。ここまでは狙い通りだが、**想定していなかったのが
confusionの急増**: ES2011aのconfusionが0.40/0.30で6.1%→25.2%/24.2%と
4倍化し、平均DERを押し上げる主因になっている。0.20ではconfusionは
9.7%まで戻るが、代わりにfaが7.2%まで増える（ES2004a・IS1009a・
TS3003aのfaも軒並み8〜10%台に悪化）。

原因の見立て: しきい値を下げると清書グループの境界に短く低エネルギーな
音声区間が新たに混入し、`GroupDiarizer`（pyannote segmentation +
CAM++ + FastClustering）のクラスタリングに揺れが生じて、話者ラベルの
入れ替わり（confusion）が増える。miss改善分をconfusion／faの悪化が
上回り、**3つの`vad-threshold`値すべてで平均DERが悪化した**
（+1.7pt〜+2.6pt、採用基準の「0.3pt以上の改善」の逆方向）。特にES2011aは
0.40/0.30で+15〜17pt級の大幅悪化で、「どの会議も0.5pt超の悪化なし」
基準にも大きく抵触する。

**コンボ実行は行わなかった**: T2の設計は「個別で0.3pt以上の改善が
出た値があればコンボを回す」だったが、0.40/0.30/0.20のいずれも改善
どころか悪化したため、組み合わせる意味のある候補が無かった。padding
ノブ自体もT1の通り今回のsherpa_onnxには存在しないため、そもそも
組み合わせる相手がない。

baseline再実測の13.9%とRound 1到達点の14.1%の差（0.2pt）について:
`GroupDiarizer`のFastClustering・`SpeakerLabeler`のマッチングは実行間で
決定論的でない要素を含む（§8以降の既知の実行間ノイズと同水準）ため、
この程度の差は測定誤差として扱う。0.2pt程度のノイズに対して、
0.40/0.30/0.20の悪化幅（+1.7〜+2.6pt）は十分大きく、ノイズでは
説明できない。

### T3: faサニティゲート

**実施しなかった**: タスク設計上、T3は「採用基準（平均0.3pt以上改善）を
クリアした設定が出た場合のみ」実施する条件付きタスクだった。T2の結果、
`vad-threshold`0.40/0.30/0.20のいずれも平均DERを悪化させており、
クリアした設定が一つも無かったため、T3のfa詳細検証・本番デフォルト化の
検討自体が対象外になった。

### 採用/不採用まとめ

| 施策 | 変更内容 | 平均DERへの効果 | 判定 |
|---|---|---|---|
| `vad_threshold`の露出（T1） | `build_vad()`/`--vad-threshold`追加 | 機能追加のみ、デフォルト0.5で本番非影響 | **採用**（評価スクリプト機能として） |
| `vad_threshold=0.40` | 0.5→0.40 | +2.6pt悪化、ES2011a +16.7pt | **不採用**（悪化基準に大きく抵触） |
| `vad_threshold=0.30` | 0.5→0.30 | +1.9pt悪化、ES2011a +15.0pt | **不採用**（悪化基準に大きく抵触） |
| `vad_threshold=0.20` | 0.5→0.20 | +1.7pt悪化、fa平均7.3%まで増加 | **不採用**（改善なし、fa/confusion双方悪化） |
| padding（speech_pad_ms相当）ノブ | 該当なし | 測定不可 | **対象外**（sherpa_onnx 1.13.6に存在しない） |

本番の`build_vad()`呼び出しはデフォルト値（`vad_threshold=0.5`）のまま
据え置く。ライブレイテンシへの影響について: `vad_threshold`自体は
Silero推論1回あたりの確率カットオフであり、`min_silence_duration`の
ような終端待ち時間には効かないため**しきい値を変えること自体は
latency-neutral**（このスイープに関する限りレイテンシ論点は無い —
今回不採用になったのはDER側の悪化のみが理由）。ただしpaddingノブは
今回のsherpa_onnxに存在しないため、「paddingがtail latencyを増やすか」
の議論自体が本ラウンドでは発生しなかった。

### 残課題

- **T3診断（§12）の「ライブVAD検出漏れ」問題自体は未解決のまま**。
  `threshold`単純な引き下げは今回confusion急増という副作用で頭打ちに
  なった。次に検討する余地があるとすれば`min_speech_duration`
  （現在0.25s固定）や、Silero単体ではなく2段VAD構成（低しきい値VADで
  候補区間を広く取り、別の軽量分類器でfaを間引く）だが、いずれも
  スコープが今回のT1〜T4より大きい。
- confusion急増のメカニズム（境界に混入した短い低エネルギー区間が
  `GroupDiarizer`のクラスタリングをどう揺らすか）は仮説止まりで、
  実際にどのセグメントがconfusionの原因になっているかの秒単位の
  切り分けは行っていない。`vad_threshold=0.40`のES2011aを対象にした
  詳細診断は次ラウンド以降の候補。
- IS1009aのconfusion 12.4%（§0の既知の最大要因）は今回のスコープ外の
  まま未着手。

## 14. Round 4: confusionの内訳診断（LOCAL/REMAP/FAST-PATH）、REMAP対策の実験、overlap対応可否調査

### 背景

§0でconfusionが「今もっとも対応の余地がある」最大要因と確認済み
（IS1009a 12.4%、ES2011a ~6-7%、他は≤5%）。だがこれまでのラウンドは
confusionを一枚岩の数字としてしか見ておらず、清書パイプラインの
どの段階（局所クラスタリングそのもの／グローバル話者へのremap／
そもそも再分離を経ないfast-pathの残差）が原因なのかを切り分けたことが
なかった。本ラウンドはこの内訳診断（T1）、内訳の支配的要因への対策
実験（T2）、overlap発話への対応可否のコード調査（T3、実装なし）の3本。

### T1: confusion内訳診断

**定義**（3カテゴリ、時間の合計はconfusion全体と一致するよう相互排他に設計）:

- **LOCAL誤り**: そのconfusion区間を生んだ清書グループについて、
  グループの時間範囲に限定した「ローカルクラスタid→参照話者」の
  最適割当（Hungarian法、重複時間最大化）を計算し、それでもなお
  参照と食い違う場合。完璧なremapがあってもこの誤りは直らない。
- **REMAP誤り**: 上記のローカル最適割当は参照と一致するのに、
  `speaker_id.SpeakerLabeler.match_embedding()`（remap_threshold経由）
  が選んだグローバルS{n}が食い違う場合。ローカルクラスタリング自体は
  正しく、remapステップだけが誤っている。
- **FAST-PATH残差**: そのhypスパンの最終ラベルが、実際の局所再分離＋
  remapを経ておらず、グループ内ローカルクラスタが1種類以下だった
  （＝多数決fast-pathラベルへのフォールバック）ことに由来する場合。

診断は`scripts/eval_diar.py`の`generate_diarize_hypothesis()`と全く
同じアルゴリズム・同じ定数を使う独自インストルメント版（スクラッチ、
コミットせず）で、各hypスパンに「どのグループ由来か」「fast-path
フォールバックか」「ローカルクラスタid」を付与記録した。confusion
区間そのものの特定（どの時間・どのラベルがconfusionか、collar=0.25s
込み）は`pyannote.metrics.errors.identification.IdentificationErrorAnalysis`
の`.difference()`を使用 — これは`eval_diar.py`の`der_breakdown()`が使う
`DiarizationErrorRate`と同じcollar/マッチング機構だが、**内部でhyp/ref
ラベル空間のHungarian最適置換（`DiarizationErrorRate.optimal_mapping()`）
を先に適用しないと使えない**ことが実装中に判明した（最初の実装では
このマッピングを忘れ、ES2011aのconfusionが354.3s＝正しい値27.0sの
13倍という明らかにおかしい値が出た。原因はhypのS{n}ラベルと参照の
A/B/C/DラベルをIdentificationErrorAnalysisが素の文字列比較していた
ため）。修正後は`DiarizationErrorRate`が返すconfusion秒数と
`IdentificationErrorAnalysis`側の合計が両meetingで完全一致（ES2011a
27.0s=27.0s、IS1009a 57.3s=57.3s）することを確認済み — 以下の内訳は
この検証済み手法による。

| meeting | confusion合計 | LOCAL | REMAP | FAST-PATH残差 |
|---|---|---|---|---|
| ES2011a | 27.0s | 6.4s（23.8%） | 13.2s（**49.0%**） | 7.4s（27.3%） |
| IS1009a | 57.3s | 22.1s（38.6%） | 35.1s（**61.3%**） | 0.04s（0.1%） |

（confusion合計はDiarizationErrorRate実測値。§0で参照した5.9%/12.4%
という比率とは実行間の非決定性ノイズで数秒ズレる — §13で確認済みの
±0.2pt級のノイズと同水準、GroupDiarizerのFastClustering・remap
マッチングは実行間で完全に決定論的ではない。)

**両meetingともREMAP誤りが最大カテゴリ**（ES2011a 49%、IS1009a 61%）。
LOCAL誤りは2位（24-39%）で、こちらは「そもそもgroup内の局所クラスタ
リングが参照と食い違う」という、remapを直しても解決しない根本的な
限界に近い。FAST-PATH残差はES2011aで27%（fallbackグループが17/32と
多い）、IS1009aではほぼゼロ（fallbackグループ4/24のみで、たまたま
それらの区間はconfusionを起こしていない）。

**T2の方針決定**: タスク設計どおりREMAP誤りが両meetingで支配的
だったため、REMAP対策の実験に進む。

### T2: REMAP対策実験 — `min_remap_update_s`（クラスタ長でcentroid更新を制限）

T1の候補2つ（duration-weighted remap／best-of-group assignment）から
前者を選択。理由: `match_embedding()`のcentroid更新は既に単純な累積平均
（`(centroids[i]*n + emb) / (n+1)`、`speaker_id.py`）で、極端に短い
ローカルクラスタ（数百msなど）の埋め込みはノイズが大きく、これが
centroidを実際の話者から動かしてしまい後続のremapを狂わせる、という
仮説が「クラスタ<1sはcentroid更新から除外する」という具体的な対策に
直結しやすかったため。

**実装**（`scripts/eval_diar.py`の`generate_diarize_hypothesis()`に
`min_remap_update_s`引数、`--min-remap-update-s`CLIフラグ。本番側
`scripts/realtime_transcribe.py`の`Refiner`にも同じ挙動を
`min_remap_update_s`コンストラクタ引数／`--speaker-min-remap-update-s`
CLIフラグとして追加し、評価専用にしない）: ローカルクラスタの合計
発話長がこの秒数未満なら、`match_embedding(update=True)`ではなく
`match_embedding(update=False)`（読み取り専用probe）を使う —
既存centroidにマッチすればそのラベルを使う（centroidの平均は動かさ
ない）が、マッチしなければ新規centroidを開かず、グループの
fast-path多数決ラベルにフォールバックする。デフォルト0.0は完全な
no-op（既存の全ラウンドの挙動と同一）。

**スイープ**（AMI 5会議、collar=0.25s、`--method refine_diarize`、
他はデフォルト）:

| min-remap-update-s | 平均DER | ES2011a | IS1008a | ES2004a | IS1009a | TS3003a |
|---|---|---|---|---|---|---|
| 0.0（baseline再実測） | 13.900% | 16.82% | 3.98% | 17.76% | 16.88% | 14.06% |
| 0.5 | 13.885%（-0.015pt） | 16.82% | 3.98% | 17.70% | 16.88% | 14.05% |
| 1.0 | 13.884%（-0.016pt） | 16.82% | 3.98% | 17.70% | 16.87% | 14.05% |
| 1.5 | 13.838%（-0.062pt） | 16.82% | 3.98% | 17.70% | 16.87% | 13.82% |
| 2.0 | 14.0%（**+0.1pt悪化**） | **17.7%**（+0.9pt） | 4.0% | 17.7% | 16.9% | 13.8% |

0.5〜1.5では改善方向だが**採用基準の0.3pt改善に遠く届かない**
（最良の1.5でも-0.062pt）。2.0まで上げるとES2011aのconfusionが
5.9%→6.8%へ悪化して回帰（+0.9pt、「0.5pt超悪化なし」基準に抵触）し、
平均も逆に悪化する。

**なぜ効かなかったかの検証**: 1.5（一番マシだった値）でT1と同じ
インストルメント診断を再実行し、REMAP誤りの秒数自体が減っているか
確認した。

| meeting | min_remap_update_s | confusion合計 | LOCAL | REMAP | FAST-PATH残差 |
|---|---|---|---|---|---|
| ES2011a | 0.0 | 27.0s | 6.4s | 13.2s | 7.4s |
| ES2011a | 1.5 | 24.3s（-2.7s） | 6.4s（±0） | 13.2s（**±0**） | 4.6s（-2.8s） |
| IS1009a | 0.0 | 57.3s | 22.1s | 35.1s | 0.04s |
| IS1009a | 1.5 | 57.3s（±0） | 21.3s | 35.9s（**+0.8s**） | 0.04s（±0） |

ES2011aの改善（confusion 27.0s→24.3s）は**REMAP秒数が全く変わらず**
（13.2s→13.2s）、FAST-PATH残差だけが減ったこと（7.4s→4.6s）で起きて
いる — この施策が意図した「remap誤りを直す」効果は測定上ゼロで、
副次的にfallback経路の挙動がわずかに変わった結果らしい。IS1009aに
至ってはREMAP秒数がむしろ微増（35.1s→35.9s）している。

**結論（仮説の反証）**: 「短いクラスタのノイズがcentroidを汚して
remapを誤らせる」という仮説は、この施策では裏付けられなかった。
`match_embedding(update=False)`で更新を止めても、そもそものマッチング
判定（centroidとの類似度がremap_threshold=0.35を超えるか）自体は
変わらないため、centroid更新の有無が結果に反映されるのは「その
centroidが後で別のクラスタのマッチングに使われた場合」のみで、
今回の2 meeting・この時間窓では効果が測定できるほど蓄積しなかった
とみられる。REMAP誤りの真因はcentroidのドリフトではなく、**その場の
埋め込み自体がどのグローバル話者にも十分近くない**（threshold=0.35を
割り込む、または僅差の誤答が生じる）ことにありそうだが、これを
確定させる追加診断は今回のスコープ外。

### 採用/不採用まとめ（T2）

| 施策 | 変更内容 | 平均DERへの効果 | 判定 |
|---|---|---|---|
| `min_remap_update_s`の露出 | `Refiner`/`generate_diarize_hypothesis()`/両CLIフラグ追加 | 機能追加のみ、デフォルト0.0で本番非影響 | **採用**（評価・本番双方の機能として） |
| `min_remap_update_s=0.5/1.0/1.5` | centroid更新をクラスタ長でゲート | 最良0.062pt改善（基準0.3pt未達） | **不採用** |
| `min_remap_update_s=2.0` | 同上、より広いゲート | +0.1pt悪化、ES2011a+0.9pt回帰 | **不採用**（悪化基準に抵触） |

ライブレイテンシへの影響: この施策はremap呼び出し自体の回数もタイミング
も変えず（`match_embedding()`の`update`引数を切り替えるだけ）、レイテンシ
中立。今回不採用になった理由はDER側の効果不足のみ。

### T3: overlap発話への対応可否（調査のみ、実装なし）

**問い**: 構造的floor（§で確認済み、~3.1pt）はhayamimiが常に1区間=1話者
のシングルラベル出力である前提から来ている。インストール済み
`sherpa_onnx`（1.13.6）の`OfflineSpeakerDiarization`は、そもそも
overlap（重複発話）区間を扱う余地があるのか？

**Python API層の確認**: `sherpa_onnx.OfflineSpeakerDiarizationSegment`
は`speaker`（単一int）・`start`・`end`・`text`・`duration`のみを持ち、
複数話者を保持するフィールドがない。`OfflineSpeakerDiarizationResult`
は`sort_by_start_time()`/`sort_by_speaker()`でこの単一話者セグメントの
列を並べ替えるだけ。`FastClusteringConfig`にoverlap関連の設定項目は
無い。Python側に公開されているのは`OfflineSpeakerSegmentationModelConfig`
と`OfflineSpeakerSegmentationPyannoteModelConfig`（＝モデルファイルの
パス設定のみ）で、セグメンテーションモデルの生の出力（フレーム単位の
確率）にアクセスするAPIは存在しない。

**C++実装層の確認**（`gh api`で`k2-fsa/sherpa-onnx`の
`sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h`を直接取得
して読んだ）: pyannote segmentation-3.0モデルは**powerset分類**
（`InitPowersetMapping()`、pyannote-audioの`utils/powerset.py`を参照する
コメント付き）を出力しており、`ToMultiLabel()`がpowersetクラスを
「フレームごとの話者0/1マトリクス」にデコードする。このデコード結果は
**構造上、1フレームに複数話者が同時にactiveになり得る**（powersetの
クラスにはペア・トリプルの組み合わせが含まれる — pyannote本来の
overlap対応の仕組みそのもの）。しかし`GetChunkSpeakerSampleIndexes()`
はこの結果に対して真っ先に`ExcludeOverlap()`を呼んでおり、その
コメントは明示的に「If there are multiple speakers at a frame, then
this frame is excluded.」— **overlapフレームは話者セグメント抽出前に
意図的に捨てられている**。つまりモデル自体はoverlap対応の情報を
出力しているが、sherpa-onnxの公開パイプラインがそれをクラスタリング
に渡す前段で破棄している。

**判定: possible-with-work**（upstream変更なしでは不可能、ではない）。
根拠:
- overlap対応に必要な情報（powersetデコード後のマルチラベル）は
  モデル出力に存在する。アルゴリズム（pyannote-audioの
  `utils/powerset.py`）も公開されている。
- ただしPython APIからこの生出力に到達する手段が無い
  （`OfflineSpeakerSegmentationModelConfig`はパス設定のみ）ため、
  sherpa_onnxの`OfflineSpeakerDiarization`を経由する限りoverlapは
  常にゼロになる。
- 回避策は「sherpa_onnxをバイパスし、`model.onnx`
  （`models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`、
  既にダウンロード済み）を自前でonnxruntime推論し、powersetデコード
  ロジックを自前実装する」こと。モデルファイルもアルゴリズムも既知・
  入手可能なため技術的には可能だが、新規のセグメンテーション専用
  推論パス（チャンク分割・受容野からのサンプルインデックス復元・
  powersetデコード・複数話者フレームのマージ）を丸ごと書く規模の
  作業で、今回のT1〜T3の範囲を大きく超える。upstream（k2-fsa/
  sherpa-onnx）にoverlap保持オプションを追加してもらう方が現実的な
  可能性もあるが、今回は問い合わせ・PR提案までは行っていない。
- 従って次ラウンド以降でこのfloorを攻めるなら、この自前実装
  （またはupstreamへの機能要望）が前提になる、という結論。今回は
  調査のみで実装はしない。

### 残課題

- REMAP誤りの真因（centroidドリフトではなさそうと分かったのみ）は
  未確定。次の候補は「remap_threshold=0.35自体を僅差で割り込む/僅差で
  跨ぐケースの分布を見る」診断（閾値そのものの再チューニングではなく、
  T1と同じ手法でREMAP誤りをさらに「閾値未満で新規centroidを開いた」
  ケースと「既存centroidに誤ってマッチした」ケースに割るなど）。
- LOCAL誤り（24-39%、2番目に大きいカテゴリ）は今回T2の対象外のまま。
  `num_clusters`ヒント（グローバル台帳の確信済み話者数を
  `FastClusteringConfig.num_clusters`に渡す）はT1のタスク設計が
  「LOCALが支配的な場合の代替案」として用意していたが、今回は
  REMAPが支配的だったため試していない。次ラウンドでLOCAL側を狙う
  ならこれが候補として残っている。
- T3で判定した「possible-with-work」の自前実装（またはupstream機能
  要望）は着手していない。構造的floor（~3.1pt）を動かす唯一の既知
  経路だが、スコープの大きい別イテレーションが必要。

## 15. Round 5: REMAP誤りへの2つの対策実験（joint remap／provisional centroid除外）— いずれも不採用

### 背景

§14（Round 4）でconfusionの内訳診断を行い、両meetingともREMAP誤り
（`SpeakerLabeler.match_embedding()`がグループ内のローカル話者クラスタを
セッション全体のグローバルS{n}へ誤って割り当てるケース）が支配的
（ES2011a 49%、IS1009a 61%）と確認済み。§14 T2の`min_remap_update_s`
（centroid更新をクラスタ長でゲート）は「短いクラスタのノイズがcentroidを
汚す」という仮説を検証したが、REMAP誤りの絶対秒数は全く動かず反証された
（§14 T2の結論）。本ラウンドはコード自体を読み直して見つけた別の仮説を
2本試す。

**メカニズム（コードで確認済み）**: `realtime_transcribe.Refiner._emit_turns()`
は清書グループ内の各ローカルクラスタを`SpeakerLabeler.match_embedding()`
経由で**独立に**（貪欲最近傍、閾値0.35）グローバルS{n}へマッチさせる。
同じグループ内の2つの異なるローカルクラスタが同じグローバル話者へ
マッチするのを妨げる仕組みが何もない — 実在する2人の話者が1人に
潰れれば、それはDER上のconfusionとして現れる。

### T1: 制約付き同時remap（Hungarian割当）

**仮説**: 上記の「2つのローカルクラスタが同じグローバル話者に潰れる」
ケースをそもそも起こさせなければ、その分のREMAP誤りが減るはず。

**実装**（`scripts/speaker_id.py`に`SpeakerLabeler.match_embeddings_joint()`
を追加。デフォルトoffのフラグ経由でのみ呼ばれる —
`scripts/realtime_transcribe.py`の`Refiner`に`joint_remap`コンストラクタ
引数／`--speaker-joint-remap`CLIフラグ、`scripts/eval_diar.py`の
`generate_diarize_hypothesis()`に同名引数／`--joint-remap`CLIフラグ）:
グループ内の全ローカルクラスタ×生存中の全グローバルcentroidの
コサイン類似度行列を一度に作り、`scipy.optimize.linear_sum_assignment`
（Hungarian法）で**合計類似度を最大化する1対1割当**を解く。既存の
remap_threshold（0.35）は割当後の適格性フィルタとしてそのまま残す —
最適割当の類似度がそれでも閾値未満のクラスタは「適格な候補なし」として
従来の独立`match_embedding()`呼び出しにフォールバック（新規speakerを開く、
または非制約の単独最近傍にマッチ）。単一ローカルクラスタのグループは
Hungarianを経由せず従来パスと完全に同一（制約すべき相手がいないため）。
`min_remap_update_s`（§14 T2）がゲートする短いクラスタは、そちらの
read-onlyプローブが先に処理し、Hungarian対象からは除外する。

**測定**（AMI 5会議、collar=0.25s、`--method refine_diarize --breakdown`）:

| meeting | baseline DER | joint-remap DER | Δ | baseline confusion | joint-remap confusion |
|---|---|---|---|---|---|
| ES2011a | 17.0% | 20.3% | **+3.3pt** | 6.1% | 8.2% |
| IS1008a | 4.0% | 5.2% | **+1.2pt** | 0.7% | 2.0% |
| ES2004a | 17.8% | 20.0% | **+2.2pt** | 5.0% | 6.7% |
| IS1009a | 16.9% | 26.6% | **+9.7pt** | 12.4% | 19.4% |
| TS3003a | 14.1% | 15.8% | **+1.7pt** | 0.7% | 2.2% |
| **平均** | **13.9%** | **17.6%** | **+3.7pt** | — | — |

**5会議全てで悪化**、しかも§13で確認済みの run-to-run ノイズ床
（±0.2pt級）を遥かに超える規模。採用基準（0.3pt以上の改善、悪化
0.5pt以内）どころか、逆方向に大きく外れているため即座に不採用と判断、
T2の全体sanity（pytest等）は実施しなかった（採用基準未達の時点で
スコープ外 — タスク定義通り）。

**なぜ悪化したか（§14と同じ手法の内部診断インストルメントで検証、
ES2011a・IS1009a）**:

| meeting | 施策 | confusion合計 | LOCAL | REMAP | FAST-PATH残差 |
|---|---|---|---|---|---|
| ES2011a | baseline | 23.5s | 8.6s（36.6%） | 11.0s（47.0%） | 3.8s（16.4%） |
| ES2011a | joint-remap | 36.3s | 21.6s（**59.6%**） | 7.3s（20.1%） | 7.4s（20.3%） |
| IS1009a | baseline | 57.3s | 22.1s（38.5%） | 35.2s（61.4%） | 0.0s |
| IS1009a | joint-remap | 89.8s | 38.0s（42.3%） | 51.7s（57.6%） | 0.0s |

ES2011aではREMAP誤りの絶対秒数自体は確かに減った（11.0s→7.3s、狙い通り）
が、LOCAL誤りがそれ以上に増えた（8.6s→21.6s）。IS1009aに至っては
REMAP・LOCAL両方が悪化した（REMAP 35.2s→51.7s、LOCAL 22.1s→38.0s）。

原因の仮説（コード上の性質から）: LOCAL誤りの定義は「グループの時間
範囲内でローカルクラスタ→参照話者の最適割当（Hungarian、重複時間
最大化）を計算しても、なお参照と食い違う」ケース — つまり**ローカル
diarizerの局所クラスタリング自体が実話者数より過分割している**場合、
本来は同一話者である2つのローカルクラスタを、joint remapは制約により
**強制的に別々のグローバル話者へ割り当てる**。これは「2つの異なる
ローカルクラスタは異なる実話者である」という前提（T1のタスク設計の
根拠）が、ローカルクラスタリングが完璧ではない現実の下では成り立たない
ことを意味する。§0以来「LOCAL誤りは局所クラスタリングそのものの限界」
と位置付けてきたが、本実験はその限界が単独では良性でも、joint制約と
組み合わさると新たな誤りを能動的に生み出す（過分割された同一話者の
片方を無理やり別人にする）ことを示した。

（run間ノイズについての注記: `GroupDiarizer`のローカルクラスタリングは
§14で述べた通り完全に決定論的ではないため、上表の内部診断（1回実行）
はLOCAL/REMAPの厳密な内訳にノイズを含む。ただしフルスイープのDER自体
（5会議全て、平均+3.7pt悪化）はそのノイズ床を遥かに超える規模なので、
「T1は悪化する」という結論自体はノイズに左右されない。）

**判定: 不採用**。`--joint-remap`/`--speaker-joint-remap`フラグは
コードとしては残す（デフォルトoff、機能追加のみで本番非影響）が、
有効化は推奨しない。

### T3: provisional centroidをremap候補から除外

T1が失敗し、かつIS1009aのconfusionが基準の8%を大きく超えたまま
（12.4%）だったため、タスク定義どおりT3に進んだ。

**仮説**: `speaker_id.PROVISIONAL_CONFIRM_HITS`未満（＝一度もマッチされて
いない、開いたばかりの一発centroid）がremapの最近傍探索に混ざっている
ことで、本来別の（既に確定した）話者にマッチすべきクラスタの類似度を
「奪って」誤ったマッチを起こしているのではないか。

**実装**（`scripts/speaker_id.py`の`SpeakerLabeler.match_embedding()`に
`exclude_provisional`引数を追加。Trueの場合、`self._counts[i] <
PROVISIONAL_CONFIRM_HITS`のcentroidを最近傍探索から完全に除外する
——マッチしても新規オープンしてもよいが、それ以外のcentroidの
邪魔をさせない。remap呼び出し（`Refiner._emit_turns()`の独立/short-id
両方、`eval_diar.py`の同等ループ）にのみ`exclude_provisional_remap`
という新規フラグ経由で渡す — fast path（`label()`）には決して渡さない
＝centroidが最初の確認ヒットを得る経路そのものを塞がないようにした。
`--exclude-provisional-remap`/`--speaker-exclude-provisional-remap`
CLIフラグ、デフォルトoff）。

**測定**（同条件）:

| meeting | baseline DER | exclude-provisional DER | Δ | baseline confusion | exclude-provisional confusion |
|---|---|---|---|---|---|
| ES2011a | 17.0% | 17.7% | +0.7pt | 6.1% | 6.8% |
| IS1008a | 4.0% | 4.0% | ±0 | 0.7% | 0.7% |
| ES2004a | 17.8% | 17.6% | -0.2pt | 5.0% | 5.0% |
| IS1009a | 16.9% | 20.5% | **+3.6pt** | 12.4% | 15.4% |
| TS3003a | 14.1% | 14.2% | +0.1pt | 0.7% | 0.9% |
| **平均** | **13.9%** | **14.8%** | **+0.9pt** | — | — |

T1ほど壊滅的ではないが、平均DERは改善どころか悪化（+0.9pt）、
IS1009aは単独で+3.6ptの明確な回帰。採用基準（0.3pt以上の改善）を
満たさないため不採用。

**内部診断**（同手法）:

| meeting | 施策 | confusion合計 | LOCAL | REMAP | FAST-PATH残差 |
|---|---|---|---|---|---|
| ES2011a | baseline | 23.5s | 8.6s | 11.0s | 3.8s |
| ES2011a | exclude-provisional | 27.0s | 8.6s（±0） | 11.0s（**±0**） | 7.4s（+3.6s） |
| IS1009a | baseline | 57.3s | 22.1s | 35.2s | 0.0s |
| IS1009a | exclude-provisional | 71.5s | 21.4s（-0.7s） | 50.1s（**+14.9s、悪化**） | 0.0s |

ES2011aはREMAP・LOCALとも全く変化がなく（remapのやり直しが発生する
ような一発centroidがこのmeetingでは元々remap候補としてほぼ選ばれて
いなかったと見られる）、増分は全てFAST-PATH残差（clustering非決定性の
ノイズと見られる、fallbackグループはそもそもremapを経由しない）。
IS1009aはREMAP誤りが**明確に悪化**（35.2s→50.1s）— 「一発centroidを
除外すれば正しい確定済み話者にマッチするはず」という仮説とは逆に、
除外によって**本来は正しいはずの一発centroid自身へのマッチ**（今
出現した話者が、たまたま直前に一度だけ登場した本人と同一で、
まだ2回目のヒットに至っていないだけのケース）が塞がれ、代わりに
別の（誤った）確定済みcentroidへ吸着されたか、無用に新規centroidを
開いたと考えられる。「provisional＝ノイズが多い一発の誤りやすい
centroid」という前提が、IS1009aでは「provisional＝正しいが、
たまたまだ1回しか出現していないだけの新規話者」というケースを
無視できないほど含んでいたということ。

**判定: 不採用**。`--exclude-provisional-remap`/
`--speaker-exclude-provisional-remap`フラグはコードとしては残す
（デフォルトoff、機能追加のみで本番非影響）が、有効化は推奨しない。

### 採用/不採用まとめ

| 施策 | 変更内容 | 平均DERへの効果 | 判定 |
|---|---|---|---|
| `match_embeddings_joint()`の追加、`--joint-remap`等 | 機能追加のみ、デフォルトoffで本番非影響 | — | **採用**（評価・本番双方の機能として） |
| `--joint-remap`を有効化 | グループ内remapを制約付きHungarian割当に | +3.7pt悪化、5会議全て悪化 | **不採用** |
| `match_embedding()`の`exclude_provisional`引数、`--exclude-provisional-remap`等 | 機能追加のみ、デフォルトoffで本番非影響 | — | **採用**（評価・本番双方の機能として） |
| `--exclude-provisional-remap`を有効化 | 未確認centroidをremap候補から除外 | +0.9pt悪化（IS1009a+3.6pt） | **不採用** |

ライブレイテンシへの影響: 両施策ともremap呼び出しの回数・タイミングは
変えない（`match_embeddings_joint()`は同じグループ内の呼び出しを1回に
まとめる分、むしろ呼び出し回数はわずかに減る）ので、デフォルトoffの
現状はもちろん、仮に有効化してもレイテンシへの影響はない。今回不採用
になった理由はいずれもDER側の効果不足（というより悪化）のみ。

### 残課題

- REMAP誤りの真因は§14に続き依然未確定。本ラウンドで判明したのは
  「REMAP誤りとLOCAL誤りは独立ではなく、remap側の制約を強めると
  LOCAL側に誤りが押し出される（少なくともES2011aでは）」という
  トレードオフの存在。今後この方向を攻めるなら、remap側だけでなく
  ローカルクラスタリングの過分割自体（§14残課題に記載済みの
  `num_clusters`ヒントなど）に踏み込む必要がありそうだが、未着手。
- provisional centroidの「ノイズが多い一発」と「正しいが未確認の
  新規話者」を区別する情報は現状`_counts`（ヒット回数）しかない。
  T3の反例（IS1009a）は、ヒット回数だけでは両者を判別できないことを
  示唆しており、次の一手があるとすれば埋め込み自体の質（分散・
  信頼度スコアなど、sherpa-onnxの埋め込み抽出器からは現状取得
  できない）のような別シグナルが必要になりそうだが、今回のスコープ外。

## 16. Round 6: overlap対応diarizationのプロトタイプ実測（powerset直接デコード）— 上限は測れたが本番投入は保留

### 背景

§6以降のDERは全て、ある構造的な床の上に乗っている。hayamimiの仮説は
**ある瞬間に必ず1人の話者しか出さない**ため、参照側で2人以上が同時に
喋っている区間は、どれだけ話者同定が完璧でも取りこぼす。この「取りこぼしが
確定している時間」を参照発話時間（話者ごとの発話長の総和、
pyannote.metricsのmiss/fa/confusionの分母と同じ取り方）で割った値が
overlap floorで、AMI 5会議では以下の通り。

| meeting | overlap floor（構造的に確定するmiss） |
|---|---|
| ES2004a | 14.2% |
| IS1009a | 12.7% |
| ES2011a | 9.7% |
| IS1008a | 2.4% |
| TS3003a | 2.1% |

§14（Round 4）のT3調査で、この床をsherpa-onnx側から崩すのは不可能と
確認済み。`OfflineSpeakerDiarization`が回しているpyannote
segmentation-3.0のpowerset出力は最大2人同時を表現できるが、C++実装の
`ExcludeOverlap()`が該当フレームを捨てており、生の多ラベル事後確率を
取り出すAPIが存在しない。そのときの結論は「segmentation ONNXを自前で
回せば可能」だった。

Round 1〜5は割当・閾値側のノブを一通り試し尽くして全て不採用（§12〜§15）。
本ラウンドは方針を変え、**その床のうち実際にどれだけ回収できるのかを
測るためだけのプロトタイプ**を書いた。本番配線は一切行っていない。

### 方式

`scripts/eval_diar_overlap.py`（新規、eval専用。本番経路からは誰も呼ばない）。
清書グループ／remapのパイプラインに手を入れたものではなく、**ファイル全体を
一括処理する別物のdiarizer**として一から書いてある。測りたいのが
「今のパイプラインをどう直すか」ではなく「overlap込みでどこまで行けるか」
だからである。

1. 会議音声を10秒窓（モデルの`window_size`=160000、メタデータから読んで
   assertしている）でスライドさせ、hop 5秒（窓の50%重なり）で
   `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`を
   onnxruntimeで直接実行する。出力は(フレーム, 7)のlog-softmax、
   フレーム間隔は`receptive_field_shift`=270サンプル=16.875ms。
2. 7クラスのargmaxで**最尤の「同時発話者の集合」**を取る。クラス順は
   pyannoteのpowerset実装どおり `[(), (0,), (1,), (2,), (0,1), (0,2), (1,2)]`。
   ペアクラスが選ばれたフレームは2人が同時にアクティブになる —
   `ExcludeOverlap()`が捨てていたのが、まさにこの情報。
3. 窓内ローカル話者ごとにCAM++埋め込みを1つ取る
   （`speaker_id.SpeakerLabeler.embed()`、本番と同じ抽出器）。ただし
   **その話者が単独で喋っているフレームだけ**を連結して埋め込む。2人分の
   混合音声から取ったCAM++ベクトルはどちらの話者にも寄らないため、
   ここを手抜きすると後段のクラスタリングが崩れる。
4. 窓をまたいだ話者の同一性はscipyの階層クラスタリング（コサイン距離、
   average linkage）で解く。**同じ窓で同時に検出された2人は定義上別人**なので、
   その組の距離を到達不能な値に固定するcannot-link制約を入れてある。
5. 重なり合う窓の投票をフレーム単位で多数決し、グローバル話者ごとの
   活動区間へ落とす。min_duration_off（短い無音の橋渡し）→
   min_duration_on（短いターンの削除）はsherpa-onnxの同名ノブと同じ順序。
6. **重なりを持ったままの**仮説をpyannote.metricsで採点（collar 0.25s、
   overlap込み、参照は§6以来と同一）。

**クラスタリングの2段化**は測定の途中で必要になった設計変更である。
最初は全埋め込みを一度にクラスタリングしていたが、閾値0.7で14話者・
0.9で2話者という崖ができ、その間に使える切り口が存在しなかった
（ES2004a、しきい値0.5では36話者）。原因は、単独発話が1秒に満たない
窓ローカル話者の埋め込みがほぼノイズで、本来離れているはずの話者同士を
橋渡ししてしまうこと。そこで**単独発話1.5秒以上の埋め込みだけがマージに
参加し、残りは出来上がったcentroidへ最近傍で割り当てる**2段構成に変えた
（`--reliable-s`）。これで崖が消え、閾値0.45〜0.65が連続的に振る舞うようになった。

**採点の注意**: `eval_diar.py`が見出しに出しているDERはsimpleder、
`--breakdown`のmiss/fa/confusionはpyannote.metricsで、**両者は同じ仮説に
対して数ポイント違う数字を出す**（分母の取り方が違う。simplederは参照発話の
和集合、pyannoteは話者ごとの発話長の総和）。§6以降ずっと引用してきた
「13.9%」はsimpleder側の数字である。混同を避けるため、本節の表は全て
**pyannote / simpleder** の順で併記する。比較対象のbaselineも本ラウンドで
測り直した。

### 数値

**baseline（本番経路、`--method refine_diarize --breakdown`、本ラウンドで再実測）**:

| meeting | DER (pyannote/simpleder) | miss | fa | confusion |
|---|---|---|---|---|
| ES2011a | 21.6% / 17.7% | 11.3% | 3.5% | 6.8% |
| IS1008a | 6.8% / 4.0% | 2.6% | 3.5% | 0.7% |
| ES2004a | 23.2% / 17.8% | 14.5% | 3.7% | 5.0% |
| IS1009a | 23.7% / 16.9% | 6.8% | 4.5% | 12.4% |
| TS3003a | 15.9% / 14.1% | 8.7% | 6.5% | 0.7% |
| **平均** | **18.2% / 14.1%** | — | — | — |

（simpleder平均14.1%は§15までの13.9%と0.2pt違うが、これは§14で述べた
`GroupDiarizer`のローカルクラスタリングが完全には決定論的でないことに
よるrun間ノイズ床の範囲内。）

**プロトタイプのクラスタリング閾値スイープ（平均DER、pyannote/simpleder）**:

| 閾値 | overlap-on | overlap除去後 |
|---|---|---|
| 0.45 | 19.6% / — | 17.3% / — |
| 0.50 | 17.4% / — | 16.3% / — |
| 0.55 | 15.5% / 12.7% | 15.1% / 12.2% |
| 0.60 | 15.2% / 12.5% | 15.2% / 12.5% |
| 0.65 | **14.6% / 12.1%** | 15.2% / 12.5% |
| 0.70 | 22.5% / — | 23.2% / — |
| 0.80 | 25.4% / — | 25.8% / — |

0.65が最良で、0.70でES2011a（2話者に潰れる）とES2004a（3話者）が崩壊する。
つまり最適点は崖のすぐ手前にあり、**この閾値はデータセットに合わせて
選んだ値でしかない**。

**閾値0.65での会議別内訳**（overlap-onとoverlap除去後は**同一の仮説**を
採点し直したもの。除去は各瞬間で局所的に長いほうのターンだけを残す）:

| meeting | floor | baseline DER | overlap-on DER | miss | fa | conf | 除去後 DER | miss | fa | conf |
|---|---|---|---|---|---|---|---|---|---|---|
| ES2011a | 9.7% | 21.6% | **15.9%** | 6.6% | 6.6% | 2.7% | 18.6% | 12.0% | 1.9% | 4.7% |
| IS1008a | 2.4% | 6.8% | 7.9% | 1.9% | 5.3% | 0.7% | **6.7%** | 2.4% | 2.9% | 1.4% |
| ES2004a | 14.2% | 23.2% | **17.6%** | 10.1% | 5.3% | 2.2% | 20.2% | 15.5% | 3.3% | 1.4% |
| IS1009a | 12.7% | 23.7% | 23.2% | 5.5% | 8.1% | 9.7% | **22.0%** | 10.7% | 2.2% | 9.1% |
| TS3003a | 2.1% | 15.9% | **8.6%** | 5.5% | 2.4% | 0.8% | 8.6% | 5.8% | 2.1% | 0.7% |
| **平均** | — | **18.2%** | **14.6%** | — | — | — | 15.2% | — | — | — |

**powersetデコードはoverlapを実際に回収している**。overlap比率の高い
2会議で、missがfloorを明確に下回った:

| meeting | floor | 除去後 miss | overlap-on miss | 回収分 |
|---|---|---|---|---|
| ES2004a | 14.2% | 15.5% | 10.1% | **5.4pt** |
| IS1009a | 12.7% | 10.7% | 5.5% | **5.2pt** |
| ES2011a | 9.7% | 12.0% | 6.6% | **5.4pt** |

overlap除去後のmissがfloorとほぼ一致している（ES2004a 15.5% vs 14.2%、
ES2011a 12.0% vs 9.7%）ことが、floorの見積もり自体の妥当性の裏取りにも
なっている。

**ただしDERの改善のほとんどはoverlapのおかげではない**。平均で見ると:

- baseline 18.2% → プロトタイプ（overlap除去後）15.2% … **-3.0pt。
  これはoverlapとは無関係で、ファイル全体を一括処理するオフライン
  クラスタリングそのものの効果。**
- overlap除去後 15.2% → overlap-on 14.6% … **-0.6pt。これがoverlap
  出力の正味の寄与。**

回収したmissが5pt級なのに正味0.6ptしか残らないのは、overlapを出すと
false alarmとconfusionがほぼ同じ量だけ増えるからである（ES2004aで
fa 3.3%→5.3%、ES2011aで fa 1.9%→6.6%）。参照が単独発話としている区間に
2人目を出してしまう誤検出が、正しい2人目の検出とほぼ相殺している。
IS1008a・TS3003aのようなoverlapがほぼ無い会議では、overlap出力は
**純粋に害**（IS1008a 6.7%→7.9%）になる。

**実行時間（このマシン、CPU、`--threads 4`、600秒の音声1本あたり）**:

| meeting | 全体 wall | segmentation | 埋め込み | クラスタリング | RTF |
|---|---|---|---|---|---|
| ES2011a | 6.9s | 1.5s | 4.6s | <0.05s | 0.011 |
| IS1008a | 8.3s | 1.5s | 6.3s | <0.05s | 0.014 |
| ES2004a | 7.1s | 1.5s | 5.1s | <0.05s | 0.012 |
| IS1009a | 8.0s | 1.5s | 5.9s | <0.05s | 0.013 |
| TS3003a | 7.3s | 1.5s | 5.2s | <0.05s | 0.012 |

参考までに、本番経路のdiarization部分は同じ音声でdiar_rtf 0.031〜0.068
（`decode`全体では35〜59秒）。プロトタイプは**ファイル全体を一括で
処理してなおその半分以下**で、コストの2/3は埋め込み抽出が占める。
segmentation ONNXを自前で回すこと自体は安い（600秒あたり1.5秒、
RTF 0.0025）。閾値スイープが実質無料なのは、segmentationと埋め込みを
1回だけ計算して全閾値で共有しているため。

### 判断

**プロトタイプとしては成功、本番投入は保留**。

達成したこと: §14が「possible-with-work」と判定したpowerset直接デコードは
実際に動き、overlap区間のmissを5pt級で回収できることを実測で示した。
overlap floorという概念自体も、除去後のmissがfloorとほぼ一致することで
裏が取れた。

一方、**この結果を「baselineより3.6pt良い」と読むのは誤り**である。改善分の
5/6はoverlapではなくオフライン一括クラスタリングに由来しており、それは
hayamimiがリアルタイム字幕である以上そのままは使えない性質の利得である。
overlap出力そのものの正味の効果は平均-0.6pt、しかもoverlapの少ない会議では
プラス（悪化）に転じる。

**本番へ入れるとしたら何が要るか**:

- 配線先は清書パス（`Refiner._emit_turns()`）。`GroupDiarizer`をこの
  powersetデコード版に差し替えれば、25秒以内の清書グループ単位でなら
  overlapを出せる。ただし本プロトタイプの利得の大半はグループ境界を
  越えたファイル全体クラスタリングから来ているので、**清書グループ単位に
  切り戻した瞬間にその分は消える**。残るのは正味-0.6ptの側だけで、
  それは採用基準（0.3pt以上の改善）をかろうじて満たす程度でしかない。
  この差分を実測せずに採用は決められない。
- 出力側の設計変更が要る。現状の`Refiner`は1スパン1話者を前提に
  ターンを組み立てており、字幕としても「同時に2人の行を出す」体験が
  未設計。DER以前にUXの決定が先。
- 埋め込みコストは1.5〜2倍程度に増える（窓ローカル話者ごとに1回、
  hop 5秒なら600秒あたり150〜200回）。RTFの絶対値は小さいが、
  リアルタイム動作中の予算としては無視できない。

**未解決のまま残ること**:

- **3人以上の同時発話は原理的に出せない**。segmentation-3.0のpowersetは
  最大2人（`powerset_max_classes`=2）で、モデルを差し替えない限り
  この上限は動かない。
- **overlap区間の埋め込み汚染**。単独フレームが不足する窓ローカル話者は
  混合音声から埋め込みを取るしかなく、実測でES2004a 181中37件、
  IS1009a 206中52件（約25%）がこのフォールバックに落ちている。
  IS1009aのconfusionが9.7%とbaseline（12.4%）からあまり下がらないのは
  ここが効いている可能性が高い。
- **クラスタリング閾値の脆さ**。0.65が最良で0.70で崩壊する以上、
  AMI以外のデータで同じ値が使える保証はない。実運用では話者数が
  未知のまま閾値だけで切ることになるので、ここは本番化の最大の
  リスク要因。

### 残課題

- 清書グループ単位（25秒窓）に制約したときのoverlap正味効果は未測定。
  本ラウンドで測ったのはファイル全体版のみ。この数字が無いと本番投入の
  可否は判断できない。
- false alarmの増加分の内訳（segmentationのペアクラス誤検出なのか、
  窓の多数決集約が甘いのか）は未診断。pyannote本家はソフトスコアを
  Hamming窓で重み付き集約するが、本プロトタイプは二値の多数決なので、
  ここは改善余地がある可能性がある。
- 埋め込み汚染への対処（重なり区間の分離、あるいは重なり区間を
  埋め込みから完全に除外して話者数を減らす）は未着手。

## 17. Round 7: セッション終了時グローバル再クラスタリングの実測 — 不採用（confusion平均+22.5pt悪化）

### 背景

§16（Round 6）の帰属分析で、offline全ファイル一括プロトタイプが
baselineより3.6pt良かった（pyannote 18.2%→14.6%）うち、**3.0ptは
overlap対応とは無関係で、清書グループの境界を越えてファイル全体を
一括クラスタリングすることそのものの効果**だと判明していた
（overlapの正味寄与は-0.6ptのみ）。本ラウンドの仮説はここから
自然に導かれる: 清書グループ境界を越えた一括クラスタリングの利得
だけを、**ライブ出力に一切手を触れずに**回収できないか。

**方式**: セッション終了時（post-hoc）のグローバル再クラスタリング。
清書パス（`Refiner._emit_turns()`）は今まで通りグループが閉じるたびに
ローカルクラスタをその場でグローバルS{n}へgreedyにremapし続ける
（ライブ出力は完全に不変）。それに加えて、各グループのローカル
クラスタの埋め込み・所属グループ・発話長を**セッション全体にわたって
溜めておき**、セッション終了時に一度だけ§16と同じ2段階制約付き
階層クラスタリング（単独発話1.5秒以上のクラスタだけがマージに参加、
閾値0.65、同一グループ内のクラスタはcannot-link、短小クラスタは
事後に最近傍centroidへ割当）を全体に対してかけ直し、その結果で
ラベルを書き換える。「清書が後から確定済みテキストを訂正する」のと
同じ設計思想 — ライブは今まで通り、セッション終了時の一回だけ
事後改訂する。

### 実装

共有化: §16のプロトタイプ（`scripts/eval_diar_overlap.py`）に直書き
されていた2段階クラスタリング（`cluster_local_speakers()`/
`assign_by_centroid()`）を新規`scripts/global_recluster.py`へ
切り出し、`two_stage_cluster()`という1つの高レベル関数にまとめた
（embeddings・group_ids・durations・threshold・reliable_sを渡すだけ）。
`eval_diar_overlap.py`は今はこのモジュールから`cluster_reliable`
（旧`cluster_local_speakers`の実体）と`assign_by_centroid`を
import するだけになり、既存の`tests/test_diar_overlap.py`はimport元が
変わっただけで無改修のまま通る（30件全て緑）。§16のロジックを
再実装せず再利用する、というタスクの要件どおり。

- `scripts/eval_diar.py`の`generate_diarize_hypothesis()`に
  `global_recluster`/`global_recluster_threshold`引数、
  `--global-recluster`/`--global-recluster-threshold`（デフォルト0.65）
  CLIフラグを追加。既存の各グループ処理ループで、remap対象になった
  ローカルクラスタ（`cluster_embs`に実体があるもの、＝実際に埋め込みが
  取れたクラスタ）ごとに`{group_idx, local_id, embedding, duration_s}`を
  `recluster_entries`へ積む。全グループを処理し終えた後、フラグが
  onなら`two_stage_cluster()`を一回呼び、返ってきたクラスタIDを
  `S{cluster_id+1}`にマップして、**そのクラスタIDを持つ出力ターンの
  ラベルだけ**を書き換える（フラグoffなら`recluster_entries`は
  空リストのまま参照されず、既存の出力は一切変わらない）。
- `scripts/realtime_transcribe.py`の`Refiner`に同じ`global_recluster`/
  `global_recluster_threshold`コンストラクタ引数、
  `--speaker-global-recluster`/`--speaker-global-recluster-threshold`
  CLIフラグを追加。`_emit_turns()`が既存のremapを行うのと全く同じ場所
  （ローカルクラスタの埋め込みを計算した直後）に、フラグが立っている
  ときだけ同じ形の行を`self._recluster_entries`へ積む（`group_idx`は
  `_emit_turns()`の呼び出し回数を数えるだけの単調カウンタ）。
  新規メソッド`Refiner.run_global_recluster()`をshutdown経路
  （`main()`の`finish()`、`refiner._task_queue.join()`の直後）から
  呼ぶ。**このメソッドは診断用の集計を1行printするだけ**で、コンソール・
  SSE配信・トランスクリプトファイルのどれも書き換えない — 本タスクの
  指示どおり「配信UXは今回のスコープ外、shutdown時に呼ばれてクラッシュ
  しなければよい」を満たす最小実装。理由も明記した:
  ラベルを書き換えるにはコンソール・SSE・トランスクリプトの各消費者に
  「後から表示済みの行を訂正する」UXが必要で、それ自体が独立した
  設計判断であり本ラウンドでは踏み込まない。

### 測定

**flag-offリグレッション確認**（`--method refine_diarize --breakdown`、
5会議、collar=0.25s）: miss/false-alarm/confusionの内訳が§16の
baseline表と完全一致（ES2011a miss=11.3/fa=3.5/confusion=6.1、
IS1008a 2.6/3.5/0.7、ES2004a 14.5/3.7/5.0、IS1009a 6.8/4.5/12.4、
TS3003a 8.7/6.5/0.7 — 全て一致）。simpleder平均は13.9%（§16の14.1%と
0.2pt差だが、これは§14で既知のローカルクラスタリング非決定性の
ノイズ床の範囲内で、本ラウンドの変更が原因ではない）。**リファクタ
（`enumerate(groups)`の導入とhyp_keysの記録）はflag-offの出力に
影響していないことを確認した。**

**flag-on（閾値0.65）** の会議別内訳（pyannote DERはmiss+fa+confusionの
概算、simplederはeval_diar.pyの主指標）:

| meeting | entries→clusters | baseline confusion | flag-on confusion | baseline DER(概算pyannote/simple) | flag-on DER(概算pyannote/simple) |
|---|---|---|---|---|---|
| ES2011a | 35→4 | 6.1% | **32.0%** | 20.9%/17.0% | 46.8%/44.7% |
| IS1008a | 51→7 | 0.7% | **5.7%** | 6.8%/4.0% | 11.8%/9.8% |
| ES2004a | 48→8 | 5.0% | **26.5%** | 23.2%/17.8% | 44.7%/41.1% |
| IS1009a | 63→10 | 12.4% | **33.7%** | 23.7%/16.9% | 45.0%/41.4% |
| TS3003a | 17→3 | 0.7% | **31.1%** | 15.9%/14.1% | 46.3%/45.1% |
| **平均** | — | — | — | **18.1%/13.9%** | **38.9%/36.4%** |

採用基準（pyannote平均+0.3pt以上改善、単一会議の悪化0.5pt以内）を
桁違いに外れて悪化（simpleder平均+22.5pt、5会議全てで+3〜31pt悪化）。
**§15（Round 5）のjoint-remap実験（+3.7pt悪化）よりさらに一桁悪い**。

**閾値スイープ**（0.55/0.65/0.75、simpleder平均）: 35.2% / 36.4% /
35.9%。**閾値には全くセンシティブでない** — どの値でも壊滅的で、
「0.65が悪いだけで別の閾値なら救える」という余地はない。

**attribution確認**: miss・false alarmはflag-on/offで完全に同一値
（上表参照、例えばES2011aのmiss=11.3%/fa=3.5%はどちらも変化なし）。
これは測定するまでもなく設計上自明でもある —
`global_recluster`はターンの開始・終了時刻を一切変更せず、`S{n}`
文字列だけを書き換えるため、miss/false alarmの計算対象（区間の有無）
は数学的に不変。**今回の悪化は100%confusion由来**であることが、
測定と実装の両面から確認できる。

**実行時間**: `recluster=`欄（全会議で0.00〜0.02秒）。対象となる
埋め込み数が17〜63件と少なく（後述）、2段階クラスタリング自体は
事実上無視できるコスト。§16の全体一括クラスタリング（600秒あたり
0.05秒未満）と同様、再クラスタリングのステップそのものは安い。

### なぜ悪化したか（メカニズム診断、指示どおり実測で検証）

**仮説（タスク指示にあった作業仮説）**: 「清書グループ単位のローカル
クラスタcentroidは§16の窓単位埋め込みより粗い粒度なので、
cannot-link／マージの判断がノイズに弱いのではないか」。

**検証**: 会議あたりの再クラスタリング対象埋め込み数を比較すると、
本ラウンドは17〜63件（上表）に対し、§16のプロトタイプは同じAMI
会議で181件（ES2004a）・206件（IS1009a）と報告されている
（§16「残課題」）。**約3〜10倍少ない**。これは仮説を支持する一つの
根拠ではある（絶対数が少ないほど1件のノイズの影響が大きい）が、
それだけでは今回の悪化の規模（confusion 5〜6倍）を説明しきれない。

そこで実際の出力ラベル集合を直接調べたところ、**より支配的な、別の
メカニズム**が見つかった。例としてES2011a（参照話者4人）を診断すると:

```
n_groups=32  n_recluster_entries=35  n_recluster_clusters=4
最終ラベル集合: {S1, S2, S3, S4, S6, S7}
（内訳: S1-S4 = 再クラスタリングが生成した4クラスタ、正しく参照の4人と一致
       S6, S7 = 再クラスタリング対象に一度も入らなかった「1話者判定
                 グループ」の多数決フォールバックラベル。fast pathの
                 オンライン漸進centroidが独自に開いた9個のcentroid
                 （`opened_by={'fast': 9}`）のうち生き残った番号）
```

つまり**再クラスタリングは対象にした35件だけを見れば4人に正しく
まとめている**（`n_recluster_clusters=4` = 参照話者数と一致）。しかし
32グループのうち、清書グループ内で2人以上のローカル話者が検出できた
グループだけが再クラスタリング対象になり（35エントリ）、**1話者としか
判定されなかったグループはそのままfast pathの多数決ラベルを引き継ぎ、
再クラスタリングには一切参加しない**。この2つのラベル空間
（再クラスタリングが新たに割り振った`S1〜S4`と、素通りしたfast path
由来の`S6・S7`）は**互いに独立に生成されており、どちらも同じ`S{n}`
という文字列空間を使っているにもかかわらず、同一の実話者を指している
という保証が一切ない**。DERのスコアリングは仮説話者ラベルと参照話者を
1対1で対応付けるため、同一の実話者の発話が`S1`（再クラスタリング側）と
`S6`（素通りfast path側）に分裂していると、そのどちらか一方しか
正しい参照話者に対応付けられず、もう一方は必ずconfusion（またはFA）に
回る。これはラベル文字列の衝突バグではない（名前空間を`RC{n}`のような
衝突しない別空間にして再実験しても、ES2011aはむしろ44.7%→48.0%と
悪化した — 衝突の有無は本質ではなく、**2つの独立したラベル空間へ
分裂すること自体**が問題）。

**結論**: 悪化の主因は「粒度が粗い」という作業仮説単体ではなく
（部分的には支持されるがそれだけでは規模を説明できない）、それに
加えて、あるいはそれ以上に、**セッション中の清書グループの相当数
（本ラウンドの5会議で35〜75%が「1話者判定」で再クラスタリング対象に
入らない）が、post-hocの再クラスタリングと一切結び付かないまま古い
fast pathのラベルを引きずり続けること**。§16のプロトタイプは
ファイル全体を最初から一括で扱うため、この「対象漏れ」という概念が
そもそも存在しなかった。本ラウンドの実装（清書グループ単位の既存
remapフローに後乗せする設計）は、タスクが要求した「ライブ出力に
手を触れない」という制約を満たす代わりに、副作用として「セッションの
一部だけが再クラスタリングされ、残りは古いラベル体系のまま孤立する」
という新しい失敗モードを持ち込んでしまった。

### 判断

**不採用**。`--global-recluster`（`eval_diar.py`）／
`--speaker-global-recluster`（`realtime_transcribe.py`）フラグは
コードとしては残す（デフォルトoff、機能追加のみで本番非影響。
flag-offのリグレッション確認は上記のとおり合格）が、有効化は
一切推奨しない。共有モジュール`scripts/global_recluster.py`への
リファクタ自体は独立して価値がある（§16プロトタイプの重複コードを
解消、`tests/test_diar_overlap.py`は無改修のまま通過）ため維持する。

pytest: 全件（159 passed, 1 skipped）で緑。今回は採用基準未達のため
タスク定義どおり新規ユニットテストの追加は行っていない
（`two_stage_cluster()`自体は既存の`test_diar_overlap.py`が
`cluster_reliable`/`assign_by_centroid`を通じて間接的に検証している）。

### 残課題

- 「対象漏れ」を解消する設計（1話者判定グループの多数決フォールバック
  ラベルも、埋め込みさえ残っていれば再クラスタリング対象に含める、
  あるいは事後にfast path centroidと再クラスタリング結果のcentroidを
  もう一段マッチングする）は未着手。ただし後者は結局§15
  （joint-remap／exclude-provisional-remapともに不採用）と同種の
  「remap側のノブを増やす」対策であり、そちらの失敗パターン
  （制約を強めるとLOCAL側に誤りが押し出される）を繰り返すリスクがある。
- 粒度不足（17〜63件 vs §16の181〜206件）を独立に切り分けて測る
  実験（例えば清書グループの単位を25秒より短くして母数を増やす）は
  未実施。ただし「対象漏れ」メカニズムの方が支配的である以上、
  優先度は低いと判断する。
- 本ラウンドで確定したのは「今回の実装方針（清書グループ単位の
  ローカルクラスタをそのまま溜めてセッション末尾で一括再クラスタ）は
  ダメ」ということであり、「ライブ出力を変えずに§16の一括クラスタリング
  利得を回収する」という上位の目標自体が原理的に不可能だと示したわけ
  ではない。この目標を追うなら、清書グループの単位を今より細かく
  保ったまま母数を増やすか、あるいは「対象漏れ」を作らない別の
  アーキテクチャ（例えば1話者判定グループも常に埋め込みを1つ取って
  再クラスタリング対象に含める）から出直す必要がある。

## 18. Round 8: 「対象漏れ」を塞いだ再クラスタリングの実測 — 不採用（プール完全化後も+8〜11pt悪化、granularity floorを直接確認）

### 背景

§17（Round 7）はconfusion+22.5pt悪化で不採用となったが、その根本原因は
2段階クラスタリングのアルゴリズム自体ではなく、**プールの不完全性**
だと特定されていた: ローカルdiarizerが「1話者」と判定した清書グループ
（あるいはdiarizerが例外で処理を諦めたグループ）は一度も埋め込みを
作らず、再クラスタリング対象プールに一切入らないまま、fast pathの
多数決ラベルを永久に引きずり続ける。§17実測のES2011aでは32グループ中
17グループ（53%）がこれに該当し、再クラスタリングが正しく再構成した
`S1〜S4`（参照4人と一致）とは完全に独立した`S6・S7`という別ラベル空間へ
分裂していた。本ラウンドの課題はこの「対象漏れ」を実装で塞ぎ、その上で
再測定すること。

### 実装

`scripts/global_recluster.py`に純粋関数`pool_audio_for_group()`を追加
（モデル非依存、`tests/test_diar_overlap.py`に4件のユニットテストを
新規追加、全件緑）。1話者判定/diarizer decline時に埋め込む音声と長さを
決める:

- ローカルdiarizerが実際に話者区間を検出したが単一クラスタだった場合
  （`turns`が非空）: その区間の音声だけを結合して埋め込む — diarizer
  自身の「発話区間」判断を尊重し、生バッファ全体（無音区間を含む）は
  使わない。
- diarizerが完全に諦めた場合（例外、または全区間が最低長未満で
  フィルタされた）: グループの生バッファ全体にフォールバック。

両呼び出し元をこの関数で更新:

- `eval_diar.py`の`generate_diarize_hypothesis()`: 1話者フォールバック
  分岐（既存の`hyp.append((majority, ...))`のすぐ後）で、
  `global_recluster`がonのときだけ`local_id=-1`という番兵キーで
  プールエントリを追加。`hyp_keys`にも`(group_idx, -1)`を積むことで、
  この分岐のターンも既存の書き換えロジック（`key_to_label`経由）で
  他のターンと全く同じ扱いになる — 新しい分岐を書き換えパスに
  作らずに済んだ。
- `realtime_transcribe.py`の`Refiner._emit_turns()`: 元のコードは
  「raw内の distinct local id が2未満」で早期return、その後「フィルタ後の
  turn数が2未満」で2度目の早期returnという、**判定条件が異なる**2つの
  decline地点を持っていた（前者はフィルタ前のdistinct id数、後者は
  フィルタ後の単純なturn数 — 同一local_idの2 turnでも通過してしまう
  既存の細かい非対称性）。当初この2つを1つの`len({t[0] for t in turns}) < 2`
  判定にまとめて実装したが、これは後者のケース（フィルタ後2 turnが
  同一local_idのときに従来はdecline**しない**）で判定結果が変わって
  しまうことに気づき、**2つのdecline地点をそのまま維持**し、それぞれの
  地点でその時点の`turns`スナップショットを使ってプールエントリを積む
  `record_pool_entry()`ヘルパーを両地点から個別に呼ぶ設計に修正した。
  flag-offの出力を一切変えないという制約（後述の回帰確認で検証）を
  守るため。

### 測定

**flag-offリグレッション確認**（`--method refine_diarize --breakdown`、
5会議、collar=0.25s）: §17文書化済みのbaselineとminas/false-alarm/
confusion内訳が完全一致（ES2011a 11.3/3.5/6.1、IS1008a 2.6/3.5/0.7、
ES2004a 14.5/3.7/5.0、IS1009a 6.8/4.5/12.4、TS3003a 8.7/6.5/0.7 —
全て一致）、simpleder平均も13.9%で§17と同一。実装のフラグoff経路は
リファクタの影響を受けていない。

**flag-on（閾値0.65）の会議別内訳**（pyannote DERはmiss+fa+confusionの
合計、simplederはeval_diar.pyの主指標）:

| meeting | entries→clusters (§17→本ラウンド) | baseline confusion | flag-on confusion | baseline DER(pyannote/simple) | flag-on DER(pyannote/simple) |
|---|---|---|---|---|---|
| ES2011a | 35→4 / **52→4** | 6.1% | **28.5%** | 20.9%/17.0% | 43.3%/40.8% |
| IS1008a | 51→7 / **53→7** | 0.7% | **4.9%** | 6.8%/4.0% | 11.0%/9.1% |
| ES2004a | 48→8 / **69→8** | 5.0% | **8.6%** | 23.2%/17.8% | 26.8%/22.4% |
| IS1009a | 63→10 / **67→11** | 12.4% | **29.5%** | 23.7%/16.9% | 40.8%/36.8% |
| TS3003a | 17→3 / **38→4** | 0.7% | **2.4%** | 15.9%/14.1% | 17.6%/16.1% |
| **平均** | — | — | — | **18.1%/13.9%** | **27.9%/25.0%** |

「対象漏れ」を塞いだことで全会議のエントリ数が増加（例: TS3003a
17→38、ほぼ倍増）し、confusion悪化幅は§17から大幅に縮小した
（simpleder平均+22.5pt → **+11.1pt**、pyannote平均+20.8pt →
**+9.8pt**）。ES2004a・TS3003aは悪化幅1桁台まで縮小。それでも
採用基準（pyannote平均+0.3pt以上**改善**、単一会議の悪化0.5pt以内）を
依然大きく外れて悪化している——「対象漏れ」修正は改善方向への一歩
だったが、baselineを上回るには全く足りなかった。

**閾値スイープ**（0.55/0.65/0.75、simpleder平均）: **21.9% / 25.0% /
23.9%**。§17（35.2/36.4/35.9%）と比べ全閾値で改善したが、最良の0.55
でもbaseline 13.9%に対して+8.0pt悪化。§17同様、「別の閾値なら救える」
余地はない。

**attribution確認**: miss・false alarmはflag-on/offで完全に同一値
（上表参照）。これはラベル文字列のみを書き換える設計上、測定するまでも
なく自明——今回もconfusionの変化だけで悪化幅が説明できる。

**実行時間**: `recluster=`欄は0.00〜0.02秒（§17と同水準）。エントリ数が
52〜69件に増えても、2段階クラスタリング自体のコストは無視できる。

### なぜそれでも悪化したか（メカニズム診断、指示どおり実測で検証）

「対象漏れ」を塞いだにもかかわらず悪化が残った理由を、ES2011a
（参照話者4人、FEE041/FEE042/FEE043/FEE044）で直接検証した。
`two_stage_cluster()`が返す4クラスタの中身を、各エントリの時間区間と
参照RTTMの重なりから求めた「真の話者」で突き合わせると:

```
cluster 0: n=23  true_spk別内訳={FEE044:7, FEE043:11, FEE041:4, FEE042:1}
cluster 1: n=2   true_spk別内訳={FEE041:2}
cluster 2: n=16  true_spk別内訳={FEE041:15, FEE044:1}
cluster 3: n=11  true_spk別内訳={FEE041:7, FEE044:3, FEE043:1}

真の話者ごとの内訳:
FEE041 (28エントリ, 会議で最も発話が多い話者) -> cluster {3:7, 1:2, 2:15, 0:4}
FEE044 (11エントリ) -> cluster {0:7, 3:3, 2:1}
FEE043 (12エントリ) -> cluster {0:11, 3:1}
FEE042 (1エントリ)  -> cluster {0:1}
```

**クラスタ数(4)は参照話者数(4)と偶然一致しているが、内容は全くの
混合**。最も発話が多いFEE041は4クラスタ全てに分散し、cluster 0は
4人全員の発話が混在している。つまり「対象漏れ」は解消され、52件全て
がプールに入って一括クラスタリングされているにもかかわらず、
**清書グループ／ローカルクラスタ粒度の埋め込みでは、そもそも
話者を正しく分離できていない**。

§17の「残課題」が示唆していた仮説——粒度不足（本ラウンド52〜69件 vs
§16のウィンドウ単位181〜206件）——が、対象漏れという交絡要因を
取り除いた上で直接支持された。清書グループ単位のローカルクラスタ
埋め込みは、1エントリが最大30秒超の音声（`pool_audio_for_group()`が
diarizer declineグループにフォールバックする生バッファ、上のES2011a
診断では最大31.9秒）を1本のCAM++埋め込みに圧縮することがあり、
これは§16のウィンドウ単位埋め込み（数秒粒度）よりはるかに粗い。
粗い埋め込みは複数話者の発話が混ざっている可能性が高く、cannot-link
制約（同一グループ内は別人）だけでは、その混ざった埋め込み自体が
別人のクラスタへ誤って引き寄せられることを防げない。

### 判断

**不採用（再確認）**。「対象漏れ」というRound 7の根本原因は本ラウンドで
実際に塞がれ、悪化幅は半分以下に縮小した（有意義な前進ではあった）が、
それでもbaselineには遠く及ばない。これは**全ファイル一括クラスタリング
の方向性そのものを閉じる**——清書グループ単位のローカルクラスタ埋め込み
という粒度を保つ限り、対象漏れをどれだけ丁寧に塞いでも、埋め込み自体の
話者分離能力が足りない。

`--global-recluster`（`eval_diar.py`）／`--speaker-global-recluster`
（`realtime_transcribe.py`）フラグはコードとして残す（デフォルトoff、
有効化は一切推奨しない）。`pool_audio_for_group()`のプール完全化は
`--global-recluster`使用時の診断精度そのものを引き上げる独立した価値が
あるため維持する。realtime_transcribe.py側は本番のsoak前提とはならない
（run_global_recluster()は診断printのみで、ライブ出力・SSE・
トランスクリプトのどれも書き換えない、§17から変更なし）。

pytest: 163 passed, 1 skipped（§17の159 passed, 1 skippedに、本ラウンド
追加の`pool_audio_for_group()`ユニットテスト4件を加えた数）。

### 残課題

- §16のウィンドウ単位一括クラスタリングが実測した-3.0ptの利得は、
  「後付けのラベル書き換え」というアーキテクチャでは（対象漏れの
  有無に関わらず）清書グループ粒度の埋め込みでは再現できないことが
  Round 7・8の2ラウンドで確定した。この利得を狙うなら、清書グループ
  という単位そのものを変える（更に細かく分割する、あるいはウィンドウ
  単位の埋め込みをそもそも清書パスとは別に保持し続ける）別アーキテクチャ
  が必要——ただしそれはリアルタイム字幕としての清書グループ設計
  （GROUP_GAP_S/GROUP_MAX_S、docs/DIARIZATION_PLAN.md本文の各所参照）
  自体の見直しになり、本ループ（Round 1〜8）のスコープを超える。
- overlap出力のUX（§16で据え置き）、セッション末尾ラベル訂正のUX
  （§17で明示化）は共に未着手のまま。後者は本ラウンドでは
  「不採用が再確認された」ことで優先度がさらに下がった——訂正UXを
  設計する動機だった再クラスタリング手法自体が、粒度floorにより
  当面採用見込みが立たないため。

### 総括（Round 1〜8、本ループの最終まとめ）

| Round | 施策 | 効果 | 判定 |
|---|---|---|---|
| 2（§12） | `min_duration_on/off`スイープ、`--skip-overlap`採点 | ノブ調整は効果なし〜悪化、採点オプションのみ有用 | ノブ**不採用**／`--skip-overlap`**採用**（評価機能） |
| 3（§13） | ライブSilero VAD検出しきい値スイープ | Miss改善はFA増加とほぼ相殺、純益なし | **不採用** |
| 4（§14） | confusion内訳診断（LOCAL/REMAP/FAST-PATH）、overlap対応可否調査 | 診断手法として確立、REMAP誤りがconfusionの主因と特定 | 診断手法**採用**、対策は§5送り |
| 5（§15） | joint remap（Hungarian割当）、provisional centroid除外 | joint remap +3.7pt悪化、provisional除外 +0.9pt悪化 | 両方**不採用**（機能はデフォルトoffで温存） |
| 6（§16） | overlap対応diarizationプロトタイプ（powerset直接デコード） | pyannote -3.6pt、うち-3.0ptはoverlap非依存の一括クラスタリング由来、overlap正味は-0.6pt | プロトタイプ**成功**、本番投入は**保留**（清書グループ単位での純利得が未測定） |
| 7（§17） | セッション終了時グローバル再クラスタリング（§16の手法をライブ出力非改変で後乗せ） | simpleder平均+22.5pt悪化、5会議全て悪化、閾値非依存。根本原因は「対象漏れ」（1話者判定グループがプールに入らず旧ラベルのまま孤立）と特定 | **不採用** |
| 8（本ラウンド、§18） | Round 7の「対象漏れ」をプール完全化で修正、再測定 | 悪化幅を半減（simpleder +22.5pt→+11.1pt、pyannote +20.8pt→+9.8pt）したがbaseline未達。診断により原因を「清書グループ粒度の埋め込みの話者分離能力不足」と特定（対象漏れとは独立の別floor） | **不採用**。全ファイル一括クラスタリング方向性を**この粒度では終了** |

**採用に至った本番動作変更は、この8ラウンドを通じて実質ゼロ**
（採用済みなのは評価スクリプトの機能や診断手法であって、`realtime_transcribe.py`
のデフォルト挙動そのものを動かしたものではない）。baseline
（pyannote 18.2% / simpleder 14.1%、§16実測）は Round 1〜8の
どの実験にも更新されていない。

**現在の床（何が動かせないと分かったか、Round 8時点）**:

- **overlap floor**（§16）: AMI 5会議平均で参照発話時間の約8.2%
  （ES2004a 14.2%〜TS3003a 2.1%）が、hayamimiの「1瞬間1話者」設計
  そのものに起因する構造的miss。overlap detectionを実装しない限り
  動かない。
- **VAD-missed quiet speech**: §13でライブVAD感度をスイープしても
  Miss改善とFA増加がほぼ相殺し純益がなかった。低エネルギー発話の
  取りこぼしはVADしきい値の単純なスイープでは解消できない床として
  残っている。
- **ローカルクラスタ信頼性**（REMAP誤りの根） （§14〜§15）: 清書
  グループのローカルクラスタをグローバルcentroidへ貪欲最近傍で
  remapする現行方式に起因するconfusionは、remap側のノブ（joint割当、
  provisional除外）をどう動かしても悪化にしかならなかった。§15の
  分析どおり、LOCAL誤り（ローカルdiarizer自体の過分割）と独立ではなく、
  remap側を締めるとLOCAL側に誤りが押し出されるトレードオフが観測
  されている。
- **清書グループ粒度の全ファイル一括クラスタリング floor**（§16〜§18で
  新たに確定）: ファイル全体一括クラスタリングの利得（§16、-3.0pt）は
  実在するが、清書グループ単位のローカルクラスタという粒度でそれを
  後付けのラベル書き換えとして回収しようとした2ラウンド（§17: 対象漏れ
  で失敗、§18: 対象漏れを塞いでも埋め込みの分離能力不足で失敗）とも
  baselineに届かなかった。この利得はこの粒度・このアーキテクチャでは
  もう狙わない、という結論が本ループの最終到達点。

**明示的に据え置きの2項目**:

1. **overlap出力のUX**（§16で据え置き）: DER以前に「同時に2人の行を
   出す」体験自体が未設計。powersetデコードの技術的な実現可能性は
   §16で実証済みだが、清書パスへの配線は出力設計が固まるまで着手
   できない。
2. **セッション末尾ラベル訂正のUX**（§17で明示化、§18で優先度低下）:
   `Refiner.run_global_recluster()`は診断ログを1行printするだけで、
   コンソール・SSE・トランスクリプトのどれも書き換えない。§18の
   不採用再確認により、この訂正UXを設計する動機（採用基準を満たす
   再クラスタリング手法）自体が当面見込めなくなったため、これ以上
   このループでは追わない。将来この方向を再開するなら、清書グループの
   単位設計（GROUP_GAP_S/GROUP_MAX_S）自体の見直しから始める必要がある。

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
- [k2-fsa/sherpa-onnx `offline-speaker-diarization-pyannote-impl.h`](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h)（Round 4 T3、`gh api`で取得して直接確認: `ExcludeOverlap()`/`ToMultiLabel()`/`InitPowersetMapping()`）
- [pyannote-audio `utils/powerset.py`](https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py)（上記C++コード内で直接参照されているpowersetデコードの元アルゴリズム）
- `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` のONNXメタデータ（Round 6で直接読み出し: `window_size`=160000, `receptive_field_shift`=270, `num_classes`=7, `powerset_max_classes`=2）
- リポジトリ内: `scripts/speaker_id.py`, `scripts/realtime_transcribe.py`,
  `scripts/download_models.py`, `README.md`, `docs/GOALS.md`,
  `docs/BENCHMARKS.md`
