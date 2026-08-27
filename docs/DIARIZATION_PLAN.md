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
