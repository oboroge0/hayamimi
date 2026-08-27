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

## 出典

- [Speaker Diarization — sherpa-onnx docs](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html)
- [k2-fsa/sherpa-onnx offline-speaker-diarization.py](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-speaker-diarization.py)
- [k2-fsa/sherpa-onnx releases: speaker-segmentation-models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)（`gh api` で実サイズ確認: `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` = 6,958,444 bytes）
- [pyannote/segmentation-3.0 (Hugging Face)](https://huggingface.co/pyannote/segmentation-3.0)
- [onnx-community/pyannote-segmentation-3.0 (Hugging Face)](https://huggingface.co/onnx-community/pyannote-segmentation-3.0)
- [3D-Speaker (modelscope, CAM++)](https://github.com/modelscope/3D-Speaker)
- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) / [AMI License (CC BY 4.0)](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- [VoxConverse (joonson/voxconverse)](https://github.com/joonson/voxconverse)
- [pyannote.metrics (GitHub)](https://github.com/pyannote/pyannote-metrics) / [pyannote-metrics (PyPI)](https://pypi.org/project/pyannote-metrics/)
- [simpleder (PyPI)](https://pypi.org/project/simpleder/)
- `.venv` 内 `sherpa_onnx` 1.13.6 実体の `dir()`/`help()` 出力（本調査で直接確認）
- リポジトリ内: `scripts/speaker_id.py`, `scripts/realtime_transcribe.py`,
  `scripts/download_models.py`, `README.md`, `docs/GOALS.md`,
  `docs/BENCHMARKS.md`
