# Engine Scorecard (end-to-end, real speech)

本番経路 (LID→ルーティング→デコード→ja句読点) のエンドツーエンド採点。
単発クリップのためプリロール・二段パスは含まない。metricは en=WER, 他=CER（yueはt2s正規化）。

## 評価データの出典

| lang | データセット | 性質 |
|---|---|---|
| ja | ReazonSpeech test split(ミラー: japanese-asr/ja_asr.reazonspeech_test) | **日本のTV放送**由来。「実放送」と呼んでいるのはこのセット |
| en | LibriSpeech dev-clean(openslr/librispeech_asr) | オーディオブック朗読(放送ではない) |
| zh / ko | google/fleurs validation split | 朗読音声(放送ではない) |
| yue | google/fleurs(yue_hant_hk) | 朗読音声(放送ではない) |

「実放送でCER 3.8%」のような表現が指すのは **ja 行(ReazonSpeech=TV放送由来)のみ**。
他言語は朗読系データでの計測であり、性質が異なる。再生成手順は
`scripts/make_realset.py` / `make_realset_zhko.py` / `make_realset_yue.py`(各スクリプト
冒頭に取得元の詳細あり)。標準ベンチ一本での5言語比較は FLEURS 統一計測
(`docs/BENCHMARKS.md` 2026-09-01節)を参照。

| lang | clips | LID正解 | 主tier | mean err | mean RTF |
|---|---|---|---|---|---|
| ja | 15 | 15/15 | rz | 0.038 | 0.090 |
| en | 15 | 15/15 | v3 | 0.023 | 0.102 |
| zh | 12 | 12/12 | pz | 0.066 | 0.084 |
| ko | 12 | 12/12 | sv | 0.081 | 0.060 |
| yue | 12 | 12/12 | sv | 0.061 | 0.043 |

## LID誤判定の内訳

誤判定なし。

### 2026-09-01 再計測の注記

- ja 7.5%→3.8%: 冒頭欠落修正(疑い時分割リトライ)とCJK数字正規化(ITN)を含む現行
  パイプラインでの再計測。参照文がアラビア数字表記のためITNが素直に効く。
  なお ja_06 は音声に実在するが参照文に無い冒頭発話まで出力されるようになり
  (参照側の不備)、このクリップ単体は見かけ上悪化している。
- zh 5.3%→6.6%: 悪化分約1.3ptは数字の表記ズレ(出力「1000多年」vs 参照「一千多年」
  など)。認識誤りではなく採点表記の不一致。FLEURS(参照が全てアラビア数字)では
  ITNはzh -2.0ptの改善。
- en/ko/yue: 誤り率は小数第3位まで前回と一致(セット再生成の決定性の傍証)。
