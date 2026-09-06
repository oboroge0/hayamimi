# Singing-Voice Scorecard

本番経路 (LID→ルーティング→デコード) を歌唱音声で採点。
ja=PJS(アカペラ自作曲、かな空間CER・pykakasi正規化)、ko/en=CSD(童謡アカペラ、CER/WER)。
ja は同一歌唱者・同一歌詞の朗読版 (PJS speech) との対照比較付き。

## 歌唱 vs 話し言葉ベースライン

| lang | clips | LID正解 | 主tier | mean err (歌唱) | mean err (朗読/話し言葉基準) | mean RTF |
|---|---|---|---|---|---|---|
| ja | 15 | 15/15 | rz | 0.241 | 0.147 (朗読ペア実測) | 0.048 |
| en | 15 | 13/15 | v3+omni+pz | 0.517 | 0.023 (SCORECARD) | 0.120 |
| ko | 15 | 10/15 | sv+rz | 0.567 | 0.081 (SCORECARD) | 0.099 |

## ja: 歌唱 vs 朗読（同一人物・同一歌詞、かな空間CER）

| wav | err 朗読 | err 歌唱 | Δ | 歌唱LID | 歌唱tier |
|---|---|---|---|---|---|
| ja_01.wav | 0.154 | 0.385 | +0.231 | ja | rz |
| ja_02.wav | 0.158 | 0.158 | +0.000 | ja | rz |
| ja_03.wav | 0.200 | 0.167 | -0.033 | ja | rz |
| ja_04.wav | 0.140 | 0.140 | +0.000 | ja | rz |
| ja_05.wav | 0.138 | 0.446 | +0.308 | ja | rz |
| ja_06.wav | 0.080 | 0.120 | +0.040 | ja | rz |
| ja_07.wav | 0.145 | 0.273 | +0.127 | ja | rz |
| ja_08.wav | 0.184 | 0.224 | +0.041 | ja | rz |
| ja_09.wav | 0.200 | 0.280 | +0.080 | ja | rz |
| ja_10.wav | 0.129 | 0.194 | +0.065 | ja | rz |
| ja_11.wav | 0.107 | 0.200 | +0.093 | ja | rz |
| ja_12.wav | 0.054 | 0.071 | +0.018 | ja | rz |
| ja_13.wav | 0.167 | 0.167 | +0.000 | ja | rz |
| ja_14.wav | 0.224 | 0.310 | +0.086 | ja | rz |
| ja_15.wav | 0.200 | 0.475 | +0.275 | ja | rz |

## 歌唱でのLID誤判定

| wav | true | detected | tier | err |
|---|---|---|---|---|
| ko_03.wav | ko | ja | rz | 1.000 |
| ko_07.wav | ko | ja | rz | 1.000 |
| ko_08.wav | ko | ja | rz | 1.000 |
| ko_11.wav | ko | ja | rz | 1.000 |
| ko_14.wav | ko | ja | rz | 1.000 |
| en_07.wav | en | la | omni | 0.818 |
| en_10.wav | en | zh | pz | 0.938 |
