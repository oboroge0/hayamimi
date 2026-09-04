# ASR Accuracy Evaluation — Real Speech (Cantonese / yue)

Comparison of **SenseVoice** (multilingual, `language="yue"`), **Omnilingual** (multilingual, no lang hint), and two **dedicated Cantonese models** on a small set of **real, human-spoken** Cantonese utterances (12 clips, 86.7s total audio, `testdata/eval_real_yue/`). Goal: determine whether SenseVoice's built-in `yue` route is good enough, or whether a dedicated Cantonese model should be swapped in, mirroring the existing zh/ko real-speech comparison in `docs/EVAL_REAL_ZHKO.md`.

## Data source

- [FLEURS](https://huggingface.co/datasets/google/fleurs) (`google/fleurs`), config `yue_hant_hk` (Cantonese, Hong Kong, Traditional-Chinese orthography), read-aloud sentences recorded by native speakers, via the `datasets-server` anonymous rows API (no auth required).
  - The `test` split returned an HTTP 500 from the datasets-server rows API (the same class of failure seen for `cmn_hans_cn`'s `test` split, which exceeds the server's parquet scan limit), so the `validation` split was used instead — same corpus/recording conditions, just a different held-out split.
- Utterances were filtered to roughly 3-9s duration (widened to 3-15s only if needed). Source audio was converted to 16kHz mono 16-bit PCM WAV via ffmpeg.

## Models

- SenseVoice: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` (non-autoregressive, use_itn=True, `language="yue"`) — the model currently routed for yue in this repo
- Omnilingual: `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, no lang hint)
- Dedicated Cantonese (WeNet CTC): `sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10` (ASLP-lab WSYue-ASR, U2++ Conformer CTC, INT8, `OfflineRecognizer.from_wenet_ctc`)
- Dedicated Cantonese (zipformer): `sherpa-onnx-zipformer-cantonese-2024-03-13` (icefall zipformer transducer, INT8, `OfflineRecognizer.from_transducer(model_type="zipformer")`)

## Metric

Character error rate (CER), computed the same way as `cer_ja` in `scripts/eval_accuracy.py`: NFKC normalization, strip punctuation, strip all whitespace, then character-level Levenshtein distance / reference length (micro-averaged: total edits / total reference characters across all files). Two variants are reported:

- **CER (raw)**: reference as-is (Traditional Chinese, per FLEURS `yue_hant_hk`) vs hypothesis as-is.
- **CER (t2s)**: both reference and hypothesis passed through `opencc`'s `t2s` (Traditional-to-Simplified) converter before comparing, then normalized/diffed the same way. This removes script-choice mismatches (e.g. a model that emits Simplified characters for the same Cantonese words) from the error count, isolating actual mistranscriptions.


## Per-file results

| file | ref | SenseVoice hyp | SenseVoice CER | SenseVoice CER(t2s) | SenseVoice RTF | Omnilingual hyp | Omnilingual CER | Omnilingual CER(t2s) | Omnilingual RTF | WenetYue-CTC hyp | WenetYue-CTC CER | WenetYue-CTC CER(t2s) | WenetYue-CTC RTF | Zipformer-yue hyp | Zipformer-yue CER | Zipformer-yue CER(t2s) | Zipformer-yue RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yue_01.wav | 如果你在冬季造訪北極或南極地區，你會經歷永夜，亦即太陽不會升到地平線以上。 | 如果你在冬季造访北极或南极地区，你会经历永夜亦即太阳不会升到地平线以上。 | 0.294 | 0.000 | 0.024 | 如果你在東季造訪不極或南極地區你會經力泳嘢 亦即太人不會升到地平線以上 | 0.176 | 0.176 | 0.088 | 如果你在冬季造访北极或南极地区你会经历永夜亦即太阳不会升到地平线以上 | 0.294 | 0.000 | 0.018 | 如果你在東貴做訪北極為男的地區呢會經歷亦即太人不會升到地平線異常 | 0.353 | 0.353 | 0.020 |
| yue_02.wav | 這些理論認為人們有一些需求或渴望，在長大成人的過程中內化了。 | 这些理论认为人们有一些需求或渴望在长大成人的过程中内化了。 | 0.286 | 0.000 | 0.024 | 就些利論認為人們有一些需求或學望助長立成人的過程中內化了 | 0.179 | 0.179 | 0.084 | 这些理论认为人们有一些需求或渴望在奖励时人的过程中内化了 | 0.357 | 0.107 | 0.018 | 這些理論認為人們有一些投忙著那時都過場中買花了 | 0.500 | 0.500 | 0.023 |
| yue_03.wav | 咖哩依據液體含量多寡，而有「乾」或「濕」之分。 | 咖喱依据液体含量多寡而有肝或湿之分。 | 0.294 | 0.118 | 0.033 | 隔利伊據亦體行量多寡而有肝或之分 | 0.412 | 0.412 | 0.085 | 咖喱依据液体含量多寡而有干或湿之分 | 0.294 | 0.059 | 0.018 | 但呢依據亦睇衡量多怪而有光滑十之分 | 0.529 | 0.529 | 0.023 |
| yue_04.wav | 然後由拉卡．辛領銜帶唱拜讚歌。 | 然后由拉卡新领盒大唱败赞歌。 | 0.538 | 0.308 | 0.028 | 然後由拉卡新領客大川敗站角 | 0.538 | 0.538 | 0.088 | 然后由拉卡申领侠大唱拜赞歌 | 0.462 | 0.231 | 0.018 | 然後由拉唱大讚哥 | 0.538 | 0.538 | 0.017 |
| yue_05.wav | 值得在這座迷人村莊漫步半小時。 | 值得在这座迷人村庄漫步半小时。 | 0.214 | 0.000 | 0.034 | 直得在這座咪人村裝慢部半小時 | 0.357 | 0.357 | 0.086 | 值得在这座迷人村庄漫步半小时 | 0.214 | 0.000 | 0.018 | 值得再這座迷人村莊萬部半小時 | 0.214 | 0.214 | 0.023 |
| yue_06.wav | 他們仍在試著判斷撞擊的規模以及對地球的影響。 | 他们仍在试着判断撞击的规模以及对地球的影响。 | 0.381 | 0.048 | 0.024 | 他們仍在試著判斷狀擊的規模以及對地球的影響 | 0.048 | 0.048 | 0.087 | 他们仍在试着判断撞击的规模以及对地球的影响 | 0.381 | 0.048 | 0.017 | 他們仍在是著判斷撞的目的規模以及對地球的影響 | 0.143 | 0.143 | 0.016 |
| yue_07.wav | 一旦您脫離了洋流，游回岸並不比平時困難。 | 一旦你脱离了洋楼游回岸并不比平时困难。 | 0.389 | 0.111 | 0.027 | 一旦你特利了樣流由會安並不比平事昆難 | 0.500 | 0.500 | 0.086 | 一旦你脱离了弱流游回岸并不比平时困难 | 0.389 | 0.111 | 0.017 | 一旦你脫離了腳爐又會啊並不比平時困難 | 0.333 | 0.333 | 0.021 |
| yue_08.wav | 對我來說這似乎不合理；這肯定不公平。 | 对我来说这似乎不合理这肯定不公平。 | 0.312 | 0.000 | 0.028 | 對我來說這似乎不合里 這很定不公平 | 0.125 | 0.125 | 0.087 | 对我内说这似乎不合理这肯定不公平 | 0.312 | 0.062 | 0.017 | 我對我露出這似乎不合理就肯定不恭朋友 | 0.438 | 0.438 | 0.015 |
| yue_09.wav | 其無所不在的能力影響所有人，上至國王，下至平民。 | 其无所不在的能力影响所有人上至国王下至平民。 | 0.143 | 0.000 | 0.028 | 其無所不在的能力影響所有人 上至國王下字平文 | 0.095 | 0.095 | 0.090 | 其无所不在的能力影响所有人上至国王下至平民 | 0.143 | 0.000 | 0.018 | 其無所不在的能力影響所有人嘗試國王下次平民 | 0.143 | 0.143 | 0.018 |
| yue_10.wav | 它也沒有權力撤銷州與州之間的稅法和關稅。 | 他也没有权力撤销州与州之间的税法和关税。 | 0.474 | 0.053 | 0.028 | 他也沒有權力設消周與洲之間的瑞法和關序 | 0.368 | 0.368 | 0.087 | 他也没有权力撤销州与州之间的税法和关税 | 0.474 | 0.053 | 0.018 | 她也沒有權力設週與舟之間的睡法和灣水 | 0.421 | 0.421 | 0.014 |
| yue_11.wav | 針對有幼童的家庭，有些節慶會場會安排專屬的露營區域。 | 针对有幼童的家庭有些节庆会场入会场会安排专属的露营区域。 | 0.583 | 0.125 | 0.032 | 針對有有同的家庭有些接興會長會安排專屬的路型區域 | 0.292 | 0.292 | 0.086 | 针对有友同的家庭有些节庆会长有会场会安排专属的露营区域 | 0.667 | 0.208 | 0.017 | 針對有同嘅家庭有些哲興活著入會場會安排穿熟的路型區域 | 0.500 | 0.500 | 0.022 |
| yue_12.wav | 亞馬遜河也是地球上最寬的河流，有些河段能達到六英里寬。 | 阿妈逊河也是地球上最宽的河流有些河段能达到六英里宽。 | 0.240 | 0.080 | 0.031 | 阿媽信河也是地球上最歡的荷流也些河轉能達到陸英你歡 | 0.400 | 0.400 | 0.086 | 哈马逊河也是地球上最宽的河流有些河段能达到六英里宽 | 0.240 | 0.040 | 0.017 | 他媽也是地球場最寬的河流也些河豚能達到六英美觀 | 0.360 | 0.360 | 0.019 |

## Aggregate

| system | CER raw (micro-avg) | CER t2s (micro-avg) | mean RTF | n |
|---|---|---|---|---|
| Omnilingual | 0.2720 | 0.2720 | 0.0866 | 12 |
| SenseVoice | 0.3400 | 0.0600 | 0.0284 | 12 |
| WenetYue-CTC | 0.3520 | 0.0720 | 0.0176 | 12 |
| Zipformer-yue | 0.3720 | 0.3720 | 0.0192 | 12 |

## Recommendation

- **Keep SenseVoice**. CER(t2s) is 6.0% (SenseVoice) vs 7.2% (best dedicated candidate, WenetYue-CTC) — not a clear enough win to justify adding a second model for yue, especially since SenseVoice already covers 5 languages (zh/en/ja/ko/yue) in one model (RTF: SenseVoice 0.028 vs WenetYue-CTC 0.018).


## Caveats

- **Small sample.** ~12 utterances. These numbers indicate relative system behavior, not statistically robust estimates — a single unusual sentence can swing the aggregate noticeably.

- **Cantonese uses the `validation` split, not `test`**, purely due to a datasets-server HTTP 500 on the `test` split's rows API — both are held-out FLEURS splits recorded under the same conditions, so this should not bias the comparison.

- **Script normalization is approximate.** The `opencc` t2s conversion is a well-established rule-based converter, but Cantonese-specific characters/idioms (e.g. 唔喰/唔啧, 嗚) don't always have a 1:1 Simplified mapping and can still register as errors post-conversion even when the transcription is semantically correct. The raw (non-t2s) CER column is also reported for transparency, but is expected to be inflated for any system that outputs Simplified characters against the Traditional FLEURS reference.

- **FLEURS is read-aloud, single-speaker-per-clip speech**, not spontaneous/conversational or noisy far-field audio — real deployment conditions (accents, code-switching with English/Mandarin, background noise) may show larger gaps between systems than measured here.

- **WenetYue-CTC and Zipformer-yue are Cantonese-only models** (plus some Mandarin/English coverage per their training data) — unlike SenseVoice, they cannot serve the other 4 languages this repo already routes through SenseVoice, so adopting one adds a second loaded model rather than replacing SenseVoice outright, unless yue is split into its own route.

