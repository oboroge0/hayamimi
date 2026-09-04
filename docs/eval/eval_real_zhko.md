# ASR Accuracy Evaluation — Real Speech (Chinese + Korean)

Comparison of **SenseVoice** (multilingual), **Omnilingual** (multilingual, no lang hint), and a **dedicated per-language model** on a small set of **real, human-spoken** utterances (12 Mandarin Chinese + 12 Korean, 169.0s total audio, `testdata/eval_real_zhko/`). Goal: determine whether swapping SenseVoice for a dedicated per-language model is worthwhile for zh and ko, mirroring the existing ja/en real-speech comparison in `docs/EVAL_REAL.md`.

## Data source

- **Both languages**: [FLEURS](https://huggingface.co/datasets/google/fleurs) (`google/fleurs`), read-aloud sentences recorded by native speakers, via the `datasets-server` anonymous rows API (no auth required).
  - Chinese: config `cmn_hans_cn`. The `test` split's parquet shard exceeds the datasets-server row-read cap (`Parquet error: Scan size limit exceeded`), so the `validation` split was used instead — same corpus/recording conditions, just a different held-out split.
  - Korean: config `ko_kr`, `test` split (served without issue).
- Utterances were filtered to roughly 3-9s duration (widened to 3-15s only if needed). Source audio was converted to 16kHz mono 16-bit PCM WAV via ffmpeg.

## Models

- SenseVoice: `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` (non-autoregressive, use_itn=True, language forced per entry)
- Omnilingual: `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, no lang hint)
- Dedicated Chinese: `sherpa-onnx-paraformer-zh-int8-2025-10-07` (Paraformer, `OfflineRecognizer.from_paraformer`)
- Dedicated Korean: `sherpa-onnx-zipformer-korean-2024-06-24` (Zipformer transducer INT8, `OfflineRecognizer.from_transducer(model_type="zipformer")`)

## Metric

Character error rate (CER), computed the same way as `cer_ja` in `scripts/eval_accuracy.py`: NFKC normalization, strip punctuation, strip all whitespace, then character-level Levenshtein distance / reference length (micro-averaged: total edits / total reference characters across all files for that language+system). For Korean this also removes spacing (어절 분리) differences between reference and hypothesis, since ASR systems don't reliably reproduce reference spacing conventions and that would otherwise dominate the score independent of actual transcription quality.


## Per-file results


### Chinese

| file | ref | SenseVoice hyp | SenseVoice CER | SenseVoice RTF | Omnilingual hyp | Omnilingual CER | Omnilingual RTF | Paraformer-zh hyp | Paraformer-zh CER | Paraformer-zh RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| zh_01.wav | 因为远离大陆，哺乳动物无法长途跋涉而来，使得巨龟成为科隆群岛主要的食草动物。 | 因为远离大陆哺乳动物无法长途跋射而来，使得巨龟成为克隆群岛主主要的食草动物。 | 0.086 | 0.023 | 因为远林大陆普乳动物无法长度拔射而来 使得聚规成为克隆群岛主要的石草动物 | 0.257 | 0.084 | 因为远离大陆哺乳动物无法长途发射而来使得巨龟成为克隆群岛主主要的食草动物 | 0.114 | 0.017 |
| zh_02.wav | 亚马逊河也是地球上最宽的河流，有些河段的宽度可达6英里。 | 亚马逊河也是地球上最宽的河流，有些河段的宽度可达6英里。 | 0.000 | 0.024 | 亚马信和也是地球上最宽的合流 有些合作的宽度可达英里 | 0.231 | 0.086 | 亚马逊河也是地球上最宽的河流有些河道的宽度可达六英里 | 0.077 | 0.017 |
| zh_03.wav | 他称，他制作了一个 WiFi 门铃。 | 他称，他制作了一个wifi门铃。 | 0.143 | 0.025 | 他称 他制作了一个歪f门龄 | 0.357 | 0.082 | 他称他制作了一个 w i i 门铃 | 0.143 | 0.018 |
| zh_04.wav | 新王国时期的古埃及人惊叹于其前辈的已有一千多年历史的纪念碑。 | 新国王时期的古埃及人惊叹于其前辈的已有1000多年的历史的纪念碑。 | 0.241 | 0.024 | 新国王时期的古埃锦人金叹于其前辈的已有多年的历史的纪念碑 | 0.241 | 0.087 | 新国王时期的古埃及人惊叹于其前辈的已有一千多年的历史的纪念碑 | 0.103 | 0.017 |
| zh_05.wav | 两小时内，政府大楼附近又发生了三起炸弹爆炸事件。 | 2小时内，政府大楼附近又发生了三起炸弹爆炸事件。 | 0.045 | 0.024 | 小时内 政府大楼附近又发生了三尺炸弹爆炸世节 | 0.182 | 0.087 | 两小时内政府大楼附近又发生了三起炸弹爆炸事件 | 0.000 | 0.016 |
| zh_06.wav | 一般来说，卫星电话不能取代移动电话，因为只有在卫星信号畅通的室外，才能进行通话。 | 一般来说，卫星电话不能取代移动电话，因为只有在卫星信号畅通的室外才能进行通话。 | 0.000 | 0.024 | 一般来说 卫星电话不能取待一动电话 因为只有在卫星信号唱通的失外才能进行通话 | 0.111 | 0.088 | 一般来说卫星电话不能取代移动电话因为只有在卫星信号畅通的室外才能进行通话 | 0.000 | 0.017 |
| zh_07.wav | 西班牙人开始了长达三个世纪的殖民时期。 | 西班牙人开始了长达3个世纪的殖民时期。 | 0.056 | 0.025 | 西班牙人开始了长达三个世纪的殖民时期 | 0.000 | 0.084 | 西班牙人开始了长达三个世纪的殖民时期 | 0.000 | 0.019 |
| zh_08.wav | 内陆水道可以作为假期游玩的一个不错的主题。 | 内陆水稻可以作为假期游玩的一个不错的主题。 | 0.050 | 0.027 | 内路水道可以作为驾期游玩的一个不错的主题 | 0.100 | 0.085 | 内陆水稻可以作为假期游玩的一个不错的主题 | 0.050 | 0.019 |
| zh_09.wav | 然后，拉卡·辛格带头唱了拜赞歌。 | 然后拉卡辛格带头唱了半战歌。 | 0.214 | 0.026 | 然后拉卡新格带头唱了半赞歌 | 0.214 | 0.085 | 然后拉卡辛格带头唱了半赞歌 | 0.143 | 0.018 |
| zh_10.wav | 西班牙人开始了长达三个世纪的殖民时期。 | 西班牙人开始了长达3个世纪的殖民时期。 | 0.056 | 0.023 | 西班亚人开始了长达三个世纪的殖民时期 | 0.056 | 0.085 | 西班牙人开始了长达三个世纪的殖民时期 | 0.000 | 0.017 |
| zh_11.wav | 绘图分析的结果将发布在公网上。 | 绘图分析的结果将发布在公网上。 | 0.000 | 0.024 | 绘图分析的结果将发布在功望上 | 0.143 | 0.087 | 绘图分析的结果将发布在公网上 | 0.000 | 0.017 |
| zh_12.wav | 有些音乐节会为带小孩的家庭设立特别的露营区。 | 有些音乐节会为带小孩子的家庭设立特别的露营区。 | 0.048 | 0.025 | 有些音乐节会为带小孩子的家庭设立特别的路营区 | 0.095 | 0.087 | 有些音乐节会为带小孩子的家庭设立特别的露营区 | 0.048 | 0.017 |

### Korean

| file | ref | SenseVoice hyp | SenseVoice CER | SenseVoice RTF | Omnilingual hyp | Omnilingual CER | Omnilingual RTF | Zipformer-ko hyp | Zipformer-ko CER | Zipformer-ko RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| ko_01.wav | 염소 사육은 대략 일만 년 전에 이란의 자그로스산맥에서 시작한 것으로 보입니다. | 염소사육은 대략 1만년 전에 이란의 자그로스 산맥에서 시작한 것으로 보입니다. | 0.030 | 0.022 | 염소 사유은 대략 년전에 일한의 작그로스 3맥에서 시작한 것으로 보입니다 | 0.212 | 0.086 | 염소사역은대략 1만년전에일한에자그로스산맥에서시작한것으로보입니다. | 0.152 | 0.019 |
| ko_02.wav | 그래도 관계자의 조언을 듣고 모든 표지판을 지키고, 안전 경고에 세심한 주의를 기울여야 합니다. | 그래도 관계자의 조언을 듣고고 모든 표지판을 지키고 안전경고에 세심한 주의를 기울여야 합니다. | 0.026 | 0.024 | 그래도 관계자의 좋언을 듣고 모든 표지판을 지키고 안정 경구의 세심한 주의를 기울려야 합니다 | 0.128 | 0.086 | 그래도관계자의조언을듣고모든큐지판을시키고안정경고에세심한주위에기울여야합니다. | 0.128 | 0.018 |
| ko_03.wav | 교전이 발발한 직후 영국은 독일에 대한 해상 봉쇄를 시작한다. | 교전이 발한 짓고 영국은 독일에 대한 해상봉세를 시작한다. | 0.160 | 0.034 | 교전히 발발한 짓고 영국은 독일에 대한 해상 봉세를 시작한다 | 0.160 | 0.084 | 교전이발발은찍고연구원독일에대한해상봉제를시작한다. | 0.280 | 0.018 |
| ko_04.wav | 사건 발생 이후, 깁슨(Gibson)은 병원으로 이송되었으나 얼마 후 숨을 거뒀다. | 사건 발생 이후깁씨는 병원으로 이송되었으나 얼마 후 숨을 거뒀다. | 0.242 | 0.026 | 사건 발생 이후 깊스는 병원으로 이송되었으나 얼마우 스을 거<unk>다 | 0.424 | 0.089 | 사건발생이후깁스는병원으로이송되어스나얼마  스물걷었다. | 0.455 | 0.015 |
| ko_05.wav | 오늘날 날개를 접을 수 없는 곤충은 잠자리와 파리밖에 없습니다. | 어느 날 날개를 접을 수 없는 곤충은 잠자리야 파리밖 없습니다. | 0.154 | 0.037 | 오느날 날개에 접을 수 없는 곤충은 잠자리야 팔이밖에 없습니다 | 0.192 | 0.084 | 어느날개를접을수없는권층은잠자리야파리밖에없습니다. | 0.231 | 0.025 |
| ko_06.wav | 파리 사람들은 자기중심적이고, 무례하고, 오만하다는 평판이 있습니다. | 파리 사람들은 자기중심적이고 무례하고 오만하다는 평판이 있습니다. | 0.000 | 0.024 | 파이 사람들은 자기 중심적이고 물래하고 오만하다는 평판이 있습니다 | 0.103 | 0.085 | 파리사람들은자기중심적이고무례하고오만하다는평판이있습니다. | 0.000 | 0.015 |
| ko_07.wav | 하지만 여전히 새에는 공룡처럼 보이게 하는 것들이 많습니다. | 하지만 여전히 새해에는 공룡처럼 보이게 하는 것들이 많습니다. | 0.040 | 0.031 | 하지만 여전히 세에는 공용처럼 보이게 하는 것들이 많습니다 | 0.080 | 0.086 | 하지만여전히새해에는공용처럼볼게하는것들이많습니다. | 0.160 | 0.027 |
| ko_08.wav | 사자는 무리 지어 사는 가장 사회적인 고양이과 동물입니다. | 사자는 무리  지어 사는 가장 사회적인 고양이과 동물입니다. | 0.000 | 0.025 | 사자는 무리지어사는 가장 사회적인 고양이과 동물입니다 | 0.000 | 0.089 |  | 1.000 | 0.019 |
| ko_09.wav | 스프링복스의 경우, 다섯 경기 연패를 기록했다. | 스프링 복수의 경우 다섯 경기 연패를 기록했다. | 0.053 | 0.026 | 후프링 복수의 경우 다섯경기 연폐를 기록했다 | 0.158 | 0.088 |  | 1.000 | 0.019 |
| ko_10.wav | 이곳은 남아프리카의 명소 중 하나로 남아프리카 국립 공원(SANParks)의 플래그십입니다. | 이곳 은 남아프리카 의 명소 중 하나로 남아프리카 국립 공원의 플래그십 입니다. | 0.200 | 0.025 | 이것은 남 아프리카의 명소 중 하나로 남 아프리카 국립 공원에 플레으십입니다 | 0.300 | 0.087 | 이곳은남아프리카에명소중하나로남아프리카국미공원에플래그쉽있니다. | 0.325 | 0.024 |
| ko_11.wav | 미국 공병대는 시간당 6인치의 강우량이 기 파손된 제방을 무너뜨릴 수 있다고 추정했다. | 미국 공병대 시간당 6인치의 강훈량이 기 파손된 제방을 무너뜨릴 수 있다고 주장했다. | 0.111 | 0.025 | 미국 공병되는 시간당 유인치의 광훈량이 깊 파손된 재방을 문너트릴 수 있다고 지정했다 | 0.250 | 0.090 | 미국공경되는시간당 6인치에광고량이기파선덴재방을무너뜨를수있다고주전했다. | 0.306 | 0.021 |
| ko_12.wav | "1940년 8월 15일, 연합군은 프랑스 남부를 침략했고, 이 침략은 ""드래군 작전""이라 불렸다. " | 1940년 8월 15일 연합권은 프랑스 남부를 침략했고 이 침략은 드래곤 작전이라 불렸다. | 0.053 | 0.029 | 년월일 연학권는 프랑스 남부를 침략했고 이 침략은 드레곤 작전이라 불렸다 | 0.316 | 0.089 | 1940년 8월 15일연합군은프랑스남부를침략했고이침략은드래곤작전이라불렀다. | 0.053 | 0.016 |

## Aggregate

| lang | system | CER (micro-avg) | mean RTF | n |
|---|---|---|---|---|
| ko | Omnilingual | 0.2071 | 0.0869 | 12 |
| ko | SenseVoice | 0.0926 | 0.0273 | 12 |
| ko | Zipformer-ko | 0.3025 | 0.0197 | 12 |
| zh | Omnilingual | 0.1685 | 0.0857 | 12 |
| zh | Paraformer-zh | 0.0562 | 0.0175 | 12 |
| zh | SenseVoice | 0.0749 | 0.0245 | 12 |

## Recommendation

- **Chinese**: **Switch to Paraformer-zh**. CER drops from 7.5% (SenseVoice) to 5.6% (Paraformer-zh), a 25% relative reduction in errors, clearly outweighing any RTF difference (SenseVoice 0.024 vs Paraformer-zh 0.018).

- **Korean**: **Keep SenseVoice**. CER is 9.3% (SenseVoice) vs 30.2% (Zipformer-ko) — not a clear enough win to justify adding/switching a second model for Korean, especially considering SenseVoice already covers 5 languages in one model (RTF: SenseVoice 0.027 vs Zipformer-ko 0.020).


## Caveats

- **Small sample.** ~12 utterances per language. These numbers indicate relative system behavior, not statistically robust estimates.

- **Chinese uses the `validation` split, not `test`**, purely due to a datasets-server row-size limitation on the `test` parquet shard — both are held-out FLEURS splits recorded under the same conditions, so this should not bias the comparison.

- **FLEURS is read-aloud, single-speaker-per-clip speech** (similar register to LibriSpeech), not spontaneous/conversational or noisy far-field audio — real deployment conditions may show larger or smaller gaps between systems than measured here.

- **Korean CER strips spacing**, which is a deliberate deviation from the raw `cer_ja` normalization (which already strips whitespace) — noted here since Korean word-spacing (띄어쓰기) is linguistically meaningful, unlike Chinese/Japanese where it's absent.

- **Zipformer-ko returned empty transcriptions on 2/12 clips** (`ko_08.wav`, `ko_09.wav`, both scored CER=1.0 and pull the aggregate down substantially). Root-caused to low input amplitude: both clips have peak amplitude ~0.015-0.022 (vs ~0.1+ for clips it transcribes correctly); manually gain-normalizing those two clips before feeding them to the model recovers reasonable output (`'사자는무리지와사는가장사회적인고양이과돈버립니다.'` / `'스프링복수에경우다섯경기연패를기록했다.'`), while SenseVoice and Omnilingual handled the same un-normalized low-volume audio without issue. This was evaluated with the same waveform-loading path used for every other system in this repo's harness (`sf.read` → `accept_waveform`, no gain normalization), so it reflects a genuine robustness gap for this specific Zipformer-ko model on quiet audio, not a harness bug — and is itself a strong reason to prefer SenseVoice for Korean regardless of CER on louder clips.

