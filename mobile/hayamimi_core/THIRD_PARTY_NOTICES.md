# Third-Party Notices

hayamimi (早耳)'s own source code is MIT-licensed (see `LICENSE`). It ships no
model weights in this repository -- `scripts/download_models.py` fetches
pretrained models from their original publishers into `models/` (git-ignored).
Those models carry their own licenses, listed below. **Read the "Translation
models" row carefully before redistributing anything that bundles model
weights** -- one of them is share-alike, not permissive.

## ASR / VAD / speaker models

| Model (dir under `models/`) | Publisher | License | Source |
|---|---|---|---|
| `sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17` (ReazonSpeech k2 Zipformer, ja) | Reazon Human Interaction Lab | Apache-2.0 | [reazon-research/reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2), packaged by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |
| `sherpa-onnx-paraformer-zh-int8-2025-10-07` (Paraformer-zh, zh) | Alibaba DAMO Academy / FunASR | Apache-2.0 | [FunASR](https://github.com/modelscope/FunASR) / ModelScope, packaged by k2-fsa/sherpa-onnx |
| `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` (SenseVoice small, ko/yue) | Alibaba / FunAudioLLM | Apache-2.0 (code) + FunASR Model Open Source License Agreement (official weights; permits commercial use with attribution) | [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice), packaged by k2-fsa/sherpa-onnx |
| `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (Parakeet TDT 0.6B v3, en + 24 EU langs) | NVIDIA | CC-BY-4.0 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), packaged by k2-fsa/sherpa-onnx |
| `omnilingual-300m-ctc-int8` (Meta Omnilingual ASR 300M CTC, ~1600-language fallback) | Meta AI | Apache-2.0 | [facebook/omniASR-CTC-300M](https://huggingface.co/facebook/omniASR-CTC-300M) / [facebookresearch/omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr), ONNX export packaged by k2-fsa/sherpa-onnx |
| `sherpa-onnx-whisper-tiny` (spoken-language ID) | OpenAI | MIT | [openai/whisper](https://github.com/openai/whisper), ONNX export packaged by k2-fsa/sherpa-onnx |
| `silero_vad.onnx` (voice activity detection) | Silero Team | MIT | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), packaged by k2-fsa/sherpa-onnx |
| `campplus_sv.onnx` (CAM++ speaker embedding, `--speakers`) | Alibaba DAMO Academy / 3D-Speaker | Apache-2.0 | [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker), packaged by k2-fsa/sherpa-onnx |

## Text models

| Model (dir under `models/`) | Publisher | License | Source |
|---|---|---|---|
| `mojicast-punct-onnx` (Japanese punctuation restoration) | Base models: Tohoku NLP + bobfromjapan; ONNX export: Mojicast (ishiki-emo) | Apache-2.0 | [tohoku-nlp/bert-base-japanese-char-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v3), [bobfromjapan/bert_japanese_punctuation](https://huggingface.co/bobfromjapan/bert_japanese_punctuation), export: [ishiki-emo/mojicast-punct-onnx](https://huggingface.co/ishiki-emo/mojicast-punct-onnx) |
| `mojicast-m2m100-ct2` (M2M-100 418M, ja->zh/ko translation) | Meta AI (base model); CTranslate2 conversion: Mojicast (ishiki-emo) | MIT | [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M), conversion: [ishiki-emo/mojicast-m2m100-ct2](https://huggingface.co/ishiki-emo/mojicast-m2m100-ct2) |
| `mojicast-fugumt-ja-en-ct2` (FuguMT, ja->en translation) | staka (base model); CTranslate2 conversion: Mojicast (ishiki-emo) | **CC BY-SA 4.0 (share-alike)** | [staka/fugumt-ja-en](https://huggingface.co/staka/fugumt-ja-en), conversion: [ishiki-emo/mojicast-fugumt-ja-en-ct2](https://huggingface.co/ishiki-emo/mojicast-fugumt-ja-en-ct2) |

> **CC BY-SA 4.0 flag:** `mojicast-fugumt-ja-en-ct2` (used for `--translate en`)
> is the one non-permissive model in this list. If you redistribute this
> model's weights (not just its runtime output), you must keep attribution to
> staka's Fugu Machine Translator and to the Mojicast conversion, and any
> redistribution of the weights themselves must remain under CC BY-SA 4.0.
> hayamimi's own code and the other models are unaffected -- this only
> applies if you re-host/re-bundle this specific model file. Live subtitle
> *output* text is not itself claimed to be encumbered by this notice; if
> your use case is sensitive to this, use `--translate zh,ko` (M2M-100, MIT)
> or skip `en` translation.

## Eval-only baseline models (not used by the runtime pipeline)

These are downloaded only if you run `scripts/download_models.py` without
`--minimal` and are referenced solely by `scripts/eval_accuracy.py` /
`scripts/make_realset_zhko.py` as comparison baselines during accuracy
evaluation -- `asr_engine.py`'s routing does not use them.

| Model (dir under `models/`) | Publisher | License | Source |
|---|---|---|---|
| `sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8` | NVIDIA NeMo (trained on ReazonSpeech data) | CC-BY-4.0 | packaged by k2-fsa/sherpa-onnx (`asr-models` release) |
| `sherpa-onnx-zipformer-korean-2024-06-24` | k2-fsa / Zipformer (Korean) | Apache-2.0 | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |

## Python runtime dependencies

| Package | License | Notes |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0 | ONNX Runtime-based inference for all ASR/VAD/speaker models above |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | used transitively by sherpa-onnx and `punct_ja.py` |
| [ctranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | translation model inference |
| [sentencepiece](https://github.com/google/sentencepiece) | Apache-2.0 | tokenization for translation models |
| [numpy](https://numpy.org/) | BSD-3-Clause | |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause | |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) | MIT | microphone capture |
| [fugashi](https://github.com/polm/fugashi) | MIT (MeCab itself is BSD/GPL/LGPL tri-license) | Japanese tokenization for punctuation restoration |
| [unidic-lite](https://github.com/polm/unidic-lite) | MIT (dictionary data: BSD-modified, per UniDic) | dictionary for fugashi |
| [kiwipiepy](https://github.com/bab2min/kiwipiepy) | **LGPL-2.1-or-later** | Korean tokenizer, used only to fix SenseVoice's spacing on `ko` output; optional at runtime (falls through silently if not installed) -- kept as an **optional/dev extra**, not a hard runtime dependency, to avoid LGPL obligations on the core install |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | memory diagnostics |

### Dev / eval-only dependencies (`requirements-dev.txt`)

| Package | License |
|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT |
| [jiwer](https://github.com/jitsi/jiwer) | Apache-2.0 |
| [edge-tts](https://github.com/rany2/edge-tts) | LGPL-3.0 |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Unlicense |
| [pytest](https://github.com/pytest-dev/pytest) | MIT |
| [OpenCC](https://github.com/BYVoid/OpenCC) | Apache-2.0 |

## Design inspiration

Several integration choices in this project (the FuguMT/M2M-100 CTranslate2
conversions, the punctuation model export, and beam-size/repetition-control
settings noted in `docs/TRANSLATE.md` and `docs/TRANSLATE_M2M.md`) are drawn
from or reuse artifacts published by the [Mojicast](https://github.com/ishiki-emo/mojicast)
project (MIT-licensed offline captioning app) by ishiki-emo. Credit and thanks
to that project -- see the README credits section.

## TODO verify

- `sherpa-onnx-whisper-tiny`'s HuggingFace model card
  (`csukuangfj/sherpa-onnx-whisper-tiny`) returned 404 during this pass; the
  MIT license listed above is inherited from upstream OpenAI Whisper
  (well-established, see openai/whisper's own `LICENSE` file), not confirmed
  directly on that specific HF mirror's model card.
