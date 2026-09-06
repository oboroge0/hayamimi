# Third-Party Notices

`hayamimi_core`'s own source code is MIT-licensed (see `LICENSE`). It ships
no model weights: a host app places pretrained models on the device itself
(automatically via `downloadProfile`, or by hand — see the README's "Model
placement guide"). Those models, and this package's Dart dependencies,
carry their own licenses, listed below.

This file covers `hayamimi_core` only — the mobile Flutter package. The
desktop hayamimi pipeline (`scripts/`) has a wider model catalog (zh, EU
languages, translation, speaker diarization) with its own
`THIRD_PARTY_NOTICES.md` at the repository root; none of those extra
models or Python dependencies are used by this package.

## ASR / VAD models

Placed under `<targetDir>/model`, `/vad`, `/sense_voice`, `/lid` by
`downloadProfile`, or manually — see the README's "Model placement guide"
for exact files and directories.

| Model | Needed by | Publisher | License | Source |
|---|---|---|---|---|
| ReazonSpeech k2 Zipformer (ja ASR) | every profile | Reazon Human Interaction Lab | Apache-2.0 | [reazon-research/reazonspeech-k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2), packaged by [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |
| Silero VAD | every profile (`HayamimiLive` only) | Silero Team | MIT | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), packaged by k2-fsa/sherpa-onnx |
| SenseVoice small (en/zh/ko/yue ASR) | `RoutingProfile.jaSenseVoice` only | Alibaba / FunAudioLLM | Apache-2.0 (code) + FunASR Model Open Source License Agreement (official weights; permits commercial use with attribution) | [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice), packaged by k2-fsa/sherpa-onnx |
| whisper-tiny (spoken-language ID probe) | `RoutingProfile.jaSenseVoice` only | OpenAI | MIT | [openai/whisper](https://github.com/openai/whisper), ONNX export packaged by k2-fsa/sherpa-onnx |

## Japanese punctuation model

Optional, and not placed by `downloadProfile` — see the README's "Japanese
punctuation restoration" and "Model placement guide" for why and where to
get it.

| Model | Publisher | License | Source |
|---|---|---|---|
| `mojicast-punct-onnx` (`punct_bert.fp16.onnx` + `vocab.txt`) | Base models: Tohoku NLP + bobfromjapan; ONNX export: Mojicast (ishiki-emo) | Apache-2.0 | [tohoku-nlp/bert-base-japanese-char-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v3), [bobfromjapan/bert_japanese_punctuation](https://huggingface.co/bobfromjapan/bert_japanese_punctuation), export: [ishiki-emo/mojicast-punct-onnx](https://huggingface.co/ishiki-emo/mojicast-punct-onnx) |

## Dart package dependencies

| Package | License | Notes |
|---|---|---|
| [sherpa_onnx](https://pub.dev/packages/sherpa_onnx) | Apache-2.0 | ONNX Runtime-based inference for the ASR/VAD models above |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | The inference engine itself, bundled by `sherpa_onnx`. This package's own Japanese punctuation code (`lib/punct/`) calls the same already-loaded copy through `dart:ffi` rather than shipping a second one — see the README's "How it reaches ONNX Runtime, and why that way." |
| [record](https://pub.dev/packages/record) | BSD-3-Clause (per the package's own `LICENSE` file) | Microphone capture for `HayamimiLive`/`HayamimiRemote` |
| [archive](https://pub.dev/packages/archive) | MIT (Copyright (c) 2013-2021 Brendan Duncan) | `.tar.bz2` extraction inside `downloadProfile` |
| [crypto](https://pub.dev/packages/crypto) | BSD-3-Clause (Copyright 2015, the Dart project authors) | sha256 verification inside `downloadProfile` |
| [ffi](https://pub.dev/packages/ffi) | BSD-3-Clause (Copyright 2019, the Dart project authors) | Native memory allocation and C string conversion for the ONNX Runtime C API calls in `lib/punct/` |
| [unorm_dart](https://pub.dev/packages/unorm_dart) | MIT (Copyright (c) 2018 Yasuhiro Shimizu) | NFKC Unicode normalization, which `dart:core` has no equivalent of and which the punctuation tokenizer needs to match the Python reference |

## Vendored source (`lib/punct/ort_bindings.dart`)

`lib/punct/ort_bindings.dart` is not original work. It is derived from the
ffigen-generated ONNX Runtime bindings in
[gtbluesky/onnxruntime_flutter](https://github.com/gtbluesky/onnxruntime_flutter)
(**MIT**, Copyright (c) 2023 gtbluesky), file
`lib/src/bindings/onnxruntime_bindings_generated.dart`, which were in turn
generated from ONNX Runtime's `onnxruntime_c_api.h`
([microsoft/onnxruntime](https://github.com/microsoft/onnxruntime), **MIT**,
Copyright (c) Microsoft Corporation).

Only the Dart binding declarations were taken; none of that package's native
code, build files, or higher-level API was copied, and the package is not a
dependency. The copy was trimmed to the ~27 C API members hayamimi_core
calls and one parameter's type was corrected for Windows — both changes are
documented in the file's own header.

## Test fixture data

`test/fixtures/punct_ja_parity.json` contains 40 Japanese sentences from the
**FLEURS** corpus ([google/fleurs](https://huggingface.co/datasets/google/fleurs),
`ja_jp`), Copyright Google, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with their
punctuation removed to make unpunctuated test input. The attribution is
repeated inside the file itself.

## Design inspiration

The Japanese punctuation model's export (`mojicast-punct-onnx`, above) is
published by the [Mojicast](https://github.com/ishiki-emo/mojicast) project
(MIT-licensed offline captioning app) by ishiki-emo. Credit and thanks to
that project.

## TODO verify

- `sherpa-onnx-whisper-tiny`'s HuggingFace model card
  (`csukuangfj/sherpa-onnx-whisper-tiny`) returned 404 during this pass; the
  MIT license listed above is inherited from upstream OpenAI Whisper
  (well-established, see openai/whisper's own `LICENSE` file), not confirmed
  directly on that specific HF mirror's model card.
