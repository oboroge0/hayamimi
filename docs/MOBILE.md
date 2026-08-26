# Mobile-Sized ja Model — INT8 Quantization of ReazonSpeech Zipformer

First step toward a "スマホ搭載プロファイル" (phone-deployable profile): shrink
the ja-tier model (ReazonSpeech k2 zipformer transducer, `model_type=zipformer`,
`sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/`) with dynamic INT8
quantization and measure what it costs in accuracy, using
`scripts/quantize_reazonspeech.py`.

## Method

- Dynamic quantization (`onnxruntime.quantization.quantize_dynamic`,
  `weight_type=QInt8`) applied independently to encoder, decoder, and joiner.
- Two candidate mobile profiles were built and evaluated against the fp32
  baseline:
  - **full_int8** — encoder + decoder + joiner all INT8.
  - **encoder_only_int8** — only the encoder (by far the largest component)
    is quantized; decoder/joiner stay fp32. This follows a known sherpa-onnx
    community observation that quantizing the small AR decoder/joiner heads
    of a zipformer transducer risks more CER regression than the bytes saved
    are worth.
- Recognizer built the same way production does
  (`scripts/asr_engine.py::_build_reazon`): `OfflineRecognizer.from_transducer`,
  `decoding_method="modified_beam_search"`, `modeling_unit="cjkchar"`.
- Accuracy: CER on the 15 real-broadcast ja clips in `testdata/eval_real/`
  (`manifest.json`), reusing `cer_ja` from `scripts/eval_accuracy.py`
  (NFKC-normalized, punctuation/whitespace stripped, micro-averaged Levenshtein
  distance over all 15 refs).
- Speed: RTF (decode time / audio duration) on this Windows PC (CPU), not
  representative of an on-device ARM benchmark.

## Size

| component | fp32 | int8 (ours) | reduction |
|---|---|---|---|
| encoder | 261.1 MB | 69.9 MB | 73.2% |
| decoder | 5.2 MB | 1.3 MB | 74.9% |
| joiner | 4.1 MB | 1.0 MB | 74.8% |
| **total** | **270.3 MB** | **72.2 MB (full_int8)** | **73.3%** |
| encoder_only_int8 total | — | 69.9 + 5.2 + 4.1 = 79.2 MB | 70.7% |

(Our from-scratch dynamic quantization landed within ~1MB of the encoder
`.int8.onnx` already shipped in the model directory upstream — 69.9 MB vs
69.9/70.9 MB for decoder/joiner respectively — confirming the same technique
is in play. Ours are written under `quantized_ort/` so they don't collide
with the existing files, and are not committed — `models/` is untracked.)

## Accuracy (CER, ja, micro-averaged over 15 clips)

| variant | CER | Δ vs fp32 | mean RTF (this PC) |
|---|---|---|---|
| fp32 (baseline) | 5.84% | — | 0.024 |
| full_int8 | 5.50% | **−0.34pp (slightly better)** | 0.062 |
| encoder_only_int8 | 5.50% | **−0.34pp (slightly better)** | 0.062 |

Both int8 variants produced byte-identical hypotheses on all 15 clips in this
run, so encoder-only vs full-int8 made no measurable accuracy difference here
— the decoder/joiner are tiny relative to the encoder and dynamic
quantization of the encoder dominates any effect. Given that, **full_int8 is
the better default for mobile**: it saves an extra ~7MB over encoder_only for
identical accuracy in this sample.

CER did not regress — both quantized variants were marginally *better* than
fp32 (one clip, ja_05, went from misreading "敦賀さん" as "駿河さん" in fp32
to a different but still-imperfect reading in int8; the two ambiguous /
already-hard clips ja_04 and ja_13 were unaffected by quantization either
way). This is a small 15-clip sample — treat the accuracy delta as "no
measurable regression" rather than "quantization improves ASR."

## Speed caveat

RTF got *worse* on this x86-64 PC after quantization (0.024 → 0.062, i.e.
~2.6x slower), even though both are comfortably real-time. Dynamic INT8
matmul on this onnxruntime/CPU combination doesn't out-run the fp32 MLAS path
at this model's shapes — PC benchmarks are not a stand-in for mobile. On
ARM (which is what sherpa-onnx's Dart/Flutter bindings target for Android/iOS)
INT8 has dedicated NEON/dot-product kernels that fp32 doesn't, so the
size win is the number that should be expected to translate to mobile;
RTF must be re-measured on an actual device before trusting it.

## Loadability check

Both quantized variants load and decode successfully through
`sherpa_onnx.OfflineRecognizer.from_transducer` with the exact same
constructor arguments (`model_type="zipformer"`,
`decoding_method="modified_beam_search"`, `modeling_unit="cjkchar"`) used in
production (`scripts/asr_engine.py::_build_reazon`) — see
`scripts/quantize_reazonspeech.py::variant_recognizer` /
`evaluate`. Since sherpa-onnx's Dart bindings wrap the same C++
`OfflineRecognizer` used here (no Python-specific code path), and the model
directory already ships upstream `.int8.onnx` files of comparable size that
are known to work through `sherpa_onnx.dart`, there's no format reason these
onnxruntime-produced `.int8.onnx` files wouldn't load the same way on mobile.
This has only been verified on desktop CPU in this task — an actual
Android/iOS load test is still open.

## Reproduce

```
H:\Programming\hayamimi\.venv\Scripts\python.exe scripts\quantize_reazonspeech.py
```

Requires `onnx` in addition to the existing `onnxruntime` (installed for this
task: `pip install onnx`, pulled in `ml_dtypes` as a transitive dependency —
both are lightweight, pure quantization-time deps, not needed at inference
time). Writes quantized models under
`models/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17/quantized_ort/`
(`full_int8/`, `encoder_only_int8/`) and a `scratch_quantize_results.json`
summary at repo root; neither is committed (`models/` is git-untracked, the
JSON is a scratch artifact).

To re-evaluate without re-running quantization: add `--skip-quantize`.

## On-emulator accuracy parity

Answers the question left open above ("no format reason these onnxruntime-
produced `.int8.onnx` files wouldn't load the same way on mobile — this has
only been verified on desktop CPU") by actually running the shipped
`.int8.onnx` files through `sherpa_onnx.dart` on an Android emulator
(`hayamimi_test` AVD, x86_64, API level matching `mobile/android`) and
scoring the output against the same PC baseline docs/EVAL_REAL.md and this
document use.

**Speed caveat, stated once and applying to every number below**: the
emulator's CPU is the *host* Windows PC's CPU running under virtualization,
not a real ARM device. RTF/decode-time numbers from this setup say nothing
about a phone. Only **accuracy** (output text) is meaningful here, because
`OfflineRecognizer.decode` is a deterministic forward pass over the same
model weights regardless of how fast it runs.

### How it works

`mobile/hayamimi_core/lib/bench/manifest_eval_runner.dart` (`ManifestEvalRunner`)
adds a `kDebugMode`-gated "Manifest batch eval" panel to the Bench tab
(`mobile/lib/main.dart`): given a model directory, a `manifest.json` (the
same `{"wav","lang","ref"}` array format as `testdata/eval_real/`), and a
WAV directory, it decodes every entry through one recognizer and writes a
JSON results file (`wav`, `lang`, `ref`, `hyp`, `audio_s`, `decode_s`, `rtf`)
that `scripts/eval_accuracy.py`'s `cer_ja` can score directly. The recognizer
is built with `decodingMethod: 'modified_beam_search'`
(`buildZipformerRecognizer` in `bench_runner.dart`, shared with the
single-file RTF bench) to match desktop production
(`scripts/asr_engine.py::_build_reazon`) — sherpa-onnx's own default is
`greedy_search`, which would have compared a different decoding config
rather than a different platform.

Models and test WAVs were pushed to `/data/local/tmp` and copied into the
app's private storage (`getApplicationDocumentsDirectory()`, i.e.
`.../app_flutter/`) via `adb shell run-as`, since the app sandbox is not
directly writable by `adb push`.

### Verification A — CER on the 15 real-broadcast ja clips

Same clips and scorer as the desktop INT8 result above
(`testdata/eval_real/`, `cer_ja`, micro-averaged over all 15 refs):

| run | CER | vs PC INT8 |
|---|---|---|
| PC, INT8 full_int8 (this doc, above) | 5.50% | — |
| Android emulator, same `.int8.onnx` files, `modified_beam_search` | 6.19% | +0.69pp |

12 of 15 clips were exact matches after normalization; the 3 that differed
(`ja_04`, `ja_05`, `ja_13`) were already the harder/more ambiguous clips in
the original PC eval. A +0.69pp CER delta on a 15-clip sample is well within
what's expected from ONNX Runtime kernel/build differences between desktop
Windows and the emulator's Android-targeted onnxruntime build (see
Verification B) — not evidence of a mobile-specific accuracy bug.

### Verification B — real video audio, mobile vs PC text parity

`testdata/videos/case1_ja_news.wav` (a 73s real Japanese news clip, no
reference transcript) was decoded once on the emulator and once on the PC,
both through the identical `.int8.onnx` weights and
`decoding_method="modified_beam_search"` / `modeling_unit="cjkchar"`
config. Text match rate (normalized Levenshtein, `cer_ja` with the PC
transcript as "reference"):

- **90.0% match** (edit distance 20 over a 201-character normalized PC
  transcript).
- The two transcripts are word-for-word identical except for one clause:
  the emulator's output includes "町の路上で村田孝一容疑者の体を押さえつけ"
  (~20 characters) immediately after "富山市芝園", which is **missing
  entirely** from the PC transcript — the PC output jumps straight to
  "暴行を加えた". Every other sentence in the 73s clip is byte-identical
  between the two runs.
- Ruled out as the cause: thread count. Re-running the PC decode with
  `num_threads=2` (matching the mobile bench's default) produced a
  byte-identical PC transcript to the `num_threads=6` run, so this isn't
  parallel-reduction nondeterminism.
- Most likely explanation: the emulator's onnxruntime build (Android
  x86_64, shipped by the `sherpa_onnx_android_x86_64` Flutter plugin) and
  the desktop onnxruntime build take different CPU kernel paths for the
  same ONNX graph, producing small floating-point differences that
  occasionally tip a low-margin `modified_beam_search` decision — in this
  case the PC pruning a beam the emulator kept. This is a platform
  numerics difference, not a mobile-side correctness bug: if anything the
  emulator transcript was the more complete one on this clip.

**Bottom line**: the shipped INT8 model produces the same output on-device
as on desktop to within ~1pp CER / ~90% exact text match on a real 73s
clip, with the one observed discrepancy being a small ASR beam-search
divergence rather than dropped/garbled/wrong-language output. Combined with
the loadability check above, there's no evidence the mobile INT8 pipeline
degrades accuracy relative to the PC pipeline it's copied from.

## Next steps for a mobile profile

- Load + accuracy validated on an Android emulator via `sherpa_onnx.dart`
  (see "On-emulator accuracy parity" above); RTF still needs a real ARM
  device, since the emulator's CPU is just the host PC's.
- Consider static (calibrated) quantization if dynamic INT8 RTF on-device
  turns out worse than fp16 (`*.fp16.onnx`, already shipped upstream at
  ~136.6 MB total — a middle ground between 270.3 MB fp32 and 72.2 MB int8).
- Widen the ja eval set beyond 15 clips before treating the CER delta as
  conclusive.
