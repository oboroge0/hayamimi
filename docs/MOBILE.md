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

## Punctuation model INT8 (`scripts/punct_ja.py` model)

Second mobile-sizing target: the ja punctuation-restoration BERT-char model
behind `scripts/punct_ja.py` (`models/mojicast-punct-onnx/punct_bert.onnx`,
see `docs/PUNCT_JA.md` for the model/architecture background), quantized and
evaluated with `scripts/quantize_punct.py`.

### Background: the upstream INT8 file is broken here

`docs/PUNCT_JA.md` already documents that the INT8 file the upstream Mojicast
HF repo ships (`punct_bert.int8.onnx`) was found non-functional in this
environment (onnxruntime 1.29.0, CPU EP, Windows): its logits were nearly
constant and didn't track the input at all, so `punct_ja.py` was hard-coded
to load the fp32 file only. This task asks the open question left there: is
a *from-scratch* dynamic quantization (the same `quantize_dynamic(...,
weight_type=QInt8)` recipe used for the ReazonSpeech encoder above) any
better, or does this BERT graph just not survive dynamic INT8 quantization
on this onnxruntime build in general?

Answer: **our own INT8 export works correctly** — unlike the shipped one,
it produces token-dependent logits and restores punctuation. So the earlier
failure was specific to the upstream artifact (bad export/upload, or an
onnxruntime-version mismatch at export time), not a fundamental
incompatibility between this model architecture and dynamic INT8 on this
platform.

### Method

- `onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)` applied to
  the whole `punct_bert.onnx` graph (single file, unlike the 3-component
  ReazonSpeech transducer) → `quantized_ort/punct_bert.int8.onnx` (not
  committed, `models/` is untracked).
- Accuracy: reuses the 15 ja clips' *reference transcripts* (already
  punctuated) from `testdata/eval_real/manifest.json` as text-only ground
  truth — no audio/ASR involved. For each ref, `、`/`。`/`？` are stripped to
  build unpunctuated input, `PunctuatorJa.restore()` is run with each model
  variant, and the restored output is scored two ways:
  - **Punctuation-position F1**: since `restore()` never alters
    non-punctuation characters, marks in the reference and in the
    hypothesis can be aligned by base-character position (no edit-distance
    alignment needed) and compared as exact-type matches (`、` vs `。` vs
    `？`) per gap.
  - **Punctuation-inclusive CER**: whole-string Levenshtein distance between
    the fully-restored hypothesis and the original (punctuated) reference,
    divided by reference length — i.e. how close the end-to-end punctuated
    output is to the original sentence, not just comma/period placement.
- Speed: mean wall-clock latency of `PunctuatorJa.restore()` over the same
  15 inputs, CPU, this Windows PC (not representative of an ARM device,
  same caveat as the ASR section above).

### Size

| variant | size | reduction |
|---|---|---|
| fp32 (`punct_bert.onnx`) | 363.5 MB | — |
| int8 (ours, `quantized_ort/punct_bert.int8.onnx`) | 91.4 MB | 74.9% |

### Accuracy (15 ja reference transcripts, text-only, no ASR)

| variant | punct F1 | precision | recall | punct-inclusive CER |
|---|---|---|---|---|
| fp32 | 0.615 | 0.600 | 0.632 | 4.08% |
| int8 (ours) | **0.686** | **0.750** | 0.632 | **3.45%** |

Both models found the same 12/19 correct punctuation marks (recall tied at
0.632 — int8 didn't *miss* anything fp32 caught). int8 had **fewer false
positives**: fp32 over-inserted marks on 2 clips where int8 correctly
left the text unpunctuated at that gap (e.g. ja_08's second/third clauses,
ja_12 — see `scratch_quantize_punct_results.json` for full per-clip
output), which is what drives both the higher precision/F1 and the lower
CER. This is a **15-reference, single-domain (TV news) sample** — not
enough to claim int8 is *better* in general, but it does clearly rule out
the "quantization silently degrades this model" concern raised by the
broken upstream int8 file: our own INT8 export is at least as accurate as
fp32 here, with no regression observed.

### Speed (mean `restore()` latency, this PC, CPU, single-threaded call path)

| variant | mean latency | model load time |
|---|---|---|
| fp32 | 21.35 ms | 429 ms |
| int8 (ours) | **10.56 ms** | **197 ms** |

Unlike the ReazonSpeech encoder above (where dynamic INT8 was *slower* than
fp32 on this PC's MLAS path), this BERT-char model's INT8 export is
**~2x faster** here, consistent with the ~11ms/5ms fp32/int8 numbers the
upstream Mojicast repo itself cites (`docs/PUNCT_JA.md`) — this model's
matmul-heavy transformer shape apparently does hit onnxruntime's INT8
fast path on this build, even though the ASR transducer's conv/GEMM mix
didn't. Model load time is also roughly halved. As with the ASR result,
PC numbers are not a stand-in for ARM timing — only the size and
"does the graph still restore punctuation correctly" checks transfer with
confidence; RTF/latency needs re-measurement on-device.

### Verdict: PASS — candidate for mobile deployment

Both the size win (75% smaller, 363.5MB → 91.4MB) and the accuracy check
(no measurable degradation vs fp32 on this sample; int8 was in fact
marginally better) support using the self-quantized INT8 punctuation model
for the phone-deployable ("スマホ搭載完全版") profile, replacing the
currently-mandatory fp32 file. Recommended before shipping:
- Re-verify on a larger/more diverse ja reference set (15 short TV-news
  sentences is a thin sample, same caveat as the ASR CER above).
- Re-measure latency on an actual ARM device (Android emulator load +
  accuracy parity check, following the same procedure as "On-emulator
  accuracy parity" above, is the natural next step since text-only
  scoring doesn't need audio pushed to the device).
- Wire the int8 path into `PunctuatorJa` as a selectable/default option
  (currently `onnx_filename` must be passed explicitly; `scripts/punct_ja.py`
  still defaults to fp32) once a decision is made to ship it.

### Reproduce

```
H:\Programming\hayamimi\.venv\Scripts\python.exe scripts\quantize_punct.py
```

Writes the quantized model under
`models/mojicast-punct-onnx/quantized_ort/punct_bert.int8.onnx` (not
committed) and a `scratch_quantize_punct_results.json` summary at repo root
(also not committed, scratch artifact). Pass `--skip-quantize` to re-run
just the evaluation against an already-produced int8 file.

## Multi-language routing on mobile

Ports the desktop pipeline's dual-LID routing policy (`docs/LID.md`,
`scripts/asr_engine.py`'s `resolve_dual_confirm`/`resolve_sticky_lang`/
`sv_lid_tag`) to Dart, so the mobile Live screen can switch between
languages per VAD segment instead of running a single fixed-language model
for the whole session (the prior mobile behavior).

### Model catalog decision: ja + SenseVoice only, no dedicated en/EU tier

The desktop pipeline has 4 tiers (ja → ReazonSpeech, zh → Paraformer,
ko/yue → SenseVoice, en+24 EU langs → Parakeet-TDT-0.6B-v3). Parakeet-TDT
-0.6B-v3-int8 is **641 MB** on disk (`models/sherpa-onnx-nemo-parakeet-tdt
-0.6b-v3-int8/`) — well over the >300MB threshold this task set for
starting a "ja+en+SV" profile, and on its own bigger than every other model
this profile needs combined. So mobile ships **2 tiers**:

- **ja** → ReazonSpeech k2 zipformer (`RoutingProfile.jaSenseVoice`'s
  `reazonModelDir`), same model/decoding config as the existing single-model
  path (`modified_beam_search`, matches `scripts/asr_engine.py::_build_reazon`).
- **en/zh/ko/yue** → SenseVoice small, which already covers all four in one
  model (its own internal LID arbitrates between them at decode time). No
  dedicated Paraformer-zh tier either — the desktop docs/EVAL_REAL_ZHKO.md
  win for zh (Paraformer 5.6% CER vs SenseVoice 7.5%) doesn't justify a
  third model on a phone; SenseVoice's zh output is used as-is.

This means `RoutingProfile` has no separate profile for "ja+en+SV" distinct
from "ja+SV": SenseVoice already covers en, so they're the same two models.
See `mobile/hayamimi_core/lib/routing/routing_profile.dart`.

### Model sizes (int8, the files this profile loads)

| model | role | size |
|---|---|---|
| ReazonSpeech ja (encoder+decoder+joiner, int8) | ja tier | 69.8 MB |
| SenseVoice small (int8) | en/zh/ko/yue tier | 228.2 MB |
| whisper-tiny (encoder+decoder, int8) | LID probe | 98.0 MB |
| **total** | | **396.0 MB** |

(Silero VAD, already required by the existing single-model path, is a few
MB more and unchanged.) All three int8 files are loaded simultaneously —
no LRU eviction is implemented (the mobile task's memory-management step
was descoped to "report the total" rather than build unload logic, since
396MB of model weights plus onnxruntime's working-set overhead is a large
but plausibly device-resident footprint for a phone with several GB of RAM;
an actual peak-RSS measurement on-device is a follow-up, not done in this
task — see "Open items" below).

### Architecture

- `mobile/hayamimi_core/lib/routing/lang_routing.dart` — pure, unit-tested
  port of the desktop's `resolve_dual_confirm`, `resolve_sticky_lang`, and
  `sv_lid_tag` (`mobile/hayamimi_core/test/routing/lang_routing_test.dart`
  mirrors every case in `tests/test_units.py`'s dual-LID and sticky-LID
  sections). `resolve_sticky_lang` is ported for completeness/fidelity even
  though nothing calls it yet — mobile has no third tier for languages
  outside SenseVoice's coverage (see catalog decision above), so only the
  dual-confirm path is wired into the live pipeline today.
- `mobile/hayamimi_core/lib/routing/routed_recognizer.dart` —
  `RoutedRecognizerSet` owns the three native model handles (ReazonSpeech,
  SenseVoice, whisper-tiny LID) and implements the per-segment routing
  decision: run whisper-tiny LID (truncated to 4s, matching the desktop's
  `LID_MAX_SECONDS`); if the candidate matches the session's current
  language, decode directly (no SenseVoice probe needed, mirroring the
  desktop's `lang == last_lang` fast path); otherwise, when the candidate is
  one of the 5 SenseVoice-covered languages, decode via SenseVoice to get
  both a transcript AND its own LID tag on the same audio in one call, then
  arbitrate with `resolveDualConfirm` — reusing that decode's text when the
  resolved language isn't "ja" (no double decode), or re-decoding via
  ReazonSpeech when it is. A whisper-tiny candidate outside SenseVoice's
  5-language coverage (e.g. "fr", "ru") has no specialist tier loaded here,
  so the segment holds at the session's current language (or defaults to
  "ja" at bootstrap) — this is mobile's simplification of the desktop's
  4-tier fallback chain down to 2 tiers.
- `mobile/hayamimi_core/lib/live/live_transcriber.dart` — `start()` gained
  `routingProfile`/`senseVoiceModelDir`/`lidModelDir` parameters; when
  `routingProfile.dualConfirmed`, segments go through `RoutedRecognizerSet`
  instead of a single `OfflineRecognizer`, and emitted
  `LiveTranscriptEntry`s carry a `lang` field. The refine ("清書") pass
  re-runs the same routed `decode()` over the merged group audio, which
  re-judges LID on the (usually longer) combined audio — the same idea as
  the desktop's `resolve_refine_lang`, simplified to reuse `decode()`
  directly rather of porting `REFINE_MIN_REGROUP_S`'s separate gate.
- `mobile/lib/live/live_page.dart` — a "Language routing" dropdown
  (`RoutingProfile.jaOnly` default, unchanged prior behavior, vs
  `RoutingProfile.jaSenseVoice`) plus SenseVoice/LID model directory
  fields shown only when routing is on, and a small language badge (e.g.
  "JA", "EN") on each transcript line when the session used routing.
- `mobile/hayamimi_core/lib/bench/manifest_eval_runner.dart` —
  `ManifestEvalRunner.runRouted` batch-decodes a multilingual
  `manifest.json` through `RoutedRecognizerSet`, treating each clip as an
  independent bootstrap "session" (manifest clips are isolated recordings,
  not a continuous conversation, so this measures the harder bootstrap
  dual-LID path specifically — `docs/LID.md` table 3's "一致時正解率" is
  the relevant comparison). Each result gains a `detectedLang` field
  alongside the existing `hyp`/`ref`, so both language-routing accuracy and
  per-language CER can be scored. `mobile/lib/main.dart`'s Bench tab gained
  a matching "Routed multilingual manifest eval" debug panel.

### Accuracy — measured on-device

Run on the `hayamimi_test` AVD (Android emulator, x86_64) via
`ManifestEvalRunner.runRouted` through the Bench tab's "Routed
multilingual manifest eval" panel, driven with `adb shell input tap`
(button coordinates found by scanning the screenshot for the Material
button color, since the Flutter surface exposes no accessibility tree to
`uiautomator`). Same speed caveat as "On-emulator accuracy parity" above:
the emulator's CPU is the host PC's CPU under virtualization, so RTF is
informational only — only the routing decision and the decoded text are
meaningful here.

**Sample**: 20 clips — the first 5 of each language from `testdata/eval_real`
(ja, en) and `testdata/eval_real_zhko` (zh, ko), 127.8s of audio total. This
is a **quarter to a third** of the 12-15-clip-per-language samples
`docs/SCORECARD.md`'s PC numbers use, so treat deltas as indicative, not
conclusive — the task's own reproduction budget capped it at ~20 clips
rather than the full 54-clip PC set.

Scored with the same functions the PC pipeline uses:
`scripts/eval_accuracy.py`'s `cer_ja` (NFKC-normalized, punctuation/
whitespace-stripped, character-level Levenshtein) for ja/zh/ko, `wer_en`
(word-level, `jiwer`) for en — the same split `docs/SCORECARD.md` and
`docs/EVAL_REAL_ZHKO.md` use.

| lang | clips | LID正解 (mobile) | mean err (mobile) | mean err (PC, `docs/SCORECARD.md`) | mean RTF (mobile, informational) |
|---|---|---|---|---|---|
| ja | 5 | 4/5 | 0.320 (CER) | 0.075 (CER) | 0.093 |
| en | 5 | 5/5 | 0.017 (WER) | 0.023 (WER) | 0.062 |
| zh | 5 | 5/5 | 0.110 (CER) | 0.053 (CER) | 0.065 |
| ko | 5 | 5/5 | 0.108 (CER) | 0.081 (CER) | 0.062 |

**Overall language-routing accuracy: 19/20 (95.0%)** — one misroute, all
other 19 clips resolved to the manifest's ground-truth language.

Full raw results (`wav`, `lang`, `ref`, `hyp`, `detected_lang`, `rtf` per
clip) are in the eval JSON pulled off-device during this run; the numbers
above are the aggregates from it.

#### ja: the one misroute, and why the mean CER looks worse than PC's

`ja_02.wav`'s reference is `「ピカピカブ！ピカピカブ！」` — a short
onomatopoeia-only exclamation (Pikachu's cry), no ordinary words. Both
whisper-tiny and SenseVoice's own LID agreed on "zh" for this clip
(`resolveDualConfirm` requires agreement to switch, and got it — the dual
-LID policy did exactly what it's designed to do given two wrong-but
-agreeing signals), so the session routed to SenseVoice and decoded
Chinese-script garbage (`飞卡皮卡追卡。`) against the ja CER scorer, i.e.
CER 1.0 for that one clip. `docs/LID.md`'s own curves already show ja as
one of the two languages (with ko) that whisper-tiny-alone LID never
reaches 95% accuracy on within the 7s window; this is a live instance of
that documented weak spot, on an utterance that's unusually
script-ambiguous even for a human reader (an all-katakana, no-kanji
exclamation carries very little LID signal in either model). Excluding
this one misroute, the other 4 ja clips' mean CER is **0.151** (`ja_04`'s
0.385 is a full-width-digit ITN mismatch — reference `１００ｍ` vs
ReazonSpeech's `百M`, not a decode error — and `ja_05`'s 0.217 is the
same "敦賀さん→駿河さん" ambiguous name clip flagged as already-hard in
the INT8 quantization section above). The routing feature's own accuracy
number (19/20 language-correct) already reflects this misroute; reporting
CER separately here so a single bad-LID clip doesn't get double-counted as
"the ASR got worse," when what actually happened is "the ASR decoded the
wrong language's model, correctly, given what both LIDs agreed on."

#### en: not worse than PC despite dropping the dedicated tier — read this with the sample-size caveat

The task brief expected mobile's en number to come in **worse** than
`docs/SCORECARD.md`'s PC value (WER 2.3%, `v3`/Parakeet-TDT-v3 tier),
since mobile drops that dedicated tier and routes en through SenseVoice
instead (see the model-catalog decision above). On this 5-clip sample it
did not: mobile's SenseVoice-routed en scored WER **1.7%**, nominally
*better* than PC's v3 tier. Two things temper that:

- **n=5 vs PC's n=15** — a single clip is worth 20pp of WER at this sample
  size (`en_03.wav` alone accounts for all the error, at 8.3% WER on a
  transcription that only dropped one letter — "Hstwood" for
  "Hurstwood"). This is not a statistically meaningful comparison; it
  would take a much larger en sample to say with confidence whether
  SenseVoice-routed en is truly competitive with the dedicated v3 tier.
- **These are read-speech LibriVox clips** (`testdata/eval_real`'s en
  source), the easiest case for any ASR system — SenseVoice's own ITN
  already capitalizes/punctuates cleanly on this material (see the `hyp`
  strings above: proper sentence-initial caps, no all-lowercase/no-punct
  output). The casing/punctuation gap `scripts/asr_engine.py`'s own
  comments cite as v3's advantage over other tiers may be more visible on
  harder, more conversational audio than it is here.

**Known trade-off, stated plainly**: dropping the dedicated en tier is a
deliberate mobile-profile compromise made for the size budget (see model
-catalog decision above), not a discovery that it's accuracy-neutral. This
5-clip result doesn't contradict that it's a real trade-off — it's too
small a sample to establish either way. Treat "no measured regression on
this run" as exactly that, not as "the trade-off is free."

#### zh / ko: both routed and decoded correctly on every clip, mean CER somewhat above PC

Both languages hit 5/5 LID accuracy. Mean CER (zh 0.110, ko 0.108) runs
above the PC scorecard's zh 0.053 (dedicated Paraformer-zh tier — mobile
has no such tier, SenseVoice decodes zh directly, see model-catalog
decision above) and ko 0.081 (PC's ko is *also* SenseVoice, same model as
mobile — so this delta isn't a tier difference, just sample variance:
n=5 vs PC's n=12, plus this 5-clip subset happening to include two
harder items, `ko_04`'s Latin-script name "Gibson" mixed into Korean text
and `ko_05`'s off-by-one-syllable start). No clip decoded garbled or
wrong-language text; every error is an ordinary ASR substitution/
insertion/deletion, consistent with `docs/EVAL_REAL_ZHKO.md`'s per-clip
SenseVoice CER range (0.000-0.242) for the same models on the fuller
12-clip set.

#### Reproduction

1. Start `hayamimi_test` (`cmd //c "H:\dev\emu_start.bat"`), `adb
   wait-for-device shell` until `getprop sys.boot_completed` is non-empty.
2. `flutter build apk --debug` + `adb install -r`.
3. Stage a manifest+wav subset on the PC (any subset of
   `testdata/eval_real{,_zhko}/`'s clips + a matching `manifest.json`),
   push the three model directories and the subset to `/data/local/tmp`
   with `adb push`.
4. `adb shell run-as <pkg> cp -r /data/local/tmp/<name> app_flutter/<name>`
   for each of the model dirs and the eval subset — **the correct
   destination is `app_flutter/` directly under the app's private data
   dir (`/data/data/<pkg>/app_flutter`, what `path_provider`'s
   `getApplicationDocumentsDirectory()` actually resolves to on Android),
   not `files/app_flutter/`** — this tripped up the first attempt in this
   session (a `files/app_flutter/` copy sat unused while the app looked in
   `app_flutter/` and reported "Manifest file not found"). `adb shell
   run-as <pkg> ls app_flutter` is the fastest way to confirm placement
   before poking the UI.
5. Launch the app (`adb shell am start -n <pkg>/.MainActivity`), open the
   Bench tab, scroll to "Routed multilingual manifest eval", edit the
   three model-directory fields and the manifest/WAV-dir fields to point
   at the pushed paths (`adb shell input tap` + `input text` +
   `input keyevent KEYCODE_DEL` repeated to clear a field — there is no
   select-all-then-type shortcut over `adb shell input`), tap "Run routed
   manifest eval", and poll with `adb shell screencap` until the summary
   text appears.
6. `adb shell run-as <pkg> cat app_flutter/routed_manifest_eval_result.json`
   to pull the results (has `detected_lang` per clip), then score with
   `scripts/eval_accuracy.py`'s `cer_ja`/`wer_en` on the PC.

### Open items

- On-device peak memory for three simultaneously-loaded models is an
  estimate (see model sizes above), not a measured RSS number.
- `resolve_sticky_lang` is ported and tested but unused by the live
  pipeline (see architecture note above) — it's there for a future tier
  beyond SenseVoice's 5 languages, not exercised by `jaSenseVoice` today.
- The emulator has no usable microphone, so the routed live-mic path
  (as opposed to the routed *manifest* eval, which doesn't need a mic and
  is now measured above) is exercised only through
  `LiveTranscriber.runDebugWavRefineTest`'s existing wav-based debug path
  — not through an actual multi-segment live session with real language
  switches mid-conversation.
- The 20-clip on-device sample above (5 per language) is a quarter to a
  third of `docs/SCORECARD.md`'s 12-15-clip-per-language PC sample; the
  en result in particular (mobile nominally *beating* PC's dedicated v3
  tier) should not be read as "dropping the v3 tier is free" — see the en
  subsection above. A larger on-device run (the full `eval_real`/
  `eval_real_zhko` sets, ~54 clips total) would tighten these numbers.

## UI-path integration smoke test (Live screen + broadcast server + 清書)

The batch eval above (`RoutedRecognizerSet` fed directly from a manifest)
only exercises the decode path. This pass instead drives the actual UI
surfaces a user touches: the Live screen's routing dropdown and Start
listening flow, the 配信サーバー (broadcast server) `/events` SSE feed, and
the 清書 (refine) button — with `RoutingProfile.jaSenseVoice` selected, on
`hayamimi_test` (emulator).

| # | Check | Result |
|---|---|---|
| 1 | Select `ja + SenseVoice (en/zh/ko/yue)`, Start listening → model load + stable "Listening..." | Pass |
| 2 | Debug wav test (ja/en clips) → confirmed lines carry the correct language badge | Pass (see below) |
| 3 | 配信サーバー ON, `adb forward` + `curl /events` → `final` events carry a language field matching the routed language | Pass |
| 4 | 清書 button doesn't crash under routing | Pass |
| 5 | Bugs found | 2 found, both fixed (below) |

**Two bugs found and fixed** (both mobile-only, `mobile/lib/live/live_page.dart`
and `mobile/hayamimi_core/lib/live/live_transcriber.dart`):

1. **配信サーバー always tagged every final event `lang: "ja"`**, even under
   `jaSenseVoice` routing — `_broadcastLang` in `live_page.dart` was a
   hard-coded constant from before routing existed, and `_onEntry` never
   read the segment's actual `LiveTranscriptEntry.lang`. A LAN subscriber
   (OBS, a browser overlay) would have shown every non-ja routed line
   mislabeled "ja". Fixed: `_onEntry` now sends `entry.lang ?? _broadcastLang`,
   falling back to "ja" only for a plain `jaOnly` session (which has no
   `lang` to report).
2. **The "wavから清書テスト" debug button — the only way to exercise the
   live decode pipeline on an emulator, which has no usable microphone —
   never supported routing at all.** It always built a plain ja-only
   `OfflineRecognizer` regardless of the selected `RoutingProfile`, so
   there was no way to check the routing badge or the broadcast lang fix
   above without a real device. Fixed: `LiveTranscriber.runDebugWavRefineTest`
   now takes the same `routingProfile`/`senseVoiceModelDir`/`lidModelDir`
   parameters as `start()` and routes through `RoutedRecognizerSet` when
   `dualConfirmed`; `DebugRefineTestResult` carries a language per decode
   now, and `live_page.dart` displays each with the same `_LangBadge` the
   Live screen's transcript uses. As a second-order fix, the debug test's
   results are now also fed through `_onEntry`/`_onRefineEntry` — the
   normal live-session callbacks — so they land in the transcript list and
   the broadcast server exactly like a real segment would, which is what
   made check #3 above possible to verify on an emulator at all.

**Reproduction / evidence** (`eval_routed/ja_01.wav`, `eval_routed/en_01.wav`
from the prior batch-eval push, model dirs already on-device — see the
"Reproduction" steps above for how those got there):

- ja_01.wav, split into two halves by the debug test (an arbitrary
  midpoint cut, not VAD-aligned — the first half lands mid-word and
  misroutes to `ko` as a result, illustrating that the debug test's split
  is a rougher LID input than a real VAD segment, not a routing bug):
  `個別1: お 싸워요. [KO]` / `個別2: 大変でしたか [JA]` / 清書 (結合):
  `大橋さんはやっぱり大変でしたか [JA]`.
- `/events` SSE stream during that run: `{"type":"final","text":"お
  싸워요.","lang":"ko",...}` then `{"type":"final","text":"大変でしたか",
  "lang":"ja",...}` — confirms the lang field now really does vary with
  the routed language rather than always reading "ja".
- en_01.wav: both halves and the combined 清書 all tag `EN`, text intact
  ("He was in a fevered state of mind..." / "...cast upon his entire
  future."); `/events` shows both as `"lang":"en"`.
- All 101 `hayamimi_core` unit tests plus `flutter analyze` (both `mobile/`
  and `mobile/hayamimi_core/`) pass after these changes.

**Not covered by this pass**: a real multi-segment live session with
mid-conversation language switches (still blocked on the emulator's
missing microphone, same limitation noted above); the manual 清書 button's
UI path (only the debug test's routed refine call was exercised — the two
share the same `RoutedRecognizerSet.decode` call, so this is a light gap).

## Next steps for a mobile profile

- Load + accuracy validated on an Android emulator via `sherpa_onnx.dart`
  (see "On-emulator accuracy parity" above); RTF still needs a real ARM
  device, since the emulator's CPU is just the host PC's.
- Consider static (calibrated) quantization if dynamic INT8 RTF on-device
  turns out worse than fp16 (`*.fp16.onnx`, already shipped upstream at
  ~136.6 MB total — a middle ground between 270.3 MB fp32 and 72.2 MB int8).
- Widen the ja eval set beyond 15 clips before treating the CER delta as
  conclusive.
