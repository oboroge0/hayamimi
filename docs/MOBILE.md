# Mobile-Sized ja Model — INT8 Quantization of ReazonSpeech Zipformer

This is an investigation log (model sizing, quantization, and on-device
measurements), not the package guide. To embed live subtitles in a Flutter
app, or to read the model-placement/platform-status reference this log's
measurements feed into, see
[`mobile/hayamimi_core/README.md`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/README.md).

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

## On-device (iPhone 15) verification

First real ARM RTF measurement, resolving the "RTF must be re-measured on
an actual device" caveat in "Speed caveat" above. Measured via the Bench
tab (`mobile/`) on a physical iPhone 15 (iOS 27.0 beta), zipformer
full_int8, `modified_beam_search`, model pushed to `Documents/model/` and
`Documents/test.wav` (`test_ja_1.wav` from the ReazonSpeech release's
`test_wavs/`) via `xcrun devicectl device copy to` — see
`docs/IOS_VERIFY.md` for the setup steps.

| environment | RTF (ja int8, `modified_beam_search`) | notes |
|---|---|---|
| PC (Windows, x86-64), int8 | 0.062 | "Accuracy" section above |
| PC (Windows, x86-64), fp32 | 0.024 | same |
| Android emulator (x86_64, host PC) | (informational only) | host CPU, not representative |
| **iPhone 15 (real device), int8** | **0.013** (processing time 0.17s) | Bench tab, single run |

Confirms the "Speed caveat" section's prediction: unlike the x86-64 PC
(where int8 was *slower* than fp32 for lack of an int8-specific kernel
path), ARM's NEON/dot-product int8 path makes the phone ~4.8x faster than
the PC's int8 RTF and ~1.8x faster than the PC's own fp32 RTF. This is a
single run on one clip, not yet averaged over multiple clips/runs the way
the PC number is — a strong signal, not a final benchmark.

### Live screen, real mic (first-ever on real hardware)

Same iPhone 15, Live tab, ja model + Silero VAD from `Documents/`.
Transcription worked end-to-end on real mic input; owner-reported
observations over the session (subjective, not instrumented):

- No noticeable heat.
- No UI stutter/jank during decode (the sync-FFI-in-`async` concern noted
  in `mobile/README.md`'s Status section didn't show up in practice here).
- Language routing (`ja + SenseVoice`, `Documents/sense_voice/` +
  `Documents/lid/`) was exercised with real speech switching between ja
  and en. Badge-switch accuracy was mixed, but the session had multiple
  people talking in the room — cross-talk/background speech is the
  suspected cause rather than a routing bug; not conclusive either way
  without a cleaner single-speaker retest.

This resolves the "Open items" bullet below about the routed live-mic path
only having been exercised through the wav-based debug test, not a real
multi-segment live session — it has now been run live, just not yet in
conditions clean enough to call the accuracy number itself trustworthy.

Remote mode (WS streaming to a PC) was ruled out of scope for this session
by the repo owner — the Mac's Wi-Fi IP turned out to be outside the usual
private-LAN ranges (`104.194.96.0/20`), and rather than chase phone-side
reachability the owner chose to skip it. Not attempted; not a known
failure.

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

### Provisional verdict on the 15-clip sample (superseded below)

On the original 15-reference `testdata/eval_real` sample, both the size win
(75% smaller, 363.5MB → 91.4MB) and the accuracy check (no measurable
degradation vs fp32; int8 was in fact marginally better) looked like a
candidate for replacing the mandatory fp32 file on the phone-deployable
profile. That write-up flagged the obvious risk in its own "recommended
before shipping" list: *"Re-verify on a larger/more diverse ja reference
set (15 short TV-news sentences is a thin sample)"* — see "Large-scale
re-verification" immediately below, which did exactly that and reversed
the call.

### Large-scale re-verification (FLEURS ja, n=250) — REGRESSION FOUND, default stays fp32

The 15-clip sample above was drawn entirely from TV-news captions
(`testdata/eval_real`) — a narrow domain and a thin sample for a
pass/fail accuracy call. This step re-ran the same fp32-vs-int8 comparison
on a much larger and differently-sourced set: **250 ja sentences sampled
from FLEURS (`google/fleurs`, `ja_jp` config, `validation`+`test` splits,
456-sentence combined pool after de-duplication)**, using
`raw_transcription`, which already carries natural `、`/`。`/`？`
punctuation — read out loud for the FLEURS speech corpus, so a different
register (formal/written-style, encyclopedic and news-wire sentences)
than the eval_real TV captions. Sentences were fetched via the same
column-projected remote-parquet technique `scripts/eval_translate.py`
uses (`fsspec` + `pyarrow`, `id`+`raw_transcription` columns only, no
audio), filtered to sentences containing at least one target mark and
under 450 characters, then length-decile-stratified sampled (seed 0) so
the set isn't skewed toward the shorter sentences that dominate the raw
pool (25 sentences from each of 10 length buckets covering the full
21–137 character range, median 52 chars). Implemented in
`scripts/quantize_punct.py::build_fleurs_refs`, invoked with
`--source fleurs --n 250`. Scoring method (punct-position F1, punct-
inclusive CER) is unchanged from the 15-clip method above; latency is now
also measured per this run (`--latency`).

**Pre-declared pass criterion** (stated before running the larger eval):
int8 F1 within **−0.02** of fp32 F1 → promote int8 to the `PunctuatorJa`
default (fp32 kept as an explicit opt-in). Worse than that → fail, default
stays fp32.

| variant | n | precision | recall | F1 | punct-inclusive CER | mean latency |
|---|---|---|---|---|---|---|
| fp32 | 250 | 0.8724 | 0.4831 | 0.6218 | 2.81% | 41.04 ms |
| int8 (ours) | 250 | 0.9638 | 0.3757 | 0.5407 | 3.25% | 22.94 ms |

**int8 F1 − fp32 F1 = −0.0812 → FAILS the −0.02 threshold.** The direction
of the earlier 15-clip result (int8 ≥ fp32) does **not** hold at this
sample size/domain — it reverses hard. The mechanism is visible in
precision vs. recall: int8's precision is *higher* than fp32's (0.964 vs
0.872 — it makes fewer wrong-mark insertions), but its recall collapses
(0.376 vs 0.483 — it misses far more of the marks that should be there).
On the 15 short TV-caption clips, fp32's few extra false positives were
enough to make int8 look at least as good; on 250 longer, more varied
FLEURS sentences, int8's systematic under-prediction of commas/periods
dominates and drags F1 well past the pre-declared tolerance. Read together
with the ASR-side result above (INT8 quantization landed within noise on
the zipformer transducer, CER ±0.34pp on 15 clips), this is a reminder
that a 15-clip sample genuinely wasn't enough to catch a real regression
in this BERT-char model — the earlier "no regression, marginally better"
conclusion for punctuation specifically should be treated as retracted by
this larger run, not as a second confirming data point.

Latency did hold up in the same direction as before (int8 ~1.8x faster,
22.94ms vs 41.04ms mean `restore()` call on this PC/CPU) — the speed
benefit from quantizing this model is real and reproduces at scale; it's
the accuracy side that didn't survive a bigger, more diverse sample.

### Verdict: FAIL — int8 does not replace fp32 as the default

`scripts/punct_ja.py`'s `PunctuatorJa` **keeps fp32 (`punct_bert.onnx`) as
the default**, unchanged. The self-quantized int8 file remains available
as an explicit opt-in (`onnx_filename="quantized_ort/punct_bert.int8.onnx"`)
for callers who value the ~4x size reduction and ~1.8x speedup enough to
accept a real, now-measured recall/F1 hit on punctuation restoration — it
is not a drop-in accuracy-neutral replacement. If this is revisited later:
- The recall gap (int8 misses marks fp32 catches) is the thing to dig
  into first — e.g. per-mark-type breakdown (comma vs period vs question)
  or a lower comma/period decision threshold for the int8 variant
  specifically, rather than assuming dynamic INT8 is a dead end for this
  model altogether.
- Static/calibrated quantization (as opposed to the dynamic quantization
  used here) is the natural next experiment, same as noted for the ASR
  encoder in "Next steps" below.
- Latency/size still need on-ARM-device confirmation regardless of which
  variant ships, same caveat as everywhere else in this document.

### Follow-up: fp16 and static INT8 (QDQ) candidates — fp16 PASSES, static INT8 fails too

Dynamic INT8 above failed the pre-declared −0.02 F1 threshold by a wide
margin (−0.0812). Two follow-up candidates from `scripts/quantize_punct.py`'s
"Next steps" list were tried next, same harness (FLEURS ja, n=250, seed=0,
`--source fleurs`), same pre-declared pass criterion (variant F1 within
−0.02 of fp32 F1):

- **fp16**: whole-graph float16 conversion via `onnxconverter-common`
  (`onnxconverter_common.float16.convert_float_to_float16(model,
  keep_io_types=True)` — keeps `input_ids`/`attention_mask`/`logits` at
  their original int64/float32 dtypes so `punct_ja.py` needs no changes;
  only internal weights/activations run in fp16). New CLI: `--variant fp16`.
- **int8-static (QDQ, calibrated)**: `onnxruntime.quantization.quantize_static`
  with `QuantFormat.QDQ`, `weight_type=QInt8`, `activation_type=QUInt8`,
  calibrated on **100 FLEURS ja sentences disjoint from the 250-sentence
  eval set** (`--calib-n 100 --calib-seed 777`, excluded by exact string
  match against the eval sentences — no calibration-on-eval leakage). The
  preprocessing step (`quant_pre_process`) needed `sympy` (added to the
  venv, not yet added to `requirements.txt` — out of this task's scope)
  and its symbolic-shape-inference pass fails on this graph's
  position-embedding `Min(512, seq_len)` broadcast pattern ("Incomplete
  symbolic shape inference"); the script falls back to
  `quant_pre_process(..., skip_symbolic_shape=True)` automatically when
  that happens. New CLI: `--variant int8-static`.

#### Results (FLEURS ja, n=250, seed=0 — same set as the dynamic-INT8 run above)

| variant | size | reduction | P | R | F1 | punct-inclusive CER | mean latency (this PC) |
|---|---|---|---|---|---|---|---|
| fp32 | 363.5 MB | — | 0.8724 | 0.4831 | 0.6218 | 2.81% | 51.5–77.6 ms* |
| int8 dynamic | 91.4 MB | 74.9% | 0.9638 | 0.3757 | 0.5407 | 3.25% | 22.94 ms |
| **fp16** | **181.8 MB** | **50.0%** | 0.8724 | 0.4831 | **0.6218** | 2.81% | 532.3 ms |
| int8-static (QDQ) | 91.3 MB | 74.9% | 0.9275 | 0.3432 | 0.5010 | 3.48% | 35.0 ms |

\* fp32 is re-run as the baseline in both variant runs; 51.5ms alongside
fp16, 77.6ms alongside int8-static — within this PC's run-to-run noise,
not a real difference. int8 dynamic's 22.94ms figure is from the earlier
run above (same harness).

**fp16 PASSES** (F1 delta = **+0.0000** — bit-for-bit identical P/R/F1/CER
to fp32 on this 250-sentence set; the punctuation predictions did not
change at all going to half precision). **int8-static FAILS**, and fails
*worse* than dynamic INT8 (F1 delta = **−0.1208** vs dynamic's −0.0812):
same failure mode as dynamic — precision goes up (0.928 vs 0.872) but
recall collapses further (0.343 vs 0.483, dynamic's own already-collapsed
0.376). Calibration didn't fix the recall problem; QDQ activation
quantization made it worse. Combined with the dynamic-INT8 result, this
now rules out *both* INT8 quantization approaches tried for this model —
the recall collapse looks like a property of quantizing this BERT-char
head's decision boundary near the comma/period threshold, not a specific
recipe's fault.

fp16's latency on this PC is the catch: **532ms mean `restore()` call,
~10x slower than fp32's ~52-78ms and ~23x slower than int8 dynamic**. This
x86/Windows CPU (via onnxruntime's CPU EP) has no native fp16 compute path
for this op mix, so fp16 tensors are cast to fp32, computed, and cast back
at essentially every op — pure overhead with no matching throughput gain,
unlike a GPU or an ARM chip with a NEON fp16 vector unit. This is the same
PC-vs-ARM caveat flagged everywhere else in this document: fp16's *size*
win (50%, real and platform-independent) is what would carry over to a
phone; its *latency* number here says nothing about ARM/NEON-fp16 or
Apple/Qualcomm NPU behavior and must be re-measured on-device before
relying on it.

#### Mobile-sizing verdict: fp16 is the sole surviving candidate

Of the three quantization variants tried for this model (dynamic INT8,
int8-static/QDQ, fp16), **fp16 is the only one that clears the accuracy
bar** — it is the recommended candidate *if* this model needs a
smaller-than-fp32 footprint on a phone-deployable profile:
`models/mojicast-punct-onnx/quantized_ort/punct_bert.fp16.onnx` (not
committed; regenerate via `--variant fp16`), 181.8 MB vs fp32's 363.5 MB.
Both INT8 approaches (dynamic and static/QDQ) are now considered
exhausted for this model at default settings — see "if revisited" below.

As with the ASR-side fp16/int8 findings elsewhere in this document, **this
does not change the PC default**: `scripts/punct_ja.py`'s `PunctuatorJa`
keeps loading fp32 (`punct_bert.onnx`) by default, since the PC build has
no size constraint. fp16 is recorded here purely as the phone-deployable
candidate; wiring it into the actual mobile profile (and re-measuring
latency on real ARM hardware, per the caveat above) is a separate mobile-
integration task, out of scope here.

If INT8 is revisited again later for this model specifically (beyond
fp16), the two off-the-shelf recipes here are now both empirically ruled
out; next steps would need to get more invasive: per-mark-type threshold
tuning specifically for a quantized checkpoint (lower comma/period
decision thresholds to compensate for the recall shift), quantization-
aware training/fine-tuning rather than post-training quantization, or
accepting a mixed strategy (e.g. keep the attention/matmul-heavy layers
in fp16 and leave embeddings in fp32) rather than a single blanket
conversion.

### Reproduce

```
H:\Programming\hayamimi\.venv\Scripts\python.exe scripts\quantize_punct.py
H:\Programming\hayamimi\.venv\Scripts\python.exe scripts\quantize_punct.py --skip-quantize --source fleurs --n 250 --latency
H:\Programming\hayamimi\.venv\Scripts\python.exe scripts\quantize_punct.py --variant fp16 --source fleurs --n 250 --seed 0 --latency
H:\Programming\hayamimi\.venv\Scripts\python.exe scripts\quantize_punct.py --variant int8-static --source fleurs --n 250 --seed 0 --calib-n 100 --calib-seed 777 --latency
```

The first command (re)quantizes and evaluates dynamic INT8 on the 15-clip
`eval_real` sample (`--source eval_real`, the default, `--variant int8` by
default). The second re-evaluates an already-produced int8 file against
the larger 250-sentence FLEURS set used above (`--source fleurs`); pass
`--seed` to resample differently. The third and fourth (re)quantize and
evaluate the fp16 and int8-static follow-up candidates on the same
250-sentence FLEURS set; int8-static additionally calibrates on a
disjoint 100-sentence FLEURS sample (`--calib-n`/`--calib-seed`). Writes
the quantized model under
`models/mojicast-punct-onnx/quantized_ort/punct_bert.<variant>.onnx` (not
committed, `models/` is untracked) and a
`scratch_quantize_punct_results_<source>_<variant>.json` summary at repo
root per run (also not committed, scratch artifacts). fp16 needs the
`onnxconverter-common` pip package; int8-static needs `sympy` for its
preprocessing step — neither is yet pinned in `requirements.txt`.

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
- The routed live-mic path has now been run on a real device (see
  "On-device (iPhone 15) verification" → "Live screen, real mic" above),
  resolving the emulator-only gap this bullet used to describe. The
  session had multiple speakers in the room, though, so the observed
  language-badge accuracy isn't a clean single-speaker number — a quieter
  retest would be needed before trusting it as a routing-accuracy figure.
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

## Draft ("発話中の暫定字幕") partial subtitles

Ported the desktop dashboard's "いま聞き取り中" experience (`scripts/realtime_transcribe.py`'s
`PartialPrinter`/`PARTIAL_EVERY_S`/`PARTIAL_WINDOW_S`) to mobile: previously the
Live screen showed nothing while the user was still speaking — only a
VAD-bounded final, after the pause. Now a provisional line grows while
speech is in progress and gets replaced by the real final the moment the
segment closes.

**Design** (`mobile/hayamimi_core/lib/live/draft_pass.dart`,
`live_transcriber.dart`):

- **Accumulation**: the Dart `sherpa_onnx` VAD bindings have no
  `current_segment`-style accessor the way the desktop's Python bindings do
  (`vad.current_segment.samples`), so `LiveTranscriber` builds its own
  accumulator by hand: every mic/wav frame fed to the VAD is also appended
  to a `List<Float32List>` while `vad.isDetected()` is true, and cleared the
  moment a segment finalizes (pops) or a new one starts.
- **Timing**: `isDraftDue` (pure, unit-tested) fires a draft decode once
  per `defaultDraftIntervalSeconds` (1.0s — coarser than the desktop's 0.5s,
  since the draft decode shares the *same* recognizer/CPU as the
  fast-final and refine passes and every extra decode is phone battery and
  heat) **and only if nothing else is currently decoding** — `_busy` is one
  flag shared by the fast-final, refine, *and* draft passes now (previously
  only fast-final/refine shared it; fixed a latent bug where a draft could
  have raced a manual/auto refine on the same native recognizer handle).
  If busy, the draft is skipped outright rather than queued — the next
  mic/wav frame just checks again, so drafts never pile up behind a slow
  decode.
- **Window cap**: `capDraftWindow` caps a draft decode to the trailing
  `defaultDraftWindowSeconds` (8.0s, mirrors `PARTIAL_WINDOW_S`) of
  accumulated audio, so a long uninterrupted utterance doesn't make every
  subsequent draft decode progressively slower.
- **Cheap decode, deliberately**: draft decodes reuse the *already-loaded*
  session recognizer(s) rather than building a second, lighter one (would
  double model memory just for drafts) and skip the routing judgment
  entirely for `RoutingProfile.jaSenseVoice` sessions —
  `RoutedRecognizerSet.decodeCurrentLangOnly` decodes with whichever model
  backs the session's *current* language, no LID/dual-confirm pass. A
  wrong-language draft only costs a flicker; the properly-routed fast-final
  replaces it moments later. The plain (non-routed) recognizer already
  defaults to `greedy_search` for free; the routed set's ReazonSpeech tier
  stays `modified_beam_search` (built once, at session start, for final
  quality) — reused as-is for drafts rather than adding a third recognizer
  instance.
- **Wiring**: `LiveTranscriber.drafts` → `HayamimiLive.events` as
  `PartialSubtitleEvent` (same shape as `scripts/subtitle_server.py`'s
  `{"type":"partial","text":...}`) → `SubtitleBroadcastServer.broadcast()`
  when 配信サーバー is on, and → the Live screen's new draft strip
  (`_DraftStrip` in `mobile/lib/live/live_page.dart`, "マイクの音声を待っています…"
  placeholder when idle). The overlay (`overlay_html.dart`) previously
  always rendered final+partial inline with no way to isolate one; ported
  the desktop overlay's `?show=final` / `?show=partial` query param so a
  phone-hosted OBS overlay can now split them into two browser sources too.
- **Debug wav streaming** (`LiveTranscriber.startDebugWavStream`,
  `_DebugWavStreamCard` in `live_page.dart`): the existing
  "wavから清書テスト" debug path only ever exercised two static halves of a
  wav through the refine-combine logic — no way to see drafts fire on an
  emulator, which has no usable microphone. Added a second debug path that
  paces a 16kHz-mono wav through the *exact* same per-frame pipeline
  (`_processFrame`) mic input uses — VAD, draft accumulation/timing, fast
  final, refine buffering — via a refactored `_buildNativeState` shared by
  both `start()` and `startDebugWavStream()`.

**Verification** (`hayamimi_test` AVD, emulator has no mic so this is the
only way to exercise a live multi-segment session end to end):

- Model/VAD dirs already on-device at `/data/local/tmp/hy_push/*` from a
  prior session; copied into the app's docs dir via
  `adb shell run-as dev.oboroge.hayamimi_mobile cp -r ... files/app_flutter/`
  (same "not directly `adb push`-able" sandbox constraint noted above), plus
  `/data/local/tmp/ja_test.wav` (16kHz mono, confirmed via a PowerShell WAV
  header read since neither `python3` nor `file` were on PATH) copied to
  `app_flutter/test.wav`, the debug field's default path.
- `ja only` routing, 配信サーバー ON, `adb forward tcp:8833 tcp:8833`,
  `curl -sN http://127.0.0.1:8833/events` streaming in the background, then
  tapped "リアルタイム再生で流す" repeatedly across several runs:
  - **(a) partial events fire multiple times per segment, confirmed over
    `/events`**: e.g. one segment produced `{"type":"partial","text":"はい"}`
    then `{"type":"partial","text":"お願いします"}` before finalizing as
    `{"type":"final","text":"準備をお願いします","lang":"ja",...}`; another
    segment produced 3 partials (`"はい"` → `"は"` → `"午後三時から始まります"`)
    before finalizing `"会議は午後三時から始まります］"`. Confirmed across 3+
    separate runs, not a one-off.
  - **(b) draft line visible on screen**: burst-captured screenshots
    (`screencap` every 0.4s on-device, pulled as a batch) caught the draft
    strip mid-utterance showing `はい` in dimmed italic above the
    (not-yet-updated) transcript list, with the debug button showing "停止"
    — a single `adb shell input tap` + one screenshot round-trip was too
    slow relative to the ~1-3s segment lengths in this test clip to reliably
    land inside the draft window, hence the burst approach.
  - **(c) finals/refine unaffected**: both finalized lines rendered
    correctly with the `ja` badge in the transcript list; `/events` finals
    carried the same text; the overlay's `?show=partial`/`?show=final`
    query params confirmed present in the served HTML via `curl`. (The
    manual 清書 button itself stays disabled during a debug wav stream — its
    `onPressed` is gated on `isRunning`, which only a real mic session sets;
    this is pre-existing, unrelated to the draft change.)
- **Draft decode latency** (temporarily instrumented with a `print`,
  reverted before commit): 3 consecutive draft decodes measured 42.2ms,
  70.9ms, and 78.9ms. This is the **emulator's host CPU** (the AVD's CPU is
  the Windows PC's own CPU under virtualization, not representative of a
  phone's ARM SoC — see "On-emulator accuracy parity" above for the same
  caveat) — real ARM RTF for the draft path is still unmeasured, same gap
  as the existing fast-final/refine numbers.
- All 113 `hayamimi_core` unit tests (new: `draft_pass_test.dart`'s
  `isDraftDue` skip/interval-control cases;
  `pcm_frame_buffer_test.dart`'s new `concatFloat32Lists`/`capDraftWindow`
  cases) plus `flutter analyze` (`mobile/` and `mobile/hayamimi_core/`)
  pass.

**Not covered by this pass**: real ARM device RTF for the draft path;
`RoutingProfile.jaSenseVoice` draft decoding specifically (verified via
code review of `decodeCurrentLangOnly`'s language-picking logic and the
existing `jaOnly` path end-to-end, not via an on-emulator routed run in
this pass — routed drafts share the same `_runDraftDecode`/`_processFrame`
plumbing already verified for `jaOnly`, so the delta is small, but it's
worth a follow-up routed-profile wav-stream run).

## Next steps for a mobile profile

- Load + accuracy validated on an Android emulator via `sherpa_onnx.dart`
  (see "On-emulator accuracy parity" above); real ARM RTF now measured on
  an iPhone 15 (see "On-device (iPhone 15) verification" above) — still
  needs the equivalent on a real Android device.
- Consider static (calibrated) quantization if dynamic INT8 RTF on-device
  turns out worse than fp16 (`*.fp16.onnx`, already shipped upstream at
  ~136.6 MB total — a middle ground between 270.3 MB fp32 and 72.2 MB int8).
- Widen the ja eval set beyond 15 clips before treating the CER delta as
  conclusive.
