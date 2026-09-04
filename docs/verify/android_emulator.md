# Android emulator verification of `hayamimi_core` (2026-09-01)

Split out of `docs/MOBILE.md` on 2026-09-03, verbatim: this file holds the
three "Android emulator verification" runs; the quantization and sizing log
they belong to stayed in [`../design/mobile_quantization.md`](../design/mobile_quantization.md).

## Android emulator verification of the decode worker and punctuation (2026-09-01)

Two things had been built for `hayamimi_core` without ever running on
Android: the persistent decode worker isolate (issue #24) and the Dart
Japanese punctuation port (issue #15). Both had a specific reason to be
doubted there. The worker spawns an isolate that loads sherpa-onnx models
itself, which is a different code path from loading them on the isolate
that spawned it. The punctuator reaches ONNX Runtime by opening
`libonnxruntime.so` over `dart:ffi` — the library `sherpa_onnx` already
ships — and if a second copy of that library ended up in the APK, two ONNX
Runtimes in one Android process is a known crash
([sherpa-onnx#3261](https://github.com/k2-fsa/sherpa-onnx/issues/3261)).
This is the run that settled both, plus what it turned up about
segmentation that had nothing to do with Android.

### Setup

| | |
|---|---|
| Device | **Android emulator x86_64, API level 35** (`ro.product.cpu.abi=x86_64`), AVD `hayamimi_test` |
| Host | Windows 11 (10.0.26200), AMD Ryzen 5 5600 |
| Toolchain | Flutter 3.47.1 stable / Dart 3.13.1, Android SDK emulator 37.1.11.0 |
| App | the package's own `example/` app, debug build, `--target-platform android-x64` |
| ASR | ReazonSpeech ja zipformer int8 (`encoder`/`decoder`/`joiner-epoch-35-avg-1.int8.onnx` + `tokens.txt`), `numThreads` 2, `decodingMethod: 'modified_beam_search'` passed explicitly to match `scripts/asr_engine.py` rather than measure the package's `greedy_search` default |
| Punctuation | `quantized_ort/punct_bert.fp16.onnx` (181.8 MB) + `vocab.txt`. The desktop reference uses the fp32 `punct_bert.onnx` (363 MB) |
| Profile | `RoutingProfile.jaOnly`. `jaSenseVoice` was out of scope |
| Input | `testdata/multi_sentence_ja.wav` (three ja sentences, 6.26 s, 16 kHz mono) through `startDebugWavStream`, since an emulator has no usable microphone |

Model files cannot be `adb push`ed straight into
`getApplicationDocumentsDirectory()`
(`/data/user/0/<pkg>/app_flutter`); they went via `/data/local/tmp` and
`run-as <pkg> cp`, laid out as `model/`, `vad/silero_vad.onnx`, `punct/`,
`wav/`.

### What was verified

**The decode worker comes up and stays up on Android.** `Isolate.spawn`
succeeded, `sherpa_onnx.initBindings()` inside the worker succeeded, and
every model was built there and reported back over the worker's `SendPort`
as `model_load` frames — which is the proof, since a failed spawn or a
failed `initBindings()` would have thrown `DecodeWorkerException` from
`start()` and no `model_load` frame could have arrived at all. Decode
results crossed the isolate boundary for the whole run, and the worker was
shut down and rebuilt three times in one process (three sequential
`startDebugWavStream` passes) with no crash, no `DecodeWorkerDied`, and no
`ErrorSubtitleEvent`.

**Punctuation reaches ONNX Runtime through the copy `sherpa_onnx` already
loads.** The APK contains exactly one `lib/x86_64/libonnxruntime.so`
(25,000,408 bytes) — no second runtime was added.
`DynamicLibrary.open('libonnxruntime.so')` resolved, `OrtGetApiBase` was
found, and the runtime served C API version 11, reporting itself as ONNX
Runtime **1.27.1** — the same version this repo records from the Windows
host. That is the Android default branch in `lib/punct/ort_library.dart`
taken as-is, with no `libraryPath`. The real evidence is the worker
though: it built the 181.8 MB float16 model through that same path and ran
`restore()` on its own decode results.

**Refines came back punctuated, and punctuated the way the reference
punctuates.** `RefineSubtitleEvent.punctuated` was `true` and the text
carried 。 :

```
refine punctuated=true audio_s=1.564 text=東京の天気は晴れです。
refine punctuated=true audio_s=2.044 text=会議は十時からです。
refine punctuated=true audio_s=1.724 text=昨日は昨日送りました。
```

A control pass over the same audio with no punctuation model returned
`punctuated=false` and no 。 , so the marks are the punctuation model's
output and not something the recognizer produced. Against the desktop
reference run of the same file, **every 。 the desktop places, this run
places, at the same sentence ends** — so the Dart `PunctuatorJa` plus the
float16 model agrees with the Python fp32 reference on this input. What
differs between the two is upstream of punctuation, in the recognizer's
input, and is the subject of the next two sections.

### Timings

**Android emulator x86_64, API 35, host Ryzen 5 5600 — not a phone.** The
AVD's CPU is the desktop's under virtualization. The float16 punctuation
numbers are the least transferable of all, because ONNX Runtime's CPU
provider has no float16 compute path on x86 and casts to float32 and back
on every operator, which an ARM chip with a float16 vector unit does not
do.

`model_load` events, in ms:

| model | conditions | n | min | max | median |
|---|---|---:|---:|---:|---:|
| `recognizer` | ja zipformer int8 (encoder 70.9 MB + decoder + joiner), 2 threads, `modified_beam_search` | 8 | 2929.0 | 3679.1 | 3223 |
| `punct` | `punct_bert.fp16.onnx` 181.8 MB, 2 intra-op threads | 7 | 855.2 | 1535.5 | 1072 |
| `vad` | `silero_vad.onnx` 0.64 MB | 8 | 24.8 | 44.5 | 33.9 |

The first load of each in a fresh process is the slow end of its range
(cold page cache).

Finals (decode only, inside the worker) and refines (re-decode plus
`restore()`, reported as one number), in ms:

| audio_s | final | refine, punctuated | refine, no punctuation model |
|---:|---|---|---|
| 1.564 | 52.6 / 54.4 | 83.1 / 95.8 | — |
| 1.724 | 54.1 / 57.4 | 80.1 / 81.4 | — |
| 2.044 | 66.5 / 65.7 | 93.1 / 85.5 | — |
| 6.134 (merged) | 370.8 / 165.7 / 142.9 / 161.8 / 174.9 | 832.5 (cold) / 539.2 / 485.2 / 217.6 | 184.6 |

Real-time factor on the split segments is roughly 0.03x.

**Punctuation was not timed directly.** `RefineSubtitleEvent.latencyMs`
covers the re-decode and `restore()` together, so the only way to get a
number for the punctuation pass alone was to subtract the final over the
identical samples — same recognizer, same audio:

| audio_s | refine (punctuated) | final (decode only) | difference | output chars |
|---:|---:|---:|---:|---:|
| 1.564 | 83.1 | 52.6 | 30.5 | 10 |
| 2.044 | 93.1 | 66.5 | 26.6 | 9 |
| 1.724 | 80.1 | 54.1 | 26.0 | 10 |
| 6.134 | 217.6 | 184.6 (no-punct refine) | 33.0 | 10 |

So roughly **26–33 ms per ~10-character line**, warm, two intra-op threads.
This is an inference from the difference of two timers, not a measurement.
For scale, the package README records 66–90 ms mean on the Windows host
over 55-character inputs; `restore()` scales with sequence length, so the
two are not in conflict.

### Two segmentation findings, and what this change did about them

The desktop reference run of the same file
(`scripts/realtime_transcribe.py --wav testdata/multi_sentence_ja.wav
--no-realtime --mode single --lang ja --threads 4`) produces three
sentences:

```
[ja/rz] 東京の天気は晴れです。      (seg=1.8s)
[ja/rz] あしたの会議は十時からです。 (seg=2.4s)
[ja/rz] 資料は昨日送りました。      (seg=2.1s)
```

With the package's defaults at the time, the emulator produced **one**
segment of 6.134 s and **one** sentence: `資料は昨日送りました`.

**Finding 1: the VAD defaults were sherpa-onnx's, not the desktop's.**
`VadSensitivity` shipped `minSilenceSeconds = 0.5` and
`maxSpeechSeconds = 5.0`; the desktop runs 0.35 and 12.0, and its own
comment records 0.35 s as measured CER-neutral against 0.5 s on real
broadcast ja while finalizing 150 ms sooner. This file's inter-sentence
pauses fall in the 0.35–0.5 s band, so the package merged what the desktop
split. Re-running with `minSilenceSeconds: 0.35` split it into three
segments of 1.564 / 2.044 / 1.724 s and all three sentences came out,
reproduced identically across two passes in one process. **The defaults are
now 0.35 and 12.0.** Also observed on the way: the emitted segment was
6.134 s under a configured `maxSpeechSeconds` of 5.0, so that value does
not force a split the way its doc comment claimed — sherpa-onnx treats
`max_speech_duration` as a nudge, and the doc now says so.

**Finding 2: the Dart final path had no pre-roll.** Even with the VAD
aligned, two sentences came out slightly wrong: `あしたの会議は十時からです`
lost `あしたの`, and `資料は昨日送りました` became `昨日は昨日送りました`. The
emulator's segments were also 0.24–0.38 s shorter than the desktop's on the
same audio. The reason is that the desktop does not decode the VAD's
samples as-is: `AudioHistory.with_preroll` prepends up to `PREROLL_S` = 1.0
s of pre-onset audio to every segment, clamped so it never re-includes the
previous one, and `tests/test_asr_segment.py` pins that behaviour on this
very fixture (its docstring records the same `資料は` → `昨日は` failure with
pre-roll forced to 0). Silero's speech-start detection lags the true onset
— measured 198 ms behind on this fixture's first sentence — so without
pre-roll the recogniser is handed audio that begins mid-word. **The package
now does the same**, in `PrerollHistory`
(`mobile/hayamimi_core/lib/live/preroll.dart`): a bounded rolling buffer on
the caller's isolate, a new `prerollSeconds` knob defaulting to 1.0, and
the extended samples used as the final's audio so `audio_s` and the refine
buffer both reflect them.

### A third finding that is not ours to fix

**The merge is what destroyed the text, and the truncation is a model
property.** Verified on the desktop, same int8 model, no VAD at all:

| input | result |
|---|---|
| whole 6.26 s file, `modified_beam_search` | `資料は昨日送りました` |
| whole 6.26 s file, `greedy_search` | `資料は昨日送りました` |
| 0.0–1.8 s | `東京の天気は晴れです` |
| 1.8–4.2 s | `〈明日の会議は十時からです` |
| 4.2 s–end | `資料は昨日送りました` |
| 0.0–4.2 s (sentences 1+2) | `〈明日の会議は十時からです` |

Given multi-utterance audio this ReazonSpeech transducer keeps only the
**last** utterance, identically on Windows x86 and Android x86_64. So VAD
segmentation is load-bearing for correctness here, not an optimisation, and
a refine over a long group can still lose its leading sentences. The
package's existing "a merged re-decode must never lose content" guard
(`isRefineTextTooShort`) catches the case where several good finals refine
to one sentence, but not the case where a single over-long segment was
already truncated on the fast path. The desktop's further remedy — split a
suspicious result in half and retry each half (`_looks_truncated` /
`_split_retry` in `scripts/asr_engine.py`, v0.3.1) — **is not ported to
`hayamimi_core` yet.**

### What remains unverified

- **No physical ARM device.** Everything above is emulated x86_64 on a
  desktop CPU. The float16 punctuation timing in particular should not be
  extrapolated to a phone.
- **`RoutingProfile.jaSenseVoice` was not exercised.** Only `jaOnly` was in
  scope, so the routed decode worker — SenseVoice plus the whisper-tiny
  language-ID probe in one isolate — is still unrun on Android.
- **`restore()` was not timed directly**, only inferred (see above). A
  direct number needs a timer inside the worker's punctuator call site.
- **No release build, and no iOS.** `libraryPath:
  OrtLibrary.processSymbols` on iOS remains unverified, and the punctuator
  still refuses to load there by default.

## Android emulator verification, run 2 — the segmentation fixes, and a third defect (2026-09-01)

The run above ended with two changes to `hayamimi_core`'s defaults (VAD
silence, and pre-roll) and no evidence that they worked, because they were
written after the measurement. This is the re-run that checks them, on the
same AVD, the same models, and the same 6.26 s three-sentence Japanese
fixture. **Both defects are fixed.** A third, which the fixes made visible,
is not — it is described at the end, along with the change this branch makes
for it.

Same setup as run 1 in every respect (emulator `hayamimi_test`,
`ro.product.cpu.abi=x86_64`, API 35, host Windows 11 / Ryzen 5 5600, Flutter
3.47.1 / Dart 3.13.1, ReazonSpeech ja zipformer int8 with
`decodingMethod: 'modified_beam_search'` and `numThreads` 2, the fp16
punctuation model, `RoutingProfile.jaOnly`, input through
`startDebugWavStream`). Nothing was pushed to the device again: the model
files were already there from run 1, and the padded wav on the device is
byte-identical (328,418 B) to the one run 1 made. ONNX Runtime reported
itself as 1.27.1, C API 11, unchanged.

Five passes were run, twice, in two fresh processes — ten sessions in total.
**Every text and every `audio_s` is identical between the two processes.** No
`ErrorSubtitleEvent`, no `DecodeWorkerDied`, no crash.

| pass | `prerollSeconds` | `vadSensitivity` | what it is for |
|---|---|---|---|
| `r1-defaults` | 1.0 (default) | package default (0.35 / 12.0) | the shipped configuration |
| `r2-preroll0` | **0** | package default | control: turn pre-roll off only |
| `r3-closing` | 1.0 | package default | package defaults with no `refineNow()` driver at all |
| `r1-defaults-2` | 1.0 | package default | repeat of the first, for stability |
| `old-defaults` | **0** | **0.5 / 5.0** | control: reproduce run 1 on this build |

### The two defects from run 1: fixed

Run 1's output, run 2's output, and the desktop reference, on the same file:

| sentence | desktop reference | run 1 (0.5 s silence, no pre-roll) | run 2 `r1-defaults` |
|---|---|---|---|
| 1 | 東京の天気は晴れです。 | 東京の天気は晴れです。 | 東京の天気は晴れです。 |
| 2 | あしたの会議は十時からです。 | 会議は十時からです。 (`あしたの` lost) | **あしたの会議は十時からです。** |
| 3 | 資料は昨日送りました。 | 昨日は昨日送りました。 (`資料`→`昨日`) | **資料は昨日送りました。** |
| segment lengths | 1.8 / 2.4 / 2.1 s | 1.564 / 2.044 / 1.724 s | **1.762 / 2.400 / 2.144 s** |

All three sentences now match the desktop character for character, and the
segment durations land within 40 ms of the desktop's.

Each fix has its own control, so neither is credited with the other's effect.
`old-defaults` puts run 1's VAD values back on this build and reproduces run
1's single merged segment exactly — one 6.134 s segment reading
`資料は昨日送りました`, the last sentence alone. `r2-preroll0` turns off only
`prerollSeconds`, on the fixed VAD defaults, and reproduces run 1's texts and
run 1's `audio_s` values exactly (1.564 / 2.044 / 1.724 s,
`東京の天気は晴れです。` / `会議は十時からです。` / `昨日は昨日送りました。`).

The pre-roll actually prepended was 0.198 / 0.356 / 0.420 s, not the full
1.0 s, because `PrerollHistory.withPreroll` clamps at the previous segment's
end so a segment never reaches back into the one before it. That is the
documented behaviour, measured.

### The closing refine fires

`r3-closing` runs on the package's true defaults — `autoRefineEnabled` false,
which is its own default, and no `refineNow()` call anywhere. A refine still
arrives over all three finals before the awaited future completes, at
`audio_s=6.306`, which is exactly `1.762 + 2.400 + 2.144`: the whole group.
Run 1's equivalent measurement was a `refineNow()` after the await that found
`refineBufferedSeconds=0.0` and did nothing, so the closing-refine change
does what it was written to do.

### Timings

**Android emulator x86_64, API 35, host Ryzen 5 5600 — not a phone.** The
AVD's CPU is the desktop's under virtualization. The float16 punctuation
numbers are the least transferable of all, for the reason run 1 gives: ONNX
Runtime's CPU provider has no float16 compute path on x86 and casts to
float32 and back on every operator, which an ARM chip with a float16 vector
unit does not do. Two processes, ten sessions.

`model_load` events, in ms:

| model | conditions | n | min | max | median | run 1 median |
|---|---|---:|---:|---:|---:|---:|
| `recognizer` | ja zipformer int8 (encoder 70.9 MB + decoder + joiner), 2 threads, `modified_beam_search` | 10 | 3390.3 | 4563.3 | **3883.7** | 3223 |
| `punct` | `punct_bert.fp16.onnx` 181.8 MB, 2 intra-op threads | 10 | 927.4 | 1994.9 | **1140.2** | 1072 |
| `vad` | `silero_vad.onnx` 0.64 MB | 10 | 21.5 | 41.8 | **32.9** | 33.9 |

The recognizer median moved up ~660 ms against run 1. None of the three
commits under test touches model building, and the `punct`/`vad` medians are
flat, so read it as host load on a shared desktop rather than a regression.
The first load of each model in a fresh process is at the slow end of its
range.

Finals (decode only, inside the worker), in ms:

| segment `audio_s` | sentence | n | min | median | max |
|---:|---|---:|---:|---:|---:|
| 1.762 | 東京の天気は晴れです | 6 | 59.3 | **61.8** | 69.0 |
| 2.400 | あしたの会議は十時からです | 6 | 74.3 | **80.4** | 91.4 |
| 2.144 | 資料は昨日送りました | 6 | 66.1 | **71.5** | 85.6 |

Real-time factor ≈ 0.033x, unchanged from run 1. Pre-roll costs 0–9 ms per
final for 0.2–0.42 s of extra audio, which is inside the noise.

Refines over those same segments, punctuated (re-decode plus `restore()`,
reported as one number), in ms: **97–114 warm**, and **407.9 / 412.7** for
the first refine in each of the two processes. That first-refine cost is a
one-time warm-up inside the fp16 punctuation model, not a per-session one —
each process ran five sessions and only the first refine paid it.

**`restore()` is still not timed directly.** `latencyMs` covers the
re-decode and the punctuation together, so subtracting the median final over
the same audio is the only number available:

| `audio_s` | refine (warm median) | final (median) | difference | output chars |
|---:|---:|---:|---:|---:|
| 1.762 | 97.2 | 61.8 | **35.4** | 11 |
| 2.400 | 114.4 | 80.4 | **34.0** | 14 |
| 2.144 | 96.6 | 71.5 | **25.1** | 11 |

So roughly **25–35 ms per 11–14 character line**, warm, two intra-op threads
— consistent with run 1's 26–33 ms per ~10 characters. It remains an
inference from the difference of two timers.

### The third defect: the fallback refine was unpunctuated

On package defaults, `r3-closing`'s refine came back **unpunctuated**,
reproduced identically in both processes:

```
[r3-closing] refine lang= punctuated=false audio_s=6.306 latency_ms=214.19 chars=35 text=東京の天気は晴れです あしたの会議は十時からです 資料は昨日送りました
```

Three sentences the fast path had recognized correctly, joined with spaces
and no 。 anywhere.

The mechanism, measured. A refine re-decodes the group's audio as one
buffer, and this ja transducer keeps only the last utterance of
multi-utterance audio — run 1 established that on the desktop, and this run
re-established it on the device through public API alone
(`LiveTranscriber.runDebugWavRefineTest`: the whole file returns
`資料は昨日送りました`, 10 characters). So the merged re-decode came back at 10
characters against a `fastText` of 35, and `10 < 0.7 × 35` fired
`isRefineTextTooShort`, which replaced the text with the fast finals joined.
That guard is doing its job — it stopped a caption that would have lost two
of three sentences. The defect is that nothing had punctuated those finals,
so the refine went out raw.

It is not confined to the debug path. Punctuation ran in the decode worker,
before the guard runs on the caller's isolate, so any refine whose group held
two or more ja segments hit this; with `defaultAutoRefineSilenceSeconds` at
4.0 s a real microphone session groups several sentences per refine, so it
was the normal case. The harness only saw punctuated refines in the other
passes because it forced `autoRefineSilenceSeconds` to 0.3 s, one refine per
segment. Defect 1's fix is what exposed it: with run 1's single merged
segment there was one sentence in the group and nothing to fall back to, so
`old-defaults` still shows `punctuated=true`.

**The desktop pipeline does not have this hole** because its fast finals are
already punctuated — `RoutedASR.transcribe` runs `punct_ja.py` over ja finals
— so `scripts/realtime_transcribe.py`'s identical `0.7 × len(fast_joined)`
fallback yields punctuated text, and its line reads
`[refine/ja] 東京の天気は晴れです。あしたの会議は十時からです。資料は昨日送りました。`

**What this branch changed.** The decode worker now punctuates
`finalSegment` results under the same condition it punctuates refines
(routed `lang == 'ja'`, or a plain session flagged Japanese), so the joined
fast text is already punctuated and the fallback keeps its marks. The refine
reports `punctuated: true` only when every final in the group was
punctuated, and the shrink comparison strips the restored marks off both
sides rather than only the refine's. Drafts stay unpunctuated. The cost is
one `restore()` per utterance rather than per refine — 25–35 ms per line at
the sizes above, on the emulator — which `JaPunctuation(applyToFinals:
false)` declines, restoring the refine-only behaviour and its unpunctuated
fallback. `LiveTranscriptEntry`/`FinalSubtitleEvent` now carry `punctuated`
(wire: `"punctuated"`) so a consumer can see which lines it applied to.

**This fix has not been run on the emulator.** It is covered by the
package's own tests only; a third run is what would confirm it on a device.

### What remains unverified after run 2

Unchanged from run 1: no physical ARM device (everything is emulated
x86_64), `RoutingProfile.jaSenseVoice` still not exercised, `restore()`
still not timed directly, and no release build and no iOS — so
`OrtLibrary.processSymbols` on iOS remains unverified. New to this run: the
punctuated-finals fix above.

## Android emulator verification, run 3 — the punctuation fix confirmed, and two small follow-ups (2026-09-01)

Run 2 ended with a fix for its third defect (the fallback refine going out
unpunctuated) that had not itself been run on the emulator. This is that
check, on the same AVD, the same models, and the same 6.26 s three-sentence
Japanese fixture, plus a fourth pass that reproduces run 2's own control
(`applyToFinals: false`) on this build. **Verdict: fixed.** On package
defaults the closing refine over the three-sentence group is now
`punctuated=true` and carries 。 between the sentences; run 2's behaviour
is still reachable, exactly, via `JaPunctuation(applyToFinals: false)`. Two
small new issues turned up alongside the fix, both addressed by this
change — see below.

Same setup as runs 1 and 2 (emulator `hayamimi_test`,
`ro.product.cpu.abi=x86_64`, API 35, host Windows 11 26H2 / Ryzen 5 5600,
Flutter 3.47.1 / Dart 3.13.1, ReazonSpeech ja zipformer int8 with
`decodingMethod: 'modified_beam_search'` and `numThreads` 2, the fp16
punctuation model, package-default VAD (0.35 / 12.0) and pre-roll (1.0),
input through `startDebugWavStream`). Models and the padded wav were
already on the device from run 1 (328,418 B, byte-identical); nothing was
pushed again. ONNX Runtime reported itself as 1.27.1, C API 11, unchanged.

The harness ran four passes, twice, in two fresh processes — eight sessions
in total. **Every text, every `chars`, and every `audio_s` is identical
between the two processes.** No `ErrorSubtitleEvent`, no `DecodeWorkerDied`,
no crash.

| tag | `applyToFinals` | `autoRefineEnabled` | explicit `refineNow` | what it shows |
|---|---|---|---|---|
| `r1-perseg` | true (default) | true (0.3 / 0.6) | yes | one refine per segment |
| `r1-closing` | true (default) | false (package default) | no | the closing refine over the group — the fix |
| `r2-perseg-nofinalpunct` | false | true (0.3 / 0.6) | yes | finals raw, refines still punctuated |
| `r2-closing-nofinalpunct` | false | false | no | the control: reproduces run 2's `[r3-closing]` |

### The fix confirmed, side by side

Run 2, package defaults, the line this run existed to re-check:

```
[r3-closing] refine lang= punctuated=false audio_s=6.306 latency_ms=214.19 chars=35 text=東京の天気は晴れです あしたの会議は十時からです 資料は昨日送りました
```

Run 3, same configuration, same fixture, package defaults:

```
[r1-closing] final  lang= punctuated=true audio_s=1.762 latency_ms=100.068 chars=11 text=東京の天気は晴れです。
[r1-closing] final  lang= punctuated=true audio_s=2.4   latency_ms=115.527 chars=14 text=あしたの会議は十時からです。
[r1-closing] final  lang= punctuated=true audio_s=2.144 latency_ms=100.563 chars=11 text=資料は昨日送りました。
[r1-closing] refine lang= punctuated=true audio_s=6.306 latency_ms=217.18  chars=38 text=東京の天気は晴れです。 あしたの会議は十時からです。 資料は昨日送りました。
```

`audio_s=6.306` is still `1.762 + 2.4 + 2.144` (the whole group), and
`chars=38` is the three punctuated finals (11 + 14 + 11 characters) joined
with one space each — this is the run before this change's join fix below,
the shape "New defect 1" describes. The mechanism is exactly run 2's: the
merged re-decode is still `資料は昨日送りました。` (confirmed again through
public API alone, `[diag] merged_decode whole_file chars=10`), which strips
to 10 characters against a fast-joined text that strips to 35, so
`10 < 0.7 × 35` fires the guard and the text falls back to the joined
finals — now punctuated, so `isCombinedFastTextPunctuated` reports the
fallback honestly as `punctuated: true`.

Turning only `applyToFinals` off, on the same build, reproduces run 2's
line character for character:

```
[r2-closing-nofinalpunct] refine lang= punctuated=false audio_s=6.306 latency_ms=243.517 chars=35 text=東京の天気は晴れです あしたの会議は十時からです 資料は昨日送りました
```

`chars=35`, `punctuated=false`, same text as run 2's `[r3-closing]` — the
flag is the whole difference, and the escape hatch its doc comment promises
works. Per-segment refines (`r1-perseg` / `r2-perseg-nofinalpunct`) are
unaffected either way: the group is a single segment there, so the guard
never fires and both configurations end at the same refine text.

### Timings

**Android emulator x86_64, API 35, host Ryzen 5 5600 — not a phone.** Two
processes, eight sessions, `modified_beam_search`, `numThreads` 2.

`model_load` events, in ms:

| model | n | median | run 2 median | run 1 median |
|---|---:|---:|---:|---:|
| `recognizer` | 8 | **3586.2** | 3883.7 | 3223 |
| `punct` | 8 | **1317.6** | 1140.2 | 1072 |
| `vad` | 8 | **37.9** | 32.9 | 33.9 |

Every model's median moved within its own run-to-run spread across the
three runs and nothing in these commits touches model building, so this
reads as host-load noise on a shared desktop, same as run 2's recognizer
jump.

Finals (decode, plus `restore()` when `applyToFinals`), in ms:

| segment `audio_s` | `applyToFinals` | n | median | vs. run 2's final median |
|---:|---|---:|---:|---:|
| 1.762 | false | 4 | **61.7** | 61.8 |
| 2.400 | false | 4 | **77.6** | 80.4 |
| 2.144 | false | 4 | **68.8** | 71.5 |
| 1.762 | true (warm) | 2 | **109.5** | — |
| 2.400 | true | 4 | **115.4** | — |
| 2.144 | true | 4 | **117.3** | — |
| 1.762 | true, **cold** (first final of the process) | 2 | **406.4** | — |

`applyToFinals: false` reproduces run 2's final medians within 3 ms, as
expected — punctuating finals has no effect on a final that stays
unpunctuated. `applyToFinals: true` costs **+47.7 / +35.0 / +45.8 ms**: a
final goes from ~62-80 ms to ~110-117 ms warm.

Refines (re-decode + `restore()`), in ms:

| `audio_s` | chars | punctuated | n | median |
|---:|---:|---|---:|---:|
| 1.762–2.144 (per-segment) | 11–14 | true | 4 each | 97.8–114.1 |
| 6.306 (closing, `r1-closing`) | 38 | true (fallback) | 2 | **211.3** |
| 6.306 (closing, `r2-closing-nofinalpunct`) | 35 | false (fallback) | 2 | **231.2** |

Run 2's equivalent fallback samples were 214.2 / 245.9 ms at `chars=35` —
consistent. No refine in this run paid a cold-start: with `applyToFinals`
on, the first final absorbed it, and by the time the `applyToFinals: false`
passes ran the process was already warm.

**The cold-start moved from the refine path to the fast path.** Run 2 found
the fp16 punctuation model's first `restore()` in a process costs ~300 ms
more than every later one, and it landed on that process's first *refine*
(407.9 / 412.7 ms there). With finals punctuated by default, this run found
the same cost now lands on the first *final* instead (402.1 / 410.8 ms,
against a warm 100–119 ms) — same one-time cost, moved onto the first
caption line a user actually sees, because that is now whichever call
happens first in a fresh process. Net work was unchanged; only where the
latency was visible changed. See "New defect 2" (now fixed) below.

### Two follow-ups, and what this change did about them

**New defect 1 — the punctuated fallback kept the space that used to be the
only separator.** `combineSegmentFastText` (`lib/live/refine_pass.dart`)
joined every group's fast finals with `join(' ')`, mirroring the desktop
joiner (`scripts/realtime_transcribe.py`'s `" ".join(...)`) for
unpunctuated text. That was fine while the space was the only thing
separating one sentence from the next in a fallback caption; now that every
sentence already ends in 。, the space is a second, redundant separator, and
`。 ` is not how Japanese is set — the desktop's own `[refine/ja]` reference
line for this fixture has no spaces
(`東京の天気は晴れです。あしたの会議は十時からです。資料は昨日送りました。`).
Severity was cosmetic but user-visible, and the default path for any group
of two or more ja segments on this recognizer.

**Fixed in this change:** `combineSegmentFastText` now joins with `''`
when `isCombinedFastTextPunctuated` says every contributing segment was
punctuated, and keeps `' '` otherwise (the condition both functions already
compute). The guard comparison is unaffected — `isRefineTextTooShort`
compares lengths after `withoutRestoredMarks`, and dropping the redundant
spaces only makes the fast side slightly shorter, i.e. the guard slightly
less eager. Run 3's `[r1-closing]` line above (`chars=38`, with spaces) is
what this branch produced *before* the join fix; after it, the same group
now emits `東京の天気は晴れです。あしたの会議は十時からです。資料は昨日送りました。`
with no spaces, matching the desktop reference exactly.

**New defect 2 — the punctuation model's one-time warm-up moved onto the
fast path.** Described in the cold-start paragraph above: the ~300 ms first
`restore()` of a process, previously absorbed by the first refine, was now
paid by the first final — the first caption line a user sees, not a 清書
pass that arrives seconds later.

**Fixed in this change:** `loadWorkerPunctuator`
(`lib/live/decode_worker.dart`) now calls `punctuator.restore('あ')` once,
right after `PunctuatorJa.load(...)` succeeds and before the `model_load`
`punct`/`done` event is sent, discarding the result. The ~300 ms then lands
inside the `model_load model=punct` number instead — already ~0.9-1.8 s and
already understood as startup cost — rather than on the first final or
refine. Cost: `model_load punct` reports ~300 ms higher; benefit: no
400 ms spike on the first caption.

### Doc-accuracy: the punctuation cost is higher than the earlier estimate

`ja_punctuation.dart`'s `applyToFinals` doc and the README's punctuation
section previously said "~25–35 ms per ~11–14 characters", inferred in run
2 by subtracting two timers (warm refine median minus final median, over
different audio paths). This run measured it directly instead — same
audio, same code path, punctuation on vs. off:

| segment | final median, `applyToFinals: false` | final median, `applyToFinals: true` (warm) | difference |
|---:|---:|---:|---:|
| 1.762 s (11 chars out) | 61.7 | 109.5 | **+47.8** |
| 2.400 s (14 chars out) | 77.6 | 115.4 | **+37.7** |
| 2.144 s (11 chars out) | 68.8 | 117.3 | **+48.5** |

**≈38-49 ms per 11-14 character line**, warm, two intra-op threads, on this
emulator — not 25-35. `ja_punctuation.dart` and the README have been
updated to "~40-50 ms per ~11-14 characters" and now say it is a direct A/B
measurement rather than a subtraction.

### A known small waste, left alone

`decode_worker.dart` punctuates a refine's merged re-decode unconditionally,
and `live_transcriber.dart`'s shrink guard may then throw that text away
entirely in favour of `payload.fastText`. On every group refine that hits
the guard — the default outcome for ≥2 ja segments on this recognizer —
one `restore()` (~40-50 ms) is spent on text nobody sees; the worker has
`payload.fastText`/`payload.fastTextPunctuated` in hand already and could
check `isRefineTextTooShort` before punctuating, skipping the call. **Not
fixed here** — it costs only time, and the closing refine's measured
latency did not get worse (211.3 ms punctuated vs. 231.2 ms unpunctuated,
noise at n=2).

### What remains unverified after run 3

Unchanged from runs 1 and 2: no physical ARM device (everything here is
emulated x86_64, and the fp16 punctuation model is the figure most likely
to move on a real phone), `RoutingProfile.jaSenseVoice` still not
exercised (every pass here went through the plain-session
`punctuatePlainSession` branch, not the routed `lang == 'ja'` branch of
`shouldPunctuate`), `restore()` still not timed directly inside the package
(the ~38-49 ms above is a difference between two configurations, not an
instrumented number), the real-microphone grouping was not exercised (the
closing refine ran through `startDebugWavStream`, not a mic, though it is
the same code path a real `defaultAutoRefineSilenceSeconds = 4.0` pause
would take), and no release build, no iOS.

