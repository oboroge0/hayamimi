# hayamimi_core

Reusable core of [hayamimi](https://github.com/oboroge0/hayamimi)'s mobile
speech recognition pipeline, extracted out of the `mobile/` demo app so
other Flutter apps — e.g. a smart-glasses companion app — can embed live
subtitles without pulling in that app's UI.

It gives you three independent pieces, all speaking the same
[`SubtitleEvent`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/server/subtitle_event.dart)
wire format (the same `partial`/`final`/`translation`/`refine` JSON shape
the desktop `scripts/subtitle_server.py` and OBS overlay use):

- **`HayamimiLive`** — on-device transcription: mic → Silero VAD → a
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) offline ASR model,
  with an optional two-pass "refine" (re-decode) step.
- **`HayamimiRemote`** — a thin client that streams this device's mic to a
  hayamimi server running on a PC (`--input ws --serve`) and surfaces the
  subtitle events streamed back, so the PC's full pipeline (multi-language
  routing, refine, translation) does the actual recognition.
- **`SubtitleBroadcastServer`** — a small LAN-facing HTTP server that
  re-broadcasts either of the above as SSE + a transparent overlay page, so
  OBS or a browser on the same network can subscribe.

Lower-level pieces (`LiveTranscriber`, `RemoteTranscriber`, `BenchRunner`,
VAD/PCM helpers, ...) are also exported for callers who want more control
than the two facades above give — see
[`lib/hayamimi_core.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/hayamimi_core.dart)
for the full export list.

## Table of contents

- [Installing](#installing)
- [Embedding guide](#embedding-guide)
  - [Initialize native bindings](#initialize-native-bindings)
  - [Platform permissions](#platform-permissions)
  - [On-device subtitles: HayamimiLive](#on-device-subtitles-hayamimilive)
  - [Remote subtitles: HayamimiRemote](#remote-subtitles-hayamimiremote)
  - [Broadcasting to the LAN: SubtitleBroadcastServer](#broadcasting-to-the-lan-subtitlebroadcastserver)
  - [Runtime configuration and session control](#runtime-configuration-and-session-control)
  - [Japanese punctuation restoration](#japanese-punctuation-restoration)
  - [Threading model and error handling](#threading-model-and-error-handling)
  - [Events](#events)
- [Model placement guide](#model-placement-guide)
- [Platform status](#platform-status)
- [API reference](#api-reference)
- [Package layout](#package-layout)
- [Consumers](#consumers)

## Installing

```yaml
dependencies:
  hayamimi_core: ^0.1.0
  # or, to track this repo directly:
  # hayamimi_core:
  #   path: ../path/to/hayamimi/mobile/hayamimi_core
  # hayamimi_core:
  #   git:
  #     url: https://github.com/oboroge0/hayamimi.git
  #     path: mobile/hayamimi_core
```

Import it as:

```dart
import 'package:hayamimi_core/hayamimi_core.dart';
```

## Embedding guide

### Initialize native bindings

`sherpa_onnx` (a dependency of this package) keeps its FFI binding table
per-isolate, so every isolate that touches its native objects has to
initialize its own copy before doing anything else. Call this once, before
`runApp`, on your app's main isolate:

```dart
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  sherpa_onnx.initBindings();
  runApp(const MyApp());
}
```

This package's own background isolates (the decode worker, and the
short-lived isolate that builds the VAD) call `initBindings()` for
themselves — you don't need to do that. The one place your isolate is
still involved is the VAD: it's built off-isolate but then lives and runs
on your caller's isolate for the rest of the session, so that isolate
needs bindings initialized before `HayamimiLive.start`/
`LiveTranscriber.start` builds it. Skipping this call surfaces as a native
crash or a "failed to lookup symbol" error the first time any sherpa-onnx
object is touched, not as a friendly Dart exception — see
[`lib/live/native_model_loader.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/live/native_model_loader.dart)
for exactly which handles cross this boundary and why doing it this way is
safe.

### Platform permissions

`sherpa_onnx` and `record` (both dependencies of this package) need their
usual platform setup — `RECORD_AUDIO` permission on Android
(`android/app/src/main/AndroidManifest.xml`), `NSMicrophoneUsageDescription`
on iOS (`ios/Runner/Info.plist`) — see the
[`mobile/`](https://github.com/oboroge0/hayamimi/tree/main/mobile) app in
this repo, or this package's own
[`example/`](https://github.com/oboroge0/hayamimi/tree/main/mobile/hayamimi_core/example),
for a working example of both. `HayamimiRemote` needs the same
`RECORD_AUDIO`/`NSMicrophoneUsageDescription` setup (it streams your mic
too, just to a PC instead of decoding locally) plus `INTERNET`, which is
implicit for Android debug builds and should be declared explicitly in
your release manifest. If you use `SubtitleBroadcastServer`, see
["iOS local network permission"](#ios-local-network-permission) below —
it needs one more entitlement beyond the microphone.

### On-device subtitles: HayamimiLive

See
[`example/`](https://github.com/oboroge0/hayamimi/tree/main/mobile/hayamimi_core/example)
for a full, runnable Flutter app built from nothing but this snippet's
shape (pub.dev's standard `example/` layout) — useful as a copy-pasteable
starting point, since the fragment below elides error handling and widget
wiring for brevity. Model files aren't included below either — see
["Model placement guide"](#model-placement-guide).

```dart
import 'package:hayamimi_core/hayamimi_core.dart';

final live = HayamimiLive();

live.events.listen((event) {
  switch (event) {
    case PartialSubtitleEvent(:final text):
      print('draft: $text');
    case FinalSubtitleEvent(:final text, :final lang):
      print('final [$lang]: $text');
    case RefineSubtitleEvent(:final text):
      print('refine (清書): $text');
    default:
      break;
  }
});

await live.start(
  modelDir: '/path/to/model',        // encoder/decoder/joiner/tokens.txt
  vadModelPath: '/path/to/silero_vad.onnx',
  // Optional multilingual routing: each segment goes to ReazonSpeech (ja)
  // or SenseVoice (en/zh/ko/yue) instead of one fixed-language model.
  routingProfile: RoutingProfile.jaSenseVoice,
  senseVoiceModelDir: '/path/to/sense_voice',
  lidModelDir: '/path/to/lid',
);

// ... later, e.g. on a button tap:
await live.refineNow();   // manual "清書" re-decode pass

// ... when done:
await live.stop();
await live.dispose();
```

#### Text post-processing hook

`HayamimiLive(textTransform: ...)` (or the settable `live.textTransform`
field, changeable mid-session) lets a host app rewrite each draft/final/
refine entry's text right where it becomes a `SubtitleEvent` — before it
reaches `events` and therefore before `SubtitleBroadcastServer` or any other
consumer sees it. The callback takes `(text, lang)` and returns the text to
publish; `null` (the default) is a no-op. This is deliberately just an
insertion point, not an implementation: CJK inverse-text-normalization
(kanji numerals -> arabic digits, `scripts/itn_cjk.py` on the desktop side)
and user find/replace dictionaries are not ported to Dart yet, so a host app
that wants them today supplies its own `textTransform`.

### Remote subtitles: HayamimiRemote

An alternative to `HayamimiLive` for when the recognition should happen on
a PC instead of the phone — e.g. a low-power device, or when you want the
desktop pipeline's full multi-language routing/refine/translation stack
rather than this package's phone-sized subset:

```dart
import 'package:hayamimi_core/hayamimi_core.dart';

final remote = HayamimiRemote();

remote.events.listen((event) {
  if (event case FinalSubtitleEvent(:final text, :final lang)) {
    print('[$lang] $text');
  }
});

await remote.connect('ws://192.168.1.10:8766/ingest');

// ... when done:
await remote.disconnect();
await remote.dispose();
```

The server side is `python scripts/realtime_transcribe.py --input ws
--serve` from the main hayamimi repo — see its own docs for the two ports
involved (`--serve`'s HTTP/SSE dashboard vs. `--input ws`'s `/ingest`
WebSocket). `HayamimiRemote` needs no model files and no
`sherpa_onnx.initBindings()` call of its own — it does no local decoding.

### Broadcasting to the LAN: SubtitleBroadcastServer

Either facade's events can be mirrored out over `SubtitleBroadcastServer`,
which speaks the exact protocol `scripts/subtitle_server.py` does (so an
OBS browser source or a browser tab that already works against the desktop
app works against your embedding app too):

```dart
final broadcast = SubtitleBroadcastServer();
await broadcast.start();

live.events.listen(broadcast.broadcast); // events are already SubtitleEvent

// GET http://<lan ip>:8833/         -> transparent overlay page
// GET http://<lan ip>:8833/events   -> SSE feed of the same events
```

`SubtitleBroadcastServer({port, bindAddress, allowOrigin})` binds
[`InternetAddress.anyIPv4`](https://api.dart.dev/dart-io/InternetAddress/anyIPv4-constant.html)
(`0.0.0.0`, all interfaces) by default, deliberately, rather than loopback:
the whole point of this class is reachability from *other* devices on the
LAN — that's what makes it useful for OBS or a browser running on a
different machine. Pass a narrower `bindAddress` (e.g.
`InternetAddress.loopbackIPv4`) only if a host app wants to restrict this to
same-device consumers, which defeats the LAN-broadcast use case this class
exists for. `allowOrigin` (default `'*'`) controls the `/events` response's
`Access-Control-Allow-Origin` header, matching the desktop
`subtitle_server.py`'s permissive default so a browser-based OBS source or a
web dashboard on a different origin works without extra configuration;
narrow it if a host app needs to restrict which origins may read its
transcript.

#### iOS local network permission

`SubtitleBroadcastServer` binds `0.0.0.0` and accepts connections from
other devices on the LAN. On iOS 14+ that requires the local network
entitlement, or the very first inbound connection is silently refused with
no error your Dart code can see. Add to `ios/Runner/Info.plist`:

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Streams live subtitles to OBS or a browser on your local network.</string>
```

`NSBonjourServices` is *not* needed for this package: the server does not
advertise itself over Bonjour, so clients connect by IP and port (see
`currentLanAddress()` in `lib/server/lan_address.dart` for showing the user
which address to type). Add `NSBonjourServices` only if your own app adds mDNS
discovery on top.

Android needs nothing extra beyond `INTERNET` (see
["Platform permissions"](#platform-permissions) above).

### Runtime configuration and session control

The pacing/VAD/decoding defaults below (see
["Model placement guide"](#model-placement-guide) for the profiles they
were chosen for) were tuned for a phone, but an embedding app doesn't
always match that shape — a noisy venue needs a less trigger-happy VAD, a
lecture-style app might want a longer draft window, and a host that already
has its own hotword list wants to plug it straight into the recognizer
instead of post-processing text after the fact. All of this used to be
hardcoded constants with no way in; issue
[#29](https://github.com/oboroge0/hayamimi/issues/29) opened it up.

#### Pacing knobs

The draft ("発話中の暫定字幕") and refine ("清書") passes are each governed by a
handful of `default*` constants in `lib/live/draft_pass.dart` and
`lib/live/refine_pass.dart`. All six are now both `LiveTranscriber`/
`HayamimiLive` constructor parameters and same-named properties that can be
reassigned mid-session — a setter takes effect from the next due-check or
buffer write onward, without touching any native model, and an invalid
(non-positive or non-finite) value throws `ArgumentError` immediately rather
than silently misbehaving later:

| knob | default | runtime-settable | what it does |
|---|---|---|---|
| `draftIntervalSeconds` | 1.0s | yes | How often a draft re-decode fires while a VAD segment is still in progress. |
| `draftWindowSeconds` | 8.0s | yes | Trailing-audio window a draft decode re-processes, so a long utterance doesn't make every draft slower. |
| `minDraftAudioSeconds` | 0.25s | yes | Minimum accumulated audio before a draft decode is worth running at all. |
| `autoRefineSilenceSeconds` | 4.0s | yes | Silence gap that fires an auto-refine, when `autoRefineEnabled` is on. |
| `autoRefineMaxBufferedSeconds` | 20.0s | yes | Buffered-duration ceiling that fires an auto-refine even without a silence gap. |
| `refineBufferMaxSeconds` | 60.0s | yes | Hard cap on how much audio the refine buffer holds before it starts dropping the oldest segment. |
| `prerollSeconds` | 1.0s | yes | How much audio from *before* a speech segment's detected onset is decoded along with it. The only knob here where `0` is a valid value — it means "no pre-roll". |

```dart
final live = HayamimiLive(autoRefineSilenceSeconds: 2.0); // shorter pauses trigger refine
// ...later, mid-session:
live.draftWindowSeconds = 4.0; // shrink the draft window on an older/hotter phone
```

**Why `prerollSeconds` is there.** Silero VAD notices speech a moment after
it starts, so the samples it hands over can begin partway into the first
word — and the recognizer transcribes what it is given. On an Android
emulator this package returned `昨日は昨日送りました` for `資料は昨日送りました`,
and lost `あしたの` off the front of another sentence. Pre-roll prepends the
audio recorded just before the onset, which is what the desktop pipeline
has always done (`AudioHistory` / `PREROLL_S` in
`scripts/realtime_transcribe.py`), clamped so two consecutive segments never
both contain the gap between them. The prepended audio is part of the
segment from then on: it is what gets decoded, what `audio_s` counts, and
what the refine buffer stores. Set it to `0` to decode exactly what the VAD
delimited.

The first six knobs were exercised as-is in the on-device session recorded
in ["Platform status"](#platform-status) below — see
[`docs/MOBILE.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/MOBILE.md)
for the full measurement conditions. `prerollSeconds` came later, from the
Android emulator run described in the same section.

#### Decoding method

sherpa-onnx's offline recognizer supports more than one search algorithm
(`'greedy_search'` is fast; `'modified_beam_search'` costs more CPU for
generally better accuracy). Before this parameter existed, the plain
(single-model) path was pinned to sherpa-onnx's own `'greedy_search'`
default and the `RoutingProfile.jaSenseVoice` profile's ja tier was
hardcoded to `'modified_beam_search'` (matching desktop production) with no
way to change either. `start`/`startDebugWavStream`'s `decodingMethod`
parameter overrides whichever of those applies; leave it `null` (the
default) to keep that existing behavior unchanged. SenseVoice's own decode
doesn't use this parameter.

#### VAD sensitivity

Silero VAD (the model deciding "is this speech or silence") shipped pinned
at its own defaults, with no way to adjust for a noisy room or a soft
speaker. `VadSensitivity` exposes its four knobs — `threshold` (speech-
probability cutoff, higher = less sensitive), `minSilenceSeconds` (how long
a pause has to last before a segment finalizes), `minSpeechSeconds`
(shorter blips are discarded before decoding), and `maxSpeechSeconds`
(roughly how long one segment may run before the VAD closes it mid-speech):

```dart
await live.start(
  modelDir: modelDir,
  vadModelPath: vadPath,
  vadSensitivity: VadSensitivity(threshold: 0.65, minSilenceSeconds: 0.8),
);

// ...later, mid-session, without restarting:
await live.setVadSensitivity(VadSensitivity(threshold: 0.4));
```

**The defaults are the desktop pipeline's, not sherpa-onnx's.**
`minSilenceSeconds` is **0.35s** and `maxSpeechSeconds` is **12.0s**,
matching `scripts/realtime_transcribe.py`; `threshold` (0.5) and
`minSpeechSeconds` (0.25s) are sherpa-onnx's own, which the desktop leaves
alone too. This changed because of what sherpa-onnx's stock 0.5s silence
default did on an Android emulator: on a three-sentence Japanese recording
whose pauses fall just under half a second, it merged all three sentences
into one 6.13s segment, and this repo's ja recognizer, handed that segment,
returned only the last sentence. Two sentences disappeared with nothing in
the output to say so. At 0.35s the same audio split into three segments and
all three came out. The desktop's own comment records that 0.35s measured
no worse for accuracy than 0.5s on real broadcast Japanese while finalizing
150ms sooner.

`maxSpeechSeconds` is worth reading as a nudge rather than a guarantee: on
the same run, a session configured with 5.0s emitted a 6.134s segment.
sherpa-onnx's `max_speech_duration` influences when the VAD force-closes a
segment; it does not bound what the VAD can hand over, so don't size a
buffer or a timeout on it. Passing
`VadSensitivity(minSilenceSeconds: 0.5, maxSpeechSeconds: 5.0)` restores the
values this package shipped before.

`setVadSensitivity` rebuilds Silero VAD off-isolate (same background-isolate
technique `start` uses for the initial VAD load — see
["Threading model and error handling"](#threading-model-and-error-handling))
and swaps it in at the next safe point: never mid-segment, so a speaker's
in-progress utterance is never silently truncated by the swap. If no
session is running yet, the value is just remembered for the next `start`.
Calling it again before an earlier call has finished replaces the pending
target rather than queueing both, so only the most recently requested
sensitivity ever actually gets installed.

#### Hotwords

`start`/`startDebugWavStream` also accept `hotwordsFile`/`hotwordsScore`,
sherpa-onnx's own recognizer-level hotword biasing, applied to the plain
path and the routed `RoutingProfile.jaSenseVoice` profile's ja tier (not
SenseVoice). Unlike the pacing knobs and VAD sensitivity, hotwords have no
runtime setter — they're compiled into the recognizer when it's built, so
changing them means a fresh `start` (or `stop` + `start`).

#### Starting a new "conversation" without reloading models

Loading `RoutingProfile.jaSenseVoice`'s three models back-to-back is the
multi-second cost `native_model_loader.dart` moves off the UI isolate — not
something a host app wants to pay again just because the user tapped "new
session." `resetSession()` clears everything about the *current*
conversation instead: the refine buffer, in-progress draft state, and (for
a routed session) which language it's currently locked to, all without
touching a single loaded native model. The clear is queued behind whatever
the decode worker already has, so a segment that was still decoding when
you called it still lands in the transcript first rather than racing a
write into the buffers being emptied; the returned future completes once
the worker has confirmed it cleared its own state. It's a no-op (nothing
cleared, no event emitted) when no session is running.

```dart
await live.resetSession(); // fresh conversation, same loaded models
```

### Japanese punctuation restoration

On the desktop pipeline, the refine ("清書") pass hands its text through
`scripts/punct_ja.py` before anything sees it, so desktop captions read as
sentences. This package had no equivalent, so its refine output arrived as
an unbroken run of characters — the same words, no 、 and no 。. Issue
[#15](https://github.com/oboroge0/hayamimi/issues/15) is about closing that
gap; it is closed now, for a session that asks for it. The model files
this needs are not part of either download profile — see
["Model placement guide"](#model-placement-guide) for where to get them.

**Turning it on.** Pass a `JaPunctuation` to `start` (or
`startDebugWavStream`, which is how you see it on an emulator):

```dart
await live.start(
  modelDir: modelDir,
  vadModelPath: vadModelPath,
  punctuation: JaPunctuation(
    modelPath: '$punctDir/punct_bert.fp16.onnx',
    vocabPath: '$punctDir/vocab.txt',
    // On a desktop host, also: libraryPath: '<...>/onnxruntime.dll'.
    // See "Punctuation platform status" below for why Android needs no
    // path here.
  ),
);
```

`null` — the default — leaves it off, which is exactly what a session did
before the parameter existed.

**What changes when it is on.**

* **Refine events only.** `RefineSubtitleEvent` (and
  `LiveTranscriber.refineEntries`) come back with 、 and 。 in them, and ？
  after a recognised question ending. `FinalSubtitleEvent`,
  `PartialSubtitleEvent` and the `entries`/`drafts` streams carry the
  recognizer's text unchanged. That is where the desktop pipeline
  punctuates, and it is the pass with the context for it: a final covers one
  speech segment and a draft part of one still being spoken, so sentence
  boundaries there would fall wherever the speaker happened to pause.
  Punctuating every final would also cost a full model run per utterance, on
  a phone, for a line the next refine replaces.
* **Each affected line says so.** `punctuated` on `LiveTranscriptEntry` and
  on `RefineSubtitleEvent`, and `"punctuated"` in that event's JSON (every
  key that was already there still is), so a consumer — including one across
  the LAN — can tell text the punctuation model wrote from text the
  recognizer produced.
* **Which refines get it.** A `RoutingProfile.jaSenseVoice` session
  punctuates the refines that came back as Japanese and leaves the rest
  alone. A plain single-model session has no language tag to test, so giving
  it a Japanese punctuation model is itself the statement that its model
  transcribes Japanese: **do not pass this to a plain session with a
  non-Japanese model**, which would run a Japanese model over, say, English
  and produce nonsense rather than nothing.
* **Where it runs.** In the decode worker isolate, beside the recognizers —
  loaded after them, and reported on `modelLoads` /
  `ModelLoadSubtitleEvent` as `model: "punct"` with its measured load time.
  `restore()` is another synchronous native call, so running it on the
  caller's isolate would hand back the pause that moving decoding off it
  removed (see
  ["Threading model and error handling"](#threading-model-and-error-handling)).
* **Memory, and what a failure does.** The model is 181.8 MB for the life of
  the session, on top of up to 396 MB of recognizer weights. Failing to load
  it fails `start()` with a message naming the file, rather than starting a
  session that quietly produces unpunctuated text: on a phone a missing
  model file is a configuration mistake, not something to degrade around.
* **`textTransform` still runs last**, on the punctuated text.

One interaction is worth knowing about. A refine falls back to the fast
finals' text when the merged re-decode comes back much shorter than they
were, and that length comparison is made on the text *without* the restored
marks. They are characters nobody said, and one per clause is enough to lift
a genuinely truncated re-decode back over the threshold. When that fallback
does fire, the entry reports `punctuated: false`, because the text that
survived is the finals', which nothing punctuated.

**Using `PunctuatorJa` directly.** The class stays public, for a caller who
wants to punctuate something the live pipeline did not produce:

```dart
final punctuator = await PunctuatorJa.load(
  modelPath: '<...>/punct_bert.fp16.onnx',
  vocabPath: '<...>/vocab.txt',
);
punctuator.restore('明日の会議は午後三時から始まります資料の準備をお願いします');
// -> 明日の会議は午後三時から始まります。資料の準備をお願いします。
punctuator.dispose();   // the session is native; nothing frees it for you
```

One instance is one ONNX Runtime session and is not safe to use from more
than one isolate — which is why the live pipeline keeps its own inside the
decode worker rather than sharing this one.

#### How it reaches ONNX Runtime, and why that way

The model runs on ONNX Runtime, the inference engine `sherpa_onnx` already
bundles for speech recognition. The obvious move — adding the `onnxruntime`
or `flutter_onnxruntime` package — is the wrong one: each ships its own
`libonnxruntime.so`, and two copies of that library in one Android process
is a known crash
([sherpa-onnx#3261](https://github.com/k2-fsa/sherpa-onnx/issues/3261)).

So this package ships no runtime at all. It calls ONNX Runtime's C API
directly through `dart:ffi` (Dart's foreign-function interface, which lets
Dart call C functions in a shared library) against the library
`sherpa_onnx` has already loaded: look up `OrtGetApiBase`, ask it for a
function table, call through it. That table only ever grows at the end
across ONNX Runtime releases, so bindings written against an older version
keep working when `sherpa_onnx` bumps the runtime it bundles — this package
asks for API version 11 (ONNX Runtime 1.10's) and got 1.27.1 in testing.

The session is created from the model's *file path*, not from bytes, so the
182 MB never passes through the Dart heap. Every native object it makes —
environment, session, tensors, the strings ONNX Runtime allocates for graph
names — is freed explicitly; `dispose()` releases the rest.

Two pieces of the reference implementation had no Dart equivalent:

- **NFKC** (a Unicode normalization that folds full-width ASCII to ASCII and
  half-width katakana to full-width) is not in `dart:core`. This package
  depends on [`unorm_dart`](https://pub.dev/packages/unorm_dart) (MIT, no
  dependencies, Unicode 17.0) for it.
- **MeCab**, the Japanese morphological analyzer the reference tokenizer
  runs, has no pure-Dart port worth carrying — and it turns out not to
  matter: the pipeline splits every morpheme back into single characters a
  line later, so MeCab's only observable effect is that it drops
  whitespace. The Dart tokenizer therefore does "NFKC, drop whitespace,
  split into code points". That is not an assumption:
  `scripts/make_punct_fixture.py` compares the two on every FLEURS ja
  sentence and refuses to write its fixture if they ever disagree.

#### Punctuation platform status

| Platform | Status |
|---|---|
| Windows (host, `flutter test`) | **Verified** — parity with the Python reference on 51 recorded cases. |
| Android | **Verified on an x86_64 emulator** (API 35): `libonnxruntime.so` resolved to the single copy `sherpa_onnx` ships, `OrtGetApiBase` was found, and the runtime served C API version 11 (ONNX Runtime 1.27.1 — the same version this host records). The 181.8 MB float16 model loaded inside the decode worker and punctuated ja refines. **No physical ARM device.** See ["Measured on an Android emulator"](#measured-on-an-android-emulator). |
| iOS | **Unverified, and refused by default.** `sherpa_onnx` links ONNX Runtime as an xcframework and it has not been confirmed that `OrtGetApiBase` stays exported from the app binary, so `PunctuatorJa.load` throws rather than guess. Pass `libraryPath: OrtLibrary.processSymbols` to try it. |
| macOS / Linux | Pass `libraryPath`; nothing puts the library on a desktop host's loader search path. |

#### Running the parity test

`test/punct/punct_ja_parity_test.dart` replays
`test/fixtures/punct_ja_parity.json` — 40 sentences from the FLEURS ja
benchmark set with their punctuation stripped, plus 11 synthetic edge cases
— and asserts the Dart output equals the recorded Python output character
for character. Regenerate the fixture with
`python scripts/make_punct_fixture.py`.

It needs the float16 model and an ONNX Runtime library, and **skips with a
reason naming whichever is missing** rather than failing, so a checkout
without them stays green. On Windows it finds the library by reading the
resolved path of `sherpa_onnx_windows` out of
`.dart_tool/package_config.json` and using the `onnxruntime.dll` that
package ships; elsewhere, point `HAYAMIMI_ORT_LIBRARY` at one. A companion
test, `punct_ja_fixture_tokens_test.dart`, checks the token ids from the
same fixture and needs only the 28 KB `vocab.txt`, so tokenizer drift is
caught even without the model.

The test prints what it measured while running: **66–90 ms mean
`restore()`** over the 51 cases (55 characters on average) across six
runs, on this Windows x86 host, ONNX Runtime 1.27.1 from
`sherpa_onnx_windows`, two intra-op threads. **This says
nothing about a phone.** ONNX Runtime's CPU provider has no float16 compute
path on x86, so float16 tensors are cast to float32 and back on every
operator; an ARM chip with a float16 vector unit is a different machine
entirely. For scale only: the Python reference (`scripts/punct_ja.py`,
ONNX Runtime 1.29.0, four intra-op threads, MeCab tokenization inside the
timed call) averaged 366.3 ms on the same 51 inputs and the same model on
this host. The two measurements differ in runtime build, thread count and
what the timed call includes, so the gap has not been attributed to
anything — it is recorded here, not explained.

### Threading model and error handling

**The problem this section used to describe.** Everything sherpa-onnx does
is a *synchronous* FFI call — a direct call into a native library that runs
to completion before the next line of Dart executes. A Dart isolate is
single-threaded, so while such a call is running, nothing else on that
isolate happens, Flutter's rendering included. Loading the models that way
froze host apps for seconds at startup. Decoding that way blocked the UI
once per utterance, and for much longer on a refine ("清書") pass, which
re-decodes a whole buffered group at once (up to 60 s of audio by default).

**Where the work runs now.**

| Work | Runs on |
| --- | --- |
| Microphone capture | the `record` plugin's own platform thread; audio chunks arrive as a stream on the caller's isolate |
| Silero VAD — one `acceptWaveform` per 32 ms frame | the caller's isolate |
| Building the recognizers | the decode worker isolate, which loads them itself |
| Building the VAD | a short-lived background isolate, which hands the handle back ([`lib/live/native_model_loader.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/live/native_model_loader.dart)) |
| Every decode — per-segment finals, drafts, refine passes, and the whisper-tiny language identification a routed session runs | the decode worker isolate ([`lib/live/decode_worker.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/live/decode_worker.dart)) |
| Building the Japanese punctuation model, when the session asked for one | the decode worker isolate, after the recognizers |
| Restoring Japanese punctuation into a refine result | the decode worker isolate, right after the decode that produced it |
| Emitting `entries`/`drafts`/`refineEntries`/`decoding`/`errors`/`modelLoads`/`sessionResets` | the caller's isolate |

Each session owns one decode worker. It is created by `start`, it owns that
session's recognizer handles from first load to last free, and it is shut
down by `stop`/`dispose`. Requests reach it as messages carrying the audio;
results come back as messages carrying the text. The worker serves them one
at a time, which is the same "only one decode at a time" guarantee the old
synchronous code got for free.

**What the queue does with each kind of request.** Being asynchronous means
work can now arrive while a decode is still running, so each kind has an
explicit rule:

* **Finals** (one per completed speech segment) are always queued, never
  dropped, and emitted in the order the segments closed.
* **Drafts** (the periodic re-decode of a segment still being spoken) are
  dropped rather than queued while anything else is outstanding, and a draft
  whose result arrives after its segment already produced a final is thrown
  away instead of overwriting that final.
* **Refines** are coalesced: asking for one while another is still waiting
  its turn returns that pass's future instead of starting a second pass over
  overlapping audio. A refine claims the buffered audio at the moment it is
  actually sent, so a segment that was still decoding when you tapped 清書 is
  part of the group rather than the next one.
* `decoding` reports whether *anything* is outstanding, so it emits one
  `true`/`false` pair per burst rather than one per decode.

**Decode latency is unchanged.** The work moved; it did not shrink. A
segment still takes as long to decode as it did, and `LiveTranscriptEntry.latencyMs`
still reports that same decode time (measured inside the worker, excluding
the message round trip). What changed is that your isolate is not the one
waiting for it.

**Not measured here.** No on-device frame-timing or jank measurement was
taken for this change — the claim above is about where the work runs, not a
measured improvement in any particular app. If you need numbers for your
own app, measure them there.

**What still runs on your isolate.** One Silero VAD `acceptWaveform` call
per 32 ms frame, and the event dispatch for the streams above. The VAD stays
put deliberately: it is a small, frequent call, and a message round trip per
frame would cost more than the call it replaced.

**Error handling.** A session reports its own failures rather than throwing
across an event stream: `HayamimiLive.events` carries an `ErrorSubtitleEvent`,
`LiveTranscriber.errors` a `LiveTranscriberException`, and
`HayamimiRemote.events`/`RemoteTranscriber` a `RemoteErrorEvent`/
`RemoteTranscriberException`, for anything that goes wrong once a session
is already running. `start`/`connect` themselves still throw synchronously
for a bad call (a missing model file, an already-connected client) — see
each method's own doc comment for exactly which exception. Specific cases:

* If the decode worker isolate dies mid-session, the session stops: a
  `LiveTranscriberException` is emitted on `errors` (an `ErrorSubtitleEvent`
  on `HayamimiLive.events`), microphone capture is torn down, and the object
  is left ready to `start` again — the same handling a revoked microphone
  gets. A single decode failing inside the worker is reported the same way
  but does not stop the session; only that utterance is lost.
* `stop`/`dispose` wait for the worker to acknowledge shutdown, and kill the
  isolate if it does not answer within a few seconds. In that case the
  worker's native memory is not freed: the recognizer handles belong to that
  isolate, and freeing them from another one while it might still be inside
  a decode would be a use-after-free. Losing the allocation of a session
  that was ending anyway is the cheaper failure.
* The debug-only `LiveTranscriber.runDebugWavRefineTest` helper still
  decodes on the isolate that calls it. It is a one-shot, explicitly
  awaited call with its own short-lived models, so a worker would buy
  nothing and cost a spawn and a full model load per invocation. Use
  `startDebugWavStream` to exercise the real pipeline, worker included.

**App lifecycle.** This package does not observe the app lifecycle: the
host app owns that policy, because only it knows whether a background
session is wanted (and whether it has arranged the background-audio
entitlements to keep one alive). The recommended default is to `stop()` on
background and `start()` again on resume, from your own
`WidgetsBindingObserver`. If you don't, the OS may take the microphone away
mid-session — on backgrounding, or when another app grabs it — and when
that happens the library emits the error event described above and stops
the session, so `isRunning`/`isConnected` tell the truth instead of
reporting a live session that is silently receiving nothing.

### Events

Every `SubtitleEvent` this package emits — from `HayamimiLive.events`,
`HayamimiRemote.events`, and re-broadcast verbatim by
`SubtitleBroadcastServer` — carries a `type` discriminator and is
wire-compatible with the desktop `scripts/subtitle_server.py`, so the same
OBS browser source or web client that already works against the desktop app
works unmodified against this package. A type this package's own overlay
page (`lib/server/overlay_html.dart`) doesn't recognize is simply ignored by
its JS, so adding new event types here (`model_load`, `session_reset`)
never broke it — but a *custom* consumer written against an earlier version
of this package should still switch on `type` defensively for the same
reason.

| type | JSON shape | emitted when |
|---|---|---|
| `partial` | `{"type":"partial","text":...}` | A draft ("発話中の暫定字幕") re-decode fires while a VAD segment is still in progress. |
| `final` | `{"type":"final","text":...,"lang":...,"speaker":...,"latency_ms":...,"audio_s":...,"switched":...}` | A VAD segment finalized. `audio_s` is the segment's duration in seconds; `switched` is `true` only when this segment is the reason a `RoutingProfile.jaSenseVoice` session's language changed. `speaker` is always empty (no diarization yet). |
| `translation` | `{"type":"translation","lang":...,"text":...}` | Reserved for wire compatibility with the desktop pipeline's machine translation — not emitted by this package today. |
| `refine` | `{"type":"refine","text":...,"lang":...,"speaker":...,"latency_ms":...,"audio_s":...,"punctuated":...}` | A refine ("清書") pass completed. `audio_s` is the total duration of the buffered group that was re-decoded; `punctuated` is `true` when Japanese punctuation was restored into `text` (see ["Japanese punctuation restoration"](#japanese-punctuation-restoration)), and `false` from every producer that does not punctuate, remote sessions included. |
| `error` | `{"type":"error","message":...}` | A session failure not attributable to a call the caller made — e.g. the OS revoking mic access mid-session. |
| `model_load` | `{"type":"model_load","model":...,"phase":"start"\|"done","ms":...}` | Right before and right after each native model finishes loading (`model` is `"vad"`, `"recognizer"`, for `RoutingProfile.jaSenseVoice`: `"ja"`/`"sensevoice"`/`"lid"`, or `"punct"` for the Japanese punctuation model), including a `setVadSensitivity` rebuild. `ms` is the elapsed build time, only set on `"done"`. |
| `session_reset` | `{"type":"session_reset"}` | `resetSession()` finished clearing the current conversation's state. |

`HayamimiRemote` speaks the same table — its underlying `RemoteEvent` parser
(`lib/remote/remote_event.dart`) understands all seven types above (with the
same field names) coming back from a desktop `--input ws --serve` server,
in addition to its own connection-lifecycle frames (`ready`,
`session_start`). The desktop pipeline also emits a few event types this
package doesn't have a typed `RemoteEvent` for yet — `model_fallback`,
`warning`, `session_summary`, `recluster` — which arrive as
`RemoteUnknownEvent` on `HayamimiRemote.rawEvents` and are silently dropped
from `HayamimiRemote.events` rather than raising an error.

## Model placement guide

`HayamimiLive` needs sherpa-onnx model files on disk; this package doesn't
bundle any, but ships a downloader so you don't have to hand-roll the
fetch/verify/unpack/placement yourself. This section covers both paths:
letting `downloadProfile` do it, and placing files yourself.

### Recommended configurations (measured)

Two supported profiles, both using **int8** model variants. On a real
iPhone 15, int8 ran at **RTF 0.013** (ja zipformer, `modified_beam_search`)
— about 4.8x faster than the same int8 files on a desktop x86-64 CPU, and
faster than fp32 anywhere we measured, so on ARM phones int8 is both the
smallest *and* the fastest choice; there's no speed reason to carry the
larger fp16/fp32 files. Full measurements and conditions:
[`docs/MOBILE.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/MOBILE.md),
"On-device (iPhone 15) verification".

| profile | models on device | disk | languages |
|---|---|---|---|
| `RoutingProfile.jaOnly` (default) | ReazonSpeech zipformer int8 + Silero VAD | ~72 MB | ja |
| `RoutingProfile.jaSenseVoice` | + SenseVoice small int8 + whisper-tiny int8 (LID probe) | ~396 MB | ja / en / zh / ko / yue |

Notes on the choice:

- The multilingual profile loads all three models simultaneously (no LRU
  eviction); 396 MB of weights is a deliberate size-vs-coverage trade-off
  that assumes a phone with several GB of RAM. Start with ja-only unless
  you need the language switching.
- The refine ("清書") defaults shipped in this package are phone-tuned —
  see ["Pacing knobs"](#pacing-knobs) above for the values and how to
  change them.
- Punctuation restoration for ja (the desktop pipeline's BERT-char model,
  181.8 MB as fp16) is wired into the refine pass — see
  ["Japanese punctuation restoration"](#japanese-punctuation-restoration) —
  but it is **not** part of either download profile: the model file is
  still a local build artifact, so a host app has to put it on the device
  itself (see below).

### Directory layout each profile expects

`downloadProfile`/manual placement both target this layout under one
`<targetDir>` (e.g. your app's Documents directory) — it's what
`HayamimiLive.start`/`LiveTranscriber.start` read their `modelDir`/
`vadModelPath`/`senseVoiceModelDir`/`lidModelDir` arguments from:

| directory | needed by | files |
|---|---|---|
| `<targetDir>/model/` | every profile | ReazonSpeech ja zipformer `encoder`/`decoder`/`joiner` (`.int8.onnx`) + `tokens.txt` |
| `<targetDir>/vad/` | every profile (`HayamimiLive` only — `HayamimiRemote` has no local VAD) | `silero_vad.onnx` |
| `<targetDir>/sense_voice/` | `RoutingProfile.jaSenseVoice` only | `model.int8.onnx` + `tokens.txt` |
| `<targetDir>/lid/` | `RoutingProfile.jaSenseVoice` only | `tiny-encoder.int8.onnx` + `tiny-decoder.int8.onnx` |
| any path you choose, passed to `JaPunctuation` | Japanese punctuation, optional, any profile | `punct_bert.fp16.onnx` (181.8 MB) + `vocab.txt` (28 KB) |

Exact filenames inside `model/`/`sense_voice/`/`lid/` don't have to match
this table character for character: `resolveZipformerTransducerFiles`/
`resolveOnnxFile` in
[`lib/bench/model_file_resolver.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/bench/model_file_resolver.dart)
match by role substring (`encoder`/`decoder`/`joiner`) and prefer an
`int8` variant when both it and a full-precision one are present, so
extracting a sherpa-onnx release archive's own filenames as-is (rather than
renaming them) is sufficient.

### Automatic download

```dart
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:path_provider/path_provider.dart';

final docsDir = await getApplicationDocumentsDirectory();

await downloadProfile(
  ModelProfile.jaOnly, // or ModelProfile.jaSenseVoice
  docsDir.path,
  onProgress: (event) {
    // event.phase: skipped/downloading/verifyingDownload/extracting/done
    // event.bytesReceived / event.totalBytes for a progress bar
    // (totalBytes is null if the server didn't send Content-Length)
  },
);

final live = HayamimiLive();
await live.start(
  modelDir: '${docsDir.path}/model',
  vadModelPath: '${docsDir.path}/vad/silero_vad.onnx',
  // Only for ModelProfile.jaSenseVoice:
  // routingProfile: RoutingProfile.jaSenseVoice,
  // senseVoiceModelDir: '${docsDir.path}/sense_voice',
  // lidModelDir: '${docsDir.path}/lid',
);
```

`downloadProfile` downloads each asset from the sherpa-onnx GitHub
releases (`asr-models` tag), verifies its sha256, extracts only the
`int8` members from the archives that bundle multiple precisions, and
places everything under `<targetDir>/model`, `/vad`, `/sense_voice`,
`/lid` — the layout in the table above, exactly what this snippet expects,
so nothing needs renaming or resolving by hand. It's idempotent: call it
again (e.g. on every app start) and it re-verifies what's already on disk
by checksum and only re-fetches what's missing or corrupt, so it's safe
to call unconditionally rather than gating it on "is this the first
launch."

See
[`lib/setup/model_downloader.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/setup/model_downloader.dart)
for the full manifest (`modelManifest`) — exact asset URLs, sha256s, and
which files each profile places where — and this package's
[`example/`](https://github.com/oboroge0/hayamimi/tree/main/mobile/hayamimi_core/example)'s
`_downloadModels` for a guarded, tap-to-download version of the snippet
above (it doesn't fetch 396 MB without the user asking for it).

### Manual placement

If you'd rather manage the files by hand (a CI step, a custom CDN, …),
`downloadProfile`'s manifest names the exact assets:

- ReazonSpeech ja zipformer transducer (int8): [`sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2)
  — extract the `encoder`/`decoder`/`joiner` `.int8.onnx` files (this
  archive also ships fp32/fp16, which you don't need on a phone) plus
  `tokens.txt` into `<targetDir>/model/`.
- [`silero_vad.onnx`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
  into `<targetDir>/vad/`, only needed for `HayamimiLive` (not
  `HayamimiRemote`, which has no ASR of its own).
- For `RoutingProfile.jaSenseVoice` only: [`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2)
  (`model.int8.onnx` + `tokens.txt`) into `<targetDir>/sense_voice/`, and
  [`sherpa-onnx-whisper-tiny.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2)
  (only its `tiny-encoder.int8.onnx`/`tiny-decoder.int8.onnx` — the LID
  probe doesn't need `tiny-tokens.txt` or the fp32 files also in that
  archive) into `<targetDir>/lid/`.
- **Japanese punctuation (optional, any profile):** `punct_bert.fp16.onnx`
  and its `vocab.txt` are not on the sherpa-onnx release page — they're a
  local build artifact of `python scripts/quantize_punct.py --variant
  fp16` in the main hayamimi repo, described in
  [`docs/PUNCT_JA.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/PUNCT_JA.md).
  There is no `ModelDownloader` entry for it yet — copy both files to
  wherever you pass as `JaPunctuation(modelPath: ..., vocabPath: ...)`, no
  fixed subdirectory required.

### Storage location and licenses

Every path above is written as `<targetDir>/...`; a typical host app uses
`path_provider`'s `getApplicationDocumentsDirectory()` for `<targetDir>`,
as the snippets in this guide do — it survives app restarts, isn't
cleared under storage pressure the way a cache directory can be, and (on
Android) isn't visible to the user or other apps without their own file
picker, which matters for multi-hundred-MB files a user didn't explicitly
save. Avoid a temporary/cache directory for anything `downloadProfile`
places: the whole point of its checksum-based idempotency is that the
files stay put between launches.

Every model file above is a third-party artifact with its own license —
this package's own code is MIT (`LICENSE`), but bundling or redistributing
model weights is a separate question. See
[`THIRD_PARTY_NOTICES.md`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/THIRD_PARTY_NOTICES.md)
in this package for the publisher, license, and source of every model
`downloadProfile`/manual placement can put on a device, including the
Japanese punctuation model.

## Platform status

What has actually been run where, as of this release — not a claim about
what *should* work elsewhere. Full write-ups:
[`docs/MOBILE.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/MOBILE.md)
and
[`docs/IOS_VERIFY.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/IOS_VERIFY.md).

| Platform | ASR bench / live mic | Multi-language routing | Remote mode | Japanese punctuation |
|---|---|---|---|---|
| **iOS** (real iPhone 15) | **Verified.** Bench RTF 0.013 (ja int8, `modified_beam_search`, single run); Live tab transcribed real mic input end-to-end with no reported heat or UI stutter. | Ran live on-device with real speech switching ja/en; badge-switch accuracy was mixed because the session had multiple people talking in the room, so treat it as "it works," not a clean accuracy number — see `docs/IOS_VERIFY.md` for a cleaner single-speaker retest plan. | **Not attempted** — ruled out of scope by the repo owner for this verification session (the Mac's Wi-Fi IP was outside normal private-LAN ranges); not a known failure. | **Unverified, and refused by default** — see ["Punctuation platform status"](#punctuation-platform-status). |
| **Android** | **Emulator only.** Load/accuracy validated on an AVD via `sherpa_onnx.dart` (CER 6.19% vs. the PC's 5.50% on the same 15-clip ja set, a +0.69pp gap attributed to onnxruntime kernel differences); RTF from an emulator reflects the *host PC's* CPU under virtualization, not a phone, so it isn't a real-device number. The **decode worker isolate** is verified on an x86_64 emulator (API 35): it spawned, initialized the sherpa-onnx bindings and built every model inside itself, decoded across the isolate boundary for a whole session, and was torn down and rebuilt three times in one process without a crash. **A real Android device has not been used for this package.** | Validated via manifest batch eval on the emulator only; the live-mic path has not been run on Android at all (only on iOS — see the iOS column). `RoutingProfile.jaSenseVoice` was **not exercised** in the emulator run below, which ran `jaOnly`. | Not covered by an Android-specific verification pass. | **Verified on an x86_64 emulator** (API 35): ja refines came back `punctuated: true` with 。 at the same sentence ends as the Python reference, and a control run without a punctuation model gave `punctuated: false`. **Not run on a physical ARM device**; see ["Punctuation platform status"](#punctuation-platform-status). |
| **Windows** (desktop host) | Not a supported target platform for this package (see `pubspec.yaml`'s `platforms:`) — used only to run this repo's own `flutter analyze`/`flutter test` gates and the punctuation parity test below. | n/a | n/a | **Verified** — parity with the Python reference on 51 recorded cases (`flutter test`). |
| **macOS / Linux** | Not a supported target platform for this package. | n/a | n/a | Would need an explicit `libraryPath`; not exercised. |

### Measured on an Android emulator

**Android emulator x86_64, API 35, host Ryzen 5 5600 — not a phone.** These
are emulated x86_64 timings on a desktop CPU, so they say what the code
does, not what a handset does. The float16 punctuation figures are the
least transferable of all: ONNX Runtime's CPU provider has no float16
compute path on x86 and casts to float32 and back on every operator, which
an ARM chip with a float16 vector unit does not do. Full write-up, including
how the models were placed and what was and wasn't exercised, in
[`docs/MOBILE.md`](https://github.com/oboroge0/hayamimi/blob/main/docs/MOBILE.md).

| what | conditions | ms |
|---|---|---|
| `model_load` `recognizer` | ReazonSpeech ja zipformer int8, 2 threads, `modified_beam_search`, n=8 | 3223 (median) |
| `model_load` `punct` | `punct_bert.fp16.onnx` (181.8 MB), 2 intra-op threads, n=7 | 1072 (median) |
| `model_load` `vad` | `silero_vad.onnx` (0.64 MB), n=8 | 33.9 (median) |
| final, decode only | segments of 1.5–2.0 s | 53–66 |
| refine, re-decode + punctuation | the same 1.5–2.0 s of audio | 80–96 |
| refine, re-decode + punctuation | one merged group of 6.13 s | 218–833 |

The first load of each model in a fresh process sits at the slow end of its
range (cold page cache), and 833 ms is the first refine of a run for the
same reason. **Punctuation was not timed directly.** A refine's
`latency_ms` covers the re-decode and the punctuation pass together, so the
cost quoted for `restore()` on this run — roughly **26–33 ms per ~10
characters**, warm, two intra-op threads — is a refine minus the final over
the identical samples. That is an inference from the difference of two
timers, not a measurement.

### Known limitation: a refine over a long group can lose its leading sentences

A refine ("清書") re-decodes a group of segments as one utterance, and when
that group is long the result can come back containing only its last
sentence. On the emulator, a 6.13 s group of three Japanese sentences
refined to the third sentence alone.

This is a property of the ReazonSpeech transducer, not of this package and
not of Android. Handed the same 6.26 s file with no VAD at all, on the
Windows host, the same int8 model returns that same single sentence under
both `greedy_search` and `modified_beam_search`; split into its three
sentences, it transcribes all three. Given multi-utterance audio, this
model keeps the last utterance.

What this package does about it today, and what it doesn't:

- The VAD defaults now split on shorter pauses (see
  ["VAD sensitivity"](#vad-sensitivity)), so a group is far less likely to
  contain several sentences in the first place. Segmentation is load-bearing
  for correctness here, not just for latency.
- When a refine comes back much shorter than the fast finals it is replacing,
  the finals' combined text is emitted instead (`isRefineTextTooShort`). That
  catches the case where a group of several good finals refines to one
  sentence — but not the case where a single over-long *segment* was already
  truncated on the fast path, because then there is no better text to fall
  back to.
- The desktop pipeline goes further: it splits a suspicious result in half
  and retries each half (`_looks_truncated` / `_split_retry` in
  `scripts/asr_engine.py`, v0.3.1). **That is not ported to this package
  yet.**

## API reference

Once published, pub.dev generates a full dartdoc API reference from this
package's doc comments at
[pub.dev/documentation/hayamimi_core/latest/](https://pub.dev/documentation/hayamimi_core/latest/)
— [`lib/hayamimi_core.dart`](https://github.com/oboroge0/hayamimi/blob/main/mobile/hayamimi_core/lib/hayamimi_core.dart)
is the export list it's generated from, and a good starting point for
browsing it.

## Package layout

- `lib/hayamimi_core.dart` — the public export surface; start here.
- `lib/hayamimi_live.dart` / `lib/hayamimi_remote.dart` — the `HayamimiLive`
  / `HayamimiRemote` facades described above.
- `lib/bench/` — offline WAV→RTF benchmarking (`BenchRunner`), the
  `ModelKind` enum, and `model_file_resolver.dart`'s pure file-matching
  logic (unit tested, no FFI).
- `lib/live/` — `LiveTranscriber` (mic → VAD → decode orchestration), the
  decode worker it sends every decode to — `decode_worker.dart` (the
  isolate that owns the recognizers, and the handle it is driven through),
  `decode_session.dart` (the caller's-isolate half: request ids, dispatch,
  the futures `refineNow`/`resetSession` hand out), `decode_protocol.dart`
  (the message types and their wire encoding) and `decode_scheduler.dart`
  (the pure queue policy) — `live_vad.dart` (the `LiveVad` interface a
  session feeds, and the Silero-via-sherpa-onnx implementation of it), and
  its other pure building blocks: `pcm_frame_buffer.dart` (PCM16→float,
  frame re-slicing), `speech_segment_filter.dart`
  (is-this-segment-worth-decoding),
  `refine_pass.dart` (the two-pass "清書" buffer/merge/due-check logic —
  mirrors the desktop pipeline's `Refiner` with phone-tuned defaults),
  `draft_pass.dart` (the "発話中の暫定字幕" pacing logic), `preroll.dart`
  (`PrerollHistory`, the rolling buffer a segment's pre-onset context comes
  from), `vad_sensitivity.dart`
  (`VadSensitivity` and the VAD-swap-timing pure function
  `shouldSwapVadNow`), `model_load_event.dart` (`ModelLoadEvent`, what
  `LiveTranscriber.modelLoads` emits), `ja_punctuation.dart`
  (`JaPunctuation`, the value object `start` takes to turn Japanese
  punctuation on), and `live_transcript_entry.dart` (one finalized/refine/
  draft line, including its `audioSeconds`/`switched`/`punctuated` fields).
- `lib/remote/` — `RemoteTranscriber` (mic → `/ingest` WebSocket
  orchestration, auto-reconnect, debug wav sender) and its pure pieces:
  `remote_event.dart` (JSON→typed event parsing), `remote_handshake.dart`
  (the ingest handshake frame), `wav_pcm_reader.dart` (16-bit PCM wav
  parsing), `remote_connection_state.dart` (the connection lifecycle enum).
- `lib/server/` — `SubtitleBroadcastServer` (the `dart:io` `HttpServer`),
  `subtitle_event.dart` (the `SubtitleEvent` hierarchy: partial/final/
  translation/refine/error/model_load/session_reset, all with
  wire-compatible JSON encoding — see ["Events"](#events) above), and
  `overlay_html.dart` (the OBS-ready transparent overlay page).
- `lib/punct/` — Japanese punctuation restoration (see
  ["Japanese punctuation restoration"](#japanese-punctuation-restoration)
  above). A live session runs this inside its decode worker; `PunctuatorJa`
  is also usable on its own. `punctuator_ja.dart` is the whole public
  surface (`PunctuatorJa.restore`);
  `punct_ja_tokenizer.dart` (NFKC + character split + vocabulary) and
  `punct_ja_text.dart` (where a mark goes, the ？ heuristic, and
  `withoutRestoredMarks`, which the refine pass's length check strips with)
  are pure Dart and unit tested without a model; `punct_ort_session.dart` and
  `ort_library.dart` are the only FFI in the package, and
  `ort_bindings.dart` is a trimmed copy of ONNX Runtime's C API bindings
  (see `THIRD_PARTY_NOTICES.md`).
- `lib/setup/model_downloader.dart` — `downloadProfile`/`downloadModelSource`
  (fetch → verify sha256 → extract → place) and `modelManifest` (the two
  profiles' exact sherpa-onnx release URLs, checksums, and on-disk layout).
- `test/` — unit tests for everything pure-logic above (`flutter test`),
  plus `lifecycle_test.dart`, which covers the start/connect/dispose
  lifecycle of `LiveTranscriber`/`RemoteTranscriber` against a fake
  `RecordPlatform` and a real `dart:io` WebSocket server;
  `decode_session_test.dart`, which drives the decode queue on its own —
  finals in order, drafts dropped and discarded, refines coalesced, a reset
  behind outstanding work, a worker dying mid-session; and
  `live_transcriber_session_test.dart`, which drives a whole *live* session
  with both of its native edges stood in for (the decode worker and the
  VAD, via the `decodeWorkerFactory`/`vadFactory` constructor hooks), from
  microphone bytes to transcript line. What is still untested is the FFI
  itself: no `flutter test` run can load the sherpa-onnx native libraries or
  open a microphone, so whether a decode produces the right *text* — and
  what any of this does to frame timing — is only ever verified on a
  device.

## Consumers

- [`example/`](https://github.com/oboroge0/hayamimi/tree/main/mobile/hayamimi_core/example)
  — a minimal, runnable Flutter app built from nothing but this package's
  public API (`hayamimi_core.dart`'s export list): one dependency, one
  `HayamimiLive` instance, a draft line + a language-tagged transcript
  list. Start here if you're embedding this package into your own app.
- [`mobile/`](https://github.com/oboroge0/hayamimi/tree/main/mobile) in
  this repo — the reference app: three tabs (Bench/Live/Remote) built
  directly on this package, useful as a fuller worked example than the
  snippets above (permission handling, error states, debug-only no-mic
  test paths, etc).
