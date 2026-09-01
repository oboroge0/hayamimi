# hayamimi_core

Reusable core of [hayamimi](https://github.com/oboroge0/hayamimi)'s mobile
speech recognition pipeline, extracted out of the `mobile/` demo app so
other Flutter apps — e.g. a smart-glasses companion app — can embed live
subtitles without pulling in that app's UI.

It gives you three independent pieces, all speaking the same
[`SubtitleEvent`](lib/server/subtitle_event.dart) wire format (the same
`partial`/`final`/`translation`/`refine` JSON shape the desktop
`scripts/subtitle_server.py` and OBS overlay use):

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

Lower-level pieces (`LiveTranscriber`, `RemoteTranscriber`,
`BenchRunner`, VAD/PCM helpers, ...) are also exported for callers who want
more control than the two facades above give — see
[`lib/hayamimi_core.dart`](lib/hayamimi_core.dart) for the full export list.

## Installing

Not yet published to pub.dev. Depend on it from a path (this repo) or git:

```yaml
dependencies:
  hayamimi_core:
    path: ../path/to/hayamimi/mobile/hayamimi_core
  # or:
  hayamimi_core:
    git:
      url: https://github.com/oboroge0/hayamimi.git
      path: mobile/hayamimi_core
```

Import it as:

```dart
import 'package:hayamimi_core/hayamimi_core.dart';
```

### Model files

`HayamimiLive` needs sherpa-onnx model files on disk (72-396 MB depending
on the profile — see "Recommended configurations" below); this package
doesn't bundle any, but ships a downloader so you don't have to hand-roll
the fetch/verify/unpack/placement yourself:

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
`/lid` — the exact layout `HayamimiLive.start` (and this snippet) expect,
so nothing needs renaming or resolving by hand. It's idempotent: call it
again (e.g. on every app start) and it re-verifies what's already on disk
by checksum and only re-fetches what's missing or corrupt, so it's safe
to call unconditionally rather than gating it on "is this the first
launch."

See [`lib/setup/model_downloader.dart`](lib/setup/model_downloader.dart)
for the full manifest (`modelManifest`) — exact asset URLs, sha256s, and
which files each profile places where — and
[`example/`](example/)'s `_downloadModels` for a guarded, tap-to-download
version of the snippet above (it doesn't fetch 396 MB without the user
asking for it).

<details>
<summary>Getting the models yourself instead</summary>

If you'd rather manage the files by hand (a CI step, a custom CDN, …),
`downloadProfile`'s manifest names the exact assets:

- ReazonSpeech ja zipformer transducer (int8): [`sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17.tar.bz2)
  — extract the `encoder`/`decoder`/`joiner` `.int8.onnx` files (this
  archive also ships fp32/fp16, which you don't need on a phone) plus
  `tokens.txt` into one directory. See `resolveZipformerTransducerFiles`
  in `lib/bench/model_file_resolver.dart` for the filename matching rules
  if you use a different sherpa-onnx model (exact names aren't required).
- [`silero_vad.onnx`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
  for VAD, only needed for `HayamimiLive` (not `HayamimiRemote`, which has
  no ASR of its own).
- For `RoutingProfile.jaSenseVoice` only: [`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2)
  (`model.int8.onnx` + `tokens.txt`) and [`sherpa-onnx-whisper-tiny.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2)
  (only its `tiny-encoder.int8.onnx`/`tiny-decoder.int8.onnx` — the LID
  probe doesn't need `tiny-tokens.txt` or the fp32 files also in that
  archive).

</details>

`sherpa_onnx` and `record` (both dependencies of this package) need their
usual platform setup — `RECORD_AUDIO` permission on Android
(`android/app/src/main/AndroidManifest.xml`), `NSMicrophoneUsageDescription`
on iOS (`ios/Runner/Info.plist`) — see the `mobile/` app in this repo for a
working example of both.

### iOS local network permission (SubtitleBroadcastServer only)

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

Android needs nothing extra beyond `INTERNET`, which is implicit for debug
builds and should be declared in your release manifest.

### App lifecycle

This package does not observe the app lifecycle: the host app owns that
policy, because only it knows whether a background session is wanted (and
whether it has arranged the background-audio entitlements to keep one
alive). The recommended default is to `stop()` on background and `start()`
again on resume, from your own `WidgetsBindingObserver`.

If you don't, the OS may take the microphone away mid-session — on
backgrounding, or when another app grabs it. When that happens the library
now emits an error event (`ErrorSubtitleEvent` on `HayamimiLive.events`, a
`LiveTranscriberException` on `LiveTranscriber.errors`, `RemoteErrorEvent`
on `HayamimiRemote.events`) and stops the session, so `isRunning` /
`isConnected` tell the truth instead of reporting a live session that is
silently receiving nothing.

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
| ja only (default) | ReazonSpeech zipformer int8 + Silero VAD | ~72 MB | ja |
| `RoutingProfile.jaSenseVoice` | + SenseVoice small int8 + whisper-tiny int8 (LID probe) | ~396 MB | ja / en / zh / ko / yue |

Notes on the choice:

- The multilingual profile loads all three models simultaneously (no LRU
  eviction); 396 MB of weights is a deliberate size-vs-coverage trade-off
  that assumes a phone with several GB of RAM. Start with ja-only unless
  you need the language switching.
- The refine ("清書") defaults shipped in this package are phone-tuned and
  were exercised as-is in the on-device session: auto-refine fires after
  **4 s** of silence or **20 s** of buffered speech, with the buffer capped
  at **60 s** (`defaultAutoRefineSilenceSeconds` /
  `defaultAutoRefineMaxBufferedSeconds` / `defaultRefineBufferMaxSeconds`
  in `lib/live/refine_pass.dart`). See "Runtime configuration and session
  control" below if your app wants different pacing — every one of these
  (plus the draft pass's own three) is both a constructor parameter and a
  mid-session-settable property now.
- Punctuation restoration for ja (the desktop pipeline's BERT-char model,
  ~182 MB as fp16) is **not** part of either profile yet — mobile
  integration is tracked in
  [#15](https://github.com/oboroge0/hayamimi/issues/15).

## Threading / known limitations

**Model loading is off the main isolate.** `HayamimiLive.start()` /
`LiveTranscriber.start()` build the sherpa-onnx recognizer(s) and the VAD on
a short-lived background isolate and hand the native handles back — see
[`lib/live/native_model_loader.dart`](lib/live/native_model_loader.dart) for
why that handoff is safe. Loading 72 MB (ja only) to 396 MB
(`RoutingProfile.jaSenseVoice`) of ONNX weights is a multi-second
*synchronous* FFI call; doing it inline on Flutter's UI isolate — which is
what this package used to do — froze the whole app for that entire time,
which is what a "freezes at startup" report on a real iPhone 15 turned out
to be.

**Per-segment decode still runs on the caller's isolate.** Once a session is
live, each VAD-bounded segment is decoded with a synchronous FFI call on
whichever isolate drives the mic stream — for a normal host app, the UI
isolate. Budget roughly **0.1–0.5 s of blocked UI per utterance** for the
fast per-segment pass on a modern phone, and noticeably longer for a refine
("清書") pass, which re-decodes a whole buffered group at once (up to 60 s of
audio by default). If your app animates continuously while transcribing,
this shows up as dropped frames. Moving the decode path onto a persistent
worker isolate is tracked upstream in the
[hayamimi issue tracker](https://github.com/oboroge0/hayamimi/issues) and is
*not* fixed by the loading change above.

## Minimal example: on-device subtitles

See [`example/`](example/) for a full, runnable Flutter app built from
nothing but this snippet's shape (pub.dev's standard `example/` layout) —
useful as a copy-pasteable starting point, since the fragment below elides
error handling and widget wiring for brevity:

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

### Text post-processing hook

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

## Runtime configuration and session control

The pacing/VAD/decoding defaults above were chosen for a phone running the
"Recommended configurations" above, but an embedding app doesn't always
match that shape — a noisy venue needs a less trigger-happy VAD, a
lecture-style app might want a longer draft window, and a host that already
has its own hotword list wants to plug it straight into the recognizer
instead of post-processing text after the fact. All of this used to be
hardcoded constants with no way in; issue
[#29](https://github.com/oboroge0/hayamimi/issues/29) opened it up.

### Pacing knobs

The draft ("発話中の暫定字幕") and refine ("清書") passes are each governed by a
handful of `default*` constants in `lib/live/draft_pass.dart` and
`lib/live/refine_pass.dart` (see "Recommended configurations" above for the
values and why they're phone-tuned). All six are now both `LiveTranscriber`/
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

```dart
final live = HayamimiLive(autoRefineSilenceSeconds: 2.0); // shorter pauses trigger refine
// ...later, mid-session:
live.draftWindowSeconds = 4.0; // shrink the draft window on an older/hotter phone
```

### Decoding method

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

### VAD sensitivity

Silero VAD (the model deciding "is this speech or silence") shipped pinned
at its own defaults, with no way to adjust for a noisy room or a soft
speaker. `VadSensitivity` exposes its four knobs — `threshold` (speech-
probability cutoff, higher = less sensitive), `minSilenceSeconds` (how long
a pause has to last before a segment finalizes), `minSpeechSeconds`
(shorter blips are discarded before decoding), and `maxSpeechSeconds` (hard
cap on one segment's length) — each defaulting to sherpa-onnx's own value,
so `VadSensitivity()` reproduces the unconfigurable behavior exactly:

```dart
await live.start(
  modelDir: modelDir,
  vadModelPath: vadPath,
  vadSensitivity: VadSensitivity(threshold: 0.65, minSilenceSeconds: 0.8),
);

// ...later, mid-session, without restarting:
await live.setVadSensitivity(VadSensitivity(threshold: 0.4));
```

`setVadSensitivity` rebuilds Silero VAD off-isolate (same background-isolate
technique `start` uses for the initial load — see "Threading / known
limitations" below) and swaps it in at the next safe point: never
mid-segment, so a speaker's in-progress utterance is never silently
truncated by the swap. If no session is running yet, the value is just
remembered for the next `start`. Calling it again before an earlier call has
finished replaces the pending target rather than queueing both, so only the
most recently requested sensitivity ever actually gets installed.

### Hotwords

`start`/`startDebugWavStream` also accept `hotwordsFile`/`hotwordsScore`,
sherpa-onnx's own recognizer-level hotword biasing, applied to the plain
path and the routed `RoutingProfile.jaSenseVoice` profile's ja tier (not
SenseVoice). Unlike the pacing knobs and VAD sensitivity, hotwords have no
runtime setter — they're compiled into the recognizer when it's built, so
changing them means a fresh `start` (or `stop` + `start`).

### Starting a new "conversation" without reloading models

Loading `RoutingProfile.jaSenseVoice`'s three models back-to-back is the
multi-second cost `native_model_loader.dart` moves off the UI isolate — not
something a host app wants to pay again just because the user tapped "new
session." `resetSession()` clears everything about the *current*
conversation instead: the refine buffer, in-progress draft state, and (for
a routed session) which language it's currently locked to, all without
touching a single loaded native model. If a decode is in flight when it's
called, it waits for that decode to finish first, so the reset can't race a
final/refine/draft write into the buffers it's about to clear. It's a no-op
(nothing cleared, no event emitted) when no session is running.

```dart
await live.resetSession(); // fresh conversation, same loaded models
```

## Minimal example: remote subtitles (PC does the recognition)

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
WebSocket).

## Broadcasting to other apps on the LAN

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

## Events

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
| `refine` | `{"type":"refine","text":...,"lang":...,"speaker":...,"latency_ms":...,"audio_s":...}` | A refine ("清書") pass completed. `audio_s` is the total duration of the buffered group that was re-decoded. |
| `error` | `{"type":"error","message":...}` | A session failure not attributable to a call the caller made — e.g. the OS revoking mic access mid-session. |
| `model_load` | `{"type":"model_load","model":...,"phase":"start"\|"done","ms":...}` | Right before and right after each native model finishes loading (`model` is `"vad"`, `"recognizer"`, or for `RoutingProfile.jaSenseVoice`: `"ja"`/`"sensevoice"`/`"lid"`), including a `setVadSensitivity` rebuild. `ms` is the elapsed build time, only set on `"done"`. |
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

## Package layout

- `lib/hayamimi_core.dart` — the public export surface; start here.
- `lib/hayamimi_live.dart` / `lib/hayamimi_remote.dart` — the `HayamimiLive`
  / `HayamimiRemote` facades described above.
- `lib/bench/` — offline WAV→RTF benchmarking (`BenchRunner`), the
  `ModelKind` enum, and `model_file_resolver.dart`'s pure file-matching
  logic (unit tested, no FFI).
- `lib/live/` — `LiveTranscriber` (mic → VAD → decode orchestration) and
  its pure building blocks: `pcm_frame_buffer.dart` (PCM16→float, frame
  re-slicing), `speech_segment_filter.dart` (is-this-segment-worth-decoding),
  `refine_pass.dart` (the two-pass "清書" buffer/merge/due-check logic —
  mirrors the desktop pipeline's `Refiner` with phone-tuned defaults),
  `draft_pass.dart` (the "発話中の暫定字幕" pacing logic), `vad_sensitivity.dart`
  (`VadSensitivity` and the VAD-swap-timing pure function
  `shouldSwapVadNow`), `model_load_event.dart` (`ModelLoadEvent`, what
  `LiveTranscriber.modelLoads` emits), and `live_transcript_entry.dart` (one
  finalized/refine/draft line, including its `audioSeconds`/`switched`
  fields).
- `lib/remote/` — `RemoteTranscriber` (mic → `/ingest` WebSocket
  orchestration, auto-reconnect, debug wav sender) and its pure pieces:
  `remote_event.dart` (JSON→typed event parsing), `remote_handshake.dart`
  (the ingest handshake frame), `wav_pcm_reader.dart` (16-bit PCM wav
  parsing), `remote_connection_state.dart` (the connection lifecycle enum).
- `lib/server/` — `SubtitleBroadcastServer` (the `dart:io` `HttpServer`),
  `subtitle_event.dart` (the `SubtitleEvent` hierarchy: partial/final/
  translation/refine/error/model_load/session_reset, all with
  wire-compatible JSON encoding — see "Events" above), and
  `overlay_html.dart` (the OBS-ready transparent overlay page).
- `lib/setup/model_downloader.dart` — `downloadProfile`/`downloadModelSource`
  (fetch → verify sha256 → extract → place) and `modelManifest` (the two
  profiles' exact sherpa-onnx release URLs, checksums, and on-disk layout).
- `test/` — unit tests for everything pure-logic above (`flutter test`),
  plus `lifecycle_test.dart`, which covers the start/connect/dispose
  lifecycle of `LiveTranscriber`/`RemoteTranscriber` against a fake
  `RecordPlatform` and a real `dart:io` WebSocket server. Their *decoding*
  paths still aren't unit tested — those need the sherpa-onnx native libs
  and a real mic — but are built from pieces that are.

## Consumers

- [`example/`](example/) — a minimal, runnable Flutter app built from
  nothing but this package's public API (`hayamimi_core.dart`'s export
  list): one dependency, one `HayamimiLive` instance, a draft line + a
  language-tagged transcript list. Start here if you're embedding this
  package into your own app.
- `mobile/` in this repo — the reference app: three tabs (Bench/Live/
  Remote) built directly on this package, useful as a fuller worked example
  than the snippets above (permission handling, error states, debug-only
  no-mic test paths, etc).
