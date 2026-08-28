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

You'll need to get model files onto the device yourself (this package
doesn't bundle any):

- A zipformer transducer ASR model (encoder/decoder/joiner/tokens.txt) from
  the [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
  — pick an `int8` variant. See `resolveZipformerTransducerFiles` in
  `lib/bench/model_file_resolver.dart` for the filename matching rules
  (exact names aren't required).
- [`silero_vad.onnx`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
  for VAD, only needed for `HayamimiLive` (not `HayamimiRemote`, which has
  no ASR of its own).

`sherpa_onnx` and `record` (both dependencies of this package) need their
usual platform setup — `RECORD_AUDIO` permission on Android
(`android/app/src/main/AndroidManifest.xml`), `NSMicrophoneUsageDescription`
on iOS (`ios/Runner/Info.plist`) — see the `mobile/` app in this repo for a
working example of both.

### Recommended configurations (measured)

Two supported profiles, both using **int8** model variants. On a real
iPhone 15, int8 ran at **RTF 0.013** (ja zipformer, `modified_beam_search`)
— about 4.8x faster than the same int8 files on a desktop x86-64 CPU, and
faster than fp32 anywhere we measured, so on ARM phones int8 is both the
smallest *and* the fastest choice; there's no speed reason to carry the
larger fp16/fp32 files. Full measurements and conditions:
[`docs/MOBILE.md`](../../docs/MOBILE.md), "On-device (iPhone 15)
verification".

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
  in `lib/live/refine_pass.dart`). They're constructor parameters if your
  app wants different pacing.
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
  mirrors the desktop pipeline's `Refiner` with phone-tuned defaults), and
  `live_transcript_entry.dart` (one finalized line).
- `lib/remote/` — `RemoteTranscriber` (mic → `/ingest` WebSocket
  orchestration, auto-reconnect, debug wav sender) and its pure pieces:
  `remote_event.dart` (JSON→typed event parsing), `remote_handshake.dart`
  (the ingest handshake frame), `wav_pcm_reader.dart` (16-bit PCM wav
  parsing), `remote_connection_state.dart` (the connection lifecycle enum).
- `lib/server/` — `SubtitleBroadcastServer` (the `dart:io` `HttpServer`),
  `subtitle_event.dart` (the `SubtitleEvent` hierarchy: partial/final/
  translation/refine/error, all with wire-compatible JSON encoding), and
  `overlay_html.dart` (the OBS-ready transparent overlay page).
- `test/` — unit tests for everything pure-logic above (`flutter test`);
  I/O-heavy classes (`LiveTranscriber`, `RemoteTranscriber`) aren't unit
  tested directly since they need a real mic/socket/native libs, but are
  built from pieces that are.

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
