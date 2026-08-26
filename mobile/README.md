# hayamimi mobile (RTF bench + live transcription)

Flutter app (Android + iOS, shared codebase) for prototyping hayamimi's
speech recognition on phones. It has three screens:

- **Bench** — runs a [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
  offline ASR model against a WAV file and reports the RTF (real-time
  factor — processing time divided by audio duration).
- **Live** — streams mic audio, cuts it into speech segments with Silero
  VAD, decodes each segment as it finalizes, and shows a running transcript.
  It can also run a "清書" (refine) second pass that re-decodes several
  segments' audio together for a cleaner result (see
  [Two-pass refine](#two-pass-refine-清書) below), and broadcast the
  transcript to other apps on the same LAN (see
  [Other app integration](#other-app-integration-broadcasting-subtitles)
  below).
- **Remote** — a thin client: streams mic audio to a hayamimi server
  running on a PC (`--input ws --serve`) and displays the subtitle events
  streamed back, so the PC's full pipeline (5-tier language routing
  included) does the actual recognition instead of the phone. See
  [Remote mode](#remote-mode-streaming-to-a-pc) below.

## Status

- Only the **Zipformer (transducer)** model family is wired up, on both
  screens. The model picker already lists SenseVoice / Paraformer / CTC as
  future entries; see `lib/bench/model_kind.dart`.
- UI is intentionally minimal. Design/UX is a later pass.
- The live screen's decode call is synchronous FFI wrapped in an `async`
  function, same as the bench screen — a long segment can cause a brief UI
  stutter while it decodes. Fine for a prototype; worth revisiting (e.g. an
  isolate) before treating this as production-ready.

## Code layout

- `lib/bench/model_kind.dart` — enum of supported ASR model families.
- `lib/bench/model_file_resolver.dart` — pure logic that picks
  encoder/decoder/joiner/tokens files out of a model directory listing
  (prefers int8 variants when both are present). Unit tested, no FFI.
- `lib/bench/bench_result.dart` — result value type + RTF calculation.
- `lib/bench/bench_runner.dart` — glues the above to the `sherpa_onnx`
  package: loads the model, decodes the WAV, times it with a `Stopwatch`.
- `lib/live/pcm_frame_buffer.dart` — pure logic: converts PCM16 mic bytes to
  normalized `Float32List` samples, and re-slices arbitrary-size mic chunks
  into the fixed-size frames Silero VAD requires. Unit tested, no FFI.
- `lib/live/speech_segment_filter.dart` — pure logic: decides whether a
  VAD-emitted speech segment is long enough to bother decoding. Unit
  tested.
- `lib/live/live_transcript_entry.dart` — one finalized transcript line
  (text + timestamp).
- `lib/live/refine_pass.dart` — pure logic for the two-pass "refine" (清書)
  feature: `RefineBuffer` (a duration-capped ring buffer of finalized
  segments awaiting a refine), `combineSegmentSamples`/
  `combineSegmentFastText` (merging a group's audio/text),
  `isRefineTextTooShort` (the "don't let a refine lose content" guard), and
  `isAutoRefineDue` (the silence-gap/max-buffered due check for auto mode).
  Mirrors the shape of the desktop pipeline's `Refiner` class
  (`scripts/realtime_transcribe.py`) with phone-tuned defaults. Unit
  tested, no FFI. See [Two-pass refine](#two-pass-refine-清書) below.
- `lib/live/live_transcriber.dart` — orchestrates the `record` mic stream,
  sherpa-onnx's `VoiceActivityDetector`, and the same `OfflineRecognizer`
  the bench uses; also owns the refine pass (`refineNow`,
  `autoRefineEnabled`, built from `refine_pass.dart`) and the debug-only
  `runDebugWavRefineTest` helper. Not unit tested itself (needs a real mic
  + native libs); built from the pure pieces above so most of its logic is
  covered indirectly.
- `lib/live/live_page.dart` — the live transcription screen UI: the
  transcript list, the "清書" (refine) section with its manual button and
  "自動清書" toggle, the "配信サーバー" (broadcast server) toggle, and (debug
  builds only) the "wavから清書テスト" card.
- `lib/server/subtitle_event.dart` — pure logic: the `partial`/`final`
  event value types and their JSON/SSE-frame encoding, wire-compatible with
  the desktop `scripts/subtitle_server.py`. Unit tested, no I/O.
- `lib/server/overlay_html.dart` — the transparent OBS overlay page, ported
  from `scripts/subtitle_server.py`'s `OVERLAY_HTML`.
- `lib/server/lan_address.dart` — picks a LAN-reachable IPv4 address to show
  the user; the selection logic (`pickLanAddress`) is pure and unit tested,
  `currentLanAddress()` wires it to `NetworkInterface.list`.
- `lib/server/subtitle_broadcast_server.dart` — the actual `dart:io`
  `HttpServer`: serves the overlay at `/` and an SSE stream at `/events`.
  Covered by an integration test that binds a real ephemeral port.
- `lib/remote/remote_event.dart` — pure logic: parses the JSON events a
  hayamimi `/ingest` WebSocket sends back (`partial`/`final`/`translation`/
  `refine`/`ready`/`error`/...) into typed values, wire-compatible with
  `scripts/subtitle_server.py`. Unit tested, no I/O.
- `lib/remote/remote_handshake.dart` — pure logic: builds the JSON
  handshake frame `/ingest` expects as the first WebSocket message. Unit
  tested.
- `lib/remote/wav_pcm_reader.dart` — pure logic: parses a 16-bit PCM `.wav`
  file's bytes (sample rate, channels, raw PCM) without touching disk. Unit
  tested. Used by the debug test-wav sender.
- `lib/remote/remote_connection_state.dart` — the connection lifecycle enum
  the Remote screen displays.
- `lib/remote/remote_transcriber.dart` — orchestrates the `record` mic
  stream and a `dart:io` `WebSocket` connection to `/ingest`: sends the
  handshake, streams PCM16 binary frames, parses incoming JSON events, and
  auto-reconnects with a fixed backoff if the connection drops. Also
  provides `sendTestWavFile`, a debug helper that streams a `.wav` file at
  real-time pace over its own one-shot connection (mirrors
  `scripts/ws_mic_client.py`) for testing without a real microphone. Not
  unit tested itself (needs a real mic/socket); built from the pure pieces
  above.
- `lib/remote/remote_page.dart` — the Remote screen UI: server URL field,
  connect/disconnect, partial text strip, finals list with language badge
  and latency, and a debug-build-only "send test wav" card.
- `lib/main.dart` — top-level app shell with the Bench/Live/Remote tab
  switcher.
- `test/` — unit tests for the pure logic (`flutter test`).

## Building on Windows (Android)

Prerequisites: Flutter SDK, Android SDK + NDK (installed automatically by
Gradle on first build if missing), a JDK.

```
flutter pub get
flutter analyze
flutter test
flutter build apk --debug
```

The debug APK lands at `build/app/outputs/flutter-apk/app-debug.apk`.

To run on a connected device/emulator instead of just building:

```
flutter run
```

## Building on macOS (iOS)

iOS builds require Xcode and can only be done on macOS. From this repo on a
Mac:

```
cd mobile
flutter pub get
open ios/Runner.xcworkspace   # first time, to set up signing in Xcode
flutter build ios --debug --no-codesign   # or `flutter run` with a device/simulator
```

`sherpa_onnx` ships a prebuilt XCFramework for iOS via the `sherpa_onnx_ios`
pub package, so no manual native build step is needed — `pod install`
happens automatically as part of `flutter build`/`flutter run`.

## Where to put the model and test audio

The app defaults to paths inside its own app-documents directory
(`getApplicationDocumentsDirectory()` from `path_provider`), but all fields
are editable text inputs, so any accessible path works:

- `<app docs dir>/model/` — must contain a zipformer transducer model:
  `encoder*.onnx`, `decoder*.onnx`, `joiner*.onnx`, `tokens.txt`. Filenames
  don't need to match exactly; see `model_file_resolver.dart` for the
  matching rules. Get one from the
  [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
  (pick an `int8` transducer variant, ideally ~100MB-class rather than the
  multi-hundred-MB ones). Used by both the Bench and Live screens.
- `<app docs dir>/test.wav` — a 16kHz mono WAV file, used by the Bench
  screen. This repo's `testdata/ja_test.wav` works for exercising the
  pipeline.
- `<app docs dir>/vad/silero_vad.onnx` — the Silero VAD model, used by the
  Live screen to detect speech segments in the mic stream. Download it from
  the sherpa-onnx project's VAD model releases, e.g.
  [`silero_vad.onnx`](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)
  (also mirrored alongside the ASR models in the
  [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)).

## Using the Live screen

1. Push (or otherwise get) the ASR model into `<app docs dir>/model/` (same
   as Bench) and `silero_vad.onnx` into `<app docs dir>/vad/` — see above.
2. Open the **Live** tab and confirm the two paths (both prefilled with the
   defaults above, both editable).
3. Tap **Start listening**. The app requests microphone permission on first
   use; grant it. Speech gets cut into segments by VAD as you talk, and each
   finalized segment is decoded and appended to the transcript list — there
   is a short "Decoding..." indicator while a segment is being transcribed.
4. Tap **Stop** to end the session; any speech still buffered in the VAD at
   that point is flushed and decoded before native resources are released.

On Android, first run will prompt for the `RECORD_AUDIO` permission
(declared in `android/app/src/main/AndroidManifest.xml`). On iOS,
`NSMicrophoneUsageDescription` is set in `ios/Runner/Info.plist`; no other
iOS-specific setup is needed beyond what's already documented above for
building on macOS.

## Two-pass refine (清書)

The Live screen can run a second decode pass — "清書" — over several
already-finalized segments' audio at once, the same "re-decode with more
context" trick the desktop pipeline's `Refiner` class uses
(`scripts/realtime_transcribe.py`; measured ~23% relative CER better on
real broadcast ja). This is re-decode only — no punctuation-restoration
model is layered on top (a separate future task).

- **清書 button** — always available while listening. Combines every
  segment buffered since the last refine, re-decodes it as one utterance,
  and appends the result to the "清書" section below the transcript.
- **自動清書 toggle** (default **off**) — when on, a refine fires
  automatically after ~4s of silence, or once ~20s of speech has buffered
  without a pause. Off by default because every refine is a full
  re-decode: firing it automatically trades battery and heat for
  convenience, so it's opt-in. The manual button always works regardless
  of this setting.
- **Memory cap** — buffered audio for the next refine is capped at 60s;
  once exceeded, the oldest buffered segment is dropped (a sliding window)
  even if no refine has happened yet, independent of when a refine
  actually fires. See `defaultRefineBufferMaxSeconds` in
  `lib/live/refine_pass.dart`.
- **Never loses content** — if a refine's re-decode comes back much
  shorter than the fast per-segment finals it's replacing (under 70% of
  their combined length), the fast finals are used instead. Mirrors the
  desktop `Refiner`'s same guard.

The refine result's format matches the Remote screen's `[清書]`-prefixed
lines in spirit, but gets its own section here rather than a prefix, since
the Live screen already separates "transcript" from other output.

### Debug: exercising refine without a mic (`wavから清書テスト`)

Debug builds (`kDebugMode`) show a card that runs
`LiveTranscriber.runDebugWavRefineTest` against a wav file: it splits the
file in half to stand in for two VAD segments, decodes each half
individually, then decodes them combined (the refine result) — so you can
see whether combining changed anything, without a live mic session or an
emulator's (nonexistent) microphone. It reuses the Model directory field
above and defaults its wav path to `<app docs dir>/test.wav` (the same
file Bench uses; push it the same way, see
[Getting files onto an Android device/emulator](#getting-files-onto-an-android-deviceemulator)).

## Other app integration: broadcasting subtitles

The Live screen can run a small in-app HTTP server so other apps on the
same Wi-Fi network — an OBS browser source, a browser tab — can subscribe
to this phone's live transcript. It speaks the exact same protocol as the
desktop app's `scripts/subtitle_server.py`, so anything that already works
against the desktop subtitle server works against the phone too:

- `GET /events` — an SSE (`text/event-stream`) feed of JSON events, one
  `final` event per finalized transcript line:
  `{"type": "final", "text": ..., "lang": "ja", "speaker": "", "latency_ms": ...}`.
  (The mobile pipeline decodes whole VAD segments rather than streaming
  incrementally, so it has no true `partial` events the way the desktop
  does — see `lib/server/subtitle_event.dart` for the `PartialSubtitleEvent`
  type kept ready for if/when that changes.)
- `GET /` — a transparent-background overlay page, drop-in usable as an OBS
  browser source, identical to the desktop server's `/`.

To use it:

1. Open the **Live** tab and start listening as usual.
2. Toggle **配信サーバー** on. The app requests no extra permission for
   this on iOS; on Android it needs `INTERNET`, already declared in the
   manifest. Once started, the card shows this phone's LAN URL, e.g.
   `http://192.168.1.42:8833/`.
3. On another device on the *same* Wi-Fi network: add that URL as an OBS
   browser source (with a transparent background), or just open it in a
   browser tab.

Notes and limitations:

- The server binds `0.0.0.0:8833` and only runs while the app is in the
  foreground with the screen on — there's no Android foreground service
  backing it, so it stops (along with mic capture) if the app is
  backgrounded or the screen locks. Fine for the "phone on a tripod next to
  the streaming PC" use case this targets; a real background service would
  be a follow-up if always-on capture is ever needed.
- No auth — anyone on the LAN who has (or guesses) the URL can watch the
  transcript. Treat it like the desktop server: fine on a trusted home/event
  network, not something to expose beyond that.
- `lang` is currently a fixed `"ja"` on every event, since the mobile
  pipeline runs one model per session rather than the desktop's
  per-utterance language routing.

### Getting files onto an Android device/emulator

The app's documents directory is app-private, so `adb push` can't write to
it directly. Push to a world-readable staging location first, then copy in
with `run-as` (requires a debug build):

```
adb push model/encoder-....onnx /data/local/tmp/hayamimi_bench/model/
adb push model/decoder-....onnx  /data/local/tmp/hayamimi_bench/model/
adb push model/joiner-....onnx   /data/local/tmp/hayamimi_bench/model/
adb push model/tokens.txt        /data/local/tmp/hayamimi_bench/model/
adb push test.wav                /data/local/tmp/hayamimi_bench/test.wav

adb shell run-as dev.oboroge.hayamimi_mobile mkdir /data/data/dev.oboroge.hayamimi_mobile/app_flutter/model
adb shell run-as dev.oboroge.hayamimi_mobile cp /data/local/tmp/hayamimi_bench/model/encoder-....onnx /data/data/dev.oboroge.hayamimi_mobile/app_flutter/model/
# ...repeat cp for decoder/joiner/tokens.txt/test.wav
```

(`adb shell run-as <pkg> <cmd>` fails cryptically if you try to read from
`/sdcard` directly on newer Android — scoped storage blocks it even for
world-readable-looking files. Staging under `/data/local/tmp` first avoids
that.)

## Remote mode: streaming to a PC

Remote mode is the opposite of Live mode: instead of running ASR on the
phone, the phone just captures mic audio and streams it to a hayamimi
server running on a PC over the `/ingest` WebSocket endpoint
(`scripts/ws_ingest.py`), and displays whatever subtitle events stream
back. This gets you the PC pipeline's full quality (multi-language routing,
refine pass, translation, speaker labels) from a phone with no model files
on it at all.

### 1. Start the server on the PC

```
python scripts/realtime_transcribe.py --input ws --serve
```

This starts two listeners:

- `--serve` (default port **8833**): the HTTP dashboard/SSE feed at
  `http://<pc>:8833/` and `http://<pc>:8833/events`, same as any other run.
- `--input ws` (default port **8766**, override with `--ws-port`): the raw
  `/ingest` WebSocket the phone connects to, at
  `ws://<pc>:8766/ingest`.

These are two different ports serving two different protocols — the Remote
tab's server URL field wants the **8766** one, not 8833.

### 2. Connect from the phone

1. Open the **Remote** tab.
2. Set **サーバーURL** to `ws://<pc address>:8766/ingest`:
   - From the Android emulator, the host PC is reachable at `10.0.2.2`
     (the field is prefilled with `ws://10.0.2.2:8766/ingest`).
   - From a real phone on the same Wi-Fi, use the PC's LAN IP instead, e.g.
     `ws://192.168.1.10:8766/ingest`.
3. Tap **接続** (Connect). The app requests mic permission on first use,
   then streams mic audio to the server continuously, auto-reconnecting
   with a fixed 2s backoff if the connection drops.
4. Subtitle events appear as they arrive: an italic partial strip for
   in-progress text, and a list of finalized lines below it with a language
   badge and decode latency. The exact same events also show up on the
   PC's `/dashboard`/`/events` and OBS overlay, since they all come from
   the same `SubtitleServer` broadcast.

### Debug: sending a test wav (no real mic needed)

An emulator has no usable microphone, so the debug build shows a "テスト
wav送信" card for exercising the full pipeline without one. It streams a
pushed 16-bit PCM `.wav` file at real-time pace over its own one-shot
`/ingest` connection (the server only accepts one audio-producing client at
a time, so this is independent of — and will conflict with — an active mic
connection).

To get a test file onto an emulator's app-private storage (see
[Getting files onto an Android device/emulator](#getting-files-onto-an-android-deviceemulator)
above for why the two-step push is needed):

```
adb push testdata/ja_test.wav /data/local/tmp/ja_test.wav
adb shell run-as dev.oboroge.hayamimi_mobile cp /data/local/tmp/ja_test.wav app_flutter/ja_test.wav
```

The wav path field defaults to `<app docs dir>/ja_test.wav`. Tap
**テストwavを送信**; the button shows "送信中..." while it streams (roughly
the audio's real duration, plus a couple of seconds for the server to
finalize and refine).

### Notes and limitations

- Only one audio-producing `/ingest` client is accepted by the server at a
  time; a second connection attempt gets a `{"type": "error", ...}` event
  and is closed immediately (see `scripts/ws_ingest.py`).
- Like Live mode's broadcast server, mic streaming only runs while the app
  is foregrounded with the screen on — no Android background service backs
  it yet.
- `refine` events (the server's second-pass, cleaned-up re-decode of a
  group of finals) are shown as their own labeled `[清書]` lines rather
  than replacing the finals they summarize, since a thin client has no
  general way to "revise a line already on screen" without more UI state
  than seemed worth it for this pass.

## Notes on RTF numbers from an emulator

RTF measured on an Android emulator reflects the host PC's CPU (via
software rendering / x86_64 translation), not real phone hardware. Treat
emulator RTF as a smoke test that the pipeline works, not as a real
performance number — always confirm on a physical device before drawing
conclusions about phone feasibility.
