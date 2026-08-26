# hayamimi mobile (RTF bench + live transcription)

Flutter app (Android + iOS, shared codebase) for prototyping hayamimi's
speech recognition on phones. It has two screens:

- **Bench** — runs a [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
  offline ASR model against a WAV file and reports the RTF (real-time
  factor — processing time divided by audio duration).
- **Live** — streams mic audio, cuts it into speech segments with Silero
  VAD, decodes each segment as it finalizes, and shows a running transcript.

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
- `lib/live/live_transcriber.dart` — orchestrates the `record` mic stream,
  sherpa-onnx's `VoiceActivityDetector`, and the same `OfflineRecognizer`
  the bench uses. Not unit tested (needs a real mic + native libs); built
  from the pure pieces above so most of its logic is covered indirectly.
- `lib/live/live_page.dart` — the live transcription screen UI.
- `lib/main.dart` — top-level app shell with a Bench/Live tab switcher.
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

## Notes on RTF numbers from an emulator

RTF measured on an Android emulator reflects the host PC's CPU (via
software rendering / x86_64 translation), not real phone hardware. Treat
emulator RTF as a smoke test that the pipeline works, not as a real
performance number — always confirm on a physical device before drawing
conclusions about phone feasibility.
