# hayamimi_core example

Minimal proof that a third-party Flutter app can embed hayamimi's live
subtitle engine with **one dependency and a few dozen lines** — no code
copied from the `mobile/` reference app.

## Embed it in your own app

1. Add the dependency to your `pubspec.yaml` (see the parent package's
   [README](../README.md#installing) for the path/git form):

   ```yaml
   dependencies:
     hayamimi_core:
       path: path/to/hayamimi/mobile/hayamimi_core
   ```

2. Look at [`lib/main.dart`](lib/main.dart) — that's the whole integration.
   The parts that matter for embedding are:

   - `sherpa_onnx.initBindings()` once in `main()`, before `runApp`.
   - One `HayamimiLive()` instance, kept alive for the life of the subtitle
     widget.
   - `live.events.listen(...)`, switching on `PartialSubtitleEvent` (the
     "still speaking" draft line) and `FinalSubtitleEvent` (a finished,
     language-tagged line).
   - `live.start(modelDir: ..., vadModelPath: ..., routingProfile:
     RoutingProfile.jaSenseVoice, senseVoiceModelDir: ..., lidModelDir:
     ...)` to begin capturing mic audio.
   - `live.dispose()` when the widget goes away.

   Everything else in the file (the `Scaffold`/`ListView`/badge widgets) is
   ordinary Flutter UI, not part of the package's API surface.

   The file also has a `kDebugMode`-gated "stream a test wav" section at
   the bottom — that's a verification aid for running on an emulator
   (no usable microphone there), not part of the embedding surface above.

3. Platform setup your own app needs regardless of hayamimi_core:
   `RECORD_AUDIO` permission
   (`android/app/src/main/AndroidManifest.xml`, already added in this
   example) on Android, `NSMicrophoneUsageDescription`
   (`ios/Runner/Info.plist`) on iOS.

## Running this example yourself

```
flutter pub get
flutter run
```

Tap **"Download models (~396 MB)"** first — this is `downloadProfile`
(`ModelProfile.jaSenseVoice`) from `lib/setup/model_downloader.dart`,
wired up in `lib/main.dart`'s `_downloadModels`. It fetches, verifies, and
extracts everything into this app's Documents directory in the layout
`_resolveModelPaths` expects (`<app docs dir>/model/`, `/vad/`,
`/sense_voice/`, `/lid/`); safe to tap again later, since re-runs are
checksum-verified and only re-fetch what's missing.

Then tap "Start listening" and speak — draft text appears above the list
while you're mid-sentence, then a finalized, language-badged line lands in
the list once you pause.

(If you'd rather place the files yourself — a CI step, an emulator with
no network, … — see the parent package's README "Model files" section for
the manual layout and exact asset names.)
