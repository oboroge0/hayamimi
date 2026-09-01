/// Constructs sherpa-onnx native objects on a short-lived background
/// isolate and hands the resulting handles back to the caller's isolate.
///
/// Why: `OfflineRecognizer(...)` / `VoiceActivityDetector(...)` /
/// `SpokenLanguageIdentification(...)` are *synchronous* FFI calls that
/// mmap and initialize the ONNX weights inline. On a phone that is 72 MB
/// (ja only) to 396 MB (`RoutingProfile.jaSenseVoice`'s three models) of
/// model loading, taking seconds. Called on Flutter's UI isolate — which is
/// what `HayamimiLive.start()` used to do — the whole app stops painting
/// and handling input for that entire time, which is what the "freezes at
/// startup" report on a real iPhone 15 was.
///
/// Why handing a raw pointer across isolates is safe here:
///
///  * sherpa-onnx's Dart classes are thin wrappers over a single
///    `Pointer<Opaque>` plus the pure-Dart config that produced it. The
///    package exposes public `.fromPtr(...)` constructors for exactly this
///    kind of rebuild, and the wrappers hold nothing isolate-local (no
///    `Finalizer`, no `NativeCallable`, no retained `Dart_Handle`).
///  * The pointee is plain C-heap state owned by the sherpa-onnx C API and
///    is not thread-affine. The building isolate exits the moment
///    `Isolate.run` returns, so from then on exactly one isolate — the
///    caller's — accepts waveforms into it and eventually frees it.
///  * FFI binding tables are per-isolate (see the sherpa_onnx library docs:
///    "You **must** call initBindings ... in every isolate"), so the
///    background isolate initializes its own. The calling isolate must
///    already have called `sherpa_onnx.initBindings()` too — it is the one
///    that later uses and frees the handle — which host apps do in
///    `main()`.
///
/// What still comes through here, and what does not: the Silero VAD does,
/// because it lives on the caller's isolate afterwards — it is fed one
/// 32 ms frame at a time, and a message round trip per frame would cost
/// more than the `acceptWaveform` call it replaced. The recognizers do
/// not. They are built, used and freed inside the decode worker isolate
/// (`decode_worker.dart`), which hands no pointer to anyone, so there is no
/// handoff there to justify. `RoutedRecognizerSet.build` still calls the
/// helpers below when its `loadOffIsolate` argument is left at its default,
/// which is what the debug-only `LiveTranscriber.runDebugWavRefineTest`
/// path does; the worker passes false and builds in place.
library;

import 'dart:ffi';
import 'dart:isolate';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

/// Builds an [sherpa_onnx.OfflineRecognizer] from [config] off-isolate.
Future<sherpa_onnx.OfflineRecognizer> buildOfflineRecognizerOffIsolate(
  sherpa_onnx.OfflineRecognizerConfig config,
) async {
  final address = await Isolate.run(() {
    sherpa_onnx.initBindings();
    return sherpa_onnx.OfflineRecognizer(config).ptr.address;
  }, debugName: 'hayamimi:load-offline-recognizer');
  return sherpa_onnx.OfflineRecognizer.fromPtr(
    ptr: Pointer.fromAddress(address).cast(),
    config: config,
  );
}

/// Builds a [sherpa_onnx.VoiceActivityDetector] from [config] off-isolate.
Future<sherpa_onnx.VoiceActivityDetector> buildVadOffIsolate({
  required sherpa_onnx.VadModelConfig config,
  required double bufferSizeInSeconds,
}) async {
  final address = await Isolate.run(() {
    sherpa_onnx.initBindings();
    return sherpa_onnx.VoiceActivityDetector(
      config: config,
      bufferSizeInSeconds: bufferSizeInSeconds,
    ).ptr.address;
  }, debugName: 'hayamimi:load-vad');
  return sherpa_onnx.VoiceActivityDetector.fromPtr(
    ptr: Pointer.fromAddress(address).cast(),
    config: config,
  );
}

/// Builds a [sherpa_onnx.SpokenLanguageIdentification] from [config]
/// off-isolate.
Future<sherpa_onnx.SpokenLanguageIdentification>
buildSpokenLanguageIdentificationOffIsolate(
  sherpa_onnx.SpokenLanguageIdentificationConfig config,
) async {
  final address = await Isolate.run(() {
    sherpa_onnx.initBindings();
    return sherpa_onnx.SpokenLanguageIdentification(config).ptr.address;
  }, debugName: 'hayamimi:load-lid');
  return sherpa_onnx.SpokenLanguageIdentification.fromPtr(
    ptr: Pointer.fromAddress(address).cast(),
    config: config,
  );
}
