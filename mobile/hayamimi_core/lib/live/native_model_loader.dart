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
///    that later decodes and frees — which host apps do in `main()`.
///
/// Only *loading* moves off the caller's isolate. Decoding still runs where
/// it is called from; see the "Threading / known limitations" section of
/// this package's README.
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
