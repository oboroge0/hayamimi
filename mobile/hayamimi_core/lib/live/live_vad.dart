/// The voice-activity detector a live session feeds, as an interface, plus
/// the Silero-via-sherpa-onnx implementation of it.
///
/// Why an interface: `LiveTranscriber` reaches `isRunning == true` only
/// after it has loaded a VAD, and loading one is an FFI call that a plain
/// `flutter test` run cannot make. That left everything a *running* session
/// does — segments becoming transcript lines in order, drafts being skipped
/// and discarded, a worker dying with the microphone live, a reply landing
/// after `stop()` — reachable only on a device. The decode worker already
/// had a seam for exactly this reason; this is the other half of it, and
/// with both in place a session can be driven end to end with no native
/// library present.
///
/// The interface is deliberately the six calls `LiveTranscriber` actually
/// makes, not a mirror of sherpa-onnx's class: feed a frame, ask whether
/// speech is in progress, take finished segments, flush, free.
library;

import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import 'native_model_loader.dart';
import 'vad_sensitivity.dart';

/// How much audio the native VAD is allowed to buffer internally. Well
/// above the longest segment [VadSensitivity.maxSpeechSeconds] permits, so
/// the ring never truncates a segment that is still being spoken.
const double _vadBufferSeconds = 30;

/// A voice-activity detector driving one live session.
///
/// Stateful across [acceptWaveform] calls: the implementation is tracking
/// where the current speech segment started. That is why a session never
/// swaps one of these out mid-segment (see [shouldSwapVadNow]).
abstract class LiveVad {
  /// Exactly how many samples each [acceptWaveform] call must be given.
  /// The mic delivers chunks of whatever size the OS chooses, so the
  /// session re-slices them to this (see `PcmFrameBuffer`).
  int get frameSize;

  /// Feeds one frame of 16 kHz mono float samples.
  void acceptWaveform(Float32List frame);

  /// Whether a speech segment is in progress right now — i.e. the speaker
  /// has started and has not yet paused long enough to end it.
  bool get isSpeechDetected;

  /// Whether at least one finished segment is waiting to be taken.
  bool get hasSegment;

  /// Removes and returns the oldest finished segment's samples.
  Float32List takeSegment();

  /// Ends any segment in progress and makes it available to [takeSegment],
  /// so stopping does not silently drop what the speaker had just said.
  void flush();

  /// Releases the native handle. Called once per instance.
  void free();
}

/// Builds a [LiveVad] for a session. `LiveTranscriber` calls this on
/// [LiveTranscriber.start] and again for every
/// [LiveTranscriber.setVadSensitivity] rebuild.
typedef LiveVadFactory =
    Future<LiveVad> Function({
      required String modelPath,
      required VadSensitivity sensitivity,
      required int sampleRate,
    });

/// The production factory: Silero VAD, loaded on a short-lived background
/// isolate so the caller is not frozen for the load (see
/// `native_model_loader.dart`).
Future<LiveVad> buildSileroLiveVad({
  required String modelPath,
  required VadSensitivity sensitivity,
  required int sampleRate,
}) async {
  final config = sileroVadConfig(
    sensitivity: sensitivity,
    modelPath: modelPath,
    sampleRate: sampleRate,
  );
  final vad = await buildVadOffIsolate(
    config: config,
    bufferSizeInSeconds: _vadBufferSeconds,
  );
  return SileroLiveVad(vad, frameSize: config.sileroVad.windowSize);
}

/// Translates a [VadSensitivity] into the sherpa-onnx config that expresses
/// it. Pure — building the config touches no native code; only handing it
/// to the constructor does.
sherpa_onnx.VadModelConfig sileroVadConfig({
  required VadSensitivity sensitivity,
  required String modelPath,
  required int sampleRate,
}) {
  return sherpa_onnx.VadModelConfig(
    sileroVad: sherpa_onnx.SileroVadModelConfig(
      model: modelPath,
      threshold: sensitivity.threshold,
      minSilenceDuration: sensitivity.minSilenceSeconds,
      minSpeechDuration: sensitivity.minSpeechSeconds,
      maxSpeechDuration: sensitivity.maxSpeechSeconds,
    ),
    sampleRate: sampleRate,
  );
}

/// [LiveVad] backed by sherpa-onnx's Silero [sherpa_onnx.VoiceActivityDetector].
class SileroLiveVad implements LiveVad {
  SileroLiveVad(this._vad, {required this.frameSize});

  final sherpa_onnx.VoiceActivityDetector _vad;

  @override
  final int frameSize;

  @override
  void acceptWaveform(Float32List frame) => _vad.acceptWaveform(frame);

  @override
  bool get isSpeechDetected => _vad.isDetected();

  @override
  bool get hasSegment => !_vad.isEmpty();

  @override
  Float32List takeSegment() {
    final segment = _vad.front();
    _vad.pop();
    return segment.samples;
  }

  @override
  void flush() => _vad.flush();

  @override
  void free() => _vad.free();
}
