import 'dart:collection';
import 'dart:typed_data';

import 'package:hayamimi_core/hayamimi_core.dart';

/// A scripted stand-in for Silero VAD.
///
/// The real one is loaded over FFI, which a `flutter test` run cannot do,
/// and until a VAD exists a `LiveTranscriber` never reaches
/// `isRunning == true`. This one lets a test say exactly when speech starts,
/// when a segment ends, and what audio that segment contained — so
/// everything a *running* session does becomes observable without a device.
class FakeLiveVad implements LiveVad {
  FakeLiveVad({this.frameSize = 512});

  @override
  final int frameSize;

  /// Every frame this VAD was fed, in order.
  final List<Float32List> frames = <Float32List>[];

  /// What [isSpeechDetected] answers. A test sets this to true to stand for
  /// "the speaker is mid-utterance", which is what makes the session
  /// accumulate draft audio.
  bool speechActive = false;

  /// Handed over one per subsequent accepted frame, as if the speaker had
  /// just paused: the segment ends and [isSpeechDetected] goes false.
  ///
  /// Each carries where it started, in samples from this VAD's first
  /// frame, because that is what the session's pre-roll works from — see
  /// `PrerollHistory`. Use [segment] to build one.
  final Queue<LiveSpeechSegment> segmentsPerFrame = Queue<LiveSpeechSegment>();

  /// Handed over on [flush] instead — what a session stopping, or a debug
  /// wav file ending, produces.
  LiveSpeechSegment? flushSegment;

  int flushCount = 0;
  bool freed = false;

  final Queue<LiveSpeechSegment> _ready = Queue<LiveSpeechSegment>();

  /// A segment of [seconds] of silence claiming to start [startSample]
  /// samples into the session.
  static LiveSpeechSegment segment(
    double seconds, {
    int startSample = 0,
    int sampleRate = 16000,
  }) => LiveSpeechSegment(
    samples: Float32List((seconds * sampleRate).round()),
    startSample: startSample,
  );

  @override
  void acceptWaveform(Float32List frame) {
    frames.add(frame);
    if (segmentsPerFrame.isNotEmpty) {
      _ready.add(segmentsPerFrame.removeFirst());
      speechActive = false;
    }
  }

  @override
  bool get isSpeechDetected => speechActive;

  @override
  bool get hasSegment => _ready.isNotEmpty;

  @override
  LiveSpeechSegment takeSegment() => _ready.removeFirst();

  @override
  void flush() {
    flushCount++;
    final pending = flushSegment;
    if (pending != null) {
      _ready.add(pending);
      flushSegment = null;
    }
    speechActive = false;
  }

  @override
  void free() {
    freed = true;
  }
}

/// Hands out [FakeLiveVad]s in place of the real loader, and records what
/// each one was asked for — which is how a `setVadSensitivity` rebuild is
/// observed.
class FakeVadSource {
  FakeVadSource({this.setUp});

  /// Applied to each VAD as it is built, so a test can script one before
  /// the session that will use it exists.
  void Function(FakeLiveVad vad)? setUp;

  final List<FakeLiveVad> built = <FakeLiveVad>[];
  final List<VadSensitivity> requestedSensitivities = <VadSensitivity>[];
  final List<String> requestedModelPaths = <String>[];

  FakeLiveVad get last => built.last;

  Future<LiveVad> build({
    required String modelPath,
    required VadSensitivity sensitivity,
    required int sampleRate,
  }) async {
    requestedModelPaths.add(modelPath);
    requestedSensitivities.add(sensitivity);
    final vad = FakeLiveVad();
    setUp?.call(vad);
    built.add(vad);
    return vad;
  }
}
