import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../bench/model_file_resolver.dart';
import '../bench/model_kind.dart';
import 'live_transcript_entry.dart';
import 'pcm_frame_buffer.dart';
import 'refine_pass.dart';
import 'speech_segment_filter.dart';

/// How often the auto-refine due check ([isAutoRefineDue]) runs while a
/// session is live and auto mode is on. Cheap (a few comparisons), so a 1s
/// tick is plenty responsive without meaningfully affecting battery.
const _autoRefineCheckInterval = Duration(seconds: 1);

class LiveTranscriberException implements Exception {
  LiveTranscriberException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Streams mic audio through Silero VAD to find speech segments, decodes
/// each finalized segment with the same [sherpa_onnx.OfflineRecognizer]
/// `BenchRunner` uses, and emits one [LiveTranscriptEntry] per segment.
///
/// This class owns the mic stream, the native VAD/recognizer handles, and
/// the glue between them, so it isn't unit-testable on its own (it needs a
/// real device/emulator mic and the sherpa-onnx native libs). The pure
/// pieces it's built from — [pcm16BytesToFloat32], [PcmFrameBuffer], and
/// [isSegmentWorthDecoding] — are unit tested separately.
class LiveTranscriber {
  LiveTranscriber({AudioRecorder? recorder})
    : _recorder = recorder ?? AudioRecorder();

  static const int sampleRate = 16000;

  final AudioRecorder _recorder;
  final _entriesController = StreamController<LiveTranscriptEntry>.broadcast();
  final _decodingController = StreamController<bool>.broadcast();
  final _refineEntriesController =
      StreamController<LiveTranscriptEntry>.broadcast();

  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.OfflineRecognizer? _recognizer;
  StreamSubscription<Uint8List>? _micSubscription;
  PcmFrameBuffer? _frameBuffer;

  // --- Two-pass "refine" (清書): buffer finalized segments' audio and, on
  // demand, re-decode the whole group together for a cleaner result than
  // any single segment got on its own (same effect the desktop pipeline's
  // `Refiner` gets from re-decoding across segment boundaries — see
  // `scripts/realtime_transcribe.py`). No punctuation-restoration model is
  // layered on top of this; it's re-decode only.
  final RefineBuffer _refineBuffer = RefineBuffer(sampleRate: sampleRate);
  DateTime? _lastSegmentAt;
  Timer? _autoRefineTimer;
  bool _autoRefineEnabled = false;

  /// Finalized transcript lines, one per detected speech segment.
  Stream<LiveTranscriptEntry> get entries => _entriesController.stream;

  /// Emits `true` right before a segment starts decoding and `false` right
  /// after, so the UI can show a busy indicator. Shared by both the fast
  /// per-segment decode and the refine pass, since only one decode ever
  /// runs at a time on this recognizer.
  Stream<bool> get decoding => _decodingController.stream;

  /// Refine ("清書") results: one entry per manual or automatic refine
  /// pass, each covering everything buffered since the previous refine (or
  /// since the session started).
  Stream<LiveTranscriptEntry> get refineEntries => _refineEntriesController.stream;

  bool get isRunning => _micSubscription != null;

  /// Total audio currently buffered for the next refine pass, in seconds.
  double get refineBufferedSeconds => _refineBuffer.totalDurationSeconds;

  /// Whether auto-refine (silence-triggered, see [refine_pass.dart]) is on.
  /// Defaults to off: on a phone every refine is a full re-decode burning
  /// battery and generating heat, so firing it automatically is opt-in —
  /// the manual "清書" button always works regardless of this setting.
  bool get autoRefineEnabled => _autoRefineEnabled;

  set autoRefineEnabled(bool value) {
    _autoRefineEnabled = value;
    if (!value) {
      _autoRefineTimer?.cancel();
      _autoRefineTimer = null;
    } else if (isRunning) {
      _startAutoRefineTimer();
    }
  }

  /// Starts capturing mic audio and transcribing it.
  ///
  /// [modelDir] must contain a zipformer transducer model (see
  /// `resolveZipformerTransducerFiles`). [vadModelPath] must point at a
  /// Silero VAD onnx model file (e.g. `silero_vad.onnx`).
  Future<void> start({
    required ModelKind modelKind,
    required String modelDir,
    required String vadModelPath,
  }) async {
    if (isRunning) {
      return;
    }

    if (!modelKind.isImplemented) {
      throw LiveTranscriberException(
        '${modelKind.label} is not implemented yet. Only Zipformer '
        '(transducer) works today.',
      );
    }

    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      throw LiveTranscriberException('Model directory not found: $modelDir');
    }

    final vadModelFile = File(vadModelPath);
    if (!await vadModelFile.exists()) {
      throw LiveTranscriberException('VAD model not found: $vadModelPath');
    }

    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      throw LiveTranscriberException('Microphone permission was not granted.');
    }

    final filenames = await dir
        .list()
        .where((e) => e is File)
        .map((e) => e.uri.pathSegments.last)
        .toList();

    final ResolvedModelFiles resolved;
    try {
      resolved = resolveZipformerTransducerFiles(filenames);
    } on ModelFileResolutionException catch (e) {
      throw LiveTranscriberException(e.message);
    }

    final sep = Platform.pathSeparator;
    _recognizer = sherpa_onnx.OfflineRecognizer(
      sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          transducer: sherpa_onnx.OfflineTransducerModelConfig(
            encoder: '$modelDir$sep${resolved.encoder}',
            decoder: '$modelDir$sep${resolved.decoder}',
            joiner: '$modelDir$sep${resolved.joiner}',
          ),
          tokens: '$modelDir$sep${resolved.tokens}',
          numThreads: 2,
          debug: false,
          provider: 'cpu',
        ),
      ),
    );

    final vadConfig = sherpa_onnx.VadModelConfig(
      sileroVad: sherpa_onnx.SileroVadModelConfig(model: vadModelPath),
      sampleRate: sampleRate,
    );
    _vad = sherpa_onnx.VoiceActivityDetector(
      config: vadConfig,
      bufferSizeInSeconds: 30,
    );
    _frameBuffer = PcmFrameBuffer(frameSize: vadConfig.sileroVad.windowSize);

    try {
      final micStream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: sampleRate,
          numChannels: 1,
        ),
      );
      _micSubscription = micStream.listen(_onMicChunk);
    } catch (_) {
      await _teardownNativeState();
      rethrow;
    }

    _refineBuffer.clear();
    _lastSegmentAt = null;
    if (_autoRefineEnabled) {
      _startAutoRefineTimer();
    }
  }

  void _startAutoRefineTimer() {
    _autoRefineTimer?.cancel();
    _autoRefineTimer = Timer.periodic(_autoRefineCheckInterval, (_) {
      final lastSegmentAt = _lastSegmentAt;
      if (lastSegmentAt == null) {
        return;
      }
      final due = isAutoRefineDue(
        sinceLastSegment: DateTime.now().difference(lastSegmentAt),
        bufferedDurationSeconds: _refineBuffer.totalDurationSeconds,
      );
      if (due) {
        unawaited(refineNow());
      }
    });
  }

  void _onMicChunk(Uint8List bytes) {
    final vad = _vad;
    final frameBuffer = _frameBuffer;
    if (vad == null || frameBuffer == null) {
      return;
    }

    frameBuffer.add(pcm16BytesToFloat32(bytes));
    for (final frame in frameBuffer.drainFrames()) {
      vad.acceptWaveform(frame);
    }
    _drainReadySegments();
  }

  void _drainReadySegments() {
    final vad = _vad;
    if (vad == null) {
      return;
    }
    while (!vad.isEmpty()) {
      final segment = vad.front();
      vad.pop();
      _decodeSegment(segment);
    }
  }

  void _decodeSegment(sherpa_onnx.SpeechSegment segment) {
    final recognizer = _recognizer;
    if (recognizer == null ||
        !isSegmentWorthDecoding(segment, sampleRate: sampleRate)) {
      return;
    }

    _decodingController.add(true);
    final stopwatch = Stopwatch()..start();
    final stream = recognizer.createStream();
    try {
      stream.acceptWaveform(samples: segment.samples, sampleRate: sampleRate);
      recognizer.decode(stream);
      final result = recognizer.getResult(stream);
      stopwatch.stop();
      final text = result.text.trim();
      if (text.isNotEmpty) {
        final now = DateTime.now();
        _entriesController.add(
          LiveTranscriptEntry(
            text: result.text,
            timestamp: now,
            latencyMs: stopwatch.elapsedMicroseconds / 1000,
          ),
        );
        _lastSegmentAt = now;
        _refineBuffer.add(
          RefineSegment(samples: segment.samples, text: text, capturedAt: now),
        );
      }
    } finally {
      stream.free();
      _decodingController.add(false);
    }
  }

  /// Runs a refine pass over everything currently buffered: combines the
  /// buffered segments' audio and re-decodes it as one utterance, using the
  /// same recognizer the fast per-segment path uses. Emits nothing if the
  /// buffer is empty, the combined audio is under half a second (mirrors
  /// the desktop `Refiner`'s `len(buf) < sr // 2` guard), or the result is
  /// empty after the too-short fallback below.
  ///
  /// Safe to call whether or not [autoRefineEnabled] is on — this is also
  /// what the manual "清書" button calls directly, and what the debug wav
  /// test path (see `runDebugWavRefineTest`) exercises indirectly via its
  /// own recognizer.
  Future<void> refineNow() async {
    final recognizer = _recognizer;
    if (recognizer == null || _refineBuffer.isEmpty) {
      return;
    }
    final segments = _refineBuffer.takeAll();
    final combined = combineSegmentSamples(segments);
    if (combined.length < sampleRate ~/ 2) {
      return;
    }
    final fastJoined = combineSegmentFastText(segments);

    _decodingController.add(true);
    final stopwatch = Stopwatch()..start();
    final stream = recognizer.createStream();
    try {
      stream.acceptWaveform(samples: combined, sampleRate: sampleRate);
      recognizer.decode(stream);
      var text = recognizer.getResult(stream).text.trim();
      stopwatch.stop();
      // A merged re-decode must never LOSE content: if it comes back much
      // shorter than the fast finals combined, trust those instead (mirrors
      // scripts/realtime_transcribe.py's Refiner).
      if (isRefineTextTooShort(text, fastJoined)) {
        text = fastJoined;
      }
      if (text.isNotEmpty) {
        _refineEntriesController.add(
          LiveTranscriptEntry(
            text: text,
            timestamp: DateTime.now(),
            latencyMs: stopwatch.elapsedMicroseconds / 1000,
          ),
        );
      }
    } finally {
      stream.free();
      _decodingController.add(false);
    }
  }

  /// Stops capturing, flushes any speech still buffered in the VAD, and
  /// releases native resources. Anything still buffered for refine is
  /// dropped unrefined — this mirrors the fast path (an in-progress VAD
  /// segment gets flushed and decoded, but there's no "finish the refine
  /// pass on shutdown" step the way the desktop's `force=True` shutdown
  /// path has, since stopping is a deliberate user action here rather than
  /// a process exit).
  Future<void> stop() async {
    if (!isRunning) {
      return;
    }
    _autoRefineTimer?.cancel();
    _autoRefineTimer = null;

    await _micSubscription?.cancel();
    _micSubscription = null;
    await _recorder.stop();

    _vad?.flush();
    _drainReadySegments();

    _refineBuffer.clear();
    _lastSegmentAt = null;

    await _teardownNativeState();
  }

  Future<void> _teardownNativeState() async {
    _vad?.free();
    _vad = null;
    _recognizer?.free();
    _recognizer = null;
    _frameBuffer = null;
  }

  /// Releases everything, including the mic recorder itself. Call once when
  /// the owning widget is disposed.
  Future<void> dispose() async {
    await stop();
    await _entriesController.close();
    await _decodingController.close();
    await _refineEntriesController.close();
    await _recorder.dispose();
  }

  /// Debug-only helper (see `kDebugMode` at the call site in `live_page.dart`):
  /// exercises the refine pass's audio-combining logic end to end against a
  /// real WAV file, without needing a live mic session or an emulator's
  /// (nonexistent) microphone.
  ///
  /// Loads its own short-lived [sherpa_onnx.OfflineRecognizer] from
  /// [modelDir] (independent of any running live session), splits
  /// [wavPath]'s samples into two halves to stand in for two VAD segments,
  /// decodes each half individually (the "fast path" comparison), then runs
  /// them back through [combineSegmentSamples] and decodes the combined
  /// audio (the "refine" result) — so a caller can see whether the refine
  /// decode differs from the two individual ones.
  static Future<DebugRefineTestResult> runDebugWavRefineTest({
    required String modelDir,
    required String wavPath,
  }) async {
    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      throw LiveTranscriberException('Model directory not found: $modelDir');
    }
    final wavFile = File(wavPath);
    if (!await wavFile.exists()) {
      throw LiveTranscriberException('WAV file not found: $wavPath');
    }

    final filenames = await dir
        .list()
        .where((e) => e is File)
        .map((e) => e.uri.pathSegments.last)
        .toList();

    final ResolvedModelFiles resolved;
    try {
      resolved = resolveZipformerTransducerFiles(filenames);
    } on ModelFileResolutionException catch (e) {
      throw LiveTranscriberException(e.message);
    }

    final sep = Platform.pathSeparator;
    final recognizer = sherpa_onnx.OfflineRecognizer(
      sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          transducer: sherpa_onnx.OfflineTransducerModelConfig(
            encoder: '$modelDir$sep${resolved.encoder}',
            decoder: '$modelDir$sep${resolved.decoder}',
            joiner: '$modelDir$sep${resolved.joiner}',
          ),
          tokens: '$modelDir$sep${resolved.tokens}',
          numThreads: 2,
          debug: false,
          provider: 'cpu',
        ),
      ),
    );

    try {
      final wave = sherpa_onnx.readWave(wavPath);
      if (wave.samples.isEmpty) {
        throw LiveTranscriberException(
          'Failed to read WAV file (unsupported format or empty): $wavPath',
        );
      }

      String decode(Float32List samples) {
        final stream = recognizer.createStream();
        try {
          stream.acceptWaveform(samples: samples, sampleRate: wave.sampleRate);
          recognizer.decode(stream);
          return recognizer.getResult(stream).text.trim();
        } finally {
          stream.free();
        }
      }

      final mid = wave.samples.length ~/ 2;
      final segment1 = RefineSegment(
        samples: Float32List.sublistView(wave.samples, 0, mid),
        text: '',
        capturedAt: DateTime.now(),
      );
      final segment2 = RefineSegment(
        samples: Float32List.sublistView(wave.samples, mid),
        text: '',
        capturedAt: DateTime.now(),
      );

      final text1 = decode(segment1.samples);
      final text2 = decode(segment2.samples);
      final combined = combineSegmentSamples([segment1, segment2]);
      final refineText = decode(combined);

      return DebugRefineTestResult(
        segment1Text: text1,
        segment2Text: text2,
        refineText: refineText,
      );
    } finally {
      recognizer.free();
    }
  }
}

/// Result of [LiveTranscriber.runDebugWavRefineTest]: the two individually
/// decoded halves of the test wav, and the combined ("清書") re-decode, for
/// visually comparing whether the refine pass changed anything.
class DebugRefineTestResult {
  const DebugRefineTestResult({
    required this.segment1Text,
    required this.segment2Text,
    required this.refineText,
  });

  final String segment1Text;
  final String segment2Text;
  final String refineText;
}
