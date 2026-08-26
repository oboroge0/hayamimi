import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../bench/model_file_resolver.dart';
import '../bench/model_kind.dart';
import 'live_transcript_entry.dart';
import 'pcm_frame_buffer.dart';
import 'speech_segment_filter.dart';

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

  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.OfflineRecognizer? _recognizer;
  StreamSubscription<Uint8List>? _micSubscription;
  PcmFrameBuffer? _frameBuffer;

  /// Finalized transcript lines, one per detected speech segment.
  Stream<LiveTranscriptEntry> get entries => _entriesController.stream;

  /// Emits `true` right before a segment starts decoding and `false` right
  /// after, so the UI can show a busy indicator.
  Stream<bool> get decoding => _decodingController.stream;

  bool get isRunning => _micSubscription != null;

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
      if (result.text.trim().isNotEmpty) {
        _entriesController.add(
          LiveTranscriptEntry(
            text: result.text,
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
  /// releases native resources.
  Future<void> stop() async {
    if (!isRunning) {
      return;
    }
    await _micSubscription?.cancel();
    _micSubscription = null;
    await _recorder.stop();

    _vad?.flush();
    _drainReadySegments();

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
    await _recorder.dispose();
  }
}
