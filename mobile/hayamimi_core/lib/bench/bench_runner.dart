import 'dart:io';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import 'bench_result.dart';
import 'model_file_resolver.dart';
import 'model_kind.dart';

class BenchRunException implements Exception {
  BenchRunException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Runs an offline ASR decode over a WAV file and measures RTF.
///
/// Model/tokenizer discovery and the sherpa-onnx FFI calls live here so the
/// UI layer only has to call [run] and render a [BenchResult]. Keeping this
/// separate from the widget tree also lets [resolveZipformerTransducerFiles]
/// (the file-picking logic) be unit tested without touching native FFI.
class BenchRunner {
  /// Runs the benchmark and returns the measured result.
  ///
  /// [modelDir] must contain the encoder/decoder/joiner/tokens files for a
  /// zipformer transducer model (see [resolveZipformerTransducerFiles] for
  /// the matching rules). [wavPath] must be a 16kHz mono WAV file.
  static Future<BenchResult> run({
    required ModelKind modelKind,
    required String modelDir,
    required String wavPath,
    int numThreads = 2,
  }) async {
    if (!modelKind.isImplemented) {
      throw BenchRunException(
        '${modelKind.label} is not implemented yet. Only Zipformer '
        '(transducer) works today.',
      );
    }

    final dir = Directory(modelDir);
    if (!await dir.exists()) {
      throw BenchRunException('Model directory not found: $modelDir');
    }

    final wavFile = File(wavPath);
    if (!await wavFile.exists()) {
      throw BenchRunException('WAV file not found: $wavPath');
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
      throw BenchRunException(e.message);
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
          numThreads: numThreads,
          debug: false,
          provider: 'cpu',
        ),
      ),
    );

    try {
      final wave = sherpa_onnx.readWave(wavPath);
      if (wave.samples.isEmpty) {
        throw BenchRunException(
          'Failed to read WAV file (unsupported format or empty): $wavPath',
        );
      }
      final audioDurationSeconds = wave.samples.length / wave.sampleRate;

      final stream = recognizer.createStream();
      final stopwatch = Stopwatch()..start();
      try {
        stream.acceptWaveform(
          samples: wave.samples,
          sampleRate: wave.sampleRate,
        );
        recognizer.decode(stream);
        stopwatch.stop();

        final result = recognizer.getResult(stream);
        return BenchResult(
          audioDurationSeconds: audioDurationSeconds,
          processingDurationSeconds: stopwatch.elapsedMicroseconds / 1e6,
          text: result.text,
        );
      } finally {
        stream.free();
      }
    } finally {
      recognizer.free();
    }
  }
}
