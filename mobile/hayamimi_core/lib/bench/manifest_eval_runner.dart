import 'dart:convert';
import 'dart:io';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../routing/routed_recognizer.dart';
import 'bench_runner.dart';
import 'manifest_eval_result.dart';

/// Batch-decodes every clip named in a `manifest.json` (the same format as
/// testdata/eval_real/manifest.json: a JSON array of `{"wav","lang","ref"}`)
/// through a single zipformer-transducer recognizer, so accuracy can be
/// compared against a PC run of the identical model.
///
/// This is a debug/dev tool, not a production code path — see the
/// `kDebugMode`-gated call site in the mobile app's Bench tab.
class ManifestEvalRunner {
  /// Decodes every entry in [manifestPath] using audio files from [wavDir]
  /// (each entry's `wav` field is joined onto [wavDir]), returning one
  /// [ManifestEvalResult] per manifest entry in manifest order.
  ///
  /// Throws [BenchRunException] if the manifest, model, or a referenced wav
  /// file cannot be found/read.
  static Future<List<ManifestEvalResult>> run({
    required String modelDir,
    required String manifestPath,
    required String wavDir,
    int numThreads = 2,
  }) async {
    final manifestFile = File(manifestPath);
    if (!await manifestFile.exists()) {
      throw BenchRunException('Manifest file not found: $manifestPath');
    }

    final List<dynamic> raw;
    try {
      raw = jsonDecode(await manifestFile.readAsString()) as List<dynamic>;
    } on FormatException catch (e) {
      throw BenchRunException('Failed to parse manifest JSON: $e');
    }
    final entries = [
      for (final item in raw)
        ManifestEntry.fromJson((item as Map).cast<String, Object?>()),
    ];

    // modified_beam_search matches desktop production
    // (scripts/asr_engine.py::_build_reazon) for ja ReazonSpeech — using
    // sherpa-onnx's greedy_search default here would compare a different
    // decoding config, not just a different platform.
    final recognizer = await buildZipformerRecognizer(
      modelDir,
      numThreads: numThreads,
      decodingMethod: 'modified_beam_search',
    );

    final results = <ManifestEvalResult>[];
    try {
      final sep = Platform.pathSeparator;
      for (final entry in entries) {
        final wavPath = '$wavDir$sep${entry.wav}';
        final wave = sherpa_onnx.readWave(wavPath);
        if (wave.samples.isEmpty) {
          throw BenchRunException(
            'Failed to read WAV file (unsupported format, empty, or '
            'missing): $wavPath',
          );
        }
        final audioDurationSeconds = wave.samples.length / wave.sampleRate;

        final stream = recognizer.createStream();
        final stopwatch = Stopwatch()..start();
        final String hyp;
        try {
          stream.acceptWaveform(
            samples: wave.samples,
            sampleRate: wave.sampleRate,
          );
          recognizer.decode(stream);
          stopwatch.stop();
          hyp = recognizer.getResult(stream).text;
        } finally {
          stream.free();
        }

        results.add(
          ManifestEvalResult(
            wav: entry.wav,
            lang: entry.lang,
            ref: entry.ref,
            hyp: hyp,
            audioDurationSeconds: audioDurationSeconds,
            decodeSeconds: stopwatch.elapsedMicroseconds / 1e6,
          ),
        );
      }
    } finally {
      recognizer.free();
    }
    return results;
  }

  /// Decodes every entry in [manifestPath] through
  /// `RoutingProfile.jaSenseVoice` routing (ReazonSpeech ja + SenseVoice
  /// en/zh/ko/yue, arbitrated per clip by the dual-LID policy in
  /// `../routing/`), so a multilingual manifest's language-routing accuracy
  /// and CER can be compared against the PC pipeline's `docs/SCORECARD.md`.
  ///
  /// Each entry is treated as its own isolated "session" (bootstrap: no
  /// prior language) since manifest clips are independent recordings, not
  /// a continuous conversation — this measures the dual-LID bootstrap path
  /// specifically, which `docs/LID.md` table 3 shows is the harder case.
  ///
  /// Throws [BenchRunException] if the manifest or any model/wav file
  /// cannot be found/read.
  static Future<List<ManifestEvalResult>> runRouted({
    required String reazonModelDir,
    required String senseVoiceModelDir,
    required String lidModelDir,
    required String manifestPath,
    required String wavDir,
    int numThreads = 2,
  }) async {
    final manifestFile = File(manifestPath);
    if (!await manifestFile.exists()) {
      throw BenchRunException('Manifest file not found: $manifestPath');
    }

    final List<dynamic> raw;
    try {
      raw = jsonDecode(await manifestFile.readAsString()) as List<dynamic>;
    } on FormatException catch (e) {
      throw BenchRunException('Failed to parse manifest JSON: $e');
    }
    final entries = [
      for (final item in raw)
        ManifestEntry.fromJson((item as Map).cast<String, Object?>()),
    ];

    final RoutedRecognizerSet routed;
    try {
      routed = await RoutedRecognizerSet.build(
        reazonModelDir: reazonModelDir,
        senseVoiceModelDir: senseVoiceModelDir,
        lidModelDir: lidModelDir,
        numThreads: numThreads,
      );
    } on RoutedRecognizerException catch (e) {
      throw BenchRunException(e.message);
    }

    final results = <ManifestEvalResult>[];
    try {
      final sep = Platform.pathSeparator;
      for (final entry in entries) {
        final wavPath = '$wavDir$sep${entry.wav}';
        final wave = sherpa_onnx.readWave(wavPath);
        if (wave.samples.isEmpty) {
          throw BenchRunException(
            'Failed to read WAV file (unsupported format, empty, or '
            'missing): $wavPath',
          );
        }
        final audioDurationSeconds = wave.samples.length / wave.sampleRate;

        // Each manifest clip is its own bootstrap session (see doc comment
        // above): reset the routed language before every clip.
        routed.currentLang = null;

        final stopwatch = Stopwatch()..start();
        final result = routed.decode(wave.samples);
        stopwatch.stop();

        results.add(
          ManifestEvalResult(
            wav: entry.wav,
            lang: entry.lang,
            ref: entry.ref,
            hyp: result.text,
            audioDurationSeconds: audioDurationSeconds,
            decodeSeconds: stopwatch.elapsedMicroseconds / 1e6,
            detectedLang: result.lang,
          ),
        );
      }
    } finally {
      routed.free();
    }
    return results;
  }

  /// Serializes [results] as the JSON array written to disk by the Bench
  /// tab's "Run manifest eval" button, so `adb pull` + `eval_accuracy.py`'s
  /// `cer_ja` can score it on the PC side without re-deriving the format.
  static String toJson(List<ManifestEvalResult> results) {
    return const JsonEncoder.withIndent(
      '  ',
    ).convert([for (final r in results) r.toJson()]);
  }
}
