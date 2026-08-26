import 'dart:io';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import '../bench/model_file_resolver.dart';
import 'lang_routing.dart';
import 'routing_profile.dart';

class RoutedRecognizerException implements Exception {
  RoutedRecognizerException(this.message);
  final String message;

  @override
  String toString() => message;
}

/// Result of decoding one segment through [RoutedRecognizerSet.decode]:
/// the transcript, the language it was decoded with, and whether this
/// segment is what caused the session to switch languages (mirrors
/// [DualConfirmResult.switched]).
class RoutedDecodeResult {
  const RoutedDecodeResult({
    required this.text,
    required this.lang,
    required this.switched,
  });

  final String text;
  final String lang;
  final bool switched;
}

/// Only whisper-tiny LID candidates the loaded tiers can actually decode.
/// A candidate outside this set (e.g. "fr", "ru") has no specialist model
/// loaded in [RoutingProfile.jaSenseVoice], so it can't drive a switch —
/// the segment holds at the session's current language instead. This is
/// mobile's simplification of the desktop's 4-tier catalog down to 2 tiers
/// (see `routing_profile.dart`).
const int _lidMaxSeconds = 4; // mirrors desktop asr_engine.LID_MAX_SECONDS

/// Owns the three native models a [RoutingProfile.jaSenseVoice] session
/// needs — ReazonSpeech ja, SenseVoice (en/zh/ko/yue), and the whisper-tiny
/// spoken-language identifier — and implements the same dual-LID switch
/// policy the desktop pipeline uses (`resolveDualConfirm`, ported in
/// `lang_routing.dart`) to decide, per VAD segment, which model decodes it.
///
/// Not unit-testable on its own (real FFI/model files needed) by design,
/// same as `LiveTranscriber` — the pure decision logic it calls into
/// ([resolveDualConfirm], [svLidTag]) is what's unit tested.
class RoutedRecognizerSet {
  RoutedRecognizerSet._({
    required sherpa_onnx.OfflineRecognizer reazonRecognizer,
    required sherpa_onnx.OfflineRecognizer senseVoiceRecognizer,
    required sherpa_onnx.SpokenLanguageIdentification lidRecognizer,
    required this.sampleRate,
  }) : _reazon = reazonRecognizer,
       _senseVoice = senseVoiceRecognizer,
       _lid = lidRecognizer;

  final sherpa_onnx.OfflineRecognizer _reazon;
  final sherpa_onnx.OfflineRecognizer _senseVoice;
  final sherpa_onnx.SpokenLanguageIdentification _lid;
  final int sampleRate;

  /// The session's current language, or `null` before the first segment
  /// resolves one (bootstrap). Exposed for the UI's language badge.
  String? currentLang;

  /// Builds the ja (ReazonSpeech, `modified_beam_search` — matches desktop
  /// production, see `bench_runner.dart`) + SenseVoice + whisper-tiny LID
  /// recognizer set from their model directories.
  static Future<RoutedRecognizerSet> build({
    required String reazonModelDir,
    required String senseVoiceModelDir,
    required String lidModelDir,
    int numThreads = 2,
    int sampleRate = 16000,
  }) async {
    final reazonDir = Directory(reazonModelDir);
    if (!await reazonDir.exists()) {
      throw RoutedRecognizerException(
        'ja model directory not found: $reazonModelDir',
      );
    }
    final filenames = await reazonDir
        .list()
        .where((e) => e is File)
        .map((e) => e.uri.pathSegments.last)
        .toList();
    final ResolvedModelFiles resolved;
    try {
      resolved = resolveZipformerTransducerFiles(filenames);
    } on ModelFileResolutionException catch (e) {
      throw RoutedRecognizerException(e.message);
    }
    final sep = Platform.pathSeparator;
    final reazon = sherpa_onnx.OfflineRecognizer(
      sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          transducer: sherpa_onnx.OfflineTransducerModelConfig(
            encoder: '$reazonModelDir$sep${resolved.encoder}',
            decoder: '$reazonModelDir$sep${resolved.decoder}',
            joiner: '$reazonModelDir$sep${resolved.joiner}',
          ),
          tokens: '$reazonModelDir$sep${resolved.tokens}',
          numThreads: numThreads,
          debug: false,
          provider: 'cpu',
        ),
        decodingMethod: 'modified_beam_search',
      ),
    );

    final svDir = Directory(senseVoiceModelDir);
    if (!await svDir.exists()) {
      reazon.free();
      throw RoutedRecognizerException(
        'SenseVoice model directory not found: $senseVoiceModelDir',
      );
    }
    final svFilenames = await svDir
        .list()
        .where((e) => e is File)
        .map((e) => e.uri.pathSegments.last)
        .toList();
    final svModel = svFilenames.firstWhere(
      (f) => f.toLowerCase().endsWith('.onnx'),
      orElse: () => '',
    );
    final svTokens = svFilenames.firstWhere(
      (f) => f.toLowerCase() == 'tokens.txt',
      orElse: () => '',
    );
    if (svModel.isEmpty || svTokens.isEmpty) {
      reazon.free();
      throw RoutedRecognizerException(
        'SenseVoice model/tokens not found in: $senseVoiceModelDir',
      );
    }
    final senseVoice = sherpa_onnx.OfflineRecognizer(
      sherpa_onnx.OfflineRecognizerConfig(
        model: sherpa_onnx.OfflineModelConfig(
          senseVoice: sherpa_onnx.OfflineSenseVoiceModelConfig(
            model: '$senseVoiceModelDir$sep$svModel',
            language: '', // auto: SenseVoice has its own internal LID
            useInverseTextNormalization: true,
          ),
          tokens: '$senseVoiceModelDir$sep$svTokens',
          numThreads: numThreads,
          debug: false,
          provider: 'cpu',
        ),
      ),
    );

    final lidDir = Directory(lidModelDir);
    if (!await lidDir.exists()) {
      reazon.free();
      senseVoice.free();
      throw RoutedRecognizerException(
        'LID model directory not found: $lidModelDir',
      );
    }
    final lidFilenames = await lidDir
        .list()
        .where((e) => e is File)
        .map((e) => e.uri.pathSegments.last)
        .toList();
    String pickLid(String needle) {
      final hits = lidFilenames.where(
        (f) =>
            f.toLowerCase().contains(needle) &&
            f.toLowerCase().contains('int8') &&
            f.toLowerCase().endsWith('.onnx'),
      );
      return hits.isNotEmpty ? hits.first : '';
    }

    final lidEncoder = pickLid('encoder');
    final lidDecoder = pickLid('decoder');
    if (lidEncoder.isEmpty || lidDecoder.isEmpty) {
      reazon.free();
      senseVoice.free();
      throw RoutedRecognizerException(
        'whisper-tiny LID encoder/decoder not found in: $lidModelDir',
      );
    }
    final lid = sherpa_onnx.SpokenLanguageIdentification(
      sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper: sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
          encoder: '$lidModelDir$sep$lidEncoder',
          decoder: '$lidModelDir$sep$lidDecoder',
        ),
        numThreads: numThreads,
      ),
    );

    return RoutedRecognizerSet._(
      reazonRecognizer: reazon,
      senseVoiceRecognizer: senseVoice,
      lidRecognizer: lid,
      sampleRate: sampleRate,
    );
  }

  /// Decodes one VAD-bounded [samples] segment, routing it to ReazonSpeech
  /// (ja) or SenseVoice (en/zh/ko/yue) per the dual-LID policy in
  /// `lang_routing.dart`.
  RoutedDecodeResult decode(Float32List samples) {
    final speechSeconds = samples.length / sampleRate;
    final whisperLang = _identifyLang(samples);

    // Fast path: the LID candidate matches the session's current language
    // already -- decode directly, no SenseVoice probe needed (mirrors the
    // desktop's `lang == last_lang` early return).
    if (whisperLang == currentLang) {
      return _decodeWith(currentLang!, samples);
    }

    // A candidate outside SenseVoice's coverage has no specialist tier
    // loaded here (see routing_profile.dart) -- hold the current language
    // (or fall back to "ja" at bootstrap, the common case for this app).
    if (!jaSenseVoiceLangs.contains(whisperLang)) {
      final held = currentLang ?? 'ja';
      currentLang = held;
      return _decodeWith(held, samples);
    }

    // Candidate is one of the 5 SenseVoice-covered languages and disagrees
    // with (or there is no) current language: decode via SenseVoice to get
    // both a transcript AND its own LID tag on this exact audio in one
    // call, then arbitrate via resolveDualConfirm.
    final svStream = _senseVoice.createStream();
    final String svText;
    final String svLang;
    try {
      svStream.acceptWaveform(samples: samples, sampleRate: sampleRate);
      _senseVoice.decode(svStream);
      final result = _senseVoice.getResult(svStream);
      svText = result.text.trim();
      svLang = svLidTag(result.lang);
    } finally {
      svStream.free();
    }

    final confirm = resolveDualConfirm(
      lang: whisperLang,
      lastLang: currentLang,
      speechSeconds: speechSeconds,
      svLang: svLang,
    );
    currentLang = confirm.lang;

    if (confirm.lang == 'ja') {
      // Reuse-avoidance: the SenseVoice decode above wasn't for ja, so
      // decode again with ReazonSpeech (desktop production quality tier).
      return _decodeWith('ja', samples, switched: confirm.switched);
    }

    // Non-ja resolved language: the SenseVoice decode already produced the
    // transcript for this exact audio -- reuse it instead of decoding
    // twice.
    return RoutedDecodeResult(
      text: svText,
      lang: confirm.lang,
      switched: confirm.switched,
    );
  }

  RoutedDecodeResult _decodeWith(
    String lang,
    Float32List samples, {
    bool switched = false,
  }) {
    final recognizer = lang == 'ja' ? _reazon : _senseVoice;
    final stream = recognizer.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: sampleRate);
      recognizer.decode(stream);
      final text = recognizer.getResult(stream).text.trim();
      return RoutedDecodeResult(text: text, lang: lang, switched: switched);
    } finally {
      stream.free();
    }
  }

  String _identifyLang(Float32List samples) {
    final maxSamples = _lidMaxSeconds * sampleRate;
    final probeSamples = samples.length > maxSamples
        ? Float32List.sublistView(samples, 0, maxSamples)
        : samples;
    final stream = _lid.createStream();
    try {
      stream.acceptWaveform(samples: probeSamples, sampleRate: sampleRate);
      final result = _lid.compute(stream);
      return result.lang;
    } finally {
      stream.free();
    }
  }

  /// Releases all three native model handles. Call once when the routed
  /// session ends.
  void free() {
    _reazon.free();
    _senseVoice.free();
    _lid.free();
  }
}
