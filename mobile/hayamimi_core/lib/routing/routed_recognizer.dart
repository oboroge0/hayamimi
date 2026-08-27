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
    final svTokens = svFilenames.firstWhere(
      (f) => f.toLowerCase() == 'tokens.txt',
      orElse: () => '',
    );
    final String svModel;
    try {
      svModel = svTokens.isEmpty ? '' : resolveOnnxFile(svFilenames);
    } on ModelFileResolutionException catch (_) {
      reazon.free();
      throw RoutedRecognizerException(
        'SenseVoice model/tokens not found in: $senseVoiceModelDir',
      );
    }
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
    final ResolvedOnnxPair lidFiles;
    try {
      // requireInt8: true -- this tier only ever ships the quantized
      // whisper-tiny variant, so a match without an int8 file is treated
      // the same as no match at all (mirrors the old hand-rolled pickLid).
      lidFiles = resolveOnnxFilePair(
        lidFilenames,
        role1: 'encoder',
        role2: 'decoder',
        requireInt8: true,
      );
    } on ModelFileResolutionException catch (_) {
      reazon.free();
      senseVoice.free();
      throw RoutedRecognizerException(
        'whisper-tiny LID encoder/decoder not found in: $lidModelDir',
      );
    }
    final lid = sherpa_onnx.SpokenLanguageIdentification(
      sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper: sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
          encoder: '$lidModelDir$sep${lidFiles.first}',
          decoder: '$lidModelDir$sep${lidFiles.second}',
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
    // loaded here (see routing_profile.dart) -- hold the current language.
    if (!jaSenseVoiceLangs.contains(whisperLang)) {
      if (currentLang == null) {
        // Bootstrap: whisper-tiny's very first guess is a language this app
        // has no tier for at all (e.g. "fr", "ru") -- blindly defaulting to
        // "ja" here would repeat the same mistake the desktop pipeline fixed
        // for its own bootstrap path (see asr_engine.py's
        // resolve_sticky_lang bootstrap_probe_lang and docs/LID.md): a
        // single whisper-tiny misfire on segment 1 seeding the whole
        // session with the wrong language. Probe with SenseVoice's own
        // internal LID on this exact audio and arbitrate with the same
        // [resolveDualConfirm] policy the 5-language coverage case below
        // uses, instead of blindly trusting whisper-tiny's out-of-coverage
        // guess.
        final svStream = _senseVoice.createStream();
        final String probeText;
        final String probeLang;
        try {
          svStream.acceptWaveform(samples: samples, sampleRate: sampleRate);
          _senseVoice.decode(svStream);
          final result = _senseVoice.getResult(svStream);
          probeText = result.text.trim();
          probeLang = svLidTag(result.lang);
        } finally {
          svStream.free();
        }
        final confirm = resolveDualConfirm(
          lang: whisperLang,
          lastLang: null,
          speechSeconds: speechSeconds,
          svLang: probeLang,
        );
        if (!jaSenseVoiceLangs.contains(confirm.lang)) {
          // Neither LID landed on a language this app has a tier for
          // (whisper-tiny said e.g. "fr" and SenseVoice's own probe agreed
          // or gave nothing usable) -- there's no specialist to route to,
          // so fall back to "ja" same as the pre-fix behavior.
          currentLang = 'ja';
          return _decodeWith('ja', samples, switched: true);
        }
        currentLang = confirm.lang;
        if (confirm.lang == 'ja') {
          // Reuse-avoidance: the probe above decoded through SenseVoice,
          // not ReazonSpeech -- re-decode with the production ja tier.
          return _decodeWith('ja', samples, switched: confirm.switched);
        }
        if (confirm.lang == probeLang && probeText.isNotEmpty) {
          // resolveDualConfirm picked the SenseVoice probe's own language --
          // the probe decode above already produced this transcript, so
          // reuse it instead of decoding twice.
          return RoutedDecodeResult(
            text: probeText,
            lang: confirm.lang,
            switched: confirm.switched,
          );
        }
        return _decodeWith(confirm.lang, samples, switched: confirm.switched);
      }
      return _decodeWith(currentLang!, samples);
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

  /// Decodes [samples] with whichever model backs the session's *current*
  /// language, without running LID or the dual-LID switch policy at all --
  /// the deliberately cheap path for draft ("発話中の暫定字幕") decodes.
  ///
  /// [decode] answers "which language is this?" for every finalized
  /// segment, which costs a whisper-tiny LID pass (and sometimes a second
  /// SenseVoice probe) on top of the transcript decode. Running that full
  /// routing judgment every ~1s while a segment is still in progress would
  /// multiply the power/heat cost of drafts for no benefit: the draft is
  /// provisional and gets replaced by the properly-routed fast-final the
  /// moment the segment actually finalizes, so an occasional wrong-language
  /// draft costs nothing but a flicker. Falls back to "ja" before any
  /// segment has resolved a language yet (mirrors [decode]'s bootstrap
  /// fallback).
  RoutedDecodeResult decodeCurrentLangOnly(Float32List samples) {
    return _decodeWith(currentLang ?? 'ja', samples);
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
