/// Outcome of decoding one manifest entry during a batch manifest eval.
///
/// Mirrors the shape scripts/eval_accuracy.py expects when scoring a run:
/// `wav`, `lang`, `ref` echo the manifest entry back so the PC-side scorer
/// doesn't need the manifest file too, and `hyp`/timings are what the
/// on-device recognizer produced.
class ManifestEvalResult {
  const ManifestEvalResult({
    required this.wav,
    required this.lang,
    required this.ref,
    required this.hyp,
    required this.audioDurationSeconds,
    required this.decodeSeconds,
    this.detectedLang,
  });

  final String wav;
  final String lang;
  final String ref;
  final String hyp;
  final double audioDurationSeconds;
  final double decodeSeconds;

  /// The language [RoutingProfile.jaSenseVoice] routing resolved this clip
  /// to, for scoring language-routing accuracy against the manifest's
  /// ground-truth `lang` field. `null` for a plain (non-routed) manifest
  /// eval run, where only one model/language was ever in play.
  final String? detectedLang;

  /// Real-time factor for this clip. Not meaningful across devices — see
  /// docs/design/mobile_quantization.md's on-emulator accuracy parity note — but useful for
  /// spotting an outlier clip within a single run.
  double get rtf =>
      audioDurationSeconds > 0 ? decodeSeconds / audioDurationSeconds : 0;

  Map<String, Object?> toJson() => {
    'wav': wav,
    'lang': lang,
    'ref': ref,
    'hyp': hyp,
    'audio_s': audioDurationSeconds,
    'decode_s': decodeSeconds,
    'rtf': rtf,
    if (detectedLang != null) 'detected_lang': detectedLang,
  };

  factory ManifestEvalResult.fromJson(Map<String, Object?> json) {
    return ManifestEvalResult(
      wav: json['wav'] as String,
      lang: json['lang'] as String,
      ref: json['ref'] as String,
      hyp: json['hyp'] as String,
      audioDurationSeconds: (json['audio_s'] as num).toDouble(),
      decodeSeconds: (json['decode_s'] as num).toDouble(),
      detectedLang: json['detected_lang'] as String?,
    );
  }
}

/// One entry parsed from a manifest.json file (see testdata/eval_real).
class ManifestEntry {
  const ManifestEntry({required this.wav, required this.lang, required this.ref});

  final String wav;
  final String lang;
  final String ref;

  factory ManifestEntry.fromJson(Map<String, Object?> json) {
    return ManifestEntry(
      wav: json['wav'] as String,
      lang: json['lang'] as String,
      ref: json['ref'] as String,
    );
  }
}
