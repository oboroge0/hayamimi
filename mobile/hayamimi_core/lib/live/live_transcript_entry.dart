/// One finalized line of live transcription: the recognized text for a
/// single VAD-bounded speech segment, plus when it was produced.
class LiveTranscriptEntry {
  const LiveTranscriptEntry({
    required this.text,
    required this.timestamp,
    this.latencyMs,
    this.lang,
    this.audioSeconds,
    this.switched = false,
  });

  final String text;
  final DateTime timestamp;

  /// Wall-clock time the decode call took for this segment, in
  /// milliseconds. Mirrors the `latency_ms` field the desktop subtitle
  /// server reports, for other-app consumers (see `lib/server/`).
  final double? latencyMs;

  /// The language this segment was decoded with (e.g. "ja", "en"), when the
  /// session used [RoutingProfile.jaSenseVoice] multilingual routing.
  /// `null` for a plain single-model session, where the caller already
  /// knows the (only) language.
  final String? lang;

  /// How much audio this entry covers, in seconds: the VAD segment's sample
  /// count for a fast-final entry, or the total buffered duration of the
  /// group a refine ("清書") pass re-decoded. `null` for a draft entry,
  /// which this package never persists or measures for duration.
  final double? audioSeconds;

  /// `true` when this entry is the reason a [RoutingProfile.jaSenseVoice]
  /// session's language changed (mirrors [RoutedDecodeResult.switched]).
  /// Always `false` for a plain single-model session and for refine
  /// entries, which re-judge a whole group rather than a single switch
  /// point.
  final bool switched;
}
