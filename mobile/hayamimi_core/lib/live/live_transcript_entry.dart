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
    this.punctuated = false,
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

  /// `true` when this entry's text had Japanese punctuation (、 。 ？)
  /// restored into it, so a caller can tell text the punctuation model
  /// wrote from text the recognizer produced — e.g. to skip its own
  /// sentence splitting, or to show that this line is the finished one.
  ///
  /// Only refine ("清書") entries can be `true`, and only when the session
  /// was started with a `JaPunctuation` model that applies to the entry's
  /// language. Always `false` for finals and drafts, and `false` again for
  /// a refine that fell back to the fast finals' text because the merged
  /// re-decode came back too short (that fallback text is unpunctuated).
  final bool punctuated;
}
