/// One finalized line of live transcription: the recognized text for a
/// single VAD-bounded speech segment, plus when it was produced.
class LiveTranscriptEntry {
  const LiveTranscriptEntry({
    required this.text,
    required this.timestamp,
    this.latencyMs,
  });

  final String text;
  final DateTime timestamp;

  /// Wall-clock time the decode call took for this segment, in
  /// milliseconds. Mirrors the `latency_ms` field the desktop subtitle
  /// server reports, for other-app consumers (see `lib/server/`).
  final double? latencyMs;
}
