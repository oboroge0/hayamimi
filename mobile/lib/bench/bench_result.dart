/// Outcome of a single RTF benchmark run.
class BenchResult {
  const BenchResult({
    required this.audioDurationSeconds,
    required this.processingDurationSeconds,
    required this.text,
  });

  final double audioDurationSeconds;
  final double processingDurationSeconds;
  final String text;

  /// Real-time factor: processing time / audio duration.
  ///
  /// RTF < 1.0 means decoding is faster than real time (good for a live
  /// pipeline); RTF >= 1.0 means it cannot keep up with real-time audio.
  /// Returns 0 when [audioDurationSeconds] is not positive, since RTF is
  /// undefined for empty/invalid audio.
  double get rtf {
    if (audioDurationSeconds <= 0) {
      return 0;
    }
    return processingDurationSeconds / audioDurationSeconds;
  }
}
