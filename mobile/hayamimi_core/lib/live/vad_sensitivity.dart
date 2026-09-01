/// Pure logic for the Live screen's VAD (voice activity detection)
/// sensitivity: how eagerly Silero VAD (the model that decides "is this
/// speech or silence") starts/ends a segment. `LiveTranscriber` shipped this
/// pinned at sherpa-onnx's own defaults with no way for a host app to change
/// it -- a noisy room or a soft-spoken user needs different values than the
/// defaults were tuned for, so this makes the four knobs configurable both
/// at [LiveTranscriber.start] and at runtime via
/// `LiveTranscriber.setVadSensitivity`.
///
/// Everything in this file is plain data/logic with no FFI dependency (even
/// though what it configures -- the native Silero VAD model -- very much
/// has one), so it's unit tested directly. The FFI glue that actually
/// rebuilds the VAD off-isolate and swaps it into a running session lives in
/// `live_transcriber.dart`.
library;

/// Silero VAD's four tunable knobs, with sherpa-onnx's own defaults as this
/// class's defaults so passing `VadSensitivity()` reproduces today's
/// unconfigurable behavior exactly.
class VadSensitivity {
  VadSensitivity({
    this.threshold = 0.5,
    this.minSilenceSeconds = 0.5,
    this.minSpeechSeconds = 0.25,
    this.maxSpeechSeconds = 5.0,
  }) {
    if (threshold <= 0 || threshold > 1 || !threshold.isFinite) {
      throw ArgumentError.value(
        threshold,
        'threshold',
        'must be in (0, 1] (Silero VAD\'s speech-probability cutoff)',
      );
    }
    _requirePositive(minSilenceSeconds, 'minSilenceSeconds');
    _requirePositive(minSpeechSeconds, 'minSpeechSeconds');
    _requirePositive(maxSpeechSeconds, 'maxSpeechSeconds');
  }

  /// Speech-probability cutoff Silero VAD's own model output must clear for
  /// a frame to count as speech. Higher = less sensitive (fewer false
  /// positives from background noise, but soft speech more easily missed).
  final double threshold;

  /// How long a silence has to last before an in-progress segment is
  /// considered finished and finalized. Lower = shorter pauses split
  /// segments; higher = a speaker's brief pauses stay inside one segment.
  final double minSilenceSeconds;

  /// Minimum speech duration before a detected segment is kept at all --
  /// shorter blips are discarded as noise before they ever reach
  /// [LiveTranscriber.entries].
  final double minSpeechSeconds;

  /// Hard cap on how long a single segment is allowed to run before Silero
  /// VAD force-closes it, even mid-speech, so one very long utterance can't
  /// block segmentation indefinitely.
  final double maxSpeechSeconds;

  static void _requirePositive(double value, String name) {
    if (!value.isFinite || value <= 0) {
      throw ArgumentError.value(value, name, 'must be a positive, finite number');
    }
  }
}

/// Whether it's safe to swap a newly built VAD into a running session right
/// now.
///
/// Two guards, both required:
///
///  * [speechActive] -- never swap mid-segment. The VAD instance that
///    started tracking a speech segment is the only one holding that
///    segment's internal state (Silero VAD is stateful across
///    `acceptWaveform` calls); swapping to a fresh instance partway through
///    would silently truncate or corrupt whatever segment is in progress.
///    The caller keeps feeding audio to the old VAD until the segment closes
///    (`speechActive` goes false), then swaps.
///  * [busy] -- never swap while a decode (final/refine/draft) is in
///    flight, mirroring [isDraftDue]'s `isDecoding` guard in
///    `draft_pass.dart`: `LiveTranscriber` reads/writes its native handles
///    from a single call stack at a time by convention, and a swap
///    interleaved with a decode would violate that.
///
/// Pure function so the "only swap at a segment boundary" decision is
/// unit-testable without the sherpa-onnx native VAD itself -- the actual
/// swap (rebuilding off-isolate, polling this check, freeing the old
/// handle) lives in `LiveTranscriber.setVadSensitivity`.
bool shouldSwapVadNow({required bool speechActive, required bool busy}) {
  return !speechActive && !busy;
}
