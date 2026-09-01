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

/// Silero VAD's four tunable knobs.
///
/// The defaults were sherpa-onnx's own (`minSilenceSeconds` 0.5,
/// `maxSpeechSeconds` 5.0) until an Android emulator run showed what that
/// costs. On a three-sentence Japanese recording whose pauses are just
/// under half a second, the 0.5 s silence default merged all three
/// sentences into one 6.13 s segment, and the recognizer given that segment
/// returned only the last sentence -- a caption that is wrong, not merely
/// rough, with nothing in the output to say two sentences went missing.
/// The desktop pipeline this package is the mobile port of does not have
/// that problem because it does not use the stock values: it runs
/// `--min-silence 0.35` and `--max-speech 12.0`
/// (`scripts/realtime_transcribe.py`), and its own comment records that
/// 0.35 s measured no worse for accuracy than 0.5 s while finalizing a
/// segment 150 ms sooner. Re-running the same audio through this package
/// with 0.35 s split it into three segments and all three sentences came
/// out.
///
/// So the two defaults here are the desktop's, not sherpa-onnx's:
/// [minSilenceSeconds] is 0.35 and [maxSpeechSeconds] is 12.0.
/// [threshold] and [minSpeechSeconds] are unchanged from sherpa-onnx's
/// values, which the desktop also leaves alone. For a caller, this means
/// segments finalize on shorter pauses than before -- more transcript
/// lines, each of them shorter, and each decoded from audio the recognizer
/// can actually handle. Pass explicit values to get the old behavior back.
class VadSensitivity {
  VadSensitivity({
    this.threshold = 0.5,
    this.minSilenceSeconds = 0.35,
    this.minSpeechSeconds = 0.25,
    this.maxSpeechSeconds = 12.0,
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
  ///
  /// Defaults to 0.35 s, matching the desktop pipeline. Raising it back
  /// toward sherpa-onnx's own 0.5 s risks merging consecutive sentences
  /// into one segment, which this repo's ja recognizer answers by
  /// transcribing only the last of them -- see this class's doc.
  final double minSilenceSeconds;

  /// Minimum speech duration before a detected segment is kept at all --
  /// shorter blips are discarded as noise before they ever reach
  /// [LiveTranscriber.entries].
  final double minSpeechSeconds;

  /// Roughly how long a single segment is allowed to run before Silero VAD
  /// closes it mid-speech, so an unbroken monologue still produces timely
  /// lines instead of one enormous one at the end.
  ///
  /// This is sherpa-onnx's `max_speech_duration`, and it is not a hard
  /// cap. On the Android emulator run behind these defaults, a session
  /// configured with 5.0 s emitted a 6.134 s segment; the value influences
  /// when the VAD force-closes a segment rather than bounding what it can
  /// hand over. Do not size a buffer, a timeout, or a UI element on the
  /// assumption that no segment can exceed it.
  ///
  /// Defaults to 12.0 s, matching the desktop pipeline, which force-splits
  /// breathless commentary at that length and lets the refine ("清書") pass
  /// re-merge the group afterwards.
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
///  * [busy] -- never swap while a decode is outstanding at the decode
///    worker, mirroring [isDraftDue]'s `isDecoding` guard in
///    `draft_pass.dart`. This one is now belt and braces rather than
///    strictly required: the recognizers moved to their own isolate, so a
///    VAD swap on the caller's isolate can no longer interleave with a
///    decode at all. It is kept because it costs nothing (between
///    utterances the queue is empty anyway) and because it keeps the swap
///    point defined by one rule -- "when nothing is happening" -- rather
///    than by which piece of state happens to be shared today.
///
/// Pure function so the "only swap at a segment boundary" decision is
/// unit-testable without the sherpa-onnx native VAD itself -- the actual
/// swap (rebuilding off-isolate, polling this check, freeing the old
/// handle) lives in `LiveTranscriber.setVadSensitivity`.
bool shouldSwapVadNow({required bool speechActive, required bool busy}) {
  return !speechActive && !busy;
}

/// Whether a VAD rebuild that started against [buildGeneration] should be
/// discarded (freed, never installed) instead of swapped in, because the
/// owning session's native state has since moved on to
/// [currentGeneration] -- torn down, or torn down and rebuilt by a fresh
/// `start()`, while the rebuild's own `buildVadOffIsolate` await was still
/// in flight.
///
/// Why this matters: `setVadSensitivity`'s rebuild is a genuinely
/// long-running async call (a real off-isolate model load), so a `stop()`
/// followed immediately by a new `start()` can easily complete *during*
/// that await. Without this check, the stale rebuild -- built against the
/// OLD session's model path/sensitivity -- would land after the new
/// session is already up, silently replacing (and freeing) the new
/// session's own freshly built VAD with one that doesn't belong to it.
///
/// `LiveTranscriber` bumps a monotonically increasing generation counter
/// every time it builds or tears down native state, captures the current
/// value right before starting a rebuild, and compares it here once the
/// rebuild's await resolves. Pure equality check, so it's unit-testable
/// without the native VAD itself, same as [shouldSwapVadNow].
bool isVadBuildStale({
  required int buildGeneration,
  required int currentGeneration,
}) {
  return buildGeneration != currentGeneration;
}
