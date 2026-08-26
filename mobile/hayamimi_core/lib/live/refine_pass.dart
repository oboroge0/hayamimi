import 'dart:typed_data';

/// Pure logic for the Live screen's "refine" (清書) pass: a second decode
/// over several already-finalized VAD segments' audio, run together instead
/// of one at a time, the same "re-decode with more context" trick the
/// desktop pipeline's `Refiner` class uses (see
/// `scripts/realtime_transcribe.py`, `GROUP_GAP_S`/`GROUP_MAX_S`). No
/// punctuation-restoration model is added here — this is re-decode only.
///
/// Everything in this file is plain data/logic with no FFI or platform
/// dependency, so it's unit tested directly. The FFI glue (actually running
/// the recognizer over the combined audio) lives in `live_transcriber.dart`.

/// Hard cap on how much audio [RefineBuffer] holds at once, in seconds.
///
/// This is a RAM safety net, independent of when a refine actually fires:
/// a phone can't afford to accumulate unbounded raw Float32 audio in memory
/// the way the desktop's `AudioHistory` can. Once the buffer would exceed
/// this, the oldest segment is dropped even if no refine has happened yet.
const double defaultRefineBufferMaxSeconds = 60.0;

/// One finalized VAD segment kept around for a later refine pass: its audio
/// samples plus the fast (single-segment) transcription already produced
/// for it.
class RefineSegment {
  const RefineSegment({
    required this.samples,
    required this.text,
    required this.capturedAt,
  });

  final Float32List samples;
  final String text;
  final DateTime capturedAt;

  double durationSeconds(int sampleRate) => samples.length / sampleRate;
}

/// Ring buffer of finalized segments awaiting a refine pass.
///
/// Segments are appended as they're decoded on the fast path; a refine
/// consumes everything currently buffered via [takeAll]. Independently of
/// that, the buffer slides (drops from the front) whenever the total
/// buffered duration would exceed [maxDurationSeconds], so a user who never
/// triggers a refine (manual or auto) can't grow this without bound.
class RefineBuffer {
  RefineBuffer({
    this.sampleRate = 16000,
    this.maxDurationSeconds = defaultRefineBufferMaxSeconds,
  });

  final int sampleRate;
  final double maxDurationSeconds;
  final List<RefineSegment> _segments = <RefineSegment>[];

  bool get isEmpty => _segments.isEmpty;
  bool get isNotEmpty => _segments.isNotEmpty;

  /// Read-only view of the currently buffered segments, oldest first.
  List<RefineSegment> get segments => List.unmodifiable(_segments);

  double get totalDurationSeconds =>
      _segments.fold(0.0, (sum, s) => sum + s.durationSeconds(sampleRate));

  /// Appends [segment], then drops the oldest buffered segments (if any)
  /// until the total is back at or under [maxDurationSeconds]. Always keeps
  /// at least the segment just added, even if it alone exceeds the cap.
  void add(RefineSegment segment) {
    _segments.add(segment);
    while (_segments.length > 1 && totalDurationSeconds > maxDurationSeconds) {
      _segments.removeAt(0);
    }
  }

  /// Removes and returns every currently buffered segment, oldest first,
  /// leaving the buffer empty. This is how a refine pass claims what it's
  /// about to decode.
  List<RefineSegment> takeAll() {
    final taken = List<RefineSegment>.of(_segments);
    _segments.clear();
    return taken;
  }

  /// Drops everything buffered without producing a refine (e.g. on stop()).
  void clear() => _segments.clear();
}

/// Concatenates [segments]' audio samples into one buffer, in order. This is
/// the "merge the group's audio" step the desktop `Refiner` does by slicing
/// its rolling `AudioHistory`; on the phone there's no rolling history
/// buffer to slice, so the segments' own samples are joined directly.
Float32List combineSegmentSamples(List<RefineSegment> segments) {
  final totalLength = segments.fold<int>(0, (sum, s) => sum + s.samples.length);
  final combined = Float32List(totalLength);
  var offset = 0;
  for (final segment in segments) {
    combined.setRange(offset, offset + segment.samples.length, segment.samples);
    offset += segment.samples.length;
  }
  return combined;
}

/// Joins segments' fast (per-segment) text with single spaces, skipping any
/// blank ones. Used both as the refine's fallback text (see
/// [isRefineTextTooShort]) and to show what fell out of the fast path for
/// comparison.
String combineSegmentFastText(List<RefineSegment> segments) {
  return segments
      .map((s) => s.text.trim())
      .where((t) => t.isNotEmpty)
      .join(' ');
}

/// Whether a merged re-decode came back suspiciously short compared to the
/// fast finals it's replacing, mirroring the desktop `Refiner`'s guard: "a
/// merged re-decode must never LOSE content; if it comes back much shorter
/// than the fast finals combined, trust those" (scripts/realtime_transcribe.py).
/// Length is compared in UTF-16 code units, which is good enough for the
/// relative-shrink check this guards (not meant as a linguistic measure).
bool isRefineTextTooShort(
  String refineText,
  String fastJoinedText, {
  double minRatio = 0.7,
}) {
  if (fastJoinedText.isEmpty) return false;
  return refineText.trim().length < minRatio * fastJoinedText.length;
}

/// Default silence gap (in seconds) that fires an auto-refine, when auto
/// mode is on. Deliberately longer than the desktop's `GROUP_GAP_S` (2.0s):
/// on a phone every refine is a full re-decode running on battery, and
/// firing on a 2s pause would refine mid-conversation constantly. Auto mode
/// defaults to OFF for the same reason (see `live_transcriber.dart`); a
/// user who wants a refine sooner can always tap the manual button.
const double defaultAutoRefineSilenceSeconds = 4.0;

/// Default buffered-duration ceiling (in seconds) that fires an auto-refine
/// even without a silence gap, so a very long uninterrupted utterance still
/// gets refined periodically instead of only at [defaultRefineBufferMaxSeconds]
/// when the buffer starts sliding (and silently losing early audio).
/// Mirrors the desktop's `GROUP_MAX_S` (25.0s) at a slightly shorter, more
/// battery-conscious value.
const double defaultAutoRefineMaxBufferedSeconds = 20.0;

/// Whether an auto-refine should fire right now, mirroring the desktop
/// pipeline's `Refiner.maybe_refine` due check (gap-since-last-segment OR
/// buffered-duration-ceiling) but with phone-tuned defaults. Pure function:
/// the caller (a `Timer` in `LiveTranscriber`) supplies the current state.
bool isAutoRefineDue({
  required Duration sinceLastSegment,
  required double bufferedDurationSeconds,
  double silenceSeconds = defaultAutoRefineSilenceSeconds,
  double maxBufferedSeconds = defaultAutoRefineMaxBufferedSeconds,
}) {
  if (bufferedDurationSeconds <= 0) {
    return false;
  }
  final sinceLastSegmentSeconds = sinceLastSegment.inMilliseconds / 1000.0;
  return sinceLastSegmentSeconds >= silenceSeconds ||
      bufferedDurationSeconds >= maxBufferedSeconds;
}
