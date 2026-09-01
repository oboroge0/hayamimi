/// Pure logic for the Live screen's "draft" (発話中の暫定字幕) pass: while a
/// VAD segment is still in progress (the speaker hasn't paused yet), decode
/// the audio accumulated so far on a timer and show it as an in-progress
/// line, the same "partial" experience `scripts/realtime_transcribe.py`
/// gives the desktop dashboard (see `PARTIAL_EVERY_S`/`PARTIAL_WINDOW_S`
/// there). Unlike the finalized/refine passes, a draft is never stored --
/// it's overwritten by the next draft or cleared by the next final.
///
/// Everything in this file is plain data/logic with no FFI or platform
/// dependency, so it's unit tested directly. The FFI glue (accumulating mic
/// frames while the VAD is mid-segment, and actually decoding them) lives in
/// `live_transcriber.dart`.
library;

import 'dart:typed_data';

/// How often (wall-clock) a draft re-decode should fire while a VAD segment
/// is in progress. Coarser than the desktop's `PARTIAL_EVERY_S` (0.5s):
/// on a phone the draft decode competes with the *same* recognizer/CPU the
/// fast-final and refine passes use, and every extra decode is battery and
/// heat, so this trades a bit of "typewriter" smoothness for headroom.
const double defaultDraftIntervalSeconds = 1.0;

/// Cap on how much trailing audio a draft decode re-processes, mirroring
/// the desktop's `PARTIAL_WINDOW_S`: without this, a long uninterrupted
/// utterance would make every subsequent draft decode slower (more audio to
/// re-decode) right when the interval needs to stay snappy.
const double defaultDraftWindowSeconds = 8.0;

/// Minimum accumulated audio, in seconds, before a draft decode is worth
/// running at all -- mirrors [isSegmentWorthDecoding]'s guard against
/// wasting a decode call on near-silence.
const double defaultMinDraftAudioSeconds = 0.25;

/// Whether it's time to fire another draft decode.
///
/// [isDecoding] is the "skip, don't queue" guard: [LiveTranscriber] sends
/// every decode -- fast-final, refine and draft alike -- to a single
/// worker isolate that serves them one at a time. If anything is already
/// outstanding there, this returns `false` rather than letting drafts pile
/// up behind it: a queued draft would only delay whichever decode actually
/// matters (the next final), and would be superseded by that final anyway.
/// The caller is expected to just try again on the next frame, which
/// naturally catches up once the worker is free.
/// Drops whole frames off the front of [frames] until their total sample
/// count is within [maxSeconds] of [sampleRate] audio (or exactly one frame
/// remains, even if that single frame alone exceeds the budget).
///
/// [LiveTranscriber] calls this every time a new frame is appended to its
/// draft accumulator, so the accumulator itself never grows past this
/// window. Without it, a long uninterrupted utterance made the accumulator
/// -- and therefore the concatFloat32Lists pass over the whole thing that
/// [_runDraftDecode] repeated on every draft tick -- grow without bound,
/// even though [capDraftWindow] only ever decodes the trailing
/// [defaultDraftWindowSeconds] of it anyway: total work across the
/// utterance was effectively O(n^2) in its length. Trimming at the source
/// keeps each tick's concatenation cheap and bounded instead.
List<Float32List> slideDraftFrames(
  List<Float32List> frames, {
  required int sampleRate,
  double maxSeconds = defaultDraftWindowSeconds,
}) {
  final maxSamples = (maxSeconds * sampleRate).round();
  var total = frames.fold<int>(0, (sum, f) => sum + f.length);
  var start = 0;
  while (total > maxSamples && start < frames.length - 1) {
    total -= frames[start].length;
    start++;
  }
  return start == 0 ? frames : frames.sublist(start);
}

bool isDraftDue({
  required bool isDecoding,
  required Duration sinceLastDraft,
  double intervalSeconds = defaultDraftIntervalSeconds,
}) {
  if (isDecoding) {
    return false;
  }
  return sinceLastDraft.inMilliseconds / 1000.0 >= intervalSeconds;
}
