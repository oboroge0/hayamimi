/// Pure logic for "pre-roll": prepending a moment of the audio that came
/// *before* a speech segment's detected onset to that segment, so the
/// recognizer sees the run-up to the first word instead of starting
/// mid-word.
///
/// **The problem.** Silero VAD (the model that decides "is this speech or
/// silence") reports a speech onset slightly after the speaker actually
/// started, so the samples it hands over can begin partway into the first
/// word. On the Android emulator run recorded in
/// `docs/verify/android_emulator.md`, `資料は昨日送りました` came back from the recognizer as
/// `昨日は昨日送りました` and `あしたの会議は十時からです` lost its `あしたの`
/// entirely — both utterance-initial words, both clipped by the onset.
///
/// **The fix, and where it comes from.** The desktop pipeline has kept a
/// rolling buffer of recent audio since long before this package existed
/// (`AudioHistory` / `PREROLL_S` in `scripts/realtime_transcribe.py`) and
/// prepends up to one second of it to every segment it decodes; its
/// `tests/test_asr_segment.py` pins the behaviour on the same fixture that
/// regressed above. This file is that idea in Dart.
///
/// Everything here is plain data/logic with no FFI or platform dependency,
/// so it is unit tested directly (`test/preroll_test.dart`). The glue that
/// feeds it live microphone frames and hands the extended segment to the
/// decode worker lives in `live_transcriber.dart`.
library;

import 'dart:collection';
import 'dart:typed_data';

/// How much audio from before a segment's detected onset is prepended to
/// it, in seconds, unless a caller says otherwise.
///
/// One second, the same value the desktop pipeline's `PREROLL_S` uses. It
/// is generous next to the onset lag it compensates for (a couple of
/// hundred milliseconds), and being generous is cheap: the extra audio is
/// silence or room tone the recognizer transcribes to nothing.
const double defaultPrerollSeconds = 1.0;

/// How much recent audio [PrerollHistory] keeps by default, in seconds.
///
/// Only the trailing [defaultPrerollSeconds] is ever read, so this is
/// mostly headroom; it is sized to match the native VAD's own internal
/// buffer (see `_vadBufferSeconds` in `live_vad.dart`) so a segment the
/// VAD can still produce is one this history can still find context for.
/// At 16 kHz mono float samples, 30 s is under 2 MB.
const double defaultPrerollKeepSeconds = 30.0;

/// A bounded rolling buffer of the most recently captured audio, able to
/// hand back a finished speech segment with its run-up attached.
///
/// A session pushes every frame it feeds the VAD into one of these
/// ([push]), then asks for the extended segment when the VAD finalizes one
/// ([withPreroll]). Sample positions are absolute: sample 0 is the first
/// frame ever pushed, and the buffer remembers how many samples it has
/// dropped off the front, so a position stays meaningful after old audio
/// has aged out.
///
/// Two rules keep the extra audio honest, both inherited from the desktop
/// implementation this mirrors:
///
///  * never reach back past what the buffer still holds, and
///  * never reach back into the previous segment. Two utterances a
///    half-second apart would otherwise both contain that half second, the
///    refine ("清書") pass would re-decode it twice, and a word spoken in
///    the gap could be transcribed into both lines.
class PrerollHistory {
  PrerollHistory({
    this.sampleRate = 16000,
    this.keepSeconds = defaultPrerollKeepSeconds,
  });

  /// The rate the pushed audio was captured at, in Hz. Only used to turn
  /// the seconds-valued knobs into sample counts.
  final int sampleRate;

  /// How much audio this keeps before dropping the oldest, in seconds.
  final double keepSeconds;

  // Frames as pushed, oldest first. Kept as separate lists rather than one
  // growing buffer: concatenating on every frame would copy the whole
  // history ~31 times a second, whereas a segment only needs the frames
  // its pre-roll actually spans, once, when it closes.
  final Queue<Float32List> _frames = Queue<Float32List>();
  int _bufferedSamples = 0;
  int _offset = 0;
  int _lastSegmentEnd = 0;

  /// Absolute index of the oldest sample still held: everything before it
  /// has been dropped and can no longer be prepended to anything.
  int get offset => _offset;

  /// How many samples are currently held.
  int get bufferedSamples => _bufferedSamples;

  /// Absolute index just past the end of the last segment [withPreroll]
  /// was called for -- the line no later pre-roll is allowed to cross.
  int get lastSegmentEnd => _lastSegmentEnd;

  /// Records one frame of captured audio, then drops whole frames off the
  /// front until the buffer is back within [keepSeconds].
  ///
  /// A frame is only dropped while what remains still covers the keep
  /// window, so this holds between [keepSeconds] and [keepSeconds] plus one
  /// frame -- never less than it promised, and never unbounded.
  void push(Float32List frame) {
    if (frame.isEmpty) {
      return;
    }
    _frames.add(frame);
    _bufferedSamples += frame.length;
    final keep = (keepSeconds * sampleRate).round();
    while (_frames.isNotEmpty &&
        _bufferedSamples - _frames.first.length >= keep) {
      final dropped = _frames.removeFirst();
      _bufferedSamples -= dropped.length;
      _offset += dropped.length;
    }
  }

  /// Returns [samples] with up to [prerollSeconds] of the audio recorded
  /// immediately before [segmentStartSample] prepended to it.
  ///
  /// [segmentStartSample] is where the VAD says the segment began, counted
  /// in samples from the start of the session (the same clock [push] feeds).
  /// The returned audio is what should actually be decoded and buffered for
  /// a later refine pass, so a final's reported duration covers the pre-roll
  /// too.
  ///
  /// Returns [samples] unchanged when there is nothing to prepend: an empty
  /// history, a segment that starts where the previous one ended, or
  /// `prerollSeconds: 0`, which is how a caller turns pre-roll off.
  ///
  /// Calling this also moves [lastSegmentEnd] past the segment, which is
  /// what stops the *next* call reaching back into it. Call it once per
  /// segment the VAD produces, in order, including for segments that are
  /// then dropped without being decoded -- skipping one would let the next
  /// segment's pre-roll cover audio this one already accounted for.
  Float32List withPreroll({
    required int segmentStartSample,
    required Float32List samples,
    double prerollSeconds = defaultPrerollSeconds,
  }) {
    final end = _clamp(segmentStartSample);
    var start = segmentStartSample - (prerollSeconds * sampleRate).round();
    if (start < _lastSegmentEnd) {
      start = _lastSegmentEnd;
    }
    start = _clamp(start);
    _lastSegmentEnd = segmentStartSample + samples.length;
    if (start >= end) {
      return samples;
    }
    final preroll = _read(start, end);
    final extended = Float32List(preroll.length + samples.length);
    extended.setRange(0, preroll.length, preroll);
    extended.setRange(preroll.length, extended.length, samples);
    return extended;
  }

  /// Pins an absolute sample position inside the audio still held, so a
  /// position from before the buffer slid (or from beyond what has been
  /// pushed, which a stand-in VAD in a test can report) reads as "nothing
  /// there" instead of throwing.
  int _clamp(int index) {
    final oldest = _offset;
    final newest = _offset + _bufferedSamples;
    if (index < oldest) return oldest;
    if (index > newest) return newest;
    return index;
  }

  /// Copies the absolute range [start], [end) out of the held frames.
  Float32List _read(int start, int end) {
    final out = Float32List(end - start);
    var framePosition = _offset;
    var written = 0;
    for (final frame in _frames) {
      final frameEnd = framePosition + frame.length;
      if (frameEnd > start && framePosition < end) {
        final from = start > framePosition ? start - framePosition : 0;
        final to = end < frameEnd ? end - framePosition : frame.length;
        out.setRange(written, written + (to - from), frame, from);
        written += to - from;
      }
      framePosition = frameEnd;
      if (framePosition >= end) {
        break;
      }
    }
    return out;
  }
}
