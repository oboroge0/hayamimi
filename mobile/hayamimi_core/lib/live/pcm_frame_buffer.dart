import 'dart:typed_data';

/// Converts a chunk of little-endian PCM16 mono bytes (the format the
/// `record` package's `startStream` emits when configured with
/// `AudioEncoder.pcm16bits`) into normalized Float32 samples in
/// `[-1.0, 1.0]`, the format sherpa-onnx's VAD and recognizer APIs expect.
Float32List pcm16BytesToFloat32(Uint8List bytes) {
  final byteData = ByteData.sublistView(bytes);
  final sampleCount = bytes.length ~/ 2;
  final samples = Float32List(sampleCount);
  for (var i = 0; i < sampleCount; i++) {
    final sample = byteData.getInt16(i * 2, Endian.little);
    samples[i] = sample / 32768.0;
  }
  return samples;
}

/// Accumulates arbitrary-length Float32 sample chunks and re-slices them
/// into fixed-size frames.
///
/// The mic stream delivers chunks whose size is decided by the OS/plugin,
/// not by us, but sherpa-onnx's Silero VAD requires a constant window size
/// per `acceptWaveform` call (e.g. 512 samples at 16kHz). This buffer sits
/// between the two: push whatever arrives, pull out only complete frames,
/// and keep the remainder for next time.
///
/// Backed by a single growable [Float32List] used as a ring: unread samples
/// live in `_buffer[_start, _start + _length)`, [add] appends after that
/// range, and [drainFrames] just advances `_start`/`_length` -- no per-frame
/// shifting of the whole backing array. This matters because [add] and
/// [drainFrames] run on every mic callback (dozens of times a second for
/// the life of a session): the previous `List<double>` + `removeRange(0,
/// frameSize)` implementation shifted every remaining buffered sample down
/// on every drained frame, an O(n) cost paid on the hot path for what's
/// normally well under one frame's worth of carryover. The backing array
/// only grows (doubling) or gets compacted back to offset 0 when there
/// isn't room to append in place, which happens rarely relative to the
/// steady stream of add/drain calls.
class PcmFrameBuffer {
  PcmFrameBuffer({required this.frameSize})
    : assert(frameSize > 0, 'frameSize must be positive'),
      _buffer = Float32List(frameSize * _initialCapacityFrames);

  static const int _initialCapacityFrames = 8;

  final int frameSize;

  Float32List _buffer;
  int _start = 0; // index of the first unread sample in _buffer
  int _length = 0; // count of unread samples starting at _start

  /// Appends new samples to the pending tail.
  void add(Float32List samples) {
    if (samples.isEmpty) {
      return;
    }
    _makeRoomFor(samples.length);
    final writeStart = _start + _length;
    _buffer.setRange(writeStart, writeStart + samples.length, samples);
    _length += samples.length;
  }

  /// Ensures `_buffer` can hold `_length + additional` samples starting at
  /// `_start`, compacting the unread region to offset 0 (if that alone
  /// makes enough room) or growing the backing array (doubling capacity
  /// until it fits) otherwise.
  void _makeRoomFor(int additional) {
    if (_start + _length + additional <= _buffer.length) {
      return;
    }
    final needed = _length + additional;
    if (needed <= _buffer.length) {
      // The unread region fits at the front; just slide it there instead
      // of growing.
      final compacted = Float32List.sublistView(
        _buffer,
        _start,
        _start + _length,
      );
      _buffer.setRange(0, _length, compacted);
      _start = 0;
      return;
    }
    var newCapacity = _buffer.isEmpty ? frameSize : _buffer.length;
    while (newCapacity < needed) {
      newCapacity *= 2;
    }
    final newBuffer = Float32List(newCapacity);
    newBuffer.setRange(
      0,
      _length,
      Float32List.sublistView(_buffer, _start, _start + _length),
    );
    _buffer = newBuffer;
    _start = 0;
  }

  /// Removes and returns as many complete [frameSize]-length frames as are
  /// currently available, leaving any leftover samples buffered for the
  /// next call. Returns an empty list when there isn't a full frame yet.
  List<Float32List> drainFrames() {
    final frames = <Float32List>[];
    while (_length >= frameSize) {
      frames.add(
        Float32List.fromList(
          _buffer.sublist(_start, _start + frameSize),
        ),
      );
      _start += frameSize;
      _length -= frameSize;
    }
    return frames;
  }

  /// Number of samples currently buffered but not yet enough for a frame.
  int get pendingSampleCount => _length;

  /// Drops any buffered, not-yet-complete samples.
  void reset() {
    _start = 0;
    _length = 0;
  }
}

/// Concatenates [chunks] into a single Float32List, in order. Used to turn
/// the frames accumulated for an in-progress draft decode (a `List` because
/// they arrive one VAD window at a time) into one buffer to feed the
/// recognizer, the same shape [combineSegmentSamples] produces for the
/// refine pass's finalized segments.
Float32List concatFloat32Lists(List<Float32List> chunks) {
  final totalLength = chunks.fold<int>(0, (sum, c) => sum + c.length);
  final combined = Float32List(totalLength);
  var offset = 0;
  for (final chunk in chunks) {
    combined.setRange(offset, offset + chunk.length, chunk);
    offset += chunk.length;
  }
  return combined;
}

/// Returns the trailing [maxSeconds] of [samples] (or all of it, if
/// shorter) -- the draft pass's equivalent of the desktop's
/// `PARTIAL_WINDOW_S` cap (see `draft_pass.dart`), kept here alongside the
/// other sample-buffer helpers since it's plain Float32List slicing.
Float32List capDraftWindow(
  Float32List samples, {
  required int sampleRate,
  required double maxSeconds,
}) {
  final maxSamples = (maxSeconds * sampleRate).round();
  if (samples.length <= maxSamples) {
    return samples;
  }
  return Float32List.sublistView(samples, samples.length - maxSamples);
}
