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
class PcmFrameBuffer {
  PcmFrameBuffer({required this.frameSize})
    : assert(frameSize > 0, 'frameSize must be positive');

  final int frameSize;
  final List<double> _pending = <double>[];

  /// Appends new samples to the pending tail.
  void add(Float32List samples) {
    _pending.addAll(samples);
  }

  /// Removes and returns as many complete [frameSize]-length frames as are
  /// currently available, leaving any leftover samples buffered for the
  /// next call. Returns an empty list when there isn't a full frame yet.
  List<Float32List> drainFrames() {
    final frames = <Float32List>[];
    while (_pending.length >= frameSize) {
      frames.add(Float32List.fromList(_pending.sublist(0, frameSize)));
      _pending.removeRange(0, frameSize);
    }
    return frames;
  }

  /// Number of samples currently buffered but not yet enough for a frame.
  int get pendingSampleCount => _pending.length;

  /// Drops any buffered, not-yet-complete samples.
  void reset() => _pending.clear();
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
