import 'dart:typed_data';

/// Minimum speech-segment duration worth sending through the recognizer.
///
/// Silero VAD's own `minSpeechDuration` setting already filters out most
/// noise blips before a segment is ever emitted, but this is a second,
/// app-level guard: a segment that straddles a VAD boundary awkwardly can
/// still come out very short, and running a full decode pass on
/// near-silence just wastes CPU and can surface junk text.
const defaultMinDecodeDurationSeconds = 0.2;

/// Whether a VAD-emitted segment's [samples] are worth turning into a
/// decode call.
///
/// This is the "VAD segment -> decode input" gate. It takes the samples
/// rather than a VAD segment object so it has no dependency on the
/// FFI-backed package at all, which is also what lets `LiveTranscriber`
/// run against a stand-in VAD in tests.
bool isSegmentWorthDecoding(
  Float32List samples, {
  int sampleRate = 16000,
  double minDurationSeconds = defaultMinDecodeDurationSeconds,
}) {
  if (samples.isEmpty) {
    return false;
  }
  final durationSeconds = samples.length / sampleRate;
  return durationSeconds >= minDurationSeconds;
}
