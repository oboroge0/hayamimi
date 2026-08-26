import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

/// Minimum speech-segment duration worth sending through the recognizer.
///
/// Silero VAD's own `minSpeechDuration` setting already filters out most
/// noise blips before a [sherpa_onnx.SpeechSegment] is ever emitted, but
/// this is a second, app-level guard: a segment that straddles a VAD
/// boundary awkwardly can still come out very short, and running a full
/// decode pass on near-silence just wastes CPU and can surface junk text.
const defaultMinDecodeDurationSeconds = 0.2;

/// Whether a VAD-emitted [segment] is worth turning into a decode call.
///
/// This is the "VAD segment -> decode input" gate: it doesn't touch the
/// recognizer, so it's plain, unit-testable logic even though
/// [sherpa_onnx.SpeechSegment] itself comes from the FFI-backed package.
bool isSegmentWorthDecoding(
  sherpa_onnx.SpeechSegment segment, {
  int sampleRate = 16000,
  double minDurationSeconds = defaultMinDecodeDurationSeconds,
}) {
  if (segment.samples.isEmpty) {
    return false;
  }
  final durationSeconds = segment.samples.length / sampleRate;
  return durationSeconds >= minDurationSeconds;
}
