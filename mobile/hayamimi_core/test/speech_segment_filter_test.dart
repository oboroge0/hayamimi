import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/speech_segment_filter.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

sherpa_onnx.SpeechSegment _segmentOfDuration(
  double seconds, {
  int sampleRate = 16000,
}) {
  final sampleCount = (seconds * sampleRate).round();
  return sherpa_onnx.SpeechSegment(samples: Float32List(sampleCount), start: 0);
}

void main() {
  group('isSegmentWorthDecoding', () {
    test('rejects an empty segment', () {
      final segment = sherpa_onnx.SpeechSegment(
        samples: Float32List(0),
        start: 0,
      );
      expect(isSegmentWorthDecoding(segment), isFalse);
    });

    test('rejects a segment shorter than the minimum duration', () {
      final segment = _segmentOfDuration(0.1);
      expect(isSegmentWorthDecoding(segment, minDurationSeconds: 0.2), isFalse);
    });

    test('accepts a segment at or above the minimum duration', () {
      final segment = _segmentOfDuration(0.2);
      expect(isSegmentWorthDecoding(segment, minDurationSeconds: 0.2), isTrue);

      final longerSegment = _segmentOfDuration(1.5);
      expect(
        isSegmentWorthDecoding(longerSegment, minDurationSeconds: 0.2),
        isTrue,
      );
    });

    test('uses defaultMinDecodeDurationSeconds when not overridden', () {
      final justUnder = _segmentOfDuration(
        defaultMinDecodeDurationSeconds - 0.01,
      );
      final justOver = _segmentOfDuration(
        defaultMinDecodeDurationSeconds + 0.01,
      );

      expect(isSegmentWorthDecoding(justUnder), isFalse);
      expect(isSegmentWorthDecoding(justOver), isTrue);
    });

    test('respects a custom sample rate when computing duration', () {
      // 1600 samples at 8kHz is 0.2s, but at 16kHz it's only 0.1s.
      final segment = sherpa_onnx.SpeechSegment(
        samples: Float32List(1600),
        start: 0,
      );

      expect(
        isSegmentWorthDecoding(
          segment,
          sampleRate: 8000,
          minDurationSeconds: 0.2,
        ),
        isTrue,
      );
      expect(
        isSegmentWorthDecoding(
          segment,
          sampleRate: 16000,
          minDurationSeconds: 0.2,
        ),
        isFalse,
      );
    });
  });
}
