import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/vad_sensitivity.dart';

void main() {
  group('VadSensitivity', () {
    test('default constructor reproduces sherpa-onnx\'s own VAD defaults', () {
      final sensitivity = VadSensitivity();
      expect(sensitivity.threshold, 0.5);
      expect(sensitivity.minSilenceSeconds, 0.5);
      expect(sensitivity.minSpeechSeconds, 0.25);
      expect(sensitivity.maxSpeechSeconds, 5.0);
    });

    test('accepts and reads back custom values', () {
      final sensitivity = VadSensitivity(
        threshold: 0.7,
        minSilenceSeconds: 0.8,
        minSpeechSeconds: 0.3,
        maxSpeechSeconds: 10.0,
      );
      expect(sensitivity.threshold, 0.7);
      expect(sensitivity.minSilenceSeconds, 0.8);
      expect(sensitivity.minSpeechSeconds, 0.3);
      expect(sensitivity.maxSpeechSeconds, 10.0);
    });

    test('threshold of exactly 1.0 is accepted (upper bound inclusive)', () {
      expect(() => VadSensitivity(threshold: 1.0), returnsNormally);
    });

    test('threshold of 0 throws ArgumentError', () {
      expect(() => VadSensitivity(threshold: 0), throwsArgumentError);
    });

    test('threshold above 1 throws ArgumentError', () {
      expect(() => VadSensitivity(threshold: 1.1), throwsArgumentError);
    });

    test('a negative threshold throws ArgumentError', () {
      expect(() => VadSensitivity(threshold: -0.1), throwsArgumentError);
    });

    test('a NaN threshold throws ArgumentError', () {
      expect(() => VadSensitivity(threshold: double.nan), throwsArgumentError);
    });

    test('a non-positive minSilenceSeconds throws ArgumentError', () {
      expect(
        () => VadSensitivity(minSilenceSeconds: 0),
        throwsArgumentError,
      );
      expect(
        () => VadSensitivity(minSilenceSeconds: -1),
        throwsArgumentError,
      );
    });

    test('a non-positive minSpeechSeconds throws ArgumentError', () {
      expect(() => VadSensitivity(minSpeechSeconds: 0), throwsArgumentError);
    });

    test('a non-finite maxSpeechSeconds throws ArgumentError', () {
      expect(
        () => VadSensitivity(maxSpeechSeconds: double.infinity),
        throwsArgumentError,
      );
    });
  });

  group('shouldSwapVadNow', () {
    test('true when idle (no active segment, nothing decoding)', () {
      expect(
        shouldSwapVadNow(speechActive: false, busy: false),
        isTrue,
      );
    });

    test('false mid-segment, even if nothing is decoding', () {
      expect(
        shouldSwapVadNow(speechActive: true, busy: false),
        isFalse,
      );
    });

    test('false while a decode is in flight, even between segments', () {
      expect(
        shouldSwapVadNow(speechActive: false, busy: true),
        isFalse,
      );
    });

    test('false when both mid-segment and decoding', () {
      expect(
        shouldSwapVadNow(speechActive: true, busy: true),
        isFalse,
      );
    });
  });
}
