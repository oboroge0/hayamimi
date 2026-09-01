import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/vad_sensitivity.dart';

void main() {
  group('VadSensitivity', () {
    test('the default constructor matches the desktop pipeline\'s tuned values', () {
      // minSilenceSeconds/maxSpeechSeconds are the desktop's
      // (scripts/realtime_transcribe.py), not sherpa-onnx's 0.5/5.0: the
      // stock silence default merged three spoken sentences into one
      // segment on an Android emulator and the recognizer then returned
      // only the last of them. threshold/minSpeechSeconds are
      // sherpa-onnx's own, which the desktop also leaves alone.
      final sensitivity = VadSensitivity();
      expect(sensitivity.threshold, 0.5);
      expect(sensitivity.minSilenceSeconds, 0.35);
      expect(sensitivity.minSpeechSeconds, 0.25);
      expect(sensitivity.maxSpeechSeconds, 12.0);
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

  group('isVadBuildStale', () {
    test('false when the generation is unchanged since the build started', () {
      expect(
        isVadBuildStale(buildGeneration: 3, currentGeneration: 3),
        isFalse,
      );
    });

    test(
      'true once the generation has advanced (stop() torn down native '
      'state while the rebuild was in flight)',
      () {
        expect(
          isVadBuildStale(buildGeneration: 3, currentGeneration: 4),
          isTrue,
        );
      },
    );

    test(
      'true after a stop()+start() cycle bumped the generation twice',
      () {
        expect(
          isVadBuildStale(buildGeneration: 3, currentGeneration: 5),
          isTrue,
        );
      },
    );

    test('false at generation 0 (nothing has ever been built/torn down)', () {
      expect(
        isVadBuildStale(buildGeneration: 0, currentGeneration: 0),
        isFalse,
      );
    });
  });
}
