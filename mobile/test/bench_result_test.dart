import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_mobile/bench/bench_result.dart';

void main() {
  group('BenchResult.rtf', () {
    test('computes processing time over audio duration', () {
      const result = BenchResult(
        audioDurationSeconds: 10.0,
        processingDurationSeconds: 2.5,
        text: 'hello',
      );
      expect(result.rtf, closeTo(0.25, 1e-9));
    });

    test('rtf above 1.0 means slower than real time', () {
      const result = BenchResult(
        audioDurationSeconds: 1.0,
        processingDurationSeconds: 3.0,
        text: '',
      );
      expect(result.rtf, closeTo(3.0, 1e-9));
    });

    test('returns 0 for zero-length audio instead of dividing by zero', () {
      const result = BenchResult(
        audioDurationSeconds: 0.0,
        processingDurationSeconds: 1.0,
        text: '',
      );
      expect(result.rtf, 0);
    });

    test('returns 0 for negative audio duration', () {
      const result = BenchResult(
        audioDurationSeconds: -1.0,
        processingDurationSeconds: 1.0,
        text: '',
      );
      expect(result.rtf, 0);
    });
  });
}
