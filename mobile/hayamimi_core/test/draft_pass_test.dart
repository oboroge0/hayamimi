import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/draft_pass.dart';

Float32List _frame(int length) => Float32List(length);

void main() {
  group('slideDraftFrames', () {
    const sampleRate = 16000;

    test('keeps everything under the window budget untouched', () {
      final frames = [_frame(1000), _frame(1000)];
      final result = slideDraftFrames(
        frames,
        sampleRate: sampleRate,
        maxSeconds: 1.0, // 16000 samples
      );
      expect(result, same(frames)); // no trim needed: same list, no copy
    });

    test('drops whole frames off the front once the budget is exceeded', () {
      // 3 frames of 8000 samples each = 24000 samples = 1.5s; window = 1.0s
      // (16000 samples) -- must drop the oldest frame(s) to fit, keeping
      // the most recent audio.
      final oldest = _frame(8000);
      final middle = _frame(8000);
      final newest = _frame(8000);
      final result = slideDraftFrames(
        [oldest, middle, newest],
        sampleRate: sampleRate,
        maxSeconds: 1.0,
      );
      expect(result, [middle, newest]);
    });

    test('keeps at least the single newest frame even if it alone exceeds '
        'the budget', () {
      final huge = _frame(32000); // 2s of audio, twice the 1s window
      final result = slideDraftFrames(
        [_frame(4000), huge],
        sampleRate: sampleRate,
        maxSeconds: 1.0,
      );
      expect(result, [huge]);
    });

    test('an empty accumulator stays empty', () {
      final result = slideDraftFrames(
        <Float32List>[],
        sampleRate: sampleRate,
        maxSeconds: 1.0,
      );
      expect(result, isEmpty);
    });
  });

  group('isDraftDue', () {
    test('skips (does not queue) while a decode is already in flight', () {
      final due = isDraftDue(
        isDecoding: true,
        sinceLastDraft: const Duration(seconds: 999),
      );
      expect(due, isFalse);
    });

    test('is not due before the interval has elapsed', () {
      final due = isDraftDue(
        isDecoding: false,
        sinceLastDraft: const Duration(milliseconds: 500),
        intervalSeconds: 1.0,
      );
      expect(due, isFalse);
    });

    test('is due once the interval has elapsed and nothing is decoding', () {
      final due = isDraftDue(
        isDecoding: false,
        sinceLastDraft: const Duration(seconds: 1),
        intervalSeconds: 1.0,
      );
      expect(due, isTrue);
    });

    test('is due for any elapsed time past the interval, not just exactly at it', () {
      final due = isDraftDue(
        isDecoding: false,
        sinceLastDraft: const Duration(seconds: 5),
        intervalSeconds: 1.0,
      );
      expect(due, isTrue);
    });

    test('busy always wins over an elapsed interval', () {
      final due = isDraftDue(
        isDecoding: true,
        sinceLastDraft: const Duration(seconds: 5),
        intervalSeconds: 1.0,
      );
      expect(due, isFalse);
    });

    test('respects a custom interval', () {
      expect(
        isDraftDue(
          isDecoding: false,
          sinceLastDraft: const Duration(milliseconds: 1900),
          intervalSeconds: 2.0,
        ),
        isFalse,
      );
      expect(
        isDraftDue(
          isDecoding: false,
          sinceLastDraft: const Duration(milliseconds: 2000),
          intervalSeconds: 2.0,
        ),
        isTrue,
      );
    });
  });
}
