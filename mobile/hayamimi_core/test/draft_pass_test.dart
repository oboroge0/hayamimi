import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/draft_pass.dart';

void main() {
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
