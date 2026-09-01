import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/punct/punct_ja_text.dart';

/// The mark-placing half of the punctuation restorer, with the model's
/// output supplied by hand. Everything here is a rule ported from
/// `scripts/punct_ja.py`; the parity test proves the rules add up to the
/// same output, these tests say what each rule is.

/// Builds a logits buffer of the shape [insertMarks] expects: two values
/// per sequence position, and the sequence is `[CLS]` + characters +
/// `[SEP]`. `commas` and `periods` are indexed by character, so index 0 is
/// the first character rather than `[CLS]`.
Float32List logitsFor(
  int characterCount, {
  Map<int, double> commas = const <int, double>{},
  Map<int, double> periods = const <int, double>{},
}) {
  // -10 sits far below the 0.5 threshold once sigmoid is applied, +10 far
  // above, so a test says "mark here" without hand-computing logits.
  final Float32List logits = Float32List((characterCount + 2) * 2);
  for (int i = 0; i < logits.length; i++) {
    logits[i] = -10.0;
  }
  commas.forEach((int index, double value) {
    logits[(index + 1) * 2] = value;
  });
  periods.forEach((int index, double value) {
    logits[(index + 1) * 2 + 1] = value;
  });
  return logits;
}

void main() {
  group('sigmoid', () {
    test('maps 0 to exactly the default threshold', () {
      expect(sigmoid(0), 0.5);
    });

    test('is monotonic across the threshold', () {
      expect(sigmoid(-1), lessThan(0.5));
      expect(sigmoid(1), greaterThan(0.5));
    });
  });

  group('insertMarks', () {
    test('writes 。 after a character the model marks with a period', () {
      final List<String> characters = <String>['あ', 'い', 'う'];
      expect(
        insertMarks(
          characters: characters,
          logits: logitsFor(3, periods: <int, double>{1: 10.0, 2: 10.0}),
        ),
        'あい。う。',
      );
    });

    test('writes 、 when only the comma clears its threshold', () {
      expect(
        insertMarks(
          characters: <String>['あ', 'い'],
          logits: logitsFor(
            2,
            commas: <int, double>{0: 10.0},
            periods: <int, double>{1: 10.0},
          ),
        ),
        'あ、い。',
      );
    });

    test('prefers the period when both clear their thresholds', () {
      expect(
        insertMarks(
          characters: <String>['あ', 'い'],
          logits: logitsFor(
            2,
            commas: <int, double>{0: 10.0},
            periods: <int, double>{0: 10.0, 1: 10.0},
          ),
        ),
        'あ。い。',
      );
    });

    test('never writes a mark after a character that is already one', () {
      expect(
        insertMarks(
          characters: <String>['あ', '」', 'い'],
          logits: logitsFor(3, periods: <int, double>{1: 10.0, 2: 10.0}),
        ),
        'あ」い。',
      );
    });

    test('never writes a mark immediately before an existing one', () {
      expect(
        insertMarks(
          characters: <String>['あ', '、', 'い'],
          logits: logitsFor(3, periods: <int, double>{0: 10.0, 2: 10.0}),
        ),
        'あ、い。',
      );
    });

    test('appends 。 when the text would otherwise end unpunctuated', () {
      expect(
        insertMarks(characters: <String>['あ', 'い'], logits: logitsFor(2)),
        'あい。',
      );
    });

    test('leaves an already-punctuated ending alone', () {
      expect(
        insertMarks(characters: <String>['あ', '！'], logits: logitsFor(2)),
        'あ！',
      );
    });

    test(
      'with forceFinalPeriod off, a final period falls through to the comma',
      () {
        // The reference implementation uses if/elif, so suppressing the
        // period on the last character lets the comma be considered
        // instead rather than skipping the character entirely.
        expect(
          insertMarks(
            characters: <String>['あ', 'い'],
            logits: logitsFor(
              2,
              commas: <int, double>{1: 10.0},
              periods: <int, double>{1: 10.0},
            ),
            forceFinalPeriod: false,
          ),
          'あい、',
        );
      },
    );

    test('with forceFinalPeriod off, nothing is appended at the end', () {
      expect(
        insertMarks(
          characters: <String>['あ', 'い'],
          logits: logitsFor(2),
          forceFinalPeriod: false,
        ),
        'あい',
      );
    });

    test('honours a raised threshold', () {
      final Float32List logits = logitsFor(
        2,
        periods: <int, double>{0: 1.0, 1: 10.0},
      );
      expect(
        insertMarks(characters: <String>['あ', 'い'], logits: logits),
        'あ。い。',
      );
      expect(
        insertMarks(
          characters: <String>['あ', 'い'],
          logits: logits,
          periodThreshold: 0.9,
        ),
        'あい。',
      );
    });

    test('rejects a logits buffer that does not match the characters', () {
      expect(
        () => insertMarks(
          characters: <String>['あ', 'い'],
          logits: logitsFor(3),
        ),
        throwsArgumentError,
      );
    });
  });

  group('applyQuestionMarks', () {
    test('turns a sentence-final 。 into ？ after a question ending', () {
      expect(applyQuestionMarks('元気ですか。'), '元気ですか？');
    });

    test('leaves a statement alone', () {
      expect(applyQuestionMarks('元気です。'), '元気です。');
    });

    test('decides per sentence', () {
      expect(
        applyQuestionMarks('もう準備はできましたか。今日中に送ります。'),
        'もう準備はできましたか？今日中に送ります。',
      );
    });

    test('appends a mark to a trailing segment that has none', () {
      // A quirk of the reference implementation, kept so both pipelines
      // produce the same text: every non-empty segment gets a mark, even
      // the last one when the input did not end with 。.
      expect(applyQuestionMarks('あい'), 'あい。');
    });

    test('drops the empty segments a leading or doubled 。 creates', () {
      expect(applyQuestionMarks('。あ'), 'あ。');
      expect(applyQuestionMarks('あ。。い'), 'あ。い。');
    });

    test('fires on the nominalizer の too, as the heuristic cannot tell', () {
      expect(applyQuestionMarks('それが問題なの。'), 'それが問題なの？');
    });

    test('returns an empty string unchanged', () {
      expect(applyQuestionMarks(''), '');
    });
  });
}
