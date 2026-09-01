import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/refine_pass.dart';

RefineSegment _segment({
  required double seconds,
  required String text,
  int sampleRate = 16000,
  DateTime? capturedAt,
  double fill = 0,
  bool punctuated = false,
}) {
  final samples = Float32List((seconds * sampleRate).round());
  if (fill != 0) {
    for (var i = 0; i < samples.length; i++) {
      samples[i] = fill;
    }
  }
  return RefineSegment(
    samples: samples,
    text: text,
    capturedAt: capturedAt ?? DateTime(2026),
    punctuated: punctuated,
  );
}

void main() {
  group('RefineBuffer', () {
    test('starts empty', () {
      final buffer = RefineBuffer();
      expect(buffer.isEmpty, isTrue);
      expect(buffer.totalDurationSeconds, 0);
    });

    test('accumulates duration as segments are added', () {
      final buffer = RefineBuffer();
      buffer.add(_segment(seconds: 1.5, text: 'a'));
      buffer.add(_segment(seconds: 2.5, text: 'b'));
      expect(buffer.isNotEmpty, isTrue);
      expect(buffer.totalDurationSeconds, closeTo(4.0, 1e-6));
      expect(buffer.segments.map((s) => s.text), ['a', 'b']);
    });

    test('slides out the oldest segments once over the cap', () {
      final buffer = RefineBuffer(maxDurationSeconds: 5.0);
      buffer.add(_segment(seconds: 3.0, text: 'first'));
      buffer.add(_segment(seconds: 3.0, text: 'second'));
      // total would be 6s > 5s cap: the oldest ("first") is dropped.
      expect(buffer.segments.map((s) => s.text), ['second']);
      expect(buffer.totalDurationSeconds, closeTo(3.0, 1e-6));
    });

    test('keeps a single segment even if it alone exceeds the cap', () {
      final buffer = RefineBuffer(maxDurationSeconds: 1.0);
      buffer.add(_segment(seconds: 5.0, text: 'long'));
      expect(buffer.segments.map((s) => s.text), ['long']);
    });

    test('takeAll drains the buffer and returns what was buffered', () {
      final buffer = RefineBuffer();
      buffer.add(_segment(seconds: 1.0, text: 'a'));
      buffer.add(_segment(seconds: 1.0, text: 'b'));
      final taken = buffer.takeAll();
      expect(taken.map((s) => s.text), ['a', 'b']);
      expect(buffer.isEmpty, isTrue);
    });

    test('clear drops everything without returning it', () {
      final buffer = RefineBuffer();
      buffer.add(_segment(seconds: 1.0, text: 'a'));
      buffer.clear();
      expect(buffer.isEmpty, isTrue);
    });
  });

  group('combineSegmentSamples', () {
    test('concatenates sample buffers in order', () {
      final s1 = RefineSegment(
        samples: Float32List.fromList([1, 2, 3]),
        text: 'a',
        capturedAt: DateTime(2026),
      );
      final s2 = RefineSegment(
        samples: Float32List.fromList([4, 5]),
        text: 'b',
        capturedAt: DateTime(2026),
      );
      final combined = combineSegmentSamples([s1, s2]);
      expect(combined, [1, 2, 3, 4, 5]);
    });

    test('returns an empty buffer for no segments', () {
      expect(combineSegmentSamples(const []), isEmpty);
    });
  });

  group('combineSegmentFastText', () {
    test('joins non-blank segment text with spaces', () {
      final segments = [
        _segment(seconds: 0.1, text: 'こんにちは'),
        _segment(seconds: 0.1, text: '  '),
        _segment(seconds: 0.1, text: '世界'),
      ];
      expect(combineSegmentFastText(segments), 'こんにちは 世界');
    });

    test('returns empty string for no usable text', () {
      final segments = [_segment(seconds: 0.1, text: ''), _segment(seconds: 0.1, text: '   ')];
      expect(combineSegmentFastText(segments), '');
    });
  });

  group('isCombinedFastTextPunctuated', () {
    test('true when every segment contributing text was punctuated', () {
      final segments = [
        _segment(seconds: 0.1, text: '東京の天気は晴れです。', punctuated: true),
        _segment(seconds: 0.1, text: '資料は昨日送りました。', punctuated: true),
      ];
      expect(isCombinedFastTextPunctuated(segments), isTrue);
    });

    test('false when any contributing segment was not', () {
      // A group half of whose finals went through the punctuation model
      // joins into a line that is partly punctuated, which is not a line a
      // consumer can treat as punctuated.
      final segments = [
        _segment(seconds: 0.1, text: '東京の天気は晴れです。', punctuated: true),
        _segment(seconds: 0.1, text: '資料は昨日送りました'),
      ];
      expect(isCombinedFastTextPunctuated(segments), isFalse);
    });

    test('ignores blank segments, which contribute nothing to the join', () {
      final segments = [
        _segment(seconds: 0.1, text: '   '),
        _segment(seconds: 0.1, text: '東京の天気は晴れです。', punctuated: true),
      ];
      expect(isCombinedFastTextPunctuated(segments), isTrue);
    });

    test('false for a group with no text at all', () {
      expect(isCombinedFastTextPunctuated(const []), isFalse);
      expect(
        isCombinedFastTextPunctuated([_segment(seconds: 0.1, text: '')]),
        isFalse,
      );
    });
  });

  group('isRefineTextTooShort', () {
    test('false when refine text is at least the ratio of fast text', () {
      expect(isRefineTextTooShort('abcdefg', 'abcdefghij', minRatio: 0.7), isFalse);
    });

    test('true when refine text shrank below the ratio', () {
      expect(isRefineTextTooShort('ab', 'abcdefghij', minRatio: 0.7), isTrue);
    });

    test('false when there is no fast text to compare against', () {
      expect(isRefineTextTooShort('', ''), isFalse);
    });
  });

  group('isAutoRefineDue', () {
    test('false when nothing is buffered', () {
      expect(
        isAutoRefineDue(
          sinceLastSegment: const Duration(seconds: 100),
          bufferedDurationSeconds: 0,
        ),
        isFalse,
      );
    });

    test('false before the silence gap and below the max-buffered ceiling', () {
      expect(
        isAutoRefineDue(
          sinceLastSegment: const Duration(seconds: 1),
          bufferedDurationSeconds: 5,
          silenceSeconds: 4,
          maxBufferedSeconds: 20,
        ),
        isFalse,
      );
    });

    test('true once the silence gap is reached', () {
      expect(
        isAutoRefineDue(
          sinceLastSegment: const Duration(seconds: 4),
          bufferedDurationSeconds: 1,
          silenceSeconds: 4,
          maxBufferedSeconds: 20,
        ),
        isTrue,
      );
    });

    test('true once the buffered-duration ceiling is reached, even mid-speech', () {
      expect(
        isAutoRefineDue(
          sinceLastSegment: const Duration(milliseconds: 200),
          bufferedDurationSeconds: 20,
          silenceSeconds: 4,
          maxBufferedSeconds: 20,
        ),
        isTrue,
      );
    });
  });
}
