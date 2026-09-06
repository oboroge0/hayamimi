import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/preroll.dart';

/// [PrerollHistory] on its own: the rolling buffer, and the two rules that
/// decide how far back a segment is allowed to reach.
///
/// Everything here runs at a made-up 100 Hz sample rate. The class only
/// uses the rate to turn seconds into sample counts, and 100 makes the
/// arithmetic readable — one second of "audio" is 100 samples, so a
/// position and an index are the same number. The desktop tests this
/// mirrors (`tests/test_units.py`) do the same.
void main() {
  const sampleRate = 100;

  /// [count] samples whose value is their own absolute position, so an
  /// assertion on a sample's value is an assertion on where it came from.
  Float32List ramp(int from, int count) =>
      Float32List.fromList(List<double>.generate(count, (i) => (from + i) * 1.0));

  /// A stand-in for one segment's own audio, distinguishable from [ramp]'s.
  Float32List speech(int count) => Float32List(count)..fillRange(0, count, -1.0);

  PrerollHistory history({double keepSeconds = defaultPrerollKeepSeconds}) =>
      PrerollHistory(sampleRate: sampleRate, keepSeconds: keepSeconds);

  group('push', () {
    test('keeps everything while under the keep window', () {
      final h = history(keepSeconds: 5.0); // 500 samples
      for (var i = 0; i < 4; i++) {
        h.push(ramp(i * 100, 100));
      }

      expect(h.bufferedSamples, 400);
      expect(h.offset, 0);
    });

    test('drops the oldest frames once past it, and says so through offset', () {
      final h = history(keepSeconds: 5.0); // 500 samples
      for (var i = 0; i < 20; i++) {
        h.push(ramp(i * 100, 100));
      }

      // 2000 samples pushed, 500 kept: the first 1500 are gone, and offset
      // is what says where the remaining audio starts.
      expect(h.bufferedSamples, 500);
      expect(h.offset, 1500);
    });

    test('never drops below the keep window it promised', () {
      // Frames that do not divide the window evenly: dropping one more
      // would leave less than 500 samples, so it is kept and the buffer
      // sits slightly above the window instead of below it.
      final h = history(keepSeconds: 5.0);
      for (var i = 0; i < 20; i++) {
        h.push(ramp(i * 30, 30));
      }

      expect(h.bufferedSamples, greaterThanOrEqualTo(500));
      expect(h.bufferedSamples, lessThan(500 + 30));
    });

    test('ignores an empty frame', () {
      final h = history();
      h.push(Float32List(0));

      expect(h.bufferedSamples, 0);
    });
  });

  group('withPreroll', () {
    test('prepends a full second of the audio before the onset', () {
      final h = history();
      h.push(ramp(0, 1000));

      final out = h.withPreroll(
        segmentStartSample: 500,
        samples: speech(10),
      );

      expect(out.length, 100 + 10);
      // The prepended samples are exactly positions 400..499, i.e. the
      // second of audio immediately before the VAD noticed anything.
      expect(out[0], 400.0);
      expect(out[99], 499.0);
      // ...followed by the segment's own audio, unchanged.
      expect(out[100], -1.0);
    });

    test('prepends only what exists when the session just started', () {
      final h = history();
      h.push(ramp(0, 30)); // less than the 100 samples a second wants

      final out = h.withPreroll(segmentStartSample: 30, samples: speech(10));

      expect(out.length, 30 + 10);
      expect(out[0], 0.0);
    });

    test('returns the segment untouched when nothing has been pushed', () {
      final h = history();
      final samples = speech(10);

      final out = h.withPreroll(segmentStartSample: 0, samples: samples);

      expect(identical(out, samples), isTrue);
    });

    test('never reaches back into the previous segment', () {
      final h = history();
      h.push(ramp(0, 1000));

      // First segment runs 300..399.
      h.withPreroll(segmentStartSample: 300, samples: speech(100));
      // The second starts 50 samples later. A full second of pre-roll would
      // start at 350, inside the first segment; it is clamped to 400.
      final out = h.withPreroll(segmentStartSample: 450, samples: speech(10));

      expect(out.length, 50 + 10);
      expect(out[0], 400.0);
    });

    test('two adjacent segments together cover their audio exactly once', () {
      final h = history();
      h.push(ramp(0, 1000));

      final first = h.withPreroll(segmentStartSample: 300, samples: speech(100));
      final second = h.withPreroll(segmentStartSample: 450, samples: speech(10));

      // 200..399 and 400..459: 260 samples for the 260 the two segments and
      // their run-ups span, with none of it counted twice.
      expect(first.length + second.length, 260);
    });

    test('does not reach past audio the buffer has already dropped', () {
      final h = history(keepSeconds: 5.0);
      for (var i = 0; i < 20; i++) {
        h.push(ramp(i * 100, 100));
      }
      expect(h.offset, 1500);

      final out = h.withPreroll(segmentStartSample: 1550, samples: speech(10));

      // A full second back would be 1450, which is gone; it starts at the
      // oldest sample still held instead.
      expect(out.length, 50 + 10);
      expect(out[0], 1500.0);
    });

    test('prerollSeconds of 0 turns it off', () {
      final h = history();
      h.push(ramp(0, 1000));
      final samples = speech(10);

      final out = h.withPreroll(
        segmentStartSample: 500,
        samples: samples,
        prerollSeconds: 0,
      );

      expect(identical(out, samples), isTrue);
    });

    test('a segment starting beyond the pushed audio is left alone', () {
      // A stand-in VAD in a test can report a position no audio was ever
      // pushed for; that must read as "no context available", not throw.
      final h = history();
      h.push(ramp(0, 100));
      final samples = speech(10);

      final out = h.withPreroll(segmentStartSample: 5000, samples: samples);

      expect(identical(out, samples), isTrue);
    });

    test('moves lastSegmentEnd past the segment, even when nothing was prepended', () {
      final h = history();
      h.push(ramp(0, 1000));

      h.withPreroll(
        segmentStartSample: 300,
        samples: speech(100),
        prerollSeconds: 0,
      );

      expect(h.lastSegmentEnd, 400);
    });

    test('reads across frame boundaries', () {
      // The pre-roll window almost never lines up with the frames it was
      // pushed in, so the copy has to stitch parts of several together.
      final h = history();
      for (var i = 0; i < 10; i++) {
        h.push(ramp(i * 7, 7)); // 70 samples in frames of 7
      }

      final out = h.withPreroll(segmentStartSample: 70, samples: speech(1));

      expect(out.length, 70 + 1);
      for (var i = 0; i < 70; i++) {
        expect(out[i], i * 1.0);
      }
    });
  });
}
