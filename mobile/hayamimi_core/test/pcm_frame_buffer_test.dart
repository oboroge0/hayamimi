import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/live/pcm_frame_buffer.dart';

void main() {
  group('pcm16BytesToFloat32', () {
    test('converts little-endian int16 bytes to normalized floats', () {
      // int16 values: 0, 32767 (max), -32768 (min), -1
      final bytes = Uint8List.fromList([
        0x00, 0x00, // 0
        0xFF, 0x7F, // 32767
        0x00, 0x80, // -32768
        0xFF, 0xFF, // -1
      ]);

      final samples = pcm16BytesToFloat32(bytes);

      expect(samples.length, 4);
      expect(samples[0], closeTo(0.0, 1e-9));
      expect(samples[1], closeTo(32767 / 32768.0, 1e-9));
      expect(samples[2], closeTo(-1.0, 1e-9));
      expect(samples[3], closeTo(-1 / 32768.0, 1e-9));
    });

    test('handles an empty chunk', () {
      final samples = pcm16BytesToFloat32(Uint8List(0));
      expect(samples, isEmpty);
    });
  });

  group('PcmFrameBuffer', () {
    test('does not emit a frame until enough samples have arrived', () {
      final buffer = PcmFrameBuffer(frameSize: 4);
      buffer.add(Float32List.fromList([1, 2]));

      expect(buffer.drainFrames(), isEmpty);
      expect(buffer.pendingSampleCount, 2);
    });

    test('emits exactly one frame once enough samples accumulate', () {
      final buffer = PcmFrameBuffer(frameSize: 4);
      buffer.add(Float32List.fromList([1, 2]));
      buffer.add(Float32List.fromList([3, 4]));

      final frames = buffer.drainFrames();

      expect(frames, hasLength(1));
      expect(frames.single, Float32List.fromList([1, 2, 3, 4]));
      expect(buffer.pendingSampleCount, 0);
    });

    test(
      'emits multiple frames from one large chunk and keeps the remainder',
      () {
        final buffer = PcmFrameBuffer(frameSize: 3);
        buffer.add(Float32List.fromList([1, 2, 3, 4, 5, 6, 7]));

        final frames = buffer.drainFrames();

        expect(frames, hasLength(2));
        expect(frames[0], Float32List.fromList([1, 2, 3]));
        expect(frames[1], Float32List.fromList([4, 5, 6]));
        expect(buffer.pendingSampleCount, 1);
      },
    );

    test('reset drops buffered samples', () {
      final buffer = PcmFrameBuffer(frameSize: 4);
      buffer.add(Float32List.fromList([1, 2, 3]));

      buffer.reset();

      expect(buffer.pendingSampleCount, 0);
      buffer.add(Float32List.fromList([4]));
      expect(buffer.pendingSampleCount, 1);
    });

    test('drainFrames called again after a partial drain stays empty until refilled', () {
      final buffer = PcmFrameBuffer(frameSize: 2);
      buffer.add(Float32List.fromList([1, 2, 3]));

      expect(buffer.drainFrames(), hasLength(1));
      expect(buffer.drainFrames(), isEmpty);
      expect(buffer.pendingSampleCount, 1);
    });

    test(
      'ring boundary: repeated small add/drain cycles past the initial '
      'capacity keep frame contents and order correct',
      () {
        // frameSize 4 * initial capacity factor 8 = 32-sample backing
        // array; looping well past that forces both the "compact toward
        // offset 0" and "grow" paths of the ring buffer.
        final buffer = PcmFrameBuffer(frameSize: 4);
        var nextValue = 1.0;
        final allFrames = <Float32List>[];

        for (var cycle = 0; cycle < 20; cycle++) {
          // Push 3 samples (not a full frame), then 1 more to complete it --
          // exercises the carry-over-then-complete boundary every cycle.
          final threeSamples = Float32List.fromList([
            nextValue,
            nextValue + 1,
            nextValue + 2,
          ]);
          buffer.add(threeSamples);
          expect(buffer.drainFrames(), isEmpty);
          expect(buffer.pendingSampleCount, 3);

          buffer.add(Float32List.fromList([nextValue + 3]));
          final frames = buffer.drainFrames();
          expect(frames, hasLength(1));
          expect(buffer.pendingSampleCount, 0);
          allFrames.add(frames.single);
          nextValue += 4;
        }

        for (var i = 0; i < allFrames.length; i++) {
          final base = 1.0 + i * 4;
          expect(
            allFrames[i],
            Float32List.fromList([base, base + 1, base + 2, base + 3]),
          );
        }
      },
    );

    test('a chunk far larger than the backing array grows it correctly', () {
      final buffer = PcmFrameBuffer(frameSize: 4);
      // Initial capacity is 4 * 8 = 32 samples; push well past that in one
      // call to force the doubling-grow path (not just compaction).
      final bigChunk = Float32List.fromList(
        List.generate(100, (i) => i.toDouble()),
      );
      buffer.add(bigChunk);

      final frames = buffer.drainFrames();
      expect(frames, hasLength(25));
      for (var i = 0; i < frames.length; i++) {
        final base = (i * 4).toDouble();
        expect(
          frames[i],
          Float32List.fromList([base, base + 1, base + 2, base + 3]),
        );
      }
      expect(buffer.pendingSampleCount, 0);
    });

    test('interleaved adds and partial drains preserve order across growth', () {
      final buffer = PcmFrameBuffer(frameSize: 5);
      final produced = <double>[];
      final consumed = <double>[];
      var counter = 0.0;

      for (var i = 0; i < 15; i++) {
        // Alternate small and large chunks to hit both the compact and
        // grow branches of _makeRoomFor across the run.
        final chunkLen = i.isEven ? 2 : 13;
        final chunk = Float32List.fromList(
          List.generate(chunkLen, (j) => counter + j),
        );
        counter += chunkLen;
        produced.addAll(chunk);
        buffer.add(chunk);

        for (final frame in buffer.drainFrames()) {
          consumed.addAll(frame);
        }
      }

      // Whatever's left in the buffer at the end should be exactly the
      // tail of everything produced, since consumed + pending must equal
      // produced in order.
      expect(consumed.length + buffer.pendingSampleCount, produced.length);
      expect(consumed, produced.sublist(0, consumed.length));
    });
  });

  group('concatFloat32Lists', () {
    test('joins chunks in order', () {
      final combined = concatFloat32Lists([
        Float32List.fromList([1, 2]),
        Float32List.fromList([3]),
        Float32List.fromList([4, 5]),
      ]);

      expect(combined, Float32List.fromList([1, 2, 3, 4, 5]));
    });

    test('returns an empty list for no chunks', () {
      expect(concatFloat32Lists([]), isEmpty);
    });

    test('handles a single chunk', () {
      final chunk = Float32List.fromList([1, 2, 3]);
      expect(concatFloat32Lists([chunk]), chunk);
    });
  });

  group('capDraftWindow', () {
    test('returns samples unchanged when under the cap', () {
      final samples = Float32List.fromList(List.filled(100, 1.0));
      final capped = capDraftWindow(samples, sampleRate: 100, maxSeconds: 2.0);
      expect(capped, samples);
    });

    test('returns exactly the cap length when equal', () {
      final samples = Float32List.fromList(List.filled(200, 1.0));
      final capped = capDraftWindow(samples, sampleRate: 100, maxSeconds: 2.0);
      expect(capped.length, 200);
    });

    test('keeps only the trailing window when over the cap', () {
      final samples = Float32List.fromList(
        List.generate(10, (i) => i.toDouble()),
      );
      final capped = capDraftWindow(samples, sampleRate: 5, maxSeconds: 1.0);

      // maxSamples = 5 * 1.0 = 5, so the last 5 samples: [5, 6, 7, 8, 9]
      expect(capped, Float32List.fromList([5, 6, 7, 8, 9]));
    });
  });
}
