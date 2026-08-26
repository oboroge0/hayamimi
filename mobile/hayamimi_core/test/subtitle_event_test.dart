import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/server/subtitle_event.dart';

void main() {
  group('PartialSubtitleEvent', () {
    test('toJson matches the desktop wire format', () {
      const event = PartialSubtitleEvent('こんに');
      expect(event.toJson(), {'type': 'partial', 'text': 'こんに'});
    });

    test('toSseFrame wraps JSON in a data: frame terminated by a blank line', () {
      const event = PartialSubtitleEvent('hi');
      expect(event.toSseFrame(), 'data: {"type":"partial","text":"hi"}\n\n');
    });
  });

  group('FinalSubtitleEvent', () {
    test('toJson matches the desktop wire format, defaults included', () {
      const event = FinalSubtitleEvent(text: 'こんにちは');
      expect(event.toJson(), {
        'type': 'final',
        'text': 'こんにちは',
        'lang': '',
        'speaker': '',
        'latency_ms': null,
      });
    });

    test('toJson carries lang/speaker/latency through', () {
      const event = FinalSubtitleEvent(
        text: 'hello',
        lang: 'ja',
        speaker: 'S1',
        latencyMs: 123.5,
      );
      expect(event.toJson(), {
        'type': 'final',
        'text': 'hello',
        'lang': 'ja',
        'speaker': 'S1',
        'latency_ms': 123.5,
      });
    });

    test('toSseFrame round-trips through JSON encoding', () {
      const event = FinalSubtitleEvent(text: 'test', lang: 'ja');
      final frame = event.toSseFrame();
      expect(frame.startsWith('data: '), isTrue);
      expect(frame.endsWith('\n\n'), isTrue);
      expect(frame.contains('"type":"final"'), isTrue);
    });
  });
}
