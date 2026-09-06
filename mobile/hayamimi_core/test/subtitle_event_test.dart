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
        'audio_s': null,
        'switched': false,
      });
    });

    test('toJson carries lang/speaker/latency/audio_s/switched through', () {
      const event = FinalSubtitleEvent(
        text: 'hello',
        lang: 'ja',
        speaker: 'S1',
        latencyMs: 123.5,
        audioSeconds: 2.5,
        switched: true,
      );
      expect(event.toJson(), {
        'type': 'final',
        'text': 'hello',
        'lang': 'ja',
        'speaker': 'S1',
        'latency_ms': 123.5,
        'audio_s': 2.5,
        'switched': true,
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

  group('RefineSubtitleEvent', () {
    test('toJson matches the desktop wire format, defaults included', () {
      const event = RefineSubtitleEvent(text: '清書結果');
      expect(event.toJson(), {
        'type': 'refine',
        'text': '清書結果',
        'lang': '',
        'speaker': '',
        'latency_ms': null,
        'audio_s': null,
      });
    });

    test('toJson carries lang/speaker/latency/audio_s through', () {
      const event = RefineSubtitleEvent(
        text: 'refined',
        lang: 'en',
        speaker: 'S2',
        latencyMs: 456.0,
        audioSeconds: 12.0,
      );
      expect(event.toJson(), {
        'type': 'refine',
        'text': 'refined',
        'lang': 'en',
        'speaker': 'S2',
        'latency_ms': 456.0,
        'audio_s': 12.0,
      });
    });
  });

  group('ModelLoadSubtitleEvent', () {
    test('toJson on "start" (no elapsed time yet)', () {
      const event = ModelLoadSubtitleEvent(model: 'ja', phase: 'start');
      expect(event.toJson(), {
        'type': 'model_load',
        'model': 'ja',
        'phase': 'start',
        'ms': null,
      });
    });

    test('toJson on "done" carries the elapsed milliseconds', () {
      const event = ModelLoadSubtitleEvent(
        model: 'vad',
        phase: 'done',
        ms: 87.4,
      );
      expect(event.toJson(), {
        'type': 'model_load',
        'model': 'vad',
        'phase': 'done',
        'ms': 87.4,
      });
    });
  });

  group('SessionResetSubtitleEvent', () {
    test('toJson has only the type discriminator', () {
      const event = SessionResetSubtitleEvent();
      expect(event.toJson(), {'type': 'session_reset'});
    });

    test('toSseFrame round-trips through JSON encoding', () {
      const event = SessionResetSubtitleEvent();
      expect(event.toSseFrame(), 'data: {"type":"session_reset"}\n\n');
    });
  });
}
