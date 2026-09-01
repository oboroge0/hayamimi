import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/remote/remote_event.dart';

void main() {
  group('parseRemoteEvent', () {
    test('parses a ready event', () {
      final event = parseRemoteEvent('{"type": "ready", "sr": 16000}');
      expect(event, isA<RemoteReadyEvent>());
      expect((event as RemoteReadyEvent).sampleRate, 16000);
    });

    test('parses a session_start event', () {
      final event = parseRemoteEvent('{"type": "session_start"}');
      expect(event, isA<RemoteSessionStartEvent>());
    });

    test('parses a partial event', () {
      final event = parseRemoteEvent('{"type": "partial", "text": "こんに"}');
      expect(event, isA<RemotePartialEvent>());
      expect((event as RemotePartialEvent).text, 'こんに');
    });

    test('parses a final event with all fields', () {
      final event = parseRemoteEvent(
        '{"type": "final", "text": "こんにちは", "lang": "ja", '
        '"speaker": "spk1", "latency_ms": 123.5, "tier": "fast", '
        '"audio_s": 2.5, "switched": true}',
      );
      expect(event, isA<RemoteFinalEvent>());
      final finalEvent = event as RemoteFinalEvent;
      expect(finalEvent.text, 'こんにちは');
      expect(finalEvent.lang, 'ja');
      expect(finalEvent.speaker, 'spk1');
      expect(finalEvent.latencyMs, 123.5);
      expect(finalEvent.tier, 'fast');
      expect(finalEvent.audioSeconds, 2.5);
      expect(finalEvent.switched, isTrue);
    });

    test('parses a final event with missing optional fields as defaults', () {
      final event = parseRemoteEvent('{"type": "final", "text": "hi"}');
      final finalEvent = event as RemoteFinalEvent;
      expect(finalEvent.lang, '');
      expect(finalEvent.speaker, '');
      expect(finalEvent.latencyMs, isNull);
      expect(finalEvent.tier, '');
      expect(finalEvent.audioSeconds, isNull);
      expect(finalEvent.switched, isFalse);
    });

    test('parses a translation event', () {
      final event = parseRemoteEvent('{"type": "translation", "lang": "en", "text": "hello"}');
      expect(event, isA<RemoteTranslationEvent>());
      final tr = event as RemoteTranslationEvent;
      expect(tr.lang, 'en');
      expect(tr.text, 'hello');
    });

    test('parses a refine event with all fields', () {
      final event = parseRemoteEvent(
        '{"type": "refine", "text": "清書済み", "lang": "ja", '
        '"speaker": "spk1", "latency_ms": 250.0, "audio_s": 6.0}',
      );
      expect(event, isA<RemoteRefineEvent>());
      final refine = event as RemoteRefineEvent;
      expect(refine.text, '清書済み');
      expect(refine.lang, 'ja');
      expect(refine.speaker, 'spk1');
      expect(refine.latencyMs, 250.0);
      expect(refine.audioSeconds, 6.0);
    });

    test('parses a refine event with missing optional fields as defaults', () {
      final event = parseRemoteEvent('{"type": "refine", "text": "hi"}');
      final refine = event as RemoteRefineEvent;
      expect(refine.latencyMs, isNull);
      expect(refine.audioSeconds, isNull);
    });

    test('parses an error event', () {
      final event = parseRemoteEvent(
        '{"type": "error", "message": "already streaming"}',
      );
      expect(event, isA<RemoteErrorEvent>());
      expect((event as RemoteErrorEvent).message, 'already streaming');
    });

    test('parses a model_load event', () {
      final event = parseRemoteEvent(
        '{"type": "model_load", "model": "ja", "phase": "start"}',
      );
      expect(event, isA<RemoteModelLoadEvent>());
      final load = event as RemoteModelLoadEvent;
      expect(load.model, 'ja');
      expect(load.phase, 'start');
      expect(load.ms, isNull);
    });

    test('parses a model_load "done" event with elapsed ms', () {
      final event = parseRemoteEvent(
        '{"type": "model_load", "model": "vad", "phase": "done", "ms": 87.4}',
      );
      final load = event as RemoteModelLoadEvent;
      expect(load.phase, 'done');
      expect(load.ms, 87.4);
    });

    test('parses a session_reset event', () {
      final event = parseRemoteEvent('{"type": "session_reset"}');
      expect(event, isA<RemoteSessionResetEvent>());
    });

    test(
      'server event types this package has no typed RemoteEvent for yet '
      '(model_fallback/warning/session_summary/recluster) fall back to '
      'RemoteUnknownEvent',
      () {
        for (final type in [
          'model_fallback',
          'warning',
          'session_summary',
          'recluster',
        ]) {
          final event = parseRemoteEvent('{"type": "$type"}');
          expect(event, isA<RemoteUnknownEvent>(), reason: type);
          expect((event as RemoteUnknownEvent).type, type);
        }
      },
    );

    test('unrecognized type falls back to RemoteUnknownEvent', () {
      final event = parseRemoteEvent('{"type": "future_event", "foo": 1}');
      expect(event, isA<RemoteUnknownEvent>());
      final unknown = event as RemoteUnknownEvent;
      expect(unknown.type, 'future_event');
      expect(unknown.raw['foo'], 1);
    });

    test('missing type also falls back to RemoteUnknownEvent', () {
      final event = parseRemoteEvent('{"foo": 1}');
      expect(event, isA<RemoteUnknownEvent>());
      expect((event as RemoteUnknownEvent).type, isNull);
    });

    test('throws RemoteEventParseException on invalid JSON', () {
      expect(() => parseRemoteEvent('not json'), throwsA(isA<RemoteEventParseException>()));
    });

    test('throws RemoteEventParseException when JSON is not an object', () {
      expect(() => parseRemoteEvent('[1, 2, 3]'), throwsA(isA<RemoteEventParseException>()));
    });
  });
}
