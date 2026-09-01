import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';

/// The wire encoding between `LiveTranscriber` and its decode worker
/// isolate. Nothing here touches FFI: the point of describing requests and
/// results as data was that the encoding could be checked without a device.
void main() {
  group('DecodeWorkerConfig', () {
    test('round-trips a routed session', () {
      const config = DecodeWorkerConfig(
        routed: true,
        modelDir: '/models/ja',
        senseVoiceModelDir: '/models/sv',
        lidModelDir: '/models/lid',
        decodingMethod: 'modified_beam_search',
        hotwordsFile: '/models/hotwords.txt',
        hotwordsScore: 2.5,
        numThreads: 4,
        sampleRate: 16000,
      );

      final decoded = DecodeWorkerConfig.fromMessage(config.toMessage());

      expect(decoded.routed, isTrue);
      expect(decoded.modelDir, '/models/ja');
      expect(decoded.senseVoiceModelDir, '/models/sv');
      expect(decoded.lidModelDir, '/models/lid');
      expect(decoded.decodingMethod, 'modified_beam_search');
      expect(decoded.hotwordsFile, '/models/hotwords.txt');
      expect(decoded.hotwordsScore, 2.5);
      expect(decoded.numThreads, 4);
      expect(decoded.sampleRate, 16000);
    });

    test('round-trips a plain session, nulls included', () {
      const config = DecodeWorkerConfig(routed: false, modelDir: '/models/ja');

      final decoded = DecodeWorkerConfig.fromMessage(config.toMessage());

      expect(decoded.routed, isFalse);
      expect(decoded.senseVoiceModelDir, isNull);
      expect(decoded.lidModelDir, isNull);
      expect(decoded.decodingMethod, isNull);
      expect(decoded.hotwordsFile, isNull);
      expect(decoded.hotwordsScore, 1.5);
    });
  });

  group('DecodeWorkerCommand', () {
    test('an audio request round-trips its samples and segment id', () {
      final samples = Float32List.fromList(<double>[0.1, -0.2, 0.3, -0.4]);
      final request = DecodeRequest(
        id: 7,
        kind: DecodeRequestKind.draft,
        samples: samples,
        segmentId: 42,
      );

      final decoded =
          DecodeWorkerCommand.fromMessage(request.toMessage()) as DecodeRequest;

      expect(decoded.id, 7);
      expect(decoded.kind, DecodeRequestKind.draft);
      expect(decoded.segmentId, 42);
      expect(decoded.samples, hasLength(4));
      expect(decoded.samples[0], closeTo(0.1, 1e-6));
      expect(decoded.samples[3], closeTo(-0.4, 1e-6));
    });

    test('a sample view is sent whole, not from the start of its buffer', () {
      // capDraftWindow hands the trailing slice of a bigger buffer straight
      // to the worker, so the encoding has to respect the view's offset.
      final backing = Float32List.fromList(<double>[9, 9, 1, 2, 3]);
      final view = Float32List.sublistView(backing, 2);
      final request = DecodeRequest(
        id: 1,
        kind: DecodeRequestKind.finalSegment,
        samples: view,
      );

      final decoded =
          DecodeWorkerCommand.fromMessage(request.toMessage()) as DecodeRequest;

      expect(decoded.samples, <double>[1, 2, 3]);
    });

    test('an empty audio request round-trips', () {
      final request = DecodeRequest(
        id: 3,
        kind: DecodeRequestKind.refine,
        samples: Float32List(0),
      );

      final decoded =
          DecodeWorkerCommand.fromMessage(request.toMessage()) as DecodeRequest;

      expect(decoded.samples, isEmpty);
    });

    test('control requests round-trip', () {
      for (final kind in <DecodeRequestKind>[
        DecodeRequestKind.resetSession,
        DecodeRequestKind.shutdown,
      ]) {
        final decoded = DecodeWorkerCommand.fromMessage(
          ControlRequest(id: 5, kind: kind).toMessage(),
        );

        expect(decoded, isA<ControlRequest>());
        expect(decoded.id, 5);
        expect(decoded.kind, kind);
      }
    });
  });

  group('DecodeWorkerMessage', () {
    test('a result round-trips every field', () {
      const result = DecodeWorkerResult(
        id: 11,
        kind: DecodeRequestKind.finalSegment,
        segmentId: 3,
        text: 'こんにちは',
        lang: 'ja',
        switched: true,
        latencyMs: 123.5,
      );

      final decoded =
          DecodeWorkerMessage.fromMessage(result.toMessage())
              as DecodeWorkerResult;

      expect(decoded.id, 11);
      expect(decoded.kind, DecodeRequestKind.finalSegment);
      expect(decoded.segmentId, 3);
      expect(decoded.text, 'こんにちは');
      expect(decoded.lang, 'ja');
      expect(decoded.switched, isTrue);
      expect(decoded.latencyMs, 123.5);
    });

    test('a plain-session result keeps its null language', () {
      const result = DecodeWorkerResult(
        id: 1,
        kind: DecodeRequestKind.draft,
        text: 'hello',
        latencyMs: 1,
      );

      final decoded =
          DecodeWorkerMessage.fromMessage(result.toMessage())
              as DecodeWorkerResult;

      expect(decoded.lang, isNull);
      expect(decoded.switched, isFalse);
    });

    test('ready, build failure, model load, failure, ack and death round-trip', () {
      expect(
        DecodeWorkerMessage.fromMessage(const DecodeWorkerReady().toMessage()),
        isA<DecodeWorkerReady>(),
      );

      final failed =
          DecodeWorkerMessage.fromMessage(
                const DecodeWorkerBuildFailed('Model directory not found: /x')
                    .toMessage(),
              )
              as DecodeWorkerBuildFailed;
      expect(failed.message, 'Model directory not found: /x');

      final load =
          DecodeWorkerMessage.fromMessage(
                const DecodeWorkerModelLoad(
                  model: 'sensevoice',
                  phase: 'done',
                  ms: 812.5,
                ).toMessage(),
              )
              as DecodeWorkerModelLoad;
      expect(load.model, 'sensevoice');
      expect(load.phase, 'done');
      expect(load.ms, 812.5);

      final startLoad =
          DecodeWorkerMessage.fromMessage(
                const DecodeWorkerModelLoad(
                  model: 'lid',
                  phase: 'start',
                ).toMessage(),
              )
              as DecodeWorkerModelLoad;
      expect(startLoad.ms, isNull);

      final failure =
          DecodeWorkerMessage.fromMessage(
                const DecodeWorkerFailure(
                  id: 9,
                  kind: DecodeRequestKind.refine,
                  message: 'boom',
                ).toMessage(),
              )
              as DecodeWorkerFailure;
      expect(failure.id, 9);
      expect(failure.kind, DecodeRequestKind.refine);
      expect(failure.message, 'boom');

      final ack =
          DecodeWorkerMessage.fromMessage(
                const DecodeWorkerAck(
                  id: 4,
                  kind: DecodeRequestKind.resetSession,
                ).toMessage(),
              )
              as DecodeWorkerAck;
      expect(ack.id, 4);
      expect(ack.kind, DecodeRequestKind.resetSession);

      final died =
          DecodeWorkerMessage.fromMessage(
                const DecodeWorkerDied('the isolate exited').toMessage(),
              )
              as DecodeWorkerDied;
      expect(died.message, 'the isolate exited');
    });

    test('an unrecognized message is rejected rather than silently ignored', () {
      expect(
        () => DecodeWorkerMessage.fromMessage(<Object?>['nonsense']),
        throwsArgumentError,
      );
    });
  });
}
