import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';

import 'fake_decode_worker.dart';

/// Everything `LiveTranscriber` does with the decode worker, driven end to
/// end against a scripted stand-in for that worker.
///
/// This is where the change from "a synchronous FFI call" to "a request
/// answered later" is actually observable: finals arriving in order,
/// drafts dropped and discarded, refine passes folding together, a reset
/// landing behind the work it must not race, and the worker dying
/// mid-session.
void main() {
  late FakeDecodeWorker worker;
  late _Log log;

  setUp(() {
    worker = FakeDecodeWorker();
    log = _Log();
  });

  Future<DecodeSession> startSession({
    RefineRequestPayload? Function()? refinePayload,
  }) {
    return DecodeSession.start(
      worker: worker,
      config: const DecodeWorkerConfig(routed: false, modelDir: '/models'),
      buildRefinePayload: () {
        log.refinePayloadCalls++;
        return refinePayload?.call();
      },
      onFinal: (samples, result) => log.finals.add((samples, result)),
      onDraft: log.drafts.add,
      onRefine: (payload, result) => log.refines.add((payload, result)),
      onReset: () => log.resets++,
      onModelLoad: (model, phase, ms) => log.modelLoads.add('$model/$phase'),
      onBusyChanged: log.busy.add,
      onDecodeError: log.decodeErrors.add,
      onWorkerDied: log.deaths.add,
    );
  }

  Float32List samples(int length) =>
      Float32List.fromList(List<double>.filled(length, 0.5));

  group('start', () {
    test('reports each model load while the models are still loading', () async {
      worker.modelLoads.addAll(const <DecodeWorkerModelLoad>[
        DecodeWorkerModelLoad(model: 'ja', phase: 'start'),
        DecodeWorkerModelLoad(model: 'ja', phase: 'done', ms: 900),
      ]);

      final session = await startSession();
      addTearDown(session.shutdown);

      expect(log.modelLoads, <String>['ja/start', 'ja/done']);
    });

    test('a build failure comes out as DecodeWorkerException and leaves no worker running', () async {
      worker.buildFailure = 'Model directory not found: /models';

      await expectLater(
        startSession(),
        throwsA(
          isA<DecodeWorkerException>().having(
            (e) => e.message,
            'message',
            'Model directory not found: /models',
          ),
        ),
      );
      expect(worker.shutdownCount, 1);
    });
  });

  group('finals', () {
    test('are sent one at a time and answered in the order they were submitted', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      session.submitFinal(samples(20));
      session.submitFinal(samples(30));

      // Only the first is at the worker; the rest are held on this side.
      expect(worker.decodeRequests, hasLength(1));
      expect(worker.lastRequest.samples, hasLength(10));

      worker.reply('one');
      await settle();
      expect(worker.decodeRequests, hasLength(2));
      expect(worker.lastRequest.samples, hasLength(20));

      worker.reply('two');
      await settle();
      worker.reply('three');
      await settle();

      expect(
        log.finals.map((f) => f.$2.text),
        <String>['one', 'two', 'three'],
      );
      // Each result is handed back with the audio that produced it, which
      // is what goes into the refine buffer.
      expect(log.finals.map((f) => f.$1.length), <int>[10, 20, 30]);
      expect(session.isBusy, isFalse);
    });

    test('are never dropped, however many pile up', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      for (var i = 0; i < 5; i++) {
        session.submitFinal(samples(8));
      }
      for (var i = 0; i < 5; i++) {
        worker.reply('segment $i');
        await settle();
      }

      expect(log.finals, hasLength(5));
    });

    test('mirror a routed session language, and leave it null for a plain one', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      expect(session.currentLang, isNull);

      session.submitFinal(samples(10));
      worker.reply('hello', lang: 'en', switched: true);
      await settle();

      expect(session.currentLang, 'en');
      expect(log.finals.single.$2.switched, isTrue);

      session.submitFinal(samples(10));
      worker.reply('こんにちは', lang: 'ja');
      await settle();

      expect(session.currentLang, 'ja');
    });
  });

  group('drafts', () {
    test('are dropped, not queued, while something else is outstanding', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      expect(session.submitDraft(samples(4)), isFalse);
      expect(worker.decodeRequests, hasLength(1));
      expect(worker.lastRequest.kind, DecodeRequestKind.finalSegment);
    });

    test('are delivered while their segment is still open', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      expect(session.submitDraft(samples(4)), isTrue);
      worker.reply('in progr');
      await settle();

      expect(log.drafts.single.text, 'in progr');
    });

    test('are discarded when their segment closed while they were decoding', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitDraft(samples(4));
      final request = worker.lastRequest;
      expect(request.segmentId, session.draftSegmentId);

      // The speaker paused: the segment closed and its final took over.
      session.beginDraftSegment();

      worker.reply('in progr');
      await settle();

      expect(log.drafts, isEmpty);
    });
  });

  group('refine', () {
    RefineRequestPayload payload() => RefineRequestPayload(
      samples: samples(16000),
      fastText: 'fast text',
    );

    test('claims its audio when it is sent, not when it is asked for', () async {
      final session = await startSession(refinePayload: payload);
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      final refined = session.refine();

      // Still behind the final, so the buffer has not been claimed yet —
      // which is what lets that final join this refine's group.
      expect(log.refinePayloadCalls, 0);

      worker.reply('final');
      await settle();
      expect(log.refinePayloadCalls, 1);
      expect(worker.lastRequest.kind, DecodeRequestKind.refine);

      worker.reply('refined text');
      await settle();
      await refined;

      expect(log.refines.single.$2.text, 'refined text');
      expect(log.refines.single.$1.fastText, 'fast text');
    });

    test('a second call while one is still queued awaits that same pass', () async {
      final session = await startSession(refinePayload: payload);
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      final first = session.refine();
      final second = session.refine();

      expect(identical(first, second), isTrue);

      worker.reply('final');
      await settle();
      worker.reply('refined');
      await settle();
      await first;
      await second;

      // One request, one result — not two passes over overlapping audio.
      expect(
        worker.decodeRequests
            .where((r) => r.kind == DecodeRequestKind.refine)
            .length,
        1,
      );
      expect(log.refines, hasLength(1));
    });

    test('with nothing to refine completes without sending anything', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      await session.refine();

      expect(log.refinePayloadCalls, 1);
      expect(worker.decodeRequests, isEmpty);
      expect(log.refines, isEmpty);
      expect(session.isBusy, isFalse);
    });

    test('a dropped refine does not block the work queued behind it', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      final refined = session.refine();
      session.submitFinal(samples(10));
      await settle();

      // The refine had nothing to do, so the final went straight out.
      expect(worker.decodeRequests, hasLength(1));
      expect(worker.lastRequest.kind, DecodeRequestKind.finalSegment);
      await refined;
    });
  });

  group('resetSession', () {
    test('waits for outstanding work, then completes on the worker ack', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      var resetDone = false;
      final reset = session.reset().then((_) => resetDone = true);

      // Queued behind the final, so the transcript line lands first.
      expect(worker.commands, hasLength(1));

      worker.reply('last line', lang: 'ja');
      await settle();
      expect(log.finals, hasLength(1));
      expect(session.currentLang, 'ja');
      expect(worker.commands.last, isA<ControlRequest>());
      expect(resetDone, isFalse);

      worker.ack();
      await settle();
      await reset;

      expect(log.resets, 1);
      expect(session.currentLang, isNull);
      expect(session.isBusy, isFalse);
    });
  });

  group('busy', () {
    test('flips once per burst, not once per decode', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      session.submitFinal(samples(10));
      worker.reply('one');
      await settle();
      worker.reply('two');
      await settle();

      expect(log.busy, <bool>[true, false]);
    });

    test('waitForIdle completes when the queue drains', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      var idle = false;
      final waiting = session.waitForIdle().then((_) => idle = true);
      await settle();
      expect(idle, isFalse);

      worker.reply('done');
      await settle();
      await waiting;
      expect(idle, isTrue);

      // Already idle: completes immediately.
      await session.waitForIdle();
    });
  });

  group('failures', () {
    test('one failed decode is reported and the queue keeps moving', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      session.submitFinal(samples(10));

      worker.failRequest('sherpa-onnx blew up');
      await settle();

      expect(log.decodeErrors, <String>['sherpa-onnx blew up']);
      expect(log.finals, isEmpty);
      // The second segment still gets its turn.
      expect(worker.decodeRequests, hasLength(2));

      worker.reply('second');
      await settle();
      expect(log.finals.single.$2.text, 'second');
    });

    test('a worker that dies is reported once, drops the queue, and settles its futures', () async {
      final session = await startSession(
        refinePayload: () =>
            RefineRequestPayload(samples: samples(16000), fastText: 'fast'),
      );
      addTearDown(session.shutdown);

      session.submitFinal(samples(10));
      var refineDone = false;
      final refined = session.refine().then((_) => refineDone = true);
      await settle();

      worker.die('the isolate exited');
      await settle();
      await refined;

      expect(log.deaths, <String>['the isolate exited']);
      expect(session.isBusy, isFalse);
      expect(log.busy.last, isFalse);
      expect(refineDone, isTrue);
      expect(log.refines, isEmpty);
    });

    test('nothing new is sent to a dead worker', () async {
      final session = await startSession();
      addTearDown(session.shutdown);

      worker.die('the isolate exited');
      await settle();
      final before = worker.commands.length;

      session.submitFinal(samples(10));
      await session.refine();
      await session.reset();

      expect(worker.commands, hasLength(before));
      expect(session.isBusy, isFalse);
    });
  });

  group('shutdown', () {
    test('shuts the worker down and settles anything still waiting', () async {
      final session = await startSession(
        refinePayload: () =>
            RefineRequestPayload(samples: samples(16000), fastText: 'fast'),
      );

      session.submitFinal(samples(10));
      final refined = session.refine();

      await session.shutdown();
      await refined;

      expect(worker.shutdownCount, 1);
      expect(session.isBusy, isFalse);
    });

    test('is idempotent', () async {
      final session = await startSession();

      await session.shutdown();
      await session.shutdown();

      expect(worker.shutdownCount, 1);
    });

    test('a reply that arrives after shutdown is ignored', () async {
      final session = await startSession();

      session.submitFinal(samples(10));
      await session.shutdown();
      worker.reply('too late');
      await settle();

      expect(log.finals, isEmpty);
    });
  });
}

class _Log {
  final List<(Float32List, DecodeWorkerResult)> finals =
      <(Float32List, DecodeWorkerResult)>[];
  final List<DecodeWorkerResult> drafts = <DecodeWorkerResult>[];
  final List<(RefineRequestPayload, DecodeWorkerResult)> refines =
      <(RefineRequestPayload, DecodeWorkerResult)>[];
  final List<String> modelLoads = <String>[];
  final List<bool> busy = <bool>[];
  final List<String> decodeErrors = <String>[];
  final List<String> deaths = <String>[];
  int resets = 0;
  int refinePayloadCalls = 0;
}
