import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_record_platform.dart';

/// Runtime-configuration surface added for the embedding API (issue #29):
/// the six pacing knobs' constructor defaults/validation/runtime setters,
/// and [LiveTranscriber.resetSession]/[LiveTranscriber.setVadSensitivity]
/// when no session is running. None of this needs the sherpa-onnx native
/// libs or a real mic -- only [LiveTranscriber]'s constructor and these
/// specific methods are exercised, following the same fake-`RecordPlatform`
/// approach `lifecycle_test.dart` uses so `AudioRecorder()` construction
/// doesn't touch a real platform channel.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late RecordPlatform originalPlatform;
  late FakeRecordPlatform fakePlatform;

  setUp(() {
    originalPlatform = RecordPlatform.instance;
    fakePlatform = FakeRecordPlatform();
    RecordPlatform.instance = fakePlatform;
  });

  tearDown(() async {
    await fakePlatform.close();
    RecordPlatform.instance = originalPlatform;
  });

  group('LiveTranscriber pacing-knob constructor', () {
    test('defaults match the default* constants from draft_pass.dart/refine_pass.dart', () {
      final transcriber = LiveTranscriber();
      addTearDown(transcriber.dispose);

      expect(transcriber.draftIntervalSeconds, defaultDraftIntervalSeconds);
      expect(transcriber.draftWindowSeconds, defaultDraftWindowSeconds);
      expect(transcriber.minDraftAudioSeconds, defaultMinDraftAudioSeconds);
      expect(
        transcriber.autoRefineSilenceSeconds,
        defaultAutoRefineSilenceSeconds,
      );
      expect(
        transcriber.autoRefineMaxBufferedSeconds,
        defaultAutoRefineMaxBufferedSeconds,
      );
      expect(transcriber.refineBufferMaxSeconds, defaultRefineBufferMaxSeconds);
    });

    test('accepts and reads back custom values', () {
      final transcriber = LiveTranscriber(
        draftIntervalSeconds: 2.0,
        draftWindowSeconds: 5.0,
        minDraftAudioSeconds: 0.5,
        autoRefineSilenceSeconds: 3.0,
        autoRefineMaxBufferedSeconds: 15.0,
        refineBufferMaxSeconds: 30.0,
      );
      addTearDown(transcriber.dispose);

      expect(transcriber.draftIntervalSeconds, 2.0);
      expect(transcriber.draftWindowSeconds, 5.0);
      expect(transcriber.minDraftAudioSeconds, 0.5);
      expect(transcriber.autoRefineSilenceSeconds, 3.0);
      expect(transcriber.autoRefineMaxBufferedSeconds, 15.0);
      expect(transcriber.refineBufferMaxSeconds, 30.0);
    });

    test('a non-positive or non-finite value throws ArgumentError', () {
      expect(
        () => LiveTranscriber(draftIntervalSeconds: 0),
        throwsArgumentError,
      );
      expect(
        () => LiveTranscriber(draftWindowSeconds: -1),
        throwsArgumentError,
      );
      expect(
        () => LiveTranscriber(minDraftAudioSeconds: double.nan),
        throwsArgumentError,
      );
      expect(
        () => LiveTranscriber(autoRefineSilenceSeconds: double.infinity),
        throwsArgumentError,
      );
      expect(
        () => LiveTranscriber(autoRefineMaxBufferedSeconds: 0),
        throwsArgumentError,
      );
      expect(
        () => LiveTranscriber(refineBufferMaxSeconds: -5),
        throwsArgumentError,
      );
    });
  });

  group('LiveTranscriber pacing-knob runtime setters', () {
    test('validate and are readable back', () {
      final transcriber = LiveTranscriber();
      addTearDown(transcriber.dispose);

      transcriber.draftIntervalSeconds = 3.0;
      expect(transcriber.draftIntervalSeconds, 3.0);

      transcriber.draftWindowSeconds = 6.0;
      expect(transcriber.draftWindowSeconds, 6.0);

      transcriber.minDraftAudioSeconds = 0.4;
      expect(transcriber.minDraftAudioSeconds, 0.4);

      transcriber.autoRefineSilenceSeconds = 2.5;
      expect(transcriber.autoRefineSilenceSeconds, 2.5);

      transcriber.autoRefineMaxBufferedSeconds = 25.0;
      expect(transcriber.autoRefineMaxBufferedSeconds, 25.0);

      transcriber.refineBufferMaxSeconds = 45.0;
      expect(transcriber.refineBufferMaxSeconds, 45.0);
    });

    test('an out-of-range value throws and leaves the previous value in place', () {
      final transcriber = LiveTranscriber(draftIntervalSeconds: 1.5);
      addTearDown(transcriber.dispose);

      expect(
        () => transcriber.draftIntervalSeconds = -1,
        throwsArgumentError,
      );
      expect(transcriber.draftIntervalSeconds, 1.5);

      expect(
        () => transcriber.refineBufferMaxSeconds = 0,
        throwsArgumentError,
      );
      expect(transcriber.refineBufferMaxSeconds, defaultRefineBufferMaxSeconds);
    });
  });

  group('LiveTranscriber.resetSession', () {
    test('is a no-op and emits nothing on sessionResets when not running', () async {
      final transcriber = LiveTranscriber();
      addTearDown(transcriber.dispose);

      final resets = <void>[];
      transcriber.sessionResets.listen(resets.add);

      await transcriber.resetSession();
      await Future<void>.delayed(Duration.zero);

      expect(resets, isEmpty);
    });
  });

  group('LiveTranscriber.setVadSensitivity', () {
    test(
      'when no session is running, stores the value for the next start() '
      'without building anything (no modelLoads events)',
      () async {
        final transcriber = LiveTranscriber();
        addTearDown(transcriber.dispose);

        final loads = <ModelLoadEvent>[];
        transcriber.modelLoads.listen(loads.add);

        await transcriber.setVadSensitivity(VadSensitivity(threshold: 0.8));
        await Future<void>.delayed(Duration.zero);

        expect(loads, isEmpty);
      },
    );
  });

  group('LiveTranscriber.start failure cleanup', () {
    // Regression coverage for the native-state-leak fix: start() used to
    // call _buildNativeState unguarded, so a failure partway through it
    // (e.g. the recognizer built fine but the VAD model didn't) left
    // whatever was already built never freed, with isRunning still false
    // so stop()/dispose() had nothing to tear down. A nonexistent model
    // directory can't reach that exact partial-build point without real
    // model files, but it does exercise the same try/catch/teardown path
    // this fix added around _buildNativeState, and confirms start() keeps
    // reporting the truth (not running) and stays retryable afterward.
    test(
      'a failed start() (model directory not found) throws and leaves '
      'isRunning false',
      () async {
        final transcriber = LiveTranscriber();
        addTearDown(transcriber.dispose);

        await expectLater(
          transcriber.start(
            modelKind: ModelKind.zipformerTransducer,
            modelDir: 'C:/definitely/does/not/exist/model',
            vadModelPath: 'C:/definitely/does/not/exist/vad.onnx',
          ),
          throwsA(isA<LiveTranscriberException>()),
        );

        expect(transcriber.isRunning, isFalse);

        // stop() must stay a safe no-op: nothing native was ever built to
        // tear down, so this must not throw.
        await transcriber.stop();
      },
    );

    test(
      'start() can be retried after a failure (the _starting guard resets '
      'even when _buildNativeState throws)',
      () async {
        final transcriber = LiveTranscriber();
        addTearDown(transcriber.dispose);

        for (var i = 0; i < 2; i++) {
          await expectLater(
            transcriber.start(
              modelKind: ModelKind.zipformerTransducer,
              modelDir: 'C:/definitely/does/not/exist/model',
              vadModelPath: 'C:/definitely/does/not/exist/vad.onnx',
            ),
            throwsA(isA<LiveTranscriberException>()),
          );
        }
        expect(transcriber.isRunning, isFalse);
      },
    );
  });

  group('LiveTranscriber.setVadSensitivity during start()', () {
    // Regression coverage for the "queued while _starting" fix:
    // setVadSensitivity used to treat _starting the same as fully idle,
    // applying its value to the _vadSensitivity field immediately -- which
    // start()'s own in-flight _buildNativeState call would then silently
    // overwrite with whatever it was originally given. A call made while
    // start() is loading (here, failing fast on a bad model dir) must not
    // throw and must not build anything of its own.
    test(
      'a call made while start() is loading does not throw and builds '
      'nothing on its own',
      () async {
        final transcriber = LiveTranscriber();
        addTearDown(transcriber.dispose);

        final loads = <ModelLoadEvent>[];
        transcriber.modelLoads.listen(loads.add);

        final startFuture = expectLater(
          transcriber.start(
            modelKind: ModelKind.zipformerTransducer,
            modelDir: 'C:/definitely/does/not/exist/model',
            vadModelPath: 'C:/definitely/does/not/exist/vad.onnx',
          ),
          throwsA(isA<LiveTranscriberException>()),
        );

        // start() is now awaiting its first `await` (the permission check)
        // with _starting already true -- call setVadSensitivity in that
        // window rather than after start() has already failed.
        await transcriber.setVadSensitivity(VadSensitivity(threshold: 0.9));

        await startFuture;
        await Future<void>.delayed(Duration.zero);

        expect(transcriber.isRunning, isFalse);
        expect(loads, isEmpty);
      },
    );
  });
}
