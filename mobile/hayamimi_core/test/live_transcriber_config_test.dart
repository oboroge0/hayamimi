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
}
