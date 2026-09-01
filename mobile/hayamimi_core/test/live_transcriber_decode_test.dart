import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';
import 'package:record/record.dart';

import 'fake_decode_worker.dart';
import 'fake_record_platform.dart';

/// How `LiveTranscriber` hands a session to its decode worker, and what it
/// does when that hand-off fails.
///
/// Only the worker side is faked here. The Silero VAD still loads over FFI,
/// which a `flutter test` run cannot do, so these tests stop at the point
/// the VAD would be built — which is exactly far enough to cover the
/// hand-off, the error text a caller sees, and the promise that a failure
/// leaves no worker behind.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late RecordPlatform originalPlatform;
  late FakeRecordPlatform fakePlatform;
  late FakeDecodeWorker worker;
  late Directory modelDir;
  late File vadModelFile;

  setUp(() {
    originalPlatform = RecordPlatform.instance;
    fakePlatform = FakeRecordPlatform();
    RecordPlatform.instance = fakePlatform;
    worker = FakeDecodeWorker();
    modelDir = Directory.systemTemp.createTempSync('hayamimi_models');
    vadModelFile = File('${modelDir.path}${Platform.pathSeparator}vad.onnx')
      ..writeAsBytesSync(<int>[0]);
  });

  tearDown(() async {
    await fakePlatform.close();
    RecordPlatform.instance = originalPlatform;
    if (modelDir.existsSync()) {
      modelDir.deleteSync(recursive: true);
    }
  });

  LiveTranscriber newTranscriber() =>
      LiveTranscriber(decodeWorkerFactory: () => worker);

  group('start hands the session to the decode worker', () {
    test('a plain session forwards its model directory and decoding method', () async {
      worker.buildFailure = 'stop here';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          decodingMethod: 'modified_beam_search',
          hotwordsFile: '/hotwords.txt',
          hotwordsScore: 3,
        ),
        throwsA(isA<LiveTranscriberException>()),
      );

      final config = worker.config!;
      expect(config.routed, isFalse);
      expect(config.modelDir, modelDir.path);
      expect(config.decodingMethod, 'modified_beam_search');
      expect(config.hotwordsFile, '/hotwords.txt');
      expect(config.hotwordsScore, 3);
      expect(config.sampleRate, LiveTranscriber.sampleRate);
    });

    test('a routed session forwards all three model directories', () async {
      worker.buildFailure = 'stop here';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          routingProfile: RoutingProfile.jaSenseVoice,
          senseVoiceModelDir: '/models/sense_voice',
          lidModelDir: '/models/lid',
        ),
        throwsA(isA<LiveTranscriberException>()),
      );

      final config = worker.config!;
      expect(config.routed, isTrue);
      expect(config.senseVoiceModelDir, '/models/sense_voice');
      expect(config.lidModelDir, '/models/lid');
    });

    test('a routed session without its extra directories never reaches the worker', () async {
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          routingProfile: RoutingProfile.jaSenseVoice,
        ),
        throwsA(
          isA<LiveTranscriberException>().having(
            (e) => e.message,
            'message',
            contains('requires senseVoiceModelDir and lidModelDir'),
          ),
        ),
      );
      expect(worker.startCount, 0);
    });
  });

  group('Japanese punctuation', () {
    late File punctModelFile;
    late File punctVocabFile;

    setUp(() {
      // Stand-ins for the 181.8 MB model and its vocabulary: start() only
      // checks that both files exist, and the fake worker never loads them.
      final sep = Platform.pathSeparator;
      punctModelFile = File('${modelDir.path}${sep}punct.onnx')
        ..writeAsBytesSync(<int>[0]);
      punctVocabFile = File('${modelDir.path}${sep}vocab.txt')
        ..writeAsStringSync('[PAD]\n');
    });

    JaPunctuation punctuation({String? libraryPath, int numThreads = 2}) =>
        JaPunctuation(
          modelPath: punctModelFile.path,
          vocabPath: punctVocabFile.path,
          libraryPath: libraryPath,
          numThreads: numThreads,
        );

    test('reaches the worker as part of its config', () async {
      worker.buildFailure = 'stop here';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          punctuation: punctuation(
            libraryPath: '/lib/onnxruntime.dll',
            numThreads: 3,
          ),
        ),
        throwsA(isA<LiveTranscriberException>()),
      );

      final config = worker.config!;
      expect(config.punctModelPath, punctModelFile.path);
      expect(config.punctVocabPath, punctVocabFile.path);
      expect(config.punctLibraryPath, '/lib/onnxruntime.dll');
      expect(config.punctNumThreads, 3);
      // A plain session says its own language is Japanese by being given a
      // Japanese punctuation model at all.
      expect(config.punctuatePlainSession, isTrue);
      expect(config.shouldPunctuateRefine(), isTrue);
    });

    test('a session started without it asks the worker for none', () async {
      worker.buildFailure = 'stop here';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
        ),
        throwsA(isA<LiveTranscriberException>()),
      );

      expect(worker.config!.hasPunctuation, isFalse);
    });

    test('a model that will not load fails start(), naming it', () async {
      // No silent degradation: a missing or unreadable 182 MB file is a
      // configuration error, and a session that came up quietly producing
      // unpunctuated text would hide it.
      worker.buildFailure =
          'Could not load the Japanese punctuation model '
          '"${punctModelFile.path}" (vocabulary "${punctVocabFile.path}"): '
          'OrtLibraryException: Could not load "libonnxruntime.so"';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          punctuation: punctuation(),
        ),
        throwsA(
          isA<LiveTranscriberException>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('Japanese punctuation model'),
              contains(punctModelFile.path),
            ),
          ),
        ),
      );

      expect(transcriber.isRunning, isFalse);
      expect(worker.shutdownCount, 1);
    });

    test('a missing file is reported before any worker is spawned', () async {
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);
      final missing = '${modelDir.path}${Platform.pathSeparator}absent.onnx';

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          punctuation: JaPunctuation(
            modelPath: missing,
            vocabPath: punctVocabFile.path,
          ),
        ),
        throwsA(
          isA<LiveTranscriberException>().having(
            (e) => e.message,
            'message',
            'Japanese punctuation model not found: $missing',
          ),
        ),
      );

      // Nothing was spawned and no recognizer weights were loaded for a
      // session that could not have worked.
      expect(worker.startCount, 0);
    });

    test('a missing vocabulary is reported the same way', () async {
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);
      final missing = '${modelDir.path}${Platform.pathSeparator}absent.txt';

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
          punctuation: JaPunctuation(
            modelPath: punctModelFile.path,
            vocabPath: missing,
          ),
        ),
        throwsA(
          isA<LiveTranscriberException>().having(
            (e) => e.message,
            'message',
            'Japanese punctuation vocabulary not found: $missing',
          ),
        ),
      );

      expect(worker.startCount, 0);
    });
  });

  group('a model that will not load', () {
    test('surfaces the worker message verbatim and stays retryable', () async {
      worker.buildFailure =
          'Could not find encoder/decoder/joiner onnx files in the model '
          'directory.';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      Future<void> attempt() => transcriber.start(
        modelKind: ModelKind.zipformerTransducer,
        modelDir: modelDir.path,
        vadModelPath: vadModelFile.path,
      );

      await expectLater(
        attempt(),
        throwsA(
          isA<LiveTranscriberException>().having(
            (e) => e.message,
            'message',
            worker.buildFailure,
          ),
        ),
      );

      expect(transcriber.isRunning, isFalse);
      expect(transcriber.currentLang, isNull);
      // The worker is gone rather than left running with models loaded.
      expect(worker.shutdownCount, 1);

      // And the guard that would otherwise wedge a retry has been released.
      await expectLater(attempt(), throwsA(isA<LiveTranscriberException>()));
      expect(worker.startCount, 2);

      // stop() on a session that never came up stays a safe no-op.
      await transcriber.stop();
    });

    test('model-load progress from before the failure still reaches modelLoads', () async {
      worker.modelLoads.addAll(const <DecodeWorkerModelLoad>[
        DecodeWorkerModelLoad(model: 'ja', phase: 'start'),
        DecodeWorkerModelLoad(model: 'ja', phase: 'done', ms: 1200),
      ]);
      worker.buildFailure = 'SenseVoice model/tokens not found in: /models/sv';
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      final loads = <ModelLoadEvent>[];
      transcriber.modelLoads.listen(loads.add);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
        ),
        throwsA(isA<LiveTranscriberException>()),
      );
      await settle();

      expect(loads.map((e) => '${e.model}/${e.phase}'), <String>[
        'ja/start',
        'ja/done',
      ]);
      expect(loads.last.ms, 1200);
    });
  });

  group('a failure after the worker is already up', () {
    test('still shuts the worker down instead of leaking it', () async {
      // The worker comes up fine; the VAD build right after it is what
      // fails, because loading Silero VAD needs the sherpa-onnx native
      // library that a `flutter test` run has no way to open. That is the
      // same shape as the real "good recognizer load, bad VAD model file"
      // case, and it is the path that used to leak.
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      await expectLater(
        transcriber.start(
          modelKind: ModelKind.zipformerTransducer,
          modelDir: modelDir.path,
          vadModelPath: vadModelFile.path,
        ),
        throwsA(anything),
      );

      expect(worker.startCount, 1);
      expect(worker.shutdownCount, 1);
      expect(transcriber.isRunning, isFalse);
    });
  });

  group('with no session running', () {
    test('refineNow and resetSession do nothing and reach no worker', () async {
      final transcriber = newTranscriber();
      addTearDown(transcriber.dispose);

      final refines = <LiveTranscriptEntry>[];
      final resets = <void>[];
      transcriber.refineEntries.listen(refines.add);
      transcriber.sessionResets.listen(resets.add);

      await transcriber.refineNow();
      await transcriber.resetSession();
      await settle();

      expect(worker.startCount, 0);
      expect(worker.commands, isEmpty);
      expect(refines, isEmpty);
      expect(resets, isEmpty);
    });
  });
}
