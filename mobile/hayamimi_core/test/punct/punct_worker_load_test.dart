import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/hayamimi_core.dart';

import 'punct_test_paths.dart';

/// The step where a session's punctuation model becomes a running
/// `PunctuatorJa`: `loadWorkerPunctuator`, given the same
/// [DecodeWorkerConfig] the decode worker isolate receives.
///
/// The rest of the worker cannot be exercised in a `flutter test` run — its
/// first act is to bind the sherpa-onnx native library, which no test VM can
/// load — but this part can, because ONNX Runtime is reachable on this host
/// (the same way the parity test reaches it). So the load is run in-process
/// instead of in an isolate: same config object, same function, no spawn.
///
/// The one test that needs the 182 MB model skips with a reason naming what
/// is missing; everything else here runs on any checkout.
void main() {
  final String? modelPath = PunctTestPaths.modelPath();
  final String? vocabPath = PunctTestPaths.vocabPath();
  final String? libraryPath = PunctTestPaths.ortLibraryPath();

  String? skipReason() {
    if (modelPath == null) {
      return 'the float16 punctuation model is not in '
          'models/mojicast-punct-onnx/quantized_ort/ — build it with '
          '"python scripts/quantize_punct.py --variant fp16"';
    }
    if (vocabPath == null) {
      return 'models/mojicast-punct-onnx/vocab.txt is missing — see '
          'docs/PUNCT_JA.md';
    }
    if (libraryPath == null) {
      return 'no ONNX Runtime shared library was found (on Windows it comes '
          'from the sherpa_onnx_windows package; elsewhere set '
          '${PunctTestPaths.libraryEnvVar})';
    }
    return null;
  }

  test('a config with punctuation paths builds a working punctuator', () async {
    final config = DecodeWorkerConfig(
      routed: false,
      modelDir: '/models/ja',
      punctModelPath: modelPath!,
      punctVocabPath: vocabPath!,
      punctLibraryPath: libraryPath,
    );
    final loads = <DecodeWorkerModelLoad>[];

    final punctuator = await loadWorkerPunctuator(
      config,
      onModelLoad: loads.add,
    );
    addTearDown(() => punctuator?.dispose());

    expect(punctuator, isNotNull);
    expect(punctuator!.restore('今日めっちゃ疲れたわもう寝る'), '今日めっちゃ疲れたわ。もう寝る。');

    // The progress a host UI sees while a session starts, reported under the
    // same model name every other model uses.
    expect(loads.map((e) => '${e.model}/${e.phase}'), <String>[
      'punct/start',
      'punct/done',
    ]);
    expect(loads.first.ms, isNull);
    expect(loads.last.ms, greaterThan(0));
  }, skip: skipReason());

  test('a config with no punctuation paths builds nothing', () async {
    const config = DecodeWorkerConfig(routed: false, modelDir: '/models/ja');

    expect(await loadWorkerPunctuator(config), isNull);
  });

  test('a model that cannot be read fails, naming both files', () async {
    const config = DecodeWorkerConfig(
      routed: false,
      modelDir: '/models/ja',
      punctModelPath: '/definitely/does/not/exist/punct.onnx',
      punctVocabPath: '/definitely/does/not/exist/vocab.txt',
    );
    final loads = <DecodeWorkerModelLoad>[];

    await expectLater(
      loadWorkerPunctuator(config, onModelLoad: loads.add),
      throwsA(
        isA<DecodeWorkerException>().having(
          (e) => e.message,
          'message',
          allOf(
            contains('Japanese punctuation model'),
            contains('/definitely/does/not/exist/punct.onnx'),
            contains('/definitely/does/not/exist/vocab.txt'),
          ),
        ),
      ),
    );

    // The failure is reported by the throw, so the load it announced is
    // left without a matching "done" -- the same shape a recognizer that
    // fails to build leaves behind.
    expect(loads.map((e) => '${e.model}/${e.phase}'), <String>['punct/start']);
  });

  test('a config cannot name the model without its vocabulary', () {
    // Both files or neither: a half-configured session would otherwise
    // reach the worker and fail there instead of where it was written.
    expect(
      () => DecodeWorkerConfig(
        routed: false,
        modelDir: '/models/ja',
        punctModelPath: '/models/punct/punct_bert.fp16.onnx',
      ),
      throwsAssertionError,
    );
    expect(
      () => DecodeWorkerConfig(
        routed: false,
        modelDir: '/models/ja',
        punctVocabPath: '/models/punct/vocab.txt',
      ),
      throwsAssertionError,
    );
  });
}
