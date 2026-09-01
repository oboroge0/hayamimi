import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/punct/punctuator_ja.dart';

import 'punct_parity_fixture.dart';
import 'punct_test_paths.dart';

/// End-to-end parity: the Dart port must return exactly what
/// `scripts/punct_ja.py` returned for the same input, character for
/// character, on real Japanese sentences.
///
/// This is the test that makes the port trustworthy — the unit tests below
/// it only check that each piece behaves as written, not that the pieces
/// add up to the same behaviour as the reference implementation.
///
/// It needs two files the repository does not ship: the 182 MB float16
/// model (a local artifact of `scripts/quantize_punct.py --variant fp16`)
/// and an ONNX Runtime shared library. Without either, it skips with a
/// reason naming what is missing, so a checkout without them stays green.
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

  group('PunctuatorJa matches scripts/punct_ja.py', () {
    late PunctParityFixture fixture;
    late PunctuatorJa punctuator;

    setUpAll(() async {
      fixture = PunctParityFixture.load();
      punctuator = await PunctuatorJa.load(
        modelPath: modelPath!,
        vocabPath: vocabPath!,
        libraryPath: libraryPath,
      );
    });

    tearDownAll(() {
      punctuator.dispose();
    });

    test('restores all recorded cases identically', () {
      final List<String> failures = <String>[];
      final Stopwatch stopwatch = Stopwatch();
      int runs = 0;

      for (final PunctParityCase testCase in fixture.cases) {
        stopwatch.start();
        final String actual = punctuator.restore(testCase.input);
        stopwatch.stop();
        runs++;
        if (actual != testCase.expected) {
          failures.add(
            '${testCase.name}\n'
            '  input   : ${testCase.input}\n'
            '  expected: ${testCase.expected}\n'
            '  actual  : $actual',
          );
        }
      }

      final double meanMs = stopwatch.elapsedMicroseconds / 1000.0 / runs;
      stdout.writeln(
        'punct parity: ${fixture.cases.length} cases '
        '(${fixture.cases.where((PunctParityCase c) => c.source == 'fleurs').length} '
        'FLEURS ja + '
        '${fixture.cases.where((PunctParityCase c) => c.source != 'fleurs').length} '
        'synthetic), ${failures.length} mismatches; '
        'mean restore() ${meanMs.toStringAsFixed(1)} ms on this Windows x86 '
        'host, ONNX Runtime ${punctuator.runtimeVersion}, model '
        '${fixture.model}. This says nothing about a phone: the CPU '
        'provider has no float16 path on x86, so float16 tensors are cast '
        'to float32 and back on every operator.',
      );

      expect(
        failures,
        isEmpty,
        reason:
            '${failures.length} of ${fixture.cases.length} cases differ from '
            'the Python reference:\n\n${failures.join('\n\n')}',
      );
    });

    test('reports the ONNX Runtime it actually loaded', () {
      expect(punctuator.runtimeVersion, matches(RegExp(r'^\d+\.\d+')));
    });
  }, skip: skipReason());
}
