import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/punct/punct_ja_tokenizer.dart';

import 'punct_parity_fixture.dart';
import 'punct_test_paths.dart';

/// Tokenizer parity, without the model.
///
/// The Dart tokenizer replaces MeCab with a plain "NFKC, drop whitespace,
/// split into code points" rule (see [PunctJaTokenizer] for why that is
/// equivalent here). This test checks the replacement against the real
/// thing on every recorded sentence: the ids Python built have to be the
/// ids Dart builds, exactly.
///
/// It is separate from the end-to-end parity test on purpose. This one
/// needs only `vocab.txt` — 28 KB — so tokenizer drift is still caught on a
/// machine that does not have the 182 MB model or an ONNX Runtime library.
void main() {
  final String? vocabPath = PunctTestPaths.vocabPath();

  group('PunctJaTokenizer matches the Python tokenizer', () {
    late PunctParityFixture fixture;
    late PunctJaTokenizer tokenizer;

    setUpAll(() async {
      fixture = PunctParityFixture.load();
      tokenizer = PunctJaTokenizer(await PunctJaVocab.loadFile(vocabPath!));
    });

    test('produces the recorded input ids for every case', () {
      final List<String> failures = <String>[];
      for (final PunctParityCase testCase in fixture.cases) {
        // restore() strips before tokenizing, so the fixture's ids were
        // recorded from the stripped text and this has to do the same.
        final PunctJaTokens tokens = tokenizer.tokenize(
          PunctJaTokenizer.pythonStrip(testCase.input),
        );
        if (!_sameIds(tokens.inputIds, testCase.inputIds)) {
          failures.add(
            '${testCase.name}\n'
            '  input   : ${testCase.input}\n'
            '  expected: ${testCase.inputIds}\n'
            '  actual  : ${tokens.inputIds}',
          );
        }
      }
      expect(
        failures,
        isEmpty,
        reason:
            '${failures.length} of ${fixture.cases.length} cases tokenize '
            'differently from scripts/punct_ja.py:\n\n${failures.join('\n\n')}',
      );
    });

    test('covers the whole recorded fixture', () {
      expect(fixture.cases.length, greaterThanOrEqualTo(40));
      expect(
        fixture.cases
            .where((PunctParityCase c) => c.source == 'fleurs')
            .length,
        40,
      );
    });
  }, skip: vocabPath == null
      ? 'models/mojicast-punct-onnx/vocab.txt is missing — see '
            'docs/design/punct_ja.md for how to fetch the model directory'
      : null);
}

bool _sameIds(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (int i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
