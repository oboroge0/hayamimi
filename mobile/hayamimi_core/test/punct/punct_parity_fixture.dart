import 'dart:convert';
import 'dart:io';

/// One recorded run of the reference implementation.
class PunctParityCase {
  const PunctParityCase({
    required this.name,
    required this.source,
    required this.input,
    required this.inputIds,
    required this.expected,
  });

  /// Identifies the case in a failure message.
  final String name;

  /// `fleurs` for a sentence taken from the FLEURS ja benchmark set,
  /// `synthetic` for one written for the fixture.
  final String source;

  final String input;

  /// `[CLS]` + one id per character + `[SEP]`, as Python built it.
  final List<int> inputIds;

  /// What `scripts/punct_ja.py`'s `restore()` returned.
  final String expected;
}

/// The recorded output of `scripts/punct_ja.py`, replayed by the Dart
/// tests to prove the port agrees with it.
///
/// Regenerate with `python scripts/make_punct_fixture.py`.
class PunctParityFixture {
  const PunctParityFixture({
    required this.model,
    required this.onnxruntimeVersion,
    required this.cases,
  });

  static const String path = 'test/fixtures/punct_ja_parity.json';

  factory PunctParityFixture.load() {
    final Map<String, dynamic> json =
        jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    return PunctParityFixture(
      model: json['_model'] as String,
      onnxruntimeVersion: json['_onnxruntime'] as String,
      cases: <PunctParityCase>[
        for (final Object? entry in json['cases'] as List<dynamic>)
          PunctParityCase(
            name: (entry as Map<String, dynamic>)['name'] as String,
            source: entry['source'] as String,
            input: entry['input'] as String,
            inputIds: (entry['input_ids'] as List<dynamic>).cast<int>(),
            expected: entry['expected'] as String,
          ),
      ],
    );
  }

  /// The model file the expectations were recorded with.
  final String model;

  /// The onnxruntime version Python used, for the record: the Dart side
  /// runs against whatever ONNX Runtime `sherpa_onnx` ships, so the two
  /// are deliberately allowed to differ.
  final String onnxruntimeVersion;

  final List<PunctParityCase> cases;
}
