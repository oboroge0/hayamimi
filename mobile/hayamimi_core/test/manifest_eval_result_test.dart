import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/bench/manifest_eval_result.dart';
import 'package:hayamimi_core/bench/manifest_eval_runner.dart';

void main() {
  group('ManifestEntry.fromJson', () {
    test('parses wav/lang/ref from a manifest.json entry', () {
      final entry = ManifestEntry.fromJson({
        'wav': 'ja_01.wav',
        'lang': 'ja',
        'ref': 'こんにちは',
      });
      expect(entry.wav, 'ja_01.wav');
      expect(entry.lang, 'ja');
      expect(entry.ref, 'こんにちは');
    });
  });

  group('ManifestEvalResult', () {
    test('rtf divides decode time by audio duration', () {
      const result = ManifestEvalResult(
        wav: 'ja_01.wav',
        lang: 'ja',
        ref: 'こんにちは',
        hyp: 'こんにちわ',
        audioDurationSeconds: 4.0,
        decodeSeconds: 1.0,
      );
      expect(result.rtf, closeTo(0.25, 1e-9));
    });

    test('rtf is 0 for zero-length audio instead of dividing by zero', () {
      const result = ManifestEvalResult(
        wav: 'ja_01.wav',
        lang: 'ja',
        ref: 'こんにちは',
        hyp: '',
        audioDurationSeconds: 0.0,
        decodeSeconds: 1.0,
      );
      expect(result.rtf, 0);
    });

    test('toJson/fromJson round-trips', () {
      const result = ManifestEvalResult(
        wav: 'ja_02.wav',
        lang: 'ja',
        ref: 'ピカピカブ',
        hyp: 'ピカピカブ',
        audioDurationSeconds: 2.5,
        decodeSeconds: 0.1,
      );
      final restored = ManifestEvalResult.fromJson(result.toJson());
      expect(restored.wav, result.wav);
      expect(restored.lang, result.lang);
      expect(restored.ref, result.ref);
      expect(restored.hyp, result.hyp);
      expect(restored.audioDurationSeconds, result.audioDurationSeconds);
      expect(restored.decodeSeconds, result.decodeSeconds);
    });
  });

  group('ManifestEvalRunner.toJson', () {
    test('serializes a list of results as a JSON array', () {
      const results = [
        ManifestEvalResult(
          wav: 'ja_01.wav',
          lang: 'ja',
          ref: 'ref1',
          hyp: 'hyp1',
          audioDurationSeconds: 1.0,
          decodeSeconds: 0.1,
        ),
        ManifestEvalResult(
          wav: 'ja_02.wav',
          lang: 'ja',
          ref: 'ref2',
          hyp: 'hyp2',
          audioDurationSeconds: 2.0,
          decodeSeconds: 0.2,
        ),
      ];
      final decoded = jsonDecode(ManifestEvalRunner.toJson(results)) as List;
      expect(decoded, hasLength(2));
      expect(decoded[0]['wav'], 'ja_01.wav');
      expect(decoded[1]['hyp'], 'hyp2');
    });
  });
}
