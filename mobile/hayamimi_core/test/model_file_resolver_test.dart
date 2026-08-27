import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/bench/model_file_resolver.dart';

void main() {
  group('resolveZipformerTransducerFiles', () {
    test('matches encoder/decoder/joiner/tokens by substring', () {
      final resolved = resolveZipformerTransducerFiles([
        'encoder-epoch-99-avg-1.int8.onnx',
        'decoder-epoch-99-avg-1.onnx',
        'joiner-epoch-99-avg-1.int8.onnx',
        'tokens.txt',
        'README.md',
      ]);

      expect(resolved.encoder, 'encoder-epoch-99-avg-1.int8.onnx');
      expect(resolved.decoder, 'decoder-epoch-99-avg-1.onnx');
      expect(resolved.joiner, 'joiner-epoch-99-avg-1.int8.onnx');
      expect(resolved.tokens, 'tokens.txt');
    });

    test('prefers int8 variant when both fp32 and int8 are present', () {
      final resolved = resolveZipformerTransducerFiles([
        'encoder-epoch-99-avg-1.onnx',
        'encoder-epoch-99-avg-1.int8.onnx',
        'decoder-epoch-99-avg-1.onnx',
        'joiner-epoch-99-avg-1.onnx',
        'joiner-epoch-99-avg-1.int8.onnx',
        'tokens.txt',
      ]);

      expect(resolved.encoder, contains('int8'));
      expect(resolved.joiner, contains('int8'));
    });

    test('throws ModelFileResolutionException when encoder is missing', () {
      expect(
        () => resolveZipformerTransducerFiles([
          'decoder-epoch-99-avg-1.onnx',
          'joiner-epoch-99-avg-1.onnx',
          'tokens.txt',
        ]),
        throwsA(isA<ModelFileResolutionException>()),
      );
    });

    test('throws ModelFileResolutionException when tokens.txt is missing', () {
      expect(
        () => resolveZipformerTransducerFiles([
          'encoder-epoch-99-avg-1.onnx',
          'decoder-epoch-99-avg-1.onnx',
          'joiner-epoch-99-avg-1.onnx',
        ]),
        throwsA(isA<ModelFileResolutionException>()),
      );
    });

    test('is case-insensitive on role and extension', () {
      final resolved = resolveZipformerTransducerFiles([
        'ENCODER-x.ONNX',
        'decoder-x.onnx',
        'joiner-x.onnx',
        'TOKENS.TXT',
      ]);

      expect(resolved.encoder, 'ENCODER-x.ONNX');
      expect(resolved.tokens, 'TOKENS.TXT');
    });
  });

  group('resolveOnnxFile', () {
    test('matches any .onnx file when role is left blank', () {
      final resolved = resolveOnnxFile(['model.onnx', 'tokens.txt']);
      expect(resolved, 'model.onnx');
    });

    test('matches by role substring, case-insensitively', () {
      final resolved = resolveOnnxFile([
        'MODEL-Encoder.ONNX',
        'model-decoder.onnx',
      ], role: 'encoder');
      expect(resolved, 'MODEL-Encoder.ONNX');
    });

    test('prefers the int8 variant when both are present', () {
      final resolved = resolveOnnxFile([
        'model.onnx',
        'model.int8.onnx',
      ]);
      expect(resolved, 'model.int8.onnx');
    });

    test('throws when no .onnx file matches the role', () {
      expect(
        () => resolveOnnxFile(['model.onnx'], role: 'decoder'),
        throwsA(isA<ModelFileResolutionException>()),
      );
    });

    test('requireInt8 rejects a match with no int8 variant', () {
      expect(
        () => resolveOnnxFile(['model.onnx'], requireInt8: true),
        throwsA(isA<ModelFileResolutionException>()),
      );
    });

    test('requireInt8 returns the int8 variant when present', () {
      final resolved = resolveOnnxFile([
        'model.onnx',
        'model.int8.onnx',
      ], requireInt8: true);
      expect(resolved, 'model.int8.onnx');
    });
  });

  group('resolveOnnxFilePair', () {
    test('resolves both roles independently', () {
      final resolved = resolveOnnxFilePair(
        [
          'whisper-encoder.int8.onnx',
          'whisper-decoder.int8.onnx',
          'README.md',
        ],
        role1: 'encoder',
        role2: 'decoder',
        requireInt8: true,
      );

      expect(resolved.first, 'whisper-encoder.int8.onnx');
      expect(resolved.second, 'whisper-decoder.int8.onnx');
    });

    test('throws when one side has no int8 match and requireInt8 is set', () {
      expect(
        () => resolveOnnxFilePair(
          ['whisper-encoder.int8.onnx', 'whisper-decoder.onnx'],
          role1: 'encoder',
          role2: 'decoder',
          requireInt8: true,
        ),
        throwsA(isA<ModelFileResolutionException>()),
      );
    });
  });
}
