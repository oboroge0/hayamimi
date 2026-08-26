import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_mobile/bench/model_file_resolver.dart';

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
}
