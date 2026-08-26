/// Picks the encoder/decoder/joiner/tokens files for an offline zipformer
/// transducer model out of a flat list of filenames in a model directory.
///
/// sherpa-onnx release archives don't use a fixed filename (e.g.
/// `encoder-epoch-99-avg-1.int8.onnx`), so this matches by substring instead
/// of an exact name. When a directory contains both a full-precision and an
/// int8 variant of the same role, the int8 one is preferred (it's the
/// variant this bench targets).
class ResolvedModelFiles {
  const ResolvedModelFiles({
    required this.encoder,
    required this.decoder,
    required this.joiner,
    required this.tokens,
  });

  final String encoder;
  final String decoder;
  final String joiner;
  final String tokens;
}

class ModelFileResolutionException implements Exception {
  ModelFileResolutionException(this.message);
  final String message;

  @override
  String toString() => 'ModelFileResolutionException: $message';
}

ResolvedModelFiles resolveZipformerTransducerFiles(List<String> filenames) {
  return ResolvedModelFiles(
    encoder: _pick(filenames, 'encoder'),
    decoder: _pick(filenames, 'decoder'),
    joiner: _pick(filenames, 'joiner'),
    tokens: _pickExact(filenames, 'tokens.txt'),
  );
}

String _pick(List<String> filenames, String role) {
  final candidates = filenames
      .where((f) => f.toLowerCase().contains(role) && f.toLowerCase().endsWith('.onnx'))
      .toList()
    ..sort();

  if (candidates.isEmpty) {
    throw ModelFileResolutionException(
      'No .onnx file containing "$role" found in model directory',
    );
  }

  final int8Candidates = candidates.where((f) => f.toLowerCase().contains('int8')).toList();
  return int8Candidates.isNotEmpty ? int8Candidates.first : candidates.first;
}

String _pickExact(List<String> filenames, String name) {
  final candidates = filenames.where((f) => f.toLowerCase() == name).toList();
  if (candidates.isEmpty) {
    throw ModelFileResolutionException('No "$name" found in model directory');
  }
  return candidates.first;
}
