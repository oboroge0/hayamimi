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
    encoder: resolveOnnxFile(filenames, role: 'encoder'),
    decoder: resolveOnnxFile(filenames, role: 'decoder'),
    joiner: resolveOnnxFile(filenames, role: 'joiner'),
    tokens: _pickExact(filenames, 'tokens.txt'),
  );
}

/// Resolves a single `.onnx` file out of a flat directory listing.
///
/// [role] is matched as a case-insensitive substring of the filename (e.g.
/// `"encoder"`); pass the default `""` to match any `.onnx` file
/// regardless of name, which is what a model shipped as one combined file
/// (e.g. SenseVoice) needs.
///
/// When multiple files match [role] and an int8-quantized variant is among
/// them, that variant is preferred -- this is the common
/// full-precision-plus-quantized layout sherpa-onnx release archives ship.
/// Pass [requireInt8]: true to instead reject the match entirely unless an
/// int8 variant is present (used where only the quantized model is ever
/// meant to be loaded, e.g. this app's whisper-tiny LID tier).
String resolveOnnxFile(
  List<String> filenames, {
  String role = '',
  bool requireInt8 = false,
}) {
  final needle = role.toLowerCase();
  final candidates =
      filenames
          .where(
            (f) =>
                f.toLowerCase().contains(needle) &&
                f.toLowerCase().endsWith('.onnx'),
          )
          .toList()
        ..sort();

  final int8Candidates = candidates
      .where((f) => f.toLowerCase().contains('int8'))
      .toList();

  if (requireInt8) {
    if (int8Candidates.isEmpty) {
      throw ModelFileResolutionException(
        'No int8 .onnx file containing "$role" found in model directory',
      );
    }
    return int8Candidates.first;
  }

  if (candidates.isEmpty) {
    throw ModelFileResolutionException(
      'No .onnx file containing "$role" found in model directory',
    );
  }
  return int8Candidates.isNotEmpty ? int8Candidates.first : candidates.first;
}

/// A pair of `.onnx` files resolved together by [resolveOnnxFilePair], e.g.
/// a whisper-style encoder + decoder.
class ResolvedOnnxPair {
  const ResolvedOnnxPair({required this.first, required this.second});

  final String first;
  final String second;
}

/// Resolves two related `.onnx` files (e.g. an encoder/decoder pair) out of
/// a flat directory listing in one call, applying [resolveOnnxFile]'s
/// int8-preference (or, with [requireInt8], int8-requirement) rule to each.
ResolvedOnnxPair resolveOnnxFilePair(
  List<String> filenames, {
  required String role1,
  required String role2,
  bool requireInt8 = false,
}) {
  return ResolvedOnnxPair(
    first: resolveOnnxFile(filenames, role: role1, requireInt8: requireInt8),
    second: resolveOnnxFile(filenames, role: role2, requireInt8: requireInt8),
  );
}

String _pickExact(List<String> filenames, String name) {
  final candidates = filenames.where((f) => f.toLowerCase() == name).toList();
  if (candidates.isEmpty) {
    throw ModelFileResolutionException('No "$name" found in model directory');
  }
  return candidates.first;
}
