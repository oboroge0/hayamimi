import 'dart:convert';
import 'dart:io';

/// Where the punctuation tests find the two large files they need.
///
/// Neither the 182 MB model nor an ONNX Runtime shared library is part of
/// the package, so every test that needs one has to be able to say "not
/// here" and skip instead of failing. These helpers return null rather
/// than throwing, and the tests turn a null into a skip reason that names
/// the missing file.
class PunctTestPaths {
  const PunctTestPaths._();

  /// Set this environment variable to point the parity test at an ONNX
  /// Runtime shared library on a platform this helper cannot find one on.
  static const String libraryEnvVar = 'HAYAMIMI_ORT_LIBRARY';

  /// The repository root, found by walking up from the current directory
  /// until a `models` directory appears. `flutter test` runs with the
  /// package directory as the working directory, so the root is two levels
  /// up — but the walk keeps this working from a worktree, a checkout at
  /// another depth, or an IDE that sets a different directory.
  static Directory? repoRoot() {
    Directory dir = Directory.current.absolute;
    for (int i = 0; i < 6; i++) {
      if (Directory('${dir.path}/models/mojicast-punct-onnx').existsSync()) {
        return dir;
      }
      final Directory parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }

  /// The float16 punctuation model, or null if it has not been generated.
  ///
  /// It is a local build artifact of
  /// `python scripts/quantize_punct.py --variant fp16`, not something the
  /// repository ships.
  static String? modelPath() {
    final Directory? root = repoRoot();
    if (root == null) return null;
    final String path =
        '${root.path}/models/mojicast-punct-onnx/quantized_ort/'
        'punct_bert.fp16.onnx';
    return File(path).existsSync() ? path : null;
  }

  /// The model's `vocab.txt`, or null if the model directory is missing.
  static String? vocabPath() {
    final Directory? root = repoRoot();
    if (root == null) return null;
    final String path = '${root.path}/models/mojicast-punct-onnx/vocab.txt';
    return File(path).existsSync() ? path : null;
  }

  /// An ONNX Runtime shared library to run the model with, or null.
  ///
  /// On a host, nothing has ONNX Runtime loaded already — the whole point
  /// of the production design is to borrow the one `sherpa_onnx` loads on
  /// a device. So the test borrows the same library from the package's own
  /// dependency instead: `sherpa_onnx_windows` ships `onnxruntime.dll`, and
  /// `.dart_tool/package_config.json` says where pub unpacked it. Reading
  /// the resolved path keeps this working across version bumps.
  static String? ortLibraryPath() {
    final String? fromEnv = Platform.environment[libraryEnvVar];
    if (fromEnv != null && fromEnv.isNotEmpty) {
      return File(fromEnv).existsSync() ? fromEnv : null;
    }
    if (!Platform.isWindows) return null;
    final String? packageRoot = _packageRoot('sherpa_onnx_windows');
    if (packageRoot == null) return null;
    final String dll = '$packageRoot/windows/onnxruntime.dll';
    return File(dll).existsSync() ? dll : null;
  }

  /// The directory pub unpacked [packageName] into, from the package
  /// config this test run was launched with.
  static String? _packageRoot(String packageName) {
    final File config = File('.dart_tool/package_config.json');
    if (!config.existsSync()) return null;
    final Object? decoded = jsonDecode(config.readAsStringSync());
    if (decoded is! Map<String, dynamic>) return null;
    final Object? packages = decoded['packages'];
    if (packages is! List<dynamic>) return null;
    for (final Object? entry in packages) {
      if (entry is! Map<String, dynamic>) continue;
      if (entry['name'] != packageName) continue;
      final Object? rootUri = entry['rootUri'];
      if (rootUri is! String) return null;
      final Uri uri = Uri.parse(rootUri);
      return uri.hasScheme
          ? uri.toFilePath()
          : config.parent.uri.resolve(rootUri).toFilePath();
    }
    return null;
  }
}
