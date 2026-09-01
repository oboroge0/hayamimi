/// How intra-op threads ONNX Runtime is told to use for the punctuation
/// model when a caller does not say.
///
/// Two, matching the recognizers' own default (`DecodeWorkerConfig.numThreads`):
/// the punctuation model runs in the same isolate as the recognizers and on
/// the same phone cores, so asking for more threads here takes them from
/// the decode that is about to follow rather than finding idle ones.
const int defaultPunctNumThreads = 2;

/// Where a session's Japanese punctuation model lives, and how to run it.
///
/// The problem: hayamimi's speech recognizer emits a bare run of
/// characters. The desktop pipeline puts 、 and 。 back with
/// `scripts/punct_ja.py` before anything sees the text, so desktop captions
/// read as sentences; this package's refine ("清書") output arrived
/// unpunctuated. Passing one of these to `LiveTranscriber.start` (or
/// `HayamimiLive.start`) is what closes that gap — `null`, the default,
/// leaves the session behaving exactly as it did before this parameter
/// existed.
///
/// This is a description of where the files are, not the model itself. The
/// session hands it to its decode worker isolate, which is where the model
/// is loaded and where `PunctuatorJa.restore` actually runs; nothing native
/// is created on the caller's isolate. Immutable, so the same instance can
/// be reused across sessions.
///
/// ```dart
/// await live.start(
///   modelDir: modelDir,
///   vadModelPath: vadPath,
///   punctuation: JaPunctuation(
///     modelPath: '$dir/punct_bert.fp16.onnx',
///     vocabPath: '$dir/vocab.txt',
///   ),
/// );
/// ```
///
/// Both files are local build artifacts today — see the README's "Japanese
/// punctuation restoration" section for where they come from and what is
/// still unverified.
class JaPunctuation {
  const JaPunctuation({
    required this.modelPath,
    required this.vocabPath,
    this.libraryPath,
    this.numThreads = defaultPunctNumThreads,
  });

  /// The punctuation model file, e.g. `punct_bert.fp16.onnx` (181.8 MB).
  final String modelPath;

  /// The model's `vocab.txt` (28 KB), which maps characters to token ids.
  final String vocabPath;

  /// Where ONNX Runtime — the inference engine that runs the model — comes
  /// from. Leave unset on Android, where the session borrows the copy
  /// `sherpa_onnx` has already loaded; pass a path on a desktop host,
  /// where nothing puts that library on the loader search path. `OrtLibrary`
  /// documents the per-platform default this `null` selects, including why
  /// iOS refuses rather than guesses.
  final String? libraryPath;

  /// How many threads ONNX Runtime may use inside a single operator.
  /// Defaults to [defaultPunctNumThreads].
  final int numThreads;
}
