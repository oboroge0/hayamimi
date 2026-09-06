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
    this.applyToFinals = true,
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

  /// Whether finalized per-segment lines are punctuated too, and not only
  /// refine ("清書") passes. Defaults to `true`.
  ///
  /// Refines were punctuated first, because that is where the desktop
  /// pipeline punctuates. But a refine does not always emit its own text: a
  /// merged re-decode that comes back much shorter than the fast finals it
  /// is replacing is discarded in favour of those finals joined together
  /// (see `isRefineTextTooShort`), and if nothing punctuated them, the
  /// refine goes out unpunctuated — losing the marks in exactly the case
  /// the guard exists to protect. The Japanese recognizer in this repo
  /// collapses multi-sentence audio to its last sentence, so on a group of
  /// two or more segments that guard is the normal outcome, not the rare
  /// one. Punctuating the finals is what gives the fallback something
  /// punctuated to fall back to — the same thing the desktop relies on,
  /// where the fast finals have already been through `punct_ja.py`.
  ///
  /// The cost is one extra run of the punctuation model per utterance,
  /// inside the decode worker, on text the next refine may replace anyway.
  /// Measured directly on an Android emulator (x86_64, API 35, host Ryzen 5
  /// 5600 — not a phone) by decoding the same audio with this flag on and
  /// off: **~40-50 ms per ~11-14 characters** (1.8-2.4 s of audio; finals
  /// went from 61.7/77.6/68.8 ms to 109.5/115.4/117.3 ms). No phone was
  /// measured. Set `false` to buy that back and
  /// keep the fast line exactly as the recognizer produced it; refines are
  /// still punctuated, and a refine that falls back to the finals then
  /// reports `punctuated: false` as it did before this flag existed.
  ///
  /// Drafts ("発話中の暫定字幕") are never punctuated either way: a draft
  /// covers part of a segment still being spoken and is re-decoded about
  /// once a second, so any mark placed in one is a guess about a sentence
  /// that has not finished.
  final bool applyToFinals;
}
