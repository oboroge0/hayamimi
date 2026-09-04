import 'dart:typed_data';

import 'punct_ja_text.dart';
import 'punct_ja_tokenizer.dart';
import 'punct_ort_session.dart';

/// Puts Japanese punctuation back into unpunctuated text.
///
/// The problem: hayamimi's speech recognizer emits a bare run of
/// characters. On the desktop pipeline `scripts/punct_ja.py` inserts
/// 、 and 。 into it before the text is shown, so desktop captions read as
/// sentences; the mobile package had no equivalent, so its refined output
/// arrived unpunctuated. This class is the mobile half of that.
///
/// How it works: a character-level BERT classifier decides, for every
/// character, whether a comma or a period should follow it. The model is
/// the one `docs/design/punct_ja.md` describes, converted to float16 by
/// `scripts/quantize_punct.py --variant fp16` (182 MB, half of the
/// original, and identical predictions on the 250-sentence FLEURS check
/// recorded in `docs/design/mobile_quantization.md`). A rule-based pass then turns a
/// sentence-final 。 into ？ after a question ending, because the model has
/// no question-mark class to predict.
///
/// It runs the model through ONNX Runtime — the copy `sherpa_onnx` has
/// already loaded, reached over `dart:ffi`, so the app does not end up with
/// two ONNX Runtimes in one process. See [PunctOrtSession] and
/// `OrtLibrary`.
///
/// ```dart
/// final punctuator = await PunctuatorJa.load(
///   modelPath: '.../punct_bert.fp16.onnx',
///   vocabPath: '.../vocab.txt',
/// );
/// punctuator.restore('今日めっちゃ疲れたわもう寝る');
/// // -> 今日めっちゃ疲れたわ。もう寝る。
/// punctuator.dispose();
/// ```
///
/// One instance holds one ONNX Runtime session and is not safe to use from
/// more than one isolate. Call [dispose] when done: the session and the
/// model's memory are native, so nothing frees them on garbage collection.
class PunctuatorJa {
  PunctuatorJa._(
    this._session,
    this._tokenizer, {
    required this.commaThreshold,
    required this.periodThreshold,
    required this.maxChars,
    required this.forceFinalPeriod,
  });

  /// Loads the model at [modelPath] and the vocabulary at [vocabPath].
  ///
  /// [libraryPath] chooses where ONNX Runtime comes from; leave it unset on
  /// Android, pass a path on desktop and in tests. See `OrtLibrary.open`.
  ///
  /// [commaThreshold] and [periodThreshold] are the probabilities above
  /// which a mark is written. [maxChars] cuts longer input — see
  /// [PunctJaTokenizer.tokenize]. [forceFinalPeriod] appends a 。 when the
  /// restored text would otherwise end without punctuation. The defaults
  /// are the desktop pipeline's, and changing them makes the two disagree.
  static Future<PunctuatorJa> load({
    required String modelPath,
    required String vocabPath,
    String? libraryPath,
    int intraOpNumThreads = 2,
    double commaThreshold = 0.5,
    double periodThreshold = 0.5,
    int maxChars = 500,
    bool forceFinalPeriod = true,
  }) async {
    final PunctJaVocab vocab = await PunctJaVocab.loadFile(vocabPath);
    final PunctOrtSession session = PunctOrtSession.open(
      modelPath,
      libraryPath: libraryPath,
      intraOpNumThreads: intraOpNumThreads,
    );
    return PunctuatorJa._(
      session,
      PunctJaTokenizer(vocab),
      commaThreshold: commaThreshold,
      periodThreshold: periodThreshold,
      maxChars: maxChars,
      forceFinalPeriod: forceFinalPeriod,
    );
  }

  final PunctOrtSession _session;
  final PunctJaTokenizer _tokenizer;

  final double commaThreshold;
  final double periodThreshold;
  final int maxChars;
  final bool forceFinalPeriod;

  /// The ONNX Runtime version that is actually running, e.g. `1.27.1`.
  String get runtimeVersion => _session.runtimeVersion;

  /// Returns [text] with 、 and 。 (and ？ where a question ending is
  /// recognised) inserted.
  ///
  /// Non-punctuation characters are never changed, but the text does come
  /// back NFKC-normalized (full-width ASCII folded to ASCII, half-width
  /// katakana folded to full-width) because that is what the model is fed.
  /// Empty or whitespace-only input comes back empty.
  String restore(String text) {
    final String stripped = PunctJaTokenizer.pythonStrip(text);
    if (stripped.isEmpty) return stripped;

    final PunctJaTokens tokens = _tokenizer.tokenize(
      stripped,
      maxChars: maxChars,
    );
    if (tokens.characters.isEmpty) return stripped;

    final Float32List logits = _session.run(tokens.inputIds);
    final String marked = insertMarks(
      characters: tokens.characters,
      logits: logits,
      commaThreshold: commaThreshold,
      periodThreshold: periodThreshold,
      forceFinalPeriod: forceFinalPeriod,
    );
    return applyQuestionMarks(marked);
  }

  /// Releases the ONNX Runtime session. Safe to call more than once.
  void dispose() => _session.dispose();
}
