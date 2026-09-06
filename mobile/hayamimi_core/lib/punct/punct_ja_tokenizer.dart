import 'dart:convert' show LineSplitter;
import 'dart:io' show File;

import 'package:unorm_dart/unorm_dart.dart' as unorm;

/// The BERT vocabulary the punctuation model was trained with: one token
/// per line, the line number is the token id.
///
/// The model is character-level, so almost every entry is a single
/// Japanese character; the handful of bracketed entries (`[CLS]`, `[SEP]`,
/// `[UNK]`, `[PAD]`) are BERT's control tokens.
class PunctJaVocab {
  PunctJaVocab._(this._ids, this.padId, this.unkId, this.clsId, this.sepId);

  /// Reads a `vocab.txt` in the format `models/mojicast-punct-onnx` ships.
  ///
  /// Throws [FormatException] if any of the four control tokens is missing,
  /// because their ids are not fixed by the format — they are wherever the
  /// file puts them, and guessing would corrupt every sequence.
  factory PunctJaVocab.parse(String vocabText) {
    final Map<String, int> ids = <String, int>{};
    final List<String> lines = const LineSplitter().convert(vocabText);
    for (int i = 0; i < lines.length; i++) {
      ids[lines[i]] = i;
    }
    int require(String token) {
      final int? id = ids[token];
      if (id == null) {
        throw FormatException('vocab.txt has no "$token" entry');
      }
      return id;
    }

    return PunctJaVocab._(
      ids,
      require('[PAD]'),
      require('[UNK]'),
      require('[CLS]'),
      require('[SEP]'),
    );
  }

  /// Reads the vocabulary from a file, decoded as UTF-8.
  static Future<PunctJaVocab> loadFile(String path) async {
    return PunctJaVocab.parse(await File(path).readAsString());
  }

  final Map<String, int> _ids;

  final int padId;
  final int unkId;
  final int clsId;
  final int sepId;

  /// How many tokens the vocabulary holds.
  int get length => _ids.length;

  /// The id for [character], or [unkId] when the vocabulary has no entry.
  int idOf(String character) => _ids[character] ?? unkId;
}

/// A tokenized sentence: the characters the model sees, and the ids fed to
/// it.
class PunctJaTokens {
  const PunctJaTokens(this.characters, this.inputIds);

  /// One entry per character, in output order. [PunctuatorJa] walks this
  /// list to decide where a mark goes, so it has to stay aligned with
  /// [inputIds] minus the two control tokens.
  final List<String> characters;

  /// `[CLS]` + one id per character + `[SEP]`, i.e. what the model's
  /// `input_ids` tensor gets.
  final List<int> inputIds;
}

/// Turns Japanese text into the model's character tokens.
///
/// The reference implementation (`scripts/punct_ja.py`) reproduces Hugging
/// Face's `BertJapaneseTokenizer`: NFKC-normalize, cut the text into
/// morphemes with MeCab, then split each morpheme into characters. Dart has
/// no MeCab, and pulling a morphological analyzer into a mobile package to
/// then throw the word boundaries away would be a large dependency for no
/// effect.
///
/// So this class implements the rule that pipeline actually reduces to:
/// **NFKC-normalize, drop whitespace, split into code points**. MeCab's
/// only observable effect here is that it swallows whitespace between
/// morphemes — the concatenation of its surfaces is otherwise the input
/// unchanged, and the word boundaries are discarded a line later anyway.
///
/// That equivalence is not assumed, it is checked:
/// `scripts/make_punct_fixture.py` compares this rule against the real
/// MeCab pipeline over every FLEURS ja sentence (punctuated and stripped)
/// plus the fixture's synthetic cases, and refuses to write the fixture if
/// they ever disagree. The recorded token ids are then replayed by
/// `test/punct/punct_ja_parity_test.dart`.
class PunctJaTokenizer {
  const PunctJaTokenizer(this.vocab);

  final PunctJaVocab vocab;

  /// Characters Python's `str.isspace()` is true for, which is the set
  /// MeCab drops and the set `str.strip()` trims. Dart's own `trim()` is a
  /// slightly different set (it trims U+FEFF, and leaves U+001C-U+001F),
  /// so the Python set is spelled out here rather than borrowed.
  static bool isPythonSpace(int codePoint) {
    return (codePoint >= 0x09 && codePoint <= 0x0D) ||
        (codePoint >= 0x1C && codePoint <= 0x20) ||
        codePoint == 0x85 ||
        codePoint == 0xA0 ||
        codePoint == 0x1680 ||
        (codePoint >= 0x2000 && codePoint <= 0x200A) ||
        codePoint == 0x2028 ||
        codePoint == 0x2029 ||
        codePoint == 0x202F ||
        codePoint == 0x205F ||
        codePoint == 0x3000;
  }

  /// Trims leading and trailing whitespace exactly as Python's
  /// `str.strip()` does.
  static String pythonStrip(String text) {
    final List<int> runes = text.runes.toList();
    int start = 0;
    int end = runes.length;
    while (start < end && isPythonSpace(runes[start])) {
      start++;
    }
    while (end > start && isPythonSpace(runes[end - 1])) {
      end--;
    }
    if (start == 0 && end == runes.length) return text;
    return String.fromCharCodes(runes.sublist(start, end));
  }

  /// NFKC-normalizes [text], drops whitespace, and returns one string per
  /// remaining code point.
  ///
  /// Code points, not UTF-16 units: the vocabulary contains characters
  /// outside the Basic Multilingual Plane (𣗄, 𥝱), and splitting those
  /// into surrogate halves would turn one token into two unknowns.
  List<String> splitCharacters(String text) {
    final String normalized = unorm.nfkc(text);
    final List<String> characters = <String>[];
    for (final int rune in normalized.runes) {
      if (isPythonSpace(rune)) continue;
      characters.add(String.fromCharCode(rune));
    }
    return characters;
  }

  /// Tokenizes [text], keeping at most [maxChars] characters.
  ///
  /// Longer input is cut, not windowed — the model's position embeddings
  /// stop at 512 and the reference implementation truncates the same way,
  /// so anything past the limit is dropped from the result rather than
  /// restored unpunctuated.
  PunctJaTokens tokenize(String text, {int maxChars = 500}) {
    List<String> characters = splitCharacters(text);
    if (characters.length > maxChars) {
      characters = characters.sublist(0, maxChars);
    }
    final List<int> inputIds = <int>[
      vocab.clsId,
      for (final String character in characters) vocab.idOf(character),
      vocab.sepId,
    ];
    return PunctJaTokens(characters, inputIds);
  }
}
