import 'package:flutter_test/flutter_test.dart';
import 'package:hayamimi_core/punct/punct_ja_tokenizer.dart';

/// A stand-in vocabulary, so these tests do not need the model directory.
/// The real file has 7027 entries in the same shape: one token per line,
/// line number is the id.
const String testVocabText =
    '[PAD]\n'
    '[UNK]\n'
    '[CLS]\n'
    '[SEP]\n'
    'あ\n'
    'い\n'
    'ア\n'
    'ガ\n'
    'A\n'
    'B\n'
    '1\n'
    '2\n'
    '。\n'
    '𣗄\n';

void main() {
  final PunctJaVocab vocab = PunctJaVocab.parse(testVocabText);
  final PunctJaTokenizer tokenizer = PunctJaTokenizer(vocab);

  group('PunctJaVocab', () {
    test('uses the line number as the token id', () {
      expect(vocab.padId, 0);
      expect(vocab.unkId, 1);
      expect(vocab.clsId, 2);
      expect(vocab.sepId, 3);
      expect(vocab.idOf('あ'), 4);
      expect(vocab.idOf('𣗄'), 13);
      expect(vocab.length, 14);
    });

    test('maps a character it does not know to [UNK]', () {
      expect(vocab.idOf('漢'), vocab.unkId);
    });

    test('reads a file written with CRLF line endings the same way', () {
      final PunctJaVocab crlf = PunctJaVocab.parse(
        testVocabText.replaceAll('\n', '\r\n'),
      );
      expect(crlf.clsId, 2);
      expect(crlf.idOf('あ'), 4);
      expect(crlf.length, vocab.length);
    });

    test('refuses a vocabulary with no control tokens', () {
      // Their ids are wherever the file puts them, so a missing one cannot
      // be defaulted — it would silently corrupt every sequence.
      expect(
        () => PunctJaVocab.parse('あ\nい\n'),
        throwsA(isA<FormatException>()),
      );
    });
  });

  group('splitCharacters', () {
    test('folds full-width ASCII to ASCII', () {
      expect(tokenizer.splitCharacters('ＡＢ１２'), <String>['A', 'B', '1', '2']);
    });

    test('folds half-width katakana to full-width, voicing included', () {
      expect(tokenizer.splitCharacters('ｱｶﾞ'), <String>['ア', 'ガ']);
    });

    test('drops every kind of whitespace', () {
      expect(tokenizer.splitCharacters('あ い\tう\n　え'), <String>[
        'あ',
        'い',
        'う',
        'え',
      ]);
    });

    test('keeps a character outside the Basic Multilingual Plane whole', () {
      // 𣗄 is a surrogate pair in Dart. Splitting by UTF-16 code unit would
      // turn this one vocabulary entry into two unknown tokens.
      final List<String> characters = tokenizer.splitCharacters('𣗄');
      expect(characters, <String>['𣗄']);
      expect(vocab.idOf(characters.single), 13);
    });

    test('returns nothing for whitespace-only input', () {
      expect(tokenizer.splitCharacters('  \t\n'), isEmpty);
    });
  });

  group('pythonStrip', () {
    test('trims spaces and ideographic spaces', () {
      expect(PunctJaTokenizer.pythonStrip(' 　あい　 '), 'あい');
    });

    test('trims the C0 separators Python counts as whitespace', () {
      // U+001C-U+001F are whitespace to Python's str.strip() but not to
      // Dart's String.trim(), which is why this class spells the set out
      // instead of reusing trim().
      final String padded =
          '${String.fromCharCode(0x1C)}あ${String.fromCharCode(0x1F)}';
      expect(PunctJaTokenizer.pythonStrip(padded), 'あ');
      expect(padded.trim(), padded);
    });

    test('keeps a byte-order mark, as Python does', () {
      final String padded = '${String.fromCharCode(0xFEFF)}あ';
      expect(PunctJaTokenizer.pythonStrip(padded), padded);
      expect(padded.trim(), 'あ');
    });

    test('leaves inner whitespace alone', () {
      expect(PunctJaTokenizer.pythonStrip('  あ い  '), 'あ い');
    });
  });

  group('tokenize', () {
    test('wraps the characters in [CLS] and [SEP]', () {
      final PunctJaTokens tokens = tokenizer.tokenize('あい');
      expect(tokens.characters, <String>['あ', 'い']);
      expect(tokens.inputIds, <int>[2, 4, 5, 3]);
    });

    test('maps unknown characters to [UNK] without dropping them', () {
      final PunctJaTokens tokens = tokenizer.tokenize('あ漢い');
      expect(tokens.characters, <String>['あ', '漢', 'い']);
      expect(tokens.inputIds, <int>[2, 4, 1, 5, 3]);
    });

    test('cuts input longer than maxChars instead of windowing it', () {
      final PunctJaTokens tokens = tokenizer.tokenize('あ' * 10, maxChars: 4);
      expect(tokens.characters.length, 4);
      expect(tokens.inputIds, <int>[2, 4, 4, 4, 4, 3]);
    });

    test('produces just the control tokens for empty input', () {
      final PunctJaTokens tokens = tokenizer.tokenize('');
      expect(tokens.characters, isEmpty);
      expect(tokens.inputIds, <int>[2, 3]);
    });
  });
}
