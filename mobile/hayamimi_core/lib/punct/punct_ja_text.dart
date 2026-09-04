import 'dart:math' as math;
import 'dart:typed_data';

/// Characters that already act as punctuation, so the restorer neither
/// writes a mark after one nor writes a mark immediately before one.
///
/// The list is the reference implementation's `_JA_PUNCT_CHARS`
/// (`scripts/punct_ja.py`) character for character, including the ASCII
/// half-widths — NFKC folds `！？（）` down to `!?()` before this set is
/// consulted, so both forms have to be in it.
const Set<String> jaPunctuationCharacters = <String>{
  '。',
  '、',
  '！',
  '？',
  '!',
  '?',
  '…',
  '「',
  '」',
  '『',
  '』',
  '（',
  '）',
  '(',
  ')',
  '【',
  '】',
  '・',
  ',',
  '.',
  '\n',
};

/// Sentence endings that mark a question.
///
/// The model has only two output classes, comma and period — there is no
/// question-mark class to predict. So `？` is decided by this list of
/// endings instead, which is a heuristic and is wrong in both directions:
/// it misses a question asked by intonation alone, and it fires on a
/// sentence that merely ends in the nominalizer `の`. It is carried over
/// unchanged from the desktop pipeline so both produce the same text; see
/// `docs/design/punct_ja.md`, "Known limitations".
const List<String> jaQuestionSuffixes = <String>[
  'ですか',
  'ますか',
  'でしょうか',
  'かな',
  'かしら',
  'かい',
  'の',
  'だろうか',
  'でしたか',
  'ましたか',
];

/// The only marks the restorer ever writes: [insertMarks] writes 、 and 。,
/// and [applyQuestionMarks] rewrites some of those 。 as ？. Everything else
/// in [jaPunctuationCharacters] is a mark it merely recognizes in its input.
const Set<String> restoredMarkCharacters = <String>{'、', '。', '？'};

final RegExp _restoredMarks = RegExp('[${restoredMarkCharacters.join()}]');

/// Removes the marks the restorer writes, giving back a string that can be
/// compared, by length, against text that was never punctuated.
///
/// Restoring punctuation makes text longer, and `isRefineTextTooShort`
/// (`live/refine_pass.dart`) decides whether a refine pass lost content by
/// comparing its length against the unpunctuated fast finals it replaces.
/// Comparing a punctuated refine directly would credit it for characters
/// nobody said — one mark per sentence or clause is enough to push a
/// genuinely truncated result back over the threshold — so the caller
/// strips them first. NFKC normalization, which the restorer also applies,
/// is not undone here: it folds character *forms*, so it leaves the count
/// alone except where it composes a two-character half-width katakana into
/// one, which no recognizer in this package emits.
String withoutRestoredMarks(String text) => text.replaceAll(_restoredMarks, '');

/// Converts a logit to a probability, the same way the reference
/// implementation's `1 / (1 + exp(-x))` does.
double sigmoid(double logit) => 1.0 / (1.0 + math.exp(-logit));

/// Rebuilds the sentence with 、 and 。 inserted where the model says.
///
/// [characters] is one entry per input character, and [logits] holds the
/// model's raw output for the whole sequence including the two control
/// tokens: `logits[(i + 1) * 2]` is the comma logit for `characters[i]`
/// and `logits[(i + 1) * 2 + 1]` the period logit, because index 0 belongs
/// to `[CLS]`.
///
/// A period wins over a comma when both clear their threshold. When
/// [forceFinalPeriod] is false, a period predicted on the very last
/// character is skipped — and, matching the reference implementation, the
/// comma is then considered in its place.
String insertMarks({
  required List<String> characters,
  required Float32List logits,
  double commaThreshold = 0.5,
  double periodThreshold = 0.5,
  bool forceFinalPeriod = true,
}) {
  final int expected = (characters.length + 2) * 2;
  if (logits.length != expected) {
    throw ArgumentError(
      'logits has ${logits.length} values but ${characters.length} '
      'characters need $expected ([CLS] + characters + [SEP], two per '
      'position)',
    );
  }
  final StringBuffer out = StringBuffer();
  for (int i = 0; i < characters.length; i++) {
    final String character = characters[i];
    out.write(character);
    final bool isLast = i == characters.length - 1;
    if (jaPunctuationCharacters.contains(character)) continue;
    if (!isLast && jaPunctuationCharacters.contains(characters[i + 1])) {
      // A mark is already coming; a second one would double up.
      continue;
    }
    final double comma = sigmoid(logits[(i + 1) * 2]);
    final double period = sigmoid(logits[(i + 1) * 2 + 1]);
    if (period >= periodThreshold && (!isLast || forceFinalPeriod)) {
      out.write('。');
    } else if (comma >= commaThreshold) {
      out.write('、');
    }
  }

  String result = out.toString();
  if (forceFinalPeriod && result.isNotEmpty) {
    final String last = String.fromCharCode(result.runes.last);
    if (!jaPunctuationCharacters.contains(last)) {
      result += '。';
    }
  }
  return result;
}

/// Rewrites a sentence-final 。 as ？ when the sentence ends in one of
/// [jaQuestionSuffixes].
///
/// This works on the finished string rather than on the model output, so
/// it is a straight port of the reference implementation, quirks included:
/// every segment between two 。 gets a mark appended, which means a text
/// that did not end in 。 gains one here.
String applyQuestionMarks(String text) {
  final StringBuffer out = StringBuffer();
  for (final String segment in text.split('。')) {
    if (segment.isEmpty) continue;
    final bool isQuestion = jaQuestionSuffixes.any(segment.endsWith);
    out.write(segment);
    out.write(isQuestion ? '？' : '。');
  }
  return out.toString();
}
