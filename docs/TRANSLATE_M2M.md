# Japanese -> Chinese / Korean and English -> Japanese Translation (M2M-100)

`scripts/translate_m2m.py` provides `TranslatorM2M(target_lang, source_lang)`, a small
wrapper around a CTranslate2-converted M2M-100 418M multilingual model for
translating live subtitle lines from Japanese into Chinese (`zh`), Korean
(`ko`), or English (`en`, included for completeness/comparison against
`translate_ja_en.py`'s dedicated FuguMT module), and from English (`en`) to
Japanese (`ja`).

The live pipeline uses this model for `--translate ja`, `--translate zh`, and
`--translate ko`. Translation runs only for matching finalized source lines;
partial ASR output remains in the recognized language.

## Model

- **Base model**: [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)
  (Meta AI, Fan et al., "Beyond English-Centric Multilingual Machine
  Translation"). A single multilingual model covering ~100 languages, used
  here only for ja->{zh,ko,en} and en->ja.
- **CTranslate2 conversion used**:
  [`ishiki-emo/mojicast-m2m100-ct2`](https://huggingface.co/ishiki-emo/mojicast-m2m100-ct2) --
  the conversion published by the Mojicast project. Confirmed to exist and
  downloaded directly (no local re-conversion with
  `ct2-transformers-converter` was needed). It bundles:
  - `model.bin` / `config.json` / `shared_vocabulary.json` -- CTranslate2
    model, int8 quantized, ~491 MB on disk (down from the original model's
    ~1.9 GB).
  - `sentencepiece.model` -- the original M2M-100 SentencePiece tokenizer
    (`sentencepiece.bpe.model` in the source repo, renamed).
- Downloaded into `models/mojicast-m2m100-ct2/` via
  `huggingface_hub.snapshot_download`.

## License

**MIT.** Per the Mojicast conversion's `README.md`/model card, this follows
the original `facebook/m2m100_418M` license (MIT). Unlike the ja-en FuguMT
module (CC BY-SA 4.0, share-alike), this model can be redistributed under
permissive terms -- still worth keeping attribution to Meta AI (original
model) and the Mojicast project (this CTranslate2 conversion) in any
redistribution.

## Tokenization and language tokens

M2M-100 is a multilingual encoder-decoder model, so CTranslate2 needs both
SentencePiece tokenization *and* explicit source/target language tokens
(this differs from the ja-en FuguMT module, which is a single-language-pair
model with no language tokens):

1. Encode source text with `sentencepiece.model` -> token pieces.
2. Prepend the source language token (`__ja__` or `__en__`) and append the end-of-sentence
   token `</s>` to the piece list -- this mirrors what the original
   `transformers` `M2M100Tokenizer` does automatically when `src_lang="ja"`
   is set and `.encode()` is called (`config.json`'s `add_source_bos` /
   `add_source_eos` are both `false`, meaning CTranslate2 does *not* add
   these automatically -- they must be added by the caller).
3. Call `translator.translate_batch(..., target_prefix=[["__<lang>__"]])`
   to force the first decoded token to be the target language token,
   selecting the output language (`__ja__`, `__zh__`, `__ko__`, or `__en__`).
4. The returned hypothesis starts with that same target-language token --
   it is stripped (along with a trailing `</s>` if present) before
   detokenizing with `sentencepiece.model`.

This is the standard CTranslate2 usage pattern for M2M-100 (see
[CTranslate2's Transformers guide, "M2M-100" section](https://opennmt.net/CTranslate2/guides/transformers.html#m2m-100)),
adapted to use `sentencepiece` directly instead of a `transformers`
tokenizer (this project doesn't depend on `transformers`).

The `__ja__` / `__zh__` / `__ko__` / `__en__` tokens are **not** part of the
SentencePiece model's own piece vocabulary (`sp.piece_to_id("__ja__")`
returns the `<unk>` id, 0) -- they only exist in CTranslate2's
`shared_vocabulary.json` (128,112 entries vs. the SentencePiece model's own
128,000). This is expected: CTranslate2's `Translator` matches token
*strings* against `shared_vocabulary.json` directly, independent of the
SentencePiece model's internal ids. Passing these strings straight into
`translate_batch` (as pure strings, not sentencepiece-encoded) is correct;
they must be excluded before calling `sp.decode()`, since sentencepiece
doesn't recognize them as pieces and would otherwise echo them back
literally as text (confirmed by testing `sp.decode(["__zh__"])` ->
`"__zh__"`).

## Beam size and repetition control

Mojicast's reported configuration (per the task brief) uses **greedy
decoding (beam_size=1) for zh** and **beam_size=4 for ko**, with
`no_repeat_ngram_size=3` to suppress repetition loops.

Re-measured against this specific model conversion on this machine (CPU,
`compute_type="int8"`), using both short lines and a filler-heavy casual
line designed to be repetition-prone
(`"いや、そのー、なんていうか、うーん、ちょっと難しいんだけど、あの、まあ、そうだね、頑張ってみるよ。"`)
plus a shorter repetition-prone line (`"そうそう、そうなんだよね。"`):

- **`no_repeat_ngram_size=3` does not fully eliminate repetition on this
  conversion**, unlike Mojicast's report for their setup. On the filler-heavy
  line, zh beam_size=4 still produced runs like `"是啊,是啊!是啊?是啊......是啊...是啊啊!"`.
  However, it **reliably prevents the catastrophic degenerate loops** seen
  at `no_repeat_ngram_size=0`:
  - zh, beam=1, n=0: 50+ repeated `"是的,"` tokens, ~2.97s (ran to the
    length cap).
  - ko, beam=1, n=0: 100+ repeated Korean jamo characters (`"ᄏᄏᄏᄏᄏ..."`),
    ~2.62s.
  - ko, beam=4, n=0: pure single-character repetition (`"ᄒᄒᄒᄒᄒ..."`).
  - With `n=3`, the same lines produced bounded, largely sane output in
    0.3-0.8s -- e.g. ko beam=4 n=3 on the filler line:
    `"아니, 아니, 뭔가, 괜찮아, 조금 어렵지만, 오, 그렇습니다, 나는 그것을 시도 할 것이다."`
    (a genuinely reasonable translation).
  - Kept `n=3` (matching the reference/Mojicast setting) rather than
    tightening further to `n=1` or `n=2`: spot checks did not show a
    consistent quality improvement from a stricter value, and `n=3` is the
    documented reference setting -- this is a smaller deviation than the
    ja-en FuguMT module needed (`n=1` there).
- **`beam_size`**: kept at Mojicast's reported split (zh greedy / ko beam=4)
  rather than re-tuning per-language beam size independently. In the
  measurements above, zh at beam=4 was not meaningfully better than beam=1
  and cost noticeably more latency (312ms vs. 262ms greedy on the short
  repetition-prone line; 744ms vs. 605ms on the long line); ko at beam=1
  produced worse degenerate output on the filler line
  (`"ᄏᄏᄏᄏᄏᄏᄏ..."`) than ko at beam=4 with `n=3`, confirming the
  higher beam size is worth its cost for Korean specifically.
- **Decode length cap**: `max_decoding_length` is capped relative to the
  source token count (`min(150, max(30, len(source_tokens) * 6 + 20))`),
  same formula as `translate_ja_en.py`. This is what turns the
  ~2.6-3.0s degenerate-loop cases above into a bounded stall rather than an
  unbounded one; combined with `no_repeat_ngram_size=3` the worst case seen
  here was well under 1s.

## Fallback behavior

`TranslatorM2M.translate()` never raises and never returns an empty string
for non-empty input:

- Empty / whitespace-only input is returned unchanged.
- Any exception during tokenization, translation, or detokenization is
  caught and the original source text is returned unchanged.
- Empty tokenization, an empty hypothesis, or empty/whitespace-only
  translation output falls back to the original text.

This matches `translate_ja_en.py`'s design goal: a subtitle line should
never go blank because of a translation failure.

## Measured latency (smoke test, `python scripts/translate_m2m.py`)

6 lines (business, casual, a scheduling line with numbers, a budget line
with numbers, a question, and a repetition-prone casual line), CPU,
`compute_type="int8"`, `no_repeat_ngram_size=3`, each target's model loaded
once (`~0.5s` load time per target), per-line `time.perf_counter()` around
`translate()`:

| Input | zh (beam=1) | zh output | ko (beam=4) | ko output |
|---|---|---|---|---|
| 本日はお集まりいただきありがとうございます。 | 220 ms | 谢谢你今天聚集。 | 271 ms | 오늘 모여 주셔서 감사합니다. |
| 今日はマジで疲れたわー。 | 181 ms | 今天我真的很累。 | 267 ms | 오늘은 정말 피곤해요. |
| 会議は午後3時から始まります。 | 223 ms | 会议将于下午3点开始。 | 294 ms | 회의는 오후 3시부터 시작된다. |
| このプロジェクトの予算は500万円です。 | 284 ms | 该项目的预算为5万英镑。 | 347 ms | 이 프로젝트의 예산은 5억원이다. |
| 明日は雨が降ると思いますか? | 234 ms | 明天会下雨吗? | 375 ms | 내일 비가 내릴 것이라고 생각하십니까? |
| そうそう、そうなんだよね。 | 201 ms | 是的,是的。 | 299 ms | 그렇다, 그렇다 그것은 그렇다. |

**zh mean: 224 ms/line** (min 181 ms, max 284 ms).
**ko mean: 309 ms/line** (min 267 ms, max 375 ms).

This is well above Mojicast's reported 115 ms (zh) / 167 ms (ko) per line --
the same pattern seen in `docs/TRANSLATE.md` for the ja-en module (203 ms
here vs. their 19 ms). Likely the same causes: different CPU/thread
configuration, no per-call CTranslate2 tuning (`inter_threads`/
`intra_threads`) applied yet, and this machine simply being slower per-call
than Mojicast's reference hardware. Worth revisiting with `inter_threads`
tuning when this is wired into the realtime pipeline.

### English -> Japanese on Apple Silicon

The `en->ja` route uses `beam_size=4` and the same repetition/length guards.
On an Apple M2 Max with Python 3.11 and CTranslate2 4.8.1, the model loaded in
about 209 ms. Three direct translations took 265-382 ms each:

| English input | Latency | Japanese output |
|---|---:|---|
| Hello, thank you for joining us today. | 382 ms | こんにちは、今日私たちに加わってくれてありがとう。 |
| The meeting starts at three in the afternoon. | 265 ms | 会議は午後3時から始まります。 |
| This translation runs completely on this Mac. | 278 ms | この翻訳はこのMacで完全に実行されます。 |

A 5.5-second locally synthesized English WAV also completed the live pipeline
on the same machine: Parakeet produced the English final, and `--translate ja`
emitted the corresponding Japanese line without any cloud API.

## Known limitations

- **Repetition is reduced, not eliminated**, as described above --
  `no_repeat_ngram_size=3` prevents catastrophic loops but casual/filler-
  heavy Japanese can still produce mildly repetitive output
  (`"是啊,是啊!是啊?是啊......"`-style runs). This is a real quality
  ceiling of the 418M model at this beam size, not purely a
  decoding-parameter issue.
- **Numbers are not reliably preserved.** `500万円` (5,000,000 yen) was
  translated as `5万英镑` (50,000 British pounds -- wrong currency *and*
  wrong magnitude) in zh, and `5억원` (500,000,000 Korean won -- 100x off)
  in ko. This is a bigger problem than the ja-en FuguMT module showed on its
  smoke test and should be treated as an open risk for any use case where
  numeric accuracy matters (e.g. financial or scheduling subtitles).
- **OpenCC conversion for zh regional notation (Simplified vs. Traditional,
  or Mainland vs. Taiwan/Hong Kong wording) is out of scope for this
  module.** Output is whatever the base model produces (observed as
  Simplified Chinese in the smoke test) with no post-processing.
- **English target (`en`) is included for API completeness / as a point of
  comparison against the dedicated `translate_ja_en.py` (FuguMT) module,
  but was not part of the smoke test or the repetition/beam-size
  measurements above.** If `en` output is needed, prefer
  `translate_ja_en.py` for quality unless a specific reason favors this
  module (e.g. wanting a single loaded model across all three targets).
- **MIT license** applies to both the base model and this CTranslate2
  conversion -- see License section above.
- The int8 `compute_type` is applied at load time via CTranslate2; no
  separate quality comparison against float32/float16 was done in this
  pass.
