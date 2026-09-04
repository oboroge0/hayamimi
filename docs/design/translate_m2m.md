# Japanese -> Multilingual Subtitle Translation (M2M-100)

`scripts/translate_m2m.py` provides `TranslatorM2M(target_lang)`, a small
wrapper around a CTranslate2-converted M2M-100 418M multilingual model for
translating live subtitle lines from Japanese into any of the ~100 target
languages M2M-100 was trained on (Chinese `zh`, Korean `ko`, Spanish `es`,
French `fr`, ... -- see below), or English (`en`, included for
completeness/comparison against `translate_ja_en.py`'s dedicated FuguMT
module).

Wired into `--translate` in `scripts/realtime_transcribe.py` (`build_translators()`).

## Target acceptance vs. validation

Any target language whose `__<lang>__` token exists in the model's own
`shared_vocabulary.json` is **accepted** -- `translate_m2m.is_supported_target(lang)`
checks this directly against the model files (no hardcoded allowlist), so
`TranslatorM2M(target_lang)` and `--translate LANGS` work for any of
M2M-100's ~100 languages the moment this model conversion supports them.

Acceptance is not the same as **quality having been measured**. Only a
subset of targets have a chrF score against a real reference (see
"Validation tiers" below), tracked in `translate_m2m.VALIDATED_TARGETS`.
Constructing a `TranslatorM2M` for a target outside that set still works,
but prints:

```
note: 'fr' is an unvalidated translation target (quality not yet measured; see docs/TRANSLATE_M2M.md)
```

to stderr. `BEAM_SIZE_BY_TARGET` similarly only has tuned entries for zh/ko/en;
any other target falls back to `DEFAULT_BEAM_SIZE` (4, matching ko's more
cautious setting) rather than a per-language-tuned value.

## Model

- **Base model**: [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)
  (Meta AI, Fan et al., "Beyond English-Centric Multilingual Machine
  Translation"). A single multilingual model covering ~100 languages; this
  module accepts ja->any target the model's vocabulary supports (see
  "Target acceptance vs. validation" above), with zh/ko/en/es exercised and
  measured so far.
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

1. Encode source Japanese text with `sentencepiece.model` -> token pieces.
2. Prepend the source language token `__ja__` and append the end-of-sentence
   token `</s>` to the piece list -- this mirrors what the original
   `transformers` `M2M100Tokenizer` does automatically when `src_lang="ja"`
   is set and `.encode()` is called (`config.json`'s `add_source_bos` /
   `add_source_eos` are both `false`, meaning CTranslate2 does *not* add
   these automatically -- they must be added by the caller).
3. Call `translator.translate_batch(..., target_prefix=[["__<lang>__"]])`
   to force the first decoded token to be the target language token,
   selecting the output language (`__zh__`, `__ko__`, or `__en__`).
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
  caught and the original Japanese text is returned unchanged.
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

## Validation tiers

`scripts/eval_translate.py` runs an automatic quality check against
[FLEURS](https://huggingface.co/datasets/google/fleurs) (`google/fleurs`),
which is **n-way parallel**: the same integer sentence `id` refers to the same
underlying sentence across every language config in a given split. This makes
it usable as a translation reference set even though it was built for ASR
evaluation, not MT: for a candidate target language, pull `ja_jp`'s
`raw_transcription` and the target config's `raw_transcription` for matching
ids, run each Japanese sentence through `TranslatorM2M(target_lang)`, and
score the output against the FLEURS reference with
[sacrebleu](https://github.com/mjpost/sacrebleu)'s **chrF** (character
n-gram F-score -- chosen over BLEU because it doesn't depend on
whitespace-delimited word tokenization, which matters for zh/ko/ja).

### Procedure

```
python scripts/eval_translate.py --targets zh,ko,es --n 50
```

- Loads `ja_jp` and the target's FLEURS config (`validation` split) via a
  **column-projected remote parquet read** (`fsspec` + `pyarrow`, reading only
  the `id` and `raw_transcription` columns straight off the Hub) rather than
  the `datasets-server` "rows" API used by `scripts/make_realset_zhko.py`.
  This was a deliberate substitution: FLEURS parquet shards embed audio
  arrays, and some configs' shards exceed `datasets-server`'s 300MB
  per-request scan cap even for a handful of rows -- confirmed here for
  `es_419` (`Parquet error: Scan size limit exceeded: attempted to read
  307959544 bytes, limit is 300000000 bytes`) on **every** split (train/
  validation/test), not just `test` as with zh's Mandarin config in
  `make_realset_zhko.py`. Reading only the two needed columns sidesteps the
  cap entirely (a few MB transferred regardless of the file's total size,
  confirmed by direct measurement: the ~308MB `es_419` validation parquet
  loaded in ~1s) and works uniformly for every target, so no per-language
  split workaround is needed and the `datasets` library was not required.
  `pyarrow` was added to `requirements-dev.txt` for this; `fsspec` /
  `requests` / `aiohttp` were already present as transitive deps of
  `huggingface_hub`.
- Matches ids present in both configs, samples up to `--n` of them
  (default 50, seeded, deterministic), translates each ja sentence, and
  reports both per-sentence and corpus-level chrF.

### Measured results (validation split, n=50, seed=0)

| Target | chrF (corpus) | mean latency | Status |
|---|---|---|---|
| zh | 20.19 | 664 ms/line | validated (existing) |
| ko | 25.20 | 963 ms/line | validated (existing) |
| es | **42.09** | 1013 ms/line | **validated (promoted here)** |

Full command: `python scripts/eval_translate.py --targets zh,ko,es --n 50`.

### Promotion decision: es

`es` scored **higher** than both existing validated targets (42.09 vs. 20.19
zh / 25.20 ko), not merely "at the same level" -- so it clears the bar for
promotion into `VALIDATED_TARGETS` (now `{"zh", "ko", "es"}` in
`scripts/translate_m2m.py`).

Caveat on comparing chrF *across* target languages: chrF is a character
n-gram overlap metric, and its practical ceiling differs by script and
typology, independent of translation quality. Spanish uses the same Latin
alphabet M2M-100's subword vocabulary was heavily trained on and shares much
more surface-level structure with the source-adjacent English-centric
training distribution than logographic Chinese or agglutinative Korean do,
so higher absolute chrF for `es` is expected even for comparable underlying
translation quality -- it is not proof that `es` translations are "twice as
good" as `zh`'s in any human-judged sense. What the comparison *does*
establish is the thing the promotion rule cares about: `es` is not an outlier
producing garbage output relative to the model's already-shipped targets,
which is what "same level as existing validated targets" is meant to guard
against. Spot-checking the printed per-sentence output (see the sample lines
above the summary table when running the script) confirms the `es` output is
fluent, mostly faithful full-sentence translation -- consistent with the
higher chrF, not an artifact of the metric.

### Network access note

`scripts/eval_translate.py` requires network access to
`huggingface.co` (parquet files, no auth). FLEURS access worked without
issue in this environment; no alternative corpus (e.g. TED-based parallel
data) was needed.

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
