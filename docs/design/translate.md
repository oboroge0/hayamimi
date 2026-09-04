# Japanese -> English Subtitle Translation

`scripts/translate_ja_en.py` provides `TranslatorJaEn`, a small wrapper around a
CTranslate2-converted FuguMT ja->en model for translating live subtitle lines.
Not yet wired into `asr_engine.py` / `realtime_transcribe.py` / `subtitle_server.py`
-- integration is a separate step.

## Model

- **Base model**: [`staka/fugumt-ja-en`](https://huggingface.co/staka/fugumt-ja-en)
  ("Fugu Machine Translator" by staka), a MarianMT ja->en model.
- **CTranslate2 conversion used**:
  [`ishiki-emo/mojicast-fugumt-ja-en-ct2`](https://huggingface.co/ishiki-emo/mojicast-fugumt-ja-en-ct2) --
  the conversion published by the Mojicast project (same author/org as the
  `ishiki-emo/mojicast-*` models already in `models/`). It bundles:
  - `model.bin` / `config.json` / `shared_vocabulary.json` -- CTranslate2 model, ~122 MB on disk (float16 weights, auto-upcast to float32 by CTranslate2 on this CPU; loaded here with `compute_type="int8"` for on-the-fly int8 quantization at runtime).
  - `source.spm` / `target.spm` -- the original SentencePiece tokenizers (same as `staka/fugumt-ja-en`), used separately for source encoding and target decoding.
- Downloaded into `models/mojicast-fugumt-ja-en-ct2/` via `huggingface_hub.snapshot_download`. Not re-converted locally -- the prebuilt repo was found on the first lookup under `ishiki-emo`, so `ct2-transformers-converter`/`transformers` were not needed.

## License

**CC BY-SA 4.0.** This is inherited from the original `staka/fugumt-ja-en`
model and applies to the CTranslate2 conversion as well. This means:

- Attribution to the original author (staka, "Fugu Machine Translator" /
  [Fugu-Machine Translator project](https://staka.jp/wordpress/?p=413)) and to
  the Mojicast conversion (`ishiki-emo/mojicast-fugumt-ja-en-ct2`) must be
  kept if this model or its outputs/weights are redistributed.
- Any redistribution of the model itself (not just its translations) must
  remain under CC BY-SA 4.0 (share-alike).
- This is **not** a permissive license like MIT/Apache -- do not bundle the
  model weights into a closed-source distribution without honoring
  attribution + share-alike.

## Tokenization

FuguMT is a Marian-family MT model, so CTranslate2 needs SentencePiece
tokenization done outside the `Translator`:

1. Encode source Japanese text with `source.spm` -> token pieces.
2. Feed pieces into `ctranslate2.Translator.translate_batch`.
3. Decode the output token pieces with `target.spm` -> detokenized English string.

`source.spm` and `target.spm` are distinct SentencePiece models even though
`shared_vocabulary.json` is a single shared vocabulary file.

## Beam size and repetition control

Mojicast's `TRANSLATION_REPORT.md` (per the task brief) found `beam_size=8`
cost only +4.7 ms over greedy on their hardware, and that
`no_repeat_ngram_size=3` eliminated repetition loops entirely.

Re-measuring against this specific model file on this machine (Ryzen 5 5600,
CPU, `compute_type="int8"`):

- **`no_repeat_ngram_size=3` did not eliminate repetition loops here.** With
  beam_size=5-8 it still regularly produced runs like
  `"Thank thank thank thank you so much thank you for..."` and, on harder
  sentences, multi-hundred-token degenerate loops that ran all the way to
  `max_decoding_length` (multi-second stalls). `no_repeat_ngram_size=1`
  (block any repeated single token, not just repeated 3-grams) was required
  to meaningfully suppress these loops. This module uses **`n=1`**, which is
  stricter than the reference report's `n=3` -- documented here as a
  deliberate deviation, not an oversight.
- **`beam_size`**: greedy (`beam_size=1`) was fastest (~60 ms/line steady
  state) but produced *worse* quality on several lines (longer degenerate
  tails) than beam search. `beam_size=5` gave a better quality/latency
  balance in manual comparison against 3 and 8 on the same 8-line sample and
  is what this module uses. `beam_size=8` was not consistently faster or
  slower than 5 on this hardware (noisy), so 5 was kept as the simpler
  choice.
- **Decode length cap**: `max_decoding_length` is capped relative to the
  source token count (`min(150, max(30, len(source_tokens) * 6 + 20))`)
  rather than left at CTranslate2's default of 256. Without this cap, a
  degenerate hypothesis can run to the full 256-token limit, turning one bad
  input line into a 2-4 second stall -- unacceptable for a live subtitle
  pipeline where a fallback to the source text is strictly better than a
  multi-second freeze.

## Fallback behavior

`TranslatorJaEn.translate()` never raises and never returns an empty string
for non-empty input:

- Empty / whitespace-only input is returned unchanged.
- Any exception during tokenization, translation, or detokenization is
  caught and the original Japanese text is returned unchanged.
- Empty tokenization or empty/whitespace-only translation output falls back
  to the original text.

This matches the design goal: a subtitle line should never go blank because
of a translation failure.

## Measured latency (smoke test, `python scripts/translate_ja_en.py`)

8 lines (business, casual, numbers, a scheduling line, a question, a long
line, and one line already in English), CPU, `compute_type="int8"`,
`beam_size=5`, `no_repeat_ngram_size=1`, model loaded once, per-line
`time.perf_counter()` around `translate()`:

| Input | Latency | Output |
|---|---|---|
| 本日はお集まりいただきありがとうございます。 | 254 ms | Thank thank you very kind thanks for your gathering today. |
| 今日はマジで疲れたわー。 | 76 ms | I'm really tired today. |
| 会議は午後3時から始まります。 | 248 ms | The meeting will start at 3:00 pm. |
| このプロジェクトの予算は500万円です。 | 102 ms | The budget for this project is five million yen. |
| 明日は雨が降ると思いますか? | 223 ms | Do you think it's going to rain tomorrow, when we expect the snow will fall in your day?do do thinking there'll be a shower t expected if I have any idea that this is scheduled for next-mor cloudforwe should see Rainfall. |
| 先週末、家族と一緒に近くの山に登って、久しぶりに自然の中でゆっくりとした時間を過ごすことができました。 | 392 ms | Last weekend, I was able to climb a nearby mountain with my family this past week and have the first time in years of nature so we had some leisurely relaxing moments. |
| ありがとうございます、それでは次のスライドに移ります。 | 282 ms | Thank thank you very much thanks, and then we will move to the next slide. Now let's go down for a second sliding! |
| This is already in English, so what happens? | 47 ms | This is this Is already in English, so what way? |

**Mean: ~203 ms/line, min 47 ms, max 392 ms** (model load: 0.28 s, one-time).

This is well above Mojicast's reported 19 ms/line. Likely causes: different
CPU/thread config, the deliberately harder/longer test sentences used above
vs. real short broadcast fragments, and no per-call CTranslate2 tuning
(`inter_threads`/`intra_threads`) applied yet. Short, simple lines in this
same run (e.g. "今日はマジで疲れたわー。" at 76 ms, "This is already in
English..." at 47 ms) are much closer to real-time budgets. The target of
"<100 ms/line" is met for short/simple lines but not for long or
structurally difficult ones -- worth revisiting with `inter_threads` tuning
or a smaller beam when this is wired into the realtime pipeline.

## Known limitations

- **Repetition is reduced, not eliminated.** Even with `no_repeat_ngram_size=1`,
  some sentences (see the rain-question example above) still produce
  visibly degenerate output ("do do thinking there'll be a shower t
  expected..."). FuguMT is a small MarianMT model and this is a real quality
  ceiling, not just a decoding-parameter issue -- confirmed by testing
  `no_repeat_ngram_size` at 0/1/2/3, `repetition_penalty` up to 1.3, and
  `beam_size` at 1/3/5/8, none of which fully fixed this particular case.
- **Already-English input is not detected/passed through specially.** The
  smoke test's English-only line ("This is already in English, so what
  happens?") was still fed through the model and came back garbled
  ("This is this Is already in English, so what way?"). If the upstream ASR
  can tag segment language, it may be worth skipping translation entirely
  for English-detected segments rather than relying on this module's output
  quality on English input.
- **Numbers/times are usually preserved correctly** in this sample (3:00 pm,
  five million yen) but this was not stress-tested beyond the smoke set.
- **CC BY-SA 4.0** applies -- see License section above before any
  redistribution of the model weights.
- The int8 `compute_type` is applied at load time via CTranslate2 (the
  distributed weights are float16); this was not compared quantitatively
  against float32 quality in this pass (a spot check on 2 lines showed
  identical repetition failure modes at both compute types, i.e.
  quantization was not the cause of the repetition issue).
