# Japanese Punctuation Restoration (`scripts/punct_ja.py`)

## What was shipped

A standalone `PunctuatorJa` class that inserts `。`(period) and `、`(comma)
into unpunctuated Japanese ASR output, plus a rule-based `？` heuristic for
obvious questions. It is CPU-only, loads once, and is fast enough for
per-utterance use in a live captioning pipeline.

## Model / approach

- **Source**: [Mojicast](https://github.com/ishiki-emo/mojicast) (MIT), an
  OSS offline real-time captioning app. Mojicast bundles a Japanese
  punctuation-restoration model, distributed on Hugging Face at
  [`ishiki-emo/mojicast-punct-onnx`](https://huggingface.co/ishiki-emo/mojicast-punct-onnx).
- **Architecture**: char-level BERT token classification. Body is
  [`tohoku-nlp/bert-base-japanese-char-v3`](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v3)
  (Apache-2.0), with a punctuation-classification head from
  [`bobfromjapan/bert_japanese_punctuation`](https://huggingface.co/bobfromjapan/bert_japanese_punctuation)
  (Apache-2.0) grafted on, exported to ONNX (opset 17) by the Mojicast author.
- **License**: Apache-2.0 (inherited from both base models; confirmed via
  the HF repo's `cardData.license` and README).
- **Downloaded into**: `models/mojicast-punct-onnx/` (`punct_bert.onnx`,
  `vocab.txt`, `README.md`). Not committed by this task — the directory is
  populated by re-running the download (see below) if needed.
- **Tokenizer**: reimplemented locally (no `transformers` dependency) to
  match `BertJapaneseTokenizer`'s pipeline exactly: NFKC-normalize -> split
  into MeCab morphemes (via `fugashi` + `unidic-lite`, installed into
  `.venv`) -> split each morpheme into individual characters -> map each
  character through `vocab.txt` (OOV -> `[UNK]`). Input to the model is
  `[CLS] + chars + [SEP]` as `input_ids`/`attention_mask` (int64); output is
  `logits` of shape `(seq_len, 2)` = `[comma_logit, period_logit]` for the
  position right after each token. Sigmoid + 0.5 threshold picks the mark.
- **Question marks (`？`)**: the model itself only classifies comma/period
  (no question-mark class exists in the label set). A small rule-based
  post-pass replaces a sentence-final `。` with `？` when the sentence ends
  in a common question suffix (`ですか`, `ますか`, `でしょうか`, `かな`,
  `の`, etc.). This is a heuristic, not a model prediction — see
  Limitations.

## Important deviation from the HF repo's stated default

The HF repo's README recommends `punct_bert.int8.onnx` (dynamic int8,
per-channel quantized) as Mojicast's default, citing ~5ms/line and 89.5%
agreement with fp32. **In this environment (onnxruntime 1.29.0, CPU EP on
Windows) the int8 file was verified broken**: for both synthetic and real
Japanese input, its output logits were nearly constant and did not vary
meaningfully with the input token sequence (confirmed by feeding different
random token-id sequences and by feeding real sentences — comma/period
probabilities stayed within ~0.31–0.36 everywhere, i.e. no signal). The
fp32 file (`punct_bert.onnx`, ~364MB) was verified correct on the same
inputs (produced near-1.0 period probability at the right character) and is
what `scripts/punct_ja.py` loads. The (apparently corrupted or
build-specific) int8 file was deleted from `models/mojicast-punct-onnx/`
to avoid confusion; re-download it yourself if you want to re-investigate
on a different onnxruntime version.

## How to (re-)download the model

```
mkdir -p models/mojicast-punct-onnx
curl -L -o models/mojicast-punct-onnx/punct_bert.onnx  https://huggingface.co/ishiki-emo/mojicast-punct-onnx/resolve/main/punct_bert.onnx
curl -L -o models/mojicast-punct-onnx/vocab.txt         https://huggingface.co/ishiki-emo/mojicast-punct-onnx/resolve/main/vocab.txt
```

Extra deps beyond what's already in `.venv`: `pip install fugashi
unidic-lite` (character-level MeCab pre-tokenization, matches the original
tokenizer exactly; both are lightweight, no `transformers`/PyTorch needed).

## Usage

```python
from punct_ja import PunctuatorJa

p = PunctuatorJa()  # loads model once (~0.2-0.4s cold start)
p.restore("明日の会議は午後三時から始まります資料の準備をお願いします")
# -> "明日の会議は午後三時から始まります。資料の準備をお願いします。"
```

## Measured latency (this machine, CPU, fp32 model)

Run via `python scripts/punct_ja.py` (`.venv/Scripts/python`, Windows 11):

| Input | Output | Latency |
|---|---|---|
| `明日の会議は午後三時から始まります資料の準備をお願いします` | `明日の会議は午後三時から始まります。資料の準備をお願いします。` | 23.0 ms |
| `今日めっちゃ疲れたわもう寝る` | `今日めっちゃ疲れたわ。もう寝る。` | 16.6 ms |
| `これって本当に大丈夫なんですか` | `これって本当に大丈夫なんですか？` | 18.7 ms |

Model load (cold start, once per process): ~0.2-0.4s. Additional spot
checks on longer 2-clause sentences (~30 chars) also landed in the
15-25ms range — well within the <100ms/utterance target. First call after
load is sometimes slightly slower due to ORT lazy init; subsequent calls
are consistently faster.

## Known limitations

- **Only comma/period are model predictions.** `？` is a hand-written
  suffix-matching heuristic (see above), not a model output — it will miss
  questions that don't end in a recognized suffix (e.g. rising-intonation
  statements used as questions) and could rarely misfire on a suffix
  coincidence (e.g. a sentence ending in `...の` used as a nominalizer, not
  a question). `！` is not handled at all.
- **fp32 model is fully quantization-free**, so it is the model-size/CPU
  tradeoff: ~364MB on disk, ~11ms/line per the source repo's own
  benchmark (matches what was measured here: ~15-25ms for full
  restore() calls including Python-side tokenization overhead).
- **The shipped int8 file was found non-functional** in this environment
  (see "Important deviation" above); it was not root-caused (could be a
  bad export, a stale/mismatched upload, or an onnxruntime CPU EP
  incompatibility with this specific quantized graph). If a working int8
  build becomes available it should cut latency roughly in half per the
  source repo's numbers.
  **Update**: a from-scratch INT8 requantization of the fp32 file (not the
  broken upstream artifact) was verified functional and measured as part of
  the mobile-sizing work — see `docs/MOBILE.md`, "Punctuation model INT8"
  section, and `scripts/quantize_punct.py`. It is 74.9% smaller (363.5MB ->
  91.4MB) and ~2x faster on this PC, with no accuracy regression on a
  15-reference sample (in fact marginally fewer false-positive marks than
  fp32). `PunctuatorJa` still defaults to fp32 (`punct_bert.onnx`); pass
  `onnx_filename="quantized_ort/punct_bert.int8.onnx"` to use the
  requantized model instead. It is not wired in as the default pending a
  larger accuracy check and an on-device latency measurement.
- **No streaming/incremental restoration.** `restore()` re-tokenizes and
  re-runs the whole given text each call; it's meant to be called on
  finalized ASR segments (e.g. one call per utterance/segment), not on a
  token-by-token streaming basis.
- **Character-level model, max ~500 chars per call** (`max_chars` param,
  under the model's 512-position limit including `[CLS]`/`[SEP]`); longer
  input is silently truncated rather than chunked. Fine for typical
  1-2 sentence ASR utterances; would need chunking logic for long-form
  transcripts.
- **No token_type_ids input** is used (the model doesn't require it) and
  no batching is implemented — one string in, one string out.
- Not integrated into `scripts/asr_engine.py` or
  `scripts/realtime_transcribe.py` by design (per task scope); this is a
  standalone, independently verified module for the orchestrator to wire
  in later.
