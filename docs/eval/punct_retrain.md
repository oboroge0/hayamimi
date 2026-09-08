# Japanese punctuation model: 4-class small retrain (improvement track C)

Dated record, 2026-09-08. Retrains the ja punctuation-restoration model
from scratch as a small token classifier that predicts one of 5 classes
per subword token -- none / 、/ 。/ ？/ ！ -- instead of the currently
shipped model (`models/mojicast-punct-onnx/`, BERT-char-base, 364MB fp32,
see `docs/design/punct_ja.md`), which predicts only comma/period from the
model, adds "？" via a suffix-matching heuristic, and has no "！" support
at all.

**Verdict: ADOPT** the round-2 model in
`models/punct-ja-4class-permissive/`. All six pre-declared acceptance
criteria pass. This does **not** change any default model path anywhere
in the repo -- see "What actually shipped".

## Round 1 vs round 2: why this was redone

This track was run twice. The first round met every accuracy target but
was **rejected on licensing**, and the second round replaces it.

Round 1 fine-tuned `ku-nlp/deberta-v2-tiny-japanese`, which is
**CC-BY-SA-4.0**, on Wikipedia ja (CC-BY-SA-3.0 + GFDL) and JSQuAD
(CC-BY-SA-4.0). A fine-tune of a share-alike base model is a derivative
of it, so the resulting weights would have had to be redistributed under
CC-BY-SA-4.0 as well. The model those weights would replace
(`mojicast-punct-onnx`) is Apache-2.0, so adopting round 1 would have
moved this repo's punctuation model *from* a permissive license *to* a
share-alike one -- and `THIRD_PARTY_NOTICES.md` currently carries exactly
one share-alike entry, flagged with its own warning paragraph, precisely
because that is a cost worth avoiding. Round 1 chose the share-alike base
deliberately, to fit a `<=100MB` fp32 size cap; it documented the
trade-off but made the wrong call on it.

Round 2 keeps the entire pipeline and swaps two things:

| | round 1 (rejected) | round 2 (adopted) |
|---|---|---|
| base model | `ku-nlp/deberta-v2-tiny-japanese`, **CC-BY-SA-4.0** | [`sbintuitions/modernbert-ja-30m`](https://huggingface.co/sbintuitions/modernbert-ja-30m), **MIT** |
| training text | Wikipedia ja (CC-BY-SA-3.0+GFDL) + JSQuAD (CC-BY-SA-4.0) + ~600 self-authored ！ templates | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) config `jpn_Jpan`, **ODC-By 1.0**, alone |
| size criterion | fp32 ONNX <= 100MB | **int8 ONNX <= 100MB** (relaxed for round 2; fp32 also recorded) |

The size criterion was relaxed because it was the constraint that pushed
round 1 into the share-alike base in the first place, and because int8 is
what a deployment would actually ship (`docs/design/mobile_quantization.md`).

Round 1's model is kept on disk at `models/punct-ja-4class/` and is
re-measured here as a comparison column; it is not shipped and nothing
points at it.

## Data and licenses

Everything below is permissive -- no share-alike, no NC, no ND.

| Source | Role | License | Notes |
|---|---|---|---|
| [`sbintuitions/modernbert-ja-30m`](https://huggingface.co/sbintuitions/modernbert-ja-30m) | base model | **MIT** | Verified by fetching the repo's own `LICENSE` file ("MIT License, Copyright (c) 2025 SB Intuitions") as well as the model card's `license: mit`. ModernBERT, 10 layers, hidden=256, vocab=102400. |
| [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), config `jpn_Jpan` | training text (all four marks) | **ODC-By 1.0** | Verified via the dataset card's `license: odc-by`. Streamed, never downloaded in full. |
| FLEURS ja (`google/fleurs`, `ja_jp`, validation+test) | **eval only, never trained on** | (unchanged from existing usage elsewhere in this repo) | Same 250-sentence, seed-0, length-stratified sample as the fp32/int8/fp16 comparisons in `docs/design/mobile_quantization.md`, via `scripts/quantize_punct.py::build_fleurs_refs`. |
| self-authored question / exclamation eval sets | eval only | n/a (authored for this task) | 50 sentences each, in `scripts/eval_punct_4class.py`. See the caveats under each results table. |

The trained weights are a derivative of an MIT-licensed base model, so
**they are distributable under MIT**, matching hayamimi's own license and
the rest of `THIRD_PARTY_NOTICES.md`'s "Text models" table except the one
flagged FuguMT entry. ODC-By governs the *training corpus*, and its
attribution obligation attaches to redistribution of the database; it is
credited here and in `scripts/make_punct_trainset.py`.

### Why this base model and not the others

The label pipeline assigns each subword token the mark that follows its
last character, which needs the tokenizer's **character offset mapping**
(`return_offsets_mapping=True`). Only a *fast* (Rust `tokenizers`-backed)
tokenizer provides it, and that turned out to be the deciding constraint,
not size:

| candidate | license | outcome |
|---|---|---|
| `sbintuitions/modernbert-ja-30m` | MIT | **chosen.** Ships `tokenizer.json`; offsets verified exact on Japanese text. |
| `sbintuitions/modernbert-ja-70m` | MIT | not needed -- the 30m variant cleared every criterion, so the larger one was never trained. |
| `line-corporation/line-distilbert-base-japanese` | Apache-2.0 | **ruled out.** No `tokenizer.json`; the repo ships a custom `distilbert_japanese_tokenizer.py` requiring `trust_remote_code=True` -- a slow tokenizer with no offsets, and arbitrary remote code execution to find that out. Round 1 had ruled it out on fp32 size instead, which the relaxed criterion would no longer have justified. |
| `tohoku-nlp/bert-base-japanese-char-v3` | Apache-2.0 | **ruled out**, same reason: `BertJapaneseTokenizer`, `is_fast=False`. Worse, it fails *silently* -- passing `return_offsets_mapping=True` raises nothing and simply returns no `offset_mapping` key, which would have labelled every token "O" without an error. (This is the body under the currently shipped model.) |

Subword tokenization does impose a ceiling: a mark can only be predicted
at a token boundary. Measured on the 250-sentence FLEURS eval set, 696 of
709 reference marks fall on a `modernbert-ja` token boundary, so the
**recall ceiling is 0.982** (round 1's finer-grained deberta tokenizer had
a ceiling of 1.000). At an achieved recall of 0.939 this is not the
binding constraint, but it is why recall cannot approach 1.0.

### Building the training corpus (`scripts/make_punct_trainset.py --corpus fineweb`)

FineWeb-2 is raw-ish web text, which is the whole point -- Japanese web
prose uses ？ and ！ naturally, so round 1's two workarounds (a
share-alike QA corpus for ？, ~600 self-authored templates for ！) are
both unnecessary. The price is noise, and `_web_sentence_ok()` pays it:

- the sentence must **end in a terminal mark** (。／？／！). This is the
  most important filter. "No mark follows this character" is already the
  overwhelming majority class, so every unterminated web fragment kept
  would teach the model to withhold a mark exactly where one belongs.
- length 10-200 chars; CJK ratio >= 0.4; latin+digit ratio <= 0.3; at
  most one "junk" glyph (bullets, pipes, decorative separators) and none
  in first position.
- mark density <= 0.2 (rejects mark spam, which is as useless as no marks).
- **reject any sentence containing an ASCII `,` or a non-decimal `.`**.
  Machine-translated pages punctuate Japanese with halfwidth marks; those
  are not target marks, so each one becomes a position labelled "no mark
  follows" sitting precisely where a mark belongs. This filter was added
  after eyeballing the first corpus draft, which was full of such text.
- halfwidth `?`/`!` *are* folded to ？/！ before labelling (web text uses
  both forms), and runs like `！！！` collapse to a single mark.

？/！-bearing windows are rare even after filtering (~1 per streamed
document), so `_collect_two_bucket()` keeps **every** one it encounters
while sampling declarative windows only up to a separate cap. This buys
real, diverse ？/！ evidence out of the same ODC-By corpus instead of
duplicating examples (which adds no signal) or importing an incompatible
one.

60,000 documents streamed in ~59s, yielding:

| | count |
|---|---|
| ？/！-free windows | 110,000 |
| ？/！-bearing windows | 45,000 |
| train examples | 150,350 |
| val examples (held out from the training pool, **not** FLEURS) | 4,650 |

Mark frequency over the corpus, against round 1's for comparison:

| mark | round 2 occurrences | round 2 % of chars | round 1 % of chars |
|---|---|---|---|
| 、 | 314,634 | 2.609% | 2.544% |
| 。 | 230,826 | 1.914% | 1.556% |
| ？ | 21,267 | 0.176% | 0.275% (from JSQuAD) |
| ！ | 34,274 | **0.284%** | 0.009% (synthetic) |

？ is somewhat thinner than round 1's JSQuAD-boosted rate but is now
naturally occurring rather than one dataset's question style; ！ is **30x**
denser and, for the first time, real.

## Training config

- Base: `sbintuitions/modernbert-ja-30m`, `AutoModelForTokenClassification`
  with a fresh 5-way head (`num_labels=5`,
  `id2label={0:"O",1:"、",2:"。",3:"？",4:"！"}`). 36,772,613 parameters.
- Loaded with `attn_implementation="eager"`: ModernBERT's default
  sdpa/flash attention path is not traceable by `torch.onnx.export`, and
  training under the same implementation the export uses makes train and
  inference numerics identical rather than merely close.
- Labels derived at train time (`PunctDataset`): strip target marks to get
  raw text plus a per-character "mark that followed this character" list,
  tokenize the raw text with `return_offsets_mapping=True`, assign each
  token the mark that followed its last covered character.
- Loss: `CrossEntropyLoss` with per-class weights `freq**-0.5` normalized
  to mean 1 and capped at 15x -- `O=0.152, 、=0.538, 。=0.621, ？=2.070,
  ！=1.618`. The exponent is now the `--class-weight-power` flag; see
  "Comma precision" below for why it was left at round 1's value.
- AdamW, `lr=5e-5`, weight decay 0.01, `OneCycleLR` (10% warmup), grad
  clipped to norm 1.0. 3 epochs, batch 64, max sequence length 320.
- RTX 3080 Ti, run as two resumable invocations via the new
  `--epochs-this-run` flag (epoch 1: 3m39s including tokenization;
  epochs 2-3: 5m49s), ~9.5 min total. Checkpoints under
  `<model-dir>/ckpt/` (not committed); only the final one is kept.

Held-out validation (own split, not FLEURS), macro-F1 over 、/。/？/！:

| epoch | macro-F1 | 、 F1 | 。 F1 | ？ F1 | ！ F1 |
|---|---|---|---|---|---|
| 1 | 0.7161 | 0.7147 | 0.8965 | 0.6839 | 0.5691 |
| 2 | 0.7345 | 0.7242 | 0.9045 | 0.7399 | 0.5693 |
| 3 | **0.7389** | 0.7130 | 0.9025 | 0.7642 | 0.5760 |

Round 1's equivalent final figure was macro-F1 **0.6405** with ！ F1
**0.231**. These two validation splits are different corpora and are not
strictly comparable; the FLEURS numbers below are.

## Export

`scripts/export_punct_4class.py`: `torch.onnx.export` (opset 17, dynamic
batch/seq axes), sanity-checked against the PyTorch model's own logits at
the traced sequence length **and at a second, different length** (a
silently-static sequence axis would only show up on the second) -- max abs
logit diff 3.1e-5 and 1.9e-5. Then
`onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)`, the same
recipe `scripts/quantize_punct.py` uses for the existing model.

| variant | size | reduction |
|---|---|---|
| fp32 | 147.33 MB | -- |
| **int8 (dynamic)** | **37.17 MB** | 74.8% |

The fp32 file is large relative to the parameter count because
`modernbert-ja`'s 102,400-token vocabulary makes the embedding matrix most
of the model; dynamic quantization compresses it along with the weight
matrices.

## Results: FLEURS ja, n=250, seed=0

Same sentences as `docs/design/mobile_quantization.md`. Scored with
`scripts/eval_punct_4class.py` (punct-position F1, exact mark type, same
alignment convention as `scripts/quantize_punct.py::evaluate`, extended to
4 marks and NFKC-bug-fixed -- see "The NFKC bug" below). **All four rows
were measured here in the same session with the same scorer**, including
the shipped baseline, rather than quoting a number from another document.
CPU, `intra_op_num_threads=2` per this task's constraint (other
improvement tracks were evaluating in parallel on this machine), so
latency figures are **provisional**, not clean-machine benchmarks.

| model | P | R | F1 | punct-inclusive CER | mean latency |
|---|---|---|---|---|---|
| existing shipped fp32 (mojicast BERT-char, 、/。 + ？ heuristic) | 0.8724 | 0.4824 | 0.6213 | 2.81% | 49.23 ms |
| round-1 fp32 (deberta-v2-tiny, **CC-BY-SA**, not shipped) | 0.6769 | 0.9041 | 0.7742 | 2.52% | 2.92 ms |
| **round-2 fp32 (modernbert-ja-30m, MIT)** | 0.8409 | 0.9394 | **0.8874** | **1.19%** | 8.73 ms |
| **round-2 int8 (dynamic)** | 0.8579 | 0.9196 | **0.8877** | **1.17%** | 4.56 ms |

**int8 vs round-2 fp32: F1 delta = +0.0003.** Quantization is a wash here
(int8 trades a little recall for a little precision), comfortably inside
the -0.02 tolerance and on the right side of zero.

**Round 2 vs the shipped model: F1 +0.2661, CER 2.81% -> 1.19%.** Round 2
improves on *both* axes rather than trading them: it roughly keeps the
shipped model's precision (0.841 vs 0.872) while nearly doubling recall
(0.939 vs 0.482).

**Round 2 vs round 1: F1 +0.1132**, and the improvement is almost entirely
precision (0.841 vs 0.677) at equal-or-better recall. Round 1's central
weakness was over-inserting 、; round 2 does not have it.

Per-mark breakdown (`support` = reference occurrences in this sample):

| mark | model | P | R | F1 | support |
|---|---|---|---|---|---|
| 、 | shipped fp32 | 0.9104 | 0.1470 | 0.2531 | 415 |
| 、 | round-1 fp32 | 0.6279 | 0.8988 | 0.7393 | 415 |
| 、 | **round-2 fp32** | **0.7641** | **0.9133** | **0.8321** | 415 |
| 、 | **round-2 int8** | 0.7811 | 0.8771 | 0.8263 | 415 |
| 。 | shipped fp32 | 0.8646 | 0.9590 | 0.9094 | 293 |
| 。 | round-1 fp32 | 0.8171 | 0.9147 | 0.8631 | 293 |
| 。 | **round-2 fp32** | **0.9829** | **0.9795** | **0.9812** | 293 |
| 。 | **round-2 int8** | 0.9863 | 0.9829 | 0.9846 | 293 |
| ？ | all | -- | -- | -- | **0** |
| ！ | all | 0.0 | 0.0 | 0.0 | **1** |

The shipped model's 、 recall of 0.147 is the headline: it finds one comma
in seven. That is what its high overall precision is buying.

This FLEURS draw contains **zero ？ and one ！** -- both marks are rare in
FLEURS ja, whose read-aloud source text is almost entirely declarative --
so the two dedicated sets below carry the ？/！ evidence. The single ！ is
missed by every model, round 2 included.

### Comma precision

Round 1's 、 precision was 0.628, and this task set 0.75 as a quality
target (not an acceptance criterion), to be pursued via class weights or
thresholds if missed. Round 2 reaches **0.764 (fp32) / 0.781 (int8)**
without any such adjustment, so no sweep was run and
`--class-weight-power` was left at round 1's 0.5. The gain came from the
data, not the loss: the terminal-mark and ASCII-punctuation filters
removed exactly the training positions that had been teaching round 1 to
sprinkle commas. The flag exists if a future run needs to trade recall
back for precision.

## Question set: ？ recall (acceptance criterion 4, threshold >= 0.7)

`--question-set` builds this set from FLEURS ja sentences ending in "か。"
or "？" first, falling back to self-authored sentences. **The FLEURS ja
pool (456 sentences after the existing filters) contains zero of either**,
so the full 50-sentence set is self-authored, as in round 1.

| model | ？ precision | ？ recall | verdict |
|---|---|---|---|
| round-2 fp32 | 0.9804 | **1.0000** | PASS (>= 0.7) |
| round-2 int8 | 0.9804 | **1.0000** | PASS (>= 0.7) |

Caveat, unchanged from round 1: this measures recognition of clearly
marked question forms (か/でしょうか/かな/の endings), not
naturally-occurring open-domain question phrasing. Read 1.0 as "reliably
catches textbook question forms", not as a general claim.

## Exclamation set: ！ (quality note, not a criterion)

New in round 2, because round 2 is the first version to learn ！ from real
text and a claim about it should be measured rather than asserted. 50
self-authored ！-terminated sentences (`build_exclaim_set()`), deliberately
*not* web text, so this measures generalization rather than recall of the
training domain.

| model | ！ P | ！ R | ！ F1 |
|---|---|---|---|
| existing shipped fp32 | 0.0 | 0.0 | **0.0** (cannot predict ！ at all) |
| round-1 fp32 (~600 synthetic templates) | 0.8485 | 0.5600 | 0.6747 |
| **round-2 fp32** (real web ！) | **0.9688** | 0.6200 | **0.7561** |
| round-2 int8 | 0.9677 | 0.6000 | 0.7407 |

Precision is high and recall is the weak half: when round 2 emits ！ it is
almost always right, but it still resolves about 38% of these sentences to
。 instead. That is a defensible failure mode for a subtitle pipeline -- a
missed ！ reads as a neutral sentence, whereas a spurious one misreads
tone -- but ！ remains the weakest of the four classes and callers should
not treat it as reliable.

## eval_real: where round 2 is *worse* (honest negative)

`testdata/eval_real`'s 15 ja TV-caption clips, the only set here with
natural ？ and ！ in real material:

| model | P | R | F1 | punct-inclusive CER |
|---|---|---|---|---|
| **existing shipped fp32** | 0.6500 | 0.5909 | **0.6190** | **4.39%** |
| round-1 fp32 | 0.3214 | 0.4091 | 0.3600 | 7.84% |
| round-2 fp32 | 0.3529 | 0.5455 | 0.4286 | 8.15% |
| round-2 int8 | 0.3939 | 0.5909 | 0.4727 | 7.52% |

The shipped model wins this set outright. Two things are going on, and
only one of them is a real weakness:

1. **The sample is far too small to conclude from.** 15 clips carry 22
   reference marks in total (7 、, 12 。, 1 ？, 2 ！). A handful of
   decisions moves F1 by 0.1. The shipped model's 、 F1 here is 0.000 --
   it never emits a comma at all across the whole set -- and it still
   wins, which shows how much of this is 。 placement on very short
   utterances.
2. **Domain mismatch is real, though.** TV captions punctuate sparsely by
   editorial convention; round 2 was trained on web prose that punctuates
   densely, so its 、 precision drops to 0.235 here. Round 2 inserts
   commas that are defensible Japanese but are not in these references.

This does not change the adoption decision -- the criteria are defined on
FLEURS, the model ships opt-in and changes no default, and 22 marks cannot
overturn 709 -- but anyone wiring this into a caption-style product should
measure on their own material first, and should expect to want fewer
commas than round 2 produces.

## Latency

`intra_op_num_threads=2`, other tracks evaluating concurrently on the same
CPU -- **provisional**:

| variant | mean `restore()` latency (n=250, FLEURS) |
|---|---|
| existing shipped fp32 | 49.23 ms |
| round-2 fp32 | 8.73 ms |
| round-2 int8 | 4.56 ms |

Round 2 is ~5.6x (fp32) to ~10.8x (int8) faster than the shipped model and
clears the 41ms/line threshold with an order of magnitude of headroom even
under contention. Round 1 was faster still (2.92 ms) because it was a
7.7M-parameter model; round 2 spends some of that headroom on accuracy.

## The NFKC bug (found in round 1, still worth knowing)

`unicodedata.normalize("NFKC", text)` folds fullwidth "？"/"！" (U+FF1F /
U+FF01) to ASCII "?"/"!" -- standard fullwidth-Latin compatibility
folding. "、"/"。" are ideographic punctuation and are unaffected.
`scripts/quantize_punct.py::strip_marks` NFKC-normalizes and *then* checks
membership in a fullwidth-only `TARGET_MARKS` set, so after folding, ？/！
are silently never recorded as marks. Round 1 inherited the pattern and it
zeroed out every ？/！ training label before being caught.

Fixed via `_safe_nfkc()` (private-use-area sentinels across the NFKC call)
in `scripts/train_punct_ja.py`, `scripts/make_punct_trainset.py`, and
`scripts/eval_punct_4class.py`. **Not** changed in
`scripts/quantize_punct.py` itself, to avoid silently altering the
historical fp32/int8/fp16 numbers published in
`docs/design/mobile_quantization.md` and `docs/design/punct_ja.md`; the
eval script reimplements a fixed `strip_marks`/`marks_from_restored` and
reuses only `build_fleurs_refs`, which was never affected.

## Adoption decision: ADOPT (all 6 criteria pass)

| # | criterion | threshold | result | pass? |
|---|---|---|---|---|
| 1 | FLEURS ja n=250 F1 | >= 0.65 | 0.8874 (fp32) / 0.8877 (int8) | **PASS** |
| 2 | int8 vs fp32 F1 delta | >= -0.02 | +0.0003 | **PASS** |
| 3 | int8 ONNX size | <= 100MB | 37.17MB (fp32 147.33MB) | **PASS** |
| 4 | question-set ？ recall | >= 0.7 | 1.0000 (both variants) | **PASS** |
| 5 | mean latency (CPU, provisional) | <= 41ms | 8.73ms (fp32) / 4.56ms (int8) | **PASS** |
| 6 | licenses | permissive base (MIT/Apache/CC-BY) + permissive text (ODC-By/CC-BY/CC0/PD) | MIT base + ODC-By 1.0 text; **no share-alike anywhere** | **PASS** |

Quality note (not a criterion): 、 precision 0.764 fp32 / 0.781 int8,
clearing the 0.75 target that round 1 missed at 0.628.

## What actually shipped

Meeting the criteria unlocks an **opt-in** extension only -- no default
model path changes anywhere in the repo:

- `scripts/punct_ja.py`'s `PunctuatorJa4Class` (added in round 1) now
  defaults to `models/punct-ja-4class-permissive/`. It remains a second,
  independent restorer alongside `PunctuatorJa`, which is untouched and
  remains the only one anything defaults to. Callers must import and
  instantiate this *different class* to opt in; nothing wires it in
  automatically. It loads ONNX via `onnxruntime` and the tokenizer via the
  lightweight `tokenizers` package (already a transitive dependency), so
  opting in does not require torch/transformers at runtime.
- `scripts/eval_punct_4class.py` now drives `PunctuatorJa4Class` itself
  rather than a private copy of its restore path, so these numbers
  describe the code that ships. It also gained `--variant baseline` (the
  shipped mojicast model through the same scorer), `--exclaim-set`,
  `--model-dir`, and scratch output under `testdata/` keyed by model
  directory.
- `tests/test_punct_4class.py`: model-free tests of the
  offsets/labels -> text reconstruction, plus skipped-if-unavailable
  end-to-end smoke tests against the real model, now including
  `test_restore_smoke_only_inserts_marks` -- deleting the marks `restore()`
  inserted must return the input character for character. That invariant
  is what makes swapping a base model (and its tokenizer) under this class
  safe: off-by-one offsets would corrupt text, not merely mispunctuate it.
- `mobile/` (Dart side) was **not** touched, per task scope.

### Proposed `THIRD_PARTY_NOTICES.md` addition

Applied at integration time (the row below is what `THIRD_PARTY_NOTICES.md`
now carries, in the **Text models** table):

```markdown
| `punct-ja-4class-permissive` (Japanese 4-class punctuation restoration, opt-in) | Base model: SB Intuitions; fine-tune: hayamimi | MIT | [sbintuitions/modernbert-ja-30m](https://huggingface.co/sbintuitions/modernbert-ja-30m), fine-tuned on [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `jpn_Jpan` (ODC-By 1.0) -- see `docs/eval/punct_retrain.md` |
```

Because the training corpus is ODC-By, redistribution of these weights
should credit FineWeb-2; the row above does that. No share-alike
obligation attaches, so the existing CC-BY-SA warning paragraph does not
need to grow.

## Reproduce

```
# 1. training venv (torch cu121 + transformers/datasets, not committed)
python -m venv .venv-train
.venv-train/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv-train/Scripts/python -m pip install transformers datasets accelerate fugashi unidic-lite sentencepiece onnx onnxruntime onnxconverter-common numpy fsspec pyarrow tqdm scikit-learn

# 2. build the corpus (streams fineweb-2 jpn_Jpan; ~59s; no FLEURS)
.venv-train/Scripts/python scripts/make_punct_trainset.py --corpus fineweb --n-docs 60000 --max-windows 110000 --max-q-windows 45000

# 3. train, in two resumable chunks
.venv-train/Scripts/python scripts/train_punct_ja.py --epochs 3 --epochs-this-run 1
.venv-train/Scripts/python scripts/train_punct_ja.py --epochs 3 --resume

# 4. export to ONNX fp32 + dynamic-int8
.venv-train/Scripts/python scripts/export_punct_4class.py

# 5. evaluate -- runs in the repo's normal .venv (onnxruntime + tokenizers)
.venv/Scripts/python scripts/eval_punct_4class.py --variant fp32 --n 250 --latency
.venv/Scripts/python scripts/eval_punct_4class.py --variant int8 --n 250 --latency
.venv/Scripts/python scripts/eval_punct_4class.py --variant baseline --n 250 --latency
.venv/Scripts/python scripts/eval_punct_4class.py --variant fp32 --question-set
.venv/Scripts/python scripts/eval_punct_4class.py --variant fp32 --exclaim-set
```

Round 1's corpus and model can be rebuilt with
`--corpus wikipedia --out-dir models/punct-ja-4class` and
`--base-model ku-nlp/deberta-v2-tiny-japanese --model-dir models/punct-ja-4class`.
Note that the rebuild will not be byte-identical to round 1's corpus:
`_clean_line()` is shared by both recipes and round 2 added URL stripping,
halfwidth `?`/`!` folding, and mark-run collapsing to it. Round 1's own
artefacts are kept on disk rather than regenerated, and its numbers in
this document were re-measured from those, not from a rebuild.

## Known limitations

- **Sparse-punctuation domains regress.** On `testdata/eval_real`'s TV
  captions the shipped model still wins (F1 0.619 vs 0.429); round 2
  over-inserts 、 relative to caption conventions. The sample is 22
  reference marks, so this is a flag to measure on your own material, not
  a measured defeat. See "eval_real" above.
- **！ recall is ~0.62** even on clearly exclamatory sentences. Precision
  is high (0.97), so the failure mode is silence rather than noise, but ！
  is not a reliable signal.
- **The ？ and ！ eval sets are self-authored** (FLEURS ja has zero
  question-form sentences and one ！ in the whole pool), so both measure
  clearly-marked forms rather than open-domain phrasing.
- **Web-corpus content is unfiltered for topic.** FineWeb-2 `jpn_Jpan` is
  raw web text, and the quality gate here filters for punctuation
  usefulness, not subject matter. The model's only output is which of four
  marks follows a position, so corpus content does not surface in its
  predictions, but the corpus is not curated.
- **Recall ceiling 0.982** on FLEURS from subword tokenization -- marks can
  only be placed at token boundaries.
- Latency numbers are provisional (measured under CPU contention).
- Like the existing model: no batching, no streaming/incremental
  restoration, and long input is truncated (not chunked) past `max_chars`.
